"""Runs scenarios: tester agent <-> agent under test, then checks, judge, scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agent_qa.checks import run_checks
from agent_qa.judge import CRITERIA, LLMJudge, ProxyJudge
from agent_qa.models import Message, RunResult, Scenario, ScenarioResult, ToolCall, Transcript
from agent_qa.tester import LLMTester, ScriptedTester
from travel_agent import build_agent
from travel_agent.data import TravelDB

Mode = Literal["mock", "llm"]


@dataclass
class RunConfig:
    agent: Literal["clean", "buggy"] = "clean"
    agent_mode: Mode = "mock"
    tester_mode: Mode = "mock"
    judge_mode: Mode = "mock"
    min_pass_rate: float = 0.8


def _llm(role: str, tracer, temperature: float = 0.0):
    from agent_qa.llm import LLMClient, model_for

    return LLMClient(role, model_for(role), tracer, temperature)


def run_scenario(sc: Scenario, cfg: RunConfig, tracer) -> ScenarioResult:
    db = TravelDB.fresh()
    t = Transcript()

    def record_tool(name, args, result):
        t.tool_calls.append(ToolCall(name=name, args=args, result=result, turn=t.turn))
        with tracer.span(f"tool.{name}", input=args) as s:
            s.update(output=result)

    chat = _llm("agent", tracer).chat if cfg.agent_mode == "llm" else None
    agent = build_agent(db, sc.customer, cfg.agent, chat=chat, recorder=record_tool)
    tester = (
        LLMTester(sc, _llm("tester", tracer, temperature=0.7))
        if cfg.tester_mode == "llm"
        else ScriptedTester(sc)
    )
    judge = LLMJudge(_llm("judge", tracer)) if cfg.judge_mode == "llm" else ProxyJudge()

    tags = [f"agent:{cfg.agent}", f"risk:{sc.risk}", f"tester:{tester.name}", *sc.tags]
    with tracer.scenario(
        sc.id,
        input={"title": sc.title, "goal": sc.goal},
        tags=tags,
        metadata={"risk": sc.risk, **cfg.__dict__},
    ) as root:
        # 1. the two agents talk
        t.stopped_reason = "max_turns"
        for turn in range(1, sc.max_turns + 1):
            with tracer.span("tester.turn") as s:
                msg = tester.next_message(t)
                s.update(output=msg)
            if msg is None:
                t.stopped_reason = "tester_done"
                break
            t.turn = turn
            t.messages.append(Message(role="user", content=msg, turn=turn))
            with tracer.span("agent.turn", input=msg) as s:
                reply = agent.respond(msg)
                s.update(output=reply)
            t.messages.append(Message(role="assistant", content=reply, turn=turn))

        # 2. deterministic checks on behaviour + final state
        with tracer.span("checks") as s:
            checks = run_checks(sc.checks, t, db)
            s.update(output=[c.model_dump() for c in checks])

        # 3. judge
        with tracer.span("judge") as s:
            verdict = judge.evaluate(sc, t, checks)
            s.update(output=verdict.model_dump())

        result = ScenarioResult(
            id=sc.id,
            title=sc.title,
            risk=sc.risk,
            transcript=t,
            checks=checks,
            judge=verdict,
            trace_id=tracer.current_trace_id(),
        )

        # 4. scores -> Langfuse (filterable dashboards per agent version / risk)
        for c in checks:
            tracer.score(f"check.{c.name}", 1.0 if c.passed else 0.0, c.detail or None)
        for k in CRITERIA:
            tracer.score(f"judge.{k}", float(verdict.scores[k]), verdict.reasoning)
        tracer.score("passed", 1.0 if result.passed else 0.0)
        root.update(output={"passed": result.passed, "transcript": t.render()})
    return result


def run(scenarios: list[Scenario], cfg: RunConfig, tracer) -> RunResult:
    results = [run_scenario(sc, cfg, tracer) for sc in scenarios]
    tracer.flush()
    return RunResult(
        agent=cfg.agent,
        mode={"agent": cfg.agent_mode, "tester": cfg.tester_mode, "judge": cfg.judge_mode},
        min_pass_rate=cfg.min_pass_rate,
        scenarios=results,
    )
