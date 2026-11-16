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
        live = "llm" in (cfg.agent_mode, cfg.tester_mode, cfg.judge_mode)
        if live:
            print(f"  {sc.id} ...", flush=True)
        for turn in range(1, sc.max_turns + 1):
            if live:
                print(f"    turn {turn}", flush=True)
            with tracer.span("tester.turn") as s:
                msg = tester.next_message(t)
                s.update(output=msg)
            if msg is None:
                t.stopped_reason = f"tester: {tester.stop_reason}"
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
        # The judge only grades quality; the release gate is the deterministic checks.
        # So a judge outage must not throw away a conversation that already ran.
        from agent_qa.llm import LLMError

        judge_error = None
        with tracer.span("judge") as s:
            try:
                verdict = judge.evaluate(sc, t, checks)
                s.update(output=verdict.model_dump())
            except LLMError as exc:
                verdict, judge_error = None, str(exc).splitlines()[0]
                print("    judge unavailable - verdict from deterministic checks only", flush=True)
                s.update(output={"error": judge_error})

        result = ScenarioResult(
            id=sc.id,
            title=sc.title,
            risk=sc.risk,
            transcript=t,
            checks=checks,
            judge=verdict,
            judge_error=judge_error,
            trace_id=tracer.current_trace_id(),
        )

        # 4. scores -> Langfuse (filterable dashboards per agent version / risk)
        for c in checks:
            tracer.score(f"check.{c.name}", 1.0 if c.passed else 0.0, c.detail or None)
        if verdict is not None:
            for k in CRITERIA:
                tracer.score(f"judge.{k}", float(verdict.scores[k]), verdict.reasoning)
        tracer.score("passed", 1.0 if result.passed else 0.0)
        root.update(output={"passed": result.passed, "transcript": t.render()})
    return result


def run_scenario_safe(sc: Scenario, cfg: RunConfig, tracer) -> ScenarioResult:
    """A provider outage marks the scenario ERROR and the run goes on."""
    from agent_qa.llm import LLMError

    try:
        return run_scenario(sc, cfg, tracer)
    except LLMError as exc:
        print(f"    ERROR: {str(exc).splitlines()[0]}", flush=True)
        return ScenarioResult(
            id=sc.id,
            title=sc.title,
            risk=sc.risk,
            transcript=Transcript(),
            checks=[],
            judge=None,
            error=str(exc),
        )


def run(scenarios: list[Scenario], cfg: RunConfig, tracer) -> RunResult:
    results = [run_scenario_safe(sc, cfg, tracer) for sc in scenarios]
    tracer.flush()
    return RunResult(
        agent=cfg.agent,
        mode={"agent": cfg.agent_mode, "tester": cfg.tester_mode, "judge": cfg.judge_mode},
        min_pass_rate=cfg.min_pass_rate,
        scenarios=results,
    )
