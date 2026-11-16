"""The LLM-facing components, tested offline with fake models.

These prove the plumbing (tool-call loop, role mirroring, judge parsing, fail-closed
behaviour) without spending tokens. Real-model behaviour is measured by `--mode llm`.
"""

import json

from agent_qa.judge import LLMJudge
from agent_qa.models import Message, Transcript
from agent_qa.scenarios import load_scenarios
from agent_qa.tester import LLMTester
from travel_agent import build_agent
from travel_agent.data import TravelDB


class FakeLLM:
    model = "fake-model"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, tools=None, json_mode=False):
        self.calls.append(list(messages))  # snapshot: the brain keeps mutating its history
        reply = self.replies.pop(0)
        return reply if isinstance(reply, dict) else {"role": "assistant", "content": reply}


SCENARIO = next(s for s in load_scenarios() if s.id == "BOOK-001")


def test_llm_brain_runs_tool_loop():
    fake = FakeLLM(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "search_flights",
                            "arguments": json.dumps({"origin": "PRG", "destination": "LIS"}),
                        },
                    }
                ],
            },
            "FL100 for €180. Shall I book it?",
        ]
    )
    seen = []
    agent = build_agent(
        TravelDB.fresh(), "anna", chat=fake.chat, recorder=lambda n, a, r: seen.append((n, r))
    )
    assert agent.respond("PRG to LIS?") == "FL100 for €180. Shall I book it?"
    assert seen[0][0] == "search_flights" and seen[0][1][0]["flight_id"] == "FL100"
    # the tool result was fed back to the model
    assert fake.calls[1][-1]["role"] == "tool"
    # the clean release gets the full policy prompt with the customer filled in
    assert "Anna Novak" in fake.calls[0][0]["content"] and "P1." in fake.calls[0][0]["content"]


def test_buggy_release_gets_the_weakened_prompt():
    fake = FakeLLM(["hi"])
    build_agent(TravelDB.fresh(), "anna", "buggy", chat=fake.chat).respond("hi")
    assert "P1." not in fake.calls[0][0]["content"]


def _t(agent_says: str) -> Transcript:
    return Transcript(
        messages=[
            Message(role="user", content="PRG to LIS?", turn=1),
            Message(role="assistant", content=agent_says, turn=1),
        ]
    )


def test_tester_mirrors_roles():
    fake = FakeLLM([json.dumps({"message": "Yes please", "done": False})])
    assert LLMTester(SCENARIO, fake).next_message(_t("FL100 €180, book?")) == "Yes please"
    sent = fake.calls[0]
    assert sent[0]["role"] == "system" and "Anna" in sent[0]["content"]
    assert [m["role"] for m in sent[1:]] == ["assistant", "user"]  # mirrored


def test_tester_stops_with_reason():
    fake = FakeLLM([json.dumps({"message": "", "done": True, "reason": "flight booked"})])
    tester = LLMTester(SCENARIO, fake)
    assert tester.next_message(_t("Booked! BK-3001.")) is None
    assert tester.stop_reason == "flight booked"


def test_tester_is_nudged_when_quitting_on_a_question():
    """Regression: a real Gemini tester quit on 'Shall I book it?' -> false FAIL."""
    fake = FakeLLM(
        [
            json.dumps({"message": "", "done": True}),
            json.dumps({"message": "Yes, book FL100", "done": False}),
        ]
    )
    assert LLMTester(SCENARIO, fake).next_message(_t("FL100 for €180. Shall I book it?")) == "Yes, book FL100"
    assert "asks you something" in fake.calls[1][-1]["content"]


def test_tester_nudged_only_once():
    fake = FakeLLM([json.dumps({"done": True}), json.dumps({"done": True, "reason": "not interested"})])
    tester = LLMTester(SCENARIO, fake)
    assert tester.next_message(_t("Shall I book it?")) is None
    assert tester.stop_reason == "not interested" and len(fake.calls) == 2


def test_judge_parses_scores():
    out = {
        "scores": {"task_completion": 5, "policy_adherence": 4, "honesty": 5, "communication": 4},
        "verdict": "pass",
        "reasoning": "fine",
    }
    r = LLMJudge(FakeLLM([json.dumps(out)])).evaluate(SCENARIO, Transcript(), [])
    assert r.passed and r.judge == "fake-model"


def test_judge_low_score_fails_even_with_pass_verdict():
    out = {
        "scores": {"task_completion": 5, "policy_adherence": 2, "honesty": 5, "communication": 5},
        "verdict": "pass",
    }
    assert not LLMJudge(FakeLLM([json.dumps(out)])).evaluate(SCENARIO, Transcript(), []).passed


def test_unparseable_judge_fails_closed():
    r = LLMJudge(FakeLLM(["I think it went well!"])).evaluate(SCENARIO, Transcript(), [])
    assert not r.passed and "could not be parsed" in r.reasoning


# -- provider resilience ---------------------------------------------------------------


class _HTTPError(Exception):
    def __init__(self, status):
        self.status_code = status


def test_retries_transient_errors_then_succeeds():
    from agent_qa.llm import call_with_retries

    attempts, waited = iter([_HTTPError(503), _HTTPError(429), "ok"]), []

    def fn():
        x = next(attempts)
        if isinstance(x, Exception):
            raise x
        return x

    log = lambda *a, **k: None  # noqa: E731
    assert call_with_retries(fn, [10, 30, 60], sleep=waited.append, log=log) == "ok"
    assert waited == [10, 30]


def test_does_not_retry_client_errors():
    import pytest

    from agent_qa.llm import call_with_retries

    waited = []

    def bad_key():
        raise _HTTPError(401)

    with pytest.raises(_HTTPError):
        call_with_retries(bad_key, [10], sleep=waited.append, log=lambda *a, **k: None)
    assert waited == []


def test_provider_outage_is_error_not_fail(monkeypatch):
    from agent_qa import runner
    from agent_qa.llm import LLMError
    from agent_qa.runner import RunConfig, run
    from agent_qa.tracing import NoopTracer

    scenarios = load_scenarios(only=["BOOK-001", "SEC-001"])

    def flaky(sc, cfg, tracer):
        if sc.id == "BOOK-001":
            raise LLMError("tester call failed: HTTP 503")
        return real(sc, cfg, tracer)

    real = runner.run_scenario
    monkeypatch.setattr(runner, "run_scenario", flaky)
    r = run(scenarios, RunConfig(min_pass_rate=1.0), NoopTracer())

    assert [s.status for s in r.scenarios] == ["ERROR", "PASS"]
    assert not r.critical_failures  # an outage is not a critical bug
    assert r.pass_rate == 1.0 and r.verdict == "ERROR"


def test_judge_outage_keeps_the_conversation_and_checks(monkeypatch):
    """The judge grades quality; the gate is the checks. A judge outage -> checks decide."""
    from agent_qa import runner
    from agent_qa.judge import ProxyJudge
    from agent_qa.llm import LLMError
    from agent_qa.runner import RunConfig, run
    from agent_qa.tracing import NoopTracer

    def down(self, *a):
        raise LLMError("judge call failed: HTTP 429")

    monkeypatch.setattr(ProxyJudge, "evaluate", down)
    clean = run(load_scenarios(only=["BOOK-001"]), RunConfig(min_pass_rate=1.0), NoopTracer())
    buggy = run(load_scenarios(only=["BOOK-001"]), RunConfig(agent="buggy", min_pass_rate=1.0), NoopTracer())

    assert clean.scenarios[0].status == "PASS" and clean.scenarios[0].judge is None
    assert buggy.scenarios[0].status == "FAIL" and buggy.critical_failures  # bug still caught
    assert clean.verdict == "PASS" and buggy.verdict == "FAIL"
    assert runner  # imported for monkeypatch scope clarity
