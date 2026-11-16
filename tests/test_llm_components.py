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


def test_tester_mirrors_roles_and_stops_when_done():
    fake = FakeLLM(
        [json.dumps({"message": "Yes please", "done": False}), json.dumps({"message": "", "done": True})]
    )
    tester = LLMTester(SCENARIO, fake)
    t = Transcript(
        messages=[
            Message(role="user", content="PRG to LIS?", turn=1),
            Message(role="assistant", content="FL100 €180, book?", turn=1),
        ]
    )
    assert tester.next_message(t) == "Yes please"
    sent = fake.calls[0]
    assert sent[0]["role"] == "system" and "Anna" in sent[0]["content"]
    assert [m["role"] for m in sent[1:]] == ["assistant", "user"]  # mirrored
    assert tester.next_message(t) is None


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
