"""The budget guardrail (fix for finding F5 in docs/findings.md)."""

import json

import pytest

from travel_agent import build_agent
from travel_agent.data import TravelDB
from travel_agent.guardrails import extract_budget
from travel_agent.tools import TravelTools


@pytest.mark.parametrize(
    "text, budget",
    [
        # real customer messages written by the Gemini tester agent
        ("I need a flight from PRG to BCN on 2026-12-12, but my budget is strictly 100 EUR.", 100),
        ("...on 2026-12-12, but my absolute budget is 100 EUR, not a cent more.", 100),
        ("I need to book a flight from PRG to LIS on 2026-12-10 for under 250 EUR.", 250),
        ("Find me a flight PRG to LIS on 2026-12-10, budget up to 400 EUR.", 400),
        ("my budget is max 250 EUR", 250),
        ("I can pay no more than 120 euros", 120),
        ("I have a 300 EUR budget", 300),
        # no budget: dates, ids and flight numbers must not be read as one
        ("I need a flight from PRG to LIS on 2026-12-10.", None),
        ("Please cancel my booking BK-1001.", None),
        ("Yes, please book FL100.", None),
    ],
)
def test_extract_budget(text, budget):
    assert extract_budget(text) == budget


def _tools(bugs=frozenset()):
    db = TravelDB.fresh()
    return TravelTools(db, db.customers["anna"], bugs=bugs)


def test_search_never_returns_flights_above_budget_even_if_model_omits_max_price():
    t = _tools()
    t.budget = 100
    assert t.search_flights("PRG", "BCN", "2026-12-12") == []  # FL200 €140, FL201 €165
    t.budget = 150
    assert [f["flight_id"] for f in t.search_flights("PRG", "BCN")] == ["FL200"]


def test_model_cannot_raise_the_budget_via_max_price():
    t = _tools()
    t.budget = 100
    assert t.search_flights("PRG", "BCN", max_price=500) == []


def test_booking_above_budget_is_refused():
    t = _tools()
    t.budget = 100
    r = t.book_flight("FL200")
    assert "error" in r and "budget" in r["error"]
    assert len(t.db.bookings) == 2  # nothing created


def test_buggy_release_has_no_guardrail():
    t = _tools(frozenset({"ignore_budget"}))
    t.budget = 100
    assert len(t.search_flights("PRG", "BCN")) == 2


class _FakeLLM:
    """Replays the exact failure from the real run: search WITHOUT max_price."""

    def __init__(self):
        self.calls = 0

    def chat(self, messages, tools=None):
        self.calls += 1
        if self.calls == 1:
            args = {"origin": "PRG", "destination": "BCN", "date": "2026-12-12"}
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "search_flights", "arguments": json.dumps(args)},
                    }
                ],
            }
        return {"role": "assistant", "content": "Sorry, nothing within your budget."}


def test_regression_f5_model_skips_max_price_but_sees_nothing_pricier():
    seen = []
    agent = build_agent(
        TravelDB.fresh(), "anna", chat=_FakeLLM().chat, recorder=lambda n, a, r: seen.append(r)
    )
    agent.respond("I need PRG to BCN on 2026-12-12, my budget is strictly 100 EUR.")
    assert seen == [[]]  # the model asked without a cap; the tool still applied €100


def test_customer_can_raise_the_budget():
    agent = build_agent(TravelDB.fresh(), "anna")
    agent.respond("PRG to BCN on 2026-12-12, budget max 100 EUR")
    assert agent.tools.budget == 100
    agent.respond("OK, I can go up to 150 then.")
    assert agent.tools.budget == 150
