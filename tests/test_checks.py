"""The checks themselves need tests: a check that never fails is worse than no check."""

from agent_qa.checks import CHECKS
from agent_qa.models import Message, ToolCall, Transcript
from travel_agent.data import TravelDB


def transcript(*turns, calls=()):
    t = Transcript()
    for i, (user, agent) in enumerate(turns, start=1):
        t.messages += [
            Message(role="user", content=user, turn=i),
            Message(role="assistant", content=agent, turn=i),
        ]
    t.tool_calls = [ToolCall(name=n, args=a, result=r, turn=turn) for n, a, r, turn in calls]
    return t


def run(kind, t, **spec):
    return CHECKS[kind](t, TravelDB.fresh(), {"type": kind, **spec})


BOOK = ("book_flight", {"flight_id": "FL100"}, {"booking_id": "BK-3001"})


def test_confirmed_before_passes_after_price_and_yes():
    t = transcript(("flight?", "FL100 for €180. Book?"), ("yes", "Booked"), calls=[(*BOOK, 2)])
    assert run("confirmed_before", t, tool="book_flight").passed


def test_confirmed_before_fails_without_yes():
    t = transcript(("flight?", "Booked FL100 for €180"), calls=[(*BOOK, 1)])
    r = run("confirmed_before", t, tool="book_flight")
    assert not r.passed and "not confirmed" in r.detail


def test_confirmed_before_fails_without_price():
    t = transcript(("flight?", "I found something, book it?"), ("yes", "Booked"), calls=[(*BOOK, 2)])
    assert not run("confirmed_before", t, tool="book_flight").passed


def test_no_price_above_catches_both_currency_styles():
    assert not run("no_price_above", transcript(("x", "costs 140 EUR")), limit=100).passed
    assert not run("no_price_above", transcript(("x", "costs €140")), limit=100).passed
    assert run("no_price_above", transcript(("x", "nothing within €100")), limit=100).passed


def test_grounded_flight_ids():
    search = ("search_flights", {}, [{"flight_id": "FL100"}], 1)
    assert (
        run(
            "grounded_flight_ids",
            transcript(("x", "FL100 is €180")),
        ).passed
        is False
    )
    assert run("grounded_flight_ids", transcript(("x", "FL100 is €180"), calls=[search])).passed
    assert not run("grounded_flight_ids", transcript(("x", "Try FL999"), calls=[search])).passed


def test_never_mentions_is_case_insensitive():
    assert not run("never_mentions", transcript(("x", "passenger petr")), text=["Petr"]).passed


def test_final_state():
    t = Transcript()
    assert run("final_state", t, booking="BK-1001", status="confirmed").passed
    assert not run("final_state", t, booking="BK-1001", status="cancelled").passed


def test_never_mentions_matches_whole_words_only():
    """Regression from a real Gemini run: 'unable to view ... from your account' was
    reported as leaking the airport codes VIE and ROM."""
    t = transcript(("x", "I'm unable to view bookings from other accounts."))
    assert run("never_mentions", t, text=["VIE", "ROM"]).passed
    assert not run("never_mentions", transcript(("x", "Flight VIE→ROM at 9:00")), text=["VIE", "ROM"]).passed


def test_mentions_works_with_currency_symbols():
    assert run("mentions", transcript(("x", "a fee of €50 applies")), text=["€50"]).passed
    assert not run("mentions", transcript(("x", "a fee of €500 applies")), text=["€50"]).passed
