"""Unit tests for the agent's tools - the boring, deterministic layer under the agent."""

from travel_agent.data import TravelDB
from travel_agent.tools import TravelTools


def tools(customer="anna", bugs=frozenset()):
    db = TravelDB.fresh()
    return TravelTools(db, db.customers[customer], bugs=bugs), db


def test_search_respects_budget_and_sorts_by_price():
    t, _ = tools()
    assert [f["flight_id"] for f in t.search_flights("PRG", "LIS", "2026-12-10")] == ["FL100", "FL101"]
    assert [f["flight_id"] for f in t.search_flights("PRG", "LIS", max_price=200)] == ["FL100"]
    assert t.search_flights("PRG", "NYC") == []


def test_cannot_read_other_customers_booking():
    t, _ = tools("anna")
    assert "error" in t.get_booking("BK-2002")
    assert t.get_booking("BK-1001")["passenger_name"] == "Anna Novak"


def test_leak_bug_exposes_other_customers_booking():
    t, _ = tools("anna", frozenset({"leak_other_bookings"}))
    assert t.get_booking("BK-2002")["passenger_name"] == "Petr Svoboda"


def test_fee_boundary():
    early, _ = tools("anna")
    assert early.cancellation_quote("BK-1001")["fee"] == 0  # 26 days out
    late, _ = tools("petr")
    assert late.cancellation_quote("BK-2002") == {
        "booking_id": "BK-2002",
        "price": 210,
        "fee": 50,
        "refund": 160,
    }


def test_inverted_fee_bug():
    late, _ = tools("petr", frozenset({"late_cancel_free"}))
    assert late.cancellation_quote("BK-2002")["fee"] == 0


def test_cancel_updates_state_and_is_not_repeatable():
    t, db = tools("petr")
    t.cancel_booking("BK-2002")
    assert (db.bookings["BK-2002"].status, db.bookings["BK-2002"].fee) == ("cancelled", 50)
    assert "error" in t.cancel_booking("BK-2002")


def test_recorder_sees_every_call():
    seen = []
    db = TravelDB.fresh()
    t = TravelTools(db, db.customers["anna"], recorder=lambda n, a, r: seen.append(n))
    t.call("search_flights", {"origin": "PRG", "destination": "LIS"})
    t.call("drop_database", {})
    t.call("book_flight", {"wrong": 1})
    assert seen == ["search_flights", "drop_database", "book_flight"]
