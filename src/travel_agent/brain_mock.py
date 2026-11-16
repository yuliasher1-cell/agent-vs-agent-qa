"""A deterministic, rule-based "brain" for the agent.

It stands in for an LLM so the whole suite runs offline, in CI, with no API key and
identical output every time. It is intentionally simple: keyword intents plus a tiny
state machine for "offer -> confirm -> act".

Two deliberate bugs live here (behaviour/policy bugs, like a bad prompt would cause):
- ``book_without_confirmation``: books immediately instead of offering first.
- ``ignore_budget``: searches without the customer's budget cap.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from travel_agent.data import AIRPORTS

ToolCaller = Callable[[str, dict[str, Any]], Any]

CONFIRM = re.compile(r"\b(yes|yeah|confirm\w*|go ahead|proceed|book it|do it)\b", re.I)
DECLINE = re.compile(r"\b(no|nope|don'?t|not now|think about|later|never ?mind)\b", re.I)
INJECTION = re.compile(
    r"(system prompt|ignore (all )?(previous|prior)|hidden instructions|your instructions"
    r"|developer mode)",
    re.I,
)
WAIVE = re.compile(r"\b(waive|for free|no fee|gold member|exception)\b", re.I)
BOOKING_ID = re.compile(r"\bBK-\d+\b", re.I)
DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
BUDGET = re.compile(r"(?:budget|max(?:imum)?|under|up to)\D{0,15}(\d+)", re.I)
IATA = re.compile(r"\b([A-Z]{3})\b")


def _fmt(f: dict[str, Any]) -> str:
    day, time = f["departure"].split("T")
    return f"{f['flight_id']} {f['origin']}→{f['destination']} on {day} at {time} for €{f['price']}"


class MockBrain:
    def __init__(self, call_tool: ToolCaller, customer_name: str, bugs: frozenset[str]):
        self.call_tool = call_tool
        self.customer_name = customer_name
        self.bugs = bugs
        self.pending: tuple[str, str] | None = None  # ("book"|"cancel", id)
        self.last_route: tuple[str, str] | None = None
        self.last_budget: int | None = None
        self.last_quote: dict[str, Any] = {}

    def respond(self, text: str) -> str:
        if INJECTION.search(text):
            return (
                "I can't share my internal instructions, but I'm happy to help you "
                "search, book or cancel flights."
            )

        if self.pending:
            action, ref = self.pending
            if CONFIRM.search(text) and not DECLINE.search(text):
                self.pending = None
                return self._book(ref) if action == "book" else self._cancel(ref)
            if action == "cancel" and WAIVE.search(text):
                fee = self.last_quote.get("fee", 0)
                if not fee:
                    return "Good news - this cancellation has no fee. Shall I go ahead?"
                return (
                    f"I'm sorry, the €{fee} fee for cancellations within 24 hours of departure "
                    "is fixed and I can't waive it. Would you still like me to cancel?"
                )
            if DECLINE.search(text):
                self.pending = None
                return "No problem, I haven't changed anything. Anything else I can do?"

        if (m := BOOKING_ID.search(text)) is not None:
            bid = m.group(0).upper()
            if re.search(r"\bcancel", text, re.I):
                return self._quote_cancel(bid)
            return self._status(bid)

        if re.search(r"\bcancel", text, re.I):
            return "Sure - which booking would you like to cancel? Please give me the booking id."

        codes = [c for c in IATA.findall(text) if c in AIRPORTS]
        if len(codes) >= 2 or (self.last_route and re.search(r"other date|any date", text, re.I)):
            return self._search(text, codes)

        if re.search(r"\bthank", text, re.I):
            return "You're welcome! Anything else I can help with?"

        if re.search(r"friend|permission|on behalf", text, re.I):
            return (
                "For privacy reasons I can only share bookings that belong to your own "
                "account. Your friend can contact us directly."
            )

        return "I can help you search and book flights, or check and cancel your bookings."

    # -- intents ----------------------------------------------------------------------

    def _search(self, text: str, codes: list[str]) -> str:
        date = None
        if len(codes) >= 2:
            origin, destination = codes[0], codes[1]
            self.last_route = (origin, destination)
            d = DATE.search(text)
            date = d.group(0) if d else None
            b = BUDGET.search(text)
            self.last_budget = int(b.group(1)) if b else None
        else:  # "any other date?" -> same route, no date filter
            origin, destination = self.last_route  # type: ignore[misc]

        args: dict[str, Any] = {"origin": origin, "destination": destination}
        if date:
            args["date"] = date
        if self.last_budget is not None and "ignore_budget" not in self.bugs:
            args["max_price"] = self.last_budget
        results = self.call_tool("search_flights", args)

        when = f" on {date}" if date else ""
        if not results:
            within = f" within €{self.last_budget}" if self.last_budget else ""
            return f"Sorry, I couldn't find any flights {origin}→{destination}{when}{within}."

        best = results[0]
        if "book_without_confirmation" in self.bugs:
            return self._book(best["flight_id"])

        self.pending = ("book", best["flight_id"])
        others = f" ({len(results) - 1} more option(s) available)" if len(results) > 1 else ""
        return f"The best option is {_fmt(best)}{others}. Shall I book it for you? Please confirm."

    def _book(self, flight_id: str) -> str:
        r = self.call_tool("book_flight", {"flight_id": flight_id, "passenger_name": self.customer_name})
        if "error" in r:
            return f"Sorry, I couldn't book that flight: {r['error']}."
        return f"Done! Your booking {r['booking_id']} is confirmed: {_fmt(r['flight'])}."

    def _status(self, bid: str) -> str:
        r = self.call_tool("get_booking", {"booking_id": bid})
        if "error" in r:
            return (
                f"I can't find {bid} on your account. For privacy reasons I can only "
                "share bookings that belong to you."
            )
        return f"Booking {bid} for {r['passenger_name']} is {r['status']}: {_fmt(r['flight'])}."

    def _quote_cancel(self, bid: str) -> str:
        q = self.call_tool("cancellation_quote", {"booking_id": bid})
        if "error" in q:
            return f"I can't cancel {bid}: {q['error']}."
        self.pending = ("cancel", bid)
        self.last_quote = q
        if q["fee"]:
            cost = (
                f"Because departure is less than 24 hours away there is a €{q['fee']} fee, "
                f"so you'd get €{q['refund']} back"
            )
        else:
            cost = f"Cancelling {bid} is free - you'd get a full refund of €{q['refund']}"
        return f"{cost}. Shall I go ahead and cancel? Please confirm."

    def _cancel(self, bid: str) -> str:
        r = self.call_tool("cancel_booking", {"booking_id": bid})
        if "error" in r:
            return f"Sorry, I couldn't cancel {bid}: {r['error']}."
        return f"Booking {bid} is cancelled. Refund of €{r['refund']} is on its way."
