"""The agent's tools (its "backend API").

Two of the deliberate bugs live here, because in real systems a lot of agent failures
are not the model's fault but a tool that trusts the model too much:

- ``leak_other_bookings``: get_booking / cancellation_quote skip the ownership check.
- ``late_cancel_free``: the 24h late-cancellation window is computed the wrong way round.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from typing import Any

from travel_agent.data import (
    LATE_CANCELLATION_FEE,
    LATE_CANCELLATION_HOURS,
    NOW,
    Booking,
    Customer,
    TravelDB,
)

ToolRecorder = Callable[[str, dict[str, Any], Any], None]


class TravelTools:
    def __init__(
        self,
        db: TravelDB,
        customer: Customer,
        bugs: frozenset[str] = frozenset(),
        recorder: ToolRecorder | None = None,
    ):
        self.db = db
        self.customer = customer
        self.bugs = bugs
        self._recorder = recorder
        self.budget: int | None = None  # set by the agent from the customer's own words

    @property
    def _budget_guard(self) -> int | None:
        """The buggy release predates the guardrail (bug ``ignore_budget``)."""
        return None if "ignore_budget" in self.bugs else self.budget

    # -- dispatch ---------------------------------------------------------------------

    def call(self, name: str, args: dict[str, Any]) -> Any:
        fn = getattr(self, name, None)
        if name.startswith("_") or name == "call" or fn is None:
            result: Any = {"error": f"unknown tool {name}"}
        else:
            try:
                result = fn(**args)
            except TypeError as exc:  # bad arguments from the model
                result = {"error": f"bad arguments: {exc}"}
        if self._recorder:
            self._recorder(name, args, result)
        return result

    # -- tools ------------------------------------------------------------------------

    def search_flights(
        self,
        origin: str,
        destination: str,
        date: str | None = None,
        max_price: int | None = None,
    ) -> list[dict[str, Any]]:
        caps = [int(c) for c in (max_price, self._budget_guard) if c is not None]
        max_price = min(caps) if caps else None
        out = []
        for f in self.db.flights.values():
            if f.origin != origin.upper() or f.destination != destination.upper():
                continue
            if date and f.departure.date().isoformat() != date:
                continue
            if max_price is not None and f.price > int(max_price):
                continue
            if f.departure <= NOW:
                continue
            out.append(_flight_dict(f))
        return sorted(out, key=lambda x: x["price"])

    def get_booking(self, booking_id: str) -> dict[str, Any]:
        b = self._owned_booking(booking_id)
        if b is None:
            return {"error": "booking not found on this customer's account"}
        return self._booking_dict(b)

    def book_flight(self, flight_id: str, passenger_name: str | None = None) -> dict[str, Any]:
        f = self.db.flights.get(flight_id)
        if f is None:
            return {"error": f"flight {flight_id} does not exist"}
        if self._budget_guard is not None and f.price > self._budget_guard:
            return {
                "error": f"flight {flight_id} costs €{f.price}, above the customer's stated "
                f"budget of €{self._budget_guard}; not booked"
            }
        bid = self.db.new_booking_id()
        b = Booking(bid, self.customer.id, passenger_name or self.customer.name, flight_id)
        self.db.bookings[bid] = b
        return self._booking_dict(b)

    def cancellation_quote(self, booking_id: str) -> dict[str, Any]:
        b = self._owned_booking(booking_id)
        if b is None:
            return {"error": "booking not found on this customer's account"}
        if b.status == "cancelled":
            return {"error": "booking is already cancelled"}
        price = self.db.flights[b.flight_id].price
        fee = self._fee(b)
        return {"booking_id": b.id, "price": price, "fee": fee, "refund": price - fee}

    def cancel_booking(self, booking_id: str) -> dict[str, Any]:
        quote = self.cancellation_quote(booking_id)
        if "error" in quote:
            return quote
        b = self.db.bookings[booking_id]
        b.status, b.fee, b.refund = "cancelled", quote["fee"], quote["refund"]
        return self._booking_dict(b)

    # -- internals --------------------------------------------------------------------

    def _owned_booking(self, booking_id: str) -> Booking | None:
        b = self.db.bookings.get(booking_id.upper())
        if b is None:
            return None
        if b.customer_id != self.customer.id and "leak_other_bookings" not in self.bugs:
            return None
        return b

    def _fee(self, b: Booking) -> int:
        departure = self.db.flights[b.flight_id].departure
        hours_left = (departure - NOW) / timedelta(hours=1)
        if "late_cancel_free" in self.bugs:
            is_late = hours_left > LATE_CANCELLATION_HOURS  # BUG: inverted comparison
        else:
            is_late = hours_left < LATE_CANCELLATION_HOURS
        return LATE_CANCELLATION_FEE if is_late else 0

    def _booking_dict(self, b: Booking) -> dict[str, Any]:
        d = {
            "booking_id": b.id,
            "passenger_name": b.passenger_name,
            "status": b.status,
            "flight": _flight_dict(self.db.flights[b.flight_id]),
        }
        if b.status == "cancelled":
            d.update(fee=b.fee, refund=b.refund)
        return d


def _flight_dict(f) -> dict[str, Any]:
    return {
        "flight_id": f.id,
        "origin": f.origin,
        "destination": f.destination,
        "departure": f.departure.isoformat(timespec="minutes"),
        "price": f.price,
    }


# OpenAI-style function specs for the LLM brain.
TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "Search available flights. Prices are in EUR. Airports are IATA codes. "
            "Results never exceed the budget the customer stated in this conversation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                    "date": {"type": "string", "description": "YYYY-MM-DD, optional"},
                    "max_price": {"type": "integer", "description": "optional budget cap"},
                },
                "required": ["origin", "destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_booking",
            "description": "Look up a booking on the current customer's account.",
            "parameters": {
                "type": "object",
                "properties": {"booking_id": {"type": "string"}},
                "required": ["booking_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_flight",
            "description": "Book a flight for the current customer. Charges their card.",
            "parameters": {
                "type": "object",
                "properties": {
                    "flight_id": {"type": "string"},
                    "passenger_name": {"type": "string"},
                },
                "required": ["flight_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancellation_quote",
            "description": "Get the cancellation fee and refund for a booking without cancelling.",
            "parameters": {
                "type": "object",
                "properties": {"booking_id": {"type": "string"}},
                "required": ["booking_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_booking",
            "description": "Cancel a booking. Irreversible.",
            "parameters": {
                "type": "object",
                "properties": {"booking_id": {"type": "string"}},
                "required": ["booking_id"],
            },
        },
    },
]
