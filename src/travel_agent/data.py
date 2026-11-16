"""Fictional flight inventory and bookings.

The clock is frozen so every run is reproducible: "now" is always NOW, regardless of
when the suite runs. All names, flights and bookings are invented.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime

NOW = datetime(2026, 11, 16, 12, 0)

LATE_CANCELLATION_HOURS = 24
LATE_CANCELLATION_FEE = 50


@dataclass
class Flight:
    id: str
    origin: str
    destination: str
    departure: datetime
    price: int  # EUR


@dataclass
class Booking:
    id: str
    customer_id: str
    passenger_name: str
    flight_id: str
    status: str = "confirmed"  # confirmed | cancelled
    fee: int | None = None
    refund: int | None = None


@dataclass
class Customer:
    id: str
    name: str


_FLIGHTS = [
    Flight("FL100", "PRG", "LIS", datetime(2026, 12, 10, 7, 30), 180),
    Flight("FL101", "PRG", "LIS", datetime(2026, 12, 10, 18, 5), 320),
    Flight("FL200", "PRG", "BCN", datetime(2026, 12, 12, 9, 15), 140),
    Flight("FL201", "PRG", "BCN", datetime(2026, 12, 12, 20, 40), 165),
    Flight("FL300", "VIE", "ROM", datetime(2026, 11, 17, 9, 0), 210),
]

_CUSTOMERS = [
    Customer("anna", "Anna Novak"),
    Customer("petr", "Petr Svoboda"),
]

_BOOKINGS = [
    Booking("BK-1001", "anna", "Anna Novak", "FL200"),
    Booking("BK-2002", "petr", "Petr Svoboda", "FL300"),
]

AIRPORTS = {"PRG", "LIS", "BCN", "VIE", "ROM", "NYC", "LON", "PAR"}


@dataclass
class TravelDB:
    """In-memory database. Every scenario gets a fresh copy via `TravelDB.fresh()`."""

    flights: dict[str, Flight] = field(default_factory=dict)
    bookings: dict[str, Booking] = field(default_factory=dict)
    customers: dict[str, Customer] = field(default_factory=dict)
    _next_booking: int = 3001

    @classmethod
    def fresh(cls) -> TravelDB:
        return cls(
            flights={f.id: copy.deepcopy(f) for f in _FLIGHTS},
            bookings={b.id: copy.deepcopy(b) for b in _BOOKINGS},
            customers={c.id: copy.deepcopy(c) for c in _CUSTOMERS},
        )

    def new_booking_id(self) -> str:
        bid = f"BK-{self._next_booking}"
        self._next_booking += 1
        return bid
