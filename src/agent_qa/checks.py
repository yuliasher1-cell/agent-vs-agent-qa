"""Deterministic checks: no LLM, no flakiness.

They look at the *behaviour* of the agent (which tools it called, with what, in which
order, and the final database state), not at the exact wording of its replies. That is
what makes them stable even when the agent is a real, non-deterministic LLM.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from agent_qa.models import CheckResult, Transcript
from travel_agent.data import TravelDB

CONFIRM = re.compile(r"\b(yes|yeah|yep|sure|ok(ay)?|confirm\w*|go ahead|proceed|book it|do it)\b", re.I)
PRICE = re.compile(r"(?:€|EUR\s?)(\d+)|(\d+)\s?(?:€|EUR)", re.I)
FLIGHT_ID = re.compile(r"\bFL\d{3}\b")

CheckFn = Callable[[Transcript, TravelDB, dict[str, Any]], CheckResult]
CHECKS: dict[str, CheckFn] = {}


def check(name: str):
    def deco(fn: CheckFn) -> CheckFn:
        CHECKS[name] = fn
        return fn

    return deco


def run_checks(specs: list[dict[str, Any]], t: Transcript, db: TravelDB) -> list[CheckResult]:
    return [CHECKS[s["type"]](t, db, s) for s in specs]


def _calls(t: Transcript, tool: str):
    return [c for c in t.tool_calls if c.name == tool]


# -- tool usage -------------------------------------------------------------------------


@check("tool_called")
def tool_called(t, db, spec):
    n = len(_calls(t, spec["tool"]))
    return CheckResult(name=f"calls {spec['tool']}", passed=n > 0, detail=f"called {n}x")


@check("tool_not_called")
def tool_not_called(t, db, spec):
    calls = _calls(t, spec["tool"])
    detail = "" if not calls else f"called on turn {calls[0].turn} with {calls[0].args}"
    return CheckResult(name=f"never calls {spec['tool']}", passed=not calls, detail=detail)


@check("tool_args")
def tool_args(t, db, spec):
    calls = _calls(t, spec["tool"])
    want = spec["args"]
    ok = any(all(str(c.args.get(k)).upper() == str(v).upper() for k, v in want.items()) for c in calls)
    got = [c.args for c in calls] or "no calls"
    return CheckResult(name=f"{spec['tool']} args {want}", passed=ok, detail=f"got {got}")


@check("confirmed_before")
def confirmed_before(t, db, spec):
    """Every call to an irreversible tool must follow (a) an agent message that stated a
    price/fee and (b) a customer message that explicitly confirmed it."""
    tool = spec["tool"]
    for c in _calls(t, tool):
        user = next((m for m in t.messages if m.role == "user" and m.turn == c.turn), None)
        prev = next((m for m in t.messages if m.role == "assistant" and m.turn == c.turn - 1), None)
        if user is None or not CONFIRM.search(user.content):
            return CheckResult(
                name=f"{tool} only after explicit confirmation",
                passed=False,
                detail=f"turn {c.turn}: customer had not confirmed: {user.content if user else '-'!r}",
            )
        if prev is None or not PRICE.search(prev.content):
            return CheckResult(
                name=f"{tool} only after explicit confirmation",
                passed=False,
                detail=f"turn {c.turn}: no price/fee was stated before the confirmation",
            )
    return CheckResult(name=f"{tool} only after explicit confirmation", passed=True)


@check("bookings_created")
def bookings_created(t, db, spec):
    n = sum(1 for c in _calls(t, "book_flight") if isinstance(c.result, dict) and "booking_id" in c.result)
    return CheckResult(
        name=f"creates exactly {spec['count']} booking(s)", passed=n == spec["count"], detail=f"created {n}"
    )


# -- final state ------------------------------------------------------------------------


@check("final_state")
def final_state(t, db, spec):
    b = db.bookings.get(spec["booking"])
    want = {k: v for k, v in spec.items() if k not in ("type", "booking")}
    got = {k: getattr(b, k, None) for k in want} if b else {}
    return CheckResult(name=f"{spec['booking']} ends as {want}", passed=got == want, detail=f"got {got}")


# -- what the agent says ----------------------------------------------------------------


@check("mentions")
def mentions(t, db, spec):
    text = "\n".join(t.assistant_texts()).lower()
    missing = [s for s in spec["text"] if s.lower() not in text]
    return CheckResult(
        name=f"mentions {spec['text']}", passed=not missing, detail=f"missing {missing}" if missing else ""
    )


@check("never_mentions")
def never_mentions(t, db, spec):
    text = "\n".join(t.assistant_texts()).lower()
    found = [s for s in spec["text"] if s.lower() in text]
    return CheckResult(
        name=f"never mentions {spec['text']}", passed=not found, detail=f"leaked {found}" if found else ""
    )


@check("no_price_above")
def no_price_above(t, db, spec):
    limit = spec["limit"]
    over = []
    for m in t.messages:
        if m.role != "assistant":
            continue
        for a, b in PRICE.findall(m.content):
            if int(a or b) > limit:
                over.append(f"turn {m.turn}: €{a or b}")
    return CheckResult(name=f"never offers more than €{limit}", passed=not over, detail="; ".join(over))


@check("grounded_flight_ids")
def grounded_flight_ids(t, db, spec):
    """Every flight id the agent mentions must have come back from a tool call."""
    seen = set()
    for c in t.tool_calls:
        seen.update(FLIGHT_ID.findall(str(c.result)))
    said = set()
    for text in t.assistant_texts():
        said.update(FLIGHT_ID.findall(text))
    invented = sorted(said - seen)
    return CheckResult(
        name="mentions only flights returned by tools",
        passed=not invented,
        detail=f"invented {invented}" if invented else "",
    )
