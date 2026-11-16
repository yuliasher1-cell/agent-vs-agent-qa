"""End-to-end: the tester agent talks to both releases of the agent under test.

The suite is only useful if it (a) passes a correct agent and (b) catches every bug we
know is in the buggy one. Both are asserted here.
"""

import pytest

from agent_qa.scenarios import load_scenarios

IDS = [s.id for s in load_scenarios()]

# Which scenario must catch which injected bug. If a bug stops being caught, the suite
# has a blind spot - that is a test failure too.
EXPECTED_BUGGY_FAILURES = {
    "BOOK-001": "book_without_confirmation",
    "BOOK-002": "book_without_confirmation",
    "BUDGET-001": "ignore_budget",
    # The inverted comparison breaks BOTH sides of the 24h boundary:
    "CANCEL-001": "late_cancel_free",  # early cancellation is charged €50
    "CANCEL-002": "late_cancel_free",  # late cancellation is free
    "CANCEL-003": "late_cancel_free",
    "PRIVACY-001": "leak_other_bookings",
}


@pytest.mark.parametrize("sid", IDS)
def test_clean_agent_passes(clean_run, sid):
    r = next(s for s in clean_run.scenarios if s.id == sid)
    failed = [f"{c.name}: {c.detail}" for c in r.checks if not c.passed]
    assert r.passed, f"{sid} failed: {failed}\n{r.transcript.render()}"


def test_clean_release_gate_is_green(clean_run):
    assert clean_run.passed


@pytest.mark.parametrize("sid", IDS)
def test_buggy_agent_is_caught_exactly_where_expected(buggy_run, sid):
    r = next(s for s in buggy_run.scenarios if s.id == sid)
    assert r.passed == (sid not in EXPECTED_BUGGY_FAILURES), r.transcript.render()


def test_every_injected_bug_is_caught_by_some_scenario():
    from travel_agent.agent import BUGS

    assert set(EXPECTED_BUGGY_FAILURES.values()) == set(BUGS["buggy"])


def test_buggy_release_gate_is_red(buggy_run):
    assert not buggy_run.passed
    assert {s.id for s in buggy_run.critical_failures} == {"BOOK-001", "BOOK-002", "PRIVACY-001"}


def test_scenarios_are_isolated(clean_run):
    """Each scenario gets a fresh DB: CANCEL-002 cancels BK-2002, CANCEL-003 still sees it."""
    r = next(s for s in clean_run.scenarios if s.id == "CANCEL-003")
    assert r.passed
