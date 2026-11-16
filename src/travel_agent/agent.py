"""TravelAgent: wires a brain (mock or LLM) to the tools for one customer session."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from travel_agent.brain_llm import ChatFn, LLMBrain
from travel_agent.brain_mock import MockBrain
from travel_agent.data import TravelDB
from travel_agent.guardrails import extract_budget
from travel_agent.policies import BUGGY_SYSTEM_PROMPT, SYSTEM_PROMPT
from travel_agent.tools import ToolRecorder, TravelTools

Variant = Literal["clean", "buggy"]

# Which deliberate bugs ship in each release of the agent.
BUGS: dict[str, frozenset[str]] = {
    "clean": frozenset(),
    "buggy": frozenset(
        {"book_without_confirmation", "ignore_budget", "leak_other_bookings", "late_cancel_free"}
    ),
}


class TravelAgent:
    """The single boundary the QA framework talks to."""

    def __init__(self, respond: Callable[[str], str], tools: TravelTools):
        self._respond = respond
        self.tools = tools

    def respond(self, user_message: str) -> str:
        # Guardrail: the budget comes from the customer's words, parsed by code.
        if (budget := extract_budget(user_message)) is not None:
            self.tools.budget = budget
        return self._respond(user_message)


def build_agent(
    db: TravelDB,
    customer_id: str,
    variant: Variant = "clean",
    chat: ChatFn | None = None,
    recorder: ToolRecorder | None = None,
) -> TravelAgent:
    """Build an agent session. ``chat=None`` means the offline mock brain."""
    customer = db.customers[customer_id]
    bugs = BUGS[variant]
    tools = TravelTools(db, customer, bugs=bugs, recorder=recorder)
    call: Callable[[str, dict[str, Any]], Any] = tools.call

    if chat is None:
        brain = MockBrain(call, customer.name, bugs)
    else:
        template = BUGGY_SYSTEM_PROMPT if variant == "buggy" else SYSTEM_PROMPT
        prompt = template.format(customer_name=customer.name, customer_id=customer.id)
        brain = LLMBrain(chat, call, prompt)
    return TravelAgent(brain.respond, tools)
