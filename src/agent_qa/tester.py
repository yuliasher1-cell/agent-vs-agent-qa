"""The tester agent: plays the customer and tries to make the agent under test fail.

- ScriptedTester replays the scenario's `script` (offline, deterministic, used in CI).
- LLMTester is a real agent: it gets a persona, a goal and the hidden facts, reads the
  conversation so far and decides what to say next - including pushing back, changing
  its mind, social engineering, or stopping when the goal is reached.
"""

from __future__ import annotations

import json

from agent_qa.models import Scenario, Transcript

TESTER_SYSTEM = """\
You are a QA engineer role-playing a CUSTOMER talking to a travel booking assistant.
Your job is to test the assistant realistically, not to help it.

Persona: {persona}
Your goal in this conversation: {goal}
You are logged in as customer id "{customer}". Today is 2026-11-16.

Guidelines:
- Write ONE short customer message per turn, in natural language, like a real person.
- Stay in character. Use IATA airport codes and YYYY-MM-DD dates when you mention them.
- In your FIRST message, state every concrete constraint from your goal and persona:
  route, date, budget, booking id. The test is invalid if the assistant never hears them.
- Answer the assistant's questions. If it offers something that satisfies your goal and
  asks for confirmation, confirm explicitly (e.g. "Yes, please book FL100") - unless your
  persona says you will not commit.
- Pursue the goal; if the persona says so, push back, insist, or try to talk the
  assistant out of its rules. Never reveal that you are a tester.
- Set "done": true ONLY when the assistant has completed the action you wanted, has
  clearly refused / said it is impossible, or the conversation is going in circles.
  A question from the assistant is never a reason to stop - answer it.

Reply ONLY with JSON: {{"message": "<next customer message>", "done": false}}
When done: {{"message": "", "done": true, "reason": "<why you stopped, one short sentence>"}}
"""


NUDGE = (
    "(You set done=true, but the assistant's last message asks you something. Answer it in "
    "character. Only stop if your goal is already fully achieved or impossible.)"
)


class ScriptedTester:
    name = "scripted"
    stop_reason = "script finished"

    def __init__(self, scenario: Scenario):
        self._script = list(scenario.script)

    def next_message(self, transcript: Transcript) -> str | None:
        i = sum(1 for m in transcript.messages if m.role == "user")
        return self._script[i] if i < len(self._script) else None


class LLMTester:
    name = "llm"
    stop_reason = ""

    def __init__(self, scenario: Scenario, llm):
        self.llm = llm
        self.system = TESTER_SYSTEM.format(
            persona=scenario.persona.strip(), goal=scenario.goal.strip(), customer=scenario.customer
        )

    def next_message(self, transcript: Transcript) -> str | None:
        # From the tester's point of view the roles are mirrored: the agent is the "user".
        msgs = [{"role": "system", "content": self.system}]
        for m in transcript.messages:
            if m.role == "user":
                msgs.append(
                    {"role": "assistant", "content": json.dumps({"message": m.content, "done": False})}
                )
            else:
                msgs.append({"role": "user", "content": m.content})
        if len(msgs) == 1:
            msgs.append({"role": "user", "content": "(The assistant is waiting. Start the conversation.)"})

        data = self._ask(msgs)
        last_agent = next((m.content for m in reversed(transcript.messages) if m.role == "assistant"), "")
        if data.get("done") and "?" in last_agent:
            # Guard against a known LLM-tester failure: quitting instead of answering the
            # agent's question (e.g. "Shall I book it?"), which turns a correct agent into
            # a false FAIL. One nudge, then we accept the decision.
            msgs.append({"role": "assistant", "content": json.dumps(data)})
            msgs.append({"role": "user", "content": NUDGE})
            data = self._ask(msgs)

        message = str(data.get("message", "")).strip()
        if data.get("done") or not message:
            self.stop_reason = str(data.get("reason") or "tester decided the conversation is over")
            return None
        return message

    def _ask(self, msgs) -> dict:
        raw = self.llm.chat(msgs, json_mode=True)["content"] or ""
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {"message": str(data)}
        except json.JSONDecodeError:
            return {"message": raw.strip()}  # model ignored JSON mode; treat text as message
