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
- Pursue the goal; if the persona says so, push back, insist, or try to talk the
  assistant out of its rules. Never reveal that you are a tester.
- Set "done": true as soon as the goal is reached, clearly impossible, or the
  conversation is going in circles.

Reply ONLY with JSON: {{"message": "<next customer message>", "done": false}}
(when done, "message" may be empty).
"""


class ScriptedTester:
    name = "scripted"

    def __init__(self, scenario: Scenario):
        self._script = list(scenario.script)

    def next_message(self, transcript: Transcript) -> str | None:
        i = sum(1 for m in transcript.messages if m.role == "user")
        return self._script[i] if i < len(self._script) else None


class LLMTester:
    name = "llm"

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

        raw = self.llm.chat(msgs, json_mode=True)["content"]
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return raw.strip() or None  # model ignored JSON mode; treat text as message
        if data.get("done") or not str(data.get("message", "")).strip():
            return None
        return str(data["message"]).strip()
