"""LLM brain: a standard tool-calling loop over any OpenAI-compatible chat API."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from travel_agent.tools import TOOL_SPECS

ToolCaller = Callable[[str, dict[str, Any]], Any]
# (messages, tools) -> assistant message dict with optional "tool_calls"
ChatFn = Callable[[list[dict[str, Any]], list[dict[str, Any]] | None], dict[str, Any]]

MAX_TOOL_ROUNDS = 6


class LLMBrain:
    def __init__(self, chat: ChatFn, call_tool: ToolCaller, system_prompt: str):
        self.chat = chat
        self.call_tool = call_tool
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    def respond(self, text: str) -> str:
        self.messages.append({"role": "user", "content": text})
        for _ in range(MAX_TOOL_ROUNDS):
            msg = self.chat(self.messages, TOOL_SPECS)
            self.messages.append(msg)
            calls = msg.get("tool_calls") or []
            if not calls:
                return msg.get("content") or ""
            for c in calls:
                try:
                    args = json.loads(c["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = self.call_tool(c["function"]["name"], args)
                self.messages.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(result)})
        return "Sorry, I got stuck. Could you rephrase that?"
