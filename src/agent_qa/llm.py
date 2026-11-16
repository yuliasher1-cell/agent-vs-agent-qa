"""Thin wrapper over any OpenAI-compatible chat API (OpenAI, Azure, OpenRouter, Ollama,
LM Studio, vLLM...). Each call is recorded as a Langfuse generation."""

from __future__ import annotations

import os
from typing import Any


class LLMClient:
    def __init__(self, role: str, model: str, tracer, temperature: float = 0.0):
        from openai import OpenAI  # optional dependency

        self.role = role
        self.model = model
        self.tracer = tracer
        self.temperature = temperature
        self.client = OpenAI(
            api_key=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL") or None,
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        json_mode: bool = False,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            kwargs["tools"] = tools
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        with self.tracer.generation(f"llm.{self.role}", model=self.model, input=messages) as gen:
            resp = self.client.chat.completions.create(**kwargs)
            m = resp.choices[0].message
            out: dict[str, Any] = {"role": "assistant", "content": m.content or ""}
            if m.tool_calls:
                out["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in m.tool_calls
                ]
            usage = resp.usage
            gen.update(
                output=out,
                usage_details={"input": usage.prompt_tokens, "output": usage.completion_tokens}
                if usage
                else None,
            )
            return out


def model_for(role: str) -> str:
    """AGENT_MODEL / TESTER_MODEL / JUDGE_MODEL, falling back to LLM_MODEL."""
    return os.getenv(f"{role.upper()}_MODEL") or os.getenv("LLM_MODEL") or "gpt-4o-mini"
