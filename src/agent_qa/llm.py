"""Thin wrapper over any OpenAI-compatible chat API (OpenAI, Azure, OpenRouter, Ollama,
LM Studio, vLLM...). Each call is recorded as a Langfuse generation."""

from __future__ import annotations

import os
import time
from typing import Any


class LLMError(RuntimeError):
    """A readable, actionable LLM failure (shown instead of a long traceback).

    It is an *infrastructure* error, not a test result: the runner marks the scenario
    as ERROR and carries on with the next one.
    """


RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


def retry_waits() -> list[float]:
    """Seconds to wait before each retry. LLM_RETRY_WAITS="10,30,60" by default."""
    raw = os.getenv("LLM_RETRY_WAITS", "10,30,60")
    return [float(x) for x in raw.split(",") if x.strip()]


def is_retryable(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None)
    if status is not None:
        return status in RETRYABLE_STATUS
    return exc.__class__.__name__ in {"APITimeoutError", "APIConnectionError"}


def call_with_retries(fn, waits: list[float], sleep=time.sleep, log=print):
    """Call fn(); on transient provider errors (overload, rate limit, timeout) wait and
    retry. Free tiers fail like this all the time - it must not kill a test run."""
    for attempt in range(len(waits) + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == len(waits) or not is_retryable(exc):
                raise
            what = getattr(exc, "status_code", None) or exc.__class__.__name__
            log(f" {what}, retry in {waits[attempt]:.0f}s ...", end="", flush=True)
            sleep(waits[attempt])


def _body(exc) -> str:
    """The provider's own error text - usually says exactly what is wrong."""
    try:
        return exc.response.text[:800]
    except Exception:
        return str(exc)


def _hint(status: int) -> str:
    return {
        400: "The provider rejected the request (unsupported parameter or model feature).",
        401: "The API key is wrong or missing (LLM_API_KEY).",
        403: "The key has no access to this model/endpoint.",
        404: "Model not found - check LLM_MODEL / AGENT_MODEL / TESTER_MODEL / JUDGE_MODEL.",
        429: "Rate limit or free-tier quota exceeded - wait a minute or use a smaller run.",
    }.get(status, "Provider-side error - try again later.")


class LLMClient:
    def __init__(self, role: str, model: str, tracer, temperature: float = 0.0):
        from openai import OpenAI  # optional dependency

        self.role = role
        self.model = model
        self.tracer = tracer
        self.temperature = temperature
        # Never hang forever: bounded timeout. Retries are ours (visible, with long
        # back-off), not the SDK's silent ones that look like a frozen terminal.
        self.client = OpenAI(
            api_key=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL") or None,
            timeout=float(os.getenv("LLM_TIMEOUT", "120")),
            max_retries=0,
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

        import openai

        with self.tracer.generation(f"llm.{self.role}", model=self.model, input=messages) as gen:
            print(f"      {self.role} -> {self.model} ...", end="", flush=True)
            started = time.monotonic()
            try:
                resp = call_with_retries(lambda: self.client.chat.completions.create(**kwargs), retry_waits())
            except openai.APIStatusError as exc:
                print(" error", flush=True)
                raise LLMError(
                    f"{self.role} call to {self.model} failed: HTTP {exc.status_code}. "
                    f"{_hint(exc.status_code)}\n{_body(exc)}"
                ) from None
            except (openai.APITimeoutError, openai.APIConnectionError) as exc:
                print(" error", flush=True)
                raise LLMError(
                    f"{self.role} call to {self.model} failed: {exc.__class__.__name__} "
                    "(no answer from the server after retries; the model may be overloaded)"
                ) from None
            print(f" {time.monotonic() - started:.1f}s", flush=True)
            m = resp.choices[0].message
            out: dict[str, Any] = {"role": "assistant", "content": m.content or ""}
            if m.tool_calls:
                out["content"] = m.content  # null, not "", next to tool calls (OpenAI format)
                # Keep the FULL tool call, including provider-specific extras (e.g. Gemini 3
                # "thought signatures" in extra_content). They must be sent back verbatim with
                # the tool result, otherwise the next request is rejected with HTTP 400.
                out["tool_calls"] = [tc.model_dump(exclude_none=True) for tc in m.tool_calls]
            extra = getattr(m, "model_extra", None) or {}
            if extra.get("extra_content") is not None:  # message-level provider extras
                out["extra_content"] = extra["extra_content"]
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
