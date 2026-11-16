"""Observability: every scenario becomes one Langfuse trace.

Trace layout (Langfuse Python SDK v3, OpenTelemetry-based):

    scenario BOOK-001                        <- trace (tags: agent variant, risk, mode)
    ├── tester.turn                          <- what the tester agent decided to say
    │   └── llm.tester                       <- generation (LLM mode only)
    ├── agent.turn                           <- agent under test answering
    │   ├── llm.agent                        <- generation(s)
    │   └── tool.search_flights              <- every tool call with args + result
    ├── checks                               <- deterministic check results
    └── judge
        └── llm.judge                        <- the judge's own generation
    scores: check.*, judge.*, passed         <- attached to the trace

If the langfuse package or the LANGFUSE_* keys are missing, a no-op tracer is used and
nothing else changes - the suite must never depend on observability to run.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any


class _NoopHandle:
    def update(self, **kwargs: Any) -> None:
        pass


class NoopTracer:
    enabled = False

    @contextmanager
    def scenario(self, name: str, input: Any = None, tags=None, metadata=None):
        yield _NoopHandle()

    @contextmanager
    def span(self, name: str, input: Any = None):
        yield _NoopHandle()

    @contextmanager
    def generation(self, name: str, model: str, input: Any = None):
        yield _NoopHandle()

    def score(self, name: str, value: float, comment: str | None = None) -> None:
        pass

    def current_trace_id(self) -> str | None:
        return None

    def flush(self) -> None:
        pass


class LangfuseTracer:
    enabled = True

    def __init__(self) -> None:
        from langfuse import get_client  # optional dependency

        self.lf = get_client()
        self._root = None

    @contextmanager
    def scenario(self, name: str, input: Any = None, tags=None, metadata=None):
        with self.lf.start_as_current_span(name=name, input=input) as span:
            span.update_trace(name=name, input=input, tags=tags or [], metadata=metadata or {})
            self._root = span
            try:
                yield span
            finally:
                self._root = None

    @contextmanager
    def span(self, name: str, input: Any = None):
        with self.lf.start_as_current_span(name=name, input=input) as span:
            yield span

    @contextmanager
    def generation(self, name: str, model: str, input: Any = None):
        with self.lf.start_as_current_generation(name=name, model=model, input=input) as gen:
            yield gen

    def score(self, name: str, value: float, comment: str | None = None) -> None:
        if self._root is not None:
            self._root.score_trace(name=name, value=value, comment=comment)

    def current_trace_id(self) -> str | None:
        return getattr(self._root, "trace_id", None)

    def flush(self) -> None:
        self.lf.flush()


def make_tracer(enabled: bool | None = None):
    """enabled=None -> auto: on when LANGFUSE_PUBLIC_KEY/SECRET_KEY are set."""
    if enabled is None:
        enabled = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))
    if not enabled:
        return NoopTracer()
    try:
        return LangfuseTracer()
    except ImportError:
        print("warning: langfuse not installed (pip install -e '.[llm]'); tracing disabled")
        return NoopTracer()
