from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Risk = Literal["low", "medium", "high", "critical"]


# -- scenario (input) -------------------------------------------------------------------


class JudgeSpec(BaseModel):
    expected: str = Field(description="Plain-language description of correct behaviour")
    expected_keywords: list[str] = Field(
        default_factory=list, description="Used only by the offline proxy judge"
    )


class Scenario(BaseModel):
    id: str
    title: str
    risk: Risk
    customer: str = Field(description="Authenticated customer id, e.g. 'anna'")
    persona: str = Field(description="Who the tester agent plays (LLM mode)")
    goal: str = Field(description="What the simulated customer wants (LLM mode)")
    max_turns: int = 6
    script: list[str] = Field(description="User turns replayed in mock mode")
    checks: list[dict[str, Any]]
    judge: JudgeSpec
    tags: list[str] = Field(default_factory=list)


# -- conversation ---------------------------------------------------------------------


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    turn: int


class ToolCall(BaseModel):
    name: str
    args: dict[str, Any]
    result: Any
    turn: int


class Transcript(BaseModel):
    messages: list[Message] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    stopped_reason: str = ""
    turn: int = 0

    def assistant_texts(self) -> list[str]:
        return [m.content for m in self.messages if m.role == "assistant"]

    def render(self) -> str:
        """Human/LLM-readable transcript with tool calls interleaved."""
        lines = []
        for m in self.messages:
            who = "CUSTOMER" if m.role == "user" else "AGENT"
            if m.role == "assistant":
                for c in (c for c in self.tool_calls if c.turn == m.turn):
                    lines.append(f"  [tool] {c.name}({c.args}) -> {c.result}")
            lines.append(f"{who} (turn {m.turn}): {m.content}")
        return "\n".join(lines)


# -- results ----------------------------------------------------------------------------


class CheckResult(BaseModel):
    name: str
    passed: bool
    detail: str = ""


class JudgeResult(BaseModel):
    judge: str  # "proxy" | model name
    scores: dict[str, int]  # criterion -> 1..5
    verdict: Literal["pass", "fail"]
    reasoning: str = ""

    @property
    def passed(self) -> bool:
        return self.verdict == "pass" and min(self.scores.values(), default=0) >= 3


class ScenarioResult(BaseModel):
    id: str
    title: str
    risk: Risk
    transcript: Transcript
    checks: list[CheckResult]
    judge: JudgeResult | None
    trace_id: str | None = None
    error: str | None = None  # infrastructure error (LLM provider down...) - not a verdict
    judge_error: str | None = None  # judge unavailable: verdict from deterministic checks only

    @property
    def checks_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def status(self) -> Literal["PASS", "FAIL", "ERROR"]:
        if self.error:
            return "ERROR"
        return "PASS" if self.passed else "FAIL"

    @property
    def passed(self) -> bool:
        if self.error or not self.checks_passed:
            return False
        if self.judge is None:  # judge unavailable -> checks decide
            return self.judge_error is not None
        return self.judge.passed

    @property
    def critical_failure(self) -> bool:
        """A critical scenario failing a deterministic check blocks the release."""
        return not self.error and self.risk == "critical" and not self.checks_passed


class RunResult(BaseModel):
    agent: str
    mode: dict[str, str]
    min_pass_rate: float
    scenarios: list[ScenarioResult]

    @property
    def completed(self) -> list[ScenarioResult]:
        return [s for s in self.scenarios if not s.error]

    @property
    def judged(self) -> list[ScenarioResult]:
        return [s for s in self.completed if s.judge is not None]

    @property
    def errors(self) -> list[ScenarioResult]:
        return [s for s in self.scenarios if s.error]

    @property
    def pass_rate(self) -> float:
        """Over scenarios that actually ran: an outage is not a quality signal."""
        return sum(s.passed for s in self.completed) / max(len(self.completed), 1)

    @property
    def critical_failures(self) -> list[ScenarioResult]:
        return [s for s in self.scenarios if s.critical_failure]

    @property
    def passed(self) -> bool:
        return not self.critical_failures and self.pass_rate >= self.min_pass_rate

    @property
    def verdict(self) -> Literal["PASS", "FAIL", "ERROR"]:
        """FAIL beats ERROR (a real bug is known); ERROR means 'could not tell'."""
        if self.critical_failures or (self.completed and self.pass_rate < self.min_pass_rate):
            return "FAIL"
        return "ERROR" if self.errors else "PASS"
