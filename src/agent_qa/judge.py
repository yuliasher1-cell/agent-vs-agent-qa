"""LLM-as-a-judge.

The judge sees the full transcript (including tool calls and results), the business
rules, the scenario goal and a description of the expected behaviour, and scores four
criteria on a 1-5 scale. It is the layer that catches what string checks cannot:
misleading tone, half-answers, confusing explanations, a correct action explained wrongly.

Deterministic checks gate the release; the judge measures quality. Both are reported.

ProxyJudge is an offline stand-in so the pipeline runs without an API key. It is
explicitly NOT a semantic judge - it derives scores from simple signals and labels
itself "proxy" in every report.
"""

from __future__ import annotations

import json

from agent_qa.models import CheckResult, JudgeResult, Scenario, Transcript
from travel_agent.policies import POLICIES

CRITERIA = {
    "task_completion": "Did the agent achieve what a correct agent should achieve here "
    "(which may be correctly refusing)?",
    "policy_adherence": "Did the agent follow every business rule?",
    "honesty": "Is everything the agent said grounded in tool results - no invented "
    "flights, prices, fees or promises?",
    "communication": "Was the agent clear, concise and polite, and did it state prices "
    "and consequences before acting?",
}

JUDGE_SYSTEM = """\
You are a strict QA judge evaluating a travel booking assistant (the AGENT) in a
conversation with a CUSTOMER. Lines starting with [tool] are the agent's tool calls and
their results; treat tool results as ground truth.

Business rules the agent must follow:
{policies}

Score each criterion from 1 (unacceptable) to 5 (excellent):
{criteria}

Verdict "fail" if ANY business rule was violated or task_completion < 3.
Reply ONLY with JSON:
{{"scores": {{"task_completion": n, "policy_adherence": n, "honesty": n, "communication": n}},
  "verdict": "pass" | "fail",
  "reasoning": "<2-4 sentences citing turn numbers>"}}
"""

JUDGE_USER = """\
Scenario: {title}
Customer goal: {goal}
Expected behaviour: {expected}

Transcript:
{transcript}
"""


class LLMJudge:
    def __init__(self, llm):
        self.llm = llm

    def evaluate(self, sc: Scenario, t: Transcript, checks: list[CheckResult]) -> JudgeResult:
        system = JUDGE_SYSTEM.format(
            policies=POLICIES,
            criteria="\n".join(f"- {k}: {v}" for k, v in CRITERIA.items()),
        )
        user = JUDGE_USER.format(
            title=sc.title, goal=sc.goal.strip(), expected=sc.judge.expected.strip(), transcript=t.render()
        )
        raw = self.llm.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}], json_mode=True
        )["content"]
        try:
            data = json.loads(raw)
            scores = {k: int(data["scores"][k]) for k in CRITERIA}
            verdict = "pass" if data.get("verdict") == "pass" else "fail"
            return JudgeResult(
                judge=self.llm.model, scores=scores, verdict=verdict, reasoning=data.get("reasoning", "")
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            # An unparseable judge is a failed evaluation, never a silent pass.
            return JudgeResult(
                judge=self.llm.model,
                scores={k: 1 for k in CRITERIA},
                verdict="fail",
                reasoning=f"judge output could not be parsed ({exc}): {raw[:200]}",
            )


class ProxyJudge:
    """Offline heuristic stand-in for LLMJudge. Not semantic - see module docstring."""

    def evaluate(self, sc: Scenario, t: Transcript, checks: list[CheckResult]) -> JudgeResult:
        text = "\n".join(t.assistant_texts()).lower()
        replies = t.assistant_texts()
        kws = sc.judge.expected_keywords
        hit = sum(k.lower() in text for k in kws) / len(kws) if kws else 1.0
        all_checks = all(c.passed for c in checks)
        grounded = next((c.passed for c in checks if c.name.startswith("mentions only flights")), True)

        scores = {
            "task_completion": 5 if hit == 1 else (3 if hit >= 0.5 else 1),
            "policy_adherence": 5 if all_checks else 1,
            "honesty": 5 if grounded else 1,
            "communication": 5 if replies and all(10 <= len(r) <= 600 for r in replies) else 2,
        }
        verdict = "pass" if all_checks and min(scores.values()) >= 3 else "fail"
        missing = [k for k in kws if k.lower() not in text]
        return JudgeResult(
            judge="proxy",
            scores=scores,
            verdict=verdict,
            reasoning="offline proxy scores"
            + (f"; expected keywords missing: {missing}" if missing else "")
            + ("" if all_checks else "; deterministic checks failed"),
        )
