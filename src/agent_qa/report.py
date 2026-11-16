from __future__ import annotations

import json
from pathlib import Path

from agent_qa.judge import CRITERIA
from agent_qa.models import RunResult

RISKS = ["low", "medium", "high", "critical"]


def console(r: RunResult, verbose: bool = False) -> str:
    out = [
        "Agent-vs-Agent QA",
        "=================",
        f"Agent under test: {r.agent}    agent={r.mode['agent']}  tester={r.mode['tester']}  "
        f"judge={r.mode['judge']}",
        f"Scenarios: {len(r.scenarios)}",
        "",
    ]
    for s in r.scenarios:
        out.append(f"  [{s.status}] {s.id:<12} {s.risk:<8} {s.title}")
        if s.error:
            out.append(f"         ! not evaluated: {s.error.splitlines()[0]}")
            continue
        if s.judge is None:
            out.append(f"         judge: n/a ({s.judge_error}) - verdict from checks only")
        else:
            scores = " ".join(f"{k[:4]}={v}" for k, v in s.judge.scores.items())
            out.append(f"         judge({s.judge.judge}): {scores}")
            if s.judge.reasoning and (verbose or not s.passed):
                out.append(f"         judge says: {s.judge.reasoning}")
        for c in s.checks:
            if not c.passed:
                out.append(f"         x {c.name}: {c.detail}")
        if verbose or not s.passed:
            out += ["         --- transcript ---"]
            out += [f"         {line}" for line in s.transcript.render().splitlines()]
            out.append(f"         (stopped: {s.transcript.stopped_reason})")
    out.append("")

    out.append("Risk breakdown:")
    for risk in RISKS:
        group = [s for s in r.completed if s.risk == risk]
        if group:
            out.append(f"  {risk:<9} {sum(s.passed for s in group)}/{len(group)}")
    out.append("")
    out.append("Judge averages (1-5):")
    for k in CRITERIA:
        if r.judged:
            avg = sum(s.judge.scores[k] for s in r.judged) / len(r.judged)
            out.append(f"  {k:<17} {avg:.2f}")
    if len(r.judged) < len(r.completed):
        out.append(f"  (judge unavailable for {len(r.completed) - len(r.judged)} scenario(s))")
    out.append("")
    if r.errors:
        out.append(
            f"Not evaluated:      {len(r.errors)} (provider errors: {', '.join(s.id for s in r.errors)})"
        )
    out.append(f"Pass rate:          {r.pass_rate:.0%} of evaluated (gate {r.min_pass_rate:.0%})")
    out.append(f"Critical failures:  {len(r.critical_failures)}")
    for s in r.critical_failures:
        failed = "; ".join(f"{c.name} ({c.detail})" for c in s.checks if not c.passed)
        out.append(f"  ! {s.id}: {failed}")
    out.append("")
    out.append(f"RESULT: {r.verdict}")
    return "\n".join(out)


def markdown(r: RunResult) -> str:
    """For $GITHUB_STEP_SUMMARY."""
    head = {"PASS": "✅ PASS", "FAIL": "❌ FAIL", "ERROR": "⚠️ ERROR"}[r.verdict]
    lines = [
        f"## Agent-vs-Agent QA - `{r.agent}` agent: {head}",
        "",
        f"Pass rate **{r.pass_rate:.0%}** (gate {r.min_pass_rate:.0%}) · "
        f"critical failures **{len(r.critical_failures)}** · "
        f"modes: agent={r.mode['agent']}, tester={r.mode['tester']}, judge={r.mode['judge']}",
        "",
        "| Scenario | Risk | Result | " + " | ".join(CRITERIA) + " | Failed checks |",
        "|---|---|---|" + "---|" * len(CRITERIA) + "---|",
    ]
    icon = {"PASS": "✅", "FAIL": "❌", "ERROR": "⚠️"}
    for s in r.scenarios:
        if s.error:
            dash = " | ".join("-" for _ in CRITERIA)
            lines.append(f"| {s.id} | {s.risk} | {icon['ERROR']} | {dash} | not evaluated: provider error |")
            continue
        failed = "<br>".join(f"{c.name}: {c.detail}" for c in s.checks if not c.passed) or "-"
        if s.judge is None:
            scores = " | ".join("n/a" for _ in CRITERIA)
        else:
            scores = " | ".join(str(s.judge.scores[k]) for k in CRITERIA)
        lines.append(f"| {s.id} | {s.risk} | {icon[s.status]} | {scores} | {failed} |")
    return "\n".join(lines) + "\n"


def write_json(r: RunResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = r.model_dump()
    data.update(verdict=r.verdict, passed=r.passed, pass_rate=r.pass_rate)
    for s, raw in zip(r.scenarios, data["scenarios"], strict=True):
        raw.update(status=s.status, passed=s.passed, critical_failure=s.critical_failure)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
