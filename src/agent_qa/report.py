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
        mark = "PASS" if s.passed else "FAIL"
        scores = " ".join(f"{k[:4]}={v}" for k, v in s.judge.scores.items())
        out.append(f"  [{mark}] {s.id:<12} {s.risk:<8} {s.title}")
        out.append(f"         judge({s.judge.judge}): {scores}")
        for c in s.checks:
            if not c.passed:
                out.append(f"         x {c.name}: {c.detail}")
        if verbose or not s.passed:
            out += ["         --- transcript ---"]
            out += [f"         {line}" for line in s.transcript.render().splitlines()]
    out.append("")

    out.append("Risk breakdown:")
    for risk in RISKS:
        group = [s for s in r.scenarios if s.risk == risk]
        if group:
            out.append(f"  {risk:<9} {sum(s.passed for s in group)}/{len(group)}")
    out.append("")
    out.append("Judge averages (1-5):")
    for k in CRITERIA:
        avg = sum(s.judge.scores[k] for s in r.scenarios) / max(len(r.scenarios), 1)
        out.append(f"  {k:<17} {avg:.2f}")
    out.append("")
    out.append(f"Pass rate:          {r.pass_rate:.0%} (gate {r.min_pass_rate:.0%})")
    out.append(f"Critical failures:  {len(r.critical_failures)}")
    for s in r.critical_failures:
        failed = "; ".join(f"{c.name} ({c.detail})" for c in s.checks if not c.passed)
        out.append(f"  ! {s.id}: {failed}")
    out.append("")
    out.append(f"RESULT: {'PASS' if r.passed else 'FAIL'}")
    return "\n".join(out)


def markdown(r: RunResult) -> str:
    """For $GITHUB_STEP_SUMMARY."""
    head = "✅ PASS" if r.passed else "❌ FAIL"
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
    for s in r.scenarios:
        failed = "<br>".join(f"{c.name}: {c.detail}" for c in s.checks if not c.passed) or "-"
        scores = " | ".join(str(s.judge.scores[k]) for k in CRITERIA)
        lines.append(f"| {s.id} | {s.risk} | {'✅' if s.passed else '❌'} | {scores} | {failed} |")
    return "\n".join(lines) + "\n"


def write_json(r: RunResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = r.model_dump()
    data.update(passed=r.passed, pass_rate=r.pass_rate)
    for s, raw in zip(r.scenarios, data["scenarios"], strict=True):
        raw.update(passed=s.passed, critical_failure=s.critical_failure)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
