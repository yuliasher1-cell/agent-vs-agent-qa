"""CLI:  python -m agent_qa run [--agent clean|buggy] [--mode mock|llm] ...  (or: agent-qa run)"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from agent_qa import report
from agent_qa.runner import RunConfig, run
from agent_qa.scenarios import load_scenarios
from agent_qa.tracing import make_tracer


def _load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="agent-qa", description="An agent that tests another agent.")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run scenarios")
    r.add_argument("--agent", choices=["clean", "buggy"], default="clean", help="release under test")
    r.add_argument(
        "--mode", choices=["mock", "llm"], default="mock", help="default for agent, tester and judge"
    )
    r.add_argument("--agent-mode", choices=["mock", "llm"])
    r.add_argument("--tester", choices=["mock", "llm"])
    r.add_argument("--judge", choices=["mock", "llm"])
    r.add_argument("--scenario", action="append", help="id or glob, repeatable (e.g. CANCEL-*)")
    r.add_argument("--min-pass-rate", type=float, default=None)
    r.add_argument("--out", type=Path, default=Path("results"))
    r.add_argument("--langfuse", choices=["auto", "on", "off"], default="auto")
    r.add_argument("-v", "--verbose", action="store_true", help="print every transcript")

    sub.add_parser("list", help="list scenarios")

    a = p.parse_args(argv)
    _load_dotenv()
    # Windows consoles default to cp1252 and choke on "→" / "€".
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if a.cmd == "list":
        for sc in load_scenarios():
            print(f"{sc.id:<12} {sc.risk:<8} {sc.title}")
        return 0

    cfg = RunConfig(
        agent=a.agent,
        agent_mode=a.agent_mode or a.mode,
        tester_mode=a.tester or a.mode,
        judge_mode=a.judge or a.mode,
        # Real LLMs are non-deterministic: tolerate some non-critical misses.
        min_pass_rate=a.min_pass_rate if a.min_pass_rate is not None else (1.0 if a.mode == "mock" else 0.8),
    )
    tracer = make_tracer({"auto": None, "on": True, "off": False}[a.langfuse])
    scenarios = load_scenarios(only=a.scenario)
    if not scenarios:
        print("no scenarios matched", file=sys.stderr)
        return 2

    result = run(scenarios, cfg, tracer)
    print(report.console(result, verbose=a.verbose))

    name = f"{cfg.agent}-{cfg.agent_mode}-{cfg.tester_mode}-{cfg.judge_mode}"
    report.write_json(result, a.out / f"{name}.json")
    if summary := os.getenv("GITHUB_STEP_SUMMARY"):
        with open(summary, "a", encoding="utf-8") as f:
            f.write(report.markdown(result))
    if tracer.enabled:
        print("Traces sent to Langfuse.")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
