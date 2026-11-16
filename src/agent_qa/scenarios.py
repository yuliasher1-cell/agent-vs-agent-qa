from __future__ import annotations

import fnmatch
from pathlib import Path

import yaml

from agent_qa.checks import CHECKS
from agent_qa.models import Scenario

SCENARIO_DIR = Path(__file__).resolve().parents[2] / "scenarios"


def load_scenarios(directory: Path = SCENARIO_DIR, only: list[str] | None = None) -> list[Scenario]:
    scenarios = []
    for path in sorted(directory.glob("*.yaml")):
        sc = Scenario.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        for c in sc.checks:
            if c.get("type") not in CHECKS:
                raise ValueError(f"{path.name}: unknown check type {c.get('type')!r}")
        scenarios.append(sc)

    ids = [s.id for s in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate scenario ids")

    if only:
        scenarios = [s for s in scenarios if any(fnmatch.fnmatch(s.id, p) for p in only)]
    return scenarios
