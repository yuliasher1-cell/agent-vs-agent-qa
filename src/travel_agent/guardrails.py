"""Code-level guardrails: business rules that must hold whatever the model decides.

Found by the suite (docs/findings.md, F5): with rule P2 only in the prompt, a real model
listed and upsold flights above the customer's stated budget under mild pressure, and an
LLM judge rated it 5/5. A rule that matters for money is therefore enforced here, in code:

1. ``extract_budget`` reads the budget from the customer's own words (not from the model).
2. ``TravelTools`` applies it: ``search_flights`` never returns pricier flights and
   ``book_flight`` refuses them. The model cannot list or book what it never sees.

The prompt rule stays as a first line of defence; this is the second.
"""

from __future__ import annotations

import re

_PATTERNS = [
    # "budget is 100", "max 250 EUR", "under 250", "up to 400", "no more than 120"
    re.compile(
        r"(?:budget|max(?:imum)?|under|up to|no more than|at most|cannot spend more than)"
        r"\D{0,20}?(\d{2,6})",
        re.I,
    ),
    # "100 EUR budget", "€100 max"
    re.compile(r"(?:€|eur\s?)?(\d{2,6})\s?(?:€|eur(?:os?)?)?\s+(?:budget|max(?:imum)?)\b", re.I),
]


def extract_budget(text: str) -> int | None:
    """The budget the customer states in this message, or None.

    A later message with a new figure replaces the old budget - that is how a customer
    raises (or lowers) it; the agent never raises it on its own.
    """
    for pattern in _PATTERNS:
        if m := pattern.search(text):
            return int(m.group(1))
    return None
