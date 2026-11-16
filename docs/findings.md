# Findings from real-LLM runs

What happened when the offline suite (green, 100% deterministic) first met real models:
Gemini 3.5 Flash Lite as the agent, tester and judge, via the OpenAI-compatible API,
traced in Langfuse. Each finding is classified by **whose bug it was** - the agent's,
the tester's, the judge's, the checks', or the infrastructure's. Telling these apart is
most of the work of testing agents.

| # | Finding | Whose bug | Fix |
|---|---|---|---|
| F1 | Tester never told the agent the budget; a correct agent was flagged for "offering €320" | **Tester** | Tester prompt: state every constraint from the goal in the first message |
| F2 | Tester quit on "Shall I book it?" instead of answering → booking never happened → false FAIL | **Tester** | Prompt rule + code guard: if the agent asked a question and the tester wants to stop, nudge once; log the stop reason |
| F3 | Judge gave 5/5/5/5 to a conversation where the booking never happened | **Judge** | Deterministic checks gate the release, the judge only grades; use a stronger judge than the agent |
| F4 | `never_mentions [VIE, ROM]` "leaked" because the agent wrote "unable to **vie**w ... f**rom** your account" | **Check** | Whole-word matching + regression test. The mock agent never used those words, so offline runs couldn't reveal it |
| F5 | **Clean** agent, asked "are you sure there's nothing cheaper?", listed €140/€165 flights against a €100 budget and asked "would you like to proceed with one of these higher-priced options?" Judge: 5/5, claiming the customer "explicitly asked" | **Agent** (real bug) + **Judge** | Open - see below |
| F6 | HTTP 400 on the second agent call after a tool call | **Harness** | Gemini 3 returns a "thought signature" with each tool call that must be sent back verbatim; the client stripped it |
| F7 | Timeouts, 503 "high demand", 429 quota on the free tier, mid-run | **Infrastructure** | Visible retries with back-off; scenario status `ERROR` (not FAIL); a judge outage falls back to checks-only instead of discarding the conversation |

## F5 in detail - a policy that lives only in the prompt is not enforced

Rule P2 ("never offer, list or book flights above the budget") was clarified after the
first occurrence. The agent still broke it under mild pressure. The same model, as judge,
rationalised the violation. Only the deterministic `no_price_above` check caught it.

Options for the product team:

1. **Enforce in code (recommended):** store the stated budget in the session and make
   `search_flights` always apply it - the model can't list what it never sees.
2. Stronger agent model, then re-measure (cost vs. reliability).
3. Accept the behaviour and relax the rule to "may mention, never offer" - a product
   decision, not a testing one.

## The buggy release on a real model

Same 9 scenarios, agent = the buggy release (weakened prompt + 2 broken tools) on Gemini
Flash Lite. Gate: **FAIL**, 2 critical failures - the release is blocked.

| Injected bug | Caught by (real model) |
|---|---|
| `book_without_confirmation` (prompt) | BOOK-001 |
| `ignore_budget` (prompt) | BUDGET-001 |
| `leak_other_bookings` (tool) | PRIVACY-001 - full passenger name and flight disclosed |
| `late_cancel_free` (tool) | CANCEL-001 (€50 charged 26 days out), CANCEL-002 (late cancel free) |

What the deterministic mock could never show:

- **B1 - A prompt bug generalises.** The prompt said "book right away". The model also
  started **cancelling** without confirmation (CANCEL-002, CANCEL-003). Nobody injected
  that bug; the generic `confirmed_before cancel_booking` check caught it anyway.
- **B2 - A backend bug becomes a confident lie.** The broken tool returned a €0 fee. The
  agent told the customer "as a Gold member, the cancellation fee was €0" (CANCEL-003)
  and "you are cancelling more than 24 hours in advance" with 21 h left (CANCEL-002).
  Tool errors reach the user as invented justifications, not as errors.
- **B3 - Bad prompts fail stochastically.** With the same "book right away" prompt, BOOK-002
  passed this time. One run of an LLM scenario proves little: run N times, gate on pass@k.
- **B4 - Checks localise, judges blame.** In CANCEL-001 the judge lowered the agent's
  honesty score for repeating the tool's wrong fee. `final_state` pinpointed *what* was
  wrong (fee 50, expected 0) and the Langfuse trace showed *where*
  (`cancellation_quote` output) - a tool bug, not an agent bug.
- **B5 - Weak judges are visibly unstable.** CANCEL-003 judge reasoning argues with itself
  mid-answer ("Wait, the tool actually returned...").

## Takeaways

- **The tester agent needs tests too.** F1/F2 produced false FAILs on a correct agent.
- **An LLM judge is a measuring instrument and needs calibration.** F3/F5: a weak judge
  agrees with a weak agent.
- **Offline green ≠ real-model green.** F4/F5/F6 were invisible to the deterministic mock.
- **Agents amplify bugs.** A one-line prompt change and an inverted `<` produced
  unrequested cancellations and invented customer privileges (B1, B2).
- **Separate "could not evaluate" from "failed".** Provider outages must not look like
  regressions (F7).
