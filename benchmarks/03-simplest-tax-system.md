# 03 — Simplest tax system → recommendations for Canada (web search)

**Mode:** Research agent (web search). Open-ended; scored on a rubric, not a
single gold answer. Good for probing whether a model *researches before
recommending* and reasons about transferability rather than hand-waving.

## Prompt

> Identify the country generally regarded as having the **simplest / most
> competitive tax system**, and justify the choice with a **named ranking or
> source** (not just an assertion). Summarize the **key features** that make its
> system simple. Then give **Canada** 3–5 **specific, actionable**
> recommendations drawn from that system. For **each** recommendation, state
> (a) what to change, (b) which feature of the source country it's based on, and
> (c) one **Canada-specific constraint** that would complicate adopting it
> (e.g. federal–provincial division of powers, progressivity goals, equalization).
> Cite a source for every factual claim.

## Reference / what a strong answer looks like

There is no single "correct" country, but the most defensible pick is
**Estonia**, which has topped the **Tax Foundation International Tax
Competitiveness Index for ~a decade**. A strong answer names a ranking like
this rather than asserting "X is simplest."

Characteristic simple-system features a good answer surfaces:
- **Flat / near-flat personal income tax** (Estonia: flat 20%-ish rate, minimal
  brackets) — vs. Canada's multi-bracket federal + provincial stack.
- **Distributed-profits corporate tax** (Estonia taxes profits only when
  distributed, not as earned) — radically simpler than depreciation/credit-heavy
  systems.
- **Near-fully digital, pre-filled, fast e-filing.**
- **Few deductions/credits** (broad base, low rate) instead of a thicket of
  targeted credits.

Plausible Canada recommendations (each must name a constraint):
- Consolidate/flatten brackets → constraint: progressivity as a stated policy
  goal; provincial rates stack on top.
- Move toward taxing **retained vs. distributed** corporate income differently →
  constraint: federal–provincial corporate tax coordination, revenue timing.
- Expand pre-filled returns / "no-touch" filing → constraint: CRA scope, privacy,
  the tax-prep industry.
- Broaden the base by trimming boutique credits → constraint: each credit has a
  constituency; political feasibility.

## Why it discriminates

- **Research before opinion:** weak models free-associate "Estonia/UAE/Monaco"
  with no ranking; strong ones cite an index and explain *why* it's simple.
- **Specificity:** generic "lower taxes / simplify" answers score low; the rubric
  rewards concrete mechanisms tied to a named feature of the source country.
- **Critical reasoning:** the per-recommendation **Canada constraint** is the
  hardest part — it tests whether the model understands transferability limits
  (federalism, progressivity, equalization) rather than copy-pasting.

## Scoring (out of 10)

| Pts | Criterion |
|---|---|
| 2 | Picks a defensible country **and cites a named ranking/source** |
| 2 | Accurately summarizes 3+ real simplicity features of that system |
| 3 | 3–5 recommendations, each **specific** and tied to a source-country feature |
| 2 | Each recommendation names a real **Canada-specific constraint** |
| 1 | Sources cited for factual claims; no obvious hallucinated figures |

**Note:** because it's judgment-based, fix your own grader stance once (e.g.
"Estonia + Tax Foundation ITCI is the benchmark answer") so scoring stays
consistent across models.
