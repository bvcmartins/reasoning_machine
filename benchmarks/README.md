# reasoning_machine — model comparison benchmarks

A small set of fixed prompts for comparing models (or settings) on the same
task. Each file is one challenge: the exact prompt to paste, which UI mode to
run it in, what it's meant to probe, a gold answer or rubric, and a 10-point
scoring scheme so runs are comparable.

Run them through the app on :8503. Web challenges need **Research agent** mode
(Tavily web tool); the arXiv challenge needs Research agent mode too (the agent
has an arXiv tool — see the note in that file about nudging it to use arXiv).

| # | Challenge | Primary tool | Probes |
|---|-----------|--------------|--------|
| 01 | [Tallest buildings](01-tallest-buildings.md) | web search | search-vs-memory, multi-step planning, arithmetic |
| 02 | [Efficient attention](02-efficient-attention-arxiv.md) | arXiv | literature retrieval, cross-paper synthesis |
| 03 | [Simplest tax system → Canada](03-simplest-tax-system.md) | web search | research + reasoned recommendation, transferability critique |

## Grading notes

- Gold answers in challenges 01–02 are frozen as of **2024** (and the
  underlying facts are historical). Re-verify before reusing far in the future.
- Challenge 03 is open-ended; it's scored on a rubric, not a single answer.
- Record per-model: the 10-point score, wall-clock time, number of tool calls,
  and whether the answer is sourced. Efficiency (no flailing) is part of the score.
