# 02 — Efficient attention (arXiv)

**Mode:** Research agent. The agent has an arXiv tool alongside web search; the
prompt explicitly says "using arXiv" to push it toward the arXiv tool. If a
model ignores arXiv and only web-searches, dock the efficiency point and note it.

## Prompt

> Using **arXiv**, find the paper that introduced the **Transformer**
> architecture. Report:
> - (a) its exact **arXiv identifier**,
> - (b) the **number of listed authors**,
> - (c) the **BLEU score** its *big* model reports on the **WMT 2014
>   English-to-German** translation task.
>
> Then find **two distinct arXiv papers from 2020 or later** that propose methods
> to reduce the Transformer's **O(n²)** self-attention cost. For each, give the
> **arXiv ID**, the **method name**, and the **asymptotic complexity** it claims.
>
> Finally, in **3 sentences**, explain the key trade-off these efficient-attention
> methods share. Cite an arXiv ID for every paper you mention.

## Gold answer

Part 1 — "Attention Is All You Need":
- (a) arXiv **1706.03762**
- (b) **8** authors
- (c) BLEU **28.4** (big model, WMT 2014 EN→DE)

Part 2 — any **two** distinct, real 2020+ efficient-attention papers, e.g.:
- **Linformer** — 2006.04768 — **O(n)**
- **Performer** — 2009.14794 — **O(n)**
- **Big Bird** — 2007.14062 — **O(n)** (sparse)
- **Reformer** — 2001.04451 — **O(n log n)** (LSH attention; 2020)

(FlashAttention, 2205.14135, is acceptable only if the model correctly notes it
reduces *memory/IO*, not asymptotic compute — catching that distinction is a
bonus signal, not required.)

Part 3 — the shared trade-off: these methods **approximate or sparsify full
attention** to gain speed/memory, **sacrificing exact all-pairs interaction**, so
they can underperform dense attention on tasks needing precise long-range or
global dependencies.

## Why it discriminates

- **Retrieval grounding:** the exact arXiv ID, the author count (8), and the
  precise BLEU (28.4) are hard to recall reliably — a model that doesn't open
  arXiv tends to fumble at least one.
- **Synthesis:** part 2 needs two *distinct, real, correctly-classified* papers,
  not a single famous one — tests breadth of retrieval.
- **Reasoning:** part 3 rewards understanding the actual trade-off over a generic
  "they're faster" answer.

## Scoring (out of 10)

| Pts | Criterion |
|---|---|
| 1 | Correct paper + arXiv ID 1706.03762 |
| 1 | Author count = 8 |
| 1 | BLEU = 28.4 on WMT14 EN→DE (big model) |
| 3 | Two valid 2020+ papers, each with correct ID + method + complexity (1.5 each) |
| 2 | Part-3 trade-off correctly identified (approximation/sparsity vs. exact global attention) |
| 1 | Every paper cited by arXiv ID |
| 1 | Actually used the arXiv tool (not pure web search / memory) |
