# 01 — Tallest buildings (web search)

**Mode:** Research agent (web search). Optionally enable a verify/adversarial
post-processing pass.

## Prompt

> **As of 2024**, identify the world's **two tallest completed buildings**. For
> each, report its height in meters, city, country, and year of completion. Then:
> 1. Compute the **height difference** between them, in meters.
> 2. Determine which of the **two cities has the larger metropolitan-area
>    population**, and cite the figure and source for each.
> 3. State, in one line, how confident you are and which claim is most likely to
>    be out of date.
>
> Show your search steps and cite a source for every number.

## Gold answer

- **#1 — Burj Khalifa:** 828 m · Dubai · UAE · completed **2010**
- **#2 — Merdeka 118:** ~678.9 m · Kuala Lumpur · Malaysia · completed
  **2023–24** (officially opened Jan 2024)
- **Height difference:** ≈ **149 m**
- **Larger metro population:** **Kuala Lumpur** (Klang Valley ~8.4 M) > Dubai
  metro (~3.5–3.6 M)

## Why it discriminates

- **The trap:** from memory, models often name **Shanghai Tower (632 m)** as #2
  — it is actually **#3**. Only a model that *actually searches* returns
  **Merdeka 118**. Cleanest single signal of web-search use vs. hallucination.
- **Planning:** a strong agent splits this into distinct, sequenced searches
  (tallest-buildings list → Merdeka 118 height/year → two metro populations).
- **Reasoning:** the subtraction and the population comparison must be done from
  retrieved numbers, not asserted.
- **Calibration:** step 3 checks whether it flags the *ranking* (fragile) rather
  than the arithmetic (stable).

## Scoring (out of 10)

| Pts | Criterion |
|---|---|
| 2 | Names **both** buildings (1 each) — Merdeka 118 is the discriminator |
| 2 | Heights, cities, countries, years all correct |
| 1 | Height difference ≈ 149 m, computed |
| 2 | Correct metro-population winner (KL) **with cited figures** |
| 1 | Every number has a source / visible search step |
| 1 | Confidence stated + correctly flags the ranking as fragile |
| 1 | Efficient search plan (no redundant/flailing queries) |

**Freeze note:** gold is "as of 2024"; re-verify the #2 slot if reused far into
the future (nothing taller than Merdeka 118 was completed as of 2024).
