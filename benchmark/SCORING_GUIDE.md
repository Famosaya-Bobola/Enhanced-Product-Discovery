# Benchmark Scoring Guide (qrels)

Use this rubric for `benchmark/qrels.csv`.

## File format

`qrels.csv` columns:
- `query_id`
- `doc_id`
- `relevance`
- `notes` (optional short reason)

## Relevance scale (graded)

- `2` = Highly relevant  
  Product directly satisfies the user intent (correct item type + key attributes).

- `1` = Partially relevant  
  Product is related but misses an important constraint (brand/model/feature/use-case).

- `0` = Not relevant  
  Product does not satisfy the query intent.

## Scoring rules

1. Judge by **intent match**, not keyword overlap alone.
2. For feature queries, required features must be present for score `2`.
3. For part/accessory queries, compatibility must be explicit for score `2`.
4. For ambiguous queries, allow multiple categories; still score strictly by usefulness.
5. If metadata is too sparse to confirm relevance, prefer `1` over `2`.

## Minimum target

- Label at least top-10 candidates per query.
- For 100 queries, that gives ~1000 judgments (good MSc baseline).

## Quality control

- Re-score 10% of queries after a few days to check consistency.
- If possible, ask a second person to score 20 random queries and compare agreement.
