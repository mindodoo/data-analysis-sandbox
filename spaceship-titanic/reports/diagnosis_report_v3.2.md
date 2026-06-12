# Agent 4 — Diagnosis Report (v3.1 → v3.2)

## Diagnosis Summary

Primary bottleneck: **Layer 2 (features)** — Earth/Deck-G zero-spend cluster.

| Layer | Likelihood | Evidence |
|---|---|---|
| Features | **High** | Earth 26.4%, Deck G 29.0%, 35.3% boundary errors |
| Data quality | Low | Imputed +3.3pp error only |
| Model / hparam | Low | Overfit gap 0.053, params match v2.1 LB winner |
| Sampling / splits | Low | Holdout 0.8117 aligns with OOF |

## Dispatch Decision

Route to **Agent 1 Phase C** — add 4 explicit cluster/interaction features (see
`improvement_brief_v3.2.md`). Avoid v2.2-style contextual features (proven noise).

## v3.2 Outcome vs v3.1

| Metric | v3.1 | v3.2 | Δ | Verdict |
|---|---:|---:|---:|---|
| OOF (train_split) | 0.8125 | 0.8112 | −0.0013 | regression (noise) |
| Holdout (eval_split) | 0.8117 | 0.8164 | +0.0047 | inconsistent |
| Earth error | 26.4% | 26.5% | +0.1pp | no improvement |
| Deck G error | 29.0% | 29.2% | +0.2pp | no improvement |

Success criteria: **NOT met.**

## Recommendation

1. **Submit `submission_catboost_v3.csv`** (v3.1 full-train refit) for LB anchor.
2. **Stop iterating** — v3.2 confirms Earth/Deck-G cluster remains irreducible with
   available attributes (consistent with v2.2/v2.4 lessons).
3. Keep **v2.1 `submission_catboost_v2.csv`** as LB canonical unless user prefers v3 notebook for reproducibility.

## LB Result (logged)

- `submission_catboost_v3.csv`: **LB 0.80383** (−0.00047 vs v2.1 0.80430; +0.00094 vs v1 best 0.80289).

## User Decisions

- Approved: generate v3.1 submission + run Agent 4 iteration + compare.
