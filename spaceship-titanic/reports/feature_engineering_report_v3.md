# Agent 1 — Phase C: Feature Engineering Report (v3)

## Objective

Build the user-approved 9-feature set on cleaned v3 partitions, evaluate signal on
train_split only, export modeling-ready artifacts.

## Context Received

- `cleaning_report_v3.md`, Phase B checkpoint: approved.
- Phase C feature list: approved (same as v2 proven set).
- Frozen split: train_split 6972 / eval_split 1721, seed 42.

## Actions Performed

1. Built 9 engineered features on train_split, eval_split, and test (identical logic).
2. `FamilySize` surname map fit on train_split only; mapped to eval/test (unknown → 1).
3. Evaluated correlation with target and mutual information on **train_split only**.
4. Multicollinearity heatmap; train/eval/test consistency checks.
5. Exported versioned parquet artifacts.

## Feature Signal (train_split)

| Feature | corr / V | MI rank |
|---|---|---|
| LuxurySpend | −0.510 | **#1** (0.155) |
| HasSpend | −0.480 | #2 |
| TotalSpend | −0.433 | #3 |
| DeckSide | V = 0.251 | — |
| BasicSpend | −0.200 | #5 |
| AgeBin | V = 0.135 | — |
| IsAlone | −0.118 | — |
| GroupSize | +0.086 | — |
| CabinRegion | −0.050 | — |
| FamilySize | −0.021 | — |

LuxurySpend remains the strongest feature (MI 0.155), ahead of CryoSleep (0.119).

## Multicollinearity

Expected spend-aggregate cluster only (|r| > 0.8): TotalSpend↔LuxurySpend 0.91,
↔HasSpend 0.88. No action needed for tree models.

## User Decisions

- Phase C: **approved and completed**.

## Files Generated

- `engineered_train_v3.parquet`, `engineered_eval_v3.parquet`, `engineered_test_v3.parquet`
- `feature_registry_v3.md`
- `figures_v3/c_feature_signal.png`, `figures_v3/c_multicollinearity.png`

## Recommendations for Agent 2

- Gradient boosting family (CatBoost canonical from v2.1).
- CV: GroupKFold-5 by `GroupId`, seed 42.
- Fixed threshold t=0.500.
- Beat-the-bar: v2.1 OOF 0.8143 / LB **0.80430**.
- Use `engineered_train_v3` + full labeled train (re-fit cleaning on full train for submission).
