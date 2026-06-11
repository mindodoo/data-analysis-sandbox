# Agent 2 — Training Report (v2, first run)

## Objective

Train and compare candidate models on the user-approved v2 feature set; beat the v1
bar (OOF 0.8108 / LB 0.80289).

## Context Received

- Read: `initial_analysis_v2.md`, `cleaning_report_v2.md`, `feature_registry_v2.md`.
- Inherited: LuxurySpend top feature; GroupKFold mandatory; fixed t=0.5; LightGBM/
  CatBoost untested in v1 (backlog).

## Split Strategy (FROZEN)

- GroupKFold(5) by `GroupId`, fold assignment computed once and reused for every
  model and every trial. Seed 42. Threshold fixed at 0.5.

## Model Comparison (identical folds, seeds, metrics)

| Model | OOF acc | fold std | train acc | overfit gap | vs v1 (0.8108) |
|---|---:|---:|---:|---:|---:|
| **catboost_tuned** | **0.8143** | 0.0076 | 0.8597 | 0.0454 | **+0.0035** |
| catboost (default) | 0.8142 | 0.0076 | 0.8899 | 0.0756 | +0.0034 |
| lightgbm_tuned | 0.8100 | 0.0059 | 0.9050 | 0.0950 | −0.0008 |
| lightgbm (default) | 0.8031 | 0.0066 | 0.9743 | 0.1712 | −0.0077 |
| hist_gradient_boosting | 0.8020 | 0.0100 | 0.9544 | 0.1523 | −0.0088 |
| logistic_regression | 0.7766 | 0.0109 | 0.7803 | 0.0037 | −0.0342 |
| majority_class | 0.5036 | — | — | — | — |

## Hyperparameters

- Tuning: random search on frozen folds — 15 LightGBM trials, 6 CatBoost trials.
- Best CatBoost: `iterations=600, learning_rate=0.08, depth=4, l2_leaf_reg=3.0`.
- Best LightGBM: `n_estimators=1200, lr=0.02, num_leaves=15, min_child_samples=10,
  colsample=0.92, subsample=0.72, reg_lambda=1.0` (still below CatBoost).

## Reasoning

- CatBoost's native categorical handling exploits DeckSide/Deck/AgeBin directly;
  ordinal encoding (HGB/LGBM path) loses information.
- CatBoost shows the smallest overfit gap among boosters (0.045) → gain is unlikely
  to be variance; fold std comparable across models.
- Dataset (8.7k rows) too small for deep learning; logistic kept as interpretable floor.

## Outputs

- `models/model_v2/final_model.joblib` (CatBoost refit on full train),
  `ordinal_encoder.joblib`, `meta.json`, `oof_predictions.npy`
- `reports/figures_v2/m_model_comparison.png`

## Risks

- v1 showed OOF↔LB mismatch; +0.0035 OOF may not fully transfer to LB.
- CatBoost native-categorical underperformed on LB in v1 (different feature set);
  Agent 3 should compare error profiles before submission.

## User Decisions

- Phase C handoff approved with this strategy. Agent 2 checkpoint: pending.

## Recommended Next Agent

Agent 3 — Evaluation, QA & Cross-Iteration Review (evaluate catboost_tuned vs ledger).
