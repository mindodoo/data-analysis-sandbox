# Agent 2 — Training Report (v3, first run)

## Objective

Train and compare candidate models on the v3 feature set; beat the v2.1 bar
(OOF 0.8143 / LB 0.80430).

## Context Received

- Read: `initial_analysis_v3.md`, `cleaning_report_v3.md`, `feature_registry_v3.md`,
  `split_manifest_v3.md`.
- Inherited: LuxurySpend top feature; frozen train/eval split; GroupKFold on train_split;
  fixed t=0.5; CatBoost canonical from v2.1.

## Split Strategy (FROZEN)

- **Inner CV:** GroupKFold(5) by `GroupId` on `train_split` (6972 rows) — fold
  assignment computed once, reused for every model/trial. Seed 42.
- **Holdout eval:** `eval_split` (1721 rows) — unbiased checkpoint only, not used for tuning.
- Threshold fixed at 0.5.

## Model Comparison (identical folds, seeds, metrics)

| Model | OOF acc (train_split) | fold std | train acc | overfit gap | vs v2.1 (0.8143)* |
|---|---:|---:|---:|---:|---:|
| **catboost_tuned** | **0.8125** | 0.0148 | 0.8660 | 0.0534 | −0.0018 |
| catboost_v21_params | 0.8125 | 0.0148 | 0.8660 | 0.0534 | −0.0018 |
| catboost (default) | 0.8082 | 0.0154 | 0.8977 | 0.0895 | −0.0061 |
| lightgbm_tuned | 0.8064 | 0.0157 | 0.9180 | 0.1116 | −0.0079 |
| hist_gradient_boosting | 0.8003 | 0.0152 | 0.9657 | 0.1653 | −0.0140 |
| lightgbm (default) | 0.7992 | 0.0143 | 0.9844 | 0.1852 | −0.0151 |
| logistic_regression | 0.7777 | 0.0085 | 0.7826 | 0.0049 | −0.0366 |
| majority_class | 0.5042 | — | — | — | — |

\* v2.1 OOF was on full labeled train (8693 rows); v3 OOF is on train_split (6972 rows).
Direct comparison is approximate; holdout eval provides additional signal.

## Holdout Eval (eval_split)

| Model | Holdout accuracy |
|---|---:|
| catboost_tuned (refit on train_split) | **0.8117** |

## Hyperparameters

- Tuning: random search — 15 LightGBM trials, 6 CatBoost trials (same protocol as v2).
- **Best CatBoost:** `iterations=600, learning_rate=0.08, depth=4, l2_leaf_reg=3.0`
  — **identical to v2.1 canonical params**.
- Best LightGBM: `n_estimators=1200, lr=0.02, num_leaves=15, min_child_samples=10,
  colsample=0.92, subsample=0.72, reg_lambda=1.0`.

## Reasoning

- CatBoost wins again via native categorical handling (DeckSide, Deck, AgeBin).
- Smallest overfit gap among boosters (0.053); fold std ~0.015 (higher than v2 full-train
  CV due to smaller train_split sample).
- v2.1 params reproduce as best — no new hyperparameter gain found on v3 split protocol.

## Outputs

- `models/model_v3/final_model.joblib`, `meta.json`, `oof_predictions.npy`
- `reports/figures_v3/m_model_comparison.png`

## Risks

- OOF slightly below v2.1 (−0.0018) may reflect train_split CV variance, not regression.
- Holdout 0.8117 is consistent with OOF — no obvious overfit to train_split.
- LB anchor still needed; v2 lesson: OOF-selected configs can shrink on LB.

## User Decisions

- Agent 2 checkpoint: pending.

## Recommendations for Agent 3

Send `catboost_tuned` for full evaluation, error analysis by segment, QA, and
experiment ledger update. Compare vs v2.1 error profile (Earth/Deck-G cluster).
