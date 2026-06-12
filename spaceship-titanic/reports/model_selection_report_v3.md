# Model Selection Report (v3)

## Selection Criteria

1. OOF accuracy on frozen GroupKFold-5 (train_split)
2. Holdout accuracy on eval_split
3. Overfit gap (train − OOF)
4. Fold stability (std)
5. Simplicity / prior LB evidence (v2.1 canonical)

## Candidates Evaluated

| Family | Rationale | Result |
|---|---|---|
| Majority class | Sanity floor | 0.5042 |
| Logistic regression | Interpretable baseline | 0.7777 |
| HistGradientBoosting | v1 canonical comparison | 0.8003 |
| LightGBM | Strong tabular baseline | 0.7992 → 0.8064 tuned |
| CatBoost | Native categoricals; v2.1 LB winner | 0.8082 → **0.8125 tuned** |

## Winner

**catboost_tuned** — OOF 0.8125, holdout 0.8117, params match v2.1 canonical.

Rejected: LightGBM (lower OOF, higher overfit gap); HGB (heavy overfit).

## Next Step

Agent 3 full evaluation before any submission.
