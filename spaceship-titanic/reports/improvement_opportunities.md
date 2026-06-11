# Improvement Opportunities Review

**Current kept result (Agent 6):** tuned `HistGradientBoostingClassifier`  
**Reference OOF accuracy (A):** 0.8108  
**Best params file:** `optimized_model/best_params.json` (unchanged)

> Kaggle test labels are hidden — all metrics are **group-safe OOF** (`GroupKFold` by `GroupId`).

## Experiment results (sorted by OOF accuracy)

```text
                                  experiment  oof_accuracy                                         notes  delta_vs_current
E. Ensemble 75% HGB + 25% LogisticRegression        0.8109                Alternative model from Agent 4            0.0001
        A. Current tuned HGB (threshold=0.5)        0.8108                          Agent 6 saved params            0.0000
          B. Threshold tune on OOF (t=0.500)        0.8108 No retrain; calibrate decision threshold only            0.0000
    F. Manual tweak (lr*0.9, +15 leaf nodes)        0.8100      Not saved; quick neighbor of best params           -0.0008
   D. Tuned HGB without group-aggregate step        0.8096                 Skips FoldSafeGroupAggregator           -0.0012
      C. Tuned HGB + drop ~0-importance cols        0.8092                                Dropped 9 cols           -0.0016
```

## Interpretation

| Finding | Implication |
|---------|-------------|
| Best experiment | **E. Ensemble 75% HGB + 25% LogisticRegression** → 0.8109 (Δ +0.0001) |
| Threshold tuning (B) | Often a **free** gain without retraining — worth applying at inference if approved |
| Feature drops (C) | Mixed; only adopt if gain is stable across reruns |
| No group agg (D) | Tests whether group step helps |
| Ensemble (E) | May help robustness; adds complexity and training cost |
| Manual tweak (F) | Small neighborhood search; not a full re-tune |

## Recommendations (for your review)

1. **Keep** Agent 6 `best_pipeline.joblib` and `best_params.json` as the canonical model unless you approve switching.
2. **Low-risk next step:** apply **threshold tuning (B)** on OOF probabilities when generating `submission.csv` (still on hold).
3. **Medium effort:** try **LightGBM/CatBoost** (not installed) with same fold-safe pipeline — often +0.01–0.02 on tabular Kaggle tasks.
4. **Higher effort:** expanded `RandomizedSearchCV` (n_iter 50+) or Optuna with group CV.

## Not evaluated in this pass (optional later)

- Target encoding with out-of-fold encoding (leakage-safe)
- Stacking multiple tree models
- Pseudo-labeling test set (risky for leaderboard)
- Re-engineering group features strictly inside each CV fold only (already done in Agent 5/6 pipeline)

## Decision needed

Please review and reply with one of:

- **Keep current** — proceed to submission generation later with saved tuned model only
- **Apply threshold** — keep model, use optimized threshold from experiment B
- **Adopt experiment X** — specify letter (B–F) to replace canonical model
- **Run extended tuning** — approve LightGBM install + deeper search

**submission.csv remains ON HOLD until you approve.**
