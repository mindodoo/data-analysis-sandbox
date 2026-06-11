# Spaceship Titanic Leaderboard Tracking

This log tracks Kaggle leaderboard (LB) outcomes for each submission artifact.

## Submission History

| Order | Submission File | Model / Variant | LB Accuracy | Notes |
|------:|------------------|-----------------|------------:|-------|
| 1 | `submission.csv` | Canonical tuned HGB (Agent 6 baseline path) | **0.80173** | First submission |
| 2 | `submission_catboost_native.csv` | CatBoost native categorical | **0.80102** | Lower than baseline |
| 3 | `submission_stacked_ensemble.csv` | Group-aware stack (best OOF threshold = 0.460) | **0.80126** | Lower than baseline |
| 4 | `submission_stacked_ensemble_t0.500.csv` | Group-aware stack (fixed threshold = 0.500) | **0.80289** | **Best so far** |
| 5 | `submission_stacked_safe_v2.csv` | Leakage-safe stack v2 (best OOF threshold = 0.465) | **0.80196** | Better than several prior stacks, below best |
| 6 | `submission_stacked_safe_v2_t0.500.csv` | Leakage-safe stack v2 (fixed threshold = 0.500) | **0.80289** | Tied best |
| 7 | `submission_catboost_v2.csv` | Multi-agent v2: new cleaning + features + tuned CatBoost (OOF 0.8143, t=0.500) | **0.80430** | **New best** — beats v1 best by +0.00141 |
| 8 | `submission_cohort_v2.csv` | v2.4 cohort-specific models by HomePlanet (cohort features + per-cohort configs, OOF 0.8181, t=0.500) | **0.80406** | 2nd best; OOF edge (+0.0038) did not transfer — consistent with per-cohort config selection bias |

## Current Best

- **Best file:** `submission_catboost_v2.csv` (multi-agent v2 global pipeline)
- **Best LB Accuracy:** **0.80430**
- Previous best: 0.80289 (v1 stacked ensembles, t=0.500)

## Quick Observations

- OOF ranking did not fully match LB ranking in this round.
- The fixed-threshold stack (`t=0.500`) generalized better than the OOF-optimized threshold (`t=0.460`).
- CatBoost native improved OOF in offline evaluation but underperformed on LB.
- Leakage-safe stack v2 improved robustness, but the `t=0.500` variant still generalized best on LB.

## Next Logging Template

Append new rows to the table above using:

| Order | Submission File | Model / Variant | LB Accuracy | Notes |
|------:|------------------|-----------------|------------:|-------|
| N | `submission_name.csv` | short description | 0.xxxxx | brief note |

