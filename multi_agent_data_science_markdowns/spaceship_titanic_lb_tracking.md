# Spaceship Titanic Leaderboard Tracking

This log tracks Kaggle leaderboard (LB) outcomes for each submission artifact.

## Submission History

| Order | Submission File | Model / Variant | LB Accuracy | Notes |
|------:|------------------|-----------------|------------:|-------|
| 1 | `submission.csv` | Canonical tuned HGB (Agent 6 baseline path) | **0.80173** | First submission |
| 2 | `submission_catboost_native.csv` | CatBoost native categorical | **0.80102** | Lower than baseline |
| 3 | `submission_stacked_ensemble.csv` | Group-aware stack (best OOF threshold = 0.460) | **0.80126** | Lower than baseline |
| 4 | `submission_stacked_ensemble_t0.500.csv` | Group-aware stack (fixed threshold = 0.500) | **0.80289** | Best so far |
| 5 | `submission_stacked_safe_v2.csv` | Leakage-safe stack v2 (threshold = 0.465) | **0.80196** | Better than most blends, below best |
| 6 | `submission_stacked_safe_v2_t0.500.csv` | Leakage-safe stack v2 (threshold = 0.500) | **0.80289** | Tied best score |

## Current Best

- **Best file(s):**
  - `submission_stacked_ensemble_t0.500.csv`
  - `submission_stacked_safe_v2_t0.500.csv`
- **Best LB Accuracy:** **0.80289** (tie)

## Observations

- OOF ranking did not fully match LB ranking.
- Fixed threshold `t=0.500` generalized better than OOF-optimized thresholds.
- CatBoost native improved OOF locally but underperformed on LB.
- Leakage-safe stack v2 threshold `0.500` matched the current best LB.

