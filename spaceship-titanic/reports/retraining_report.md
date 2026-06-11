# Retraining Report — Agent 6

## Hyperparameter tuning
Tuned `HistGradientBoostingClassifier` inside the fold-safe pipeline (row features + fold-safe group aggregates + OHE).

## Best parameters
```json
{
  "l2_regularization": 0.4998745790900144,
  "learning_rate": 0.08510986703590406,
  "max_bins": 128,
  "max_depth": 7,
  "max_leaf_nodes": 118,
  "min_samples_leaf": 28
}
```

## Submission
- `submission.csv`: **ON HOLD** (not generated in this run)
