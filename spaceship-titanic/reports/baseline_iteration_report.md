# Baseline Iteration Report

Using baseline engineered features (no advanced v2/v3 features).

- OOF accuracy @0.5: **0.8108**
- OOF ROC-AUC: **0.9002**

## Top threshold candidates (by OOF accuracy)

```text
 threshold  oof_accuracy  oof_roc_auc  positive_rate
     0.500        0.8108       0.9002         0.4920
     0.480        0.8102       0.9002         0.5091
     0.485        0.8098       0.9002         0.5056
     0.445        0.8097       0.9002         0.5365
     0.455        0.8097       0.9002         0.5278
```

## Generated submission variants

- `submission_baseline_t0.500.csv`
- `submission_baseline_t0.480.csv`
- `submission_baseline_t0.485.csv`
- `submission_baseline_t0.445.csv`
- `submission_baseline_t0.455.csv`
- `submission_baseline_best_threshold.csv` (alias for top threshold)
