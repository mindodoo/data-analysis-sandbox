# Model Comparison — Agent 6

Metric: **group-safe OOF** (`GroupKFold` by `GroupId`).

> Kaggle **test labels are hidden**, so true test accuracy cannot be computed locally. OOF accuracy is the best proxy we optimize.

```text
                                  stage  accuracy     f1  precision  recall  roc_auc
  Agent 5 baseline (default HGB params)    0.8069 0.8075     0.8106  0.8045   0.8990
Agent 6 tuned (best RandomizedSearchCV)    0.8108 0.8099     0.8195  0.8006   0.9002
```

**Accuracy delta (tuned - baseline):** +0.0039
