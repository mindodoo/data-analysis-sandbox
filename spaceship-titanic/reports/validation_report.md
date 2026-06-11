# Validation Report — Agent 5

Primary model: `HistGradientBoostingClassifier` (with OHE, fold-safe group aggregates)

CV: `GroupKFold(n_splits=5)` by `GroupId`

## Fold metrics

```text
 fold  n_train  n_val  accuracy     f1  precision  recall  roc_auc
    1     6954   1739    0.8028 0.8028     0.7914  0.8145   0.8965
    2     6954   1739    0.8131 0.8062     0.8325  0.7815   0.8963
    3     6954   1739    0.8120 0.8186     0.8110  0.8264   0.9051
    4     6955   1738    0.8078 0.8035     0.8319  0.7770   0.9048
    5     6955   1738    0.7986 0.8060     0.7902  0.8224   0.8939
```

## Overall OOF metrics

```text
accuracy     0.8069
f1           0.8075
precision    0.8106
recall       0.8045
roc_auc      0.8990
```
