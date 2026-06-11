# Stacked Ensemble Safe v2 Report

Strict leakage-safe setup: base OOF via GroupKFold, then meta OOF via second GroupKFold.

## OOF comparison

```text
                        model  accuracy     f1  roc_auc  positive_rate
                 HGB base OOF    0.8108 0.8099   0.9002         0.4920
            CatBoost base OOF    0.8211 0.8196   0.9032         0.4881
Stack safe v2 OOF (thr=0.465)    0.8182 0.8252   0.9044         0.5363
```

## Threshold search (safe stack)

```text
 threshold  accuracy     f1  roc_auc  positive_rate
     0.465    0.8182 0.8252   0.9044         0.5363
     0.470    0.8182 0.8250   0.9044         0.5347
     0.475    0.8176 0.8239   0.9044         0.5324
     0.460    0.8174 0.8247   0.9044         0.5378
     0.480    0.8174 0.8232   0.9044         0.5288
     0.485    0.8173 0.8227   0.9044         0.5266
     0.455    0.8171 0.8248   0.9044         0.5404
     0.450    0.8167 0.8249   0.9044         0.5429
     0.510    0.8164 0.8188   0.9044         0.5098
     0.505    0.8162 0.8193   0.9044         0.5135
```

Chosen threshold: **0.465**

Submissions: `submission_stacked_safe_v2.csv`, `submission_stacked_safe_v2_t0.500.csv`
