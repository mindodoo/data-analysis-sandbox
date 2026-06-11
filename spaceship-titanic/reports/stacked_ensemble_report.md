# Stacked Ensemble Report

Group-aware stacking using OOF probabilities from HGB, CatBoost-native, and Logistic models.

## OOF comparison

```text
                       model  accuracy     f1  roc_auc  positive_rate
                HGB baseline    0.8086 0.8093   0.8994         0.5004
             CatBoost native    0.8170 0.8179   0.9007         0.5014
           Logistic baseline    0.7820 0.8022   0.8287         0.5985
Stacked ensemble (thr=0.460)    0.8163 0.8238   0.9028         0.5389
```

## Threshold sweep (stacked)

```text
 threshold  accuracy     f1  roc_auc  positive_rate
     0.460    0.8163 0.8238   0.9028         0.5389
     0.490    0.8163 0.8212   0.9028         0.5238
     0.465    0.8159 0.8231   0.9028         0.5370
     0.455    0.8158 0.8237   0.9028         0.5410
     0.485    0.8156 0.8210   0.9028         0.5267
     0.450    0.8155 0.8237   0.9028         0.5432
     0.495    0.8154 0.8197   0.9028         0.5205
     0.470    0.8150 0.8218   0.9028         0.5345
     0.475    0.8150 0.8213   0.9028         0.5315
     0.425    0.8149 0.8250   0.9028         0.5539
```

Selected threshold: **0.460**

## Submission files
- `submission_stacked_ensemble.csv`
- `submission_stacked_ensemble_t0.500.csv`
