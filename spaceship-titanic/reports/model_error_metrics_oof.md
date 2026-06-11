# OOF Error Metrics (Proxy for Hidden Test Labels)

Kaggle test labels are hidden, so these are computed on **group-safe OOF predictions**.

- `mse_prob` / `brier_score`: mean squared error between label and predicted probability

- `mape_eps(%)`: epsilon-stabilized MAPE (not standard for classification; interpret cautiously)


```text
            model  oof_accuracy@0.5  oof_roc_auc  mse_prob  mae_prob  log_loss  brier_score  mape_eps(%)  positive_rate@0.5
  CatBoost native          0.816979     0.900678  0.126495  0.265751  0.393307     0.126495 12904.907822           0.501438
 Stacked ensemble          0.814678     0.902771  0.125925  0.249294  0.392963     0.125925 12477.785595           0.518003
     HGB baseline          0.808582     0.899357  0.126927  0.250392  0.390220     0.126927 12433.376338           0.500403
Logistic baseline          0.782009     0.828691  0.167910  0.346123  0.519996     0.167910 15265.735655           0.598528
```
