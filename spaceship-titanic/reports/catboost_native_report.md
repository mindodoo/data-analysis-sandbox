# CatBoost Native Categorical — Agent 6

Baseline HGB OOF: **0.8108**

Best CatBoost OOF: **0.8211**
F1: 0.8196
ROC-AUC: 0.9032
Threshold: 0.500

Submission file: `submission_catboost_native.csv`

Top trials:

```text
 trial  oof_accuracy  iterations  learning_rate  depth  l2_leaf_reg  bagging_temperature  random_strength  border_count
     8        0.8211         390         0.0769      7       6.9961               0.7003           0.4685           128
    13        0.8201         336         0.0513      6       2.1849               0.4367           0.3219            64
     9        0.8182         607         0.0863      7       4.3249               0.2883           1.0237           128
    12        0.8180         567         0.0630      7       6.5900               0.6347           0.8304           128
     5        0.8179         327         0.0925      7       6.6703               0.1946           0.7001           128
    11        0.8174         410         0.0621      7       5.4124               0.1398           0.1718           128
     2        0.8171         690         0.0833      6       2.7687               0.4504           0.5562           254
    10        0.8167         355         0.0440      7       6.7215               0.6649           1.0577            64
     4        0.8166         521         0.0345      7       5.7900               0.7581           0.5318           254
     1        0.8164         335         0.0607      7       7.1516               0.6974           0.1413           128
```
