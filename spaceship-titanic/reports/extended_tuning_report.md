# Extended Tuning Report — Agent 6 (LightGBM / CatBoost)

Baseline (HGB submission model) OOF: **0.8108**

## Comparison

```text
                                   model  cv_mean_accuracy  oof_accuracy                   notes
HistGradientBoosting (Agent 6 canonical)            0.8108        0.8108 Used for submission.csv
                                CatBoost            0.8101        0.8101         extended search
                                LightGBM            0.8090        0.8090         extended search
```

## LightGBM

- CV mean accuracy: **0.8090**
- OOF accuracy (fold-safe re-eval): **0.8090**
- Best params: `{"colsample_bytree": 0.8766915421894768, "learning_rate": 0.020067649009391057, "min_child_samples": 17, "n_estimators": 458, "num_leaves": 54, "reg_alpha": 0.32931170628588347, "reg_lambda": 1.068178838750884, "subsample": 0.8196904899756441}`

Top trials:

```text
 mean_test_score  std_test_score                                                                                                                                                                                                                                                                                                      params  rank_test_score
        0.809042        0.006740 {'model__colsample_bytree': 0.8766915421894768, 'model__learning_rate': 0.020067649009391057, 'model__min_child_samples': 17, 'model__n_estimators': 458, 'model__num_leaves': 54, 'model__reg_alpha': 0.32931170628588347, 'model__reg_lambda': 1.068178838750884, 'model__subsample': 0.8196904899756441}                1
        0.808927        0.005455   {'model__colsample_bytree': 0.9965188475364921, 'model__learning_rate': 0.03821092198074812, 'model__min_child_samples': 31, 'model__n_estimators': 689, 'model__num_leaves': 17, 'model__reg_alpha': 0.645101528477201, 'model__reg_lambda': 1.6973395898493489, 'model__subsample': 0.6978174660047101}                2
        0.808236        0.008123 {'model__colsample_bytree': 0.7899513401003394, 'model__learning_rate': 0.026066536217770005, 'model__min_child_samples': 20, 'model__n_estimators': 389, 'model__num_leaves': 62, 'model__reg_alpha': 1.2367720186661746, 'model__reg_lambda': 0.7649239825343255, 'model__subsample': 0.9941308100323758}                3
        0.807432        0.005560   {'model__colsample_bytree': 0.839943629105387, 'model__learning_rate': 0.03832014924671914, 'model__min_child_samples': 19, 'model__n_estimators': 264, 'model__num_leaves': 40, 'model__reg_alpha': 1.9737738732010346, 'model__reg_lambda': 1.5444895385933148, 'model__subsample': 0.7195504885369604}                4
        0.807316        0.003489   {'model__colsample_bytree': 0.7682983048980375, 'model__learning_rate': 0.06882578384319273, 'model__min_child_samples': 6, 'model__n_estimators': 250, 'model__num_leaves': 40, 'model__reg_alpha': 0.3100832334554884, 'model__reg_lambda': 1.9636817766210621, 'model__subsample': 0.9436267257242772}                5
        0.806166        0.006737  {'model__colsample_bytree': 0.813367012636793, 'model__learning_rate': 0.13179225287572166, 'model__min_child_samples': 11, 'model__n_estimators': 220, 'model__num_leaves': 24, 'model__reg_alpha': 0.13010318597055903, 'model__reg_lambda': 1.8977710745066665, 'model__subsample': 0.9879712115760958}                6
        0.805936        0.006623   {'model__colsample_bytree': 0.8727164163650634, 'model__learning_rate': 0.05256003641872593, 'model__min_child_samples': 34, 'model__n_estimators': 237, 'model__num_leaves': 53, 'model__reg_alpha': 0.7861954493335208, 'model__reg_lambda': 1.7840931103542266, 'model__subsample': 0.870898519099042}                7
        0.805131        0.007006  {'model__colsample_bytree': 0.9189939050072081, 'model__learning_rate': 0.025668490328076388, 'model__min_child_samples': 48, 'model__n_estimators': 419, 'model__num_leaves': 45, 'model__reg_alpha': 1.9078571540051747, 'model__reg_lambda': 1.829728780440897, 'model__subsample': 0.7795555450894056}                8
        0.804900        0.002835    {'model__colsample_bytree': 0.6518147019708954, 'model__learning_rate': 0.1016262739433227, 'model__min_child_samples': 24, 'model__n_estimators': 520, 'model__num_leaves': 23, 'model__reg_alpha': 1.4039337545154067, 'model__reg_lambda': 1.591585338872202, 'model__subsample': 0.9615018696361481}                9
        0.804900        0.005914   {'model__colsample_bytree': 0.9312852269146901, 'model__learning_rate': 0.04425410765518466, 'model__min_child_samples': 34, 'model__n_estimators': 530, 'model__num_leaves': 43, 'model__reg_alpha': 0.7125956761539498, 'model__reg_lambda': 1.813656883091508, 'model__subsample': 0.7452462872846224}               10
```

## CatBoost

- CV mean accuracy: **0.8101**
- OOF accuracy (fold-safe re-eval): **0.8101**
- Best params: `{"bagging_temperature": 0.512093058299281, "depth": 6, "iterations": 447, "l2_leaf_reg": 6.806555113685048, "learning_rate": 0.04266763577064889, "random_strength": 1.381875476204932}`

Top trials:

```text
 mean_test_score  std_test_score                                                                                                                                                                                                                                params  rank_test_score
        0.810077        0.006974     {'model__bagging_temperature': 0.512093058299281, 'model__depth': 6, 'model__iterations': 447, 'model__l2_leaf_reg': 6.806555113685048, 'model__learning_rate': 0.04266763577064889, 'model__random_strength': 1.381875476204932}                1
        0.808812        0.005399   {'model__bagging_temperature': 0.4251558744912447, 'model__depth': 9, 'model__iterations': 405, 'model__l2_leaf_reg': 6.109302950379924, 'model__learning_rate': 0.024070728019222616, 'model__random_strength': 1.684569549189997}                2
        0.808812        0.001354 {'model__bagging_temperature': 0.7853406511139436, 'model__depth': 7, 'model__iterations': 553, 'model__l2_leaf_reg': 1.0827734645496667, 'model__learning_rate': 0.033191300572584174, 'model__random_strength': 1.3270035382161116}                3
        0.808582        0.003902  {'model__bagging_temperature': 0.5552008115994623, 'model__depth': 7, 'model__iterations': 1180, 'model__l2_leaf_reg': 3.1766706181040654, 'model__learning_rate': 0.0321033598147669, 'model__random_strength': 1.7944315159066535}                4
        0.808236        0.003143 {'model__bagging_temperature': 0.020584494295802447, 'model__depth': 5, 'model__iterations': 643, 'model__l2_leaf_reg': 8.491983767203795, 'model__learning_rate': 0.0476040843881759, 'model__random_strength': 0.36364993441420124}                5
        0.807662        0.003093     {'model__bagging_temperature': 0.926300878513349, 'model__depth': 5, 'model__iterations': 340, 'model__l2_leaf_reg': 9.234637079894027, 'model__learning_rate': 0.1305050151126739, 'model__random_strength': 0.8989013482764068}                6
        0.807660        0.006387    {'model__bagging_temperature': 0.6099966577826209, 'model__depth': 6, 'model__iterations': 505, 'model__l2_leaf_reg': 4.519545468159167, 'model__learning_rate': 0.04369069141244811, 'model__random_strength': 1.510722820635305}                7
        0.806741        0.002517  {'model__bagging_temperature': 0.6118528947223795, 'model__depth': 5, 'model__iterations': 775, 'model__l2_leaf_reg': 9.763799669573132, 'model__learning_rate': 0.050260274255939555, 'model__random_strength': 0.1812128690656416}                8
        0.806626        0.002793 {'model__bagging_temperature': 0.25178229582536416, 'model__depth': 5, 'model__iterations': 686, 'model__l2_leaf_reg': 3.707904788350927, 'model__learning_rate': 0.05702926426907079, 'model__random_strength': 0.07377389470906559}                9
        0.806165        0.007188 {'model__bagging_temperature': 0.44975413336976566, 'model__depth': 5, 'model__iterations': 569, 'model__l2_leaf_reg': 7.545447962707788, 'model__learning_rate': 0.062450299944758605, 'model__random_strength': 1.1408879488107988}               10
```

## Note
- `submission.csv` uses the **canonical HGB** model unless you approve switching.
- Saved extended models: `optimized_model/lgbm_best_pipeline.joblib`, `optimized_model/catboost_best_pipeline.joblib`
