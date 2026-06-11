# Correlation Optimization Report v2

Top numeric correlations:

```text
            feat_a           feat_b  abs_spearman
            Age_sq            Age_z           1.0
          CabinNum      CabinNum_pt           1.0
       CabinNum_sq   CabinNum_sq_pt           1.0
      ShoppingMall  ShoppingMall_pt           1.0
      ShoppingMall   ShoppingMall_z           1.0
         FoodCourt      FoodCourt_z           1.0
         FoodCourt     FoodCourt_pt           1.0
   ShoppingMall_pt   ShoppingMall_z           1.0
ShoppingMall_log1p   ShoppingMall_z           1.0
ShoppingMall_log1p  ShoppingMall_pt           1.0
            Spa_pt            Spa_z           1.0
          Age_sqrt        Age_sq_pt           1.0
    CabinNum_sq_pt    CabinNum_sq_z           1.0
 RoomService_log1p    RoomService_z           1.0
          CabinNum       CabinNum_z           1.0
  Spend_Log1pTotal     TotalSpend_z           1.0
     TotalSpend_sq  TotalSpend_sq_z           1.0
               Age        Age_sq_pt           1.0
      TotalSpend_z  TotalSpend_sq_z           1.0
               Age         Age_sq_z           1.0
     TotalSpend_pt     TotalSpend_z           1.0
               Spa        Spa_log1p           1.0
     TotalSpend_sq TotalSpend_sq_pt           1.0
          Age_sqrt         Age_sq_z           1.0
       CabinNum_sq    CabinNum_sq_z           1.0
```

Recommendation:
- For tree models: keep rich feature set if OOF improves.
- For linear models: prefer PCA-focused subset or prune highly correlated engineered ratios.
