# Correlation Optimization Report — Agent 3

Top absolute Spearman correlations among numeric features (train):

```text
             feat_a                  feat_b  abs_spearman
    Side_is_missing     CabinNum_is_missing      1.000000
    Deck_is_missing     CabinNum_is_missing      1.000000
                Spa               Spa_log1p      1.000000
          Spend_Any              Spend_Zero      1.000000
         TotalSpend        Spend_Log1pTotal      1.000000
          FoodCourt         FoodCourt_log1p      1.000000
          CryoSleep CryoSleep_and_ZeroSpend      1.000000
        RoomService       RoomService_log1p      1.000000
             VRDeck            VRDeck_log1p      1.000000
       ShoppingMall      ShoppingMall_log1p      1.000000
    Deck_is_missing         Side_is_missing      1.000000
           CabinNum          CabinNumBin100      0.995549
                Age                  AgeBin      0.978870
  Spend_MeanNonZero        Spend_Log1pTotal      0.971945
         TotalSpend       Spend_MeanNonZero      0.971945
      GroupSize_all         GroupIsSolo_all      0.949814
GroupTotalSpend_all GroupSpendPerPerson_all      0.907987
 Spend_NonZeroCount              Spend_Zero      0.897847
 Spend_NonZeroCount               Spend_Any      0.897847
         Spend_Zero        Spend_Log1pTotal      0.888534
```

Recommendation: keep both raw spend columns and `TotalSpend` for tree models; for linear models, prefer `TotalSpend` + `Spend_NonZeroCount` + selected logs to reduce redundancy.
