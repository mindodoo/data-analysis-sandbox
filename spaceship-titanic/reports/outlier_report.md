# Outlier Report — Agent 2

We apply quantile winsorization on:
- `Age`, `TotalSpend`, and all spend columns

Train-only bounds:
- lower_q=0.005
- upper_q=0.995

Top clipped rows (train):
```text
         col  lower_q  upper_q  lower    upper  train_clipped_rows
  TotalSpend    0.005    0.995    0.0 18148.52                  44
 RoomService    0.005    0.995    0.0  3896.72                  44
   FoodCourt    0.005    0.995    0.0 10853.96                  44
ShoppingMall    0.005    0.995    0.0  3121.34                  44
         Spa    0.005    0.995    0.0  7803.28                  44
      VRDeck    0.005    0.995    0.0  8011.94                  44
         Age    0.005    0.995    0.0    70.00                  37
```
