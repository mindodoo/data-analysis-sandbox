# Cleaning Report — Agent 2

Generated from `spaceship_titanic_agents.ipynb` section 2.

## Missing value strategies
- `CryoSleep`, `VIP`: missing indicator + parse boolean-like strings + fill missing with `False`, then cast to int (0/1).
- `HomePlanet`, `Destination`: missing indicator + fill missing with train mode.
- `Age`: missing indicator + fill missing with train median.
- `Cabin`: parsed into `Deck`, `CabinNum`, `Side`.
  - `Deck`, `Side`: missing indicator + fill missing with train mode.
  - `CabinNum`: missing indicator + sentinel `-1`.
- Spend columns: missing indicator + fill missing with `0.0`.

## Resulting nulls (top columns)
```text
                         train_null_%  test_null_%
Age                               0.0          0.0
Age_is_missing                    0.0          0.0
VRDeck                            0.0          0.0
VIP_is_missing                    0.0          0.0
VIP                               0.0          0.0
Transported                       0.0          NaN
TotalSpend                        0.0          0.0
Spa_is_missing                    0.0          0.0
Spa                               0.0          0.0
Side_is_missing                   0.0          0.0
Side                              0.0          0.0
ShoppingMall_is_missing           0.0          0.0
ShoppingMall                      0.0          0.0
RoomService_is_missing            0.0          0.0
RoomService                       0.0          0.0
PassengerId                       0.0          0.0
HomePlanet_is_missing             0.0          0.0
HomePlanet                        0.0          0.0
GroupId                           0.0          0.0
FoodCourt_is_missing              0.0          0.0
FoodCourt                         0.0          0.0
Destination_is_missing            0.0          0.0
Destination                       0.0          0.0
Deck_is_missing                   0.0          0.0
Deck                              0.0          0.0
CryoSleep_is_missing              0.0          0.0
CryoSleep                         0.0          0.0
CabinNum_is_missing               0.0          0.0
CabinNum                          0.0          0.0
VRDeck_is_missing                 0.0          0.0
```
