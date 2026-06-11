# Schema Report — Agent 1

## Column inventory

| Column | Train dtype | Test dtype | Unique (train) | Notes |
|--------|-------------|------------|----------------|-------|
| PassengerId | object | object | 8693 |  |
| HomePlanet | object | object | 3 |  |
| CryoSleep | object | object | 2 |  |
| Cabin | object | object | 6560 |  |
| Destination | object | object | 3 |  |
| Age | float64 | float64 | 80 |  |
| VIP | object | object | 2 |  |
| RoomService | float64 | float64 | 1273 |  |
| FoodCourt | float64 | float64 | 1507 |  |
| ShoppingMall | float64 | float64 | 1115 |  |
| Spa | float64 | float64 | 1327 |  |
| VRDeck | float64 | float64 | 1306 |  |
| Name | object | object | 8473 |  |
| Transported | bool | — | 2 | **Target** |

## High-cardinality / ID columns

- `PassengerId`: 8,693 unique — composite id `group_index`
- `Cabin`: 6,560 unique — parse to Deck / CabinNum / Side
- `Name`: 8,474 unique — optional surname feature

## Near-constant columns

| Column | Dominant value | Share |
|--------|----------------|-------|
| VIP | False | 95.4% |
| CryoSleep | False | 62.6% |

No fully constant feature columns detected.

## Boolean columns (raw CSV)

- `Transported`: native bool in train
- `CryoSleep`, `VIP`: object with `True`/`False` strings and NaN — coerce after imputation
