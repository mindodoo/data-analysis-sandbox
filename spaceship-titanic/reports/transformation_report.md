# Transformation Report — Agent 2

Columns added:
- `Deck`, `CabinNum`, `Side` from raw `Cabin`
- `GroupId` from `PassengerId`
- `TotalSpend` from spend columns
- missing indicator flags: CryoSleep_is_missing, VIP_is_missing, HomePlanet_is_missing, Destination_is_missing, Age_is_missing, Deck_is_missing, Side_is_missing, CabinNum_is_missing, RoomService_is_missing, FoodCourt_is_missing, ShoppingMall_is_missing, Spa_is_missing, VRDeck_is_missing

Columns dropped:
- raw `Cabin`
- `Name` (not needed after parsing)
