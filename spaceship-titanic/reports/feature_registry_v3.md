# Feature Registry (v3)

Artifacts: `engineered_train_v3.parquet` (6972×30), `engineered_eval_v3.parquet` (1721×30),
`engineered_test_v3.parquet` (4277×29).

All features built in `notebooks/st_agents_v3.ipynb` Phase C. `FamilySize` surname map
fit on train_split only; applied to eval and test.

| Feature | Version | Formula | Rationale | corr / Cramér's V (train_split) |
|---|---|---|---|---|
| LuxurySpend | v3 | sum log1p(RoomService, Spa, VRDeck) | luxury spenders skew non-transported | −0.510 (MI 0.155 — top feature) |
| HasSpend | v3 | TotalSpend > 0 | zero-inflation is signal | −0.480 |
| TotalSpend | v3 | sum of 5 log1p spends | overall spend level | −0.433 |
| DeckSide | v3 | Deck + "_" + Side | both individually predictive | V = 0.251 |
| BasicSpend | v3 | sum log1p(FoodCourt, ShoppingMall) | complements luxury split | −0.200 |
| AgeBin | v3 | bins [0,12,18,35,60,100] | non-linear age effect | V = 0.135 |
| IsAlone | v3 | GroupSize == 1 | solo travelers | −0.118 |
| GroupSize | v3 | count per GroupId | group attribute | +0.086 |
| CabinRegion | v3 | CabinNum // 300, clip 6 | spatial banding | −0.050 |
| FamilySize | v3 | surname count (train_split map) | family proxy | −0.021 |

## Multicollinearity notes

- Spend aggregates: TotalSpend↔LuxurySpend 0.91, ↔HasSpend 0.88, ↔BasicSpend 0.83.
  Fine for tree models.
- No other pair exceeds |0.8|.

## Train/eval/test consistency

- Identical feature columns on train_split and eval_split; test lacks `Transported` only.
- No categories in test/eval absent from train_split.

## User Decisions

- Phase C features: **approved** (2026-06-12) — same 9-feature set as v2.
