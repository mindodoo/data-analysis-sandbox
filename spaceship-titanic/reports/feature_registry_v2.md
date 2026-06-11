# Feature Registry (v2)

Artifacts: `engineered_train_v2.parquet` (8693×30), `engineered_test_v2.parquet` (4277×29).
Raw `train.csv` / `test.csv` untouched (user requirement). All features built in
`notebooks/st_agents_v2.ipynb` Phase C.

| Feature | Version | Formula | Rationale | corr / Cramér's V with target |
|---|---|---|---|---|
| LuxurySpend | v2 | sum log1p(RoomService, Spa, VRDeck) | luxury spenders skew non-transported | −0.513 (MI 0.160 — top feature) |
| HasSpend | v2 | TotalSpend > 0 | 61–64% zero-inflation is signal | −0.482 |
| TotalSpend | v2 | sum of 5 log1p spends | overall spend level separates classes | −0.439 |
| DeckSide | v2 | Deck + "_" + Side | both individually predictive | V = 0.238 |
| BasicSpend | v2 | sum log1p(FoodCourt, ShoppingMall) | complements luxury split | −0.209 |
| AgeBin | v2 | bins [0,12,18,35,60,100] | non-linear age effect (children spike) | V = 0.133 |
| IsAlone | v2 | GroupSize == 1 | solo travelers behave differently | −0.114 |
| GroupSize | v2 | count per GroupId | group attribute (identity weak: 43.6% agreement) | +0.083 |
| CabinRegion | v2 | CabinNum // 300, clip 6 | spatial banding from Phase A | −0.043 |
| FamilySize | v2 | count per Surname ("Unknown" → 1) | surname families cross train/test | −0.031 |

## Multicollinearity notes

- Spend aggregates correlate as expected: TotalSpend↔LuxurySpend 0.91, ↔HasSpend 0.88,
  ↔BasicSpend 0.83. Fine for tree models; for linear models use either TotalSpend or
  {LuxurySpend, BasicSpend}, not both.
- No other pair exceeds |0.8|.

## Train/test consistency

- Identical feature columns and dtypes; no categories present in test but absent in train.
