# Model Selection — Agent 4 (recommendation)

**Holdout split:** stratified `train/val` with test_size=0.2, random_state=42

## Candidate comparison

```text
                                    model  holdout_accuracy
HistGradientBoostingClassifier(OHE dense)          0.807361
                  LogisticRegression(OHE)          0.784359
```

## Recommendation (for Agent 5/6)

**Primary model:** `HistGradientBoostingClassifier(OHE dense)`

### Reasoning
- Non-linear decision boundaries: HistGradientBoosting can capture non-linear feature interactions that a linear model may miss (especially with OHE-expanded categorical signals and engineered interactions).
- Robust on moderate-sized tabular data: with ~8.7k training rows, this model is usually a strong accuracy booster without requiring large-scale deep learning.
- Handles engineered numeric structure: features like `TotalSpend`, spend logs, and interaction flags often work well with gradient-boosted tree ensembles.
- Feature engineering created both numeric and categorical derived signals; tree boosting often leverages these effectively.

- Note: this ranking is based on a quick stratified holdout, not group-safe CV. For final training, Agent 5 should revisit with `GroupKFold` using `GroupId` and ensure group aggregates are computed fold-safely.

## Alternatives to revisit

- `LogisticRegression(OHE)`

### Why revisit?
- The alternative may capture non-linearities that a linear model misses.
- Final validation should use the correct split strategy (likely `GroupKFold` by `GroupId`) to avoid leakage when group-based engineered features are used.
- If group-safe CV changes the ranking, Agent 5/6 should re-select.

## Feature columns (model input)

- Categorical columns (OHE): 5
- Numeric columns: 44
- Dropped id-like columns: `PassengerId`, `GroupId`
