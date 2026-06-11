# Feature Engineering Report — Agent 3

Generated from `spaceship_titanic_agents.ipynb` section 3.

## Summary
- Engineered spend structure features (log1p, non-zero counts, mean non-zero spend)
- Parsed/derived cabin location features (binning + deck/side composition)
- Added age bin feature
- Added group-level aggregates (computed from the passenger manifest, no target used)

## Leakage / overfitting notes
- Group-level aggregates (e.g. `GroupSize_all`) can **inflate CV** if group members appear in both folds.
  Use `GroupKFold` by `GroupId` (Agent 5) if you keep these features.
