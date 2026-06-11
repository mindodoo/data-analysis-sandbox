# Risk Assessment (v2 · Phase A)

| # | Risk | Severity | Mitigation |
|---|------|----------|------------|
| 1 | Group leakage in CV (members of one group split across folds) | High | GroupKFold by GroupId, as in v1 |
| 2 | OOF↔LB ranking mismatch (seen in v1) | Medium | Fixed threshold 0.5 by default; treat OOF deltas < ~0.002 as noise |
| 3 | 24% of rows have ≥1 missing value | Medium | Impute, never drop rows; deterministic rules first |
| 4 | Deterministic CryoSleep back-fill may mislabel zero-spenders | Medium | Only back-fill when corroborated (e.g. group/deck evidence); add missing indicator |
| 5 | Spending outliers (max ≈ 30k) could dominate linear models | Low | log1p transform; tree models robust |
| 6 | Over-reliance on group identity (agreement only 43.6%) | Medium | Use group attributes (size, shared deck/planet/surname), not group target stats |
| 7 | Surname features cross train/test — potential overfit to family names | Low | Use family size/aggregates only, never surname identity |
| 8 | Train/test drift | Low | Checked: numeric distributions aligned |
