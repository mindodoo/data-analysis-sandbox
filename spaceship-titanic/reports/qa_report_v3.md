# QA Report (v3.1)

## Split & Leakage

- Internal split: stratified group holdout 80/20, seed 42, 0 group overlap ✓
- Inner CV: GroupKFold(5) on train_split only; no GroupId crosses fold boundary ✓
- Phase B/C transforms fit on train_split only; eval/test applied without refit ✓
- External test.csv untouched until submission ✓

## Reproducibility

- Seed 42 fixed across split, CV, tuning, final model ✓
- Fold-0 CatBoost refit reproduces stored OOF predictions (atol 1e-6) ✓
- Artifacts versioned: `*_v3.parquet`, `models/model_v3/` ✓

## Preprocessing Consistency

- Engineered train/eval/test: identical feature columns ✓
- No NaNs in model feature matrix on test ✓
- No unseen categories in test vs train_split ✓

## Protocol Compliance (updated workflow)

- Phase A checkpoint before split ✓
- Split Preparation before Phase B ✓
- Eval partition not used for tuning ✓
- Fixed threshold t=0.500 (v1/v2 evidence) ✓

## Issues Found

None blocking. Destination group-fill weak prior (47.9%) — documented in cleaning report.
