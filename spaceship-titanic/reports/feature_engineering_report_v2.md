# Feature Engineering Report v2

- Added multiplicative interactions for spend columns (pairwise products).
- Added ratio features (safe-divide).
- Added polynomial terms (`Age_sq`, `TotalSpend_sq`, `CabinNum_sq`).
- Added transformed variants using PowerTransformer (`*_pt`) and z-score (`*_z`).
- Added PCA components (`PCA_1..PCA_12`) from transformed numeric features.

Holdout accuracy (importance model): **0.8108**
