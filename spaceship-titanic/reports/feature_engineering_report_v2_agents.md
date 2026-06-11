# Agent 1 — Phase C: Feature Engineering Report (v2 multi-agent run)

## Objective

Build the 9 user-approved features, quantify their signal, validate multicollinearity
and train/test consistency, and propose a modeling strategy for Agent 2.

## Context Received

- `cleaning_report_v2.md` + Phase B checkpoint: all 9 features approved; user
  constraint — do NOT modify raw `train.csv`/`test.csv`, work in notebook only.
- Inherited: group identity weak (43.6% agreement) → use group attributes;
  CabinNum spatial banding; zero-inflated spends.

## Actions Performed

1. Built 10 feature columns (9 approved concepts) identically on cleaned train/test.
2. Signal evaluation: point-biserial correlation, Cramér's V, mutual information
   against all base features.
3. Multicollinearity matrix; flagged pairs |corr| > 0.8.
4. Train/test consistency checks (columns, dtypes, category sets).
5. Exported versioned artifacts.

## Key Results

- **LuxurySpend is the strongest predictor overall** (MI 0.160 > CryoSleep 0.121;
  corr −0.513) — v2's main feature win.
- HasSpend −0.482, TotalSpend −0.439, DeckSide V=0.238 (> Deck alone 0.212).
- Weak features kept for tree evaluation: CabinRegion −0.043, FamilySize −0.031.
- Multicollinearity confined to spend aggregates (expected, harmless for trees).

## Recommendations for Next Agent (Agent 2 — Modeling)

- Models: LogisticRegression baseline → HistGradientBoosting (v1 canonical) →
  LightGBM/CatBoost (v1 backlog item).
- CV: GroupKFold by GroupId, 5 folds. Fixed threshold 0.5.
- Targets to beat: v1 OOF 0.8108, LB 0.80289.
- For linear baseline drop TotalSpend (keep LuxurySpend+BasicSpend) to avoid
  multicollinearity.

## Risks / Concerns

- Spend-aggregate collinearity if a linear meta-model is used later.
- FamilySize uses surnames shared across train/test — aggregate only, no identity
  encoding (leakage-safe).

## User Decisions

- Phase B checkpoint: all 9 features approved; raw CSVs untouched (confirmed).
- Phase C checkpoint: pending.

## Files Generated

- `reports/engineered_train_v2.parquet`, `reports/engineered_test_v2.parquet`
- `reports/feature_registry_v2.md`, this report
- `reports/figures_v2/c_feature_signal.png`, `c_multicollinearity.png`
