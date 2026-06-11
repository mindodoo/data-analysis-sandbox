# Agent 1 — Data Preparation · Phase A: Initial Analysis (v2)

## Objective

Re-run EDA from raw data for the v2 improvement iteration, focusing on the areas most
likely to yield new feature-engineering gains (group structure, spending behavior,
CryoSleep constraints, Cabin decomposition).

## Context Received

- Reports read: `final_recommendations.md`, `leaderboard_tracking.md`, `improvement_opportunities.md` (v1).
- Inherited findings: best LB 0.80289 (stacked, t=0.500); tuned HGB OOF 0.8108; OOF↔LB
  ranking mismatch; model/threshold tweaks plateaued.
- Impact on this session: v2 searches for gains in data prep + features rather than
  model tweaks; group-safe CV remains mandatory.

## Inputs Received

- `data/train.csv` (8,693 × 14), `data/test.csv` (4,277 × 13)

## Dataset Summary

- Binary classification, target `Transported`, balanced at 50.36%.
- No duplicate rows, no constant/near-constant columns.
- All features have ~2–2.5% missing; 24.0% of train rows (23.3% test) have ≥1 gap.

## Actions Performed

1. Problem-type identification and target balance check.
2. Schema/dtype/cardinality/duplicate audit (train + test).
3. Missing-value analysis: per-column %, co-missingness correlation, missingness-vs-target.
4. Distribution analysis: Age by target; log1p spending histograms; zero-inflation stats.
5. Categorical target rates: HomePlanet, CryoSleep, Destination, VIP, Deck, Side; Cabin decomposed.
6. CryoSleep↔spending logical-constraint verification.
7. Correlation analysis: Pearson/Spearman heatmaps; Cramér's V for categoricals.
8. Leakage & risk checks: train/test group overlap, surname overlap, numeric drift.

## Key Findings

| Finding | Evidence |
|---|---|
| Spending cols extremely skewed, zero-inflated | skew 6.3–12.6; 61–64% zeros |
| CryoSleep=True ⇒ spend = 0 (hard rule) | 0 violations in 8,693 rows |
| CryoSleep strongest categorical predictor | Cramér's V = 0.463 |
| Deck and HomePlanet meaningful | V = 0.212 / 0.193; VIP weak (0.037) |
| Missingness ≈ MCAR, low target relation | co-missingness corr ≈ 0 |
| No train/test group overlap | 0 shared groups (no direct leakage) |
| Group outcome agreement only 43.6% | group identity weaker than v1 assumed; use group *attributes* |
| Surnames cross train/test (1,536 shared) | family features transfer across split |
| CabinNum spatial banding vs target | histogram banding by Transported |
| No numeric train/test drift | means/medians aligned |

## Insights Shared With User

See Phase A checkpoint cell in `notebooks/st_agents_v2.ipynb` and figures in
`reports/figures_v2/` (`a_target_groups.png`, `a_missingness.png`, `a_numeric_dists.png`,
`a_categorical_rates.png`, `a_cabinnum.png`, `a_correlations.png`).

## Proposed Phase B Cleaning Strategy (pending user approval)

1. Decompose `Cabin` → `Deck`/`CabinNum`/`Side` before imputation.
2. Deterministic imputation: spending→0 when CryoSleep=True; CryoSleep→True when total
   spend = 0 and evidence agrees; propagate HomePlanet/Deck/Side within group.
3. Age: median by HomePlanet; spending: 0 + missing indicator; categoricals: group mode
   else "Unknown".
4. log1p on spending columns; no outlier removal (extremes are signal); no scaling
   (tree models planned).

## Risks / Concerns

- Group outcome agreement near chance → avoid over-weighting raw group identity.
- OOF↔LB mismatch from v1 → keep thresholds fixed at 0.5 unless strong evidence.
- Deterministic CryoSleep back-fill could mislabel passengers who simply spent nothing.

## User Decisions

- Pending (Phase A checkpoint awaiting approval).

## Files Generated

- `notebooks/st_agents_v2.ipynb`
- `reports/initial_analysis_v2.md`, `reports/risk_assessment_v2.md`
- `reports/figures_v2/*.png`
