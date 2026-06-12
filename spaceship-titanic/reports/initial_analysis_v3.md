# Agent 1 — Data Preparation · Phase A: Initial Analysis (v3)

## Objective

Fresh EDA for iteration **v3** using the updated multi-agent data science workflow
(`multi_agent_data_science_markdowns/02_agent_1_data_preparation.md`). Focus on
information-poor segments flagged in v2 error analysis (Earth/Deck-G cluster,
boundary errors) and validate whether new cleaning or feature ideas remain viable.

## Context Received

Reports read: `experiment_ledger.md`, `leaderboard_tracking.md`, `initial_analysis_v2.md`,
`qa_report_v2.md`, `improvement_brief_v2.2.md`.

Key inherited findings:

1. **Best LB: 0.80430** (v2.1 global CatBoost — canonical model in `models/model_v2/`).
2. Per-cohort OOF gains (+0.0038) did **not** transfer to LB — selection bias confirmed;
   simpler fixed-protocol models generalize better.
3. Contextual features (v2.2) added +0.0001 OOF — noise floor; Earth/Deck-G cluster
   unchanged → likely information-poor with current attributes.
4. Group-safe CV (`GroupKFold` by `GroupId`) remains mandatory; fixed threshold t=0.500
   generalized best across v1 and v2.

How this shapes Phase A: re-validate raw-data patterns with fresh eyes; prioritize
segments where v2 still fails (Earth, Deck G, imputed rows); propose an **internal
train/eval split** (new workflow requirement) before any cleaning.

## Inputs Received

- `data/train.csv` (8,693 × 14), `data/test.csv` (4,277 × 13)
- Kaggle `test.csv` remains untouched (external holdout)

## Dataset Summary

- Binary classification, target `Transported`, balanced at **50.36%**.
- No duplicate rows, no spurious constant columns.
- All features ~2–2.5% missing; **24.0%** of train rows have ≥1 gap (23.3% test).
- 6,217 travel groups; 1,412 multi-member groups; within-group outcome agreement **43.6%**.

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
| No train/test group overlap | 0 shared groups |
| Group outcome agreement only 43.6% | use group *attributes*, not raw group ID |
| Surnames cross train/test (1,536 shared) | family features transfer across split |
| CabinNum spatial banding vs target | histogram banding by Transported |
| No numeric train/test drift | means/medians aligned |

## Proposed Internal Split Strategy (pending user approval)

Per updated Agent 1 workflow — execute **after** Phase A approval, **before** Phase B:

- **Type:** stratified **group** holdout — keep all passengers in a `GroupId` together,
  stratify by target rate at the group level.
- **Ratio:** 80/20 (~6,955 / ~1,738 rows expected).
- **Seed:** 42 (documented in `split_manifest_v3.md`).
- **Artifacts:** `train_split_v3.parquet`, `eval_split_v3.parquet`.
- **Rationale:** matches GroupKFold protocol used in v2 modeling; prevents cleaning/feature
  decisions from overfitting to the same rows used for internal validation.
- **Frozen for v3:** same split reused across all v3 iterations unless user approves change.

## Proposed Phase B Cleaning Strategy (pending user approval)

1. Decompose `Cabin` → `Deck` / `CabinNum` / `Side` before imputation.
2. Deterministic imputation: spending→0 when CryoSleep=True; CryoSleep→True when total
   spend = 0 and evidence agrees; propagate HomePlanet/Deck/Side within group.
3. Age: median by HomePlanet; spending: 0 + missing indicator; categoricals: group mode
   else "Unknown".
4. log1p on spending columns; no outlier removal; no scaling (tree models planned).
5. **Fit all transform parameters on `train_split_v3` only**; apply to eval + external test.

## Insights Shared With User

See Phase A checkpoint in `notebooks/st_agents_v3.ipynb` and figures in
`reports/figures_v3/` (`a_target_groups.png`, `a_missingness.png`, `a_numeric_dists.png`,
`a_categorical_rates.png`, `a_cabinnum.png`, `a_correlations.png`).

## User Decisions

- Phase A checkpoint: **approved** (2026-06-12).
- Split strategy: **approved** — executed (6972 / 1721 rows, 0 group overlap).
- Phase B: **completed** — awaiting Phase C approval.

## Risks / Concerns

- v2 plateau: gains below 1 SE for two consecutive iterations — v3 must target genuinely
  new signal, not re-tuning the same pipeline.
- Earth/Deck-G cluster (26–29% error vs 18.6% overall) may be irreducible without new
  information sources.
- Deterministic CryoSleep back-fill could mislabel passengers who simply spent nothing.

## Recommendations for Next Step

After user approves Phase A: execute Split Preparation → Phase B cleaning (fit on train
split only).

## Files Generated

- `notebooks/st_agents_v3.ipynb`
- `reports/initial_analysis_v3.md`, `reports/risk_assessment_v3.md`
- `reports/figures_v3/*.png`
