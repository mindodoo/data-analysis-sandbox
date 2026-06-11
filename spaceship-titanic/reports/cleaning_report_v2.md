# Agent 1 — Phase B: Cleaning Report (v2)

## Objective

Apply the user-approved Phase A cleaning strategy, producing leakage-free
modeling-ready data with every action citing Phase A evidence.

## Context Received

- `initial_analysis_v2.md` and Phase A checkpoint decision: **strategy approved as proposed**.
- Key inherited findings: CryoSleep⇒zero-spend hard rule; ≈MCAR missingness across
  24% of rows; spend skew 6–13 with 61–64% zeros; group members share planet/cabin.

## Actions Performed (with evidence)

| # | Action | Train rows | Test rows | Evidence |
|---|--------|-----------:|----------:|----------|
| B1 | Cabin → Deck/CabinNum/Side; GroupId, Surname extracted | all | all | Deck V=0.21, CabinNum banding |
| B2 | spend←0 where CryoSleep=True | 361 | 170 | hard rule, 0 violations |
| B3 | CryoSleep←spend evidence | 217 | 93 | P(Cryo=True \| zero spend)=0.851 (n=3160) |
| B4 | group-propagate HomePlanet/Deck/Side/Destination/Surname | ~465 | ~199 | 100% within-group agreement (HomePlanet, Side) |
| B5 | Age←median by HomePlanet (train-fitted) | 179 | 91 | ≈MCAR; planet-specific age profiles |
| B6 | spend←0 + SpendNACount indicator | 582 | 297 | zero is modal value (61–64%) |
| B7 | remaining categoricals←"Unknown"; VIP←False | ~520 | ~280 | low residual missingness; VIP 97.7% False |
| B8 | log1p on 5 spend columns (replace raw) | all | all | skew 6.3–12.6 |

Not done (by design): outlier removal (extremes are real signal), scaling (tree models).

## Verification

- Corroboration check B3: 85.1% of known zero-spenders are CryoSleep=True.
- Within-group agreement: HomePlanet 100%, Side 100%, **Destination only 49.2%** (flagged).
- Post-clean assertion: CryoSleep=True ⇒ total spend = 0 holds; row counts unchanged.
- NaNs: train 2,324 → 0; test 1,117 → 0 (Cabin/Name kept raw, excluded from features).

## Risks / Concerns

1. Destination group-fill (79 train / 41 test rows) is a weak prior (49.2% agreement).
2. CryoSleep ambiguous remainder filled with majority class False — small bias possible.
3. SpendNACount retained so models can learn from MNAR effects if any.

## User Decisions

- Phase A checkpoint: approved as proposed.
- Phase B checkpoint: pending.

## Files Generated

- `reports/cleaned_train_v2.parquet`, `reports/cleaned_test_v2.parquet`
- `reports/figures_v2/b_before_after.png`
- `reports/cleaning_report_v2.md`

## Recommendations for Next Phase (C)

TotalSpend/HasSpend, Luxury-vs-Basic spend split, GroupSize/IsAlone, FamilySize
(surname), CabinRegion bins, AgeBin, Deck×Side interaction — see notebook checkpoint
cell for formulas and rationale.
