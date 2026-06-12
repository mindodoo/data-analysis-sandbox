# Agent 1 — Phase B: Cleaning Report (v3)

## Objective

Apply the user-approved cleaning strategy with **train-split-only fitting** per the
updated multi-agent workflow.

## Context Received

- `initial_analysis_v3.md`, Phase A checkpoint: **approved**.
- Split manifest: stratified group holdout 80/20, seed 42 (`split_manifest_v3.md`).
- Key inherited findings: CryoSleep⇒zero-spend hard rule; ≈MCAR missingness; spend
  skew 6–13 with 61–64% zeros; group members share planet/cabin.

## Split Preparation (completed)

| Partition | Rows | Groups | Target rate |
|---|---:|---:|---:|
| train_split | 6972 | 4973 | 0.5042 |
| eval_split | 1721 | 1244 | 0.5015 |

Group overlap: 0. Target balance preserved across partitions.

## Actions Performed (fit on train_split only)

| # | Action | train_split | eval_split | test | Evidence |
|---|--------|------------:|-----------:|-----:|----------|
| B1 | Cabin → Deck/CabinNum/Side; GroupId, Surname | all | all | all | Deck V=0.21, CabinNum banding |
| B2 | spend←0 where CryoSleep=True | 272 | 89 | 170 | hard rule, 0 violations |
| B3 | CryoSleep←spend evidence | 186 | 31 | 93 | P(Cryo=True \| zero spend)=0.845 (n=2540) |
| B4 | group-propagate HomePlanet/Deck/Side/Destination/Surname | ~306 | ~99 | ~199 | HomePlanet/Side 100% within-group |
| B5 | Age←median by HomePlanet (train_split-fitted) | 140 | 39 | 91 | ≈MCAR; planet-specific profiles |
| B6 | spend←0 + SpendNACount indicator | 471 | 111 | 297 | zero is modal value |
| B7 | remaining categoricals←"Unknown"; VIP←False | ~428 | ~88 | ~274 | low residual missingness |
| B8 | log1p on 5 spend columns | all | all | all | skew 6.3–12.6 |

Not done (by design): outlier removal, scaling (tree models planned).

## Verification

- Corroboration B3: **84.5%** of known zero-spenders are CryoSleep=True (train_split).
- Within-group agreement (train_split): HomePlanet 100%, Side 100%, **Destination 47.9%** (flagged).
- Post-clean assertion: CryoSleep=True ⇒ total spend = 0; row counts unchanged.
- All transform parameters learned from train_split only — no eval refitting.

## Risks / Concerns

1. Destination group-fill is a weak prior (47.9% agreement) — same caveat as v2.
2. CryoSleep ambiguous remainder → False (majority class).
3. SpendNACount retained for MNAR signal.

## User Decisions

- Phase A checkpoint: approved.
- Split strategy: approved (executed).
- Phase B checkpoint: pending.

## Files Generated

- `reports/train_split_v3.parquet`, `reports/eval_split_v3.parquet`
- `reports/cleaned_train_v3.parquet`, `reports/cleaned_eval_v3.parquet`, `reports/cleaned_test_v3.parquet`
- `reports/split_manifest_v3.md`
- `reports/figures_v3/split_partitions.png`, `figures_v3/b_before_after.png`

## Recommendations for Phase C

Same feature set as v2 (proven on LB 0.80430), evaluated on train_split:

| Feature | Formula | Rationale |
|---|---|---|
| `TotalSpend` | sum of 5 log1p spends | spend level separates classes |
| `HasSpend` | TotalSpend > 0 | zero-inflation signal |
| `LuxurySpend` / `BasicSpend` | RoomService+Spa+VRDeck vs FoodCourt+ShoppingMall | luxury vs basic split |
| `GroupSize` | members per GroupId | group-attribute signal |
| `IsAlone` | GroupSize == 1 | solo traveler behavior |
| `FamilySize` | members per Surname | surname crosses train/test |
| `CabinRegion` | CabinNum binned (~300-wide) | spatial banding |
| `AgeBin` | child/teen/adult/senior | non-linear Age relation |
| `DeckSide` | Deck × Side | both individually predictive |

Consider v3-specific additions only if train_split evidence supports them (Earth/Deck-G segments).
