# Improvement Brief — Iteration v2.2 (Agent 4)

## Diagnosis

**Layer 2 — Features.** Evidence from Agent 3 (v2.1):

- Failure cluster: Earth passengers 26.2% error, Deck G 28.9% (overall 18.6%);
  Earth×G is dominated by zero-spend, low-information rows.
- 35.7% of errors sit within ±0.10 of the threshold → the model lacks separating
  signal for these rows; it is not making confident wrong calls there.
- Other layers ruled less likely: smallest overfit gap among boosters (0.045) →
  not Layer 4/5; QA clean splits → not Layer 3; imputation effect minor
  (+4.7pp on ~6% of rows) → Layer 1 secondary.

## Hypothesis

For zero-spend Earth/Deck-G passengers, the remaining signal is contextual:
outcomes of/attributes of cabin-mates, group companions, and physical neighbors.
Self-excluding neighborhood features should separate these rows.

## Instructions for Agent 1 — Phase C (re-entry mode)

Add ONLY these features to engineered v3 artifacts (everything else unchanged):

1. `CabinMatesCount` — same exact Cabin, excluding self.
2. `GroupSpendMean` — mean TotalSpend of other group members (solo → global mean).
3. `GroupCryoShare` — share of other group members in CryoSleep (solo → global rate).
4. `NeighborCount` — passengers within ±1 CabinNum, same Deck/Side, excluding self.
5. `FamilySizeOnDeck` — same surname on same deck, excluding self.

All computed from feature columns only (no target) → leakage-safe to compute on
full train + test jointly is NOT allowed across the target; group stats use
attributes, not labels, so standard computation per dataset is acceptable.

## What Must Stay Frozen

- GroupKFold(5) fold assignment (`FOLDS`), seed 42, threshold 0.5.
- Phase B cleaning, all existing v2 features.
- CatBoost params from v2.1 (`iterations=600, lr=0.08, depth=4, l2_leaf_reg=3`).

## Expected Outcome & Success Criteria

- OOF accuracy ≥ 0.8160 AND Earth error < 25%, no segment regressing > 1pp.
- If unmet → Agent 4 will recommend stopping and submitting the best model.
