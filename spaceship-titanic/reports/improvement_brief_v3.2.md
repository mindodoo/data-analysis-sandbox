# Improvement Brief — Iteration v3.2 (Agent 4)

## Diagnosis

**Layer 2 — Features.** Evidence from Agent 3 (v3.1):

- Failure cluster: Earth 26.4% error, Deck G 29.0% (overall 18.8%).
- 35.3% of errors are boundary cases → model lacks separating signal for Earth/Deck-G zero-spend rows.
- v2.2 contextual features (+0.0001 OOF) already failed on similar hypothesis — this iteration uses **explicit cluster flags and planet×deck interactions** instead of neighborhood stats.
- Other layers ruled out: overfit gap 0.053 (not hparam); holdout matches OOF (not split); imputation minor.

## Hypothesis

Explicit encoding of the Earth/Deck-G failure cluster (`EarthZeroSpend`, `DeckGFlag`) plus
`HomePlanetDeck` interaction gives CatBoost direct splits for the worst segments without
the sample-size cost of per-cohort models (v2.4 lesson).

## Instructions for Agent 1 — Phase C (re-entry mode)

Add ONLY these 4 features (all else frozen):

1. `HomePlanetDeck` — HomePlanet + "_" + Deck
2. `EarthZeroSpend` — int(HomePlanet=="Earth" & HasSpend==0)
3. `DeckGFlag` — int(Deck=="G")
4. `SpendPerGroupMember` — TotalSpend / GroupSize

## What Must Stay Frozen

- train/eval split (seed 42), GroupKFold folds on train_split, threshold 0.5.
- Phase B cleaning, all existing v3.1 features.
- CatBoost params: v2.1 canonical.

## Expected Outcome & Success Criteria

- OOF ≥ 0.8160 AND Earth error < 25%, no segment regressing > 1pp.
- If unmet → stop; keep v3.1 submission as canonical.

## Result (post-iteration)

**Criteria NOT met.** v3.2 OOF 0.8112 (−0.0013 vs v3.1), Earth error 26.5% (+0.1pp).
Holdout 0.8164 (+0.0047) — inconsistent; treat as noise. **Recommend v3.1 canonical.**
