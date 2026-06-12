# Error Analysis Report (v3.1)

Model: `catboost_tuned` (model_v3) · OOF on train_split · t=0.5

## Root-Cause Classification

| Pattern | Evidence | Layer |
|---|---|---|
| Earth / Deck G failure cluster | 26.4% / 29.0% error vs 18.8% overall | feature gap |
| Boundary errors | 35.3% of errors within ±0.10 of threshold | feature gap |
| Confident wrong predictions | 294 rows at >0.30 from threshold | data quality / label noise |
| Imputed rows err more | 21.8% vs 18.5% clean | data quality (minor) |
| HomePlanet=Unknown | 27.2% error (n=92) | data quality (small n) |

## Segment Detail

Worst: Deck G, HomePlanet Unknown/Earth, Deck E, imputed rows.
Best segments unchanged from v2 (Europa, Decks B/C low error).

## Comparison to v2.1

Error profile is **substantially unchanged** — v3 workflow did not resolve the
information-poor Earth/Deck-G cluster. Contextual features (v2.2) also failed here.

## Implication for Agent 4

Primary bottleneck remains **Layer 2 (features)** for Earth/Deck-G zero-spend
passengers. Secondary: irreducible label noise (~294 confident errors).
