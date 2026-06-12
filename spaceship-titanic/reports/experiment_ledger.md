# Experiment Ledger (owned by Agent 3)

One row per iteration. Single source of truth for cross-iteration comparison.
v1 rows reconstructed from `leaderboard_tracking.md`, `improvement_opportunities.md`,
`final_recommendations.md`.

| Iteration | What changed | Layer changed | OOF acc | LB acc | F1 | ROC-AUC | Verdict vs previous |
|---|---|---|---:|---:|---:|---:|---|
| v1.0 | Baseline HGB on v1 features | model | 0.8069 | — | — | — | baseline |
| v1.1 | Tuned HGB (Agent 6 params) | hparam | 0.8108 | 0.80173 | — | — | improved |
| v1.2 | CatBoost native (v1 features) | model | — | 0.80102 | — | — | regressed on LB |
| v1.3 | Group-aware stack, t=0.500 | model | — | 0.80289 | — | — | improved (v1 LB best) |
| v1.4 | Leakage-safe stack v2, t=0.500 | model | — | 0.80289 | — | — | tied |
| v2.1 | New cleaning (deterministic rules) + 10 new features + CatBoost tuned | data + feature + model | 0.8143 | **0.80430** | 0.8165 | 0.9029 | **LB BEST — beats v1 best (+0.00141), canonical model** |
| v2.2 | +5 contextual features (improvement brief v2.2) | feature | 0.8144 | — | — | — | tie (+0.0001, noise) — success criteria NOT met → STOP recommended |
| v2.3 | Cohort-specific models by HomePlanet, per-cohort features + per-cohort tuned CatBoost | feature + model + hparam | 0.8162 | — | — | — | +0.0019 vs v2.1 — < 1 SE, OOF-selected configs |
| v2.4 | + cohort-specific features (spend shares, CryoDeck, group flags, CabinNumFine) + 13-config search per cohort | feature + hparam | 0.8181 | 0.80406 | — | — | best OOF but LB < v2.1 (−0.00024) — OOF edge did not transfer (selection bias confirmed); 2nd best LB overall |
| v3.1 | Updated workflow (mandatory train/eval split) + same cleaning/features/hparams as v2.1 | methodology | 0.8125† | **0.80383** | 0.8164 | 0.9005 | **LB −0.00047 vs v2.1** (within noise); beats v1 best (+0.00094); v2.1 remains canonical |
| v3.2 | +4 cluster features (HomePlanetDeck, EarthZeroSpend, DeckGFlag, SpendPerGroupMember) | feature | 0.8112 | — | — | — | **NOT met criteria** — OOF −0.0013, Earth error +0.1pp; holdout +0.0047 (noise); STOP recommended |

† v3.1 OOF is on train_split (6972 rows); v2.1 OOF on full train (8693 rows) — not directly comparable.
Holdout eval_split accuracy: **0.8117** (v3.1), **0.8164** (v3.2).

## v3 LB Outcome (closes the v3 loop)

- **`submission_catboost_v3.csv`: LB 0.80383** — 3rd best overall, **−0.00047 vs v2.1** (0.80430).
- Still beats v1 best (0.80289) by +0.00094; essentially tied with v2.1 given LB noise (~±0.001).
- Same features + hparams as v2.1; updated train/eval split workflow adds rigor without LB gain.
- v3.2 iteration correctly stopped — would not have helped (OOF regressed).
- **Canonical model remains v2.1 / v3.1-equivalent:** `submission_catboost_v2.csv` (LB best) or v3 pipeline for reproducibility.

## v3 Submission (logged)

- **`submission_catboost_v3.csv`** — v3.1 pipeline refit on full train · **LB 0.80383**

## v3.2 Comparison Analysis

- Explicit Earth/Deck-G flags did not beat v3.1 on OOF or segment errors.
- Holdout uptick (+0.0047) without OOF gain mirrors v2 OOF↔holdout noise pattern — do not chase.
- Earth/Deck-G cluster likely irreducible (same conclusion as v2.2 Agent 4).
- **Canonical for LB:** v2.1 `submission_catboost_v2.csv` (LB **0.80430**). v3.1 logged at **0.80383** (−0.00047).

- **v2.1 global CatBoost: LB 0.80430 — new project best** (v1 best was 0.80289).
- v2.4 cohort models: LB 0.80406 — also beats v1 best, but loses the head-to-head.
- Lesson recorded: +0.0038 OOF from per-cohort OOF-selected configs shrank to
  −0.00024 on LB — the selection-bias caveat flagged at the v2.3/v2.4 checkpoints
  was real. Fixed-protocol, simpler models keep generalizing better (same lesson
  as v1's threshold tuning).
- **Canonical model: v2.1 `catboost_tuned` (`models/model_v2/`).**

### v2.3 notes

- Naive cohort splits (frozen params) all LOST to global: CryoSleep 0.8078,
  HasSpend 0.8085, HomePlanet 0.8112 — sample-size cost > specialization benefit.
- Gain came only from per-cohort hyperparameters: Earth/Europa prefer strongly
  regularized configs (depth 3, 300 iters, l2=9); Mars keeps global config.
- Per-cohort feature sets exclude dead features (e.g. all spend features constant
  zero inside Cryo / zero-spend cohorts).

## Stopping Decision (Agent 4, after v2.2)

- Two consecutive gains below 1 SE (v2.1 +0.0035 ≈ 0.85 SE; v2.2 +0.0001) → plateau.
- Earth/Deck-G cluster unchanged by contextual features → likely irreducible with
  available attributes (information-poor rows).
- Canonical model: **v2.1 catboost_tuned** (`models/model_v2/`) — equal accuracy,
  simpler than v2.2.

## v2.1 Comparison Analysis

- Improvement vs v1.1 OOF: +0.0035 across all 5 folds (every fold ≥ its v1-era
  counterpart range); within ~1 SE individually → likely real but small.
- No metric trade-off: precision 0.8130 / recall 0.8200 balanced.
- Trend across iterations: slow convergence; gains shifted from hparam layer (v1.1)
  to data+feature+model layer (v2.1), consistent with Agent 4 v1 diagnosis.
- Caution: v1.2 showed CatBoost OOF gains did not transfer to LB on the v1 feature
  set. v2.1 uses a different feature set; LB anchor needed.

## Error Root-Cause Summary (v2.1, input for Agent 4)

| Pattern | Evidence | Layer |
|---|---|---|
| Earth / Deck G failure cluster | 26.2% / 28.9% error vs 18.6% overall | feature gap |
| Boundary errors | 35.7% of errors within ±0.10 of threshold | feature gap |
| Confident wrong predictions | 358 rows at >0.30 from threshold | data quality / label noise (likely irreducible) |
| Imputed rows err more | 23.0% (SpendNACount>0) vs 18.3% clean | data quality (minor) |
| HomePlanet=Unknown | 30.6% error (n=111) | data quality (small n) |
