# Iteration v2.3 — Cohort-Specific Models (Agent 1 + Agent 2)

## Objective

User-directed iteration: define cohorts (Agent 1), build cohort-specific feature
sets if needed, and optimize a separate model per cohort (Agent 2).

## Context Received

- v2.1 error heterogeneity: Earth 26.2% vs Europa 7.2% error.
- Frozen protocol: GroupKFold(5) `FOLDS`, seed 42, threshold 0.5.
- Risk: per-cohort training shrinks samples (1.8k–4.7k rows per cohort).

## Agent 1 — Cohort Definition & Per-Cohort Features

Schemes evaluated: CryoSleep (2 cohorts), HomePlanet (3, Unknown folded into Earth),
HasSpend (2).

Key finding — cohorts have structurally different feature spaces:

| Cohort | Dead features (excluded) | Top signal inside cohort |
|---|---|---|
| Cryo / zero-spend | all 9 spend features | CabinNum, GroupSize, Age, CabinRegion |
| Awake / spender | — | LuxurySpend, BasicSpend, per-venue spends |
| Earth | — | LuxurySpend, TotalSpend, HasSpend, CryoSleep |

## Agent 2 — Per-Cohort Optimization (frozen folds)

| Variant | Combined OOF | vs global v2.1 (0.8143) |
|---|---:|---:|
| CryoSleep cohorts, frozen params | 0.8078 | −0.0065 |
| HasSpend cohorts, frozen params | 0.8085 | −0.0058 |
| HomePlanet cohorts, frozen params | 0.8112 | −0.0031 |
| **HomePlanet cohorts, per-cohort tuned** | **0.8162** | **+0.0019** |

Per-cohort tuned configs (4-config grid, selected by cohort OOF):

| Cohort | Config | Cohort error: global → cohort-tuned |
|---|---|---|
| Earth (n=4745) | depth 3, 300 iters, lr 0.08, l2 9 | 26.3% → 25.7% |
| Europa (n=2161) | depth 3, 300 iters, lr 0.08, l2 9 | 7.2% → 7.1% |
| Mars (n=1787) | depth 4, 600 iters, lr 0.08, l2 3 (global) | 11.8% → 12.5% |

## Interpretation

1. Specialization alone loses: smaller training sets cost more than cohort fit gains.
2. The gain is a regularization story: the info-poor Earth cohort wants a much
   simpler model than the global config.
3. Caveats: per-cohort config selection used OOF (selection bias); +0.0019 < 1 SE
   (0.0042) → inconclusive without LB confirmation.

## Recommendation

Generate `submission_cohort_v2.csv` (per-cohort refit on full train) as a second
LB probe; keep v2.1 canonical until LB arbitrates.

## Files Generated

- `reports/figures_v2/v23_cohort_models.png`
- Ledger row v2.3 in `experiment_ledger.md`

---

# Iteration v2.4 — Cohort-Specific Features + Wider Search

User directed further iteration instead of submitting v2.3.

## Agent 1 — cohort-specific features

`Share_<venue>` ×5 (venue spend / total, scale-free), `CryoDeck` interaction,
`GroupAllCryo`/`GroupAnySpend` (self-excluding group flags), `CabinNumFine` (//100).

Signal is strongly cohort-dependent (validates the cohort hypothesis):

| Feature | Earth | Europa | Mars |
|---|---:|---:|---:|
| Share_VRDeck | −0.26 | **−0.62** | −0.29 |
| Share_Spa | −0.27 | **−0.60** | −0.39 |
| Share_RoomService | −0.23 | −0.27 | **−0.61** |
| CabinNumFine | 0.00 | 0.03 | **0.22** |

## Agent 2 — wider per-cohort search (13 configs, frozen FOLDS)

| Cohort | v2.3 acc | v2.4 acc | winning config |
|---|---:|---:|---|
| Earth | 0.7427 | 0.7446 | depth 5, 500 iters, lr 0.05, l2 12 |
| Europa | 0.9287 | 0.9269 | (kept v2.3-style; plateaued) |
| Mars | 0.8752 | 0.8819 | depth 8, 300 iters, lr 0.03, l2 1 |

**Combined OOF = 0.8181** (+0.0019 vs v2.3, +0.0038 vs v2.1 global ≈ 0.9 SE).

## Verdict

Best ledger entry. Selection-bias caveat remains; recommend stop + LB arbitration
via `submission_cohort_v2.csv` vs `submission_catboost_v2.csv`.
