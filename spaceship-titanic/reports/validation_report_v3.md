# Agent 3 — Evaluation, QA & Review Report (v3.1)

## Objective

Evaluate `catboost_tuned` (model_v3) on the updated split protocol; update experiment
ledger; recommend next step.

## Context Received

- `training_report_v3.md`: OOF 0.8125 (train_split), holdout 0.8117 (eval_split).
- v2.1 canonical: OOF 0.8143 / LB 0.80430.
- v2 error patterns: Earth/Deck-G cluster, boundary errors, OOF↔LB selection bias.

## Metrics

### OOF (train_split, GroupKFold-5, t=0.5)

| accuracy | precision | recall | F1 | ROC-AUC | PR-AUC |
|---:|---:|---:|---:|---:|---:|
| 0.8125 | 0.8063 | 0.8267 | 0.8164 | 0.9005 | 0.9118 |

### Holdout (eval_split, t=0.5)

| accuracy | precision | recall | F1 | ROC-AUC | PR-AUC |
|---:|---:|---:|---:|---:|---:|
| 0.8117 | 0.8018 | 0.8297 | 0.8155 | 0.9055 | 0.9184 |

Significance vs v2.1 full-train OOF: Δ −0.0018 ≈ **−0.38 SE** → **tie / within noise**.
Note: v3 OOF is on train_split (6972 rows); v2.1 on full train (8693) — not strictly
comparable. Holdout confirms OOF is not inflated.

## Error Analysis (OOF, train_split)

| Segment | Error rate | n |
|---|---:|---:|
| Deck=G | 29.0% | 2098 |
| HomePlanet=Unknown | 27.2% | 92 |
| HomePlanet=Earth | 26.4% | 3753 |
| Deck=E | 22.6% | 695 |
| Imputed rows | 21.8% | 454 |
| Clean rows | 18.5% | 6518 |

- **35.3%** of errors are boundary cases (|p−0.5| < 0.10) → feature-gap layer.
- **294** confident errors (|p−0.5| > 0.30) → label noise / irreducible.
- Same Earth/Deck-G failure cluster as v2.1 — no new segment resolved.

## QA Verification

| Check | Result |
|---|---|
| Group leakage across folds (train_split) | none ✓ |
| Reproducibility (fold-0 refit) | OOF matches to 1e-6 ✓ |
| Train/eval split integrity | 0 group overlap ✓ |
| Transforms fit on train_split only | verified ✓ |
| Test features complete | no NaNs ✓ |
| Seeds / threshold | 42 / 0.5 fixed ✓ |

## Cross-Iteration Verdict

| vs v2.1 | Result |
|---|---|
| OOF | tie (−0.0018, < 1 SE) |
| Features / model / hparams | identical |
| Methodology | improved (mandatory train/eval split) |
| Error profile | unchanged (Earth/Deck-G) |
| LB | pending |

**Verdict:** v3.1 reproduces v2.1 performance within noise on a more rigorous protocol.
No regression detected. LB submission recommended to confirm transfer.

## User Decisions

- Agent 2 → Agent 3 handoff: **approved**.

## Recommendations

1. **Generate submission** (`submission_catboost_v3.csv`) for LB anchor — same params
   as v2.1 canonical; refit on full labeled train.
2. Then **Agent 4** to decide: stop (plateau confirmed) or target Earth/Deck-G feature gap.

## Files Generated

- `reports/qa_report_v3.md`, `reports/error_analysis_report_v3.md`
- `reports/experiment_ledger.md` (updated)
- `figures_v3/e_metrics.png`, `e_error_segments.png`, `e_iterations.png`
