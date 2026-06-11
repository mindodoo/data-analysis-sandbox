# Agent 3 — Evaluation, QA & Review Report (v2.1)

## Objective

Evaluate `catboost_tuned` (model_v2) honestly, verify QA, update the experiment
ledger, and recommend next step.

## Context Received

- `training_report_v2.md`: candidate OOF 0.8143, frozen GroupKFold(5), t=0.5.
- v1 ledger: OOF↔LB mismatch warning; v1 CatBoost-native regressed on LB.
- Phase B flags: imputed rows and group-filled Destination to be checked in errors.

## Metrics (OOF, t=0.5)

| accuracy | precision | recall | F1 | ROC-AUC | PR-AUC |
|---:|---:|---:|---:|---:|---:|
| 0.8143 | 0.8130 | 0.8200 | 0.8165 | 0.9029 | 0.9149 |

Significance: Δ +0.0035 vs v1 ≈ 0.85 SE → directional, not individually conclusive.

## Error Analysis

- Worst segments: HomePlanet=Unknown 30.6%, Deck G 28.9%, Earth 26.2%,
  Deck=Unknown 24.2%, imputed rows 23.0%.
- Best segments: Deck B 5.8%, Deck C 6.2%, Europa 7.2%.
- 35.7% of errors are boundary cases (|p−0.5| < 0.10) → feature gap.
- 358 confident errors (|p−0.5| > 0.30) → probable label noise / irreducible.

Root-cause layers: see `experiment_ledger.md` table (primary: feature gap for
Earth/Deck-G zero-spend passengers).

## QA Verification

| Check | Result |
|---|---|
| Group leakage across folds | none (asserted per fold) ✓ |
| Reproducibility | fold-0 refit reproduces OOF predictions to 1e-6 ✓ |
| Split matches Agent 2 | identical frozen FOLDS object ✓ |
| Seeds | fixed (42) everywhere ✓ |
| Test features | no NaNs, columns/categories consistent ✓ |

## Verdict

Best OOF iteration to date; sound QA. Recommend generating a submission to anchor
the v2 pipeline on LB (v1 showed OOF↔LB mismatch), then routing to Agent 4 with the
Earth/Deck-G feature-gap hypothesis.

## User Decisions

- Pending at checkpoint.

## Files Generated

- `reports/experiment_ledger.md`, `reports/qa_report_v2.md`
- `reports/figures_v2/e_metrics.png`, `e_error_segments.png`, `e_iterations.png`
