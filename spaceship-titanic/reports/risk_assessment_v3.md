# Risk Assessment — Agent 1 Phase A (v3)

## Context

Iteration v3 follows v2.1 canonical model (LB 0.80430). v2.2–v2.4 showed diminishing
returns and OOF↔LB selection bias. This assessment covers risks identified during fresh EDA.

## Data Quality Risks

| Risk | Severity | Evidence | Mitigation |
|---|---|---|---|
| Missing values (~2.5% per col, 24% rows affected) | Medium | MCAR-like; low MNAR signal | Group-aware imputation; missing indicators |
| CryoSleep↔spending logical constraint | Low | 0 violations when Cryo=True | Deterministic rule; document edge cases |
| VIP extremely rare (~2.3%) | Low | Cramér's V = 0.037 | Keep as categorical; do not over-engineer |
| Name/surname as proxy | Medium | 1,536 surnames shared train/test | OK for features; do not use as direct ID |

## Leakage Risks

| Risk | Severity | Evidence | Mitigation |
|---|---|---|---|
| Group leakage in CV | **High** | 6,217 groups; 43.6% within-group agreement | GroupKFold + group holdout split |
| Train/test group overlap | None | 0 shared groups | External test is clean |
| Target encoding leakage | Medium | If used | Fit encodings on train split only |
| Eval partition peeking | Medium | New workflow requirement | Freeze split before Phase B; no eval-driven feature hunting |

## Modeling / Iteration Risks

| Risk | Severity | Evidence | Mitigation |
|---|---|---|---|
| OOF↔LB mismatch | **High** | v1 threshold tuning; v2.4 cohort configs | Fixed t=0.500; fixed protocol; log LB |
| Plateau / noise floor | **High** | v2.2 +0.0001; v2.4 OOF gain lost on LB | Require ≥1 SE gain; honest stopping |
| Earth/Deck-G irreducible error | Medium | 26–29% error vs 18.6% overall | Target in Phase C only if EDA shows new signal |
| Over-complexity | Medium | v2.4 cohort models lost head-to-head | Prefer simpler global pipeline unless clear LB gain |

## Split Strategy Risk

Using a random row split (instead of group split) would leak group information between
train and eval partitions — **reject**. Proposed stratified group holdout aligns with
GroupKFold used in modeling.

## Overall Assessment

Dataset is clean and well-understood. Primary risks are **methodological** (group leakage,
OOF selection bias, eval peeking) rather than raw data quality. The updated workflow's
mandatory internal split directly addresses eval peeking. Proceed to Split Preparation
after user approval.
