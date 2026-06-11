# Risk Assessment — Agent 1

## Data quality risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| ~2% missing across many columns | Medium | Documented imputation + optional missing flags (Agent 2) |
| MNAR on spend columns | Medium | Compare model with/without missing indicators |
| High-cardinality `Cabin` | Medium | Parse deck/side; target encoding with CV (Agent 3) |
| Boolean columns as strings | Low | Explicit cast after imputation |

## Modeling risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Group leakage in aggregates | High | GroupKFold or fold-safe group stats |
| Overfitting rare VIP | Low | VIP rate ~2%; may merge or drop |
| Class imbalance | Low | ~50/50 target; standard accuracy OK |

## Competition / process risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Train/test distribution shift | Low | Compare marginals in Agent 2 |
| Submission format | Low | Validate against `sample_submission.csv` |

## Outliers (IQR rule)

- **Age**: 0.9% outliers
- **TotalSpend**: 10.7% outliers

Extreme spend values are plausible (VIP/luxury); **do not clip without analysis**.

## Handoff

Proceed to **Agent 2 — Data Cleaning & Transformation** with priorities: boolean coercion, cabin parsing, missing-value strategy, train/test alignment.
