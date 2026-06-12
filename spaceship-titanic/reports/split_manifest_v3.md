# Split Manifest v3

## Configuration
- **Split type:** stratified group holdout (GroupId kept together)
- **Ratio:** 80/20 groups
- **Seed:** 42
- **Stratum:** group majority class (transported_rate >= 0.5)

## Partition sizes
| Partition | Rows | Groups | Target rate |
|---|---:|---:|---:|
| train_split | 6972 | 4973 | 0.5042 |
| eval_split | 1721 | 1244 | 0.5015 |

## Integrity checks
- Group overlap train/eval: **0** (required)
- External test.csv: untouched

## Usage
- Phase B/C: fit transforms on `train_split_v3` only; apply to eval + test.
- Frozen for v3 unless user approves change (note in experiment ledger).

## User Decisions
- Phase A checkpoint: **approved** (2026-06-12).
