# QA Report — Agent 5

## Checks

```text
                  check   ok
no_nulls_in_train_clean True
    passenger_id_unique True
      group_col_present True
         target_present True
```

## Leakage prevention
- Group aggregates computed on *training fold only* via `FoldSafeGroupAggregator`.
- CV split is `GroupKFold` by `GroupId`.
