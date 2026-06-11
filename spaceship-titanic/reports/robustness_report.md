# Robustness Report — Agent 5

Robustness in this competition primarily depends on split strategy (group-safe). The model is deterministic given the fixed data and random_state; additional robustness tests can include:
- Vary model `random_state` and compare OOF metrics
- Remove group aggregate features and compare
- Compare performance on solo vs non-solo groups
