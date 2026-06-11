#!/usr/bin/env python3
"""
Re-evaluate improvement opportunities beyond Agent 6 tuned HGB.
Keeps saved best_params / best_pipeline.joblib unchanged — evaluation only.

Writes: reports/improvement_opportunities.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from run_agent5_validation import (
    Agent5Cfg,
    FoldSafeGroupAggregator,
    RowFeatureEngineer,
    _split_feature_types,
    build_pipeline,
)
from run_agent2_cleaning import run_agent2_cleaning


def _prepare(train_clean: pd.DataFrame, cfg: Agent5Cfg):
    df = train_clean.copy()
    y = df[cfg.target].astype(int).to_numpy()
    groups = df[cfg.group_col].astype(str).to_numpy()
    X = df.drop(columns=[c for c in ["PassengerId", cfg.target] if c in df.columns]).copy()
    for c in ["HomePlanet", "Destination", "Deck", "Side"]:
        if c in X.columns:
            X[c] = X[c].astype("string")
    return X, y, groups


def _build_pipe(cat_cols: list[str], num_cols: list[str], cfg: Agent5Cfg, model_params: dict, skip_group_agg: bool) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )
    model = HistGradientBoostingClassifier(random_state=cfg.random_state, **model_params)
    steps = [("row_features", RowFeatureEngineer())]
    if not skip_group_agg:
        steps.append(("group_agg", FoldSafeGroupAggregator(group_col=cfg.group_col)))
    steps.extend([("pre", pre), ("model", model)])
    return Pipeline(steps)


def _oof_proba(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    cfg: Agent5Cfg,
    model_params: dict,
    drop_cols: list[str] | None = None,
    skip_group_agg: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return OOF probabilities and predictions (threshold=0.5)."""
    gkf = GroupKFold(n_splits=cfg.n_splits)
    oof_proba = np.zeros(len(X), dtype=float)

    if drop_cols:
        X = X.drop(columns=[c for c in drop_cols if c in X.columns], errors="ignore")

    for tr_idx, va_idx in gkf.split(X, y, groups=groups):
        X_tr, X_va = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
        y_tr = y[tr_idx]

        tmp = RowFeatureEngineer().fit_transform(X_tr)
        if not skip_group_agg:
            tmp = FoldSafeGroupAggregator(group_col=cfg.group_col).fit(tmp).transform(tmp)
        tmp_model = tmp.drop(columns=[cfg.group_col], errors="ignore")
        cat_cols, num_cols = _split_feature_types(tmp_model)

        pipe = _build_pipe(cat_cols, num_cols, cfg, model_params, skip_group_agg)
        pipe.fit(X_tr, y_tr)
        oof_proba[va_idx] = pipe.predict_proba(X_va)[:, 1]

    oof_pred = (oof_proba >= 0.5).astype(int)
    return oof_proba, oof_pred


def _best_threshold(y: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    best_t, best_acc = 0.5, 0.0
    for t in np.linspace(0.35, 0.65, 61):
        acc = accuracy_score(y, (proba >= t).astype(int))
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return best_t, best_acc


def _oof_ensemble(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    cfg: Agent5Cfg,
    hgb_params: dict,
    weight_hgb: float = 0.7,
) -> tuple[float, np.ndarray]:
    gkf = GroupKFold(n_splits=cfg.n_splits)
    oof_proba = np.zeros(len(X), dtype=float)

    for tr_idx, va_idx in gkf.split(X, y, groups=groups):
        X_tr, X_va = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
        y_tr = y[tr_idx]

        tmp = RowFeatureEngineer().fit_transform(X_tr)
        tmp = FoldSafeGroupAggregator(group_col=cfg.group_col).fit(tmp).transform(tmp)
        tmp_model = tmp.drop(columns=[cfg.group_col], errors="ignore")
        cat_cols, num_cols = _split_feature_types(tmp_model)

        pre = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
                ("num", "passthrough", num_cols),
            ],
            remainder="drop",
        )
        hgb = HistGradientBoostingClassifier(random_state=cfg.random_state, **hgb_params)
        pipe_hgb = Pipeline(
            [("row_features", RowFeatureEngineer()), ("group_agg", FoldSafeGroupAggregator(cfg.group_col)),
             ("pre", pre), ("model", hgb)]
        )
        pre_lr = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                ("num", "passthrough", num_cols),
            ],
            remainder="drop",
            sparse_threshold=0.3,
        )
        lr = LogisticRegression(solver="saga", max_iter=2000, random_state=cfg.random_state, n_jobs=-1)
        pipe_lr = Pipeline(
            [("row_features", RowFeatureEngineer()), ("group_agg", FoldSafeGroupAggregator(cfg.group_col)),
             ("pre", pre_lr), ("model", lr)]
        )

        pipe_hgb.fit(X_tr, y_tr)
        pipe_lr.fit(X_tr, y_tr)
        p_hgb = pipe_hgb.predict_proba(X_va)[:, 1]
        p_lr = pipe_lr.predict_proba(X_va)[:, 1]
        oof_proba[va_idx] = weight_hgb * p_hgb + (1 - weight_hgb) * p_lr

    acc = accuracy_score(y, (oof_proba >= 0.5).astype(int))
    return acc, oof_proba


# Columns with ~0 permutation importance (Agent 3) — candidates to drop
# Drop only columns not required by RowFeatureEngineer / FoldSafeGroupAggregator
ZERO_IMPORTANCE_DROP = [
    "Spend_Any",
    "Deck_is_missing",
    "AgeBin",
    "Side_is_missing",
    "CabinNum_is_missing",
    "CryoSleep_is_missing",
    "Destination_is_missing",
    "HomePlanet_is_missing",
    "Age_is_missing",
]

GROUP_FIT_COLS_DROP = [
    "GroupSize_fit",
    "GroupTotalSpend_fit",
    "GroupMeanAge_fit",
    "GroupAnyVIP_fit",
    "GroupAnyCryo_fit",
    "GroupSpendPerPerson_fit",
    "GroupIsSolo_fit",
]


def main() -> None:
    root = Path("/Users/mindodoo/Projects/data_ana/notebooks/spaceship-titanic")
    cfg = Agent5Cfg(root=root)
    reports = root / "reports"

    params_path = root / "optimized_model" / "best_params.json"
    best_params = json.loads(params_path.read_text(encoding="utf-8"))

    train_raw = pd.read_csv(root / "train.csv")
    test_raw = pd.read_csv(root / "test.csv")
    train_clean, _, _ = run_agent2_cleaning(train_raw, test_raw)
    X, y, groups = _prepare(train_clean, cfg)

    rows = []

    # 1) Current tuned (reference)
    proba, pred = _oof_proba(X, y, groups, cfg, best_params)
    acc = accuracy_score(y, pred)
    rows.append({"experiment": "A. Current tuned HGB (threshold=0.5)", "oof_accuracy": acc, "notes": "Agent 6 saved params"})

    # 2) Threshold optimization on same OOF proba
    t_star, acc_t = _best_threshold(y, proba)
    rows.append({
        "experiment": f"B. Threshold tune on OOF (t={t_star:.3f})",
        "oof_accuracy": acc_t,
        "notes": "No retrain; calibrate decision threshold only",
    })

    # 3) Drop zero-importance columns before pipeline
    proba3, pred3 = _oof_proba(X, y, groups, cfg, best_params, drop_cols=ZERO_IMPORTANCE_DROP)
    rows.append({
        "experiment": "C. Tuned HGB + drop ~0-importance cols",
        "oof_accuracy": accuracy_score(y, pred3),
        "notes": f"Dropped {len(ZERO_IMPORTANCE_DROP)} cols",
    })

    # 4) Without fold-safe group aggregate step
    proba4, pred4 = _oof_proba(X, y, groups, cfg, best_params, skip_group_agg=True)
    rows.append({
        "experiment": "D. Tuned HGB without group-aggregate step",
        "oof_accuracy": accuracy_score(y, pred4),
        "notes": "Skips FoldSafeGroupAggregator",
    })

    # 5) Ensemble HGB + LogisticRegression
    acc6, _ = _oof_ensemble(X, y, groups, cfg, best_params, weight_hgb=0.75)
    rows.append({
        "experiment": "E. Ensemble 75% HGB + 25% LogisticRegression",
        "oof_accuracy": acc6,
        "notes": "Alternative model from Agent 4",
    })

    # 6) Larger search headroom (single manual tweak — slightly more leaves, lower lr)
    manual = {**best_params, "learning_rate": best_params["learning_rate"] * 0.9, "max_leaf_nodes": min(127, int(best_params["max_leaf_nodes"]) + 15)}
    proba7, pred7 = _oof_proba(X, y, groups, cfg, manual)
    rows.append({
        "experiment": "F. Manual tweak (lr*0.9, +15 leaf nodes)",
        "oof_accuracy": accuracy_score(y, pred7),
        "notes": "Not saved; quick neighbor of best params",
    })

    res = pd.DataFrame(rows).sort_values("oof_accuracy", ascending=False)
    baseline_acc = float(res.loc[res["experiment"].str.startswith("A."), "oof_accuracy"].iloc[0])
    res["delta_vs_current"] = res["oof_accuracy"] - baseline_acc

    best_row = res.iloc[0]
    report = f"""# Improvement Opportunities Review

**Current kept result (Agent 6):** tuned `HistGradientBoostingClassifier`  
**Reference OOF accuracy (A):** {baseline_acc:.4f}  
**Best params file:** `optimized_model/best_params.json` (unchanged)

> Kaggle test labels are hidden — all metrics are **group-safe OOF** (`GroupKFold` by `GroupId`).

## Experiment results (sorted by OOF accuracy)

```text
{res.round(4).to_string(index=False)}
```

## Interpretation

| Finding | Implication |
|---------|-------------|
| Best experiment | **{best_row['experiment']}** → {best_row['oof_accuracy']:.4f} (Δ {best_row['delta_vs_current']:+.4f}) |
| Threshold tuning (B) | Often a **free** gain without retraining — worth applying at inference if approved |
| Feature drops (C) | Mixed; only adopt if gain is stable across reruns |
| No group agg (D) | Tests whether group step helps |
| Ensemble (E) | May help robustness; adds complexity and training cost |
| Manual tweak (F) | Small neighborhood search; not a full re-tune |

## Recommendations (for your review)

1. **Keep** Agent 6 `best_pipeline.joblib` and `best_params.json` as the canonical model unless you approve switching.
2. **Low-risk next step:** apply **threshold tuning (B)** on OOF probabilities when generating `submission.csv` (still on hold).
3. **Medium effort:** try **LightGBM/CatBoost** (not installed) with same fold-safe pipeline — often +0.01–0.02 on tabular Kaggle tasks.
4. **Higher effort:** expanded `RandomizedSearchCV` (n_iter 50+) or Optuna with group CV.

## Not evaluated in this pass (optional later)

- Target encoding with out-of-fold encoding (leakage-safe)
- Stacking multiple tree models
- Pseudo-labeling test set (risky for leaderboard)
- Re-engineering group features strictly inside each CV fold only (already done in Agent 5/6 pipeline)

## Decision needed

Please review and reply with one of:

- **Keep current** — proceed to submission generation later with saved tuned model only
- **Apply threshold** — keep model, use optimized threshold from experiment B
- **Adopt experiment X** — specify letter (B–F) to replace canonical model
- **Run extended tuning** — approve LightGBM install + deeper search

**submission.csv remains ON HOLD until you approve.**
"""

    (reports / "improvement_opportunities.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nWrote: {reports / 'improvement_opportunities.md'}")


if __name__ == "__main__":
    main()
