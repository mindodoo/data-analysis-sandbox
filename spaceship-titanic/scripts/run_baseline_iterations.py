#!/usr/bin/env python3
"""
Baseline-only iteration loop:
- Keep baseline engineered features (Agent 3 v1)
- Evaluate OOF probabilities with GroupKFold
- Sweep decision thresholds for accuracy
- Generate multiple submission variants for Kaggle A/B
- Write iteration report
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import json
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from run_agent2_cleaning import run_agent2_cleaning
from run_agent5_validation import Agent5Cfg, build_pipeline, FoldSafeGroupAggregator, RowFeatureEngineer, _split_feature_types


@dataclass(frozen=True)
class IterCfg:
    root: Path
    target: str = "Transported"
    group_col: str = "GroupId"
    n_splits: int = 5
    random_state: int = 42
    top_k_thresholds: int = 5


def _prepare_features(df: pd.DataFrame, target: str | None = None) -> pd.DataFrame:
    out = df.copy()
    drop_cols = ["PassengerId"]
    if target and target in out.columns:
        drop_cols.append(target)
    out = out.drop(columns=drop_cols, errors="ignore")
    for c in ["HomePlanet", "Destination", "Deck", "Side"]:
        if c in out.columns:
            out[c] = out[c].astype("string")
    return out


def _load_best_hgb_params(root: Path) -> dict:
    p = root / "optimized_model" / "best_params.json"
    if not p.exists():
        return {
            "learning_rate": 0.08,
            "max_depth": 6,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "l2_regularization": 0.0,
            "max_bins": 255,
        }
    return json.loads(p.read_text(encoding="utf-8"))


def _fit_predict_oof(train_clean: pd.DataFrame, cfg: IterCfg, model_params: dict) -> tuple[np.ndarray, np.ndarray]:
    y = train_clean[cfg.target].astype(int).to_numpy()
    groups = train_clean[cfg.group_col].astype(str).to_numpy()
    X_raw = _prepare_features(train_clean, target=cfg.target)

    gkf = GroupKFold(n_splits=cfg.n_splits)
    oof_proba = np.zeros(len(X_raw), dtype=float)

    for tr_idx, va_idx in gkf.split(X_raw, y, groups=groups):
        X_tr_raw, X_va_raw = X_raw.iloc[tr_idx].copy(), X_raw.iloc[va_idx].copy()
        y_tr = y[tr_idx]

        # Infer categorical/numeric columns using fold-safe transformations
        tmp = RowFeatureEngineer().fit_transform(X_tr_raw)
        tmp = FoldSafeGroupAggregator(group_col=cfg.group_col).fit(tmp).transform(tmp)
        tmp_model = tmp.drop(columns=[cfg.group_col], errors="ignore")
        cat_cols, num_cols = _split_feature_types(tmp_model)

        agent5_cfg = Agent5Cfg(root=cfg.root, n_splits=cfg.n_splits, random_state=cfg.random_state)
        pipe = build_pipeline(cat_cols=cat_cols, num_cols=num_cols, cfg=agent5_cfg)
        pipe.set_params(**{f"model__{k}": v for k, v in model_params.items()})
        pipe.fit(X_tr_raw, y_tr)
        oof_proba[va_idx] = pipe.predict_proba(X_va_raw)[:, 1]

    oof_pred = (oof_proba >= 0.5).astype(int)
    return oof_proba, oof_pred


def _threshold_grid(y: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    rows = []
    for t in np.linspace(0.35, 0.65, 61):
        pred = (proba >= t).astype(int)
        rows.append(
            {
                "threshold": round(float(t), 3),
                "oof_accuracy": float(accuracy_score(y, pred)),
                "oof_roc_auc": float(roc_auc_score(y, proba)),
                "positive_rate": float(pred.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["oof_accuracy", "threshold"], ascending=[False, True])


def _fit_full_and_predict_test(train_clean: pd.DataFrame, test_clean: pd.DataFrame, cfg: IterCfg, model_params: dict, threshold: float) -> pd.DataFrame:
    y = train_clean[cfg.target].astype(int).to_numpy()
    X_train_raw = _prepare_features(train_clean, target=cfg.target)
    X_test_raw = _prepare_features(test_clean, target=None)

    # Infer columns from full-train fold-safe style transform
    tmp = RowFeatureEngineer().fit_transform(X_train_raw)
    tmp = FoldSafeGroupAggregator(group_col=cfg.group_col).fit(tmp).transform(tmp)
    tmp_model = tmp.drop(columns=[cfg.group_col], errors="ignore")
    cat_cols, num_cols = _split_feature_types(tmp_model)

    agent5_cfg = Agent5Cfg(root=cfg.root, n_splits=cfg.n_splits, random_state=cfg.random_state)
    pipe = build_pipeline(cat_cols=cat_cols, num_cols=num_cols, cfg=agent5_cfg)
    pipe.set_params(**{f"model__{k}": v for k, v in model_params.items()})
    pipe.fit(X_train_raw, y)

    proba = pipe.predict_proba(X_test_raw)[:, 1]
    pred = (proba >= threshold)
    return pd.DataFrame({"PassengerId": test_clean["PassengerId"].values, "Transported": pred})


def run_iterations() -> dict:
    cfg = IterCfg(root=Path("/Users/mindodoo/Projects/data_ana/notebooks/spaceship-titanic"))
    reports = cfg.root / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    train_raw = pd.read_csv(cfg.root / "train.csv")
    test_raw = pd.read_csv(cfg.root / "test.csv")
    train_clean, test_clean, _ = run_agent2_cleaning(train_raw, test_raw)

    params = _load_best_hgb_params(cfg.root)

    oof_proba, oof_pred = _fit_predict_oof(train_clean, cfg, params)
    y = train_clean[cfg.target].astype(int).to_numpy()
    base_acc = float(accuracy_score(y, oof_pred))
    base_auc = float(roc_auc_score(y, oof_proba))

    grid = _threshold_grid(y, oof_proba)
    top = grid.head(cfg.top_k_thresholds).copy()

    submission_files = []
    for _, row in top.iterrows():
        thr = float(row["threshold"])
        sub = _fit_full_and_predict_test(train_clean, test_clean, cfg, params, threshold=thr)
        out = cfg.root / f"submission_baseline_t{thr:.3f}.csv"
        sub.to_csv(out, index=False)
        submission_files.append(str(out))

    # keep a canonical "best-threshold" baseline submission alias
    best_thr = float(top.iloc[0]["threshold"])
    best_sub = _fit_full_and_predict_test(train_clean, test_clean, cfg, params, threshold=best_thr)
    best_alias = cfg.root / "submission_baseline_best_threshold.csv"
    best_sub.to_csv(best_alias, index=False)

    report = (
        "# Baseline Iteration Report\n\n"
        "Using baseline engineered features (no advanced v2/v3 features).\n\n"
        f"- OOF accuracy @0.5: **{base_acc:.4f}**\n"
        f"- OOF ROC-AUC: **{base_auc:.4f}**\n\n"
        "## Top threshold candidates (by OOF accuracy)\n\n"
        "```text\n"
        + top.round(4).to_string(index=False)
        + "\n```\n\n"
        "## Generated submission variants\n\n"
        + "\n".join([f"- `{Path(p).name}`" for p in submission_files])
        + f"\n- `{best_alias.name}` (alias for top threshold)\n"
    )
    (reports / "baseline_iteration_report.md").write_text(report, encoding="utf-8")

    return {
        "base_acc": base_acc,
        "base_auc": base_auc,
        "top_thresholds": top,
        "submission_files": submission_files,
        "best_alias": str(best_alias),
    }


def main() -> None:
    res = run_iterations()
    print(f"OOF accuracy @0.5: {res['base_acc']:.4f}")
    print(f"OOF ROC-AUC: {res['base_auc']:.4f}")
    print("\nTop thresholds:")
    print(res["top_thresholds"].to_string(index=False))
    print("\nBest alias:", res["best_alias"])


if __name__ == "__main__":
    main()

