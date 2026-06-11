#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from run_agent2_cleaning import run_agent2_cleaning
from run_agent5_validation import Agent5Cfg, build_pipeline, FoldSafeGroupAggregator, RowFeatureEngineer, _split_feature_types


@dataclass(frozen=True)
class SafeCfg:
    root: Path
    target: str = "Transported"
    group_col: str = "GroupId"
    n_splits: int = 5
    random_state: int = 42


CAT_NATIVE = ["HomePlanet", "Destination", "Deck", "Side", "GroupId", "DeckSide", "AgeBin", "CabinNumBin100"]


def _load_hgb_params(root: Path) -> dict:
    p = root / "optimized_model" / "best_params.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"learning_rate": 0.08, "max_depth": 6, "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 0.0, "max_bins": 255}


def _load_cat_params(root: Path) -> dict:
    p = root / "optimized_model" / "catboost_native_best_params.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"iterations": 390, "learning_rate": 0.0769, "depth": 7, "l2_leaf_reg": 6.9961, "bagging_temperature": 0.7003, "random_strength": 0.4685, "border_count": 128}


def _prep_clean(df: pd.DataFrame, target: str | None) -> pd.DataFrame:
    drop = ["PassengerId"]
    if target and target in df.columns:
        drop.append(target)
    X = df.drop(columns=drop, errors="ignore").copy()
    for c in ["HomePlanet", "Destination", "Deck", "Side"]:
        if c in X.columns:
            X[c] = X[c].astype("string")
    return X


def _prep_cat_native(df: pd.DataFrame, target: str | None) -> pd.DataFrame:
    X = _prep_clean(df, target)
    X = RowFeatureEngineer().fit_transform(X)
    for c in ["HomePlanet", "Destination", "Deck", "Side", "GroupId", "DeckSide"]:
        if c in X.columns:
            X[c] = X[c].astype(str)
    return X


def _cat_idx(cols: list[str]) -> list[int]:
    return [i for i, c in enumerate(cols) if c in CAT_NATIVE]


def _metrics(y: np.ndarray, proba: np.ndarray, thr: float = 0.5) -> dict:
    pred = (proba >= thr).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "positive_rate": float(pred.mean()),
    }


def run_safe_v2() -> dict:
    cfg = SafeCfg(root=Path("/Users/mindodoo/Projects/data_ana/notebooks/spaceship-titanic"))
    reports = cfg.root / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    train_raw = pd.read_csv(cfg.root / "train.csv")
    test_raw = pd.read_csv(cfg.root / "test.csv")
    train_clean, test_clean, _ = run_agent2_cleaning(train_raw, test_raw)

    y = train_clean[cfg.target].astype(int).to_numpy()
    groups = train_clean[cfg.group_col].astype(str).to_numpy()
    X_hgb = _prep_clean(train_clean, cfg.target)
    X_hgb_test = _prep_clean(test_clean, None)
    X_cat = _prep_cat_native(train_clean, cfg.target)
    X_cat_test = _prep_cat_native(test_clean, None)

    hgb_params = _load_hgb_params(cfg.root)
    cat_params = _load_cat_params(cfg.root)

    gkf = GroupKFold(n_splits=cfg.n_splits)
    oof_hgb = np.zeros(len(X_hgb), dtype=float)
    oof_cat = np.zeros(len(X_hgb), dtype=float)

    # base model OOF predictions (strict fold-safe)
    for tr_idx, va_idx in gkf.split(X_hgb, y, groups):
        Xh_tr, Xh_va = X_hgb.iloc[tr_idx].copy(), X_hgb.iloc[va_idx].copy()
        Xc_tr, Xc_va = X_cat.iloc[tr_idx].copy(), X_cat.iloc[va_idx].copy()
        y_tr = y[tr_idx]

        tmp = RowFeatureEngineer().fit_transform(Xh_tr)
        tmp = FoldSafeGroupAggregator(group_col=cfg.group_col).fit(tmp).transform(tmp)
        cat_cols, num_cols = _split_feature_types(tmp.drop(columns=[cfg.group_col], errors="ignore"))
        agent5_cfg = Agent5Cfg(root=cfg.root, n_splits=cfg.n_splits, random_state=cfg.random_state)
        hgb = build_pipeline(cat_cols=cat_cols, num_cols=num_cols, cfg=agent5_cfg)
        hgb.set_params(**{f"model__{k}": v for k, v in hgb_params.items()})
        hgb.fit(Xh_tr, y_tr)
        oof_hgb[va_idx] = hgb.predict_proba(Xh_va)[:, 1]

        agg = FoldSafeGroupAggregator(group_col=cfg.group_col)
        Xc_tr_g = agg.fit(Xc_tr).transform(Xc_tr)
        Xc_va_g = agg.transform(Xc_va)
        cat = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="Accuracy",
            random_seed=cfg.random_state,
            verbose=0,
            allow_writing_files=False,
            **cat_params,
        )
        cat.fit(Xc_tr_g, y_tr, cat_features=_cat_idx(list(Xc_tr_g.columns)), eval_set=(Xc_va_g, y[va_idx]), use_best_model=True, early_stopping_rounds=50)
        oof_cat[va_idx] = cat.predict_proba(Xc_va_g)[:, 1]

    # second-level strict OOF for meta model
    meta_X = pd.DataFrame({"p_hgb": oof_hgb, "p_cat": oof_cat})
    oof_stack = np.zeros(len(meta_X), dtype=float)
    for tr_idx, va_idx in gkf.split(meta_X, y, groups):
        meta = LogisticRegression(max_iter=2000, random_state=cfg.random_state)
        meta.fit(meta_X.iloc[tr_idx], y[tr_idx])
        oof_stack[va_idx] = meta.predict_proba(meta_X.iloc[va_idx])[:, 1]

    # choose threshold from strict OOF
    thr_rows = []
    for t in np.linspace(0.45, 0.55, 21):
        m = _metrics(y, oof_stack, thr=float(t))
        thr_rows.append({"threshold": float(t), **m})
    thr_df = pd.DataFrame(thr_rows).sort_values(["accuracy", "threshold"], ascending=[False, True])
    best_thr = float(thr_df.iloc[0]["threshold"])

    # fit base models on full train then meta on full OOF meta_X
    tmp = RowFeatureEngineer().fit_transform(X_hgb)
    agg_full = FoldSafeGroupAggregator(group_col=cfg.group_col)
    tmp = agg_full.fit(tmp).transform(tmp)
    cat_cols, num_cols = _split_feature_types(tmp.drop(columns=[cfg.group_col], errors="ignore"))
    agent5_cfg = Agent5Cfg(root=cfg.root, n_splits=cfg.n_splits, random_state=cfg.random_state)
    hgb_full = build_pipeline(cat_cols=cat_cols, num_cols=num_cols, cfg=agent5_cfg)
    hgb_full.set_params(**{f"model__{k}": v for k, v in hgb_params.items()})
    hgb_full.fit(X_hgb, y)
    p_hgb_test = hgb_full.predict_proba(X_hgb_test)[:, 1]

    agg_cat = FoldSafeGroupAggregator(group_col=cfg.group_col)
    Xc_tr_g = agg_cat.fit(X_cat).transform(X_cat)
    Xc_te_g = agg_cat.transform(X_cat_test)
    cat_full = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Accuracy",
        random_seed=cfg.random_state,
        verbose=0,
        allow_writing_files=False,
        **cat_params,
    )
    cat_full.fit(Xc_tr_g, y, cat_features=_cat_idx(list(Xc_tr_g.columns)))
    p_cat_test = cat_full.predict_proba(Xc_te_g)[:, 1]

    meta_full = LogisticRegression(max_iter=2000, random_state=cfg.random_state)
    meta_full.fit(meta_X, y)
    p_stack_test = meta_full.predict_proba(pd.DataFrame({"p_hgb": p_hgb_test, "p_cat": p_cat_test}))[:, 1]

    sub_best = pd.DataFrame({"PassengerId": test_clean["PassengerId"].values, "Transported": (p_stack_test >= best_thr)})
    sub_05 = pd.DataFrame({"PassengerId": test_clean["PassengerId"].values, "Transported": (p_stack_test >= 0.5)})
    path_best = cfg.root / "submission_stacked_safe_v2.csv"
    path_05 = cfg.root / "submission_stacked_safe_v2_t0.500.csv"
    sub_best.to_csv(path_best, index=False)
    sub_05.to_csv(path_05, index=False)

    comp = pd.DataFrame(
        [
            {"model": "HGB base OOF", **_metrics(y, oof_hgb)},
            {"model": "CatBoost base OOF", **_metrics(y, oof_cat)},
            {"model": f"Stack safe v2 OOF (thr={best_thr:.3f})", **_metrics(y, oof_stack, best_thr)},
        ]
    )
    (reports / "stacked_ensemble_safe_v2_report.md").write_text(
        "# Stacked Ensemble Safe v2 Report\n\n"
        "Strict leakage-safe setup: base OOF via GroupKFold, then meta OOF via second GroupKFold.\n\n"
        "## OOF comparison\n\n```text\n"
        + comp.round(4).to_string(index=False)
        + "\n```\n\n"
        "## Threshold search (safe stack)\n\n```text\n"
        + thr_df.head(10).round(4).to_string(index=False)
        + "\n```\n\n"
        f"Chosen threshold: **{best_thr:.3f}**\n\n"
        f"Submissions: `{path_best.name}`, `{path_05.name}`\n",
        encoding="utf-8",
    )
    pd.DataFrame({"y": y, "group": groups, "p_hgb": oof_hgb, "p_cat": oof_cat, "p_stack_safe_v2": oof_stack}).to_parquet(
        reports / "stacked_safe_v2_oof.parquet", index=False
    )

    return {"comparison": comp, "threshold_table": thr_df, "best_threshold": best_thr, "submission_best": str(path_best), "submission_05": str(path_05)}


def main() -> None:
    res = run_safe_v2()
    print(res["comparison"].round(4).to_string(index=False))
    print("Best threshold:", round(res["best_threshold"], 3))
    print("Submission:", res["submission_best"])
    print("Submission@0.5:", res["submission_05"])


if __name__ == "__main__":
    main()

