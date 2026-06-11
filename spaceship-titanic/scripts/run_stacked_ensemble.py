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
from run_agent3_feature_engineering import add_engineered_features
from run_agent5_validation import Agent5Cfg, build_pipeline, FoldSafeGroupAggregator, RowFeatureEngineer, _split_feature_types


@dataclass(frozen=True)
class StackCfg:
    root: Path
    target: str = "Transported"
    group_col: str = "GroupId"
    n_splits: int = 5
    random_state: int = 42


CATBOOST_NATIVE_CATS = [
    "HomePlanet",
    "Destination",
    "Deck",
    "Side",
    "DeckSide",
    "GroupId",
    "AgeBin",
    "CabinNumBin100",
]


def _load_hgb_params(root: Path) -> dict:
    p = root / "optimized_model" / "best_params.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {
        "learning_rate": 0.08,
        "max_depth": 6,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 0.0,
        "max_bins": 255,
    }


def _load_catboost_params(root: Path) -> dict:
    p = root / "optimized_model" / "catboost_native_best_params.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {
        "iterations": 390,
        "learning_rate": 0.0769,
        "depth": 7,
        "l2_leaf_reg": 6.9961,
        "bagging_temperature": 0.7003,
        "random_strength": 0.4685,
        "border_count": 128,
    }


def _prepare_hgb_features(df: pd.DataFrame, target: str | None = None) -> pd.DataFrame:
    drop = ["PassengerId"]
    if target and target in df.columns:
        drop.append(target)
    X = df.drop(columns=drop, errors="ignore").copy()
    for c in ["HomePlanet", "Destination", "Deck", "Side"]:
        if c in X.columns:
            X[c] = X[c].astype("string")
    return X


def _prepare_cat_features(df: pd.DataFrame, target: str | None = None) -> pd.DataFrame:
    drop = ["PassengerId"]
    if target and target in df.columns:
        drop.append(target)
    X = df.drop(columns=drop, errors="ignore").copy()
    for c in ["HomePlanet", "Destination", "Deck", "Side", "GroupId"]:
        if c in X.columns:
            X[c] = X[c].astype(str)
    X = RowFeatureEngineer().fit_transform(X)
    return X


def _cat_idx(cols: list[str]) -> list[int]:
    return [i for i, c in enumerate(cols) if c in CATBOOST_NATIVE_CATS]


def run_stacking() -> dict:
    cfg = StackCfg(root=Path("/Users/mindodoo/Projects/data_ana/notebooks/spaceship-titanic"))
    reports = cfg.root / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    # Data
    train_raw = pd.read_csv(cfg.root / "train.csv")
    test_raw = pd.read_csv(cfg.root / "test.csv")
    train_clean, test_clean, _ = run_agent2_cleaning(train_raw, test_raw)
    train_v1, test_v1, _ = add_engineered_features(train_clean, test_clean)

    y = train_v1[cfg.target].astype(int).to_numpy()
    groups = train_v1[cfg.group_col].astype(str).to_numpy()

    X_hgb = _prepare_hgb_features(train_v1, cfg.target)
    X_hgb_test = _prepare_hgb_features(test_v1, None)

    X_cat_raw = _prepare_cat_features(train_v1, cfg.target)
    X_cat_test_raw = _prepare_cat_features(test_v1, None)

    gkf = GroupKFold(n_splits=cfg.n_splits)
    oof_hgb = np.zeros(len(train_v1), dtype=float)
    oof_cat = np.zeros(len(train_v1), dtype=float)
    oof_lr = np.zeros(len(train_v1), dtype=float)

    hgb_params = _load_hgb_params(cfg.root)
    cat_params = _load_catboost_params(cfg.root)

    for tr_idx, va_idx in gkf.split(X_hgb, y, groups):
        Xh_tr, Xh_va = X_hgb.iloc[tr_idx].copy(), X_hgb.iloc[va_idx].copy()
        Xc_tr, Xc_va = X_cat_raw.iloc[tr_idx].copy(), X_cat_raw.iloc[va_idx].copy()
        y_tr = y[tr_idx]

        # HGB + LR share same sklearn-preprocessed path
        tmp = RowFeatureEngineer().fit_transform(Xh_tr)
        tmp = FoldSafeGroupAggregator(group_col=cfg.group_col).fit(tmp).transform(tmp)
        tmp_model = tmp.drop(columns=[cfg.group_col], errors="ignore")
        cat_cols, num_cols = _split_feature_types(tmp_model)

        agent5_cfg = Agent5Cfg(root=cfg.root, n_splits=cfg.n_splits, random_state=cfg.random_state)
        hgb_pipe = build_pipeline(cat_cols=cat_cols, num_cols=num_cols, cfg=agent5_cfg)
        hgb_pipe.set_params(**{f"model__{k}": v for k, v in hgb_params.items()})
        hgb_pipe.fit(Xh_tr, y_tr)
        oof_hgb[va_idx] = hgb_pipe.predict_proba(Xh_va)[:, 1]

        lr_pipe = build_pipeline(cat_cols=cat_cols, num_cols=num_cols, cfg=agent5_cfg)
        lr_pipe.set_params(
            model=LogisticRegression(solver="saga", max_iter=3000, n_jobs=-1, random_state=cfg.random_state)
        )
        lr_pipe.fit(Xh_tr, y_tr)
        oof_lr[va_idx] = lr_pipe.predict_proba(Xh_va)[:, 1]

        # CatBoost native with fold-safe group aggregation
        agg = FoldSafeGroupAggregator(group_col=cfg.group_col)
        Xc_tr_g = agg.fit(Xc_tr).transform(Xc_tr)
        Xc_va_g = agg.transform(Xc_va)
        cat_idx = _cat_idx(list(Xc_tr_g.columns))
        cat_model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="Accuracy",
            random_seed=cfg.random_state,
            verbose=0,
            allow_writing_files=False,
            **cat_params,
        )
        cat_model.fit(
            Xc_tr_g,
            y_tr,
            cat_features=cat_idx,
            eval_set=(Xc_va_g, y[va_idx]),
            use_best_model=True,
            early_stopping_rounds=50,
        )
        oof_cat[va_idx] = cat_model.predict_proba(Xc_va_g)[:, 1]

    # Meta model on OOF predictions
    meta_X = pd.DataFrame({"p_hgb": oof_hgb, "p_cat": oof_cat, "p_lr": oof_lr})
    meta = LogisticRegression(max_iter=3000, random_state=cfg.random_state)
    meta.fit(meta_X, y)
    oof_stack = meta.predict_proba(meta_X)[:, 1]

    def metrics_from_proba(proba: np.ndarray, thr: float = 0.5) -> dict:
        pred = (proba >= thr).astype(int)
        return {
            "accuracy": float(accuracy_score(y, pred)),
            "f1": float(f1_score(y, pred)),
            "roc_auc": float(roc_auc_score(y, proba)),
            "positive_rate": float(pred.mean()),
        }

    # threshold sweep for stacked model
    rows = []
    for t in np.linspace(0.40, 0.60, 41):
        m = metrics_from_proba(oof_stack, thr=float(t))
        rows.append({"threshold": float(t), **m})
    thr_df = pd.DataFrame(rows).sort_values(["accuracy", "threshold"], ascending=[False, True])
    best_thr = float(thr_df.iloc[0]["threshold"])

    m_hgb = metrics_from_proba(oof_hgb)
    m_cat = metrics_from_proba(oof_cat)
    m_lr = metrics_from_proba(oof_lr)
    m_stack = metrics_from_proba(oof_stack, best_thr)

    # Fit full models and generate test submissions
    # HGB full
    tmp_full = RowFeatureEngineer().fit_transform(X_hgb)
    agg_full_hgb = FoldSafeGroupAggregator(group_col=cfg.group_col)
    tmp_full = agg_full_hgb.fit(tmp_full).transform(tmp_full)
    cat_cols, num_cols = _split_feature_types(tmp_full.drop(columns=[cfg.group_col], errors="ignore"))
    agent5_cfg = Agent5Cfg(root=cfg.root, n_splits=cfg.n_splits, random_state=cfg.random_state)
    hgb_full = build_pipeline(cat_cols=cat_cols, num_cols=num_cols, cfg=agent5_cfg)
    hgb_full.set_params(**{f"model__{k}": v for k, v in hgb_params.items()})
    hgb_full.fit(X_hgb, y)
    p_hgb_test = hgb_full.predict_proba(X_hgb_test)[:, 1]

    # LR full
    lr_full = build_pipeline(cat_cols=cat_cols, num_cols=num_cols, cfg=agent5_cfg)
    lr_full.set_params(model=LogisticRegression(solver="saga", max_iter=3000, n_jobs=-1, random_state=cfg.random_state))
    lr_full.fit(X_hgb, y)
    p_lr_test = lr_full.predict_proba(X_hgb_test)[:, 1]

    # Cat full
    agg_full_cat = FoldSafeGroupAggregator(group_col=cfg.group_col)
    X_cat_train_g = agg_full_cat.fit(X_cat_raw).transform(X_cat_raw)
    X_cat_test_g = agg_full_cat.transform(X_cat_test_raw)
    cat_idx = _cat_idx(list(X_cat_train_g.columns))
    cat_full = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Accuracy",
        random_seed=cfg.random_state,
        verbose=0,
        allow_writing_files=False,
        **cat_params,
    )
    cat_full.fit(X_cat_train_g, y, cat_features=cat_idx)
    p_cat_test = cat_full.predict_proba(X_cat_test_g)[:, 1]

    test_meta = pd.DataFrame({"p_hgb": p_hgb_test, "p_cat": p_cat_test, "p_lr": p_lr_test})
    p_stack_test = meta.predict_proba(test_meta)[:, 1]

    # Output submissions
    sub_stack = pd.DataFrame(
        {
            "PassengerId": test_clean["PassengerId"].values,
            "Transported": (p_stack_test >= best_thr),
        }
    )
    sub_stack_path = cfg.root / "submission_stacked_ensemble.csv"
    sub_stack.to_csv(sub_stack_path, index=False)

    sub_stack_05 = pd.DataFrame(
        {
            "PassengerId": test_clean["PassengerId"].values,
            "Transported": (p_stack_test >= 0.5),
        }
    )
    sub_stack05_path = cfg.root / "submission_stacked_ensemble_t0.500.csv"
    sub_stack_05.to_csv(sub_stack05_path, index=False)

    # Reports
    comp = pd.DataFrame(
        [
            {"model": "HGB baseline", **m_hgb},
            {"model": "CatBoost native", **m_cat},
            {"model": "Logistic baseline", **m_lr},
            {"model": f"Stacked ensemble (thr={best_thr:.3f})", **m_stack},
        ]
    )
    (reports / "stacked_ensemble_report.md").write_text(
        "# Stacked Ensemble Report\n\n"
        "Group-aware stacking using OOF probabilities from HGB, CatBoost-native, and Logistic models.\n\n"
        "## OOF comparison\n\n"
        "```text\n"
        + comp.round(4).to_string(index=False)
        + "\n```\n\n"
        "## Threshold sweep (stacked)\n\n"
        "```text\n"
        + thr_df.head(10).round(4).to_string(index=False)
        + "\n```\n\n"
        f"Selected threshold: **{best_thr:.3f}**\n\n"
        "## Submission files\n"
        f"- `{sub_stack_path.name}`\n"
        f"- `{sub_stack05_path.name}`\n",
        encoding="utf-8",
    )

    oof_df = pd.DataFrame(
        {
            "y": y,
            "group": groups,
            "p_hgb": oof_hgb,
            "p_cat": oof_cat,
            "p_lr": oof_lr,
            "p_stack": oof_stack,
        }
    )
    oof_df.to_parquet(reports / "stacked_oof_predictions.parquet", index=False)

    return {
        "comparison": comp,
        "threshold_table": thr_df,
        "best_threshold": best_thr,
        "submission_stack": str(sub_stack_path),
        "submission_stack05": str(sub_stack05_path),
    }


def main() -> None:
    res = run_stacking()
    print(res["comparison"].round(4).to_string(index=False))
    print("Best threshold:", round(res["best_threshold"], 3))
    print("Submission:", res["submission_stack"])
    print("Submission@0.5:", res["submission_stack05"])


if __name__ == "__main__":
    main()

