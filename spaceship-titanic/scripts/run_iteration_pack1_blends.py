#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from run_agent2_cleaning import run_agent2_cleaning
from run_agent3_feature_engineering import add_engineered_features
from run_agent5_validation import Agent5Cfg, FoldSafeGroupAggregator, RowFeatureEngineer, _split_feature_types, build_pipeline
from run_stacked_ensemble import _load_catboost_params, _load_hgb_params, _prepare_cat_features, _prepare_hgb_features, _cat_idx


@dataclass(frozen=True)
class BlendCfg:
    root: Path
    target: str = "Transported"
    group_col: str = "GroupId"
    n_splits: int = 5
    random_state: int = 42
    top_k: int = 8


def get_model_probs(cfg: BlendCfg):
    train_raw = pd.read_csv(cfg.root / "train.csv")
    test_raw = pd.read_csv(cfg.root / "test.csv")
    train_clean, test_clean, _ = run_agent2_cleaning(train_raw, test_raw)
    train_v1, test_v1, _ = add_engineered_features(train_clean, test_clean)

    y = train_v1[cfg.target].astype(int).to_numpy()
    groups = train_v1[cfg.group_col].astype(str).to_numpy()
    X_hgb = _prepare_hgb_features(train_v1, cfg.target)
    X_hgb_test = _prepare_hgb_features(test_v1, None)
    X_cat = _prepare_cat_features(train_v1, cfg.target)
    X_cat_test = _prepare_cat_features(test_v1, None)

    gkf = GroupKFold(n_splits=cfg.n_splits)
    oof_hgb = np.zeros(len(train_v1), dtype=float)
    oof_cat = np.zeros(len(train_v1), dtype=float)
    oof_stack = np.zeros(len(train_v1), dtype=float)

    hgb_params = _load_hgb_params(cfg.root)
    cat_params = _load_catboost_params(cfg.root)

    # To generate stack probs we also need logistic component per fold
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
        p_hgb = hgb.predict_proba(Xh_va)[:, 1]
        oof_hgb[va_idx] = p_hgb

        lr = build_pipeline(cat_cols=cat_cols, num_cols=num_cols, cfg=agent5_cfg)
        lr.set_params(model=LogisticRegression(solver="saga", max_iter=3000, n_jobs=-1, random_state=cfg.random_state))
        lr.fit(Xh_tr, y_tr)
        p_lr = lr.predict_proba(Xh_va)[:, 1]

        agg = FoldSafeGroupAggregator(group_col=cfg.group_col)
        Xc_tr_g = agg.fit(Xc_tr).transform(Xc_tr)
        Xc_va_g = agg.transform(Xc_va)
        cat_idx = _cat_idx(list(Xc_tr_g.columns))
        cat = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="Accuracy",
            random_seed=cfg.random_state,
            verbose=0,
            allow_writing_files=False,
            **cat_params,
        )
        cat.fit(Xc_tr_g, y_tr, cat_features=cat_idx, eval_set=(Xc_va_g, y[va_idx]), use_best_model=True, early_stopping_rounds=50)
        p_cat = cat.predict_proba(Xc_va_g)[:, 1]
        oof_cat[va_idx] = p_cat

        # reconstruct same stack meta model fold-locally to avoid leakage
        # simple equalized local fit on train partition probs
        p_hgb_tr = hgb.predict_proba(Xh_tr)[:, 1]
        p_lr_tr = lr.predict_proba(Xh_tr)[:, 1]
        p_cat_tr = cat.predict_proba(Xc_tr_g)[:, 1]
        meta = LogisticRegression(max_iter=3000, random_state=cfg.random_state)
        meta.fit(np.c_[p_hgb_tr, p_cat_tr, p_lr_tr], y_tr)
        oof_stack[va_idx] = meta.predict_proba(np.c_[p_hgb, p_cat, p_lr])[:, 1]

    # Fit full models for test predictions
    Xh_full = X_hgb.copy()
    Xc_full = X_cat.copy()
    tmp = RowFeatureEngineer().fit_transform(Xh_full)
    agg_hgb = FoldSafeGroupAggregator(group_col=cfg.group_col)
    tmp = agg_hgb.fit(tmp).transform(tmp)
    cat_cols, num_cols = _split_feature_types(tmp.drop(columns=[cfg.group_col], errors="ignore"))
    agent5_cfg = Agent5Cfg(root=cfg.root, n_splits=cfg.n_splits, random_state=cfg.random_state)

    hgb_full = build_pipeline(cat_cols=cat_cols, num_cols=num_cols, cfg=agent5_cfg)
    hgb_full.set_params(**{f"model__{k}": v for k, v in hgb_params.items()})
    hgb_full.fit(Xh_full, y)
    p_hgb_test = hgb_full.predict_proba(X_hgb_test)[:, 1]

    lr_full = build_pipeline(cat_cols=cat_cols, num_cols=num_cols, cfg=agent5_cfg)
    lr_full.set_params(model=LogisticRegression(solver="saga", max_iter=3000, n_jobs=-1, random_state=cfg.random_state))
    lr_full.fit(Xh_full, y)
    p_lr_test = lr_full.predict_proba(X_hgb_test)[:, 1]

    agg_cat = FoldSafeGroupAggregator(group_col=cfg.group_col)
    Xc_tr_g = agg_cat.fit(Xc_full).transform(Xc_full)
    Xc_te_g = agg_cat.transform(X_cat_test)
    cat_idx = _cat_idx(list(Xc_tr_g.columns))
    cat_full = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Accuracy",
        random_seed=cfg.random_state,
        verbose=0,
        allow_writing_files=False,
        **cat_params,
    )
    cat_full.fit(Xc_tr_g, y, cat_features=cat_idx)
    p_cat_test = cat_full.predict_proba(Xc_te_g)[:, 1]

    meta_full = LogisticRegression(max_iter=3000, random_state=cfg.random_state)
    meta_full.fit(np.c_[oof_hgb, oof_cat, oof_stack], y)
    p_stack_test = meta_full.predict_proba(np.c_[p_hgb_test, p_cat_test, p_lr_test])[:, 1]

    return {
        "y": y,
        "oof_hgb": oof_hgb,
        "oof_cat": oof_cat,
        "oof_stack": oof_stack,
        "p_hgb_test": p_hgb_test,
        "p_cat_test": p_cat_test,
        "p_stack_test": p_stack_test,
        "test_ids": test_clean["PassengerId"].values,
    }


def run_pack1():
    cfg = BlendCfg(root=Path("/Users/mindodoo/Projects/data_ana/notebooks/spaceship-titanic"))
    reports = cfg.root / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    d = get_model_probs(cfg)
    y = d["y"]

    # Blend grid over three strong components
    grid = []
    w_vals = np.arange(0.0, 1.01, 0.1)
    for wh, wc, ws in product(w_vals, w_vals, w_vals):
        if abs((wh + wc + ws) - 1.0) > 1e-9:
            continue
        p = wh * d["oof_hgb"] + wc * d["oof_cat"] + ws * d["oof_stack"]
        auc = roc_auc_score(y, p)
        for t in np.linspace(0.45, 0.55, 21):
            pred = (p >= t).astype(int)
            acc = accuracy_score(y, pred)
            grid.append(
                {
                    "w_hgb": round(float(wh), 2),
                    "w_cat": round(float(wc), 2),
                    "w_stack": round(float(ws), 2),
                    "threshold": round(float(t), 3),
                    "oof_accuracy": float(acc),
                    "oof_roc_auc": float(auc),
                    "positive_rate": float(pred.mean()),
                }
            )

    grid_df = pd.DataFrame(grid).sort_values(["oof_accuracy", "oof_roc_auc"], ascending=[False, False])
    top = grid_df.head(cfg.top_k).copy()

    # Create submissions for top blends
    submission_files = []
    for i, row in top.reset_index(drop=True).iterrows():
        p_test = row.w_hgb * d["p_hgb_test"] + row.w_cat * d["p_cat_test"] + row.w_stack * d["p_stack_test"]
        sub = pd.DataFrame(
            {"PassengerId": d["test_ids"], "Transported": (p_test >= row.threshold)}
        )
        fn = cfg.root / f"submission_blend_{i+1:02d}_wh{row.w_hgb:.2f}_wc{row.w_cat:.2f}_ws{row.w_stack:.2f}_t{row.threshold:.3f}.csv"
        sub.to_csv(fn, index=False)
        submission_files.append(fn.name)

    # write reports
    (reports / "iteration_pack1_report.md").write_text(
        "# Iteration Pack #1 — Blend Search\n\n"
        "Blend search across HGB + CatBoost + Stacked probabilities.\n\n"
        "## Top candidates\n\n"
        "```text\n"
        + top.round(4).to_string(index=False)
        + "\n```\n\n"
        "## Generated submissions\n\n"
        + "\n".join([f"- `{f}`" for f in submission_files])
        + "\n",
        encoding="utf-8",
    )
    top.to_csv(reports / "iteration_pack1_top_candidates.csv", index=False)

    return top, submission_files


def main():
    top, files = run_pack1()
    print(top.round(4).to_string(index=False))
    print("\nGenerated:")
    for f in files:
        print("-", f)


if __name__ == "__main__":
    main()

