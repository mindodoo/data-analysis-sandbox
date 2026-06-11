#!/usr/bin/env python3
"""
Agent 6 — Iterative Retraining & Optimization

Hyperparameter tuning for approved HistGradientBoostingClassifier pipeline
using GroupKFold (group-safe). Does NOT write submission.csv (on hold).

Writes:
  reports/retraining_report.md
  reports/iteration_history.md
  reports/model_comparison.md
  reports/final_recommendations.md
  optimized_model/best_pipeline.joblib
  optimized_model/best_params.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.model_selection import GroupKFold, RandomizedSearchCV

from run_agent5_validation import (
    Agent5Cfg,
    FoldSafeGroupAggregator,
    RowFeatureEngineer,
    _split_feature_types,
    build_pipeline,
    run_groupkfold_validation,
)


@dataclass(frozen=True)
class Agent6Cfg:
    root: Path
    target: str = "Transported"
    group_col: str = "GroupId"
    n_splits: int = 5
    random_state: int = 42
    n_iter: int = 24
    generate_submission: bool = False  # on hold per user request


def _prepare_xy(train_clean: pd.DataFrame, cfg: Agent6Cfg):
    df = train_clean.copy()
    y = df[cfg.target].astype(int).to_numpy()
    groups = df[cfg.group_col].astype(str).to_numpy()
    drop_cols = ["PassengerId", cfg.target]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()
    for c in ["HomePlanet", "Destination", "Deck", "Side"]:
        if c in X.columns:
            X[c] = X[c].astype("string")
    return X, y, groups


def _infer_columns(X: pd.DataFrame, cfg: Agent6Cfg) -> tuple[list[str], list[str]]:
    tmp = RowFeatureEngineer().fit_transform(X)
    tmp = FoldSafeGroupAggregator(group_col=cfg.group_col).fit(tmp).transform(tmp)
    tmp_model = tmp.drop(columns=[cfg.group_col], errors="ignore")
    return _split_feature_types(tmp_model)


def _baseline_params() -> dict:
    return {
        "learning_rate": 0.08,
        "max_depth": 6,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 0.0,
        "max_bins": 255,
    }


def run_hyperparameter_search(
    train_clean: pd.DataFrame,
    cfg: Agent6Cfg,
) -> dict:
    agent5_cfg = Agent5Cfg(root=cfg.root, n_splits=cfg.n_splits, random_state=cfg.random_state)
    X, y, groups = _prepare_xy(train_clean, cfg)
    cat_cols, num_cols = _infer_columns(X, cfg)

    pipe = build_pipeline(cat_cols=cat_cols, num_cols=num_cols, cfg=agent5_cfg)

    param_distributions = {
        "model__learning_rate": uniform(0.03, 0.12),
        "model__max_depth": randint(4, 11),
        "model__max_leaf_nodes": randint(15, 128),
        "model__min_samples_leaf": randint(5, 41),
        "model__l2_regularization": uniform(0.0, 5.0),
        "model__max_bins": [128, 255],
    }

    gkf = GroupKFold(n_splits=cfg.n_splits)
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=cfg.n_iter,
        scoring="accuracy",
        cv=gkf,
        random_state=cfg.random_state,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    search.fit(X, y, groups=groups)

    cv_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}

    return {
        "search": search,
        "best_params": best_params,
        "best_cv_accuracy": float(search.best_score_),
        "cv_results_top": cv_results[
            ["mean_test_score", "std_test_score", "params", "rank_test_score"]
        ].head(10),
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "agent5_cfg": agent5_cfg,
    }


def _oof_with_params(train_clean: pd.DataFrame, model_params: dict, cfg: Agent6Cfg) -> dict:
    """Re-run fold-safe OOF with explicit model hyperparameters."""
    agent5_cfg = Agent5Cfg(root=cfg.root, n_splits=cfg.n_splits, random_state=cfg.random_state)
    df = train_clean.copy()
    y = df[cfg.target].astype(int).to_numpy()
    groups = df[cfg.group_col].astype(str).to_numpy()
    drop_cols = ["PassengerId", cfg.target]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import GroupKFold

    gkf = GroupKFold(n_splits=cfg.n_splits)
    oof_pred = np.zeros(len(X), dtype=int)
    oof_proba = np.zeros(len(X), dtype=float)

    for tr_idx, va_idx in gkf.split(X, y, groups=groups):
        X_tr, X_va = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
        y_tr, y_va = y[tr_idx], y[va_idx]

        tmp = RowFeatureEngineer().fit_transform(X_tr)
        tmp = FoldSafeGroupAggregator(group_col=cfg.group_col).fit(tmp).transform(tmp)
        tmp_model = tmp.drop(columns=[cfg.group_col], errors="ignore")
        cat_cols, num_cols = _split_feature_types(tmp_model)

        pipe = build_pipeline(cat_cols=cat_cols, num_cols=num_cols, cfg=agent5_cfg)
        pipe.set_params(
            **{
                f"model__{k}": v
                for k, v in model_params.items()
                if k in model_params
            }
        )
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_va)[:, 1]
        oof_pred[va_idx] = (proba >= 0.5).astype(int)
        oof_proba[va_idx] = proba

    return {
        "accuracy": float(accuracy_score(y, oof_pred)),
        "f1": float(f1_score(y, oof_pred)),
        "precision": float(precision_score(y, oof_pred)),
        "recall": float(recall_score(y, oof_pred)),
        "roc_auc": float(roc_auc_score(y, oof_proba)),
    }


def write_agent6_reports(
    cfg: Agent6Cfg,
    baseline_oof: dict,
    tuned_oof: dict,
    search_result: dict,
) -> Path:
    reports = cfg.root / "reports"
    model_dir = cfg.root / "optimized_model"
    reports.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    best_params = search_result["best_params"]
    joblib.dump(search_result["search"].best_estimator_, model_dir / "best_pipeline.joblib")
    (model_dir / "best_params.json").write_text(json.dumps(best_params, indent=2), encoding="utf-8")

    delta_acc = tuned_oof["accuracy"] - baseline_oof["accuracy"]

    comparison = pd.DataFrame(
        [
            {"stage": "Agent 5 baseline (default HGB params)", **baseline_oof},
            {"stage": "Agent 6 tuned (best RandomizedSearchCV)", **tuned_oof},
        ]
    )

    (reports / "model_comparison.md").write_text(
        "# Model Comparison — Agent 6\n\n"
        "Metric: **group-safe OOF** (`GroupKFold` by `GroupId`).\n\n"
        "> Kaggle **test labels are hidden**, so true test accuracy cannot be computed locally. "
        "OOF accuracy is the best proxy we optimize.\n\n"
        "```text\n"
        + comparison.round(4).to_string(index=False)
        + "\n```\n\n"
        f"**Accuracy delta (tuned - baseline):** {delta_acc:+.4f}\n",
        encoding="utf-8",
    )

    (reports / "iteration_history.md").write_text(
        "# Iteration History — Agent 6\n\n"
        f"- RandomizedSearchCV iterations: {cfg.n_iter}\n"
        f"- Best CV mean accuracy (search): {search_result['best_cv_accuracy']:.4f}\n"
        "- Top 10 trials:\n\n"
        "```text\n"
        + search_result["cv_results_top"].to_string(index=False)
        + "\n```\n",
        encoding="utf-8",
    )

    (reports / "retraining_report.md").write_text(
        "# Retraining Report — Agent 6\n\n"
        "## Hyperparameter tuning\n"
        "Tuned `HistGradientBoostingClassifier` inside the fold-safe pipeline "
        "(row features + fold-safe group aggregates + OHE).\n\n"
        "## Best parameters\n"
        "```json\n"
        + json.dumps(best_params, indent=2)
        + "\n```\n\n"
        "## Submission\n"
        "- `submission.csv`: **ON HOLD** (not generated in this run)\n",
        encoding="utf-8",
    )

    (reports / "final_recommendations.md").write_text(
        "# Final Recommendations — Agent 6\n\n"
        f"- Use tuned parameters for final training when ready to submit.\n"
        f"- Best OOF accuracy after tuning: **{tuned_oof['accuracy']:.4f}** "
        f"(baseline was **{baseline_oof['accuracy']:.4f}**).\n"
        "- Before generating `submission.csv`, retrain `best_pipeline` on full `train_clean` "
        "with group stats fit on all training passengers.\n"
        "- Consider a threshold sweep on OOF probabilities if accuracy plateaus.\n",
        encoding="utf-8",
    )

    return model_dir


def run_agent6(train_clean: pd.DataFrame, cfg: Agent6Cfg | None = None) -> dict:
    cfg = cfg or Agent6Cfg(root=Path("/Users/mindodoo/Projects/data_ana/notebooks/spaceship-titanic"))

    print("Agent 6: baseline OOF (Agent 5 default params)...")
    baseline_res = run_groupkfold_validation(train_clean)
    baseline_oof = baseline_res["overall"]

    print(f"Agent 6: RandomizedSearchCV (n_iter={cfg.n_iter})...")
    search_result = run_hyperparameter_search(train_clean, cfg)

    print("Agent 6: OOF evaluation with tuned params...")
    tuned_oof = _oof_with_params(train_clean, search_result["best_params"], cfg)

    model_dir = write_agent6_reports(cfg, baseline_oof, tuned_oof, search_result)

    return {
        "baseline_oof": baseline_oof,
        "tuned_oof": tuned_oof,
        "best_params": search_result["best_params"],
        "best_cv_accuracy": search_result["best_cv_accuracy"],
        "model_dir": model_dir,
        "generate_submission": cfg.generate_submission,
        "cv_results_top": search_result["cv_results_top"],
    }


if __name__ == "__main__":
    root = Path("/Users/mindodoo/Projects/data_ana/notebooks/spaceship-titanic")
    from run_agent2_cleaning import run_agent2_cleaning

    train_raw = pd.read_csv(root / "train.csv")
    test_raw = pd.read_csv(root / "test.csv")
    train_clean, test_clean, _ = run_agent2_cleaning(train_raw, test_raw)
    out = run_agent6(train_clean)
    print("Done. Tuned OOF accuracy:", out["tuned_oof"]["accuracy"])
