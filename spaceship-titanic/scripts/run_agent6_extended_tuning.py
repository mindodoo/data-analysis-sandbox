#!/usr/bin/env python3
"""
Agent 6 extended tuning — LightGBM & CatBoost with larger RandomizedSearchCV.

Does NOT replace submission model unless results beat current OOF and user approves.
Writes reports/extended_tuning_report.md and saves search artifacts under optimized_model/.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from run_agent2_cleaning import run_agent2_cleaning
from run_agent5_validation import Agent5Cfg, FoldSafeGroupAggregator, RowFeatureEngineer, _split_feature_types
from run_agent6_optimization import Agent6Cfg, _infer_columns, _prepare_xy
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold


@dataclass(frozen=True)
class ExtendedCfg:
    root: Path
    n_iter_lgbm: int = 40
    n_iter_catboost: int = 40
    n_splits: int = 5
    random_state: int = 42
    current_oof_baseline: float = 0.8108


def _build_extended_pipeline(cat_cols: list[str], num_cols: list[str], cfg: Agent5Cfg, model) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("row_features", RowFeatureEngineer()),
            ("group_agg", FoldSafeGroupAggregator(group_col=cfg.group_col)),
            ("pre", pre),
            ("model", model),
        ]
    )


def _oof_extended(
    train_clean: pd.DataFrame,
    model,
    best_params: dict,
    cfg: ExtendedCfg,
) -> dict:
    agent6_cfg = Agent6Cfg(root=cfg.root, n_splits=cfg.n_splits, random_state=cfg.random_state)
    agent5_cfg = Agent5Cfg(root=cfg.root, random_state=cfg.random_state, n_splits=cfg.n_splits)

    df = train_clean.copy()
    y = df[agent6_cfg.target].astype(int).to_numpy()
    groups = df[agent6_cfg.group_col].astype(str).to_numpy()
    X = df.drop(columns=[c for c in ["PassengerId", agent6_cfg.target] if c in df.columns]).copy()

    gkf = GroupKFold(n_splits=cfg.n_splits)
    oof_pred = np.zeros(len(X), dtype=int)
    oof_proba = np.zeros(len(X), dtype=float)

    for tr_idx, va_idx in gkf.split(X, y, groups=groups):
        X_tr, X_va = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
        y_tr = y[tr_idx]

        tmp = RowFeatureEngineer().fit_transform(X_tr)
        tmp = FoldSafeGroupAggregator(group_col=agent6_cfg.group_col).fit(tmp).transform(tmp)
        cat_cols, num_cols = _split_feature_types(tmp.drop(columns=[agent6_cfg.group_col], errors="ignore"))

        base = {"random_state": cfg.random_state}
        if model.__class__.__name__ == "LGBMClassifier":
            base.update({"n_jobs": -1, "verbose": -1})
        if model.__class__.__name__ == "CatBoostClassifier":
            base.update({"verbose": 0, "allow_writing_files": False})
        m = model.__class__(**{**base, **best_params})
        pipe = _build_extended_pipeline(cat_cols, num_cols, agent5_cfg, m)
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_va)[:, 1]
        oof_proba[va_idx] = proba
        oof_pred[va_idx] = (proba >= 0.5).astype(int)

    return {
        "accuracy": float(accuracy_score(y, oof_pred)),
        "f1": float(f1_score(y, oof_pred)),
        "roc_auc": float(roc_auc_score(y, oof_proba)),
    }


def _run_search(
    name: str,
    pipe: Pipeline,
    param_dist: dict,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    cfg: ExtendedCfg,
    n_iter: int,
) -> dict:
    gkf = GroupKFold(n_splits=cfg.n_splits)
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="accuracy",
        cv=gkf,
        random_state=cfg.random_state,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    search.fit(X, y, groups=groups)

    best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    top = (
        pd.DataFrame(search.cv_results_)
        .sort_values("rank_test_score")[["mean_test_score", "std_test_score", "params", "rank_test_score"]]
        .head(10)
    )
    return {
        "name": name,
        "search": search,
        "best_cv_accuracy": float(search.best_score_),
        "best_params": best_params,
        "top_trials": top,
    }


def run_extended_tuning(train_clean: pd.DataFrame, cfg: ExtendedCfg | None = None) -> dict:
    cfg = cfg or ExtendedCfg(root=Path("/Users/mindodoo/Projects/data_ana/notebooks/spaceship-titanic"))
    agent6_cfg = Agent6Cfg(root=cfg.root, random_state=cfg.random_state, n_splits=cfg.n_splits)
    agent5_cfg = Agent5Cfg(root=cfg.root, random_state=cfg.random_state, n_splits=cfg.n_splits)

    X, y, groups = _prepare_xy(train_clean, agent6_cfg)
    cat_cols, num_cols = _infer_columns(X, agent6_cfg)

    results = []

    # --- LightGBM ---
    try:
        from lightgbm import LGBMClassifier

        lgbm = LGBMClassifier(
            random_state=cfg.random_state,
            n_jobs=-1,
            verbose=-1,
        )
        pipe_lgbm = _build_extended_pipeline(cat_cols, num_cols, agent5_cfg, lgbm)
        lgbm_dist = {
            "model__n_estimators": randint(200, 900),
            "model__learning_rate": uniform(0.02, 0.13),
            "model__num_leaves": randint(16, 64),
            "model__min_child_samples": randint(5, 50),
            "model__subsample": uniform(0.65, 0.35),
            "model__colsample_bytree": uniform(0.65, 0.35),
            "model__reg_alpha": uniform(0.0, 2.0),
            "model__reg_lambda": uniform(0.0, 2.0),
        }
        res_lgbm = _run_search("LightGBM", pipe_lgbm, lgbm_dist, X, y, groups, cfg, cfg.n_iter_lgbm)
        oof_lgbm = _oof_extended(train_clean, lgbm, res_lgbm["best_params"], cfg)
        res_lgbm["oof"] = oof_lgbm
        joblib.dump(res_lgbm["search"].best_estimator_, cfg.root / "optimized_model" / "lgbm_best_pipeline.joblib")
        json.dumps(res_lgbm["best_params"], indent=2)
        (cfg.root / "optimized_model" / "lgbm_best_params.json").write_text(
            json.dumps(res_lgbm["best_params"], indent=2), encoding="utf-8"
        )
        results.append(res_lgbm)
    except (ImportError, OSError) as e:
        results.append({"name": "LightGBM", "error": str(e)})

    # --- CatBoost ---
    try:
        from catboost import CatBoostClassifier

        cat = CatBoostClassifier(
            random_state=cfg.random_state,
            verbose=0,
            allow_writing_files=False,
        )
        pipe_cat = _build_extended_pipeline(cat_cols, num_cols, agent5_cfg, cat)
        cat_dist = {
            "model__iterations": randint(300, 1200),
            "model__learning_rate": uniform(0.02, 0.13),
            "model__depth": randint(4, 10),
            "model__l2_leaf_reg": uniform(1.0, 9.0),
            "model__bagging_temperature": uniform(0.0, 1.0),
            "model__random_strength": uniform(0.0, 2.0),
        }
        res_cat = _run_search("CatBoost", pipe_cat, cat_dist, X, y, groups, cfg, cfg.n_iter_catboost)
        oof_cat = _oof_extended(train_clean, cat, res_cat["best_params"], cfg)
        res_cat["oof"] = oof_cat
        joblib.dump(res_cat["search"].best_estimator_, cfg.root / "optimized_model" / "catboost_best_pipeline.joblib")
        (cfg.root / "optimized_model" / "catboost_best_params.json").write_text(
            json.dumps(res_cat["best_params"], indent=2), encoding="utf-8"
        )
        results.append(res_cat)
    except ImportError as e:
        results.append({"name": "CatBoost", "error": str(e)})

    # --- Report ---
    rows = [
        {
            "model": "HistGradientBoosting (Agent 6 canonical)",
            "cv_mean_accuracy": cfg.current_oof_baseline,
            "oof_accuracy": cfg.current_oof_baseline,
            "notes": "Used for submission.csv",
        }
    ]
    for r in results:
        if "error" in r:
            rows.append({"model": r["name"], "cv_mean_accuracy": np.nan, "oof_accuracy": np.nan, "notes": f"SKIP: {r['error']}"})
        else:
            rows.append(
                {
                    "model": r["name"],
                    "cv_mean_accuracy": r["best_cv_accuracy"],
                    "oof_accuracy": r["oof"]["accuracy"],
                    "notes": "extended search",
                }
            )

    comp = pd.DataFrame(rows).sort_values("oof_accuracy", ascending=False, na_position="last")
    reports = cfg.root / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    sections = []
    for r in results:
        if "error" in r:
            sections.append(f"## {r['name']}\n\nSkipped: {r['error']}\n")
            continue
        sections.append(
            f"## {r['name']}\n\n"
            f"- CV mean accuracy: **{r['best_cv_accuracy']:.4f}**\n"
            f"- OOF accuracy (fold-safe re-eval): **{r['oof']['accuracy']:.4f}**\n"
            f"- Best params: `{json.dumps(r['best_params'])}`\n\n"
            f"Top trials:\n\n```text\n{r['top_trials'].to_string(index=False)}\n```\n"
        )

    (reports / "extended_tuning_report.md").write_text(
        "# Extended Tuning Report — Agent 6 (LightGBM / CatBoost)\n\n"
        f"Baseline (HGB submission model) OOF: **{cfg.current_oof_baseline:.4f}**\n\n"
        "## Comparison\n\n"
        "```text\n"
        + comp.round(4).to_string(index=False)
        + "\n```\n\n"
        + "\n".join(sections)
        + "\n## Note\n"
        "- `submission.csv` uses the **canonical HGB** model unless you approve switching.\n"
        "- Saved extended models: `optimized_model/lgbm_best_pipeline.joblib`, `optimized_model/catboost_best_pipeline.joblib`\n",
        encoding="utf-8",
    )

    print("\n=== Extended tuning summary ===")
    print(comp.to_string(index=False))
    print(f"\nWrote: {reports / 'extended_tuning_report.md'}")

    return {"comparison": comp, "results": results}


def main() -> None:
    root = Path("/Users/mindodoo/Projects/data_ana/notebooks/spaceship-titanic")
    train_raw = pd.read_csv(root / "train.csv")
    test_raw = pd.read_csv(root / "test.csv")
    train_clean, _, _ = run_agent2_cleaning(train_raw, test_raw)
    run_extended_tuning(train_clean)


if __name__ == "__main__":
    main()
