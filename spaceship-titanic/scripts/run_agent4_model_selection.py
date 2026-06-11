#!/usr/bin/env python3
"""
Agent 4 — Modeling Strategy & Training Agent (recommendation-only)

Loads engineered features from:
  reports/engineered_features.parquet

Runs a quick holdout evaluation (accuracy) on a small set of candidate models
to recommend a primary model for Agent 5/6.

Writes:
  reports/model_selection_report.md
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass(frozen=True)
class ModelCfg:
    root: Path
    parquet_relpath: str = "reports/engineered_features.parquet"
    reports_dir_relpath: str = "reports"
    target: str = "Transported"
    random_state: int = 42
    test_size: float = 0.2


def _load(engineered_parquet_path: Path) -> pd.DataFrame:
    if not engineered_parquet_path.exists():
        raise FileNotFoundError(f"Missing engineered features parquet: {engineered_parquet_path}")
    df = pd.read_parquet(engineered_parquet_path)
    if "split" not in df.columns:
        raise ValueError("Expected column `split` in engineered_features.parquet")
    return df


def _pick_feature_columns(df_train: pd.DataFrame, target: str) -> tuple[list[str], list[str], list[str]]:
    # Drop id-like columns; keep group-level aggregates but avoid OHE-ing raw GroupId.
    drop_cols = [c for c in ["PassengerId", "GroupId"] if c in df_train.columns]
    X_cols = [c for c in df_train.columns if c not in drop_cols + [target, "split"]]

    cat_cols = []
    num_cols = []
    for c in X_cols:
        dt = df_train[c].dtype
        # pandas string dtype or object => categorical
        if str(dt).startswith("string") or str(dt) == "object":
            cat_cols.append(c)
        else:
            num_cols.append(c)
    return X_cols, cat_cols, num_cols


def _build_logreg(cat_cols: list[str], num_cols: list[str], random_state: int) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    # saga handles sparse; keep max_iter high for stability.
    clf = LogisticRegression(
        solver="saga",
        max_iter=3000,
        n_jobs=-1,
        random_state=random_state,
    )
    return Pipeline([("pre", pre), ("model", clf)])


def _build_hgb(cat_cols: list[str], num_cols: list[str], random_state: int) -> Pipeline:
    # HistGradientBoosting requires dense numeric input => dense one-hot.
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )
    clf = HistGradientBoostingClassifier(
        random_state=random_state,
        learning_rate=0.08,
        max_depth=6,
        max_leaf_nodes=31,
    )
    return Pipeline([("pre", pre), ("model", clf)])


def evaluate_candidates(train_df: pd.DataFrame, target: str, cfg: ModelCfg) -> dict:
    df_train = train_df.copy()
    y = df_train[target].astype(int)

    X_cols, cat_cols, num_cols = _pick_feature_columns(df_train, target)
    X = df_train[X_cols]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )

    candidates = [
        ("LogisticRegression(OHE)", _build_logreg(cat_cols, num_cols, cfg.random_state)),
        ("HistGradientBoostingClassifier(OHE dense)", _build_hgb(cat_cols, num_cols, cfg.random_state)),
    ]

    rows = []
    models_out = {}

    for name, pipe in candidates:
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_val)
        acc = float(accuracy_score(y_val, pred))
        rows.append({"model": name, "holdout_accuracy": acc})
        models_out[name] = pipe

    res = {
        "candidates_df": pd.DataFrame(rows).sort_values("holdout_accuracy", ascending=False),
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "X_cols": X_cols,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
    }
    return res


def write_report(report_path: Path, eval_res: dict, primary_model: str, alternatives: list[str], meta: dict) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cand = eval_res["candidates_df"]

    if primary_model.startswith("HistGradientBoostingClassifier"):
        primary_bullets = [
            "Non-linear decision boundaries: HistGradientBoosting can capture non-linear feature interactions that a linear model may miss (especially with OHE-expanded categorical signals and engineered interactions).",
            "Robust on moderate-sized tabular data: with ~8.7k training rows, this model is usually a strong accuracy booster without requiring large-scale deep learning.",
            "Handles engineered numeric structure: features like `TotalSpend`, spend logs, and interaction flags often work well with gradient-boosted tree ensembles.",
            "Feature engineering created both numeric and categorical derived signals; tree boosting often leverages these effectively.",
        ]
    else:
        primary_bullets = [
            "Stable baseline: a regularized linear model with one-hot encoding performs well on tabular binary classification.",
            "Less prone to overfitting than heavier ensembles when validation is limited.",
            "Fast to train and easy to reproduce for later tuning.",
        ]

    report_path.write_text(
        "# Model Selection — Agent 4 (recommendation)\n\n"
        f"**Holdout split:** stratified `train/val` with test_size={meta['test_size']}, random_state={meta['random_state']}\n\n"
        "## Candidate comparison\n\n"
        "```text\n"
        + cand.to_string(index=False)
        + "\n```\n\n"
        "## Recommendation (for Agent 5/6)\n\n"
        f"**Primary model:** `{primary_model}`\n\n"
        "### Reasoning\n"
        + "\n".join([f"- {b}" for b in primary_bullets])
        + "\n\n"
        "- Note: this ranking is based on a quick stratified holdout, not group-safe CV. For final training, Agent 5 should revisit "
        "with `GroupKFold` using `GroupId` and ensure group aggregates are computed fold-safely.\n\n"
        "## Alternatives to revisit\n\n"
        + "\n".join([f"- `{m}`" for m in alternatives])
        + "\n\n"
        "### Why revisit?\n"
        "- The alternative may capture non-linearities that a linear model misses.\n"
        "- Final validation should use the correct split strategy (likely `GroupKFold` by `GroupId`) to avoid leakage when group-based engineered features are used.\n"
        "- If group-safe CV changes the ranking, Agent 5/6 should re-select.\n\n"
        "## Feature columns (model input)\n\n"
        f"- Categorical columns (OHE): {len(eval_res['cat_cols'])}\n"
        f"- Numeric columns: {len(eval_res['num_cols'])}\n"
        "- Dropped id-like columns: `PassengerId`, `GroupId`\n",
        encoding="utf-8",
    )


def main() -> None:
    cfg = ModelCfg(root=Path("/Users/mindodoo/Projects/data_ana/notebooks/spaceship-titanic"))
    reports_dir = cfg.root / cfg.reports_dir_relpath
    parquet_path = cfg.root / cfg.parquet_relpath

    df = _load(parquet_path)
    train_df = df[df["split"] == "train"].copy()

    eval_res = evaluate_candidates(train_df=train_df, target=cfg.target, cfg=cfg)
    cand_df = eval_res["candidates_df"]

    primary_model = str(cand_df.iloc[0]["model"])
    alternatives = [str(x) for x in cand_df["model"].tolist()[1:]]

    write_report(
        report_path=reports_dir / "model_selection_report.md",
        eval_res=eval_res,
        primary_model=primary_model,
        alternatives=alternatives,
        meta={"test_size": cfg.test_size, "random_state": cfg.random_state},
    )

    # Console output for notebook cell
    print("Candidate model comparison (holdout accuracy):")
    print(cand_df.to_string(index=False))
    print("\nPrimary recommendation:")
    print(primary_model)
    if alternatives:
        print("\nAlternatives to revisit (Agent 5/6):")
        for m in alternatives:
            print("-", m)
    print(f"\nWrote: {reports_dir / 'model_selection_report.md'}")


if __name__ == "__main__":
    main()

