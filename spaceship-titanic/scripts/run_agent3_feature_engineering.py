#!/usr/bin/env python3
"""
Agent 3 — Feature Engineering & Correlation Optimization

Creates engineered features from Agent 2 cleaned data and computes feature importance.

Outputs (reports/):
- feature_engineering_report.md
- feature_selection_report.md
- correlation_optimization_report.md
- engineered_features.parquet
- feature_registry.md
- feature_importance_report.md
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass(frozen=True)
class Agent3Config:
    root: Path
    target: str = "Transported"
    random_state: int = 42
    test_size: float = 0.2


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def add_engineered_features(train_clean: pd.DataFrame, test_clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Feature engineering that does NOT use labels.

    Note: Some group-level aggregates can create leakage under naive CV if group members
    appear in both train/val folds. We keep them minimal and document this in reports.
    """
    cfg = Agent3Config(root=Path("/Users/mindodoo/Projects/data_ana/notebooks/spaceship-titanic"))
    spend = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

    train = train_clean.copy()
    test = test_clean.copy()

    # Combine to compute group-level aggregates in a consistent way for final training/inference.
    # (These do not use the target; but can affect CV design later.)
    combined = pd.concat(
        [
            train.drop(columns=[cfg.target]).assign(split="train"),
            test.assign(split="test"),
        ],
        ignore_index=True,
    )

    # Basic spend structure
    for df in (train, test):
        df["Spend_NonZeroCount"] = (df[spend] > 0).sum(axis=1).astype(int)
        df["Spend_Any"] = (df["TotalSpend"] > 0).astype(int)
        df["Spend_Zero"] = (df["TotalSpend"] == 0).astype(int)
        df["Spend_MeanNonZero"] = _safe_div(df["TotalSpend"], df["Spend_NonZeroCount"]).fillna(0.0)
        df["Spend_Log1pTotal"] = np.log1p(df["TotalSpend"])
        for c in spend:
            df[f"{c}_log1p"] = np.log1p(df[c])

        # Cabin binning: coarse location feature
        df["CabinNumBin100"] = np.where(df["CabinNum"] >= 0, (df["CabinNum"] // 100).astype(int), -1)

        # Interaction heuristics (domain-inspired)
        df["CryoSleep_and_ZeroSpend"] = ((df["CryoSleep"] == 1) & (df["TotalSpend"] == 0)).astype(int)
        df["VIP_and_HighSpend"] = ((df["VIP"] == 1) & (df["TotalSpend"] > df["TotalSpend"].quantile(0.9))).astype(int)

        # Composite categorical
        df["DeckSide"] = (df["Deck"].astype("string") + "_" + df["Side"].astype("string")).astype("string")

        # Age bins (coarse nonlinearity)
        df["AgeBin"] = pd.cut(df["Age"], bins=[-0.1, 12, 18, 25, 35, 50, 65, 200], labels=False).astype(int)

    # Group aggregates (computed on combined to apply to both splits)
    group = combined.groupby("GroupId", dropna=False)
    group_stats = pd.DataFrame(
        {
            "GroupSize_all": group.size(),
            "GroupTotalSpend_all": group["TotalSpend"].sum(),
            "GroupMeanAge_all": group["Age"].mean(),
            "GroupAnyVIP_all": group["VIP"].max(),
            "GroupAnyCryo_all": group["CryoSleep"].max(),
        }
    ).reset_index()

    def attach_group(df: pd.DataFrame) -> pd.DataFrame:
        out = df.merge(group_stats, on="GroupId", how="left")
        # Derived ratios
        out["GroupSpendPerPerson_all"] = _safe_div(out["GroupTotalSpend_all"], out["GroupSize_all"]).fillna(0.0)
        out["GroupIsSolo_all"] = (out["GroupSize_all"] == 1).astype(int)
        return out

    train = attach_group(train)
    test = attach_group(test)

    engineered = pd.concat(
        [
            train.assign(split="train"),
            test.assign(split="test").assign(**{cfg.target: np.nan}),
        ],
        ignore_index=True,
    )

    return train, test, engineered


def compute_feature_importance(train_fe: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Compute permutation importance on a holdout split.
    Uses a sklearn pipeline (OHE for categoricals + HistGradientBoostingClassifier).
    """
    cfg = Agent3Config(root=Path("/Users/mindodoo/Projects/data_ana/notebooks/spaceship-titanic"))
    target = cfg.target

    # Features: drop identifiers/labels
    drop_cols = ["PassengerId", target]
    X = train_fe.drop(columns=[c for c in drop_cols if c in train_fe.columns])
    y = train_fe[target].astype(int)

    cat_cols = [c for c in X.columns if str(X[c].dtype) in ("string", "object")]
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )

    model = HistGradientBoostingClassifier(random_state=cfg.random_state)
    pipe = Pipeline([("pre", pre), ("model", model)])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_val)
    acc = float(accuracy_score(y_val, pred))

    # Permutation importance on validation set
    perm = permutation_importance(pipe, X_val, y_val, n_repeats=10, random_state=cfg.random_state, scoring="accuracy")

    # Map importances back to input feature columns (not expanded OHE columns) using grouping:
    # permutation_importance returns per input column since we permute X_val columns before preprocessing.
    imp = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    meta = {
        "holdout_accuracy": acc,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "categorical_cols": cat_cols,
        "numeric_cols": num_cols,
        "model": "HistGradientBoostingClassifier + OneHotEncoder",
        "random_state": cfg.random_state,
    }

    return imp, meta


def write_agent3_reports(
    engineered: pd.DataFrame,
    importance: pd.DataFrame,
    meta: dict,
    feature_registry: list[tuple[str, str]],
) -> Path:
    cfg = Agent3Config(root=Path("/Users/mindodoo/Projects/data_ana/notebooks/spaceship-titanic"))
    reports = cfg.root / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    parquet_path = reports / "engineered_features.parquet"
    engineered.to_parquet(parquet_path, index=False)

    (reports / "feature_registry.md").write_text(
        "# Feature registry — Agent 3\n\n"
        + "\n".join([f"- **`{name}`**: {desc}" for name, desc in feature_registry])
        + "\n",
        encoding="utf-8",
    )

    (reports / "feature_engineering_report.md").write_text(
        f"""# Feature Engineering Report — Agent 3

Generated from `spaceship_titanic_agents.ipynb` section 3.

## Summary
- Engineered spend structure features (log1p, non-zero counts, mean non-zero spend)
- Parsed/derived cabin location features (binning + deck/side composition)
- Added age bin feature
- Added group-level aggregates (computed from the passenger manifest, no target used)

## Leakage / overfitting notes
- Group-level aggregates (e.g. `GroupSize_all`) can **inflate CV** if group members appear in both folds.
  Use `GroupKFold` by `GroupId` (Agent 5) if you keep these features.
""",
        encoding="utf-8",
    )

    (reports / "feature_selection_report.md").write_text(
        f"""# Feature Selection Report — Agent 3

This stage does not permanently remove columns yet. Instead, we rank features by **permutation importance**
on a holdout split (accuracy metric) to guide Agent 4 modeling.

Holdout accuracy (quick check): **{meta['holdout_accuracy']:.4f}**
Model: {meta['model']}
""",
        encoding="utf-8",
    )

    # Correlation analysis (numeric only)
    numeric_cols = meta["numeric_cols"]
    corr = engineered.loc[engineered["split"] == "train", numeric_cols].corr(method="spearman").abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    top_pairs = (
        upper.stack()
        .sort_values(ascending=False)
        .head(20)
        .rename("abs_spearman")
        .reset_index()
        .rename(columns={"level_0": "feat_a", "level_1": "feat_b"})
    )

    (reports / "correlation_optimization_report.md").write_text(
        "# Correlation Optimization Report — Agent 3\n\n"
        "Top absolute Spearman correlations among numeric features (train):\n\n"
        "```text\n"
        + top_pairs.to_string(index=False)
        + "\n```\n\n"
        "Recommendation: keep both raw spend columns and `TotalSpend` for tree models; for linear models, "
        "prefer `TotalSpend` + `Spend_NonZeroCount` + selected logs to reduce redundancy.\n",
        encoding="utf-8",
    )

    (reports / "feature_importance_report.md").write_text(
        "# Feature Importance Report — Agent 3\n\n"
        f"Holdout accuracy (quick check): {meta['holdout_accuracy']:.4f}\n\n"
        "Top permutation importances (validation):\n\n"
        "```text\n"
        + importance.head(40).to_string(index=False)
        + "\n```\n",
        encoding="utf-8",
    )

    return parquet_path


def run_agent3(train_clean: pd.DataFrame, test_clean: pd.DataFrame) -> dict:
    train_fe, test_fe, engineered = add_engineered_features(train_clean, test_clean)
    importance, meta = compute_feature_importance(train_fe)

    feature_registry = [
        ("Spend_NonZeroCount", "Count of spend columns > 0"),
        ("Spend_Any", "Indicator: TotalSpend > 0"),
        ("Spend_Zero", "Indicator: TotalSpend == 0"),
        ("Spend_MeanNonZero", "TotalSpend / Spend_NonZeroCount (0 if none)"),
        ("Spend_Log1pTotal", "log1p(TotalSpend)"),
        ("CabinNumBin100", "CabinNum bucketed by floor(CabinNum/100); -1 if missing"),
        ("CryoSleep_and_ZeroSpend", "Interaction: CryoSleep==1 and TotalSpend==0"),
        ("VIP_and_HighSpend", "Interaction: VIP==1 and TotalSpend above 90th percentile (within split)"),
        ("DeckSide", "Composite categorical: Deck + '_' + Side"),
        ("AgeBin", "Coarse age bin (child/teen/young adult/...)"),
        ("GroupSize_all", "Group size computed on combined passenger manifest (train+test)"),
        ("GroupTotalSpend_all", "Sum TotalSpend per GroupId on manifest"),
        ("GroupMeanAge_all", "Mean Age per GroupId on manifest"),
        ("GroupAnyVIP_all", "Max VIP per GroupId on manifest"),
        ("GroupAnyCryo_all", "Max CryoSleep per GroupId on manifest"),
        ("GroupSpendPerPerson_all", "GroupTotalSpend_all / GroupSize_all"),
        ("GroupIsSolo_all", "Indicator: GroupSize_all == 1"),
    ] + [(f"{c}_log1p", f"log1p({c})") for c in ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]]

    parquet_path = write_agent3_reports(engineered, importance, meta, feature_registry)

    return {
        "train_fe": train_fe,
        "test_fe": test_fe,
        "engineered": engineered,
        "importance": importance,
        "meta": meta,
        "parquet_path": parquet_path,
    }

