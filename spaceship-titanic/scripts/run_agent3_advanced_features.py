#!/usr/bin/env python3
"""
Agent 3 (advanced): extra feature engineering, feature importance refresh,
and PCA/reduction experiments.

Outputs:
- reports/feature_engineering_report_v2.md
- reports/feature_selection_report_v2.md
- reports/correlation_optimization_report_v2.md
- reports/feature_importance_report_v2.md
- reports/advanced_feature_comparison.md
- reports/engineered_features_v2.parquet
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler

from run_agent2_cleaning import run_agent2_cleaning
from run_agent3_feature_engineering import add_engineered_features


@dataclass(frozen=True)
class AdvCfg:
    root: Path
    target: str = "Transported"
    random_state: int = 42
    n_splits: int = 5


SPEND = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def build_v2_features(train_clean: pd.DataFrame, test_clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, test, _ = add_engineered_features(train_clean, test_clean)

    for df in (train, test):
        # Interaction features (multiplicative and ratios)
        for c1, c2 in combinations(SPEND, 2):
            df[f"{c1}_x_{c2}"] = df[c1] * df[c2]
            df[f"{c1}_div_{c2}"] = _safe_div(df[c1], df[c2]).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        df["Age_x_TotalSpend"] = df["Age"] * df["TotalSpend"]
        df["Age_div_TotalSpend"] = _safe_div(df["Age"], df["TotalSpend"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        df["CryoSleep_x_Age"] = df["CryoSleep"] * df["Age"]
        df["VIP_x_TotalSpend"] = df["VIP"] * df["TotalSpend"]

        # Polynomial terms
        df["Age_sq"] = df["Age"] ** 2
        df["TotalSpend_sq"] = df["TotalSpend"] ** 2
        df["CabinNum_sq"] = df["CabinNum"] ** 2

        # Power-like transforms
        df["Age_sqrt"] = np.sqrt(np.clip(df["Age"], 0, None))
        df["TotalSpend_sqrt"] = np.sqrt(np.clip(df["TotalSpend"], 0, None))

    # Fit transform numeric distribution transforms on train only
    numeric_for_transform = [
        "Age",
        "TotalSpend",
        "CabinNum",
        *SPEND,
        "Age_sq",
        "TotalSpend_sq",
        "CabinNum_sq",
        "Age_x_TotalSpend",
        "VIP_x_TotalSpend",
    ]
    numeric_for_transform = [c for c in numeric_for_transform if c in train.columns]

    pt = PowerTransformer(method="yeo-johnson", standardize=True)
    scaler = StandardScaler()

    train_pt = pt.fit_transform(train[numeric_for_transform])
    test_pt = pt.transform(test[numeric_for_transform])

    train_sc = scaler.fit_transform(train[numeric_for_transform])
    test_sc = scaler.transform(test[numeric_for_transform])

    for i, c in enumerate(numeric_for_transform):
        train[f"{c}_pt"] = train_pt[:, i]
        test[f"{c}_pt"] = test_pt[:, i]
        train[f"{c}_z"] = train_sc[:, i]
        test[f"{c}_z"] = test_sc[:, i]

    # PCA on transformed numeric features (train fit only)
    pca_source_cols = [f"{c}_pt" for c in numeric_for_transform]
    pca = PCA(n_components=min(12, len(pca_source_cols)), random_state=42)
    train_pca = pca.fit_transform(train[pca_source_cols])
    test_pca = pca.transform(test[pca_source_cols])
    for i in range(train_pca.shape[1]):
        col = f"PCA_{i+1}"
        train[col] = train_pca[:, i]
        test[col] = test_pca[:, i]

    engineered = pd.concat(
        [
            train.assign(split="train"),
            test.assign(split="test").assign(**{AdvCfg(root=Path('.')).target: np.nan}),
        ],
        ignore_index=True,
    )
    return train, test, engineered


def _group_cv_metrics(train_df: pd.DataFrame, use_pca_only_numeric: bool = False) -> dict:
    cfg = AdvCfg(root=Path("/Users/mindodoo/Projects/data_ana/notebooks/spaceship-titanic"))
    y = train_df[cfg.target].astype(int).to_numpy()
    groups = train_df["GroupId"].astype(str).to_numpy()

    X = train_df.drop(columns=[cfg.target, "PassengerId"], errors="ignore").copy()

    if use_pca_only_numeric:
        pca_cols = [c for c in X.columns if c.startswith("PCA_")]
        keep_cats = [c for c in ["HomePlanet", "Destination", "Deck", "Side", "DeckSide"] if c in X.columns]
        keep_core = [c for c in ["CryoSleep", "VIP", "GroupId"] if c in X.columns]
        X = X[keep_cats + keep_core + pca_cols].copy()

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

    gkf = GroupKFold(n_splits=cfg.n_splits)
    oof = np.zeros(len(X), dtype=int)
    oof_proba = np.zeros(len(X), dtype=float)
    for tr_idx, va_idx in gkf.split(X, y, groups):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr = y[tr_idx]
        pipe.fit(X_tr, y_tr)
        oof[va_idx] = pipe.predict(X_va)
        oof_proba[va_idx] = pipe.predict_proba(X_va)[:, 1]
    return {
        "accuracy": float(accuracy_score(y, oof)),
        "roc_auc": float(roc_auc_score(y, oof_proba)),
    }


def compute_importance(train_v2: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    cfg = AdvCfg(root=Path("/Users/mindodoo/Projects/data_ana/notebooks/spaceship-titanic"))
    X = train_v2.drop(columns=[cfg.target, "PassengerId"], errors="ignore").copy()
    y = train_v2[cfg.target].astype(int)
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

    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=cfg.random_state, stratify=y)
    pipe.fit(X_tr, y_tr)
    acc = float(accuracy_score(y_va, pipe.predict(X_va)))

    perm = permutation_importance(pipe, X_va, y_va, n_repeats=5, random_state=cfg.random_state, scoring="accuracy")
    imp = pd.DataFrame({"feature": X.columns, "importance_mean": perm.importances_mean, "importance_std": perm.importances_std})
    imp = imp.sort_values("importance_mean", ascending=False)
    return imp, acc


def write_reports(
    train_v2: pd.DataFrame,
    engineered: pd.DataFrame,
    imp: pd.DataFrame,
    holdout_acc: float,
    cv_base: dict,
    cv_v2: dict,
    cv_pca: dict,
) -> None:
    root = Path("/Users/mindodoo/Projects/data_ana/notebooks/spaceship-titanic")
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    (reports / "engineered_features_v2.parquet").write_bytes(b"")
    engineered.to_parquet(reports / "engineered_features_v2.parquet", index=False)

    comp = pd.DataFrame(
        [
            {
                "variant": "baseline engineered v1",
                "group_oof_accuracy": cv_base["accuracy"],
                "group_oof_roc_auc": cv_base["roc_auc"],
            },
            {
                "variant": "advanced v2 (all engineered)",
                "group_oof_accuracy": cv_v2["accuracy"],
                "group_oof_roc_auc": cv_v2["roc_auc"],
            },
            {
                "variant": "advanced v2 + PCA-focused subset",
                "group_oof_accuracy": cv_pca["accuracy"],
                "group_oof_roc_auc": cv_pca["roc_auc"],
            },
        ]
    )

    (reports / "advanced_feature_comparison.md").write_text(
        "# Advanced Feature Comparison (Agent 3 v2)\n\n"
        "```text\n" + comp.round(4).to_string(index=False) + "\n```\n",
        encoding="utf-8",
    )

    (reports / "feature_engineering_report_v2.md").write_text(
        "# Feature Engineering Report v2\n\n"
        "- Added multiplicative interactions for spend columns (pairwise products).\n"
        "- Added ratio features (safe-divide).\n"
        "- Added polynomial terms (`Age_sq`, `TotalSpend_sq`, `CabinNum_sq`).\n"
        "- Added transformed variants using PowerTransformer (`*_pt`) and z-score (`*_z`).\n"
        "- Added PCA components (`PCA_1..PCA_12`) from transformed numeric features.\n\n"
        f"Holdout accuracy (importance model): **{holdout_acc:.4f}**\n",
        encoding="utf-8",
    )

    (reports / "feature_selection_report_v2.md").write_text(
        "# Feature Selection Report v2\n\n"
        "Top 40 permutation importances:\n\n"
        "```text\n" + imp.head(40).round(4).to_string(index=False) + "\n```\n",
        encoding="utf-8",
    )

    corr_num = train_v2.select_dtypes(include=[np.number]).corr(method="spearman").abs()
    upper = corr_num.where(np.triu(np.ones(corr_num.shape), k=1).astype(bool))
    top_pairs = upper.stack().sort_values(ascending=False).head(25)
    top_pairs_df = top_pairs.reset_index().rename(columns={"level_0": "feat_a", "level_1": "feat_b", 0: "abs_spearman"})

    (reports / "correlation_optimization_report_v2.md").write_text(
        "# Correlation Optimization Report v2\n\n"
        "Top numeric correlations:\n\n"
        "```text\n" + top_pairs_df.round(4).to_string(index=False) + "\n```\n\n"
        "Recommendation:\n"
        "- For tree models: keep rich feature set if OOF improves.\n"
        "- For linear models: prefer PCA-focused subset or prune highly correlated engineered ratios.\n",
        encoding="utf-8",
    )

    (reports / "feature_importance_report_v2.md").write_text(
        "# Feature Importance Report v2\n\n"
        f"Holdout accuracy: **{holdout_acc:.4f}**\n\n"
        "```text\n" + imp.head(60).round(4).to_string(index=False) + "\n```\n",
        encoding="utf-8",
    )


def run_agent3_advanced(train_clean: pd.DataFrame, test_clean: pd.DataFrame) -> dict:
    # baseline v1
    train_v1, _, _ = add_engineered_features(train_clean, test_clean)
    cv_base = _group_cv_metrics(train_v1, use_pca_only_numeric=False)

    train_v2, test_v2, engineered_v2 = build_v2_features(train_clean, test_clean)
    cv_v2 = _group_cv_metrics(train_v2, use_pca_only_numeric=False)
    cv_pca = _group_cv_metrics(train_v2, use_pca_only_numeric=True)
    imp, holdout_acc = compute_importance(train_v2)

    write_reports(train_v2, engineered_v2, imp, holdout_acc, cv_base, cv_v2, cv_pca)

    return {
        "train_v2": train_v2,
        "test_v2": test_v2,
        "cv_base": cv_base,
        "cv_v2": cv_v2,
        "cv_pca": cv_pca,
        "importance": imp,
        "holdout_acc": holdout_acc,
        "parquet": Path("/Users/mindodoo/Projects/data_ana/notebooks/spaceship-titanic/reports/engineered_features_v2.parquet"),
    }


def main() -> None:
    root = Path("/Users/mindodoo/Projects/data_ana/notebooks/spaceship-titanic")
    train_raw = pd.read_csv(root / "train.csv")
    test_raw = pd.read_csv(root / "test.csv")
    train_clean, test_clean, _ = run_agent2_cleaning(train_raw, test_raw)
    res = run_agent3_advanced(train_clean, test_clean)
    print("Group OOF baseline v1:", round(res["cv_base"]["accuracy"], 4), "| AUC:", round(res["cv_base"]["roc_auc"], 4))
    print("Group OOF advanced v2:", round(res["cv_v2"]["accuracy"], 4), "| AUC:", round(res["cv_v2"]["roc_auc"], 4))
    print("Group OOF advanced + PCA:", round(res["cv_pca"]["accuracy"], 4), "| AUC:", round(res["cv_pca"]["roc_auc"], 4))
    print("Reports written under reports/*.md and engineered_features_v2.parquet")


if __name__ == "__main__":
    main()

