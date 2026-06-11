#!/usr/bin/env python3
"""
Agent 3.1 targeted feature pass:
- Small, curated interactions
- No large ratio explosion
- No PCA
- Compare against v1 baseline using group-safe CV (accuracy + ROC-AUC)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from run_agent2_cleaning import run_agent2_cleaning
from run_agent3_feature_engineering import add_engineered_features


ROOT = Path("/Users/mindodoo/Projects/data_ana/notebooks/spaceship-titanic")
TARGET = "Transported"
GROUP = "GroupId"
SEED = 42


def add_targeted_features(train_v1: pd.DataFrame, test_v1: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train_v1.copy()
    test = test_v1.copy()

    for df in (train, test):
        # Spend interactions (curated only)
        df["Spa_x_VRDeck"] = df["Spa"] * df["VRDeck"]
        df["FoodCourt_x_ShoppingMall"] = df["FoodCourt"] * df["ShoppingMall"]
        df["RoomService_x_TotalSpend"] = df["RoomService"] * df["TotalSpend"]

        # Carefully chosen nonlinear transforms
        df["Age_sq_small"] = (df["Age"] ** 2) / 100.0
        df["TotalSpend_log1p_sq"] = np.log1p(df["TotalSpend"]) ** 2

        # Domain interactions
        df["CryoSleep_x_DeckSide"] = (
            df["CryoSleep"].astype(str) + "_" + df["DeckSide"].astype(str)
        ).astype("string")
        df["HomePlanet_x_Destination"] = (
            df["HomePlanet"].astype(str) + "_" + df["Destination"].astype(str)
        ).astype("string")

        # Compact spend concentration feature
        spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
        max_spend = df[spend_cols].max(axis=1)
        sum_spend = df[spend_cols].sum(axis=1).replace(0, np.nan)
        df["Spend_Concentration"] = (max_spend / sum_spend).fillna(0.0)

    return train, test


def group_cv_metrics(train_df: pd.DataFrame) -> dict:
    y = train_df[TARGET].astype(int).to_numpy()
    groups = train_df[GROUP].astype(str).to_numpy()
    X = train_df.drop(columns=[TARGET, "PassengerId"], errors="ignore").copy()

    cat_cols = [c for c in X.columns if str(X[c].dtype) in ("object", "string")]
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )
    pipe = Pipeline([("pre", pre), ("model", HistGradientBoostingClassifier(random_state=SEED))])

    gkf = GroupKFold(n_splits=5)
    oof_pred = np.zeros(len(X), dtype=int)
    oof_proba = np.zeros(len(X), dtype=float)

    for tr_idx, va_idx in gkf.split(X, y, groups):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr = y[tr_idx]
        pipe.fit(X_tr, y_tr)
        oof_pred[va_idx] = pipe.predict(X_va)
        oof_proba[va_idx] = pipe.predict_proba(X_va)[:, 1]

    return {
        "accuracy": float(accuracy_score(y, oof_pred)),
        "roc_auc": float(roc_auc_score(y, oof_proba)),
    }


def main() -> None:
    train_raw = pd.read_csv(ROOT / "train.csv")
    test_raw = pd.read_csv(ROOT / "test.csv")
    train_clean, test_clean, _ = run_agent2_cleaning(train_raw, test_raw)

    train_v1, test_v1, _ = add_engineered_features(train_clean, test_clean)
    base = group_cv_metrics(train_v1)

    train_t, test_t = add_targeted_features(train_v1, test_v1)
    targeted = group_cv_metrics(train_t)

    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    comp = pd.DataFrame(
        [
            {"variant": "baseline v1", "group_oof_accuracy": base["accuracy"], "group_oof_roc_auc": base["roc_auc"]},
            {"variant": "targeted v3.1", "group_oof_accuracy": targeted["accuracy"], "group_oof_roc_auc": targeted["roc_auc"]},
        ]
    )

    (reports / "targeted_feature_comparison.md").write_text(
        "# Targeted Feature Comparison (Agent 3.1)\n\n"
        "```text\n" + comp.round(4).to_string(index=False) + "\n```\n\n"
        f"Accuracy delta: {targeted['accuracy'] - base['accuracy']:+.4f}\n\n"
        f"ROC-AUC delta: {targeted['roc_auc'] - base['roc_auc']:+.4f}\n",
        encoding="utf-8",
    )

    train_t.assign(split="train").to_parquet(reports / "engineered_features_targeted.parquet", index=False)

    print(comp.round(4).to_string(index=False))
    print("Wrote:", reports / "targeted_feature_comparison.md")


if __name__ == "__main__":
    main()

