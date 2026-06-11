#!/usr/bin/env python3
"""
Agent 5 — Validation, QA & Performance Review

Runs group-safe CV for the approved primary model:
  HistGradientBoostingClassifier with OneHotEncoder (dense)

Key point: group-derived aggregates are computed *fold-safely* (fit on training fold only).

Writes to reports/:
  validation_report.md
  error_analysis_report.md
  qa_report.md
  robustness_report.md
  deployment_readiness.md
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass(frozen=True)
class Agent5Cfg:
    root: Path
    target: str = "Transported"
    group_col: str = "GroupId"
    n_splits: int = 5
    random_state: int = 42


SPEND = ("RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck")


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    b = np.where(b == 0, np.nan, b)
    out = a / b
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


class RowFeatureEngineer(BaseEstimator, TransformerMixin):
    """Row-level engineered features that don't require fitting."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        spend = list(SPEND)
        df["Spend_NonZeroCount"] = (df[spend] > 0).sum(axis=1).astype(int)
        df["Spend_Any"] = (df["TotalSpend"] > 0).astype(int)
        df["Spend_Zero"] = (df["TotalSpend"] == 0).astype(int)
        df["Spend_MeanNonZero"] = _safe_div(df["TotalSpend"].to_numpy(), df["Spend_NonZeroCount"].to_numpy())
        df["Spend_Log1pTotal"] = np.log1p(df["TotalSpend"].to_numpy())

        for c in spend:
            df[f"{c}_log1p"] = np.log1p(df[c].to_numpy())

        df["CabinNumBin100"] = np.where(df["CabinNum"] >= 0, (df["CabinNum"] // 100).astype(int), -1)
        df["CryoSleep_and_ZeroSpend"] = ((df["CryoSleep"] == 1) & (df["TotalSpend"] == 0)).astype(int)
        df["DeckSide"] = (df["Deck"].astype("string") + "_" + df["Side"].astype("string")).astype("string")
        df["AgeBin"] = pd.cut(df["Age"], bins=[-0.1, 12, 18, 25, 35, 50, 65, 200], labels=False).astype(int)

        return df


class FoldSafeGroupAggregator(BaseEstimator, TransformerMixin):
    """Compute group aggregates on fit data only, merge at transform time."""

    def __init__(self, group_col: str):
        self.group_col = group_col

    def fit(self, X: pd.DataFrame, y=None):
        g = X.groupby(self.group_col, dropna=False)
        self.group_stats_ = pd.DataFrame(
            {
                self.group_col: g.size().index,
                "GroupSize_fit": g.size().values,
                "GroupTotalSpend_fit": g["TotalSpend"].sum().values,
                "GroupMeanAge_fit": g["Age"].mean().values,
                "GroupAnyVIP_fit": g["VIP"].max().values,
                "GroupAnyCryo_fit": g["CryoSleep"].max().values,
            }
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df = df.merge(self.group_stats_, on=self.group_col, how="left")
        # Fill groups unseen in fit (should be rare in GroupKFold) with safe defaults
        df["GroupSize_fit"] = df["GroupSize_fit"].fillna(1).astype(float)
        df["GroupTotalSpend_fit"] = df["GroupTotalSpend_fit"].fillna(0.0)
        df["GroupMeanAge_fit"] = df["GroupMeanAge_fit"].fillna(df["Age"])
        df["GroupAnyVIP_fit"] = df["GroupAnyVIP_fit"].fillna(0.0)
        df["GroupAnyCryo_fit"] = df["GroupAnyCryo_fit"].fillna(0.0)
        df["GroupSpendPerPerson_fit"] = _safe_div(
            df["GroupTotalSpend_fit"].to_numpy(), df["GroupSize_fit"].to_numpy()
        )
        df["GroupIsSolo_fit"] = (df["GroupSize_fit"] == 1).astype(int)
        return df


def build_pipeline(cat_cols: list[str], num_cols: list[str], cfg: Agent5Cfg) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )
    model = HistGradientBoostingClassifier(
        random_state=cfg.random_state,
        learning_rate=0.08,
        max_depth=6,
        max_leaf_nodes=31,
    )
    return Pipeline(
        steps=[
            ("row_features", RowFeatureEngineer()),
            ("group_agg", FoldSafeGroupAggregator(group_col=cfg.group_col)),
            ("pre", pre),
            ("model", model),
        ]
    )


def _split_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    cat_cols = []
    num_cols = []
    for c in X.columns:
        dt = X[c].dtype
        if str(dt).startswith("string") or str(dt) == "object":
            cat_cols.append(c)
        else:
            num_cols.append(c)
    return cat_cols, num_cols


def run_groupkfold_validation(train_clean: pd.DataFrame) -> dict:
    cfg = Agent5Cfg(root=Path("/Users/mindodoo/Projects/data_ana/notebooks/spaceship-titanic"))

    df = train_clean.copy()
    y = df[cfg.target].astype(int).to_numpy()
    groups = df[cfg.group_col].astype(str).to_numpy()

    drop_cols = ["PassengerId", cfg.target]  # keep GroupId for group-safe aggregation but drop before preproc
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()

    # Ensure categoricals are string for OHE
    for c in ["HomePlanet", "Destination", "Deck", "Side", "DeckSide"]:
        if c in X.columns:
            X[c] = X[c].astype("string")

    # Pipeline adds new columns, so infer feature types after transformation isn't straightforward.
    # We'll define types on the pre-aggregation dataframe after RowFeatureEngineer+GroupAgg by sampling a small fit.
    gkf = GroupKFold(n_splits=cfg.n_splits)

    oof_pred = np.zeros(len(X), dtype=int)
    oof_proba = np.zeros(len(X), dtype=float)
    fold_rows = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        X_tr, X_va = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
        y_tr, y_va = y[tr_idx], y[va_idx]

        # Fit row+group transformations to define feature types for preprocess
        tmp = RowFeatureEngineer().fit_transform(X_tr)
        tmp = FoldSafeGroupAggregator(group_col=cfg.group_col).fit(tmp).transform(tmp)

        # Remove group id from model inputs (still needed for group agg step only)
        tmp_model = tmp.drop(columns=[cfg.group_col], errors="ignore")
        cat_cols, num_cols = _split_feature_types(tmp_model)

        pipe = build_pipeline(cat_cols=cat_cols, num_cols=num_cols, cfg=cfg)
        pipe.fit(X_tr, y_tr)

        proba = pipe.predict_proba(X_va)[:, 1]
        pred = (proba >= 0.5).astype(int)

        oof_pred[va_idx] = pred
        oof_proba[va_idx] = proba

        fold_rows.append(
            {
                "fold": fold,
                "n_train": int(len(tr_idx)),
                "n_val": int(len(va_idx)),
                "accuracy": float(accuracy_score(y_va, pred)),
                "f1": float(f1_score(y_va, pred)),
                "precision": float(precision_score(y_va, pred)),
                "recall": float(recall_score(y_va, pred)),
                "roc_auc": float(roc_auc_score(y_va, proba)),
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    overall = {
        "accuracy": float(accuracy_score(y, oof_pred)),
        "f1": float(f1_score(y, oof_pred)),
        "precision": float(precision_score(y, oof_pred)),
        "recall": float(recall_score(y, oof_pred)),
        "roc_auc": float(roc_auc_score(y, oof_proba)),
    }
    cm = confusion_matrix(y, oof_pred)

    return {
        "cfg": cfg,
        "fold_df": fold_df,
        "overall": overall,
        "confusion_matrix": cm,
        "oof_pred": oof_pred,
        "oof_proba": oof_proba,
    }


def write_reports(train_clean: pd.DataFrame, res: dict) -> None:
    cfg: Agent5Cfg = res["cfg"]
    reports = cfg.root / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    fold_df = res["fold_df"]
    overall = res["overall"]
    cm = res["confusion_matrix"]

    (reports / "validation_report.md").write_text(
        "# Validation Report — Agent 5\n\n"
        "Primary model: `HistGradientBoostingClassifier` (with OHE, fold-safe group aggregates)\n\n"
        f"CV: `GroupKFold(n_splits={cfg.n_splits})` by `{cfg.group_col}`\n\n"
        "## Fold metrics\n\n"
        "```text\n"
        + fold_df.round(4).to_string(index=False)
        + "\n```\n\n"
        "## Overall OOF metrics\n\n"
        "```text\n"
        + pd.Series(overall).round(4).to_string()
        + "\n```\n",
        encoding="utf-8",
    )

    (reports / "error_analysis_report.md").write_text(
        "# Error Analysis Report — Agent 5\n\n"
        "## Confusion matrix (OOF)\n\n"
        "```text\n"
        + np.array2string(cm)
        + "\n```\n\n"
        "## Segment checks to run next\n"
        "- Errors by `CryoSleep` (0/1)\n"
        "- Errors by `Deck` and `Side`\n"
        "- Errors by spend deciles (`TotalSpend`)\n",
        encoding="utf-8",
    )

    qa_checks = []
    qa_checks.append(("no_nulls_in_train_clean", bool(train_clean.isnull().sum().sum() == 0)))
    qa_checks.append(("passenger_id_unique", bool(train_clean["PassengerId"].is_unique)))
    qa_checks.append(("group_col_present", bool(cfg.group_col in train_clean.columns)))
    qa_checks.append(("target_present", bool(cfg.target in train_clean.columns)))

    (reports / "qa_report.md").write_text(
        "# QA Report — Agent 5\n\n"
        "## Checks\n\n"
        "```text\n"
        + pd.DataFrame(qa_checks, columns=["check", "ok"]).to_string(index=False)
        + "\n```\n\n"
        "## Leakage prevention\n"
        "- Group aggregates computed on *training fold only* via `FoldSafeGroupAggregator`.\n"
        "- CV split is `GroupKFold` by `GroupId`.\n",
        encoding="utf-8",
    )

    # Robustness: repeat CV with different seeds affecting the model only (splits fixed by groups)
    (reports / "robustness_report.md").write_text(
        "# Robustness Report — Agent 5\n\n"
        "Robustness in this competition primarily depends on split strategy (group-safe). "
        "The model is deterministic given the fixed data and random_state; additional robustness tests "
        "can include:\n"
        "- Vary model `random_state` and compare OOF metrics\n"
        "- Remove group aggregate features and compare\n"
        "- Compare performance on solo vs non-solo groups\n",
        encoding="utf-8",
    )

    (reports / "deployment_readiness.md").write_text(
        "# Deployment Readiness — Agent 5\n\n"
        "This is a Kaggle competition workflow (not production), but readiness checks are:\n\n"
        "- [x] Reproducible preprocessing pipeline (Agent 2 + Agent 3 + fold-safe group aggregates)\n"
        "- [x] Leakage-aware validation (`GroupKFold`)\n"
        "- [ ] Final model retrain on full train (Agent 6)\n"
        "- [ ] Submission generation and schema validation (Agent 6)\n",
        encoding="utf-8",
    )


def run_agent5(train_clean: pd.DataFrame) -> dict:
    res = run_groupkfold_validation(train_clean)
    write_reports(train_clean, res)
    return res


def main() -> None:
    cfg = Agent5Cfg(root=Path("/Users/mindodoo/Projects/data_ana/notebooks/spaceship-titanic"))
    train = pd.read_csv(cfg.root / "train.csv")
    raise SystemExit("Run from notebook with train_clean instead of raw train.csv.")


if __name__ == "__main__":
    main()

