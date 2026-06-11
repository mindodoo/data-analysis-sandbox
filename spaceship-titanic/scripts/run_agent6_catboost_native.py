#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from run_agent2_cleaning import run_agent2_cleaning
from run_agent5_validation import FoldSafeGroupAggregator, RowFeatureEngineer


@dataclass(frozen=True)
class CatBoostNativeCfg:
    root: Path
    target: str = "Transported"
    group_col: str = "GroupId"
    n_splits: int = 5
    random_state: int = 42
    n_iter: int = 15
    baseline_oof: float = 0.8108


CAT_FEATURES = [
    "HomePlanet",
    "Destination",
    "Deck",
    "Side",
    "DeckSide",
    "GroupId",
    "AgeBin",
    "CabinNumBin100",
]


def build_feature_frame(clean_df: pd.DataFrame, target: str | None = None) -> pd.DataFrame:
    drop = ["PassengerId"]
    if target and target in clean_df.columns:
        drop.append(target)
    base = clean_df.drop(columns=drop).copy()
    for c in ["HomePlanet", "Destination", "Deck", "Side", "GroupId"]:
        if c in base.columns:
            base[c] = base[c].astype(str)
    return RowFeatureEngineer().fit_transform(base)


def fold_transform(X_tr: pd.DataFrame, X_va: pd.DataFrame, group_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    agg = FoldSafeGroupAggregator(group_col=group_col)
    tr = agg.fit(X_tr).transform(X_tr)
    va = agg.transform(X_va)
    return tr, va


def _cat_indices(columns: list[str]) -> list[int]:
    return [i for i, c in enumerate(columns) if c in CAT_FEATURES]


def sample_params(rng: np.random.Generator) -> dict:
    return {
        "iterations": int(rng.integers(300, 700)),
        "learning_rate": float(rng.uniform(0.03, 0.10)),
        "depth": int(rng.integers(4, 8)),
        "l2_leaf_reg": float(rng.uniform(2.0, 8.0)),
        "bagging_temperature": float(rng.uniform(0.0, 1.0)),
        "random_strength": float(rng.uniform(0.0, 1.5)),
        "border_count": int(rng.choice([64, 128, 254])),
    }


def _make_model(params: dict, random_state: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Accuracy",
        random_seed=random_state,
        verbose=0,
        allow_writing_files=False,
        **params,
    )


def oof_evaluate(train_clean: pd.DataFrame, params: dict, cfg: CatBoostNativeCfg) -> dict:
    y = train_clean[cfg.target].astype(int).to_numpy()
    groups = train_clean[cfg.group_col].astype(str).to_numpy()
    X_base = build_feature_frame(train_clean, cfg.target)
    gkf = GroupKFold(n_splits=cfg.n_splits)
    oof_proba = np.zeros(len(X_base), dtype=float)

    for tr_idx, va_idx in gkf.split(X_base, y, groups=groups):
        X_tr = X_base.iloc[tr_idx].copy()
        X_va = X_base.iloc[va_idx].copy()
        y_tr = y[tr_idx]

        X_tr, X_va = fold_transform(X_tr, X_va, cfg.group_col)
        cols = list(X_tr.columns)
        cat_idx = _cat_indices(cols)

        model = _make_model(params, cfg.random_state)
        model.fit(
            Pool(X_tr, y_tr, cat_features=cat_idx),
            eval_set=Pool(X_va, y[va_idx], cat_features=cat_idx),
            use_best_model=True,
            early_stopping_rounds=50,
        )
        oof_proba[va_idx] = model.predict_proba(X_va)[:, 1]

    oof_pred = (oof_proba >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, oof_pred)),
        "f1": float(f1_score(y, oof_pred)),
        "roc_auc": float(roc_auc_score(y, oof_proba)),
        "oof_proba": oof_proba,
        "y": y,
    }


def randomized_search(train_clean: pd.DataFrame, cfg: CatBoostNativeCfg) -> dict:
    rng = np.random.default_rng(cfg.random_state)
    rows = []
    best_score = -1.0
    best_params = None

    print(f"CatBoost native search: {cfg.n_iter} trials", flush=True)
    for trial in range(1, cfg.n_iter + 1):
        params = sample_params(rng)
        res = oof_evaluate(train_clean, params, cfg)
        rows.append({"trial": trial, "oof_accuracy": res["accuracy"], **params})
        if res["accuracy"] > best_score:
            best_score = res["accuracy"]
            best_params = params.copy()
            print(f"  trial {trial}: new best OOF={best_score:.4f}", flush=True)
        else:
            print(f"  trial {trial}: OOF={res['accuracy']:.4f}", flush=True)

    history = pd.DataFrame(rows).sort_values("oof_accuracy", ascending=False)
    return {"best_params": best_params, "best_oof": best_score, "history": history}


def generate_submission_catboost(
    train_clean: pd.DataFrame, test_clean: pd.DataFrame, params: dict, cfg: CatBoostNativeCfg, threshold: float
) -> Path:
    y = train_clean[cfg.target].astype(int).to_numpy()
    X_train = build_feature_frame(train_clean, cfg.target)
    X_test = build_feature_frame(test_clean, None)

    agg = FoldSafeGroupAggregator(group_col=cfg.group_col)
    X_train = agg.fit(X_train).transform(X_train)
    X_test = agg.transform(X_test)
    cat_idx = _cat_indices(list(X_train.columns))

    model = _make_model(params, cfg.random_state)
    model.fit(Pool(X_train, y, cat_features=cat_idx))

    proba = model.predict_proba(X_test)[:, 1]
    out = cfg.root / "submission_catboost_native.csv"
    sub = pd.DataFrame({"PassengerId": test_clean["PassengerId"].values, "Transported": (proba >= threshold)})
    sub.to_csv(out, index=False)

    model_dir = cfg.root / "optimized_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_dir / "catboost_native_model.cbm"))
    (model_dir / "catboost_native_best_params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    return out


def run_catboost_native(train_clean: pd.DataFrame, test_clean: pd.DataFrame, cfg: CatBoostNativeCfg) -> dict:
    search = randomized_search(train_clean, cfg)
    final_oof = oof_evaluate(train_clean, search["best_params"], cfg)

    best_t, best_acc = 0.5, final_oof["accuracy"]
    for t in np.linspace(0.40, 0.60, 41):
        acc = accuracy_score(final_oof["y"], (final_oof["oof_proba"] >= t).astype(int))
        if acc > best_acc:
            best_t, best_acc = float(t), float(acc)
    final_oof["accuracy"] = best_acc

    submission_path = generate_submission_catboost(train_clean, test_clean, search["best_params"], cfg, best_t)

    report = cfg.root / "reports" / "catboost_native_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# CatBoost Native Categorical — Agent 6\n\n"
        f"Baseline HGB OOF: **{cfg.baseline_oof:.4f}**\n\n"
        f"Best CatBoost OOF: **{final_oof['accuracy']:.4f}**\n"
        f"F1: {final_oof['f1']:.4f}\n"
        f"ROC-AUC: {final_oof['roc_auc']:.4f}\n"
        f"Threshold: {best_t:.3f}\n\n"
        f"Submission file: `{submission_path.name}`\n\n"
        "Top trials:\n\n```text\n"
        + search["history"].head(10).round(4).to_string(index=False)
        + "\n```\n",
        encoding="utf-8",
    )

    print(f"Wrote: {submission_path}")
    print(f"Wrote: {report}")
    return {
        "best_oof": final_oof["accuracy"],
        "threshold": best_t,
        "best_params": search["best_params"],
        "submission_path": submission_path,
        "history": search["history"],
    }


def main() -> None:
    root = Path("/Users/mindodoo/Projects/data_ana/notebooks/spaceship-titanic")
    cfg = CatBoostNativeCfg(root=root)
    train_raw = pd.read_csv(root / "train.csv")
    test_raw = pd.read_csv(root / "test.csv")
    train_clean, test_clean, _ = run_agent2_cleaning(train_raw, test_raw)
    run_catboost_native(train_clean, test_clean, cfg)


if __name__ == "__main__":
    main()

