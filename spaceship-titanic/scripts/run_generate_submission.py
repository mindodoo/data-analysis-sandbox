#!/usr/bin/env python3
"""Generate submission.csv using the saved Agent 6 tuned HGB pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from run_agent2_cleaning import run_agent2_cleaning
from run_agent5_validation import Agent5Cfg, build_pipeline
from run_agent6_optimization import Agent6Cfg, _infer_columns, _prepare_xy


def _prepare_test(test_clean: pd.DataFrame, target: str) -> pd.DataFrame:
    X = test_clean.drop(columns=[c for c in ["PassengerId", target] if c in test_clean.columns]).copy()
    for c in ["HomePlanet", "Destination", "Deck", "Side"]:
        if c in X.columns:
            X[c] = X[c].astype("string")
    return X


def generate_submission(
    train_clean: pd.DataFrame,
    test_clean: pd.DataFrame,
    root: Path | None = None,
    use_saved_pipeline: bool = True,
) -> Path:
    root = root or Path("/Users/mindodoo/Projects/data_ana/notebooks/spaceship-titanic")
    cfg = Agent6Cfg(root=root)
    out_path = root / "submission.csv"
    sample_path = root / "sample_submission.csv"

    if use_saved_pipeline and (root / "optimized_model" / "best_pipeline.joblib").exists():
        pipe = joblib.load(root / "optimized_model" / "best_pipeline.joblib")
    else:
        params = json.loads((root / "optimized_model" / "best_params.json").read_text(encoding="utf-8"))
        X_train, y_train, _ = _prepare_xy(train_clean, cfg)
        agent5_cfg = Agent5Cfg(root=root, random_state=cfg.random_state)
        cat_cols, num_cols = _infer_columns(X_train, cfg)
        pipe = build_pipeline(cat_cols=cat_cols, num_cols=num_cols, cfg=agent5_cfg)
        pipe.set_params(**{f"model__{k}": v for k, v in params.items()})
        pipe.fit(X_train, y_train)

    X_test = _prepare_test(test_clean, cfg.target)
    proba = pipe.predict_proba(X_test)[:, 1]
    transported = (proba >= 0.5)

    submission = pd.DataFrame(
        {
            "PassengerId": test_clean["PassengerId"].values,
            "Transported": transported,
        }
    )

    # Validation
    sample = pd.read_csv(sample_path)
    assert list(submission.columns) == list(sample.columns)
    assert len(submission) == len(test_clean) == len(sample)
    assert submission["PassengerId"].tolist() == test_clean["PassengerId"].tolist()
    assert submission["PassengerId"].tolist() == sample["PassengerId"].tolist()
    assert submission["Transported"].dtype == bool

    submission.to_csv(out_path, index=False)

    print(f"Wrote: {out_path}")
    print(f"Rows: {len(submission)}")
    print(f"Transported=True: {submission['Transported'].mean():.2%}")
    return out_path


def main() -> None:
    root = Path("/Users/mindodoo/Projects/data_ana/notebooks/spaceship-titanic")
    train_raw = pd.read_csv(root / "train.csv")
    test_raw = pd.read_csv(root / "test.csv")
    train_clean, test_clean, _ = run_agent2_cleaning(train_raw, test_raw)
    generate_submission(train_clean, test_clean, root=root)


if __name__ == "__main__":
    main()
