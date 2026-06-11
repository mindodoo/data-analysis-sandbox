#!/usr/bin/env python3
"""
Agent 2 — Data Cleaning & Transformation

Runs inside the collaborative notebook (section 2) to:
- handle missing values
- parse Cabin into Deck/CabinNum/Side
- coerce CryoSleep/VIP into integer 0/1
- apply gentle outlier clipping on numeric columns (train-only bounds)
- export audit markdown reports and `reports/cleaned_dataset.parquet`
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CleaningConfig:
    root: Path
    spend_cols: tuple[str, ...]
    target: str = "Transported"
    clip_lower_q: float = 0.005
    clip_upper_q: float = 0.995


def to_bool_or_nan(x) -> bool | float:
    """Convert Kaggle boolean-like strings to bool, else NaN."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, str):
        v = x.strip().lower()
        if v == "true":
            return True
        if v == "false":
            return False
    return np.nan


def fill_mode(series: pd.Series):
    s = series.dropna()
    if s.empty:
        return np.nan
    return s.mode().iloc[0]


def run_agent2_cleaning(train_raw: pd.DataFrame, test_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = CleaningConfig(
        root=Path("/Users/mindodoo/Projects/data_ana/notebooks/spaceship-titanic"),
        spend_cols=("RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"),
    )
    reports_dir = cfg.root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    train = train_raw.copy()
    test = test_raw.copy()

    target = cfg.target
    SPEND = list(cfg.spend_cols)

    # -----------------------------
    # Missing indicators + imputations
    # -----------------------------
    missing_flags: list[str] = []

    # Boolean-like categoricals
    for col in ["CryoSleep", "VIP"]:
        train[f"{col}_is_missing"] = train[col].isnull()
        test[f"{col}_is_missing"] = test[col].isnull()
        missing_flags.append(f"{col}_is_missing")

        train[col] = train[col].map(to_bool_or_nan)
        test[col] = test[col].map(to_bool_or_nan)

        # Empirically, "present" values are strongly skewed to False; keep simple and fold-safe
        fill_value = False
        train[col] = train[col].fillna(fill_value).astype(bool).astype(int)
        test[col] = test[col].fillna(fill_value).astype(bool).astype(int)

    # Categorical planets
    for col in ["HomePlanet", "Destination"]:
        train[f"{col}_is_missing"] = train[col].isnull()
        test[f"{col}_is_missing"] = test[col].isnull()
        missing_flags.append(f"{col}_is_missing")

        mode_val = fill_mode(train[col])
        train[col] = train[col].fillna(mode_val).astype("string")
        test[col] = test[col].fillna(mode_val).astype("string")

    # Age
    age_missing = "Age_is_missing"
    train[age_missing] = train["Age"].isnull()
    test[age_missing] = test["Age"].isnull()
    missing_flags.append(age_missing)

    age_median = float(train["Age"].median(skipna=True))
    train["Age"] = train["Age"].fillna(age_median)
    test["Age"] = test["Age"].fillna(age_median)

    # Cabin parsing
    for df in (train, test):
        cabin = df["Cabin"].astype("string")
        parts = cabin.str.split("/", expand=True)
        df["Deck"] = parts[0]
        df["CabinNum"] = pd.to_numeric(parts[1], errors="coerce")
        df["Side"] = parts[2]

    train["Deck_is_missing"] = train["Deck"].isnull()
    test["Deck_is_missing"] = test["Deck"].isnull()
    train["Side_is_missing"] = train["Side"].isnull()
    test["Side_is_missing"] = test["Side"].isnull()
    train["CabinNum_is_missing"] = train["CabinNum"].isnull()
    test["CabinNum_is_missing"] = test["CabinNum"].isnull()

    missing_flags += ["Deck_is_missing", "Side_is_missing", "CabinNum_is_missing"]

    deck_mode = fill_mode(train["Deck"]) if train["Deck"].notnull().any() else "Unknown"
    side_mode = fill_mode(train["Side"]) if train["Side"].notnull().any() else "Unknown"

    train["Deck"] = train["Deck"].fillna(deck_mode).astype("string")
    test["Deck"] = test["Deck"].fillna(deck_mode).astype("string")
    train["Side"] = train["Side"].fillna(side_mode).astype("string")
    test["Side"] = test["Side"].fillna(side_mode).astype("string")

    # Missing cabin number sentinel
    train["CabinNum"] = train["CabinNum"].fillna(-1).astype(float)
    test["CabinNum"] = test["CabinNum"].fillna(-1).astype(float)

    # Spend columns: missing -> 0, but keep missing flags
    for col in SPEND:
        train[f"{col}_is_missing"] = train[col].isnull()
        test[f"{col}_is_missing"] = test[col].isnull()
        missing_flags.append(f"{col}_is_missing")

        train[col] = train[col].fillna(0.0).astype(float)
        test[col] = test[col].fillna(0.0).astype(float)

    # Derived numeric
    train["TotalSpend"] = train[SPEND].sum(axis=1)
    test["TotalSpend"] = test[SPEND].sum(axis=1)

    # GroupId (light transformation, no target usage)
    train["GroupId"] = train["PassengerId"].str.split("_").str[0].astype("string")
    test["GroupId"] = test["PassengerId"].str.split("_").str[0].astype("string")

    # -----------------------------
    # Gentle outlier clipping (train-only bounds)
    # -----------------------------
    numeric_cols = ["Age", "TotalSpend", *SPEND]
    clip_rows: list[dict[str, object]] = []

    for col in numeric_cols:
        lower = float(train[col].quantile(cfg.clip_lower_q))
        upper = float(train[col].quantile(cfg.clip_upper_q))

        before = train[col].copy()
        train[col] = train[col].clip(lower, upper)
        test[col] = test[col].clip(lower, upper)

        clipped_count = int(((before < lower) | (before > upper)).sum())
        clip_rows.append(
            {
                "col": col,
                "lower_q": cfg.clip_lower_q,
                "upper_q": cfg.clip_upper_q,
                "lower": round(lower, 4),
                "upper": round(upper, 4),
                "train_clipped_rows": clipped_count,
            }
        )

    outlier_report_df = pd.DataFrame(clip_rows).sort_values("train_clipped_rows", ascending=False)

    # -----------------------------
    # Build cleaned frames
    # -----------------------------
    keep_cols = [
        "PassengerId",
        "HomePlanet",
        "CryoSleep",
        "Deck",
        "CabinNum",
        "Side",
        "Destination",
        "Age",
        "VIP",
        *SPEND,
        "TotalSpend",
        "GroupId",
        *missing_flags,
    ]

    train_clean = train[keep_cols + [target]].copy()
    test_clean = test[keep_cols].copy()

    # Safety: eliminate any parsing NaNs that might remain
    train_clean = train_clean.fillna(
        {
            "Deck": "Unknown",
            "Side": "Unknown",
            "CabinNum": -1,
            "TotalSpend": 0,
        }
    )
    test_clean = test_clean.fillna(
        {
            "Deck": "Unknown",
            "Side": "Unknown",
            "CabinNum": -1,
            "TotalSpend": 0,
        }
    )

    # -----------------------------
    # Export
    # -----------------------------
    cleaned_dataset = pd.concat(
        [
            train_clean.assign(split="train"),
            test_clean.assign(split="test").assign(**{target: np.nan}),
        ],
        ignore_index=True,
    )

    parquet_path = reports_dir / "cleaned_dataset.parquet"
    cleaned_dataset.to_parquet(parquet_path, index=False)

    # Markdown reports
    nulls_train = train_clean.isnull().mean() * 100
    nulls_test = test_clean.isnull().mean() * 100
    missing_table = pd.DataFrame({"train_null_%": nulls_train.round(3), "test_null_%": nulls_test.round(3)})
    missing_table = missing_table.sort_values("train_null_%", ascending=False)

    (reports_dir / "cleaning_report.md").write_text(
        f"""# Cleaning Report — Agent 2

Generated from `spaceship_titanic_agents.ipynb` section 2.

## Missing value strategies
- `CryoSleep`, `VIP`: missing indicator + parse boolean-like strings + fill missing with `False`, then cast to int (0/1).
- `HomePlanet`, `Destination`: missing indicator + fill missing with train mode.
- `Age`: missing indicator + fill missing with train median.
- `Cabin`: parsed into `Deck`, `CabinNum`, `Side`.
  - `Deck`, `Side`: missing indicator + fill missing with train mode.
  - `CabinNum`: missing indicator + sentinel `-1`.
- Spend columns: missing indicator + fill missing with `0.0`.

## Resulting nulls (top columns)
```text
{missing_table.head(30).to_string()}
```
""",
        encoding="utf-8",
    )

    (reports_dir / "transformation_report.md").write_text(
        f"""# Transformation Report — Agent 2

Columns added:
- `Deck`, `CabinNum`, `Side` from raw `Cabin`
- `GroupId` from `PassengerId`
- `TotalSpend` from spend columns
- missing indicator flags: {', '.join(missing_flags)}

Columns dropped:
- raw `Cabin`
- `Name` (not needed after parsing)
""",
        encoding="utf-8",
    )

    (reports_dir / "outlier_report.md").write_text(
        f"""# Outlier Report — Agent 2

We apply quantile winsorization on:
- `Age`, `TotalSpend`, and all spend columns

Train-only bounds:
- lower_q={cfg.clip_lower_q}
- upper_q={cfg.clip_upper_q}

Top clipped rows (train):
```text
{outlier_report_df.head(25).to_string(index=False)}
```
""",
        encoding="utf-8",
    )

    (reports_dir / "encoding_report.md").write_text(
        """# Encoding Report — Agent 2

No one-hot encoding or target encoding is performed here.

Instead, Agent 2 focuses on:
- type coercion (boolean-like -> int 0/1)
- parsing cabin into structured columns
- adding missing indicator flags
- keeping categorical fields as strings for downstream agents
""",
        encoding="utf-8",
    )

    return train_clean, test_clean, cleaned_dataset


def main() -> None:
    root = Path("/Users/mindodoo/Projects/data_ana/notebooks/spaceship-titanic")
    train_raw = pd.read_csv(root / "train.csv")
    test_raw = pd.read_csv(root / "test.csv")
    run_agent2_cleaning(train_raw, test_raw)
    print("Agent 2 cleaning complete.")


if __name__ == "__main__":
    main()

