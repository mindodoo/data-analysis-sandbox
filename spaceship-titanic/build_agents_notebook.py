#!/usr/bin/env python3
"""Build spaceship_titanic_agents.ipynb from template sections."""
import json
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "spaceship_titanic_agents.ipynb"

def md(s: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [s]}

def code(s: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "source": [s], "outputs": [], "execution_count": None}

cells = [
md("""# Spaceship Titanic — Multi-Agent Workflow Notebook

**Single notebook for all agents** on this competition. Run cells top-to-bottom within each section.

| Section | Agent | Status |
|---------|-------|--------|
| 0 | Environment & shared setup | Ready |
| 1 | Agent 1 — Data analysis (EDA) | Ready |
| 2 | Agent 2 — Data cleaning | Placeholder |
| 3 | Agent 3 — Feature engineering | Placeholder |
| 4 | Agent 4 — Modeling | Placeholder |
| 5 | Agent 5 — Validation | Placeholder |
| 6 | Agent 6 — Optimization | Placeholder |

**Kernel:** Select **Python (data_ana)** in Cursor/Jupyter (project venv at `data_ana/.venv`).

Context: [COMPETITION_BRIEF.md](./COMPETITION_BRIEF.md)
"""),

md("## 0. Environment & shared setup"),
code("""# Ensure plots render in the notebook and dependencies are available
import sys
from pathlib import Path

print("Python:", sys.executable)

# If your kernel is not data_ana/.venv, install deps into the active kernel (run once)
try:
    import matplotlib
    import seaborn
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib>=3.7", "seaborn>=0.13", "pandas>=2.0", "numpy>=1.24"])
    import matplotlib
    import seaborn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display

%matplotlib inline
sns.set_theme(style="whitegrid", palette="muted")

# --- paths (all agents use these) ---
ROOT = Path("/Users/mindodoo/Projects/data_ana/notebooks/spaceship-titanic")
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
REPORTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

SPEND = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
TARGET = "Transported"

assert (ROOT / "train.csv").exists(), f"Missing train.csv in {ROOT}"

train_raw = pd.read_csv(ROOT / "train.csv")
test_raw = pd.read_csv(ROOT / "test.csv")
sample_sub = pd.read_csv(ROOT / "sample_submission.csv")

print(f"ROOT: {ROOT}")
print(f"train_raw: {train_raw.shape} | test_raw: {test_raw.shape}")
display(train_raw.head())
"""),

md("""---
## 1. Agent 1 — Data Intake & Initial Analysis (EDA)

Exploratory analysis on **raw** data (no cleaning yet). Downstream agents use `train_raw` / `test_raw`.
"""),

md("### 1.1 Problem type"),
code("""display(Markdown('''
| Item | Value |
|------|-------|
| Task | **Binary classification** (supervised) |
| Target | `Transported` |
| Metric | Categorization Accuracy |
| Train labels | Yes |
| Test labels | No (hidden by Kaggle) |
'''))

print("Reasoning: one boolean outcome per passenger; tabular cross-sectional features.")
"""),

md("### 1.2 Schema & dataset overview"),
code("""overview = pd.DataFrame({
    "column": train_raw.columns,
    "dtype": train_raw.dtypes.astype(str).values,
    "non_null": train_raw.notnull().sum().values,
    "unique": [train_raw[c].nunique(dropna=True) for c in train_raw.columns],
    "missing_%": (train_raw.isnull().mean() * 100).round(2).values,
})
display(overview)

checks = pd.Series({
    "train rows": len(train_raw),
    "test rows": len(test_raw),
    "duplicate rows (train)": train_raw.duplicated().sum(),
    "duplicate PassengerId (train)": train_raw["PassengerId"].duplicated().sum(),
    "train/test id overlap": len(set(train_raw["PassengerId"]) & set(test_raw["PassengerId"])),
    "memory MB (train)": round(train_raw.memory_usage(deep=True).sum() / 1e6, 2),
})
display(checks.to_frame("value"))
"""),

md("### 1.3 Target distribution"),
code("""print(train_raw[TARGET].value_counts())
print(train_raw[TARGET].value_counts(normalize=True).round(4))

fig, ax = plt.subplots(figsize=(5, 4))
train_raw[TARGET].value_counts().plot(kind="bar", ax=ax, color=["#4c72b0", "#dd8452"])
ax.set_title("Target: Transported")
ax.set_ylabel("Count")
plt.tight_layout()
plt.show()
"""),

md("### 1.4 Missing values"),
code("""feature_cols = [c for c in train_raw.columns if c != TARGET]
miss_compare = pd.DataFrame({
    "train_%": train_raw[feature_cols].isnull().mean().sort_values(ascending=False) * 100,
    "test_%": test_raw.isnull().mean().sort_values(ascending=False) * 100,
}).round(2)
display(miss_compare)

rows = []
for c in feature_cols:
    if train_raw[c].isnull().any():
        m = train_raw[c].isnull()
        rows.append({
            "column": c,
            "p_transported_missing": train_raw.loc[m, TARGET].mean(),
            "p_transported_present": train_raw.loc[~m, TARGET].mean(),
            "delta": train_raw.loc[m, TARGET].mean() - train_raw.loc[~m, TARGET].mean(),
        })
display(pd.DataFrame(rows).sort_values("delta", key=abs, ascending=False).round(4))

fig, ax = plt.subplots(figsize=(8, 5))
(miss_compare["train_%"].sort_values()).plot(kind="barh", ax=ax, color="#55a868")
ax.set_xlabel("Missing (%)")
ax.set_title("Missing values — train")
plt.tight_layout()
plt.show()
"""),

md("### 1.5 Numeric distributions & outliers"),
code("""eda = train_raw.copy()
eda["GroupId"] = eda["PassengerId"].str.split("_").str[0]
eda["TotalSpend"] = eda[SPEND].sum(axis=1, min_count=1)

num_cols = ["Age"] + SPEND + ["TotalSpend"]
stats = eda[num_cols].agg(["mean", "median", "std", "min", "max"]).T
stats["skew"] = eda[num_cols].skew().round(3)
stats["kurtosis"] = eda[num_cols].kurtosis().round(3)
display(stats.round(2))

fig, axes = plt.subplots(2, 3, figsize=(12, 7))
for ax, col in zip(axes.flat, num_cols):
    sns.histplot(eda[col].dropna(), bins=40, ax=ax)
    ax.set_title(col)
fig.suptitle("Numeric distributions (train)", y=1.02)
plt.tight_layout()
plt.show()

def iqr_outlier_pct(s):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean() * 100

display(pd.Series({c: f"{iqr_outlier_pct(eda[c].dropna()):.1f}%" for c in ["Age", "TotalSpend"]}, name="IQR outlier %"))
"""),

md("### 1.6 Categorical features & transport rates"),
code("""for col in ["HomePlanet", "CryoSleep", "Destination", "VIP"]:
    print(f"\\n=== {col} ===")
    display(train_raw.groupby(col, dropna=False)[TARGET].mean().sort_values(ascending=False).round(3).to_frame("P(Transported)"))

parts = train_raw["Cabin"].dropna().astype(str).str.split("/", expand=True)
eda.loc[train_raw["Cabin"].notna(), "Deck"] = parts[0].values
eda.loc[train_raw["Cabin"].notna(), "Side"] = parts[2].values
print("\\n=== Cabin Side ===")
display(eda.groupby("Side", dropna=False)[TARGET].mean().round(3).to_frame("P(Transported)"))

fig, ax = plt.subplots(figsize=(6, 4))
train_raw.groupby("HomePlanet", dropna=False)[TARGET].mean().sort_values().plot(kind="bar", ax=ax, color="#8172b3")
ax.set_ylabel("P(Transported)")
ax.set_title("Transport rate by HomePlanet")
plt.tight_layout()
plt.show()
"""),

md("### 1.7 Group structure (PassengerId)"),
code("""eda["GroupSize"] = eda.groupby("GroupId")["PassengerId"].transform("count")
print("Unique groups:", eda["GroupId"].nunique())
display(eda["GroupSize"].value_counts().sort_index().head(10).to_frame("passengers"))
display(eda.groupby("GroupSize")[TARGET].mean().round(3).to_frame("P(Transported)"))
"""),

md("### 1.8 Correlations"),
code("""num = eda[num_cols].copy()
num[TARGET] = eda[TARGET].astype(int)

display(num.corr(method="spearman")[TARGET].drop(TARGET).sort_values(key=abs, ascending=False).round(3).to_frame("spearman"))

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(num.corr(method="spearman"), annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
ax.set_title("Spearman correlation")
plt.tight_layout()
plt.show()

print("Max spend pairwise corr:", eda[SPEND].corr().where(~np.eye(len(SPEND), dtype=bool)).stack().max().round(3))
"""),

md("### 1.9 Leakage & risk checks"),
code("""risk = pd.Series({
    "PassengerId unique (train)": train_raw["PassengerId"].is_unique,
    "PassengerId unique (test)": test_raw["PassengerId"].is_unique,
    "No train/test id overlap": len(set(train_raw["PassengerId"]) & set(test_raw["PassengerId"])) == 0,
    "Target has no nulls": train_raw[TARGET].notnull().all(),
})
display(risk.to_frame("ok"))

display(Markdown('''
**Agent 1 summary (handoff to Agent 2):**
- CryoSleep=True → much higher transport rate (~82% vs ~33% when False)
- HomePlanet / Destination strong univariate signals
- Spend negatively correlated with transport
- ~2% missing per column; missing spend may be MNAR
- Parse Cabin; use group features with fold-safe CV
'''))
"""),

md("### 1.10 Agent 1 — optional export markdown reports"),
code("""# Writes reports/ for audit trail; primary results are in cells above.
from datetime import date

def write_agent1_reports():
    miss_lines = []
    for c in feature_cols:
        if train_raw[c].isnull().any():
            m = train_raw[c].isnull()
            miss_lines.append(
                f"| {c} | {train_raw[c].isnull().mean()*100:.2f}% | "
                f"{train_raw.loc[m, TARGET].mean():.3f} | {train_raw.loc[~m, TARGET].mean():.3f} | "
                f"{train_raw.loc[m, TARGET].mean() - train_raw.loc[~m, TARGET].mean():+.3f} |"
            )
    spearman = num.corr(method="spearman")[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)

    (REPORTS / "initial_analysis.md").write_text(
        f"# Initial Analysis — Agent 1\\n\\nGenerated {date.today()} from `spaceship_titanic_agents.ipynb`.\\n\\n"
        f"Train {len(train_raw):,} rows; target balance {train_raw[TARGET].mean():.2%} True.\\n"
        f"See notebook section 1 for full tables and plots.\\n",
        encoding="utf-8",
    )
    (REPORTS / "correlation_report.md").write_text(
        "# Correlation Report\\n\\n" + spearman.round(3).to_string() + "\\n",
        encoding="utf-8",
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    train_raw[TARGET].value_counts().plot(kind="bar", ax=ax)
    fig.savefig(FIGURES / "target_distribution.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("Wrote:", REPORTS / "initial_analysis.md")

write_agent1_reports()
"""),

md("""---
## 2. Agent 2 — Data Cleaning & Transformation

*Placeholder — Agent 2 adds cleaned `train` / `test` DataFrames here.*
"""),
code("""# Agent 2: impute missing values, parse Cabin, coerce booleans
# train = ...
# test = ...
print("Agent 2 not implemented yet.")
"""),

md("""---
## 3. Agent 3 — Feature Engineering

*Placeholder.*
"""),
code("""print("Agent 3 not implemented yet.")
"""),

md("""---
## 4. Agent 4 — Modeling

*Placeholder.*
"""),
code("""print("Agent 4 not implemented yet.")
"""),

md("""---
## 5. Agent 5 — Validation & QA

*Placeholder.*
"""),
code("""print("Agent 5 not implemented yet.")
"""),

md("""---
## 6. Agent 6 — Retraining & Optimization

*Placeholder.*
"""),
code("""print("Agent 6 not implemented yet.")
"""),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python (data_ana)",
            "language": "python",
            "name": "data-ana",
        },
        "language_info": {
            "name": "python",
            "version": "3.9.0",
        },
    },
    "cells": cells,
}

OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("Wrote", OUT)
