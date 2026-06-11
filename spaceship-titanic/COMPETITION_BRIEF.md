# Spaceship Titanic — AI Agent Competition Brief

**Competition:** [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic/overview)  
**Category:** Getting Started · Tabular · Binary classification  
**Workspace:** `data_ana/notebooks/spaceship-titanic/`  
**Last verified:** 2026-05-21 (against local CSVs and Kaggle API metadata)

---

## 1. Business narrative (problem context)

In the year **2912**, the passenger liner **Spaceship Titanic** collided with a spacetime anomaly. Roughly half of the ~13,000 passengers were **transported to another dimension**; the rest remained aboard or were otherwise not transported.

Recovered records describe passengers (home planet, cabin, spending, cryosleep, etc.). Your job is to build a model that predicts, for each passenger in the test set, whether they were **transported** (`True`) or **not** (`False`).

This is a pedagogical “Getting Started” competition: tabular data, binary target, modest size, and heavy emphasis on **feature engineering** and **missing-data handling** rather than deep learning scale.

---

## 2. Machine learning objective

| Item | Definition |
|------|------------|
| **Task** | Binary classification |
| **Target** | `Transported` — was the passenger transported to the alternate dimension? |
| **Evaluation metric** | **Categorization Accuracy** (Kaggle) = fraction of test rows where predicted `Transported` equals the hidden label |
| **Optimization direction** | Maximize accuracy |
| **Class balance (train)** | ~50.4% `True`, ~49.6% `False` — roughly balanced; accuracy is a reasonable proxy for model quality |

**Implication for agents:** No need for heavy class-weighting solely due to imbalance, but still report precision/recall/F1 for insight. Leaderboard scores for strong solutions are often in the **~0.80–0.83** accuracy range (public leaderboard varies over time).

---

## 3. Data inventory

### 3.1 Files

All paths are relative to `notebooks/spaceship-titanic/` unless noted.

| File | Rows | Columns | Notes |
|------|------|---------|-------|
| `train.csv` | 8,693 | 14 | Includes label `Transported` |
| `test.csv` | 4,277 | 13 | No label column |
| `sample_submission.csv` | 4,277 | 2 | `PassengerId`, `Transported` (placeholder values) |

### 3.2 Column dictionary

| Column | Type (raw CSV) | Description | Modeling notes |
|--------|----------------|-------------|----------------|
| `PassengerId` | string | Unique id `gggg_pp` — `gggg` = travel **group**, `pp` = index within group | Often **dropped** as a direct feature; **parse group id** for group-level aggregates. Groups are often family but not always. ~6,217 groups in train; mean ~1.4 passengers per group. |
| `HomePlanet` | string (nullable) | Planet of residence / departure | 3 planets: Earth, Europa, Mars. ~2.3% missing in train, ~2.0% in test. |
| `CryoSleep` | string (nullable) | In suspended animation for the voyage | Values `True` / `False` when present; read as boolean after imputation. ~2.5% missing train. Cryo passengers are cabin-confined; spending while in cryo is often zero. |
| `Cabin` | string (nullable) | `deck/num/side` — side `P` (Port) or `S` (Starboard) | High cardinality (~6.5k unique in train). Parse into `Deck`, `CabinNum`, `Side`. ~2.3% missing. |
| `Destination` | string (nullable) | Debarkation planet | 3 values: `TRAPPIST-1e`, `55 Cancri e`, `PSO J318.5-22`. ~2.1% missing train. |
| `Age` | float (nullable) | Age in years | ~2.1% missing train. |
| `VIP` | string (nullable) | Paid VIP service | `True` / `False`; rare positive class (~2.3% VIP in train). |
| `RoomService` | float (nullable) | Billed amount | Amenity spend; heavy right skew, many zeros. |
| `FoodCourt` | float (nullable) | Billed amount | Same pattern. |
| `ShoppingMall` | float (nullable) | Billed amount | Same pattern. |
| `Spa` | float (nullable) | Billed amount | Same pattern. |
| `VRDeck` | float (nullable) | Billed amount | Same pattern. |
| `Name` | string (nullable) | First and last name | Usually **dropped** or used only for derived features (e.g. surname = group proxy). |
| `Transported` | bool (train only) | **Target** | `True` = transported to other dimension |

### 3.3 Missing values (counts)

**Train** (~2.1–2.4% per feature column; **label has no nulls**):

- Highest missing counts: `ShoppingMall` (208), `CryoSleep` (217), others ~179–203.

**Test** (~1.9–2.5% per feature column).

Agents should document imputation strategy and validate that the same pipeline applies to train and test.

### 3.4 Raw dtype caveats

When loading with pandas:

- `Transported` loads as **boolean** in train.
- `CryoSleep` and `VIP` often load as **object** because of NaN; coerce to boolean after imputation.
- Amenity columns are **float64** with NaN for missing spend.

Always validate `dtypes` after `read_csv` and after cleaning.

---

## 4. Submission requirements

### 4.1 Format

- One row per `PassengerId` in **test.csv** (4,277 rows).
- Columns exactly: `PassengerId`, `Transported`.
- `Transported` must be boolean **`True` / `False`** (Kaggle accepts boolean strings in CSV; match `sample_submission.csv` style).

Example:

```csv
PassengerId,Transported
0013_01,False
0018_01,True
```

### 4.2 Local validation before upload

```python
import pandas as pd

sub = pd.read_csv("submission.csv")
test = pd.read_csv("test.csv")
assert list(sub.columns) == ["PassengerId", "Transported"]
assert len(sub) == len(test)
assert sub["PassengerId"].tolist() == test["PassengerId"].tolist()
assert sub["Transported"].dtype == bool or sub["Transported"].isin([True, False, "True", "False"]).all()
```

### 4.3 Kaggle submission

```bash
cd /Users/mindodoo/Projects/data_ana
.venv/bin/kaggle competitions submit -c spaceship-titanic -f notebooks/spaceship-titanic/submission.csv -m "message"
```

**Limits:** 10 submissions per day (per Kaggle API metadata). Merge deadline and competition deadline are listed on the competition page (long-running Getting Started competition).

---

## 5. Modeling guidance (non-binding but high-value)

These are common strong approaches; agents should justify choices in markdown logs.

### 5.1 Feature ideas

- **Group features:** Aggregate statistics per `PassengerId` group (`gggg`): mean age, any cryosleep, sum spend, mode home planet, group size.
- **Cabin parsing:** `Deck`, `Side`, optionally `CabinNum`.
- **Spend totals:** Sum of five amenity columns; “zero spend” indicator (especially under cryosleep).
- **Interactions:** `HomePlanet` × `Destination`, `CryoSleep` × total spend.
- **Name:** Extract surname if used; beware overlap with group id.

### 5.2 Algorithms

Tabular models work well: **Random Forest**, **Gradient Boosting** (XGBoost, LightGBM, CatBoost), **Logistic Regression** on encoded features. The included `spaceship-titanic-with-tfdf.ipynb` baseline uses TensorFlow Decision Forests.

### 5.3 Validation strategy

- **Stratified K-fold** on `Transported` (e.g. 5–10 folds).
- Optionally **GroupKFold** on parsed group id to avoid leakage across related passengers in the same fold.
- Report **out-of-fold accuracy** aligned with the competition metric.
- Do not tune on test.csv labels (unavailable).

### 5.4 Pitfalls

| Risk | Mitigation |
|------|------------|
| Leakage via group aggregates | Fit group stats **inside CV folds** on training folds only |
| Different missingness train vs test | Same imputation/encoding pipeline fit on train |
| Treating `PassengerId` as numeric | Use parsed group only |
| Probability threshold tuning | Default 0.5 is fine for accuracy with balanced classes |

---

## 6. Agent operating procedures

### 6.1 Required reading order

1. This file (`COMPETITION_BRIEF.md`)
2. `README.md` in this directory
3. Relevant stage agent from `data_ana/multi_agent_data_science_markdowns/` when following the multi-agent workflow

### 6.2 Deliverables by stage (suggested)

| Stage | Agent role | Outputs in this directory |
|-------|------------|---------------------------|
| 1 | Data analysis | `spaceship_titanic_agents.ipynb` §1, optional `reports/*.md` |
| 2 | Cleaning | `reports/02_cleaning.md`, saved cleaned parquet/csv if useful |
| 3 | Feature engineering | `reports/03_features.md`, feature list |
| 4 | Modeling | `reports/04_modeling.md`, trained model artifacts |
| 5 | Validation | `reports/05_validation.md`, CV metrics table |
| 6 | Retraining / optimization | `reports/06_optimization.md` |
| Final | Submission | `submission.csv`, `reports/07_submission.md` |

Create `reports/` as needed. Prefer **notebooks + short markdown summaries** so humans can audit agent work.

### 6.3 Notebook conventions

- Set working directory to this folder at the top of each notebook:

```python
from pathlib import Path
ROOT = Path("/Users/mindodoo/Projects/data_ana/notebooks/spaceship-titanic")
# or: ROOT = Path.cwd()  # when kernel cwd is already this folder
```

- Use `data_ana/.venv` for dependencies.
- Do not commit large artifacts or secrets; `kaggle.json` stays outside the repo (see `.gitignore`).

### 6.4 Reproducibility checklist

- [ ] Record random seeds
- [ ] Pin library versions in a cell or `requirements-spaceship.txt` if added
- [ ] Log CV scheme and fold scores
- [ ] Document every imputation and encoding rule
- [ ] Version-control notebooks; data remains local/gitignored

### 6.5 When uncertain

- State unknowns explicitly; do not invent column meanings or metric definitions.
- Re-read Kaggle overview/rules: https://www.kaggle.com/competitions/spaceship-titanic/overview
- Ask the user before submitting to Kaggle or overwriting a prior `submission.csv` they rely on.

---

## 7. Quick EDA snapshot (local train.csv)

Use for sanity checks; re-run EDA in notebooks for full distributions.

| Fact | Value |
|------|-------|
| Train rows | 8,693 |
| Test rows | 4,277 |
| Target rate `Transported=True` | 50.36% |
| `HomePlanet` counts | Earth 4602, Europa 2131, Mars 1759 (+201 null) |
| `Destination` counts | TRAPPIST-1e 5915, 55 Cancri e 1800, PSO J318.5-22 796 (+182 null) |
| Spend columns | Highly skewed; median 0 on all five amenities |

---

## 8. Success criteria (project-level)

| Level | Accuracy (CV / LB) | Notes |
|-------|-------------------|-------|
| Baseline | ~0.75+ | Simple tree model + minimal cleaning |
| Solid | ~0.79+ | Group + cabin features, careful imputation |
| Competitive | ~0.81+ | Tuned boosting + robust CV |

Exact leaderboard rank is not required for internal agent workflows unless the user specifies a target score.

---

## 9. References

- Competition overview: https://www.kaggle.com/competitions/spaceship-titanic/overview
- Data tab: https://www.kaggle.com/competitions/spaceship-titanic/data
- Rules: https://www.kaggle.com/competitions/spaceship-titanic/rules
- Baseline kernel (TF-DF): `spaceship-titanic-with-tfdf.ipynb` in this directory

---

## 10. Glossary

| Term | Meaning in this competition |
|------|----------------------------|
| Transported | Passenger sent to the “alternate dimension” (positive class) |
| CryoSleep | Suspended animation; affects cabin confinement and spending |
| Group | First four digits of `PassengerId` before `_` |
| Categorization Accuracy | Per-row classification accuracy on the hidden test labels |
