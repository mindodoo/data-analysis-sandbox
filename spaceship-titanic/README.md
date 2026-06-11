# Spaceship Titanic — Project Home

This directory is the **canonical workspace** for the [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) Kaggle competition.

## Start here

1. Read **[COMPETITION_BRIEF.md](./COMPETITION_BRIEF.md)**
2. Open **`spaceship_titanic_agents.ipynb`** — single notebook for all agents (EDA, cleaning, modeling, …)
3. Select kernel **Python (data_ana)** (project venv)

### Fix `ModuleNotFoundError: matplotlib`

The project venv at `data_ana/.venv` already includes matplotlib. If Jupyter uses another Python:

1. In Cursor: kernel picker → **Python (data_ana)**
2. Or run once from `data_ana/`:
   ```bash
   .venv/bin/python -m ipykernel install --user --name=data-ana --display-name="Python (data_ana)"
   .venv/bin/pip install -r requirements.txt
   ```
3. Section **0** of the notebook also auto-installs matplotlib/seaborn into the *active* kernel if missing.

## Main notebook

| Notebook | Purpose |
|----------|---------|
| **`spaceship_titanic_agents.ipynb`** | **All agents** — run sections 0→6; outputs appear in cells |
| `spaceship-titanic-with-tfdf.ipynb` | Legacy TF-DF baseline (Kaggle kernel) |

Rebuild notebook from template (optional): `python build_agents_notebook.py`

## Data files

| File | Rows (approx.) | Role |
|------|----------------|------|
| `train.csv` | 8,693 | Labeled training set |
| `test.csv` | 4,277 | Hold-out features |
| `sample_submission.csv` | 4,277 | Submission template |

```bash
cd /Users/mindodoo/Projects/data_ana
.venv/bin/kaggle competitions download -c spaceship-titanic -p notebooks/spaceship-titanic
unzip -o notebooks/spaceship-titanic/spaceship-titanic.zip -d notebooks/spaceship-titanic
```

## Agent sections in `spaceship_titanic_agents.ipynb`

| Section | Agent | Status |
|---------|-------|--------|
| 0 | Environment & paths | Ready |
| 1 | Data analysis (EDA) | Ready |
| 2 | Data cleaning | Placeholder |
| 3 | Feature engineering | Placeholder |
| 4 | Modeling | Placeholder |
| 5 | Validation | Placeholder |
| 6 | Optimization + extended LGBM/CatBoost search | Ready |
| 6.2 | First `submission.csv` (canonical HGB) | Generated |

Optional markdown exports: `reports/` (written by section 1.10).

## Related docs

- `COMPETITION_BRIEF.md` — competition context for agents
- `data_ana/multi_agent_data_science_markdowns/` — agent playbooks

## Paths

- **Competition dir:** `/Users/mindodoo/Projects/data_ana/notebooks/spaceship-titanic`
- **Python venv:** `/Users/mindodoo/Projects/data_ana/.venv`
