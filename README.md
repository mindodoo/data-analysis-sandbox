# Multi-Agent Data Science — Tested on Spaceship Titanic

This repository is an experiment: **can a team of specialized AI agents run a full,
disciplined data science project end-to-end?** We test the
`multi_agent_data_science` workflow on the
[Kaggle Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic)
competition (binary classification, accuracy-scored).

## Executive Summary

**What worked:** the agents are effective at running the analysis step by step.
Each agent (data preparation → modeling → evaluation → improvement strategy)
produced transparent, evidence-cited work — EDA findings fed cleaning decisions,
cleaning fed feature engineering, and every handoff went through a user checkpoint
with plots, tables, and a joint recommendation. The workflow's QA discipline
(frozen group-safe CV splits, fixed seeds, fixed decision threshold, experiment
ledger) caught real issues: out-of-fold "improvements" that were actually
selection bias, and cohort models whose offline edge did not transfer to the
leaderboard.

**Result:** the agent-driven v2 pipeline reached **LB 0.80430**, beating the
previous best (0.80289) — the gain came from the data and feature layer
(deterministic CryoSleep-rule imputation, luxury/basic spend split) plus a model
family switch to CatBoost, exactly where the improvement-strategist agent's
diagnosis pointed.

**What still needs research:** the *iteration* loop. Single-pass analysis is
strong, but improvement iterations showed diminishing and sometimes illusory
gains — contextual features added nothing (+0.0001), and per-cohort hyperparameter
optimization looked like +0.0038 offline yet lost head-to-head on the leaderboard.
Open questions: how agents should size expected gains honestly (selection bias,
noise floors ≈ 1 SE), when to stop iterating, and how to find genuinely new signal
(e.g. better imputation of information-poor segments) rather than re-tuning the
same pipeline. Accuracy beyond ~0.805 LB likely requires ideas the current agent
loop does not yet generate on its own.

## Final Results

| Iteration | Approach | OOF (GroupKFold-5) | Kaggle LB |
|---|---|---:|---:|
| v1 | Tuned HistGradientBoosting + stacked ensembles | 0.8108 | 0.80289 |
| **v2.1** | **New cleaning + 10 features + tuned CatBoost (canonical)** | 0.8143 | **0.80430** |
| v2.2 | + contextual features | 0.8144 | — (stopped, noise) |
| v2.4 | Cohort-specific models by HomePlanet | 0.8181 | 0.80406 |

## The Agent Workflow

Defined in `multi_agent_data_science_markdowns/` (start at
`01_multi_agent_orchestrator.md`):

1. **Agent 1 — Data Preparation**: EDA → cleaning → feature engineering, in three
   phases with a user checkpoint after each.
2. **Agent 2 — Modeling Strategy & Training**: baselines, candidate models, tuning
   on a frozen split protocol.
3. **Agent 3 — Evaluation, QA & Cross-Iteration Review**: metrics, error analysis
   by root-cause layer, leakage/reproducibility QA, owns `experiment_ledger.md`.
4. **Agent 4 — Improvement Strategist**: diagnoses the weak layer
   (data / features / sampling / model / hyperparameters), writes an improvement
   brief, and routes work back — it never retrains anything itself.

Core rules: every agent echoes the context it inherited, cites evidence for every
decision, and no handoff happens without user review.

## Repository Layout

```text
multi_agent_data_science_markdowns/   the agent role definitions (the "workflow")
spaceship-titanic/
    data/            raw Kaggle train/test (never modified by agents)
    notebooks/       st_agents_v2.ipynb — the full v2 agent run, end to end
    reports/         agent reports, experiment ledger, figures, versioned artifacts
    models/          canonical model (model_v2: tuned CatBoost)
    submissions/     generated submission files + leaderboard tracking
scripts/             helper scripts
```

Key reading order for the v2 experiment: `spaceship-titanic/notebooks/st_agents_v2.ipynb`
(narrative + code), then `spaceship-titanic/reports/experiment_ledger.md` (what was
tried and what actually worked), then `reports/leaderboard_tracking.md` (LB ground
truth).
