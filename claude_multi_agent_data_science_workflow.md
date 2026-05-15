# Multi-Agent Data Science Workflow for Claude

## Purpose

This document defines a structured multi-agent workflow for solving data science problems using Claude or similar AI systems. The workflow is designed so that each agent performs a specialized role while documenting every action, decision, assumption, and reasoning in markdown format.

The objective is to:

- Break complex data science work into modular sessions
- Preserve continuity between agents
- Create transparent and reproducible analysis
- Allow iterative improvement of models
- Enable human review and intervention at every stage
- Support both ML and non-ML problem-solving approaches

Every agent must:

1. Read all previous markdown documents before starting
2. Continue work from the latest state
3. Document all findings and reasoning
4. Save outputs in structured markdown
5. Clearly explain WHY decisions were made
6. Avoid hidden assumptions
7. Provide reproducible methodology

---

# Global Workflow Overview

```text
Agent 1 → Data Intake & Initial Analysis
Agent 2 → Data Cleaning & Transformation
Agent 3 → Feature Engineering & Correlation Review
Agent 4 → Modeling Strategy & Training
Agent 5 → Validation, QA & Performance Review
Agent 6 → Iterative Retraining & Optimization
```

Each stage produces markdown artifacts for the next agent.

---

# Global Rules for All Agents

## Core Principles

Every agent must:

- Explain all reasoning
- Avoid making silent assumptions
- Prefer interpretable decisions first
- Log failed experiments
- Log uncertainty and risks
- Use reproducible transformations
- Track dataset versions
- Track feature versions
- Track model versions
- Explain tradeoffs

---

# Mandatory Documentation Structure

Every agent must generate markdown using this structure:

```markdown
# Agent Name

## Objective

## Inputs Received

## Dataset Summary

## Actions Performed

## Reasoning Behind Decisions

## Methods Used

## Outputs Generated

## Risks / Concerns

## Recommendations for Next Agent

## Files Generated
```

---

# Shared Project State

All agents should maintain:

```text
/project
    /raw_data
    /cleaned_data
    /features
    /models
    /validation
    /reports
    /experiments
    /logs
```

---

# Agent 1 — Data Intake & Initial Analysis

## Role

This agent is responsible for understanding the dataset before any modification occurs.

This stage focuses on:

- Dataset structure
- Data quality
- Statistical understanding
- Initial risks
- Problem type detection
- Target label understanding
- Distribution analysis
- Missing value analysis
- Leakage detection
- Class imbalance review
- Business context understanding

---

## Objectives

1. Understand the problem
2. Understand the target variable
3. Understand the data schema
4. Detect obvious issues early
5. Create baseline observations
6. Prepare guidance for cleaning

---

## Tasks

### 1. Identify Problem Type

The agent must determine whether the problem is:

- Classification
- Regression
- Time series forecasting
- Ranking
- Recommendation
- Clustering
- Survival analysis
- Anomaly detection
- Causal inference
- Optimization
- Statistical inference

### Required Reasoning

The agent must explain:

- Why the problem belongs to that category
- Which columns support that conclusion
- Whether labels exist
- Whether the task is supervised or unsupervised

---

### 2. Dataset Overview

The agent should inspect:

- Number of rows
- Number of columns
- Column names
- Data types
- Memory usage
- Duplicate rows
- Constant columns
- Near-constant columns
- Unique value counts

---

### 3. Statistical Analysis

Perform:

- Descriptive statistics
- Mean
- Median
- Mode
- Variance
- Standard deviation
- Quantiles
- Skewness
- Kurtosis

For categorical data:

- Frequency distribution
- Cardinality
- Dominant categories

---

### 4. Missing Value Analysis

Analyze:

- Null percentage
- Missing patterns
- Whether missingness is random
- Correlation of missingness with target

---

### 5. Distribution Analysis

Analyze:

- Histograms
- Boxplots
- Density plots
- Class distribution
- Target distribution

---

### 6. Correlation Analysis

Analyze:

- Pearson correlation
- Spearman correlation
- Categorical associations
- Multicollinearity
- Target relationships

---

# Agent 2 — Data Cleaning & Transformation

## Role

This agent transforms raw data into modeling-ready data.

This stage focuses on:

- Missing value handling
- Outlier treatment
- Encoding
- Scaling
- Data normalization
- Winsorization
- Leakage prevention
- Invalid value handling
- Consistent formatting

---

## Tasks

### Missing Value Handling

Possible approaches:

- Mean imputation
- Median imputation
- Mode imputation
- KNN imputation
- Model-based imputation
- Row removal
- Column removal
- Missing indicators

### Required Reasoning

Explain:

- Why a method was selected
- Why alternatives were rejected
- Risk of bias introduction
- Risk of information loss

---

### Outlier Handling

Possible methods:

- IQR filtering
- Z-score
- Isolation Forest
- Quantile clipping
- Winsorization

---

### Encoding

Possible methods:

- One-hot encoding
- Label encoding
- Target encoding
- Frequency encoding
- Binary encoding

---

### Scaling & Normalization

Possible methods:

- StandardScaler
- MinMaxScaler
- RobustScaler
- Log transform
- Quantile transform

---

# Agent 3 — Feature Engineering & Correlation Optimization

## Role

This agent creates stronger predictive features and improves signal quality.

This stage focuses on:

- Feature creation
- Feature selection
- Feature interaction
- Correlation fixing
- Dimensionality reduction
- Signal amplification

---

## Tasks

### Feature Engineering

Possible techniques:

- Ratios
- Polynomial features
- Squares
- Cubes
- Log transforms
- Interaction terms
- Time-based features
- Rolling statistics
- Aggregation features
- Group statistics

### Required Reasoning

Explain:

- Why the new feature may improve prediction
- Whether domain logic supports it
- Risk of overfitting

---

### Correlation & Multicollinearity

Analyze:

- Correlation matrices
- VIF
- Feature clustering
- Redundant feature groups

---

# Agent 4 — Modeling Strategy & Training

## Role

This agent determines the best modeling approach.

Possible model categories:

- Classical ML
- Deep learning
- Statistical modeling
- Ensemble systems
- Hybrid approaches
- Non-ML systems

---

## Tasks

### Baseline Modeling

Possible baselines:

- Mean predictor
- Majority class
- Logistic regression
- Linear regression
- Decision tree

---

### Advanced Models

Possible models:

- Random Forest
- XGBoost
- LightGBM
- CatBoost
- SVM
- ElasticNet
- Neural Networks
- Transformers
- Bayesian Models
- ARIMA/SARIMA

### Required Reasoning

Explain:

- Why the model fits the data
- Dataset size considerations
- Interpretability tradeoffs
- Computational tradeoffs
- Risk of overfitting

---

### Data Splitting Strategy

Possible approaches:

- Train/validation/test
- Stratified split
- Group split
- Time-based split
- Cross-validation

---

### Hyperparameter Tuning

Possible methods:

- Grid search
- Random search
- Bayesian optimization
- Optuna

---

# Agent 5 — Validation, QA & Performance Review

## Role

This agent validates model quality and checks reliability.

---

## Tasks

### Metrics Evaluation

Classification metrics:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC

Regression metrics:

- RMSE
- MAE
- MAPE
- R²

---

### Error Analysis

Analyze:

- False positives
- False negatives
- Residual patterns
- Segment failures
- Edge cases

---

### QA Verification

Verify:

- No leakage
- Correct splits
- Reproducibility
- Stable preprocessing
- Proper random seeds

---

# Agent 6 — Iterative Retraining & Optimization

## Role

This agent continuously improves the system.

---

## Iterative Improvement Loop

```text
1. Select random subset from validation/test review pool
2. Analyze failure patterns
3. Add informative samples
4. Retrain pipeline
5. Re-evaluate metrics
6. Compare against previous version
7. Document improvement or regression
8. Repeat until stopping criteria met
```

---

## Stopping Criteria

Possible stopping conditions:

- Minimal performance gains
- Stable convergence
- Compute constraints
- User satisfaction
- Overfitting increase

---

# Cross-Agent Communication Protocol

Each agent must begin with:

```markdown
# Previous Agent Review

## Documents Reviewed

## Key Findings From Previous Agent

## Assumptions Continued

## Risks Inherited
```

And end with:

```markdown
# Handoff Summary

## What Was Completed

## What Needs Attention Next

## Known Risks

## Recommended Next Actions
```

---

# Experiment Tracking Standards

Every experiment must include:

```markdown
# Experiment ID

## Objective

## Dataset Version

## Feature Version

## Model Version

## Parameters

## Metrics

## Result Summary

## Lessons Learned
```

---

# Recommended Folder Structure

```text
/project
    /raw_data
    /intermediate_data
    /cleaned_data
    /feature_store
    /models
    /validation
    /reports
    /monitoring
    /experiments
    /logs
    /configs
```

---

# Recommended Thinking Process for Claude

At every stage, Claude should explicitly reason about:

1. What problem is being solved?
2. What assumptions exist?
3. What risks exist?
4. What evidence supports the decision?
5. What alternatives were considered?
6. Why was this approach selected?
7. What are the tradeoffs?
8. How reliable is the conclusion?
9. What additional data would help?
10. What should the next agent focus on?

---

# Preferred Decision-Making Style

Claude should prioritize:

1. Reliability over complexity
2. Reproducibility over experimentation
3. Generalization over training accuracy
4. Interpretability before black-box modeling
5. Evidence-based transformations
6. Conservative leakage prevention
7. Incremental improvement

---

# Final Notes

This workflow is designed to:

- Create modular AI data science systems
- Preserve reasoning transparency
- Improve collaboration between agents
- Support long-running projects
- Allow interruption and continuation
- Maintain reproducibility across sessions
- Produce auditable AI-driven data science pipelines

Every markdown file should be written as if another AI agent or human data scientist will continue the work later.
