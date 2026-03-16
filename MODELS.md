# Classical Survival Modeling Pipeline

## Overview

Production-grade implementations of classical survival models for bulk transcriptomics MM risk stratification.

**Key Objective**: Sparse group-lasso Cox model for pathway-aware survival prediction:
```
min_β -ℓ_Cox(β) + λ₁‖β‖₁ + λ₂ Σ_g w_g ‖β_g‖₂
```

## Module Structure

### 1. Base Models (`src/models/baselines/`)

#### `base_model.py`
Abstract base class for all survival models.

**Key Classes**:
- `BaseSurvivalModel`: Interface with `fit()`, `predict_risk()`, `predict_survival_function()`
- `SurvivalModelCV`: Patient-level cross-validation wrapper

**Features**:
- Standardized feature scaling
- MLflow logging integration
- Model serialization (joblib)
- Cross-validation with stratification

```python
from src.models.baselines.base_model import BaseSurvivalModel

class MyModel(BaseSurvivalModel):
    def fit(self, X, y_time, y_event, **kwargs):
        X_array, y_time_array, y_event_array = self._validate_input(X, y_time, y_event)
        X_scaled = self._fit_scaler(X_array)
        # ... model-specific fit logic
        return self

    def predict_risk(self, X):
        X_array, _, _ = self._validate_input(X)
        X_scaled = self._transform_scaler(X_array)
        # ... return risk scores
```

---

#### `sparse_group_lasso_cox.py` — KEY MODEL
**Sparse Group Lasso Cox Proportional Hazards**

Implements the main objective:
```
min_β -ℓ_Cox(β) + λ₁‖β‖₁ + λ₂ Σ_g w_g ‖β_g‖₂
```

**Parameters**:
- `lambda1`: L1 (Lasso) penalty weight
- `lambda2`: L2 (Group Lasso) penalty weight
- `groups`: Pathway ontology (gene → group mapping)
- `group_weights`: w_g = sqrt(|group_g|) by default

**Features**:
- Proximal gradient descent solver
- Pathway-level sparsity
- Group identification and interpretation
- Nested CV for (λ₁, λ₂) selection

**Usage**:
```python
from src.models.baselines.sparse_group_lasso_cox import SparseGroupLassoCoxModel

# Define pathway groups
groups = {
    "mRNA_splicing": [0, 1, 2, 5],
    "DNA_repair": [3, 4, 10],
    "immune_response": [6, 7, 8, 9],
    # ... more pathways
}

model = SparseGroupLassoCoxModel(
    lambda1=0.1,
    lambda2=0.05,
    groups=groups,
)

model.fit(X_pathways, y_time, y_event)

# Get selected pathways
selected = model.get_selected_pathways()
print(selected)

# Nested CV for optimal (λ₁, λ₂)
best_l1, best_l2, cv_score, cv_scores = model.nested_cv_lambda_selection(
    X_pathways, y_time, y_event,
    lambda1_grid=np.logspace(-4, 0, 20),
    lambda2_grid=np.logspace(-4, 0, 20),
    cv_inner=3,
    cv_outer=5,
)
```

---

#### `lasso_cox.py`
**L1-Penalized Cox Model**

Pure Lasso (L1 only, l1_ratio=1.0).

**Features**:
- Feature-level sparsity
- Nested CV for lambda selection
- Feature importance extraction

```python
from src.models.baselines.lasso_cox import LassoCoxModel

model = LassoCoxModel(alpha=0.5, l1_ratio=1.0)
model.fit(X, y_time, y_event)

# Get selected features
selected = model.get_selected_features(threshold=0.01)

# Nested CV
best_alpha, cv_score, cv_scores = model.fit_nested_cv(
    X, y_time, y_event,
    alpha_grid=np.logspace(-4, 2, 20),
    cv_inner=3,
    cv_outer=5,
)
```

---

#### `elastic_net_cox.py`
**Elastic Net Cox Model**

Combines L1 and L2 penalties (configurable via `l1_ratio`).

**Features**:
- Flexible L1/L2 balance
- Grid search for (alpha, l1_ratio)
- Stable on correlated features

```python
from src.models.baselines.elastic_net_cox import ElasticNetCoxModel

model = ElasticNetCoxModel(alpha=0.1, l1_ratio=0.5)
model.fit(X, y_time, y_event)

# Grid search
best_alpha, best_l1_ratio, results = model.grid_search_cv(
    X, y_time, y_event,
    alpha_grid=np.logspace(-4, 2, 15),
    l1_ratio_grid=np.linspace(0, 1, 6),
    cv=5,
)
```

---

#### `de_enrichment.py`
**Differential Expression + Enrichment Baseline**

Interpretable baseline: stratify patients, DE analysis, pathway scoring.

**Workflow**:
1. Stratify patients by median/tertile survival time
2. Perform limma-style t-tests and correlation analysis
3. Score pathways by association with outcome
4. Combine into weighted risk score

**Usage**:
```python
from src.models.baselines.de_enrichment import DEEnrichmentBaseline

model = DEEnrichmentBaseline(
    risk_stratification="median",
    n_pathways=50,
)

model.fit(X_pathways, y_time, y_event)

# Inspect DE results
print(model.de_stats_)  # t-stats, p-values, correlation
print(model.selected_pathways_)  # Top pathways
```

---

#### `random_survival_forest.py`
**Random Survival Forest**

Tree-based ensemble via scikit-survival.

**Features**:
- Nonlinear risk modeling
- Permutation importance
- Hyperparameter grid search
- Robust to feature scale

```python
from src.models.baselines.random_survival_forest import RandomSurvivalForestModel

model = RandomSurvivalForestModel(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=5,
)

model.fit(X, y_time, y_event)

# Feature importance
importance = model.feature_importance_

# Permutation importance with CI
perm_imp = model.compute_permutation_importance(
    X, y_time, y_event,
    n_repeats=10,
)

# Grid search
best_params, best_score = model.grid_search_cv(
    X, y_time, y_event,
    param_grid={
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 20, None],
        "min_samples_leaf": [1, 5, 10],
    },
    cv=5,
)
```

---

#### `gradient_boosting_survival.py`
**XGBoost and CatBoost Survival Models**

Modern gradient boosting for survival.

**XGBoost**:
- AFT (Accelerated Failure Time) objective
- Cox objective
- SHAP explanations

```python
from src.models.baselines.gradient_boosting_survival import XGBoostSurvivalModel

model = XGBoostSurvivalModel(
    objective="survival:cox",  # or "survival:aft"
    n_estimators=100,
    max_depth=6,
)

model.fit(X, y_time, y_event)

# SHAP explanations
shap_dict = model.explain_with_shap(X[:50], sample_size=100)
shap_values = shap_dict["shap_values"]
base_value = shap_dict["base_value"]
```

**CatBoost**:
- Native Cox loss function
- Categorical feature support
- Grid search

```python
from src.models.baselines.gradient_boosting_survival import CatBoostSurvivalModel

model = CatBoostSurvivalModel(
    iterations=100,
    depth=6,
    l2_leaf_reg=3.0,
)

model.fit(X, y_time, y_event)

best_params, best_score = model.grid_search_cv(
    X, y_time, y_event,
    param_grid={
        "iterations": [50, 100, 200],
        "depth": [4, 6, 8],
        "learning_rate": [0.01, 0.1, 0.3],
    },
    cv=5,
)
```

---

### 2. Evaluation (`src/evaluation/`)

#### `splits.py`
**Patient-Level Cross-Validation**

Ensures no patient-level data leakage.

**Classes**:
- `PatientLevelSplitter`: Basic patient-level CV
- `TimeAwareSplitter`: Split by follow-up time windows
- `StratifiedTimeAwareSplitter`: Stratified by ISS/cytogenetics within time windows
- `LeaveOneStudyOutSplitter`: LOSO for multi-cohort validation
- `NestedCVSplitter`: Outer CV for evaluation, inner CV for tuning

**Usage**:
```python
from src.evaluation.splits import PatientLevelSplitter, LeaveOneStudyOutSplitter

# Basic patient-level CV
splitter = PatientLevelSplitter(n_splits=5)
splits = splitter.split(X, y_event)

# Stratified by clinical variables
stratify_df = pd.DataFrame({
    "ISS_stage": y["ISS"],
    "cytogenetics": y["cytogenetics"],
})
splitter = PatientLevelSplitter(
    n_splits=5,
    stratify_by=stratify_df,
)
splits = splitter.split(X, y_event)

# Leave-one-study-out
loso_splitter = LeaveOneStudyOutSplitter()
splits = loso_splitter.split(X, y["study_id"])

# Nested CV
nested = NestedCVSplitter(n_splits_outer=5, n_splits_inner=3)
outer_splits = nested.split(X, y_event)
# For each outer fold, use nested.inner_split(X_train, y_train_event)
```

---

#### `metrics.py`
**Comprehensive Survival Metrics**

**Static Methods**:
- `concordance_index()`: Harrell's C-index
- `time_dependent_auc()`: AUC(t) at specific time points
- `brier_score()`: MSE of survival probability
- `integrated_brier_score()`: IBS over follow-up
- `calibration_metrics()`: Slope, intercept, D-calibration
- `subgroup_robustness()`: C-index per patient subgroup
- `bootstrap_ci()`: Confidence intervals via bootstrap
- `pairwise_model_comparison()`: Statistical test between two models
- `evaluate_summary()`: Comprehensive metrics table

**Usage**:
```python
from src.evaluation.metrics import SurvivalMetrics

# C-index
c_index = SurvivalMetrics.concordance_index(y_event, y_time, predictions)

# Time-dependent AUC
auc_dict = SurvivalMetrics.time_dependent_auc(
    y_event, y_time, predictions,
    times=[1, 3, 5],  # years
)

# Calibration
cal_metrics = SurvivalMetrics.calibration_metrics(y_event, y_time, predictions)
print(f"Slope: {cal_metrics['calibration_slope']:.3f}")
print(f"Intercept: {cal_metrics['calibration_intercept']:.3f}")

# Subgroup robustness
subgroup_scores = SurvivalMetrics.subgroup_robustness(
    y_event, y_time, predictions,
    subgroups=y["ISS_stage"],
)

# Bootstrap CI
point, ci_low, ci_high = SurvivalMetrics.bootstrap_ci(
    y_event, y_time, predictions,
    SurvivalMetrics.concordance_index,
    n_bootstraps=1000,
)

# Pairwise comparison
comparison = SurvivalMetrics.pairwise_model_comparison(
    y_event, y_time,
    pred_model1, pred_model2,
)
print(comparison["winner"])

# Summary
summary = SurvivalMetrics.evaluate_summary(
    y_event, y_time, predictions,
    survival_probs=surv_probs,
    subgroups=y["ISS_stage"],
)
```

---

#### `benchmark.py`
**Cross-Study Benchmarking**

**Classes**:
- `CrossStudyBenchmark`: Train-test across cohorts, LOSO-CV, forest plots
- `ModelComparisonReport`: Statistical comparisons

**Usage**:
```python
from src.evaluation.benchmark import CrossStudyBenchmark

benchmark = CrossStudyBenchmark()

# Train on CoMMpass, test on external cohorts
external_cohorts = {
    "GSE24080": (X_gse, y_time_gse, y_event_gse),
    "GSE4204": (X_gse4204, y_time_gse4204, y_event_gse4204),
}

results = benchmark.train_test_external(
    X_commpass, y_time_commpass, y_event_commpass,
    models={
        "sparse_group_lasso": model_sgl,
        "lasso_cox": model_lasso,
        "rsf": model_rsf,
    },
    external_cohorts=external_cohorts,
)

# Create comparison table
comparison_table = benchmark.create_comparison_table(results, metric="c_index")
print(comparison_table)

# Forest plot data
forest_data = benchmark.create_forest_plot_data(results)

# Publication summary
summary = benchmark.get_publication_summary(results)
print(summary)

# Leave-one-study-out CV
y_with_study_id = y.copy()
y_with_study_id["study_id"] = study_ids

loso_results = benchmark.loso_cv(
    X, y_with_study_id["time"], y_with_study_id["event"],
    study_ids=y_with_study_id["study_id"],
    models={...},
)
```

---

### 3. Tests (`tests/test_models.py`)

Comprehensive unit and integration tests with synthetic survival data.

**Test Classes**:
- `TestBaseSurvivalModel`: Input validation, scaling
- `TestDEEnrichmentBaseline`: Fit, predict, pathway selection
- `TestLassoCoxModel`: Fit, predict, nested CV, feature selection
- `TestElasticNetCoxModel`: Fit, grid search
- `TestSparseGroupLassoCoxModel`: Fit, predict, pathway selection, nested lambda selection
- `TestRandomSurvivalForest`: Fit, feature importance, grid search
- `TestPatientLevelSplitter`: Basic and stratified splits
- `TestLeaveOneStudyOutSplitter`: LOSO splits
- `TestSurvivalMetrics`: All evaluation metrics
- `TestCVWrapper`: Integration test

**Run Tests**:
```bash
# All tests
pytest tests/test_models.py -v

# Specific test class
pytest tests/test_models.py::TestSparseGroupLassoCoxModel -v

# With coverage
pytest tests/test_models.py --cov=src --cov-report=html
```

---

## Quick Start Example

```python
import numpy as np
import pandas as pd
from src.models.baselines.sparse_group_lasso_cox import SparseGroupLassoCoxModel
from src.evaluation.splits import PatientLevelSplitter
from src.evaluation.metrics import SurvivalMetrics

# 1. Load data (GSVA/ssGSEA pathway scores)
X = pd.read_parquet("pathway_scores.parquet")  # (n_samples, n_pathways)
y = pd.read_csv("clinical_data.csv", index_col=0)  # OS, PFS, ISS, cytogenetics, etc.

# 2. Define pathway groups
groups = {
    "mRNA_splicing": X.columns[X.columns.str.contains("splicing", case=False)].tolist(),
    "immune_response": X.columns[X.columns.str.contains("immune", case=False)].tolist(),
    # ... more groups from pathway ontology
}

# 3. Train sparse group lasso Cox model
model = SparseGroupLassoCoxModel(
    lambda1=0.1,
    lambda2=0.05,
    groups=groups,
)

# Nested CV for optimal lambda
best_l1, best_l2, cv_score, cv_scores = model.nested_cv_lambda_selection(
    X, y["OS_time"], y["OS_event"],
    lambda1_grid=np.logspace(-4, 0, 20),
    lambda2_grid=np.logspace(-4, 0, 20),
)

print(f"Best λ₁: {best_l1:.4f}, λ₂: {best_l2:.4f}")
print(f"CV C-index: {cv_score:.3f} (per-fold: {cv_scores})")

# 4. Get selected pathways
pathways = model.get_selected_pathways()
print(pathways)

# 5. Cross-study benchmark
from src.evaluation.benchmark import CrossStudyBenchmark

benchmark = CrossStudyBenchmark()
results = benchmark.train_test_external(
    X, y["OS_time"], y["OS_event"],
    models={"sparse_group_lasso": model},
    external_cohorts={
        "GSE24080": (X_ext, y_ext_time, y_ext_event),
    },
)

comparison = benchmark.create_comparison_table(results)
print(comparison)
```

---

## Model Selection Guide

| Use Case | Recommended Model | Reason |
|----------|-------------------|--------|
| **Pathway-aware interpretation** | Sparse Group Lasso Cox | Identifies selected pathways, structured sparsity |
| **Feature selection, single-level** | Lasso Cox | Simple, interpretable feature selection |
| **Balanced L1/L2** | Elastic Net Cox | Stable with correlated features |
| **Interpretable baseline** | DE + Enrichment | Transparent workflow (DE → ORA/GSEA → risk) |
| **Nonlinear relationships** | Random Survival Forest | Captures feature interactions |
| **Production + SHAP** | XGBoost/CatBoost | State-of-the-art, explainable |

---

## File Locations

```
/sessions/sweet-stoic-cray/r2/
├── src/
│   └── models/
│       └── baselines/
│           ├── __init__.py
│           ├── base_model.py                      # Abstract base + CV wrapper
│           ├── sparse_group_lasso_cox.py          # KEY MODEL
│           ├── lasso_cox.py
│           ├── elastic_net_cox.py
│           ├── de_enrichment.py
│           ├── random_survival_forest.py
│           └── gradient_boosting_survival.py
│   └── evaluation/
│       ├── __init__.py
│       ├── splits.py                               # Patient-level CV
│       ├── metrics.py                              # Comprehensive metrics
│       └── benchmark.py                            # Cross-study validation
├── tests/
│   └── test_models.py                             # Unit + integration tests
├── requirements.txt
└── MODELS.md                                       # This file
```

---

## Dependencies

See `requirements.txt`:
- **Core**: numpy, pandas, scipy
- **Survival**: scikit-survival, scikit-learn
- **Boosting**: xgboost, catboost
- **Interpretation**: shap
- **MLflow**: experiment tracking
- **Testing**: pytest

Install:
```bash
pip install -r requirements.txt
```

---

## Citation

If using the sparse group lasso Cox model in publications:

> [Cite your paper when sparse group lasso implementation is used]

Classical Cox: Cox, D. R. (1972). Regression models and life tables. JRSS-B, 34(2), 187-220.

Scikit-survival: Pölsterl, S. (2020). scikit-survival: A library for time-to-event analysis. JMLR, 21(212), 1-6.
