"""
Notebook: Model Comparison & Evaluation
Forest plots, calibration, DCA, subgroup robustness, ablation studies.
"""

# %% Imports & Setup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import concordance_index, roc_auc_score, brier_score_loss
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.visualization import (
    set_publication_theme,
    get_mm_palette,
    plot_forest,
    plot_calibration,
    plot_dca,
    export_figure,
)
from src.utils.reporting import ReportGenerator

set_publication_theme(style='nature')

# %% Generate Mock Model Results
"""
Simulate results from multiple models:
- Gene-only (univariate/multivariate)
- Pathway-only
- Fusion (gene + pathway)
- Published signatures (GEP70, EMC92, SKY92)
"""

np.random.seed(42)

n_samples = 500
n_test_samples = 200

# Survival outcome
y_true = np.random.binomial(1, 0.6, n_test_samples)

# Mock predictions from different models
models_dict = {
    'Gene-only (Cox)': {
        'y_pred': np.random.uniform(0.3, 0.9, n_test_samples),
        'c_index': 0.72,
        'ci_lower': 0.68,
        'ci_upper': 0.76,
    },
    'Pathway-only (Cox)': {
        'y_pred': np.random.uniform(0.4, 0.85, n_test_samples),
        'c_index': 0.68,
        'ci_lower': 0.64,
        'ci_upper': 0.72,
    },
    'Gene + Pathway (Fusion)': {
        'y_pred': np.random.uniform(0.25, 0.95, n_test_samples),
        'c_index': 0.76,
        'ci_lower': 0.72,
        'ci_upper': 0.80,
    },
    'Gene + Pathway + Clinical': {
        'y_pred': np.random.uniform(0.2, 0.98, n_test_samples),
        'c_index': 0.78,
        'ci_lower': 0.74,
        'ci_upper': 0.82,
    },
    'GEP70': {
        'y_pred': np.random.uniform(0.35, 0.88, n_test_samples),
        'c_index': 0.70,
        'ci_lower': 0.66,
        'ci_upper': 0.74,
    },
    'SKY92': {
        'y_pred': np.random.uniform(0.32, 0.86, n_test_samples),
        'c_index': 0.71,
        'ci_lower': 0.67,
        'ci_upper': 0.75,
    },
}

print("Mock model results generated!")
print(f"Test samples: {n_test_samples}")
print(f"Event rate: {y_true.mean():.1%}")

# %% Forest Plot: C-index Comparison
"""Plot concordance index with 95% CI across models"""

print("\n" + "="*60)
print("MODEL COMPARISON: CONCORDANCE INDEX")
print("="*60)

model_names = list(models_dict.keys())
c_indices = [models_dict[m]['c_index'] for m in model_names]
ci_lowers = [models_dict[m]['ci_lower'] for m in model_names]
ci_uppers = [models_dict[m]['ci_upper'] for m in model_names]

print("\nC-index with 95% CI:")
for name, c_idx, ci_l, ci_u in zip(model_names, c_indices, ci_lowers, ci_uppers):
    print(f"  {name}: {c_idx:.3f} ({ci_l:.3f}-{ci_u:.3f})")

fig, ax = plt.subplots(figsize=(10, 7))
plot_forest(
    model_names,
    c_indices,
    ci_lowers,
    ci_uppers,
    ax=ax,
    reference_line=0.5,
    xlabel='C-index (95% CI)',
    title='Model Comparison: Concordance Index',
)
plt.tight_layout()

# %% Calibration Plots
"""Plot calibration curves for top models"""

top_models = ['Gene-only (Cox)', 'Gene + Pathway (Fusion)', 'Gene + Pathway + Clinical']

fig, axes = plt.subplots(1, len(top_models), figsize=(15, 5))
if len(top_models) == 1:
    axes = [axes]

for idx, model in enumerate(top_models):
    y_pred = models_dict[model]['y_pred']
    plot_calibration(
        y_true,
        y_pred,
        ax=axes[idx],
        n_bins=5,
        title=f'{model}',
        fontsize=9,
    )

plt.tight_layout()

# %% Decision Curve Analysis
"""Plot net benefit across prediction thresholds"""

fig, ax = plt.subplots(figsize=(9, 6))
plot_dca(
    y_true,
    models_dict['Gene + Pathway + Clinical']['y_pred'],
    ax=ax,
    title='Decision Curve Analysis: Fusion Model',
)
plt.tight_layout()

# %% Performance Metrics Table
"""Comprehensive model evaluation metrics"""

metrics_list = []

for model in model_names:
    y_pred = models_dict[model]['y_pred']
    c_idx = concordance_index(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_pred)

    metrics_list.append({
        'Model': model,
        'C-index': f"{c_idx:.3f}",
        'ROC-AUC': f"{roc_auc:.3f}",
        'Brier Score': f"{brier:.3f}",
        'Sensitivity': f"{(y_pred[y_true == 1] > 0.5).mean():.3f}",
        'Specificity': f"{(y_pred[y_true == 0] <= 0.5).mean():.3f}",
    })

metrics_df = pd.DataFrame(metrics_list)

print("\n" + "="*60)
print("COMPREHENSIVE MODEL METRICS")
print("="*60)
print(metrics_df.to_string(index=False))

# %% Subgroup Analysis
"""Robustness across patient subgroups"""

# Define subgroups
subgroups = {
    'ISS-I': np.random.binomial(1, 0.7, n_test_samples),  # Better prognosis
    'ISS-III': np.random.binomial(1, 0.4, n_test_samples),  # Poor prognosis
    'Age <65': np.random.binomial(1, 0.55, n_test_samples),
    'Age >=65': np.random.binomial(1, 0.65, n_test_samples),
}

subgroup_performance = []

for model in ['Gene-only (Cox)', 'Gene + Pathway (Fusion)', 'Gene + Pathway + Clinical']:
    y_pred = models_dict[model]['y_pred']

    for subgroup_name, subgroup_mask in subgroups.items():
        c_idx = concordance_index(y_true[subgroup_mask.astype(bool)],
                                 y_pred[subgroup_mask.astype(bool)])
        subgroup_performance.append({
            'Model': model,
            'Subgroup': subgroup_name,
            'C-index': c_idx,
        })

subgroup_perf_df = pd.DataFrame(subgroup_performance)

# Heatmap of subgroup performance
pivot_subgroup = subgroup_perf_df.pivot(index='Model', columns='Subgroup', values='C-index')

fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(
    pivot_subgroup,
    annot=True,
    fmt='.3f',
    cmap='RdYlGn',
    center=0.6,
    vmin=0.5, vmax=0.8,
    ax=ax,
    cbar_kws={'label': 'C-index'},
    linewidths=1,
    linecolor='white'
)
ax.set_title('Subgroup Robustness Analysis', fontweight='bold')
ax.set_ylabel('Model')
ax.set_xlabel('Subgroup')
plt.tight_layout()

# %% Ablation Study
"""Component importance by removing features"""

ablation_results = []

components = {
    'Gene features': 1.0,
    'Pathway features': 0.2,
    'Clinical variables': 0.15,
    'Gene + Pathway': 0.8,
    'Gene + Clinical': 0.75,
    'Pathway + Clinical': 0.35,
    'All features': 1.0,
}

for component, importance in components.items():
    # Simulate performance drop without component
    c_idx_ablated = 0.76 * importance + 0.5 * (1 - importance)
    ablation_results.append({
        'Component': component,
        'Importance (%)': importance * 100,
        'C-index': c_idx_ablated,
    })

ablation_df = pd.DataFrame(ablation_results)

fig, ax = plt.subplots(figsize=(10, 6))
y_pos = np.arange(len(ablation_df))
colors = ['#3498db' if 'All' not in c else '#e74c3c' for c in ablation_df['Component']]
ax.barh(y_pos, ablation_df['C-index'].values, color=colors, alpha=0.7, edgecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(ablation_df['Component'].values)
ax.set_xlabel('C-index')
ax.set_title('Ablation Study: Feature Component Importance', fontweight='bold')
ax.set_xlim([0.4, 0.8])
ax.grid(True, alpha=0.2, axis='x')
for i, v in enumerate(ablation_df['C-index'].values):
    ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)
plt.tight_layout()

print("\n" + "="*60)
print("ABLATION STUDY")
print("="*60)
print(ablation_df.to_string(index=False))

# %% Model Ranking
"""Final model ranking by multiple criteria"""

ranking_scores = []

for model in model_names:
    c_idx = models_dict[model]['c_index']
    # Calculate weighted score (C-index 80%, robustness 20%)
    robustness_score = 1 - np.std(
        [subgroup_perf_df[
            (subgroup_perf_df['Model'] == model)
        ]['C-index'].values]
    )
    weighted_score = c_idx * 0.8 + (0.5 + robustness_score * 0.5) * 0.2

    ranking_scores.append({
        'Rank': 0,  # Will fill after sorting
        'Model': model,
        'C-index': f"{c_idx:.3f}",
        'Robustness': f"{robustness_score:.3f}",
        'Weighted Score': f"{weighted_score:.3f}",
    })

ranking_df = pd.DataFrame(ranking_scores).sort_values(
    'Weighted Score', ascending=False
).reset_index(drop=True)
ranking_df['Rank'] = np.arange(1, len(ranking_df) + 1)

print("\n" + "="*60)
print("FINAL MODEL RANKING")
print("="*60)
print(ranking_df.to_string(index=False))

# %% Report Generation
"""Generate model comparison report"""

report = ReportGenerator(output_dir='reports', title='Model Comparison Report')

# Metrics
best_model = ranking_df.iloc[0]
report.add_metric('Best Model', best_model['Model'], section='Summary')
report.add_metric('Best C-index', best_model['C-index'], section='Summary')
report.add_metric('Number of Models', len(model_names), section='Summary')

# Tables
report.add_table('Model Performance Metrics', metrics_df, section='Results')
report.add_table('Subgroup Robustness', pivot_subgroup, section='Results')
report.add_table('Ablation Study', ablation_df, section='Results')
report.add_table('Model Ranking', ranking_df, section='Results')

print("\n" + "="*60)
print("MODEL COMPARISON COMPLETE")
print("="*60)
