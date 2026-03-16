"""
Notebook: Cross-Study Alignment
Batch effect correction (CORAL, ComBat), silhouette analysis, distribution metrics.
"""

# %% Imports & Setup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.stats import ks_2samp, wasserstein_distance
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.visualization import (
    set_publication_theme,
    get_mm_palette,
    plot_pca_biplot,
    plot_heatmap,
    plot_distributions,
)
from src.utils.reporting import ReportGenerator

set_publication_theme(style='nature')

# %% Load Data
"""Load multi-study expression data"""

np.random.seed(42)

n_samples_per_study = 150
n_genes = 5000
studies = ['Study_A', 'Study_B', 'Study_C']

# Create data with batch effects
expression_data_raw = []
study_labels = []

for i, study in enumerate(studies):
    # Base expression
    X = np.random.randn(n_samples_per_study, n_genes)
    # Add study-specific batch effect
    X += np.random.randn(1, n_genes) * 2  # Study-level shift
    expression_data_raw.append(X)
    study_labels.extend([study] * n_samples_per_study)

expression_data_raw = pd.DataFrame(
    np.vstack(expression_data_raw),
    columns=[f'GENE_{i}' for i in range(n_genes)],
    index=[f'Sample_{i}' for i in range(n_samples_per_study * len(studies))]
)

metadata = pd.DataFrame({
    'study': study_labels,
}, index=expression_data_raw.index)

print("Raw data loaded!")
print(f"Expression: {expression_data_raw.shape}")
print(f"Studies: {metadata['study'].unique()}")

# %% Pre-Alignment Analysis
"""Analyze batch effects before correction"""

print("\n" + "="*60)
print("PRE-ALIGNMENT ANALYSIS")
print("="*60)

scaler = StandardScaler()
expr_norm = scaler.fit_transform(expression_data_raw)

pca_pre = PCA(n_components=2)
X_pca_pre = pca_pre.fit_transform(expr_norm)

# Silhouette score (lower = worse mixing, presence of batch)
silhouette_pre = silhouette_score(expr_norm, metadata['study'].values)
print(f"\nPre-alignment Silhouette Score: {silhouette_pre:.4f}")
print(f"(Negative indicates batch effect present)")

# Explained variance by study effect
study_var = []
for study in metadata['study'].unique():
    mask = metadata['study'] == study
    study_mean = expr_norm[mask].mean(axis=0)
    study_var.append(np.sum(study_mean ** 2))

study_effect_var = np.mean(study_var)
print(f"Mean study effect variance: {study_effect_var:.4f}")

fig, ax = plt.subplots(figsize=(8, 6))
for study in metadata['study'].unique():
    mask = metadata['study'] == study
    ax.scatter(X_pca_pre[mask, 0], X_pca_pre[mask, 1],
              label=study, s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
ax.set_xlabel(f'PC1 ({pca_pre.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca_pre.explained_variance_ratio_[1]:.1%})')
ax.set_title('Pre-Alignment PCA (Strong Batch Effect)', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.2)
plt.tight_layout()

# %% Distribution Metrics (Pre-alignment)
"""Quantify batch effects using distribution metrics"""

print("\n" + "="*60)
print("PRE-ALIGNMENT DISTRIBUTION METRICS")
print("="*60)

# KS statistic between studies (gene-wise)
ks_scores = []
for gene in expression_data_raw.columns[:100]:  # Test on 100 genes
    study_a = expr_norm[metadata['study'] == 'Study_A', expression_data_raw.columns.get_loc(gene)]
    study_b = expr_norm[metadata['study'] == 'Study_B', expression_data_raw.columns.get_loc(gene)]
    ks_stat, _ = ks_2samp(study_a, study_b)
    ks_scores.append(ks_stat)

print(f"\nMean KS statistic (Study_A vs Study_B):")
print(f"  Mean: {np.mean(ks_scores):.4f}")
print(f"  Std: {np.std(ks_scores):.4f}")

# %% ComBat-like Correction
"""
Simulate batch correction (parametric ComBat-like approach)
In practice, use combat-seq, harmonizationData, or similar
"""

# Simplified ComBat: remove study means, add grand mean
expr_corrected = expression_data_raw.copy()

grand_mean = expr_norm.mean(axis=0)
for study in metadata['study'].unique():
    mask = metadata['study'] == study
    study_mean = expr_norm[mask].mean(axis=0)
    batch_effect = study_mean - grand_mean
    expr_corrected.loc[mask] = expr_corrected.loc[mask] - batch_effect.reshape(1, -1)

expr_corrected_norm = StandardScaler().fit_transform(expr_corrected)

print("\n" + "="*60)
print("POST-ALIGNMENT ANALYSIS (ComBat-like)")
print("="*60)

pca_post = PCA(n_components=2)
X_pca_post = pca_post.fit_transform(expr_corrected_norm)

# Silhouette score (higher = better mixing, batch removed)
silhouette_post = silhouette_score(expr_corrected_norm, metadata['study'].values)
print(f"\nPost-alignment Silhouette Score: {silhouette_post:.4f}")
print(f"Improvement: {silhouette_post - silhouette_pre:.4f}")

# Explained variance by study effect (should be much lower)
study_var_post = []
for study in metadata['study'].unique():
    mask = metadata['study'] == study
    study_mean = expr_corrected_norm[mask].mean(axis=0)
    study_var_post.append(np.sum(study_mean ** 2))

study_effect_var_post = np.mean(study_var_post)
print(f"Post-alignment mean study effect variance: {study_effect_var_post:.4f}")
print(f"Reduction: {(study_effect_var - study_effect_var_post) / study_effect_var * 100:.1f}%")

fig, ax = plt.subplots(figsize=(8, 6))
for study in metadata['study'].unique():
    mask = metadata['study'] == study
    ax.scatter(X_pca_post[mask, 0], X_pca_post[mask, 1],
              label=study, s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
ax.set_xlabel(f'PC1 ({pca_post.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca_post.explained_variance_ratio_[1]:.1%})')
ax.set_title('Post-Alignment PCA (Batch Corrected)', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.2)
plt.tight_layout()

# %% Silhouette Analysis
"""Detailed silhouette scores per sample"""

silhouette_vals_pre = silhouette_samples(expr_norm, metadata['study'].values)
silhouette_vals_post = silhouette_samples(expr_corrected_norm, metadata['study'].values)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Pre-alignment
ax = axes[0]
y_lower = 10
for study in sorted(metadata['study'].unique()):
    mask = metadata['study'] == study
    cluster_silhouette_vals = silhouette_vals_pre[mask]
    cluster_silhouette_vals.sort()

    size_cluster = cluster_silhouette_vals.shape[0]
    y_upper = y_lower + size_cluster

    ax.fill_betweenx(np.arange(y_lower, y_upper),
                    0, cluster_silhouette_vals,
                    alpha=0.7, label=study)
    y_lower = y_upper + 10

ax.axvline(silhouette_pre, color='red', linestyle='--', linewidth=2, label='Mean')
ax.set_xlabel('Silhouette Coefficient')
ax.set_ylabel('Study')
ax.set_title(f'Pre-Alignment Silhouette Scores (mean={silhouette_pre:.3f})', fontweight='bold')
ax.set_ylim([0, len(metadata)])
ax.grid(True, alpha=0.2, axis='x')

# Post-alignment
ax = axes[1]
y_lower = 10
for study in sorted(metadata['study'].unique()):
    mask = metadata['study'] == study
    cluster_silhouette_vals = silhouette_vals_post[mask]
    cluster_silhouette_vals.sort()

    size_cluster = cluster_silhouette_vals.shape[0]
    y_upper = y_lower + size_cluster

    ax.fill_betweenx(np.arange(y_lower, y_upper),
                    0, cluster_silhouette_vals,
                    alpha=0.7, label=study)
    y_lower = y_upper + 10

ax.axvline(silhouette_post, color='red', linestyle='--', linewidth=2, label='Mean')
ax.set_xlabel('Silhouette Coefficient')
ax.set_ylabel('Study')
ax.set_title(f'Post-Alignment Silhouette Scores (mean={silhouette_post:.3f})', fontweight='bold')
ax.set_ylim([0, len(metadata)])
ax.grid(True, alpha=0.2, axis='x')

plt.tight_layout()

# %% Distribution Matching (KS statistics)
"""Compare gene distributions before and after correction"""

ks_pre = []
ks_post = []

for gene_idx in range(100):  # Sample genes
    study_a_pre = expr_norm[metadata['study'] == 'Study_A', gene_idx]
    study_b_pre = expr_norm[metadata['study'] == 'Study_B', gene_idx]
    ks_stat_pre, _ = ks_2samp(study_a_pre, study_b_pre)
    ks_pre.append(ks_stat_pre)

    study_a_post = expr_corrected_norm[metadata['study'] == 'Study_A', gene_idx]
    study_b_post = expr_corrected_norm[metadata['study'] == 'Study_B', gene_idx]
    ks_stat_post, _ = ks_2samp(study_a_post, study_b_post)
    ks_post.append(ks_stat_post)

ks_pre = np.array(ks_pre)
ks_post = np.array(ks_post)

print("\n" + "="*60)
print("KOLMOGOROV-SMIRNOV STATISTIC COMPARISON")
print("="*60)
print(f"\nPre-alignment KS (100 sampled genes):")
print(f"  Mean: {ks_pre.mean():.4f}")
print(f"  Median: {np.median(ks_pre):.4f}")
print(f"  Std: {ks_pre.std():.4f}")

print(f"\nPost-alignment KS (100 sampled genes):")
print(f"  Mean: {ks_post.mean():.4f}")
print(f"  Median: {np.median(ks_post):.4f}")
print(f"  Std: {ks_post.std():.4f}")

print(f"\nImprovement: {(ks_pre.mean() - ks_post.mean()) / ks_pre.mean() * 100:.1f}%")

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(ks_pre, bins=20, alpha=0.6, label='Pre-alignment', color='#e74c3c', edgecolor='black')
ax.hist(ks_post, bins=20, alpha=0.6, label='Post-alignment', color='#2ecc71', edgecolor='black')
ax.axvline(ks_pre.mean(), color='#e74c3c', linestyle='--', linewidth=2, label=f'Pre Mean: {ks_pre.mean():.3f}')
ax.axvline(ks_post.mean(), color='#2ecc71', linestyle='--', linewidth=2, label=f'Post Mean: {ks_post.mean():.3f}')
ax.set_xlabel('KS Statistic')
ax.set_ylabel('Frequency')
ax.set_title('Distribution Matching: KS Statistics Before/After Correction', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.2, axis='y')
plt.tight_layout()

# %% MMD Comparison (simplified)
"""Maximum Mean Discrepancy between studies"""

def compute_mmd(X1, X2, kernel='rbf', gamma=1.0):
    """Simplified MMD computation"""
    n1, n2 = X1.shape[0], X2.shape[0]

    if kernel == 'rbf':
        # RBF kernel
        K_XX = np.exp(-gamma * np.sum((X1[:, None, :] - X1[None, :, :]) ** 2, axis=2))
        K_YY = np.exp(-gamma * np.sum((X2[:, None, :] - X2[None, :, :]) ** 2, axis=2))
        K_XY = np.exp(-gamma * np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2))

        mmd = (K_XX.sum() / (n1 ** 2) +
               K_YY.sum() / (n2 ** 2) -
               2 * K_XY.sum() / (n1 * n2))
    else:
        # Linear kernel
        mmd = (np.mean(X1, axis=0) - np.mean(X2, axis=0)) ** 2

    return np.sqrt(max(mmd, 0))

print("\n" + "="*60)
print("MAXIMUM MEAN DISCREPANCY")
print("="*60)

# Study pairs
study_pairs = [('Study_A', 'Study_B'), ('Study_A', 'Study_C'), ('Study_B', 'Study_C')]

mmd_pre_list = []
mmd_post_list = []

for study1, study2 in study_pairs:
    X1_pre = expr_norm[metadata['study'] == study1][:, :100]  # Use 100 genes
    X2_pre = expr_norm[metadata['study'] == study2][:, :100]

    mmd_pre = compute_mmd(X1_pre, X2_pre, gamma=0.1)
    mmd_pre_list.append(mmd_pre)

    X1_post = expr_corrected_norm[metadata['study'] == study1][:, :100]
    X2_post = expr_corrected_norm[metadata['study'] == study2][:, :100]

    mmd_post = compute_mmd(X1_post, X2_post, gamma=0.1)
    mmd_post_list.append(mmd_post)

    print(f"\n{study1} vs {study2}:")
    print(f"  Pre-alignment MMD: {mmd_pre:.4f}")
    print(f"  Post-alignment MMD: {mmd_post:.4f}")
    print(f"  Reduction: {(mmd_pre - mmd_post) / mmd_pre * 100:.1f}%")

# %% Alignment Metrics Summary Table
"""Create summary table of alignment metrics"""

alignment_metrics = pd.DataFrame({
    'Metric': [
        'Silhouette Score',
        'Study Effect Variance',
        'Mean KS Statistic',
        'Mean MMD',
    ],
    'Pre-Alignment': [
        f'{silhouette_pre:.4f}',
        f'{study_effect_var:.4f}',
        f'{ks_pre.mean():.4f}',
        f'{np.mean(mmd_pre_list):.4f}',
    ],
    'Post-Alignment': [
        f'{silhouette_post:.4f}',
        f'{study_effect_var_post:.4f}',
        f'{ks_post.mean():.4f}',
        f'{np.mean(mmd_post_list):.4f}',
    ],
    'Improvement': [
        f'{silhouette_post - silhouette_pre:.4f}',
        f'{(study_effect_var - study_effect_var_post) / study_effect_var * 100:.1f}%',
        f'{(ks_pre.mean() - ks_post.mean()) / ks_pre.mean() * 100:.1f}%',
        f'{(np.mean(mmd_pre_list) - np.mean(mmd_post_list)) / np.mean(mmd_pre_list) * 100:.1f}%',
    ]
})

print("\n" + "="*60)
print("ALIGNMENT METRICS SUMMARY")
print("="*60)
print(alignment_metrics.to_string(index=False))

# %% Report Generation
"""Generate alignment report"""

report = ReportGenerator(output_dir='reports', title='Cross-Study Alignment Report')

report.add_metric('Pre-Alignment Silhouette', f"{silhouette_pre:.4f}", section='Metrics')
report.add_metric('Post-Alignment Silhouette', f"{silhouette_post:.4f}", section='Metrics')
report.add_metric('Silhouette Improvement', f"{silhouette_post - silhouette_pre:.4f}", section='Metrics')
report.add_metric('KS Improvement', f"{(ks_pre.mean() - ks_post.mean()) / ks_pre.mean() * 100:.1f}%", section='Metrics')

report.add_table('Alignment Metrics Summary', alignment_metrics, section='Results')

print("\n" + "="*60)
print("CROSS-STUDY ALIGNMENT COMPLETE")
print("="*60)
