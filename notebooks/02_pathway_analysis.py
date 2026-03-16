"""
Notebook: Pathway Analysis
Pathway score distributions, GSVA vs ssGSEA, pathway correlations, PCA comparison.
"""

# %% Imports & Setup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, pearsonr
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.visualization import (
    set_publication_theme,
    get_mm_palette,
    plot_heatmap,
    plot_pca_biplot,
    plot_distributions,
    export_figure,
)
from src.utils.reporting import ReportGenerator

set_publication_theme(style='nature')

# %% Load Data
"""Load expression data and compute pathway scores"""

np.random.seed(42)

n_samples = 500
n_genes = 20000
n_pathways = 50

# Expression data
expression_data = pd.DataFrame(
    np.random.randn(n_samples, n_genes),
    columns=[f'GENE_{i}' for i in range(n_genes)],
    index=[f'Sample_{i}' for i in range(n_samples)]
)

# Sample metadata
studies = np.random.choice(['Study_A', 'Study_B', 'Study_C'], n_samples)
iss = np.random.choice(['ISS-I', 'ISS-II', 'ISS-III'], n_samples)

metadata = pd.DataFrame({
    'study': studies,
    'ISS': iss,
}, index=expression_data.index)

print("Data loaded successfully!")
print(f"Expression: {expression_data.shape}")

# %% Generate Pathway Scores
"""
Simulate pathway scores using two methods: GSVA and ssGSEA
In practice, use gseapy or decoupler packages
"""

# Create pathway definitions (genes per pathway)
pathways = {
    f'pathway_{i}': np.random.choice(
        expression_data.columns,
        size=np.random.randint(20, 100),
        replace=False
    )
    for i in range(n_pathways)
}

# Method 1: GSVA-style scores (normalized mean expression)
gsva_scores = pd.DataFrame(
    np.random.randn(n_samples, n_pathways),
    columns=list(pathways.keys()),
    index=expression_data.index
)

# Method 2: ssGSEA-style scores (similar but different normalization)
ssgsea_scores = gsva_scores.copy()
ssgsea_scores = (ssgsea_scores - ssgsea_scores.mean()) / (ssgsea_scores.std() + 1e-8)

# Add study effect
for study in metadata['study'].unique():
    study_mask = metadata['study'] == study
    gsva_scores.loc[study_mask] += np.random.randn(n_pathways) * 0.3
    ssgsea_scores.loc[study_mask] += np.random.randn(n_pathways) * 0.2

print(f"\nPathway scores: {gsva_scores.shape}")
print(f"GSVA mean: {gsva_scores.values.mean():.3f}")
print(f"ssGSEA mean: {ssgsea_scores.values.mean():.3f}")

# %% Pathway Score Distributions
"""Violin plots of pathway scores across studies"""

# Select top variable pathways
pathway_var = gsva_scores.var()
top_pathways = pathway_var.nlargest(6).index.tolist()

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, pathway in enumerate(top_pathways):
    ax = axes[idx]

    data_by_study = {
        study: gsva_scores.loc[metadata['study'] == study, pathway].values
        for study in metadata['study'].unique()
    }

    positions = list(range(len(data_by_study)))
    parts = ax.violinplot(
        list(data_by_study.values()),
        positions=positions,
        showmeans=True,
        showmedians=True
    )

    # Color violins
    colors = list(get_mm_palette('study'))
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.6)

    ax.set_xticks(positions)
    ax.set_xticklabels(list(data_by_study.keys()), rotation=45)
    ax.set_ylabel('Pathway Score')
    ax.set_title(pathway, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')

plt.tight_layout()

# %% GSVA vs ssGSEA Correlation
"""Compare two pathway scoring methods"""

print("\n" + "="*60)
print("PATHWAY SCORING METHOD COMPARISON")
print("="*60)

# Compute correlations
method_corr = []
for pathway in gsva_scores.columns:
    r, p = pearsonr(gsva_scores[pathway], ssgsea_scores[pathway])
    method_corr.append({'pathway': pathway, 'correlation': r, 'p_value': p})

method_corr_df = pd.DataFrame(method_corr).sort_values('correlation', ascending=False)

print("\nTop 10 pathways - GSVA vs ssGSEA correlation:")
print(method_corr_df.head(10)[['pathway', 'correlation']].to_string(index=False))

# Scatter plot
fig, ax = plt.subplots(figsize=(8, 8))
scatter = ax.scatter(
    gsva_scores.mean(),
    ssgsea_scores.mean(),
    s=100,
    c=method_corr_df.set_index('pathway').loc[gsva_scores.columns, 'correlation'],
    cmap='RdYlGn',
    alpha=0.6,
    edgecolors='black',
    linewidth=1
)
ax.set_xlabel('GSVA (mean score)', fontsize=11)
ax.set_ylabel('ssGSEA (mean score)', fontsize=11)
ax.set_title('GSVA vs ssGSEA: Mean Pathway Scores', fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Method Correlation', fontsize=10)
ax.grid(True, alpha=0.2)
plt.tight_layout()

# %% Pathway-Pathway Correlation (GSVA)
"""Correlation structure between pathways"""

pathway_corr = gsva_scores.corr()

fig, ax = plt.subplots(figsize=(12, 10))
plot_heatmap(
    pathway_corr,
    ax=ax,
    cmap='coolwarm',
    title='Pathway-Pathway Correlation (GSVA)',
    cbar_label='Pearson r',
    vmin=-1, vmax=1,
    dendogram=True,
)
plt.tight_layout()

# %% Top Variable Pathways
"""Identify most variable pathways across samples"""

pathway_stats = pd.DataFrame({
    'pathway': gsva_scores.columns,
    'mean': gsva_scores.mean(),
    'std': gsva_scores.std(),
    'var': gsva_scores.var(),
    'cv': gsva_scores.std() / (gsva_scores.mean().abs() + 1e-8),
})

pathway_stats = pathway_stats.sort_values('var', ascending=False)

print("\n" + "="*60)
print("TOP VARIABLE PATHWAYS")
print("="*60)
print(pathway_stats.head(15).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Variance
top_var_pathways = pathway_stats.head(10)['pathway'].tolist()
ax = axes[0]
y_pos = np.arange(len(top_var_pathways))
ax.barh(y_pos, pathway_stats.head(10)['var'].values, color='#3498db', alpha=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels(top_var_pathways, fontsize=9)
ax.set_xlabel('Variance', fontsize=10)
ax.set_title('Top 10 Most Variable Pathways', fontweight='bold')
ax.grid(True, alpha=0.2, axis='x')

# CV
ax = axes[1]
top_cv_pathways = pathway_stats.nlargest(10, 'cv')['pathway'].tolist()
y_pos = np.arange(len(top_cv_pathways))
ax.barh(y_pos, pathway_stats[pathway_stats['pathway'].isin(top_cv_pathways)]['cv'].values,
       color='#e74c3c', alpha=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels(top_cv_pathways, fontsize=9)
ax.set_xlabel('Coefficient of Variation', fontsize=10)
ax.set_title('Top 10 Highest CV Pathways', fontweight='bold')
ax.grid(True, alpha=0.2, axis='x')

plt.tight_layout()

# %% PCA: Gene-level vs Pathway-level
"""Compare dimensionality reduction on genes vs pathways"""

# Gene-level PCA
scaler = StandardScaler()
expr_norm = scaler.fit_transform(expression_data)
pca_genes = PCA(n_components=2)
X_pca_genes = pca_genes.fit_transform(expr_norm)

# Pathway-level PCA
pathway_norm = StandardScaler().fit_transform(gsva_scores)
pca_pathways = PCA(n_components=2)
X_pca_pathways = pca_pathways.fit_transform(pathway_norm)

print("\n" + "="*60)
print("DIMENSIONALITY REDUCTION COMPARISON")
print("="*60)
print(f"\nGene-level PCA:")
print(f"  PC1 variance: {pca_genes.explained_variance_ratio_[0]:.1%}")
print(f"  PC2 variance: {pca_genes.explained_variance_ratio_[1]:.1%}")
print(f"  Total: {sum(pca_genes.explained_variance_ratio_):.1%}")

print(f"\nPathway-level PCA:")
print(f"  PC1 variance: {pca_pathways.explained_variance_ratio_[0]:.1%}")
print(f"  PC2 variance: {pca_pathways.explained_variance_ratio_[1]:.1%}")
print(f"  Total: {sum(pca_pathways.explained_variance_ratio_):.1%}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Gene-level
for study in metadata['study'].unique():
    mask = metadata['study'] == study
    ax1.scatter(X_pca_genes[mask, 0], X_pca_genes[mask, 1],
               label=study, s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
ax1.set_xlabel(f"PC1 ({pca_genes.explained_variance_ratio_[0]:.1%})", fontsize=10)
ax1.set_ylabel(f"PC2 ({pca_genes.explained_variance_ratio_[1]:.1%})", fontsize=10)
ax1.set_title('Gene-level PCA', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.2)

# Pathway-level
for study in metadata['study'].unique():
    mask = metadata['study'] == study
    ax2.scatter(X_pca_pathways[mask, 0], X_pca_pathways[mask, 1],
               label=study, s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
ax2.set_xlabel(f"PC1 ({pca_pathways.explained_variance_ratio_[0]:.1%})", fontsize=10)
ax2.set_ylabel(f"PC2 ({pca_pathways.explained_variance_ratio_[1]:.1%})", fontsize=10)
ax2.set_title('Pathway-level PCA', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.2)

plt.tight_layout()

# %% Pathway Scores by ISS
"""Pathway enrichment by ISS stage"""

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, pathway in enumerate(top_pathways):
    ax = axes[idx]

    data_by_iss = {
        iss_stage: gsva_scores.loc[metadata['ISS'] == iss_stage, pathway].values
        for iss_stage in metadata['ISS'].unique()
    }

    positions = list(range(len(data_by_iss)))
    parts = ax.violinplot(
        list(data_by_iss.values()),
        positions=positions,
        showmeans=True,
        showmedians=True
    )

    # Color
    colors = list(get_mm_palette('iss').values())
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.6)

    ax.set_xticks(positions)
    ax.set_xticklabels(list(data_by_iss.keys()), rotation=45)
    ax.set_ylabel('Pathway Score')
    ax.set_title(pathway, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')

plt.tight_layout()

# %% Pathway Score Summary Table
"""Summary statistics for all pathways"""

pathway_summary = pd.DataFrame({
    'Pathway': gsva_scores.columns,
    'Mean': gsva_scores.mean().values,
    'Std': gsva_scores.std().values,
    'Min': gsva_scores.min().values,
    'Max': gsva_scores.max().values,
    'CV': (gsva_scores.std() / (gsva_scores.mean().abs() + 1e-8)).values,
})

pathway_summary = pathway_summary.sort_values('Std', ascending=False)

print("\n" + "="*60)
print("PATHWAY SUMMARY STATISTICS")
print("="*60)
print(pathway_summary.head(15).round(4).to_string(index=False))

# %% Report Generation
"""Generate pathway analysis report"""

report = ReportGenerator(output_dir='reports', title='Pathway Analysis Report')

# Metrics
report.add_metric('Total Pathways', n_pathways, section='Summary')
report.add_metric('Top Pathway Variance', f"{pathway_stats.iloc[0]['var']:.3f}", section='Summary')
report.add_metric('Mean Method Correlation',
                 f"{method_corr_df['correlation'].mean():.3f}", section='Summary')

# Tables
report.add_table('Top Variable Pathways', pathway_stats.head(15).round(4), section='Results')
report.add_table('Method Comparison', method_corr_df.head(10).round(4), section='Results')
report.add_table('Pathway Summary', pathway_summary.head(20).round(4), section='Results')

print("\n" + "="*60)
print("PATHWAY ANALYSIS COMPLETE")
print("="*60)
