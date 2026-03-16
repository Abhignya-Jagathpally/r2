"""
Notebook: Data Exploration & Quality Control
Load datasets, summarize, generate KM curves, PCA, density plots, missing data.
"""

# %% Imports & Setup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys

# Add source to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.visualization import (
    set_publication_theme,
    get_mm_palette,
    plot_km_curve,
    plot_distributions,
    plot_pca_biplot,
    plot_heatmap,
    export_figure,
)
from src.utils.reporting import ReportGenerator, MLflowReporter

# Set theme
set_publication_theme(style='nature')

# %% Load Data
"""
Load multi-study transcriptomics data.
Expected structure:
- expression_data: (n_samples, n_genes)
- metadata: (n_samples,) with columns: study, ISS, cytogenetics, survival_time, event
"""

# Placeholder - replace with actual data loading
np.random.seed(42)

n_samples = 500
n_genes = 20000

# Mock data
expression_data = pd.DataFrame(
    np.random.randn(n_samples, n_genes),
    columns=[f'GENE_{i}' for i in range(n_genes)],
    index=[f'Sample_{i}' for i in range(n_samples)]
)

studies = np.random.choice(['Study_A', 'Study_B', 'Study_C'], n_samples)
iss = np.random.choice(['ISS-I', 'ISS-II', 'ISS-III'], n_samples)
cytogenetics = np.random.choice(
    ['Standard', 'Intermediate', 'High-risk'],
    n_samples, p=[0.5, 0.3, 0.2]
)

# Realistic survival data
survival_time = np.random.exponential(scale=36, size=n_samples)  # months
event = np.random.binomial(1, 0.6, n_samples)

metadata = pd.DataFrame({
    'study': studies,
    'ISS': iss,
    'cytogenetics': cytogenetics,
    'survival_time': survival_time,
    'event': event,
}, index=expression_data.index)

print("Data loaded successfully!")
print(f"Expression: {expression_data.shape}")
print(f"Metadata: {metadata.shape}")

# %% Data Summary
"""Basic data characteristics"""

print("\n" + "="*60)
print("DATA SUMMARY")
print("="*60)

print(f"\nSamples: {n_samples}")
print(f"Genes: {n_genes}")
print(f"Studies: {metadata['study'].nunique()}")

print("\nStudy distribution:")
print(metadata['study'].value_counts())

print("\nISS distribution:")
print(metadata['ISS'].value_counts())

print("\nCytogenetics distribution:")
print(metadata['cytogenetics'].value_counts())

print("\nSurvival statistics:")
print(f"  Median follow-up: {metadata['survival_time'].median():.1f} months")
print(f"  Event rate: {metadata['event'].mean():.1%}")
print(f"  Events: {metadata['event'].sum()}")

print("\nExpression statistics:")
print(f"  Mean: {expression_data.values.mean():.3f}")
print(f"  Std: {expression_data.values.std():.3f}")
print(f"  Min: {expression_data.values.min():.3f}")
print(f"  Max: {expression_data.values.max():.3f}")

# %% Missing Data Analysis
"""Check for missing values"""

print("\n" + "="*60)
print("MISSING DATA ANALYSIS")
print("="*60)

missing_expr = expression_data.isna().sum()
missing_meta = metadata.isna().sum()

print(f"\nMissing in expression: {missing_expr.sum()}")
print(f"Missing in metadata:\n{missing_meta}")

# Heatmap of missing data by sample and gene
if (missing_expr > 0).any():
    missing_genes = missing_expr[missing_expr > 0].index
    missing_samples = expression_data[missing_genes].isna().any(axis=1)

    fig, ax = plt.subplots(figsize=(12, 6))
    missing_matrix = expression_data.loc[missing_samples, missing_genes].isna().astype(int)
    sns.heatmap(missing_matrix, cbar_kws={'label': 'Missing'}, ax=ax, cmap='RdYlGn_r')
    ax.set_title('Missing Data Heatmap', fontweight='bold')
    plt.tight_layout()
else:
    print("No missing data detected!")

# %% Kaplan-Meier by ISS
"""Survival curves stratified by ISS stage"""

fig, ax = plt.subplots(figsize=(8, 6))
plot_km_curve(
    metadata['survival_time'].values,
    metadata['event'].values,
    metadata['ISS'].values,
    ax=ax,
    group_labels={
        'ISS-I': 'ISS-I (Standard)',
        'ISS-II': 'ISS-II (Intermediate)',
        'ISS-III': 'ISS-III (High)',
    },
    palette='iss',
    title='Kaplan-Meier: ISS Stage',
    xlabel='Time (months)',
    ylabel='Overall Survival',
    risk_table=True,
)
plt.tight_layout()

# %% Kaplan-Meier by Cytogenetics
"""Survival curves by cytogenetic risk"""

fig, ax = plt.subplots(figsize=(8, 6))
plot_km_curve(
    metadata['survival_time'].values,
    metadata['event'].values,
    metadata['cytogenetics'].values,
    ax=ax,
    palette='cytogenetics',
    title='Kaplan-Meier: Cytogenetic Risk',
    xlabel='Time (months)',
    ylabel='Overall Survival',
    risk_table=True,
)
plt.tight_layout()

# %% Kaplan-Meier by Study
"""Survival curves by study to assess study effects"""

fig, ax = plt.subplots(figsize=(8, 6))
plot_km_curve(
    metadata['survival_time'].values,
    metadata['event'].values,
    metadata['study'].values,
    ax=ax,
    palette='study',
    title='Kaplan-Meier: By Study',
    xlabel='Time (months)',
    ylabel='Overall Survival',
    risk_table=True,
)
plt.tight_layout()

# %% Survival Distribution by Study
"""Distribution of survival times"""

data_by_study = {
    study: metadata[metadata['study'] == study]['survival_time'].values
    for study in metadata['study'].unique()
}

fig, ax = plt.subplots(figsize=(10, 6))
plot_distributions(
    data_by_study,
    ax=ax,
    title='Survival Time Distribution by Study',
    xlabel='Time (months)',
    ylabel='Density',
    palette='study',
)
plt.tight_layout()

# %% Expression Distribution by Study
"""Expression-level distributions"""

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Overall distribution
ax1.hist(expression_data.values.flatten(), bins=100, alpha=0.7, color='#3498db',
        edgecolor='black')
ax1.set_xlabel('Expression (log2 scale)')
ax1.set_ylabel('Frequency')
ax1.set_title('Overall Expression Distribution')
ax1.grid(True, alpha=0.2)

# By study
for study in metadata['study'].unique():
    study_samples = metadata[metadata['study'] == study].index
    ax2.hist(expression_data.loc[study_samples].values.flatten(),
            bins=100, alpha=0.3, label=study)
ax2.set_xlabel('Expression (log2 scale)')
ax2.set_ylabel('Frequency')
ax2.set_title('Expression Distribution by Study')
ax2.legend()
ax2.grid(True, alpha=0.2)

plt.tight_layout()

# %% PCA by Study
"""Principal component analysis colored by study"""

# Normalize
scaler = StandardScaler()
expr_normalized = scaler.fit_transform(expression_data)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(expr_normalized)

print(f"\nPCA explained variance:")
print(f"  PC1: {pca.explained_variance_ratio_[0]:.1%}")
print(f"  PC2: {pca.explained_variance_ratio_[1]:.1%}")

fig, ax = plt.subplots(figsize=(8, 8))
plot_pca_biplot(
    X_pca,
    pca.components_[:, :2],
    labels=metadata['study'].values,
    ax=ax,
    palette='study',
    title='PCA: By Study',
    feature_scale=0.3,
)
plt.tight_layout()

# %% PCA by ISS
"""PCA colored by ISS stage"""

fig, ax = plt.subplots(figsize=(8, 8))
plot_pca_biplot(
    X_pca,
    pca.components_[:, :2],
    labels=metadata['ISS'].values,
    ax=ax,
    palette='iss',
    title='PCA: By ISS Stage',
    feature_scale=0.3,
)
plt.tight_layout()

# %% Correlation Analysis
"""Gene-gene correlation structure"""

# Sample subset for visualization
sample_genes = np.random.choice(expression_data.columns, 30, replace=False)
corr_matrix = expression_data[sample_genes].corr()

fig, ax = plt.subplots(figsize=(10, 9))
plot_heatmap(
    corr_matrix,
    ax=ax,
    cmap='coolwarm',
    title='Gene-Gene Correlation (30 random genes)',
    cbar_label='Pearson r',
    vmin=-1, vmax=1,
    dendogram=False,
)
plt.tight_layout()

# %% Summary Statistics Table
"""Clinical variable summary"""

summary_stats = pd.DataFrame({
    'Variable': ['Age', 'ISS-I', 'ISS-II', 'ISS-III', 'Standard CG', 'Intermediate CG', 'High-risk CG'],
    'N': [
        n_samples,
        (metadata['ISS'] == 'ISS-I').sum(),
        (metadata['ISS'] == 'ISS-II').sum(),
        (metadata['ISS'] == 'ISS-III').sum(),
        (metadata['cytogenetics'] == 'Standard').sum(),
        (metadata['cytogenetics'] == 'Intermediate').sum(),
        (metadata['cytogenetics'] == 'High-risk').sum(),
    ],
    'Percent': [
        '100%',
        f"{(metadata['ISS'] == 'ISS-I').sum() / n_samples * 100:.1f}%",
        f"{(metadata['ISS'] == 'ISS-II').sum() / n_samples * 100:.1f}%",
        f"{(metadata['ISS'] == 'ISS-III').sum() / n_samples * 100:.1f}%",
        f"{(metadata['cytogenetics'] == 'Standard').sum() / n_samples * 100:.1f}%",
        f"{(metadata['cytogenetics'] == 'Intermediate').sum() / n_samples * 100:.1f}%",
        f"{(metadata['cytogenetics'] == 'High-risk').sum() / n_samples * 100:.1f}%",
    ]
})

print("\n" + "="*60)
print("CLINICAL CHARACTERISTICS")
print("="*60)
print(summary_stats.to_string(index=False))

# %% Clinical Associations
"""Survival by clinical variables"""

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# ISS
plot_km_curve(
    metadata['survival_time'].values,
    metadata['event'].values,
    metadata['ISS'].values,
    ax=axes[0],
    palette='iss',
    title='ISS Stage',
)

# Cytogenetics
plot_km_curve(
    metadata['survival_time'].values,
    metadata['event'].values,
    metadata['cytogenetics'].values,
    ax=axes[1],
    palette='cytogenetics',
    title='Cytogenetics',
)

# Study
plot_km_curve(
    metadata['survival_time'].values,
    metadata['event'].values,
    metadata['study'].values,
    ax=axes[2],
    palette='study',
    title='Study',
)

plt.tight_layout()

# %% Report Generation
"""Generate HTML report with all figures and tables"""

report = ReportGenerator(output_dir='reports', title='Data Exploration Report')

# Add metrics
report.add_metric('Total Samples', n_samples, section='Summary')
report.add_metric('Total Genes', n_genes, section='Summary')
report.add_metric('Studies', metadata['study'].nunique(), section='Summary')
report.add_metric('Median Follow-up (months)', f"{metadata['survival_time'].median():.1f}", section='Summary')
report.add_metric('Event Rate', f"{metadata['event'].mean():.1%}", section='Summary')

# Add tables
report.add_table('Clinical Characteristics', summary_stats, section='Summary')

report.add_table(
    'Survival Statistics',
    pd.DataFrame({
        'Metric': ['Median Follow-up', 'Events', 'Censored', 'Event Rate'],
        'Value': [
            f"{metadata['survival_time'].median():.1f} months",
            int(metadata['event'].sum()),
            int((metadata['event'] == 0).sum()),
            f"{metadata['event'].mean():.1%}"
        ]
    }),
    section='Summary'
)

# Add figures would go here with: report.add_figure(name, fig, section)

print("\n" + "="*60)
print("NOTEBOOK COMPLETE")
print("="*60)
print("Figures and tables generated successfully!")
