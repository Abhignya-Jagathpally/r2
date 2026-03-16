"""
Visualization utilities for MM risk-signature pipeline.
Includes KM curves, forest plots, calibration plots, publication-ready themes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy.stats import gaussian_kde
from lifelines import KaplanMeierFitter
from sklearn.metrics import auc
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# PUBLICATION THEMES & PALETTES
# ============================================================================

def set_publication_theme(style='nature', figsize=(8, 6), dpi=300):
    """
    Set matplotlib rcParams for Nature/Cell-style publication figures.

    Parameters
    ----------
    style : str
        'nature' (single column ~3.4") or 'cell' (wide ~7")
    figsize : tuple
        Figure size in inches
    dpi : int
        DPI for rasterized elements

    Returns
    -------
    dict
        rcParams dict for reference
    """
    if style == 'nature':
        width, height = 3.4, 2.8
    elif style == 'cell':
        width, height = 7.0, 5.0
    else:
        width, height = figsize

    params = {
        'figure.figsize': (width, height),
        'figure.dpi': dpi,
        'savefig.dpi': dpi,
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'axes.linewidth': 0.8,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.width': 0.8,
        'xtick.major.size': 4,
        'ytick.major.width': 0.8,
        'ytick.major.size': 4,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica'],
    }

    plt.rcParams.update(params)
    return params


def get_mm_palette(palette_type='risk'):
    """
    Get publication-ready color palettes for MM analysis.

    Parameters
    ----------
    palette_type : str
        'risk' (low/high), 'iss' (3-class), 'cytogenetics' (5-class),
        'study' (continuous), 'pathways' (diverging)

    Returns
    -------
    dict or list
        Color palette
    """
    palettes = {
        'risk': {'Low': '#2ecc71', 'High': '#e74c3c'},
        'iss': {
            'ISS-I': '#3498db',
            'ISS-II': '#f39c12',
            'ISS-III': '#e74c3c'
        },
        'cytogenetics': {
            'Standard': '#3498db',
            'Intermediate': '#95a5a6',
            'High-risk': '#e74c3c',
            't(4;14)': '#9b59b6',
            'del(17p)': '#c0392b'
        },
        'study': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f'],
        'pathways': sns.diverging_palette(250, 10, as_cmap=False, n_colors=11),
    }

    return palettes.get(palette_type, palette_type)


# ============================================================================
# KAPLAN-MEIER CURVES
# ============================================================================

def plot_km_curve(
    durations,
    event_observed,
    groups,
    ax=None,
    figsize=(8, 6),
    group_labels=None,
    palette='risk',
    title='Kaplan-Meier Curve',
    xlabel='Time (months)',
    ylabel='Probability of Survival',
    ci_show=True,
    risk_table=True,
    at_risk_rows=3,
    fontsize=9,
):
    """
    Plot Kaplan-Meier survival curves with risk table.

    Parameters
    ----------
    durations : array-like
        Time to event or censoring
    event_observed : array-like
        Event indicator (1=event, 0=censored)
    groups : array-like
        Group assignment
    ax : matplotlib axis, optional
    figsize : tuple
        Figure size if ax is None
    group_labels : dict
        Mapping of group values to display names
    palette : str or dict
        Color palette
    title : str
    xlabel, ylabel : str
    ci_show : bool
        Show 95% confidence intervals
    risk_table : bool
        Show number at risk table below
    at_risk_rows : int
        Number of rows in risk table
    fontsize : int

    Returns
    -------
    fig, ax : matplotlib objects
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if isinstance(palette, str):
        palette = get_mm_palette(palette)

    unique_groups = sorted(set(groups))
    group_labels = group_labels or {g: str(g) for g in unique_groups}

    kmf = KaplanMeierFitter()
    colors = [palette.get(g, f'C{i}') for i, g in enumerate(unique_groups)]

    ax_main = ax
    if risk_table:
        ax.set_position([0.12, 0.25, 0.8, 0.7])

    for i, group in enumerate(unique_groups):
        mask = np.array(groups) == group
        kmf.fit(
            durations[mask],
            event_observed[mask],
            label=group_labels[group]
        )
        kmf.plot_survival_function(
            ax=ax_main,
            ci_show=ci_show,
            linewidth=2,
            color=colors[i],
            alpha=0.8
        )

    ax_main.set_xlabel(xlabel, fontsize=fontsize)
    ax_main.set_ylabel(ylabel, fontsize=fontsize)
    ax_main.set_title(title, fontsize=fontsize + 1, fontweight='bold')
    ax_main.legend(loc='best', fontsize=fontsize - 1)
    ax_main.grid(True, alpha=0.2, linestyle='--')
    ax_main.set_ylim([0, 1.05])

    # Risk table
    if risk_table:
        ax_risk = fig.add_axes([0.12, 0.05, 0.8, 0.18])

        time_points = np.percentile(durations[event_observed == 1],
                                    np.linspace(0, 100, at_risk_rows))

        risk_data = []
        for group in unique_groups:
            mask = np.array(groups) == group
            n_at_risk = []
            for tp in time_points:
                n = np.sum((np.array(durations)[mask] >= tp) &
                          np.isfinite(np.array(durations)[mask]))
                n_at_risk.append(n)
            risk_data.append(n_at_risk)

        y_pos = np.arange(len(unique_groups))
        ax_risk.invert_yaxis()

        for i, group in enumerate(unique_groups):
            label = f"{group_labels[group]} ({risk_data[i][0]})"
            ax_risk.text(-0.02, i, label, ha='right', fontsize=fontsize - 1)

        for j, tp in enumerate(time_points):
            ax_risk.axvline(tp, color='gray', linestyle='--', alpha=0.3)
            for i, group in enumerate(unique_groups):
                ax_risk.text(tp, i, str(risk_data[i][j]),
                           ha='center', fontsize=fontsize - 1)

        ax_risk.set_xlim(ax_main.get_xlim())
        ax_risk.set_xticks([])
        ax_risk.set_yticks([])
        ax_risk.spines['left'].set_visible(False)
        ax_risk.spines['bottom'].set_visible(False)
        ax_risk.spines['top'].set_visible(False)
        ax_risk.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig, ax_main


# ============================================================================
# FOREST PLOTS
# ============================================================================

def plot_forest(
    models,
    c_indices,
    ci_lower,
    ci_upper,
    ax=None,
    figsize=(10, 6),
    reference_line=0.5,
    xlabel='C-index',
    title='Model Comparison: Concordance Index',
    fontsize=10,
    marker_size=8,
):
    """
    Plot forest plot for model comparison.

    Parameters
    ----------
    models : list
        Model names
    c_indices : array-like
        Point estimates (C-indices)
    ci_lower, ci_upper : array-like
        95% confidence interval bounds
    ax : matplotlib axis, optional
    figsize : tuple
    reference_line : float
        Reference value (typically 0.5 for random)
    xlabel : str
    title : str
    fontsize : int
    marker_size : int

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    y_pos = np.arange(len(models))

    # Plot error bars
    ax.errorbar(
        c_indices, y_pos,
        xerr=[np.array(c_indices) - np.array(ci_lower),
              np.array(ci_upper) - np.array(c_indices)],
        fmt='o',
        markersize=marker_size,
        linewidth=2,
        capsize=5,
        capthick=2,
        color='#2c3e50',
        ecolor='#34495e',
        alpha=0.8
    )

    # Reference line
    ax.axvline(reference_line, color='red', linestyle='--', linewidth=1.5,
               label=f'Random (C={reference_line})', alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize + 1, fontweight='bold')
    ax.set_xlim([0.4, 0.8])
    ax.grid(True, axis='x', alpha=0.2, linestyle='--')
    ax.legend(loc='best', fontsize=fontsize - 1)

    plt.tight_layout()
    return fig, ax


# ============================================================================
# CALIBRATION PLOTS
# ============================================================================

def plot_calibration(
    y_true,
    y_pred,
    n_bins=10,
    ax=None,
    figsize=(7, 7),
    title='Calibration Plot',
    fontsize=10,
):
    """
    Plot calibration curve (predicted vs observed event rate).

    Parameters
    ----------
    y_true : array-like
        Binary outcome
    y_pred : array-like
        Predicted probabilities [0, 1]
    n_bins : int
        Number of probability bins
    ax : matplotlib axis, optional
    figsize : tuple
    title : str
    fontsize : int

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Binning
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    obs_rate = []
    pred_rate = []
    bin_counts = []

    for i in range(len(bins) - 1):
        mask = (y_pred >= bins[i]) & (y_pred < bins[i + 1])
        if mask.sum() > 0:
            obs_rate.append(y_true[mask].mean())
            pred_rate.append(y_pred[mask].mean())
            bin_counts.append(mask.sum())

    obs_rate = np.array(obs_rate)
    pred_rate = np.array(pred_rate)
    bin_counts = np.array(bin_counts)

    # Plot
    ax.scatter(pred_rate, obs_rate, s=bin_counts * 2, alpha=0.6,
              color='#3498db', edgecolors='black', linewidth=1)

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect calibration')

    # Marginal calibration
    ax.axhline(y_true.mean(), color='gray', linestyle='--', linewidth=1,
              alpha=0.7, label=f'Observed rate: {y_true.mean():.2f}')

    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('Predicted Probability', fontsize=fontsize)
    ax.set_ylabel('Observed Event Rate', fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize + 1, fontweight='bold')
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.legend(loc='best', fontsize=fontsize - 1)
    ax.set_aspect('equal')

    plt.tight_layout()
    return fig, ax


# ============================================================================
# DECISION CURVE ANALYSIS
# ============================================================================

def plot_dca(
    y_true,
    y_pred,
    ax=None,
    figsize=(8, 6),
    title='Decision Curve Analysis',
    fontsize=10,
):
    """
    Plot decision curve analysis.

    Parameters
    ----------
    y_true : array-like
        Binary outcome
    y_pred : array-like
        Predicted probabilities [0, 1]
    ax : matplotlib axis, optional
    figsize : tuple
    title : str
    fontsize : int

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    thresholds = np.linspace(0, 1, 101)
    n_events = (y_true == 1).sum()
    n_total = len(y_true)

    model_benefit = []
    all_positive_benefit = []

    for thresh in thresholds:
        # Model
        tp = ((y_pred >= thresh) & (y_true == 1)).sum()
        fp = ((y_pred >= thresh) & (y_true == 0)).sum()

        # Net benefit
        if (tp + fp) > 0:
            nb = (tp / n_total) - ((fp / n_total) * (thresh / (1 - thresh)))
        else:
            nb = 0
        model_benefit.append(nb)

        # All positive (treat all)
        nb_all = (n_events / n_total) - ((n_total - n_events) / n_total) * (thresh / (1 - thresh))
        all_positive_benefit.append(nb_all)

    model_benefit = np.array(model_benefit)
    all_positive_benefit = np.array(all_positive_benefit)

    ax.plot(thresholds, model_benefit, linewidth=2.5, label='Model',
           color='#3498db')
    ax.plot(thresholds, all_positive_benefit, linewidth=2, linestyle='--',
           label='Treat all', color='gray', alpha=0.7)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

    ax.fill_between(thresholds, 0, model_benefit, alpha=0.2, color='#3498db')
    ax.fill_between(thresholds, 0, all_positive_benefit, alpha=0.1, color='gray')

    ax.set_xlabel('Threshold Probability', fontsize=fontsize)
    ax.set_ylabel('Net Benefit', fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize + 1, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.legend(loc='best', fontsize=fontsize - 1)

    plt.tight_layout()
    return fig, ax


# ============================================================================
# HEATMAPS
# ============================================================================

def plot_heatmap(
    data,
    ax=None,
    figsize=(10, 8),
    cmap='RdBu_r',
    title='Heatmap',
    xlabel='',
    ylabel='',
    cbar_label='',
    vmin=None,
    vmax=None,
    fontsize=9,
    dendogram=False,
):
    """
    Plot annotated heatmap with optional clustering.

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        Data matrix
    ax : matplotlib axis, optional
    figsize : tuple
    cmap : str
    title : str
    xlabel, ylabel : str
    cbar_label : str
    vmin, vmax : float
        Color scale limits
    fontsize : int
    dendogram : bool
        Perform hierarchical clustering

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Clustering
    if dendogram and isinstance(data, pd.DataFrame):
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import pdist

        Z = linkage(pdist(data.values, metric='euclidean'), method='ward')
        dendro = dendrogram(Z, no_plot=True)
        row_order = dendro['leaves']

        Z_col = linkage(pdist(data.values.T, metric='euclidean'), method='ward')
        dendro_col = dendrogram(Z_col, no_plot=True)
        col_order = dendro_col['leaves']

        data_clustered = data.iloc[row_order, col_order]
    else:
        data_clustered = data

    # Plot
    if isinstance(data_clustered, pd.DataFrame):
        sns.heatmap(
            data_clustered,
            ax=ax,
            cmap=cmap,
            cbar_kws={'label': cbar_label},
            vmin=vmin,
            vmax=vmax,
            square=True,
            linewidths=0.5,
            linecolor='white',
            xticklabels=True,
            yticklabels=True,
        )
    else:
        sns.heatmap(
            data_clustered,
            ax=ax,
            cmap=cmap,
            cbar_kws={'label': cbar_label},
            vmin=vmin,
            vmax=vmax,
            square=True,
            linewidths=0.5,
            linecolor='white',
        )

    ax.set_title(title, fontsize=fontsize + 1, fontweight='bold', pad=10)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=fontsize - 1)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=fontsize - 1)

    plt.tight_layout()
    return fig, ax


# ============================================================================
# PCA PLOTS
# ============================================================================

def plot_pca_biplot(
    X_pca,
    components,
    labels=None,
    ax=None,
    figsize=(8, 8),
    palette='study',
    title='PCA Biplot',
    fontsize=10,
    feature_scale=0.4,
):
    """
    Plot PCA biplot with feature vectors.

    Parameters
    ----------
    X_pca : array, shape (n_samples, 2)
        PCA-transformed data (PC1, PC2)
    components : array, shape (n_features, 2)
        PCA loadings
    labels : array-like, optional
        Sample class labels
    ax : matplotlib axis, optional
    figsize : tuple
    palette : str or dict
    title : str
    fontsize : int
    feature_scale : float
        Scaling for feature vectors

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if isinstance(palette, str):
        palette = get_mm_palette(palette)

    # Sample points
    if labels is not None:
        unique_labels = sorted(set(labels))
        colors = palette if isinstance(palette, list) else \
                [palette.get(l, f'C{i}') for i, l in enumerate(unique_labels)]

        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                      s=50, alpha=0.6, label=str(label),
                      color=colors[i], edgecolors='black', linewidth=0.5)
    else:
        ax.scatter(X_pca[:, 0], X_pca[:, 1],
                  s=50, alpha=0.6, color='#3498db',
                  edgecolors='black', linewidth=0.5)

    # Feature vectors
    for i, (comp_x, comp_y) in enumerate(components):
        ax.arrow(0, 0, comp_x * feature_scale, comp_y * feature_scale,
                head_width=0.02, head_length=0.02, fc='red', ec='red',
                alpha=0.5, linewidth=1.5)

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('PC1', fontsize=fontsize)
    ax.set_ylabel('PC2', fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize + 1, fontweight='bold')
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.legend(loc='best', fontsize=fontsize - 1)

    plt.tight_layout()
    return fig, ax


# ============================================================================
# DISTRIBUTION PLOTS
# ============================================================================

def plot_distributions(
    data_dict,
    ax=None,
    figsize=(10, 6),
    title='Distribution Comparison',
    xlabel='Value',
    ylabel='Density',
    palette='study',
    fontsize=10,
):
    """
    Plot overlaid density distributions by group.

    Parameters
    ----------
    data_dict : dict
        {group_name: array of values}
    ax : matplotlib axis, optional
    figsize : tuple
    title : str
    xlabel, ylabel : str
    palette : str or dict
    fontsize : int

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if isinstance(palette, str):
        palette = get_mm_palette(palette)

    groups = list(data_dict.keys())
    colors = palette if isinstance(palette, list) else \
            [palette.get(g, f'C{i}') for i, g in enumerate(groups)]

    for group, color in zip(groups, colors):
        data = np.asarray(data_dict[group])
        data = data[np.isfinite(data)]

        if len(data) > 1:
            ax.hist(data, bins=30, alpha=0.3, color=color, label=group,
                   density=True, edgecolor='black', linewidth=0.5)

            # KDE
            try:
                kde = gaussian_kde(data)
                x_range = np.linspace(data.min(), data.max(), 200)
                ax.plot(x_range, kde(x_range), color=color, linewidth=2)
            except:
                pass

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize + 1, fontweight='bold')
    ax.legend(loc='best', fontsize=fontsize - 1)
    ax.grid(True, alpha=0.2, linestyle='--')

    plt.tight_layout()
    return fig, ax


# ============================================================================
# EXPORT UTILITIES
# ============================================================================

def export_figure(fig, filepath, dpi=300, formats=['pdf', 'png']):
    """
    Export figure in multiple formats at publication quality.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    filepath : str
        Base path without extension
    dpi : int
    formats : list
        File formats to export
    """
    for fmt in formats:
        full_path = f"{filepath}.{fmt}"
        fig.savefig(full_path, dpi=dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Saved: {full_path}")


# ============================================================================
# SHAP SUMMARY PLOT
# ============================================================================

def plot_shap_summary(
    shap_values,
    feature_names=None,
    ax=None,
    figsize=(10, 8),
    title='SHAP Summary Plot',
    max_display=20,
    fontsize=9,
):
    """
    Plot SHAP feature importance (bar plot style).

    Parameters
    ----------
    shap_values : array, shape (n_samples, n_features)
        SHAP values (mean absolute impact)
    feature_names : list, optional
    ax : matplotlib axis, optional
    figsize : tuple
    title : str
    max_display : int
    fontsize : int

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Mean absolute SHAP values
    if len(shap_values.shape) == 2:
        importance = np.abs(shap_values).mean(axis=0)
    else:
        importance = np.abs(shap_values)

    # Sort and truncate
    sorted_idx = np.argsort(importance)[::-1][:max_display]
    importance_sorted = importance[sorted_idx]

    if feature_names is not None:
        feature_names_sorted = [feature_names[i] for i in sorted_idx]
    else:
        feature_names_sorted = [f'Feature {i}' for i in sorted_idx]

    # Plot
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(importance_sorted)))
    ax.barh(range(len(importance_sorted)), importance_sorted, color=colors)
    ax.set_yticks(range(len(importance_sorted)))
    ax.set_yticklabels(feature_names_sorted, fontsize=fontsize)
    ax.set_xlabel('Mean |SHAP value|', fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize + 1, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.2, linestyle='--')

    plt.tight_layout()
    return fig, ax
