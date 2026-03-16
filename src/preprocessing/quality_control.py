"""
Quality Control
===============
Sample QC: PCA outliers, Mahalanobis distance, batch effect visualization.
Pathway distribution QC. Missing data analysis.
HTML QC report generation.

Author: PhD Researcher 2
Date: 2026
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import json

try:
    from scipy.spatial.distance import mahalanobis, cdist
    from scipy.stats import chi2
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    raise ImportError("scipy and matplotlib not installed.")

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class QualityController:
    """Quality control and outlier detection."""

    def __init__(self, output_dir: Path = Path("./qc_reports")):
        """
        Initialize quality controller.

        Parameters
        ----------
        output_dir : Path
            Directory for QC report output.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.qc_results = {}
        self.outliers = []

    def detect_pca_outliers(
        self, expression_df: pd.DataFrame, n_pcs: int = 10, threshold_sd: float = 3.0
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Detect outlier samples via PCA.

        Parameters
        ----------
        expression_df : pd.DataFrame
            Expression matrix (samples × genes or pathways).
        n_pcs : int
            Number of principal components to compute.
        threshold_sd : float
            Z-score threshold for outlier detection.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, Dict]
            (pca_scores, outlier_flags, stats)
        """
        logger.info("Detecting PCA outliers...")

        try:
            from sklearn.decomposition import PCA
            from scipy.stats import zscore
        except ImportError:
            raise ImportError("scikit-learn not installed. Run: pip install scikit-learn")

        # Standardize
        expr_scaled = (expression_df - expression_df.mean(axis=0)) / (expression_df.std(axis=0) + 1e-8)

        # PCA
        n_pcs_actual = min(n_pcs, min(expression_df.shape) - 1)
        pca = PCA(n_components=n_pcs_actual)
        pca_scores = pca.fit_transform(expr_scaled)

        # Detect outliers via Mahalanobis distance
        outlier_flags = np.zeros(pca_scores.shape[0], dtype=bool)
        outlier_indices = []

        try:
            # Compute Mahalanobis distance
            mean = pca_scores.mean(axis=0)
            cov = np.cov(pca_scores.T)
            cov_inv = np.linalg.pinv(cov)

            mahal_dist = np.array([
                np.sqrt(np.dot(np.dot((row - mean), cov_inv), (row - mean).T))
                for row in pca_scores
            ])

            # Chi-square threshold (df = n_pcs)
            threshold = np.sqrt(chi2.ppf(0.95, n_pcs_actual))
            outlier_flags = mahal_dist > threshold
            outlier_indices = np.where(outlier_flags)[0].tolist()
        except Exception as e:
            logger.warning(f"Mahalanobis distance computation failed: {e}. Using Z-score instead.")
            # Fallback: Z-score on PCs
            pc_z = np.abs(zscore(pca_scores, axis=0))
            outlier_flags = (pc_z > threshold_sd).any(axis=1)
            outlier_indices = np.where(outlier_flags)[0].tolist()

        stats = {
            "n_pcs": n_pcs_actual,
            "variance_explained": pca.explained_variance_ratio_.tolist() if hasattr(pca, 'explained_variance_ratio_') else [],
            "n_outliers": outlier_flags.sum(),
            "outlier_rate": float(outlier_flags.sum() / len(outlier_flags)),
            "outlier_indices": outlier_indices,
        }

        logger.info(f"Detected {stats['n_outliers']} outliers ({stats['outlier_rate']*100:.1f}%)")
        return pca_scores, outlier_flags, stats

    def plot_pca(
        self, pca_scores: np.ndarray, outlier_flags: np.ndarray,
        sample_ids: List[str], metadata: Optional[pd.DataFrame] = None
    ) -> Path:
        """
        Plot PCA with outliers highlighted.

        Parameters
        ----------
        pca_scores : np.ndarray
            PCA scores (samples × PCs).
        outlier_flags : np.ndarray
            Boolean array of outliers.
        sample_ids : List[str]
            Sample identifiers.
        metadata : Optional[pd.DataFrame]
            Sample metadata for coloring.

        Returns
        -------
        Path
            Path to saved plot.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # PC1 vs PC2
        ax = axes[0]
        normal_mask = ~outlier_flags
        ax.scatter(pca_scores[normal_mask, 0], pca_scores[normal_mask, 1],
                  alpha=0.6, s=50, label="Normal", color="blue")
        ax.scatter(pca_scores[outlier_flags, 0], pca_scores[outlier_flags, 1],
                  alpha=0.8, s=100, label="Outlier", color="red", marker="X")
        ax.set_xlabel(f"PC1")
        ax.set_ylabel(f"PC2")
        ax.set_title("PCA: PC1 vs PC2")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # PC1 vs PC3
        if pca_scores.shape[1] >= 3:
            ax = axes[1]
            ax.scatter(pca_scores[normal_mask, 0], pca_scores[normal_mask, 2],
                      alpha=0.6, s=50, label="Normal", color="blue")
            ax.scatter(pca_scores[outlier_flags, 0], pca_scores[outlier_flags, 2],
                      alpha=0.8, s=100, label="Outlier", color="red", marker="X")
            ax.set_xlabel(f"PC1")
            ax.set_ylabel(f"PC3")
            ax.set_title("PCA: PC1 vs PC3")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = self.output_dir / "pca_outliers.png"
        plt.savefig(plot_path, dpi=100)
        plt.close()
        logger.info(f"Saved PCA plot: {plot_path}")
        return plot_path

    def analyze_batch_effects(
        self, expression_df: pd.DataFrame, batch_column: str, metadata: pd.DataFrame
    ) -> Dict:
        """
        Analyze batch effects.

        Parameters
        ----------
        expression_df : pd.DataFrame
            Expression matrix (samples × features).
        batch_column : str
            Column name in metadata for batch labels.
        metadata : pd.DataFrame
            Sample metadata with batch information.

        Returns
        -------
        Dict
            Batch effect analysis results.
        """
        logger.info("Analyzing batch effects...")

        if batch_column not in metadata.columns:
            logger.warning(f"Batch column {batch_column} not found in metadata.")
            return {}

        batch_labels = metadata[batch_column].values
        unique_batches = np.unique(batch_labels)

        # Batch-wise statistics
        batch_stats = {}
        for batch in unique_batches:
            batch_mask = batch_labels == batch
            batch_expr = expression_df[batch_mask]

            batch_stats[str(batch)] = {
                "n_samples": batch_mask.sum(),
                "mean_expr": float(batch_expr.mean().mean()),
                "std_expr": float(batch_expr.std().mean()),
                "median_expr": float(batch_expr.median().mean()),
            }

        # Compute batch variance (proportion of variance explained by batch)
        try:
            from sklearn.decomposition import PCA
            expr_scaled = (expression_df - expression_df.mean(axis=0)) / (expression_df.std(axis=0) + 1e-8)
            pca = PCA(n_components=min(10, min(expression_df.shape) - 1))
            pca_scores = pca.fit_transform(expr_scaled)

            # Correlation of first PC with batch
            batch_numeric = pd.factorize(batch_labels)[0]
            batch_corr = np.corrcoef(pca_scores[:, 0], batch_numeric)[0, 1]

            batch_stats["batch_variance_pc1"] = float(np.abs(batch_corr))
        except Exception as e:
            logger.warning(f"Batch variance computation failed: {e}")

        logger.info(f"Batch effects: {len(unique_batches)} batches detected")
        return {"batches": batch_stats}

    def analyze_missing_data(self, expression_df: pd.DataFrame) -> Dict:
        """
        Analyze missing data patterns.

        Parameters
        ----------
        expression_df : pd.DataFrame
            Expression matrix.

        Returns
        -------
        Dict
            Missing data statistics.
        """
        logger.info("Analyzing missing data...")

        total_values = expression_df.shape[0] * expression_df.shape[1]
        missing_count = expression_df.isna().sum().sum()
        missing_pct = 100 * missing_count / total_values if total_values > 0 else 0

        # Per-sample missing rate
        sample_missing = expression_df.isna().sum(axis=0) / expression_df.shape[0] * 100
        feature_missing = expression_df.isna().sum(axis=1) / expression_df.shape[1] * 100

        stats = {
            "total_missing": int(missing_count),
            "missing_pct": float(missing_pct),
            "sample_missing_mean": float(sample_missing.mean()),
            "sample_missing_max": float(sample_missing.max()),
            "feature_missing_mean": float(feature_missing.mean()),
            "feature_missing_max": float(feature_missing.max()),
        }

        if missing_pct > 0:
            logger.warning(f"Missing data detected: {missing_pct:.2f}%")

        return stats

    def plot_feature_distributions(
        self, expression_df: pd.DataFrame, n_features: int = 20
    ) -> Path:
        """
        Plot distributions of features.

        Parameters
        ----------
        expression_df : pd.DataFrame
            Expression matrix.
        n_features : int
            Number of features to plot.

        Returns
        -------
        Path
            Path to saved plot.
        """
        fig, axes = plt.subplots(4, 5, figsize=(15, 12))
        axes = axes.flatten()

        features_to_plot = expression_df.columns[:n_features]
        for idx, feature in enumerate(features_to_plot):
            ax = axes[idx]
            ax.hist(expression_df[feature].dropna(), bins=30, alpha=0.7, edgecolor='black')
            ax.set_title(feature[:20])  # Truncate long names
            ax.set_ylabel("Frequency")

        # Hide unused subplots
        for idx in range(len(features_to_plot), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plot_path = self.output_dir / "feature_distributions.png"
        plt.savefig(plot_path, dpi=100)
        plt.close()
        logger.info(f"Saved feature distribution plot: {plot_path}")
        return plot_path

    def generate_qc_report(
        self,
        expression_df: pd.DataFrame,
        metadata: Optional[pd.DataFrame] = None,
        dataset_id: str = "unknown",
        batch_column: Optional[str] = None,
    ) -> Path:
        """
        Generate comprehensive HTML QC report.

        Parameters
        ----------
        expression_df : pd.DataFrame
            Expression matrix.
        metadata : Optional[pd.DataFrame]
            Sample metadata.
        dataset_id : str
            Dataset identifier.
        batch_column : Optional[str]
            Batch column name in metadata.

        Returns
        -------
        Path
            Path to HTML report.
        """
        logger.info(f"Generating QC report for {dataset_id}...")

        # Run analyses
        pca_scores, outlier_flags, pca_stats = self.detect_pca_outliers(expression_df)
        missing_stats = self.analyze_missing_data(expression_df)

        batch_stats = {}
        if metadata is not None and batch_column:
            batch_stats = self.analyze_batch_effects(expression_df, batch_column, metadata)

        # Generate plots
        pca_plot = self.plot_pca(pca_scores, outlier_flags, expression_df.index.tolist(), metadata)
        feat_plot = self.plot_feature_distributions(expression_df, n_features=20)

        # Create HTML report
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>QC Report: {dataset_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; border-bottom: 2px solid #ddd; padding-bottom: 5px; }}
        .stats {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        img {{ max-width: 100%; height: auto; margin: 20px 0; }}
        .warning {{ color: #d32f2f; font-weight: bold; }}
        .success {{ color: #388e3c; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Quality Control Report: {dataset_id}</h1>
    <p>Generated: {datetime.now().isoformat()}</p>

    <h2>Dataset Overview</h2>
    <div class="stats">
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Samples</td><td>{expression_df.shape[0]}</td></tr>
            <tr><td>Features (genes/pathways)</td><td>{expression_df.shape[1]}</td></tr>
        </table>
    </div>

    <h2>Missing Data Analysis</h2>
    <div class="stats">
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Missing Values</td><td>{missing_stats.get('total_missing', 0)}</td></tr>
            <tr><td>Missing Percentage</td><td>{missing_stats.get('missing_pct', 0):.2f}%</td></tr>
            <tr><td>Sample Missing Mean</td><td>{missing_stats.get('sample_missing_mean', 0):.2f}%</td></tr>
            <tr><td>Feature Missing Mean</td><td>{missing_stats.get('feature_missing_mean', 0):.2f}%</td></tr>
        </table>
    </div>

    <h2>PCA-based Outlier Detection</h2>
    <div class="stats">
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Principal Components</td><td>{pca_stats.get('n_pcs', 0)}</td></tr>
            <tr><td>Outliers Detected</td><td>{pca_stats.get('n_outliers', 0)}</td></tr>
            <tr><td>Outlier Rate</td><td>{pca_stats.get('outlier_rate', 0)*100:.2f}%</td></tr>
        </table>
    </div>

    <h3>PCA Plot</h3>
    <img src="./{pca_plot.name}" alt="PCA Plot" style="max-width:100%">

    <h2>Feature Distributions</h2>
    <img src="./{feat_plot.name}" alt="Feature Distributions" style="max-width:100%">

    <h2>Conclusion</h2>
    <p>
        {'<span class="warning">WARNING: Outliers detected. Consider sample review.</span>'
         if pca_stats.get('outlier_rate', 0) > 0.1
         else '<span class="success">No major quality issues detected.</span>'}
    </p>
</body>
</html>
        """

        # Save report
        report_path = self.output_dir / f"qc_report_{dataset_id}.html"
        with open(report_path, "w") as f:
            f.write(html_content)

        logger.info(f"Saved QC report: {report_path}")

        # Save summary as JSON
        summary = {
            "dataset_id": dataset_id,
            "timestamp": datetime.now().isoformat(),
            "pca_stats": pca_stats,
            "missing_stats": missing_stats,
            "batch_stats": batch_stats,
        }

        summary_path = self.output_dir / f"qc_summary_{dataset_id}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        return report_path

    def identify_outlier_samples(self, outlier_flags: np.ndarray, sample_ids: List[str]) -> List[str]:
        """
        Identify outlier sample IDs.

        Parameters
        ----------
        outlier_flags : np.ndarray
            Boolean array of outlier flags.
        sample_ids : List[str]
            Sample identifiers.

        Returns
        -------
        List[str]
            List of outlier sample IDs.
        """
        outlier_samples = [sid for sid, is_outlier in zip(sample_ids, outlier_flags) if is_outlier]
        return outlier_samples


def main():
    """Example usage."""
    logger.info("Quality controller initialized.")


if __name__ == "__main__":
    main()
