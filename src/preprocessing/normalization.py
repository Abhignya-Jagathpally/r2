"""
Expression Normalization
========================
Within-platform normalization:
- Microarrays: quantile normalization
- RNA-seq: TMM (edgeR) / voom transformation

Low-expression filtering, log2 transformation, QC plotting.
Freeze preprocessing contract (serialized parameters for reproducibility).

Author: PhD Researcher 2
Date: 2026
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from scipy.stats import zscore
    from scipy.spatial.distance import pdist, squareform
except ImportError:
    raise ImportError("scipy not installed. Run: pip install scipy")

try:
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    import rpy2.robjects.numpy2ri as numpy2ri
    numpy2ri.activate()
    RPY2_AVAILABLE = True
except ImportError:
    RPY2_AVAILABLE = False

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class NormalizationContract:
    """Frozen preprocessing contract for reproducibility (Karpathy autoresearch pattern)."""

    def __init__(self, contract_id: Optional[str] = None):
        """
        Initialize normalization contract.

        Parameters
        ----------
        contract_id : Optional[str]
            Unique identifier for this contract. If None, generated as timestamp.
        """
        self.contract_id = contract_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.contract_params = {}
        self.is_frozen = False

    def add_params(self, **kwargs) -> None:
        """
        Add parameters to contract.

        Parameters
        ----------
        **kwargs
            Parameter key-value pairs.
        """
        if self.is_frozen:
            raise RuntimeError("Contract is frozen. Cannot modify.")
        self.contract_params.update(kwargs)

    def freeze(self) -> None:
        """Freeze contract (prevent further modifications)."""
        self.is_frozen = True
        logger.info(f"Contract {self.contract_id} frozen.")

    def save(self, output_path: Path) -> None:
        """
        Serialize contract to disk.

        Parameters
        ----------
        output_path : Path
            Output pickle file path.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Contract saved: {output_path}")

    @staticmethod
    def load(input_path: Path) -> "NormalizationContract":
        """
        Load contract from disk.

        Parameters
        ----------
        input_path : Path
            Input pickle file path.

        Returns
        -------
        NormalizationContract
            Loaded contract.
        """
        with open(input_path, "rb") as f:
            contract = pickle.load(f)
        logger.info(f"Contract loaded: {input_path}")
        return contract

    def verify(self, new_params: Dict) -> bool:
        """
        Verify that preprocessing matches contract.

        Parameters
        ----------
        new_params : Dict
            Parameters to verify against contract.

        Returns
        -------
        bool
            True if parameters match contract.
        """
        for key, expected_value in self.contract_params.items():
            if key not in new_params or new_params[key] != expected_value:
                logger.warning(f"Contract mismatch for {key}: "
                              f"expected {expected_value}, got {new_params.get(key)}")
                return False
        return True


class ExpressionNormalizer:
    """Normalize expression matrices (microarray and RNA-seq)."""

    def __init__(self, contract: Optional[NormalizationContract] = None):
        """
        Initialize normalizer.

        Parameters
        ----------
        contract : Optional[NormalizationContract]
            Frozen preprocessing contract for reproducibility.
        """
        self.contract = contract or NormalizationContract()
        self.fitted_params = {}
        self.qc_plots = {}

    def quantile_normalize_array(
        self, expression_df: pd.DataFrame, target_quantiles: Optional[np.ndarray] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Quantile normalization for microarray data.

        Parameters
        ----------
        expression_df : pd.DataFrame
            Expression matrix (genes × samples).
        target_quantiles : Optional[np.ndarray]
            Pre-computed target quantiles. If None, uses all samples.

        Returns
        -------
        Tuple[pd.DataFrame, Dict]
            (normalized_df, fitted_params)
        """
        logger.info("Quantile normalizing expression matrix...")

        from sklearn.preprocessing import QuantileTransformer

        n_samples = expression_df.shape[1]
        qt = QuantileTransformer(
            n_quantiles=min(n_samples, 1000),
            output_distribution='normal',
            random_state=42,
        )

        # expression_df is genes(rows) x samples(cols).
        # QuantileTransformer expects samples(rows) x features(cols).
        # Transpose so each column = one gene, each row = one sample.
        # This normalizes each gene's distribution ACROSS samples (cross-sample quantile normalization).
        samples_by_genes = expression_df.T.values          # (n_samples, n_genes)
        normalized_sbg = qt.fit_transform(samples_by_genes) # (n_samples, n_genes)
        normalized_values = normalized_sbg.T                # (n_genes, n_samples)

        normalized_df = pd.DataFrame(
            normalized_values,
            index=expression_df.index,
            columns=expression_df.columns,
        )

        fitted_params = {"method": "quantile_transformer", "n_quantiles": qt.n_quantiles_}
        self.fitted_params["quantile_norm"] = fitted_params
        self.contract.add_params(normalization_method="quantile")

        logger.info(f"Quantile normalization complete.")
        return normalized_df, fitted_params

    def tmm_normalize_rnaseq(
        self, count_matrix: pd.DataFrame, use_voom: bool = True
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        TMM (trimmed mean of M-values) normalization for RNA-seq (via edgeR).
        Optionally apply voom transformation.

        Parameters
        ----------
        count_matrix : pd.DataFrame
            Count matrix (genes × samples).
        use_voom : bool
            Apply voom transformation after TMM (recommended).

        Returns
        -------
        Tuple[pd.DataFrame, Dict]
            (normalized_log_cpm, fitted_params)
        """
        logger.info("TMM normalizing RNA-seq counts...")

        # Try rpy2+edgeR first
        if RPY2_AVAILABLE:
            try:
                # Import R packages
                edger = importr("edgeR")
                limma = importr("limma")

                # Convert to R DataFrame
                r_counts = ro.convert2ri(count_matrix.values)
                r_genes = ro.StrVector(count_matrix.index.tolist())
                r_samples = ro.StrVector(count_matrix.columns.tolist())

                # Create DGEList
                dge = edger.DGEList(counts=r_counts)
                dge = edger.calcNormFactors(dge, method="TMM")

                # Get norm factors
                norm_factors = np.array(dge.slots["samples"].slots["norm.factors"])
                log_cpm_r = edger.cpm(dge, log=True, prior_count=2)
                log_cpm = np.array(log_cpm_r)

                # Create normalized dataframe
                normalized_df = pd.DataFrame(
                    log_cpm,
                    index=count_matrix.index,
                    columns=count_matrix.columns,
                )

                fitted_params = {
                    "norm_factors": norm_factors.tolist(),
                    "method": "TMM",
                    "prior_count": 2,
                }

                if use_voom:
                    logger.info("Applying voom transformation...")
                    n_samples = count_matrix.shape[1]
                    design_r = ro.r(f"matrix(1, nrow={n_samples}, ncol=1)")
                    v_obj = limma.voom(dge, design=design_r)
                    voom_expr = np.array(v_obj.slots["E"])
                    normalized_df = pd.DataFrame(
                        voom_expr,
                        index=count_matrix.index,
                        columns=count_matrix.columns,
                    )
                    fitted_params["voom_applied"] = True

                self.fitted_params["tmm_norm"] = fitted_params
                self.contract.add_params(
                    normalization_method="TMM",
                    voom_applied=use_voom,
                    prior_count=2,
                )

                logger.info(f"TMM normalization complete.")
                return normalized_df, fitted_params

            except Exception as e:
                logger.warning(f"edgeR TMM failed: {e}")

        # Fallback: use pydeseq2 median-of-ratios (better than naive log-CPM)
        try:
            from pydeseq2.dds import DeseqDataSet

            logger.info("Falling back to pydeseq2 median-of-ratios normalization...")

            # pydeseq2 expects samples x genes
            counts_t = count_matrix.T
            metadata = pd.DataFrame({"condition": ["A"] * counts_t.shape[0]}, index=counts_t.index)
            dds = DeseqDataSet(counts=counts_t, metadata=metadata, design="~1")
            dds.fit_size_factors()

            size_factors = dds.obsm["size_factors"]
            normalized = count_matrix.div(size_factors, axis=1)
            log_normalized = np.log2(normalized + 1)

            fitted_params = {"method": "pydeseq2_median_ratios", "size_factors": size_factors.tolist()}
            self.fitted_params["tmm_norm"] = fitted_params
            return log_normalized, fitted_params
        except ImportError:
            pass

        # Final fallback: proper log-CPM with library size normalization
        logger.info("Falling back to log-CPM normalization...")
        lib_sizes = count_matrix.sum(axis=0)
        cpm = count_matrix.div(lib_sizes, axis=1) * 1e6
        log_cpm = np.log2(cpm + 1)
        fitted_params = {"method": "log_cpm", "fallback": True}
        self.fitted_params["tmm_norm"] = fitted_params
        return log_cpm, fitted_params

    def low_expression_filter(
        self, expression_df: pd.DataFrame, percentile: float = 25, min_threshold: float = 0.0
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Filter out low-expression genes.

        Parameters
        ----------
        expression_df : pd.DataFrame
            Normalized expression matrix.
        percentile : float
            Filter genes with mean expression below this percentile (0-100).
        min_threshold : float
            Absolute minimum expression threshold.

        Returns
        -------
        Tuple[pd.DataFrame, Dict]
            (filtered_df, filter_stats)
        """
        logger.info(f"Filtering low-expression genes (percentile={percentile})...")

        mean_expr = expression_df.mean(axis=1)
        threshold = np.percentile(mean_expr, percentile)
        threshold = max(threshold, min_threshold)

        keep_genes = mean_expr >= threshold
        filtered_df = expression_df[keep_genes]

        filter_stats = {
            "total_genes": expression_df.shape[0],
            "filtered_genes": (keep_genes == False).sum(),
            "retained_genes": keep_genes.sum(),
            "retention_rate": keep_genes.sum() / expression_df.shape[0],
            "threshold": float(threshold),
        }

        self.contract.add_params(
            low_expr_filter_percentile=percentile,
            low_expr_filter_threshold=float(threshold),
        )

        logger.info(f"Filtering complete: {filter_stats['retained_genes']} "
                   f"genes retained ({filter_stats['retention_rate']*100:.1f}%)")

        return filtered_df, filter_stats

    def log2_transform(self, expression_df: pd.DataFrame, pseudocount: float = 0.0) -> pd.DataFrame:
        """
        Log2 transformation.

        Parameters
        ----------
        expression_df : pd.DataFrame
            Expression matrix (should be normalized first).
        pseudocount : float
            Pseudocount to add before log transformation.

        Returns
        -------
        pd.DataFrame
            Log2-transformed expression matrix.
        """
        logger.info(f"Log2 transformation (pseudocount={pseudocount})...")

        log_expr = np.log2(expression_df + pseudocount)
        self.contract.add_params(log_transform=True, pseudocount=pseudocount)
        return log_expr

    def plot_qc(
        self, expression_before: pd.DataFrame, expression_after: pd.DataFrame,
        sample_metadata: Optional[pd.DataFrame] = None, output_dir: Optional[Path] = None
    ) -> Dict:
        """
        Generate QC plots (density, boxplot, MA-plot if applicable).

        Parameters
        ----------
        expression_before : pd.DataFrame
            Expression before normalization.
        expression_after : pd.DataFrame
            Expression after normalization.
        sample_metadata : Optional[pd.DataFrame]
            Sample metadata for coloring.
        output_dir : Optional[Path]
            Directory to save plots.

        Returns
        -------
        Dict
            Dictionary of saved plot paths.
        """
        if output_dir is None:
            output_dir = Path("./qc_plots")
        output_dir.mkdir(parents=True, exist_ok=True)

        plots = {}

        # Density plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for i, (ax, expr_data) in enumerate([(axes[0], expression_before), (axes[1], expression_after)]):
            for col in expr_data.columns[:20]:  # Plot first 20 samples
                ax.plot(expr_data[col].values, alpha=0.3, linewidth=1)
            ax.set_xlabel("Gene index")
            ax.set_ylabel("Expression level")
            ax.set_title("Before" if i == 0 else "After normalization")
        plt.tight_layout()
        density_path = output_dir / "density_before_after.png"
        plt.savefig(density_path, dpi=100)
        plt.close()
        plots["density"] = str(density_path)
        logger.info(f"Saved density plot: {density_path}")

        # Box plot (subset of genes)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for i, (ax, expr_data) in enumerate([(axes[0], expression_before), (axes[1], expression_after)]):
            expr_subset = expr_data.iloc[:50].T  # First 50 genes
            expr_subset.boxplot(ax=ax, rot=90)
            ax.set_title("Before" if i == 0 else "After normalization")
            ax.set_ylabel("Expression")
        plt.tight_layout()
        boxplot_path = output_dir / "boxplot_before_after.png"
        plt.savefig(boxplot_path, dpi=100)
        plt.close()
        plots["boxplot"] = str(boxplot_path)
        logger.info(f"Saved boxplot: {boxplot_path}")

        # Mean-SD plot
        fig, ax = plt.subplots(figsize=(8, 6))
        mean_before = expression_before.mean(axis=1)
        sd_before = expression_before.std(axis=1)
        mean_after = expression_after.mean(axis=1)
        sd_after = expression_after.std(axis=1)

        ax.scatter(mean_before, sd_before, alpha=0.3, s=10, label="Before")
        ax.scatter(mean_after, sd_after, alpha=0.3, s=10, label="After")
        ax.set_xlabel("Mean expression")
        ax.set_ylabel("Std dev")
        ax.set_title("Mean-SD trend")
        ax.legend()
        plt.tight_layout()
        meansd_path = output_dir / "mean_sd_trend.png"
        plt.savefig(meansd_path, dpi=100)
        plt.close()
        plots["mean_sd"] = str(meansd_path)
        logger.info(f"Saved mean-SD plot: {meansd_path}")

        self.qc_plots = plots
        return plots

    def normalize_pipeline(
        self,
        expression_df: pd.DataFrame,
        platform_type: str = "array",
        low_expr_filter_percentile: float = 25,
        output_dir: Optional[Path] = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Complete normalization pipeline.

        Parameters
        ----------
        expression_df : pd.DataFrame
            Raw expression matrix.
        platform_type : str
            "array" for microarray, "rnaseq" for RNA-seq.
        low_expr_filter_percentile : float
            Percentile threshold for low-expression filtering.
        output_dir : Optional[Path]
            Directory for saving QC plots.

        Returns
        -------
        Tuple[pd.DataFrame, Dict]
            (normalized_df, pipeline_stats)
        """
        logger.info(f"Starting normalization pipeline ({platform_type})...")

        expr_before = expression_df.copy()
        pipeline_stats = {}

        # Step 1: Platform-specific normalization
        if platform_type.lower() == "array":
            norm_expr, norm_params = self.quantile_normalize_array(expression_df)
            pipeline_stats["normalization"] = norm_params
        elif platform_type.lower() == "rnaseq":
            norm_expr, norm_params = self.tmm_normalize_rnaseq(expression_df)
            pipeline_stats["normalization"] = norm_params
        else:
            raise ValueError(f"Unknown platform_type: {platform_type}")

        # Step 2: Low-expression filtering
        filt_expr, filt_stats = self.low_expression_filter(
            norm_expr, percentile=low_expr_filter_percentile
        )
        pipeline_stats["filtering"] = filt_stats

        # Step 3: Log2 transformation
        final_expr = self.log2_transform(filt_expr, pseudocount=0.0)
        pipeline_stats["log_transform"] = {"applied": True, "pseudocount": 0.0}

        # Step 4: QC plots
        if output_dir:
            self.plot_qc(expr_before, final_expr, output_dir=output_dir)

        # Freeze contract
        self.contract.freeze()

        pipeline_stats["contract_id"] = self.contract.contract_id
        logger.info(f"Normalization pipeline complete. Contract ID: {self.contract.contract_id}")

        return final_expr, pipeline_stats


def main():
    """Example usage."""
    logger.info("Expression normalizer initialized.")


if __name__ == "__main__":
    main()
