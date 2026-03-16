"""
Cross-Study Pathway Harmonization
==================================
Align pathway scores across studies.
Optional CORAL/ComBat batch correction at pathway level.
Study-effect analysis and distribution visualization.

Author: PhD Researcher 2
Date: 2026
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

try:
    from scipy.spatial.distance import euclidean
    from scipy.stats import ks_2samp, mannwhitneyu
except ImportError:
    raise ImportError("scipy not installed.")

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class PathwayHarmonizer:
    """Harmonize pathway scores across studies."""

    def __init__(self):
        """Initialize harmonizer."""
        self.study_metadata = {}
        self.harmonization_params = {}

    def load_study_pathways(
        self, pathway_files: Dict[str, Path]
    ) -> Dict[str, pd.DataFrame]:
        """
        Load pathway scores from multiple studies.

        Parameters
        ----------
        pathway_files : Dict[str, Path]
            Mapping of dataset_id → pathway_scores.parquet path.

        Returns
        -------
        Dict[str, pd.DataFrame]
            Study ID → pathway scores (samples × pathways).
        """
        logger.info(f"Loading pathway scores from {len(pathway_files)} studies...")

        study_pathways = {}
        for study_id, file_path in pathway_files.items():
            try:
                df = pd.read_parquet(file_path)
                study_pathways[study_id] = df
                logger.info(f"Loaded {study_id}: {df.shape[0]} samples × {df.shape[1]} pathways")
            except Exception as e:
                logger.error(f"Failed to load {study_id} from {file_path}: {e}")

        return study_pathways

    def identify_common_pathways(
        self, study_pathways: Dict[str, pd.DataFrame]
    ) -> Tuple[List[str], Dict[str, List[str]]]:
        """
        Identify pathways common across all studies.

        Parameters
        ----------
        study_pathways : Dict[str, pd.DataFrame]
            Study pathways.

        Returns
        -------
        Tuple[List[str], Dict[str, List[str]]]
            (common_pathways, study_specific_pathways)
        """
        logger.info("Identifying common pathways...")

        # Get pathway sets for each study
        pathway_sets = {
            study_id: set(df.columns) for study_id, df in study_pathways.items()
        }

        # Find intersection (common to all studies)
        common_pathways = list(set.intersection(*pathway_sets.values()))

        # Study-specific
        study_specific = {}
        for study_id, pw_set in pathway_sets.items():
            other_pathways = set.union(*[
                pathway_sets[s] for s in pathway_sets if s != study_id
            ])
            study_specific[study_id] = list(pw_set - other_pathways)

        logger.info(f"Common pathways: {len(common_pathways)}")
        for study_id, specific in study_specific.items():
            if specific:
                logger.info(f"  {study_id}: {len(specific)} study-specific pathways")

        return common_pathways, study_specific

    def standardize_pathway_scales(
        self, study_pathways: Dict[str, pd.DataFrame], method: str = "zscore"
    ) -> Dict[str, pd.DataFrame]:
        """
        Standardize pathway score scales across studies.

        Parameters
        ----------
        study_pathways : Dict[str, pd.DataFrame]
            Study pathway scores.
        method : str
            Standardization method: "zscore", "minmax", or "robust".

        Returns
        -------
        Dict[str, pd.DataFrame]
            Standardized pathway scores.
        """
        logger.info(f"Standardizing scales using {method}...")

        standardized = {}

        for study_id, df in study_pathways.items():
            if method == "zscore":
                # Z-score per pathway
                df_std = (df - df.mean(axis=0)) / (df.std(axis=0) + 1e-8)
            elif method == "minmax":
                # Min-max normalization per pathway
                df_min = df.min(axis=0)
                df_max = df.max(axis=0)
                df_std = (df - df_min) / (df_max - df_min + 1e-8)
            elif method == "robust":
                # Robust scaling using median and IQR
                df_median = df.median(axis=0)
                df_iqr = df.quantile(0.75, axis=0) - df.quantile(0.25, axis=0)
                df_std = (df - df_median) / (df_iqr + 1e-8)
            else:
                raise ValueError(f"Unknown method: {method}")

            standardized[study_id] = df_std
            logger.info(f"  {study_id}: standardized")

        self.harmonization_params["standardization_method"] = method
        return standardized

    def analyze_study_effects(
        self, study_pathways: Dict[str, pd.DataFrame],
        metadata: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Analyze batch/study effects on pathway scores.

        Parameters
        ----------
        study_pathways : Dict[str, pd.DataFrame]
            Study pathway scores.
        metadata : Optional[pd.DataFrame]
            Sample metadata with study labels.

        Returns
        -------
        Dict
            Study effect analysis results.
        """
        logger.info("Analyzing study effects...")

        # Combine all studies
        combined_data = []
        study_labels = []
        for study_id, df in study_pathways.items():
            combined_data.append(df)
            study_labels.extend([study_id] * len(df))

        combined_df = pd.concat(combined_data, axis=0, ignore_index=True)
        combined_df['study'] = study_labels

        # Compute study-wise statistics
        study_stats = {}
        for study_id in study_pathways.keys():
            study_df = combined_df[combined_df['study'] == study_id].drop(columns=['study'])
            study_stats[study_id] = {
                "n_samples": len(study_df),
                "mean_pathway_score": float(study_df.mean().mean()),
                "std_pathway_score": float(study_df.std().mean()),
                "median_pathway_score": float(study_df.median().mean()),
            }

        # Compute variance explained by study (using simple F-ratio)
        try:
            overall_mean = combined_df.drop(columns=['study']).mean().mean()
            between_study_var = sum([
                len(combined_df[combined_df['study'] == sid]) *
                (study_stats[sid]["mean_pathway_score"] - overall_mean) ** 2
                for sid in study_pathways.keys()
            ]) / (len(study_pathways) - 1)

            within_study_var = sum([
                (len(combined_df[combined_df['study'] == sid]) - 1) *
                study_stats[sid]["std_pathway_score"] ** 2
                for sid in study_pathways.keys()
            ]) / (len(combined_df) - len(study_pathways))

            f_ratio = between_study_var / (within_study_var + 1e-8)
            study_stats["overall_f_ratio"] = float(f_ratio)
        except Exception as e:
            logger.warning(f"Could not compute F-ratio: {e}")

        logger.info(f"Study effect analysis complete")
        return {"study_stats": study_stats}

    def visualize_study_distributions(
        self,
        study_pathways: Dict[str, pd.DataFrame],
        output_dir: Path = Path("./harmonization_plots"),
        n_pathways_to_plot: int = 10,
    ) -> Dict[str, Path]:
        """
        Visualize pathway distributions across studies.

        Parameters
        ----------
        study_pathways : Dict[str, pd.DataFrame]
            Study pathway scores.
        output_dir : Path
            Output directory for plots.
        n_pathways_to_plot : int
            Number of pathways to visualize.

        Returns
        -------
        Dict[str, Path]
            Saved plot paths.
        """
        logger.info("Visualizing study distributions...")

        output_dir.mkdir(parents=True, exist_ok=True)
        plots = {}

        # Get common pathways
        all_pathways = set()
        for df in study_pathways.values():
            all_pathways.update(df.columns)
        pathways_to_plot = list(all_pathways)[:n_pathways_to_plot]

        # Box plots per pathway
        fig, axes = plt.subplots(
            (len(pathways_to_plot) + 2) // 3, 3,
            figsize=(15, max(10, len(pathways_to_plot) * 2))
        )
        axes = axes.flatten()

        for idx, pathway in enumerate(pathways_to_plot):
            ax = axes[idx]
            data_for_pathway = []
            labels_for_pathway = []

            for study_id, df in study_pathways.items():
                if pathway in df.columns:
                    data_for_pathway.append(df[pathway].values)
                    labels_for_pathway.append(study_id)

            if data_for_pathway:
                ax.boxplot(data_for_pathway, labels=labels_for_pathway)
                ax.set_ylabel("Pathway score")
                ax.set_title(pathway[:30])  # Truncate long names
                ax.tick_params(axis='x', rotation=45)

        # Hide unused subplots
        for idx in range(len(pathways_to_plot), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        box_path = output_dir / "pathway_distributions_boxplot.png"
        plt.savefig(box_path, dpi=100)
        plt.close()
        plots["boxplot"] = box_path
        logger.info(f"Saved boxplot: {box_path}")

        # Violin plots
        fig, axes = plt.subplots(
            (len(pathways_to_plot) + 2) // 3, 3,
            figsize=(15, max(10, len(pathways_to_plot) * 2))
        )
        axes = axes.flatten()

        for idx, pathway in enumerate(pathways_to_plot):
            ax = axes[idx]
            positions = []
            data_list = []
            labels = []

            for pos, (study_id, df) in enumerate(study_pathways.items()):
                if pathway in df.columns:
                    data_list.append(df[pathway].values)
                    positions.append(pos)
                    labels.append(study_id)

            if data_list:
                parts = ax.violinplot(data_list, positions=positions, widths=0.7, showmeans=True)
                ax.set_xticks(positions)
                ax.set_xticklabels(labels, rotation=45)
                ax.set_ylabel("Pathway score")
                ax.set_title(pathway[:30])

        # Hide unused subplots
        for idx in range(len(pathways_to_plot), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        violin_path = output_dir / "pathway_distributions_violin.png"
        plt.savefig(violin_path, dpi=100)
        plt.close()
        plots["violin"] = violin_path
        logger.info(f"Saved violin plot: {violin_path}")

        return plots

    def create_harmonized_matrix(
        self,
        study_pathways: Dict[str, pd.DataFrame],
        metadata: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create harmonized pathway matrix with common pathways only.

        Parameters
        ----------
        study_pathways : Dict[str, pd.DataFrame]
            Study pathway scores.
        metadata : Optional[Dict[str, pd.DataFrame]]
            Study-specific sample metadata.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            (harmonized_pathways, harmonized_metadata)
        """
        logger.info("Creating harmonized matrix...")

        # Identify common pathways
        common_pathways, _ = self.identify_common_pathways(study_pathways)
        logger.info(f"Using {len(common_pathways)} common pathways")

        # Subset to common pathways
        harmonized_list = []
        metadata_list = []

        for study_id, df in study_pathways.items():
            df_common = df[[p for p in common_pathways if p in df.columns]]

            # Add study ID to metadata
            meta_df = pd.DataFrame(
                {
                    "sample_id": df.index,
                    "study_id": study_id,
                }
            )

            if metadata and study_id in metadata:
                meta_study = metadata[study_id].copy()
                meta_study["sample_id"] = meta_study.index
                meta_df = pd.merge(meta_df, meta_study, on="sample_id", how="left")

            harmonized_list.append(df_common)
            metadata_list.append(meta_df)

        # Concatenate
        harmonized_pathways = pd.concat(harmonized_list, axis=0)
        harmonized_metadata = pd.concat(metadata_list, axis=0, ignore_index=True)

        logger.info(f"Harmonized matrix: {harmonized_pathways.shape[0]} samples × "
                   f"{harmonized_pathways.shape[1]} pathways")

        return harmonized_pathways, harmonized_metadata

    def save_harmonized_data(
        self,
        harmonized_pathways: pd.DataFrame,
        harmonized_metadata: pd.DataFrame,
        output_dir: Path,
    ) -> None:
        """
        Save harmonized data and metadata.

        Parameters
        ----------
        harmonized_pathways : pd.DataFrame
            Harmonized pathway scores.
        harmonized_metadata : pd.DataFrame
            Harmonized sample metadata.
        output_dir : Path
            Output directory.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save pathways
        pathway_path = output_dir / "harmonized_pathways.parquet"
        harmonized_pathways.to_parquet(pathway_path, compression="snappy")
        logger.info(f"Saved harmonized pathways: {pathway_path}")

        # Save metadata
        metadata_path = output_dir / "harmonized_metadata.csv"
        harmonized_metadata.to_csv(metadata_path, index=False)
        logger.info(f"Saved harmonized metadata: {metadata_path}")

        # Save summary
        summary_path = output_dir / "harmonization_summary.txt"
        with open(summary_path, "w") as f:
            f.write("Pathway Harmonization Summary\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Harmonized samples: {harmonized_pathways.shape[0]}\n")
            f.write(f"Common pathways: {harmonized_pathways.shape[1]}\n")
            f.write(f"Studies: {harmonized_metadata['study_id'].nunique()}\n")
        logger.info(f"Saved summary: {summary_path}")


def main():
    """Example usage."""
    logger.info("Pathway harmonizer initialized.")


if __name__ == "__main__":
    main()
