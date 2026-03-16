"""
End-to-End Preprocessing Pipeline
==================================
Orchestrates all preprocessing steps:
1. Download GEO datasets
2. Probe mapping (arrays only)
3. Normalization (per-platform)
4. Quality control
5. Pathway scoring (GSVA/ssGSEA per study)
6. Cross-study harmonization

CLI with checkpoint recovery.

Author: PhD Researcher 2
Date: 2026
"""

import logging
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.download_geo import GEODownloader
from src.preprocessing.probe_mapping import ProbeMapper
from src.preprocessing.normalization import ExpressionNormalizer, NormalizationContract
from src.preprocessing.quality_control import QualityController
from src.preprocessing.pathway_scoring import PathwayScorer
from src.preprocessing.data_contract import DataValidator, DataContract, PreprocessingCodeHasher
from src.preprocessing.harmonization import PathwayHarmonizer

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("preprocessing.log"),
    ]
)


class PreprocessingPipeline:
    """Orchestrate full preprocessing pipeline."""

    def __init__(
        self,
        data_dir: Path = Path("./data"),
        output_dir: Path = Path("./data/analysis_ready"),
        config_file: Optional[Path] = None,
    ):
        """
        Initialize pipeline.

        Parameters
        ----------
        data_dir : Path
            Base data directory.
        output_dir : Path
            Output directory for processed data.
        config_file : Optional[Path]
            Configuration JSON file.
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.standardized_dir = self.data_dir / "standardized"
        self.output_dir = Path(output_dir)

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.standardized_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.config = {}
        if config_file and config_file.exists():
            with open(config_file, "r") as f:
                self.config = json.load(f)

        self.checkpoints = {}
        self.load_checkpoints()

    def load_checkpoints(self) -> None:
        """Load preprocessing checkpoints (if any exist)."""
        checkpoint_file = self.output_dir / ".checkpoints.json"
        if checkpoint_file.exists():
            with open(checkpoint_file, "r") as f:
                self.checkpoints = json.load(f)
            logger.info(f"Loaded {len(self.checkpoints)} checkpoints")

    def save_checkpoints(self) -> None:
        """Save preprocessing checkpoints."""
        checkpoint_file = self.output_dir / ".checkpoints.json"
        with open(checkpoint_file, "w") as f:
            json.dump(self.checkpoints, f, indent=2, default=str)
        logger.info(f"Saved {len(self.checkpoints)} checkpoints")

    def step_download(self) -> None:
        """Step 1: Download GEO datasets."""
        if self.checkpoints.get("download"):
            logger.info("Download already completed. Skipping.")
            return

        logger.info("=" * 60)
        logger.info("STEP 1: Download GEO datasets")
        logger.info("=" * 60)

        downloader = GEODownloader(output_dir=str(self.raw_dir))
        results = downloader.download_all()

        logger.info(f"Downloaded {len(results)} GEO datasets")
        self.checkpoints["download"] = True
        self.save_checkpoints()

    def step_probe_mapping(self) -> None:
        """Step 2: Map probes to genes (arrays only)."""
        if self.checkpoints.get("probe_mapping"):
            logger.info("Probe mapping already completed. Skipping.")
            return

        logger.info("=" * 60)
        logger.info("STEP 2: Probe mapping for microarray data")
        logger.info("=" * 60)

        mapper = ProbeMapper(species="human")

        # Probe mapping for each array dataset
        array_datasets = {
            "GSE19784": ("GPL570", "U133Plus2"),
            "GSE39754": ("GPL17077", "AgilentHE_v2"),
            "GSE2658": ("GPL570", "U133Plus2"),
        }

        for dataset_id, (platform, platform_name) in array_datasets.items():
            dataset_dir = self.raw_dir / dataset_id
            expr_file = dataset_dir / f"{dataset_id}_expression.parquet"

            if not expr_file.exists():
                logger.warning(f"Expression file not found: {expr_file}")
                continue

            logger.info(f"Mapping {dataset_id} ({platform_name})...")
            expr_df = pd.read_parquet(expr_file)
            expr_df.index.name = "probe_id"

            # Map probes to genes
            gene_expr, stats = mapper.map_affymetrix_probes(expr_df, platform=platform)

            # Save mapped data
            mapped_file = self.standardized_dir / f"{dataset_id}_genes.parquet"
            mapper.save_mapping(gene_expr, mapped_file, platform=platform)

        self.checkpoints["probe_mapping"] = True
        self.save_checkpoints()

    def step_normalization(self) -> None:
        """Step 3: Normalize expression data."""
        if self.checkpoints.get("normalization"):
            logger.info("Normalization already completed. Skipping.")
            return

        logger.info("=" * 60)
        logger.info("STEP 3: Expression normalization")
        logger.info("=" * 60)

        # Dataset-specific normalization (arrays)
        array_datasets = {
            "GSE19784": "array",
            "GSE39754": "array",
            "GSE2658": "array",
        }

        normalization_contracts = {}

        for dataset_id, platform_type in array_datasets.items():
            # Load mapped/raw expression
            dataset_dir = self.raw_dir / dataset_id
            mapped_file = self.standardized_dir / f"{dataset_id}_genes.parquet"

            if mapped_file.exists():
                expr_file = mapped_file
            else:
                expr_file = dataset_dir / f"{dataset_id}_expression.parquet"

            if not expr_file.exists():
                logger.warning(f"Expression file not found: {expr_file}")
                continue

            logger.info(f"Normalizing {dataset_id} ({platform_type})...")
            expr_df = pd.read_parquet(expr_file)

            # Create and apply normalization
            contract = NormalizationContract()
            normalizer = ExpressionNormalizer(contract=contract)

            norm_expr, pipeline_stats = normalizer.normalize_pipeline(
                expr_df,
                platform_type=platform_type,
                low_expr_filter_percentile=25,
                output_dir=self.output_dir / "qc_plots" / dataset_id,
            )

            # Save normalized data
            norm_file = self.standardized_dir / f"{dataset_id}_normalized.parquet"
            norm_expr.to_parquet(norm_file, compression="snappy")
            logger.info(f"Saved normalized data: {norm_file}")

            # Save contract
            contract_file = self.standardized_dir / f"{dataset_id}_normalization_contract.pkl"
            contract.save(contract_file)

            normalization_contracts[dataset_id] = contract

        self.checkpoints["normalization"] = True
        self.save_checkpoints()

    def step_quality_control(self) -> None:
        """Step 4: Quality control."""
        if self.checkpoints.get("quality_control"):
            logger.info("Quality control already completed. Skipping.")
            return

        logger.info("=" * 60)
        logger.info("STEP 4: Quality control")
        logger.info("=" * 60)

        qc_controller = QualityController(output_dir=self.output_dir / "qc_reports")

        datasets = ["GSE19784", "GSE39754", "GSE2658"]

        for dataset_id in datasets:
            # Load normalized expression
            norm_file = self.standardized_dir / f"{dataset_id}_normalized.parquet"
            pheno_file = self.raw_dir / dataset_id / f"{dataset_id}_phenotype.csv"

            if not norm_file.exists():
                logger.warning(f"Normalized file not found: {norm_file}")
                continue

            logger.info(f"QC for {dataset_id}...")
            expr_df = pd.read_parquet(norm_file)

            metadata = None
            if pheno_file.exists():
                metadata = pd.read_csv(pheno_file)

            # Generate QC report
            report_path = qc_controller.generate_qc_report(
                expr_df,
                metadata=metadata,
                dataset_id=dataset_id,
                batch_column="dataset_id" if metadata is not None else None,
            )

            logger.info(f"QC report: {report_path}")

        self.checkpoints["quality_control"] = True
        self.save_checkpoints()

    def step_pathway_scoring(self) -> None:
        """Step 5: Pathway scoring (GSVA/ssGSEA per study)."""
        if self.checkpoints.get("pathway_scoring"):
            logger.info("Pathway scoring already completed. Skipping.")
            return

        logger.info("=" * 60)
        logger.info("STEP 5: Pathway scoring")
        logger.info("=" * 60)

        datasets = ["GSE19784", "GSE39754", "GSE2658"]

        for dataset_id in datasets:
            # Load normalized expression
            norm_file = self.standardized_dir / f"{dataset_id}_normalized.parquet"
            pheno_file = self.raw_dir / dataset_id / f"{dataset_id}_phenotype.csv"

            if not norm_file.exists():
                logger.warning(f"Normalized file not found: {norm_file}")
                continue

            logger.info(f"Pathway scoring for {dataset_id}...")
            expr_df = pd.read_parquet(norm_file)

            # Score pathways
            scorer = PathwayScorer(method="ssgsea")  # or "gsva"
            pathway_scores, metadata = scorer.score_pathways(
                expr_df,
                pathway_source="all",
            )

            # Load phenotype
            if pheno_file.exists():
                pheno_df = pd.read_csv(pheno_file, index_col="sample_id")
                # Align with pathway scores
                common_samples = pathway_scores.index.intersection(pheno_df.index)
                pathway_scores = pathway_scores.loc[common_samples]

            # Save pathway scores
            pathway_file = self.output_dir / f"{dataset_id}_pathways.parquet"
            scorer.save_pathway_scores(pathway_scores, metadata, pathway_file, dataset_id=dataset_id)

        self.checkpoints["pathway_scoring"] = True
        self.save_checkpoints()

    def step_harmonization(self) -> None:
        """Step 6: Cross-study harmonization."""
        if self.checkpoints.get("harmonization"):
            logger.info("Harmonization already completed. Skipping.")
            return

        logger.info("=" * 60)
        logger.info("STEP 6: Cross-study harmonization")
        logger.info("=" * 60)

        harmonizer = PathwayHarmonizer()

        # Load pathway scores
        pathway_files = {
            "GSE19784": self.output_dir / "GSE19784_pathways.parquet",
            "GSE39754": self.output_dir / "GSE39754_pathways.parquet",
            "GSE2658": self.output_dir / "GSE2658_pathways.parquet",
        }

        # Filter to existing files
        pathway_files = {k: v for k, v in pathway_files.items() if v.exists()}

        if not pathway_files:
            logger.warning("No pathway score files found. Skipping harmonization.")
            return

        logger.info(f"Loading pathway scores from {len(pathway_files)} studies...")
        study_pathways = harmonizer.load_study_pathways(pathway_files)

        # Standardize scales
        standardized = harmonizer.standardize_pathway_scales(study_pathways, method="zscore")

        # Analyze study effects
        study_effects = harmonizer.analyze_study_effects(standardized)
        logger.info(f"Study effects: {study_effects}")

        # Visualize
        plots = harmonizer.visualize_study_distributions(
            standardized,
            output_dir=self.output_dir / "harmonization_plots",
        )

        # Create harmonized matrix
        harmonized_pathways, harmonized_metadata = harmonizer.create_harmonized_matrix(
            standardized
        )

        # Save
        harmonizer.save_harmonized_data(
            harmonized_pathways,
            harmonized_metadata,
            output_dir=self.output_dir / "harmonized",
        )

        self.checkpoints["harmonization"] = True
        self.save_checkpoints()

    def run_full_pipeline(self) -> None:
        """Run complete preprocessing pipeline."""
        logger.info("=" * 60)
        logger.info("STARTING FULL PREPROCESSING PIPELINE")
        logger.info("=" * 60)

        try:
            self.step_download()
            self.step_probe_mapping()
            self.step_normalization()
            self.step_quality_control()
            self.step_pathway_scoring()
            self.step_harmonization()

            logger.info("=" * 60)
            logger.info("PIPELINE COMPLETE")
            logger.info("=" * 60)
            logger.info(f"Output directory: {self.output_dir}")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise

    def run_selective(self, steps: list) -> None:
        """
        Run selected pipeline steps.

        Parameters
        ----------
        steps : list
            List of step names to run (e.g., ["download", "normalization"]).
        """
        step_methods = {
            "download": self.step_download,
            "probe_mapping": self.step_probe_mapping,
            "normalization": self.step_normalization,
            "quality_control": self.step_quality_control,
            "pathway_scoring": self.step_pathway_scoring,
            "harmonization": self.step_harmonization,
        }

        for step in steps:
            if step not in step_methods:
                logger.warning(f"Unknown step: {step}")
                continue
            step_methods[step]()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MM Risk Signature Preprocessing Pipeline"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default="./data",
        help="Base data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="./data/analysis_ready",
        help="Output directory",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Configuration JSON file",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        help="Specific steps to run (download, probe_mapping, normalization, quality_control, pathway_scoring, harmonization)",
    )

    args = parser.parse_args()

    pipeline = PreprocessingPipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        config_file=args.config,
    )

    if args.steps:
        pipeline.run_selective(args.steps)
    else:
        pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
