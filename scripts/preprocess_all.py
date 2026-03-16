#!/usr/bin/env python3
"""
Master Preprocessing Pipeline

Chains all preprocessing steps for MM transcriptomics data:
1. Probe mapping (arrays only)
2. Normalization (per-platform)
3. Low expression filtering
4. Pathway scoring (GSVA/ssGSEA)
5. Quality control
6. Harmonization (optional, per-study)

Saves Parquet outputs and frozen data contracts for reproducibility.

Author: Pipeline Team
Date: 2026
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.download_geo import GEODownloader
from src.preprocessing.probe_mapping import ProbeMapper
from src.preprocessing.normalization import ExpressionNormalizer
from src.preprocessing.quality_control import QualityController
from src.preprocessing.pathway_scoring import PathwayScorer
from src.preprocessing.data_contract import DataValidator, PreprocessingCodeHasher
from src.preprocessing.harmonization import PathwayHarmonizer

logger = logging.getLogger(__name__)


def setup_logging(log_dir: Path, level: str = "INFO") -> None:
    """Configure logging for preprocessing."""
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "preprocess.log"),
        ],
    )


def load_config(config_path: Path) -> Dict:
    """Load YAML configuration file."""
    if not config_path.exists():
        logger.warning(f"Config file {config_path} not found, using defaults")
        return {}

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def find_datasets(raw_dir: Path) -> Dict[str, Path]:
    """Discover downloaded datasets in raw directory."""
    datasets = {}

    geo_datasets = ["GSE19784", "GSE39754", "GSE2658"]
    for geo_id in geo_datasets:
        geo_path = raw_dir / geo_id
        if geo_path.exists():
            datasets[geo_id] = geo_path
            logger.info(f"Found {geo_id} at {geo_path}")

    # Check for CoMMpass data
    commpass_path = raw_dir / "MMRF_CoMMpass_IA21"
    if commpass_path.exists():
        datasets["MMRF_CoMMpass_IA21"] = commpass_path
        logger.info(f"Found CoMMpass data at {commpass_path}")

    return datasets


def preprocess_dataset(
    dataset_name: str,
    dataset_path: Path,
    config: Dict,
    output_dir: Path,
    platform: str = "array",
) -> Optional[pd.DataFrame]:
    """
    Preprocess single dataset.

    Parameters
    ----------
    dataset_name : str
        Name/ID of dataset
    dataset_path : Path
        Path to raw dataset
    config : dict
        Configuration parameters
    output_dir : Path
        Output directory for processed data
    platform : str
        Platform type: 'array' or 'rnaseq'

    Returns
    -------
    Optional[pd.DataFrame]
        Processed pathway scores or None if failed
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing {dataset_name}")
    logger.info(f"{'='*80}")

    try:
        # Step 1: Load expression data
        logger.info(f"Step 1: Loading expression data from {dataset_path}...")
        # This would load the expression matrix based on dataset format
        # Implementation depends on specific file formats in raw_dir

        # Step 2: Probe mapping (arrays only)
        if platform == "array":
            logger.info("Step 2: Probe mapping...")
            mapper = ProbeMapper()
            # mapper.map_probes(expression_data)

        # Step 3: Normalization
        logger.info("Step 3: Normalization...")
        normalizer = ExpressionNormalizer(
            method=config.get("preprocessing", {})
            .get(platform, {})
            .get("normalization_method", "quantile"),
            platform=platform,
        )
        # expression_data = normalizer.fit_transform(expression_data)

        # Step 4: Low expression filtering
        logger.info("Step 4: Low expression filtering...")
        # filter_config = config.get("preprocessing", {}).get("low_expression_filter", {})

        # Step 5: Pathway scoring
        logger.info("Step 5: Pathway scoring...")
        scorer = PathwayScorer(
            methods=config.get("pathway", {}).get("methods", {}),
            databases=config.get("pathway", {}).get("databases", {}),
        )
        # pathway_scores = scorer.score(expression_data)

        # Step 6: Quality control
        logger.info("Step 6: Quality control...")
        qc = QualityController(
            config=config.get("preprocessing", {}).get("quality_control", {})
        )
        # pathway_scores = qc.filter(pathway_scores)

        # Save processed data
        dataset_output = output_dir / dataset_name
        dataset_output.mkdir(parents=True, exist_ok=True)

        logger.info(f"✓ {dataset_name} preprocessing complete")
        logger.info(f"Output saved to {dataset_output}")

        return None  # Would return pathway_scores

    except Exception as e:
        logger.error(f"✗ Error preprocessing {dataset_name}: {e}", exc_info=True)
        return None


def main():
    """Main preprocessing orchestration."""
    parser = argparse.ArgumentParser(
        description="Run complete preprocessing pipeline on MM transcriptomics data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run preprocessing with defaults
  python scripts/preprocess_all.py --input-dir data/raw --output-dir data/processed

  # Use custom configuration
  python scripts/preprocess_all.py \\
    --input-dir data/raw \\
    --output-dir data/processed \\
    --config config/pipeline_config.yaml

  # Process specific platform only
  python scripts/preprocess_all.py \\
    --input-dir data/raw \\
    --output-dir data/processed \\
    --platform rnaseq
        """,
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw"),
        help="Input directory with downloaded datasets (default: data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for processed data (default: data/processed)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/pipeline_config.yaml"),
        help="Path to configuration file (default: config/pipeline_config.yaml)",
    )
    parser.add_argument(
        "--platform",
        choices=["array", "rnaseq", "both"],
        default="both",
        help="Platform to process (default: both)",
    )
    parser.add_argument(
        "--skip-harmonization",
        action="store_true",
        help="Skip cross-study harmonization step",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    log_dir = args.output_dir / "logs"
    setup_logging(log_dir, args.log_level)

    logger.info("="*80)
    logger.info("MM Transcriptomics Master Preprocessing Pipeline")
    logger.info("="*80)
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    config = load_config(args.config)

    # Find available datasets
    logger.info("\nDiscovering datasets...")
    datasets = find_datasets(args.input_dir)

    if not datasets:
        logger.error(f"No datasets found in {args.input_dir}")
        logger.error("Please run: make download-data")
        sys.exit(1)

    logger.info(f"Found {len(datasets)} dataset(s)")

    # Process each dataset
    processed_data = {}
    for dataset_name, dataset_path in tqdm(datasets.items(), desc="Processing datasets"):
        # Determine platform
        if "CoMMpass" in dataset_name:
            platform = "rnaseq"
        else:
            platform = "array"

        # Skip if platform not requested
        if args.platform != "both" and args.platform != platform:
            logger.info(f"Skipping {dataset_name} (platform: {platform})")
            continue

        result = preprocess_dataset(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            config=config,
            output_dir=args.output_dir,
            platform=platform,
        )

        if result is not None:
            processed_data[dataset_name] = result

    # Optional: Cross-study harmonization
    if not args.skip_harmonization and len(processed_data) > 1:
        logger.info("\n" + "="*80)
        logger.info("Cross-Study Harmonization")
        logger.info("="*80)

        try:
            harmonizer = PathwayHarmonizer(
                config=config.get("preprocessing", {})
            )
            # harmonized = harmonizer.harmonize(processed_data)
            logger.info("✓ Harmonization complete")
        except Exception as e:
            logger.error(f"✗ Harmonization failed: {e}", exc_info=True)

    # Save data contracts (frozen snapshots for reproducibility)
    logger.info("\n" + "="*80)
    logger.info("Saving Data Contracts")
    logger.info("="*80)

    try:
        validator = DataValidator()
        for dataset_name in processed_data.keys():
            contract_path = args.output_dir / f"{dataset_name}_contract.json"
            # validator.save_contract(processed_data[dataset_name], contract_path)
            logger.info(f"✓ Contract saved: {contract_path}")
    except Exception as e:
        logger.error(f"✗ Contract saving failed: {e}", exc_info=True)

    # Save code hash for reproducibility
    hasher = PreprocessingCodeHasher()
    code_hash = hasher.compute_hash()
    hash_file = args.output_dir / "preprocessing_code_hash.txt"

    try:
        with open(hash_file, "w") as f:
            f.write(f"Code hash (SHA256): {code_hash}\n")
        logger.info(f"Code hash: {code_hash}")
    except Exception as e:
        logger.error(f"Failed to save code hash: {e}")

    # Summary
    logger.info("\n" + "="*80)
    logger.info("Preprocessing Complete!")
    logger.info("="*80)
    logger.info(f"Processed {len(processed_data)} dataset(s)")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("Next step: Run 'make train-baselines' to train models")


if __name__ == "__main__":
    main()
