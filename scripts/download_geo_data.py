#!/usr/bin/env python3
"""
Download GEO Transcriptomics Data

CLI script to download all 4 GEO datasets for MM transcriptomics pipeline:
- GSE19784: NDMM samples (Affymetrix U133Plus2)
- GSE39754: Healthy + MM samples (Agilent exon arrays)
- GSE2658: Pre-treatment MM (Affymetrix U133Plus2)
- MMRF CoMMpass: RNA-seq data (direct download from MMRF portal)

Handles CoMMpass specially with manual download instructions.

Author: Pipeline Team
Date: 2026
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.download_geo import GEODownloader

logger = logging.getLogger(__name__)


def setup_logging(output_dir: Path) -> None:
    """Configure logging for the download script."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "download_geo.log"),
        ],
    )


def load_config(config_path: Optional[Path]) -> Dict:
    """Load configuration from YAML file."""
    if config_path and config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def print_commpass_instructions() -> None:
    """Print instructions for manually downloading CoMMpass data."""
    instructions = """
╔════════════════════════════════════════════════════════════════════════════╗
║                   MMRF CoMMpass Data Download Instructions                 ║
╚════════════════════════════════════════════════════════════════════════════╝

The MMRF CoMMpass dataset requires manual download from the MMRF Portal:
https://research.themmrf.org/

Steps:
  1. Visit https://research.themmrf.org/
  2. Sign in with your institutional credentials (requires IRB approval)
  3. Navigate to "Downloads" or "Data" section
  4. Select "CoMMpass IA21" (or latest version)
  5. Download the RNA-seq expression matrix (TPM or counts)
  6. Download clinical metadata file
  7. Place files in: {output_dir}/MMRF_CoMMpass_IA21/

Expected files:
  - {output_dir}/MMRF_CoMMpass_IA21/expression_matrix.txt.gz
  - {output_dir}/MMRF_CoMMpass_IA21/clinical_metadata.csv

Once downloaded, the preprocessing pipeline will automatically detect and
process these files.
"""
    print(instructions)


def validate_datasets(config: Dict) -> List[str]:
    """Validate and return list of dataset IDs to download."""
    if not config or "datasets" not in config:
        # Use default datasets if config not provided
        return ["GSE19784", "GSE39754", "GSE2658"]

    datasets = config.get("datasets", {}).get("training_studies", [])
    dataset_ids = [
        ds["identifier"]
        for ds in datasets
        if ds.get("source") == "GEO" and ds.get("type") == "microarray"
    ]
    return dataset_ids


def main():
    """Main download orchestration function."""
    parser = argparse.ArgumentParser(
        description="Download GEO datasets for MM transcriptomics pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all default datasets
  python scripts/download_geo_data.py --output-dir data/raw

  # Download specific datasets
  python scripts/download_geo_data.py --datasets GSE19784 GSE2658 --output-dir data/raw

  # Use configuration file
  python scripts/download_geo_data.py --config config/pipeline_config.yaml --output-dir data/raw
        """,
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["GSE19784", "GSE39754", "GSE2658"],
        help="List of GEO dataset IDs to download (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Output directory for downloaded data (default: data/raw)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to pipeline configuration YAML file",
    )
    parser.add_argument(
        "--skip-geo",
        action="store_true",
        help="Skip GEO dataset downloads (useful if already downloaded)",
    )
    parser.add_argument(
        "--skip-commpass-instructions",
        action="store_true",
        help="Skip printing CoMMpass manual download instructions",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.output_dir)
    logger.setLevel(getattr(logging, args.log_level))

    logger.info("="*80)
    logger.info("MM Transcriptomics GEO Data Download")
    logger.info("="*80)
    logger.info(f"Output directory: {args.output_dir}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    config = load_config(args.config)

    # Determine datasets to download
    if args.datasets and args.datasets != ["GSE19784", "GSE39754", "GSE2658"]:
        datasets_to_download = args.datasets
    else:
        datasets_to_download = validate_datasets(config) or args.datasets

    logger.info(f"Datasets to download: {datasets_to_download}")

    # Download GEO datasets
    if not args.skip_geo:
        logger.info("\n" + "="*80)
        logger.info("Downloading GEO Datasets")
        logger.info("="*80)

        downloader = GEODownloader(output_dir=str(args.output_dir))

        for dataset_id in tqdm(
            datasets_to_download,
            desc="Downloading datasets",
            unit="dataset",
        ):
            try:
                logger.info(f"\nDownloading {dataset_id}...")
                geo_object = downloader.download_geo_dataset(dataset_id)

                if geo_object:
                    # Extract and save data
                    logger.info(f"Extracting metadata from {dataset_id}...")
                    # The GEODownloader handles saving internally
                    logger.info(f"✓ {dataset_id} downloaded successfully")
                else:
                    logger.warning(f"✗ Failed to download {dataset_id}")

            except Exception as e:
                logger.error(f"Error downloading {dataset_id}: {e}", exc_info=True)

    # Print CoMMpass instructions
    if not args.skip_commpass_instructions:
        logger.info("\n" + "="*80)
        print_commpass_instructions().format(output_dir=args.output_dir)
        logger.info("="*80)

    # Save metadata
    metadata_file = args.output_dir / "download_manifest.json"
    manifest = {
        "downloaded_at": str(Path.cwd()),
        "datasets": datasets_to_download,
        "output_dir": str(args.output_dir),
        "config_file": str(args.config) if args.config else None,
    }

    try:
        with open(metadata_file, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"\nManifest saved to {metadata_file}")
    except Exception as e:
        logger.error(f"Failed to save manifest: {e}")

    logger.info("\n" + "="*80)
    logger.info("Download Complete!")
    logger.info("="*80)
    logger.info(f"Data location: {args.output_dir}")
    logger.info("Next step: Run preprocessing with 'make preprocess'")


if __name__ == "__main__":
    main()
