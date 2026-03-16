"""
GEO Dataset Downloader
======================
Downloads Multiple Myeloma datasets from GEO and MMRF, extracts clinical/survival metadata.

Datasets:
- GSE19784: 320 NDMM samples (Affymetrix U133Plus2)
- GSE39754: Healthy + MM samples (Agilent exon arrays)
- GSE2658: Pre-treatment MM (Affymetrix U133Plus2)
- MMRF_CoMMpass_IA21: RNA-seq (via direct download from MMRF portal)

Extracts: OS (overall survival), PFS (progression-free survival), ISS stage, cytogenetics.

Author: PhD Researcher 2
Date: 2026
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, TypeVar, Any
import pandas as pd
import numpy as np
from datetime import datetime
import requests
from urllib.parse import urljoin
from functools import wraps

try:
    import GEOparse
except ImportError:
    raise ImportError("GEOparse not installed. Run: pip install GEOparse")

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Type variable for retry decorator
F = TypeVar('F', bound=Callable[..., Any])


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 5.0,
    max_delay: float = 60.0,
    timeout: float = 30.0,
) -> Callable[[F], F]:
    """
    Decorator for exponential backoff retry logic.

    Parameters
    ----------
    max_retries : int
        Maximum number of retry attempts (default: 3).
    base_delay : float
        Base delay in seconds before first retry (default: 5.0).
    max_delay : float
        Maximum delay cap in seconds (default: 60.0).
    timeout : float
        Timeout in seconds for operations (default: 30.0).

    Returns
    -------
    Callable
        Decorated function with retry logic.
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    # Build kwargs for this attempt, inject timeout if accepted
                    call_kwargs = kwargs.copy()
                    import inspect
                    sig = inspect.signature(func)
                    if 'timeout' in sig.parameters and 'timeout' not in call_kwargs:
                        call_kwargs['timeout'] = timeout

                    return func(*args, **call_kwargs)

                except (requests.RequestException, TimeoutError, ConnectionError, OSError) as e:
                    last_exception = e

                    if attempt < max_retries:
                        # Calculate exponential backoff with jitter
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {str(e)}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        # Final attempt failed
                        logger.error(
                            f"All {max_retries + 1} attempts failed for {func.__name__}. "
                            f"Final error: {str(e)}"
                        )
                        raise

                except Exception as e:
                    # Non-retryable exceptions
                    logger.error(f"Non-retryable error in {func.__name__}: {str(e)}")
                    raise

            # Should not reach here, but just in case
            if last_exception:
                raise last_exception

        return wrapper  # type: ignore
    return decorator


# Dataset IDs and metadata
DATASETS = {
    "GSE19784": {
        "title": "NDMM_320samples",
        "platform": "Affymetrix U133Plus2",
        "n_samples": 320,
        "description": "Newly diagnosed multiple myeloma",
    },
    "GSE39754": {
        "title": "Healthy_MM_exon",
        "platform": "Agilent exon arrays",
        "description": "Healthy controls and MM samples",
    },
    "GSE2658": {
        "title": "Pretreatment_MM",
        "platform": "Affymetrix U133Plus2",
        "description": "Pre-treatment multiple myeloma",
    },
}

CLINICAL_FIELDS = {
    "OS": ["os_time_months", "overall_survival_months", "os.time", "survival_time"],
    "OS_event": ["os_event", "overall_survival", "vital_status", "event_overall"],
    "PFS": ["pfs_time_months", "progression_free_survival", "pfs.time", "efs_time"],
    "PFS_event": ["pfs_event", "progression", "event_progression", "relapse_event"],
    "ISS": ["iss_stage", "iss.stage", "stage_iss"],
    "Cytogenetics": ["cyto", "cytogenetics", "fish_result", "del17p", "t1114"],
}


class GEODownloader:
    """Download and parse GEO datasets for MM."""

    def __init__(self, output_dir: str = "./data/raw"):
        """
        Initialize downloader.

        Parameters
        ----------
        output_dir : str
            Base directory for raw data storage.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = {}

    @retry_with_backoff(max_retries=3, base_delay=5.0, max_delay=60.0, timeout=30.0)
    def _fetch_geo_with_retry(self, accession: str, cache_dir: str) -> GEOparse.GEOTypes.GSeries:
        """
        Internal method to fetch GEO data with retry logic.

        Parameters
        ----------
        accession : str
            GEO accession (e.g., "GSE19784").
        cache_dir : str
            Cache directory for GEO files.

        Returns
        -------
        GEOparse.GEOTypes.GSeries
            Parsed GEO Series object.

        Raises
        ------
        Exception
            If all retry attempts fail.
        """
        logger.info(f"Fetching {accession}...")
        geo = GEOparse.get_GEO(
            geo=accession,
            destdir=cache_dir,
            silent=False,
            keep_series_matrix=True,
        )
        return geo

    def download_geo_dataset(self, accession: str) -> Optional[GEOparse.GEOTypes.GSeries]:
        """
        Download GEO dataset using GEOparse with retry logic.

        Parameters
        ----------
        accession : str
            GEO accession (e.g., "GSE19784").

        Returns
        -------
        Optional[GEOparse.GEOTypes.GSeries]
            Parsed GEO Series object or None if download fails after all retries.
        """
        try:
            cache_dir = str(self.output_dir / "geo_cache")
            geo = self._fetch_geo_with_retry(accession, cache_dir)
            logger.info(f"Successfully downloaded {accession}")
            return geo
        except Exception as e:
            logger.error(
                f"Failed to download {accession} after all retry attempts. "
                f"Last error: {str(e)}"
            )
            return None

    def extract_phenotype_data(self, geo: GEOparse.GEOTypes.GSeries) -> pd.DataFrame:
        """
        Extract phenotype/clinical data from GEO object.

        Parameters
        ----------
        geo : GEOparse.GEOTypes.GSeries
            Parsed GEO Series object.

        Returns
        -------
        pd.DataFrame
            Sample metadata with clinical annotations.
        """
        phenotype_df = geo.phenotype_data.copy()
        phenotype_df.index.name = "sample_id"
        phenotype_df = phenotype_df.reset_index()
        return phenotype_df

    def standardize_clinical_metadata(
        self, phenotype_df: pd.DataFrame, dataset_id: str
    ) -> pd.DataFrame:
        """
        Standardize clinical metadata across datasets.

        Maps dataset-specific column names to standard names: OS, OS_event, PFS, PFS_event, ISS, Cytogenetics.

        Parameters
        ----------
        phenotype_df : pd.DataFrame
            Raw phenotype data from GEO.
        dataset_id : str
            GEO accession for context-specific parsing.

        Returns
        -------
        pd.DataFrame
            Standardized metadata with canonical column names.
        """
        std_df = phenotype_df.copy()

        # Standardize column names by finding matches
        for std_col, possible_names in CLINICAL_FIELDS.items():
            existing_col = None
            for col in std_df.columns:
                col_lower = str(col).lower()
                if any(pname.lower() in col_lower for pname in possible_names):
                    existing_col = col
                    break

            if existing_col and existing_col != std_col:
                std_df = std_df.rename(columns={existing_col: std_col})

        # Standardize numeric fields
        for col in ["OS", "PFS"]:
            if col in std_df.columns:
                std_df[col] = pd.to_numeric(std_df[col], errors="coerce")

        # Standardize event indicators
        for col in ["OS_event", "PFS_event"]:
            if col in std_df.columns:
                std_df[col] = std_df[col].astype(str).str.lower()
                std_df[col] = std_df[col].map(
                    {"1": 1, "true": 1, "yes": 1, "event": 1, "0": 0, "false": 0, "no": 0}
                )

        # Add dataset identifier
        std_df["dataset_id"] = dataset_id

        return std_df

    def process_gse19784(self, phenotype_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process GSE19784-specific metadata.

        Parameters
        ----------
        phenotype_df : pd.DataFrame
            Raw phenotype from GEO.

        Returns
        -------
        pd.DataFrame
            Standardized metadata.
        """
        df = phenotype_df.copy()
        df = self.standardize_clinical_metadata(df, "GSE19784")

        # Extract ISS stage from title if present
        if "title" in df.columns:
            df["ISS"] = df["title"].str.extract(r"ISS\s*([1-3])", expand=False)

        return df

    def process_gse39754(self, phenotype_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process GSE39754-specific metadata (healthy + MM).

        Parameters
        ----------
        phenotype_df : pd.DataFrame
            Raw phenotype from GEO.

        Returns
        -------
        pd.DataFrame
            Standardized metadata with disease status.
        """
        df = phenotype_df.copy()
        df = self.standardize_clinical_metadata(df, "GSE39754")

        # Add disease status (healthy vs MM)
        if "title" in df.columns:
            df["disease_status"] = df["title"].str.extract(
                r"(healthy|normal|control|mm|myeloma)", expand=False
            )

        return df

    def process_gse2658(self, phenotype_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process GSE2658-specific metadata (pre-treatment MM).

        Parameters
        ----------
        phenotype_df : pd.DataFrame
            Raw phenotype from GEO.

        Returns
        -------
        pd.DataFrame
            Standardized metadata.
        """
        df = phenotype_df.copy()
        df = self.standardize_clinical_metadata(df, "GSE2658")

        return df

    def save_raw_data(
        self,
        accession: str,
        expression_matrix: pd.DataFrame,
        phenotype_df: pd.DataFrame,
    ) -> None:
        """
        Save raw expression matrix and phenotype data as Parquet.

        Parameters
        ----------
        accession : str
            GEO accession ID.
        expression_matrix : pd.DataFrame
            Gene expression matrix (genes × samples).
        phenotype_df : pd.DataFrame
            Sample metadata.
        """
        dataset_dir = self.output_dir / accession
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Save expression matrix
        expr_file = dataset_dir / f"{accession}_expression.parquet"
        expression_matrix.to_parquet(expr_file, compression="snappy")
        logger.info(f"Saved expression matrix: {expr_file}")

        # Save phenotype data
        pheno_file = dataset_dir / f"{accession}_phenotype.csv"
        phenotype_df.to_csv(pheno_file, index=False)
        logger.info(f"Saved phenotype data: {pheno_file}")

        # Save metadata JSON
        meta_file = dataset_dir / f"{accession}_metadata.json"
        meta_info = {
            "accession": accession,
            "downloaded": datetime.now().isoformat(),
            "n_samples": expression_matrix.shape[1],
            "n_genes": expression_matrix.shape[0],
            "phenotype_columns": phenotype_df.columns.tolist(),
        }
        with open(meta_file, "w") as f:
            json.dump(meta_info, f, indent=2)
        logger.info(f"Saved metadata: {meta_file}")

    def download_and_process(self, accession: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Download and process a single GEO dataset.

        Parameters
        ----------
        accession : str
            GEO accession ID.

        Returns
        -------
        Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]
            (expression_matrix, phenotype_df) or (None, None) if failed.
        """
        # Download
        geo = self.download_geo_dataset(accession)
        if geo is None:
            return None, None

        # Extract phenotype
        phenotype_df = self.extract_phenotype_data(geo)

        # Dataset-specific processing
        if accession == "GSE19784":
            phenotype_df = self.process_gse19784(phenotype_df)
        elif accession == "GSE39754":
            phenotype_df = self.process_gse39754(phenotype_df)
        elif accession == "GSE2658":
            phenotype_df = self.process_gse2658(phenotype_df)

        # Extract expression matrix
        expression_matrix = geo.pivot_samples("VALUE").T  # genes × samples
        expression_matrix.index.name = "gene_id"
        expression_matrix.columns.name = "sample_id"

        # Save
        self.save_raw_data(accession, expression_matrix, phenotype_df)

        self.metadata[accession] = {
            "n_samples": expression_matrix.shape[1],
            "n_genes": expression_matrix.shape[0],
            "phenotype_columns": phenotype_df.columns.tolist(),
        }

        return expression_matrix, phenotype_df

    def download_all(self) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Download and process all registered datasets.

        Returns
        -------
        Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]
            Dictionary mapping accession → (expression, phenotype).
        """
        results = {}
        for accession in DATASETS.keys():
            expr, pheno = self.download_and_process(accession)
            if expr is not None and pheno is not None:
                results[accession] = (expr, pheno)

        # Save overall metadata
        meta_file = self.output_dir / "download_metadata.json"
        with open(meta_file, "w") as f:
            json.dump(self.metadata, f, indent=2)
        logger.info(f"Saved overall metadata: {meta_file}")

        return results

    def _print_commpass_instructions(self) -> None:
        """
        Print step-by-step instructions for manual CoMMpass dataset download.
        """
        instructions = """
╔════════════════════════════════════════════════════════════════════════════╗
║                   MMRF CoMMpass (IA21) Dataset Download Guide             ║
╚════════════════════════════════════════════════════════════════════════════╝

IMPORTANT: The MMRF CoMMpass RNA-seq dataset requires manual download from the
MMRF Researcher Gateway. Automated download is not available.

STEP-BY-STEP REGISTRATION AND DOWNLOAD:
───────────────────────────────────────

1. VISIT THE MMRF RESEARCHER GATEWAY:
   https://research.themmrf.org/

2. REGISTER OR LOG IN:
   - Create a new account with your institution email
   - Verify your email address
   - Complete researcher profile information

3. ACCEPT DATA USE AGREEMENT:
   - Review and accept the MMRF Data Use Agreement (DUA)
   - Specify intended research use

4. DOWNLOAD THE FOLLOWING FILES:

   a) Gene Expression Data (REQUIRED):
      - File: MMRF_CoMMpass_IA21_E74GTF_Salmon_entrezID_TPM_matrix.tsv
      - Size: ~150-200 MB (expected)
      - Format: Gene ID × Sample matrix with TPM values

   b) Patient Clinical Data (REQUIRED):
      - File: MMRF_CoMMpass_IA21_PER_PATIENT.tsv
      - Contains: Demographics, ISS stage, cytogenetics, OS/PFS

   c) Treatment Response Data (RECOMMENDED):
      - File: MMRF_CoMMpass_IA21_STAND_ALONE_TRTRESP.tsv
      - Contains: Treatment type, response classification, outcomes

5. AFTER DOWNLOAD:
   Place downloaded files in a directory, then run:

   python download_geo.py --commpass-dir /path/to/downloaded/files

   OR specify the directory when calling download_commpass():

   downloader.download_commpass(commpass_dir="/path/to/files")

EXPECTED FILE LOCATIONS:
───────────────────────
Place all three files in the same directory:

  /path/to/your/commpass/
  ├── MMRF_CoMMpass_IA21_E74GTF_Salmon_entrezID_TPM_matrix.tsv
  ├── MMRF_CoMMpass_IA21_PER_PATIENT.tsv
  └── MMRF_CoMMpass_IA21_STAND_ALONE_TRTRESP.tsv

DATA ACCESS AND PRIVACY:
───────────────────────
- You must have institutional affiliation
- Data is subject to MMRF Data Use Agreement
- Results must acknowledge MMRF in publications
- Individual-level data must NOT be shared publicly

QUESTIONS:
──────────
Contact MMRF Researcher Gateway support:
https://research.themmrf.org/contact
        """
        logger.info(instructions)

    def _validate_commpass_files(self, commpass_dir: Path) -> Tuple[bool, List[str]]:
        """
        Validate that required CoMMpass files exist in the provided directory.

        Parameters
        ----------
        commpass_dir : Path
            Directory containing CoMMpass files.

        Returns
        -------
        Tuple[bool, List[str]]
            (all_files_exist, missing_files_list)
        """
        required_files = [
            "MMRF_CoMMpass_IA21_E74GTF_Salmon_entrezID_TPM_matrix.tsv",
            "MMRF_CoMMpass_IA21_PER_PATIENT.tsv",
            "MMRF_CoMMpass_IA21_STAND_ALONE_TRTRESP.tsv",
        ]

        missing_files = []
        for filename in required_files:
            filepath = commpass_dir / filename
            if not filepath.exists():
                missing_files.append(filename)
                logger.warning(f"Missing: {filename}")
            else:
                logger.info(f"Found: {filename}")

        return len(missing_files) == 0, missing_files

    def _load_commpass_expression(self, filepath: Path) -> Optional[pd.DataFrame]:
        """
        Load CoMMpass gene expression matrix.

        Parameters
        ----------
        filepath : Path
            Path to expression matrix TSV file.

        Returns
        -------
        Optional[pd.DataFrame]
            Expression matrix with genes as rows, samples as columns.
        """
        try:
            logger.info(f"Loading expression matrix: {filepath.name}")
            df = pd.read_csv(filepath, sep="\t", index_col=0)
            logger.info(
                f"Loaded expression matrix: {df.shape[0]} genes × {df.shape[1]} samples"
            )
            return df
        except Exception as e:
            logger.error(f"Failed to load expression matrix: {str(e)}")
            return None

    def _load_commpass_clinical(self, filepath: Path) -> Optional[pd.DataFrame]:
        """
        Load and standardize CoMMpass clinical data.

        Parameters
        ----------
        filepath : Path
            Path to clinical metadata TSV file.

        Returns
        -------
        Optional[pd.DataFrame]
            Standardized clinical metadata.
        """
        try:
            logger.info(f"Loading clinical data: {filepath.name}")
            df = pd.read_csv(filepath, sep="\t")
            logger.info(f"Loaded clinical data: {len(df)} patients")

            # Standardize to match GEO dataset format
            df = self.standardize_clinical_metadata(df, "MMRF_CoMMpass_IA21")
            logger.info(f"Standardized clinical metadata with {len(df.columns)} columns")

            return df
        except Exception as e:
            logger.error(f"Failed to load clinical data: {str(e)}")
            return None

    def _load_commpass_treatment(self, filepath: Path) -> Optional[pd.DataFrame]:
        """
        Load CoMMpass treatment response data.

        Parameters
        ----------
        filepath : Path
            Path to treatment response TSV file.

        Returns
        -------
        Optional[pd.DataFrame]
            Treatment response metadata.
        """
        try:
            logger.info(f"Loading treatment response data: {filepath.name}")
            df = pd.read_csv(filepath, sep="\t")
            logger.info(f"Loaded treatment response: {len(df)} records")
            return df
        except Exception as e:
            logger.error(f"Failed to load treatment response data: {str(e)}")
            return None

    def download_commpass(
        self, commpass_dir: Optional[str] = None
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Download and process MMRF CoMMpass IA21 dataset from manually downloaded files.

        The CoMMpass dataset requires manual registration and download from the MMRF
        Researcher Gateway (https://research.themmrf.org). This method loads, validates,
        and standardizes the manually downloaded files.

        Parameters
        ----------
        commpass_dir : Optional[str]
            Path to directory containing CoMMpass files. If None, prints instructions.
            Expected files:
            - MMRF_CoMMpass_IA21_E74GTF_Salmon_entrezID_TPM_matrix.tsv (expression)
            - MMRF_CoMMpass_IA21_PER_PATIENT.tsv (clinical)
            - MMRF_CoMMpass_IA21_STAND_ALONE_TRTRESP.tsv (treatment response)

        Returns
        -------
        Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]
            (expression_matrix, clinical_df, treatment_df) or (None, None, None) if failed.
        """
        # Print instructions if no directory provided
        if commpass_dir is None:
            self._print_commpass_instructions()
            return None, None, None

        commpass_path = Path(commpass_dir)

        # Validate directory exists
        if not commpass_path.exists():
            logger.error(f"CoMMpass directory not found: {commpass_path}")
            self._print_commpass_instructions()
            return None, None, None

        if not commpass_path.is_dir():
            logger.error(f"CoMMpass path is not a directory: {commpass_path}")
            return None, None, None

        # Validate required files
        logger.info(f"Validating CoMMpass files in: {commpass_path}")
        files_valid, missing_files = self._validate_commpass_files(commpass_path)

        if not files_valid:
            logger.error(f"Missing required CoMMpass files: {', '.join(missing_files)}")
            self._print_commpass_instructions()
            return None, None, None

        # Load all three data types
        expression_path = commpass_path / "MMRF_CoMMpass_IA21_E74GTF_Salmon_entrezID_TPM_matrix.tsv"
        clinical_path = commpass_path / "MMRF_CoMMpass_IA21_PER_PATIENT.tsv"
        treatment_path = commpass_path / "MMRF_CoMMpass_IA21_STAND_ALONE_TRTRESP.tsv"

        expression_df = self._load_commpass_expression(expression_path)
        clinical_df = self._load_commpass_clinical(clinical_path)
        treatment_df = self._load_commpass_treatment(treatment_path)

        # Check if all loaded successfully
        if expression_df is None or clinical_df is None or treatment_df is None:
            logger.error("Failed to load one or more CoMMpass files")
            return None, None, None

        # Save processed data
        try:
            dataset_dir = self.output_dir / "MMRF_CoMMpass_IA21"
            dataset_dir.mkdir(parents=True, exist_ok=True)

            # Save expression matrix
            expr_file = dataset_dir / "MMRF_CoMMpass_IA21_expression.parquet"
            expression_df.to_parquet(expr_file, compression="snappy")
            logger.info(f"Saved expression matrix: {expr_file}")

            # Save clinical data
            clinical_file = dataset_dir / "MMRF_CoMMpass_IA21_clinical.csv"
            clinical_df.to_csv(clinical_file, index=False)
            logger.info(f"Saved clinical data: {clinical_file}")

            # Save treatment data
            treatment_file = dataset_dir / "MMRF_CoMMpass_IA21_treatment.csv"
            treatment_df.to_csv(treatment_file, index=False)
            logger.info(f"Saved treatment response data: {treatment_file}")

            # Save metadata
            meta_file = dataset_dir / "MMRF_CoMMpass_IA21_metadata.json"
            meta_info = {
                "accession": "MMRF_CoMMpass_IA21",
                "loaded_from": str(commpass_path),
                "loaded_at": datetime.now().isoformat(),
                "n_samples_expr": expression_df.shape[1],
                "n_genes": expression_df.shape[0],
                "n_patients_clinical": len(clinical_df),
                "n_treatment_records": len(treatment_df),
                "clinical_columns": clinical_df.columns.tolist(),
            }
            with open(meta_file, "w") as f:
                json.dump(meta_info, f, indent=2)
            logger.info(f"Saved metadata: {meta_file}")

            logger.info(
                f"Successfully processed CoMMpass dataset: {expression_df.shape[0]} genes, "
                f"{expression_df.shape[1]} samples, {len(clinical_df)} patients"
            )

            self.metadata["MMRF_CoMMpass_IA21"] = {
                "n_samples": expression_df.shape[1],
                "n_genes": expression_df.shape[0],
                "n_patients": len(clinical_df),
                "clinical_columns": clinical_df.columns.tolist(),
            }

            return expression_df, clinical_df, treatment_df

        except Exception as e:
            logger.error(f"Failed to save CoMMpass data: {str(e)}")
            return None, None, None

    def download_commpass_info(self) -> Optional[pd.DataFrame]:
        """
        Legacy method for backward compatibility.
        Downloads and processes CoMMpass data from manually downloaded files.

        Deprecated: Use download_commpass() instead.

        Returns
        -------
        Optional[pd.DataFrame]
            Clinical metadata or None if unavailable.
        """
        logger.warning(
            "download_commpass_info() is deprecated. Use download_commpass(commpass_dir=...) instead."
        )

        # Try to load from default location
        commpass_dir = self.output_dir / "COMMPASS"
        if commpass_dir.exists():
            _, clinical_df, _ = self.download_commpass(str(commpass_dir))
            return clinical_df

        # If no default location, print instructions
        self._print_commpass_instructions()
        return None


def main():
    """
    Download all datasets.

    Supports command-line arguments:
    --commpass-dir PATH  : Path to manually downloaded CoMMpass files
    --help              : Print usage information
    """
    import sys

    downloader = GEODownloader(output_dir="./data/raw")

    # Parse command-line arguments
    commpass_dir = None
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--commpass-dir" and i + 1 < len(sys.argv[1:]):
            commpass_dir = sys.argv[i + 2]
        elif arg == "--help":
            print(
                "Usage: python download_geo.py [OPTIONS]\n\n"
                "Options:\n"
                "  --commpass-dir PATH  Path to directory with manually downloaded CoMMpass files\n"
                "  --help               Show this help message\n\n"
                "Examples:\n"
                "  python download_geo.py\n"
                "  python download_geo.py --commpass-dir /path/to/commpass/files\n"
            )
            return

    # Download GEO datasets
    logger.info("Starting GEO dataset downloads...")
    results = downloader.download_all()
    logger.info(f"Downloaded {len(results)} GEO datasets.")

    # Process CoMMpass if directory provided
    if commpass_dir:
        logger.info(f"Processing CoMMpass dataset from: {commpass_dir}")
        expr, clinical, treatment = downloader.download_commpass(commpass_dir)
        if expr is not None:
            logger.info("CoMMpass dataset successfully processed.")
        else:
            logger.error("Failed to process CoMMpass dataset.")
    else:
        logger.info(
            "CoMMpass dataset not specified. To process CoMMpass data, use:\n"
            "  python download_geo.py --commpass-dir /path/to/files\n"
            "For manual download instructions, run:\n"
            "  python -c 'from download_geo import GEODownloader; "
            "GEODownloader()._print_commpass_instructions()'"
        )

    logger.info("All downloads complete.")


if __name__ == "__main__":
    main()
