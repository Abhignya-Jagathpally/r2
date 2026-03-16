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
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import requests
from urllib.parse import urljoin

try:
    import GEOparse
except ImportError:
    raise ImportError("GEOparse not installed. Run: pip install GEOparse")

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

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

    def download_geo_dataset(self, accession: str) -> Optional[GEOparse.GEOTypes.GSeries]:
        """
        Download GEO dataset using GEOparse.

        Parameters
        ----------
        accession : str
            GEO accession (e.g., "GSE19784").

        Returns
        -------
        Optional[GEOparse.GEOTypes.GSeries]
            Parsed GEO Series object or None if download fails.
        """
        try:
            logger.info(f"Downloading {accession}...")
            geo = GEOparse.get_GEO(
                geo=accession,
                destdir=str(self.output_dir / "geo_cache"),
                silent=False,
                keep_series_matrix=True,
            )
            logger.info(f"Successfully downloaded {accession}")
            return geo
        except Exception as e:
            logger.error(f"Failed to download {accession}: {str(e)}")
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

    def download_commpass_info(self) -> Optional[pd.DataFrame]:
        """
        Download MMRF CoMMpass metadata (clinical info only; expression data requires manual download).

        Note: Full RNA-seq data requires MMRF Portal access. This retrieves available metadata.

        Returns
        -------
        Optional[pd.DataFrame]
            CoMMpass sample metadata or None if unavailable.
        """
        logger.info("CoMMpass data requires manual download from MMRF Portal.")
        logger.info("Please download from: https://research.themmrf.org/")
        logger.info("Expected file: MMRF_CoMMpass_IA21_PATIENT_VISIT.csv (clinical data)")
        logger.info("Expected file: gene expression matrix in standardized format")

        # Placeholder: If clinical file is provided manually
        commpass_clinical_path = self.output_dir / "COMMPASS" / "MMRF_CoMMpass_IA21_PATIENT_VISIT.csv"
        if commpass_clinical_path.exists():
            try:
                df = pd.read_csv(commpass_clinical_path)
                logger.info(f"Loaded CoMMpass clinical data: {len(df)} records")
                return df
            except Exception as e:
                logger.error(f"Failed to load CoMMpass data: {str(e)}")
                return None
        else:
            logger.warning(f"CoMMpass clinical file not found at {commpass_clinical_path}")
            return None


def main():
    """Download all datasets."""
    downloader = GEODownloader(output_dir="./data/raw")
    results = downloader.download_all()

    # Try to load CoMMpass metadata if available
    commpass_meta = downloader.download_commpass_info()

    logger.info(f"Download complete. Processed {len(results)} GEO datasets.")


if __name__ == "__main__":
    main()
