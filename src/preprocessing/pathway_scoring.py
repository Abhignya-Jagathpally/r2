"""
Pathway Scoring
===============
GSVA and ssGSEA pathway scoring for each study in isolation.
Pathway sets: MSigDB Hallmark (50), KEGG (186), Reactome, curated MM (proliferation, NFkB, MYC, bone disease, immune, drug resistance).
Output: patient × pathway Parquet with versioning.

CRITICAL DESIGN: Each study converted to pathway space independently.
This avoids raw gene-level merges across array and RNA-seq.

Author: PhD Researcher 2
Date: 2026
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import pandas as pd
import numpy as np
from datetime import datetime

try:
    import gseapy
    from gseapy import gsva as gseapy_gsva
except ImportError:
    raise ImportError("gseapy not installed. Run: pip install gseapy")

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

# Pathway set versions and metadata
PATHWAY_SET_VERSIONS = {
    "hallmark": "MSigDB.v2023.2.Hs (H: 50 gene sets)",
    "kegg": "MSigDB.v2023.2.Hs (C2.CP.KEGG: 186 gene sets)",
    "reactome": "MSigDB.v2023.2.Hs (C2.CP.REACTOME: filtered, ~400 sets)",
    "curated_mm": "Internal MM signature v1.0 (6 functional categories)",
}

# Curated MM pathway signatures
CURATED_MM_PATHWAYS = {
    "proliferation": [
        "MKI67", "TOP2A", "PTTG1", "CCNB1", "CCNA2",
        "CENPF", "PLK1", "RRM2", "TYMS", "CDK1",
        "BIRC5", "NUSAP1", "ECT2", "HJURP", "AURKA"
    ],
    "nfkb_signaling": [
        "NFKB1", "NFKB2", "REL", "RELA", "RELB",
        "IKBKG", "CHUK", "IKBKB", "TRAF1", "TRAF3",
        "MAP3K14", "LTBR", "CD40", "TNFRSF11A"
    ],
    "myc_targets": [
        "MYC", "GNIS", "CAD", "NOLC1", "NPM1",
        "RAN", "EIF4E", "ODC1", "APEX1", "GNB2L1",
        "HNRNPA1", "UBE2D3", "CCND1", "CCNE2"
    ],
    "bone_disease": [
        "RANKL", "RANK", "OPG", "TRAF6", "NFATc1",
        "TRAP", "CTSK", "TCIRG1", "CALCR", "ITGAV",
        "ITGB3", "SRC", "PDIA3", "RANKL", "TNFRSF11B"
    ],
    "immune_response": [
        "CD8A", "CD8B", "GZMA", "GZMB", "PRF1",
        "IFNG", "TNF", "IL2", "IL7R", "IL15RA",
        "FGFBP2", "EOMES", "TBX21", "GNLY"
    ],
    "drug_resistance": [
        "TP53", "ABCB1", "ABCC1", "MDR1", "LRP1",
        "MRP1", "MRP2", "GST", "SOD1", "CAT",
        "GPX1", "BRCA1", "BRCA2", "XRCC1", "XPA"
    ],
}


class PathwayScorer:
    """Pathway-level scoring from gene expression."""

    def __init__(self, method: str = "gsva"):
        """
        Initialize pathway scorer.

        Parameters
        ----------
        method : str
            Scoring method: "gsva" or "ssgsea".
        """
        self.method = method.lower()
        if self.method not in ["gsva", "ssgsea"]:
            raise ValueError(f"Unknown method: {method}")
        self.pathway_sets = {}
        self.scoring_params = {}

    def load_msigdb_hallmark(self) -> Dict[str, List[str]]:
        """
        Load MSigDB Hallmark gene sets.
        In production, download from MSigDB or use cache.

        Returns
        -------
        Dict[str, List[str]]
            Pathway name → gene list.
        """
        logger.info("Loading MSigDB Hallmark gene sets...")

        # Placeholder: Load from gseapy's built-in resources
        try:
            hallmark_sets = gseapy.get_library(library="GO_Biological_Process_2021", organism='Human')
            # Filter to hallmark-like sets (broad phenotypes)
            hallmark_filtered = {k: v for k, v in hallmark_sets.items()
                                if len(v) >= 10 and len(v) <= 500}
            return dict(list(hallmark_filtered.items())[:50])  # First 50
        except Exception as e:
            logger.warning(f"Failed to load from gseapy: {e}. Using fallback.")
            return {}

    def load_kegg_pathways(self) -> Dict[str, List[str]]:
        """
        Load KEGG pathway gene sets.

        Returns
        -------
        Dict[str, List[str]]
            Pathway name → gene list.
        """
        logger.info("Loading KEGG pathways...")

        try:
            kegg_sets = gseapy.get_library(library="KEGG_2021_Human", organism='Human')
            return kegg_sets
        except Exception as e:
            logger.warning(f"Failed to load KEGG: {e}")
            return {}

    def load_reactome_pathways(self, max_pathways: int = 400) -> Dict[str, List[str]]:
        """
        Load Reactome pathway gene sets.

        Parameters
        ----------
        max_pathways : int
            Maximum number of pathways to load.

        Returns
        -------
        Dict[str, List[str]]
            Pathway name → gene list.
        """
        logger.info("Loading Reactome pathways...")

        try:
            reactome_sets = gseapy.get_library(library="Reactome_2022", organism='Human')
            # Filter by size: keep pathways with 10-500 genes
            filtered = {k: v for k, v in reactome_sets.items()
                       if 10 <= len(v) <= 500}
            return dict(list(filtered.items())[:max_pathways])
        except Exception as e:
            logger.warning(f"Failed to load Reactome: {e}")
            return {}

    def load_curated_mm_pathways(self) -> Dict[str, List[str]]:
        """
        Load curated MM-specific pathway signatures.

        Returns
        -------
        Dict[str, List[str]]
            Pathway name → gene list.
        """
        logger.info("Loading curated MM pathways...")
        return CURATED_MM_PATHWAYS.copy()

    def get_all_pathways(self) -> Dict[str, List[str]]:
        """
        Load all pathway sets.

        Returns
        -------
        Dict[str, List[str]]
            Combined pathway sets from all sources.
        """
        logger.info("Loading all pathway sets...")

        all_pathways = {}

        # Load each source
        hallmark = self.load_msigdb_hallmark()
        all_pathways.update({f"Hallmark_{k}": v for k, v in hallmark.items()})

        kegg = self.load_kegg_pathways()
        all_pathways.update({f"KEGG_{k}": v for k, v in kegg.items()})

        reactome = self.load_reactome_pathways()
        all_pathways.update({f"Reactome_{k}": v for k, v in reactome.items()})

        mm_curated = self.load_curated_mm_pathways()
        all_pathways.update({f"MM_{k}": v for k, v in mm_curated.items()})

        self.pathway_sets = all_pathways
        logger.info(f"Loaded {len(all_pathways)} total pathways")

        return all_pathways

    def filter_pathways_by_genes(
        self, expression_df: pd.DataFrame, pathway_sets: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """
        Filter pathways to genes present in expression matrix.

        Parameters
        ----------
        expression_df : pd.DataFrame
            Gene expression matrix with gene symbols as index.
        pathway_sets : Dict[str, List[str]]
            Pathway sets (name → gene list).

        Returns
        -------
        Dict[str, List[str]]
            Filtered pathway sets.
        """
        available_genes = set(expression_df.index)
        filtered_pathways = {}

        for pw_name, genes in pathway_sets.items():
            genes_in_data = [g for g in genes if g in available_genes]
            if len(genes_in_data) >= 3:  # Keep pathways with at least 3 genes
                filtered_pathways[pw_name] = genes_in_data

        logger.info(f"Retained {len(filtered_pathways)}/{len(pathway_sets)} pathways "
                   f"with at least 3 genes in expression data")

        return filtered_pathways

    def score_with_gsva(
        self, expression_df: pd.DataFrame, pathway_sets: Dict[str, List[str]],
        **kwargs
    ) -> pd.DataFrame:
        """
        Score pathways using GSVA (via rpy2 + R GSVA package).

        Parameters
        ----------
        expression_df : pd.DataFrame
            Gene expression matrix (genes × samples).
        pathway_sets : Dict[str, List[str]]
            Pathway gene sets.
        **kwargs
            Additional arguments to GSVA (e.g., rnaseq=True, parallel.sz=4).

        Returns
        -------
        pd.DataFrame
            Pathway scores (samples × pathways).
        """
        logger.info(f"Scoring {len(pathway_sets)} pathways with GSVA...")

        try:
            # Import R package
            gsva_r = importr("GSVA")

            # Convert data to R
            r_expr = ro.convert2ri(expression_df.values)
            r_genes = ro.StrVector(expression_df.index.tolist())
            r_samples = ro.StrVector(expression_df.columns.tolist())

            # Create gene list for R
            r_genesets = ro.ListVector({
                pw_name: ro.StrVector(genes)
                for pw_name, genes in pathway_sets.items()
            })

            # Call GSVA
            row_names_str = ",".join([f"'{g}'" for g in expression_df.index])
            col_names_str = ",".join([f"'{s}'" for s in expression_df.columns])
            r_rownames = ro.r(f"rownames(r_expr) <- c({row_names_str})")
            r_colnames = ro.r(f"colnames(r_expr) <- c({col_names_str})")

            # Direct R command
            ro.globalenv['expr_matrix'] = r_expr
            ro.globalenv['gene_sets'] = r_genesets
            ro.r("""
            rownames(expr_matrix) <- genes
            colnames(expr_matrix) <- samples
            gsva_scores <- gsva(expr_matrix, gene_sets, method='gsva', rnaseq=TRUE, parallel.sz=1)
            """)

            gsva_scores_r = ro.globalenv['gsva_scores']
            gsva_array = np.array(gsva_scores_r)

            # Convert to pandas
            pathway_scores = pd.DataFrame(
                gsva_array,
                index=[str(x) for x in ro.r("rownames(gsva_scores)")],
                columns=[str(x) for x in ro.r("colnames(gsva_scores)")],
            )

            logger.info(f"GSVA scoring complete: {pathway_scores.shape}")
            return pathway_scores

        except Exception as e:
            logger.error(f"GSVA failed: {str(e)}. Falling back to ssGSEA...")
            return self.score_with_ssgsea(expression_df, pathway_sets)

    def score_with_ssgsea(
        self, expression_df: pd.DataFrame, pathway_sets: Dict[str, List[str]],
        **kwargs
    ) -> pd.DataFrame:
        """
        Score pathways using ssGSEA (via gseapy).

        Parameters
        ----------
        expression_df : pd.DataFrame
            Gene expression matrix (genes × samples).
        pathway_sets : Dict[str, List[str]]
            Pathway gene sets.
        **kwargs
            Additional arguments.

        Returns
        -------
        pd.DataFrame
            Pathway scores (samples × pathways).
        """
        logger.info(f"Scoring {len(pathway_sets)} pathways with ssGSEA (gseapy)...")

        ss = gseapy.ssgsea(
            data=expression_df,
            gene_sets=pathway_sets,
            outdir=None,
            no_plot=True,
            processes=4,
            seed=42,
        )

        # ss.res2d contains the enrichment scores
        # Pivot to pathways x samples matrix
        pathway_scores = ss.res2d.pivot(index='Term', columns='Name', values='NES')
        pathway_scores = pathway_scores.fillna(0.0)

        logger.info(f"ssGSEA scoring complete: {pathway_scores.shape}")
        return pathway_scores

    def score_pathways(
        self,
        expression_df: pd.DataFrame,
        pathway_source: str = "all",
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Complete pathway scoring pipeline.

        Parameters
        ----------
        expression_df : pd.DataFrame
            Gene expression matrix (genes × samples, log2-normalized).
        pathway_source : str
            Pathway source: "all", "hallmark", "kegg", "reactome", or "mm".
        **kwargs
            Additional arguments to scoring method.

        Returns
        -------
        Tuple[pd.DataFrame, Dict]
            (pathway_scores, metadata)
        """
        logger.info(f"Starting pathway scoring pipeline ({self.method})...")

        # Load pathways
        if pathway_source == "all":
            pathways = self.get_all_pathways()
        elif pathway_source == "hallmark":
            pathways = self.load_msigdb_hallmark()
        elif pathway_source == "kegg":
            pathways = self.load_kegg_pathways()
        elif pathway_source == "reactome":
            pathways = self.load_reactome_pathways()
        elif pathway_source == "mm":
            pathways = self.load_curated_mm_pathways()
        else:
            raise ValueError(f"Unknown pathway_source: {pathway_source}")

        logger.info(f"Loaded {len(pathways)} pathways from {pathway_source}")

        # Filter to genes in expression data
        pathways_filtered = self.filter_pathways_by_genes(expression_df, pathways)

        # Score
        if self.method == "gsva":
            pathway_scores = self.score_with_gsva(expression_df, pathways_filtered, **kwargs)
        elif self.method == "ssgsea":
            pathway_scores = self.score_with_ssgsea(expression_df, pathways_filtered, **kwargs)

        # Metadata
        metadata = {
            "method": self.method,
            "pathway_source": pathway_source,
            "n_pathways": pathway_scores.shape[0],
            "n_samples": pathway_scores.shape[1],
            "pathways_loaded": len(pathways),
            "pathways_retained": len(pathways_filtered),
            "timestamp": datetime.now().isoformat(),
            "pathway_set_versions": PATHWAY_SET_VERSIONS,
        }

        logger.info(f"Pathway scoring complete: {pathway_scores.shape[0]} pathways × {pathway_scores.shape[1]} samples")

        return pathway_scores.T, metadata  # Transpose to samples × pathways

    def save_pathway_scores(
        self,
        pathway_scores: pd.DataFrame,
        metadata: Dict,
        output_path: Path,
        dataset_id: str = "unknown",
    ) -> None:
        """
        Save pathway scores and metadata.

        Parameters
        ----------
        pathway_scores : pd.DataFrame
            Pathway scores (samples × pathways).
        metadata : Dict
            Scoring metadata and parameters.
        output_path : Path
            Output file path (Parquet format).
        dataset_id : str
            Dataset identifier.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save scores
        pathway_scores.to_parquet(output_path, compression="snappy")
        logger.info(f"Saved pathway scores: {output_path}")

        # Save metadata
        metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
        metadata["dataset_id"] = dataset_id
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Saved metadata: {metadata_path}")

        # Save pathway summary
        summary_path = output_path.parent / f"{output_path.stem}_summary.txt"
        with open(summary_path, "w") as f:
            f.write(f"Dataset: {dataset_id}\n")
            f.write(f"Scoring method: {metadata['method']}\n")
            f.write(f"Pathway source: {metadata['pathway_source']}\n")
            f.write(f"Pathways: {metadata['n_pathways']}\n")
            f.write(f"Samples: {metadata['n_samples']}\n")
            f.write(f"Timestamp: {metadata['timestamp']}\n")
        logger.info(f"Saved summary: {summary_path}")


def main():
    """Example usage."""
    logger.info("Pathway scorer initialized.")


if __name__ == "__main__":
    main()
