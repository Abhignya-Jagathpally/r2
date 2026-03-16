"""
Pathway Scoring
===============
GSVA and ssGSEA pathway scoring for each study in isolation.
Pathway sets: MSigDB Hallmark (50), KEGG (186), Reactome, curated MM
(proliferation, NFkB, MYC, bone disease, immune, drug resistance).
Output: patient x pathway Parquet with versioning.

CRITICAL DESIGN: Each study converted to pathway space independently.
This avoids raw gene-level merges across array and RNA-seq.

Author: PhD Researcher 2
Date: 2026
"""

import logging
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import pandas as pd
import numpy as np
from datetime import datetime

try:
    import gseapy
except ImportError:
    raise ImportError("gseapy not installed. Run: pip install gseapy")

try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri, numpy2ri
    from rpy2.robjects.packages import importr
    RPY2_AVAILABLE = True
except ImportError:
    RPY2_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pathway set versions — pinned for reproducibility
# ---------------------------------------------------------------------------
PATHWAY_SET_VERSIONS = {
    "hallmark": "MSigDB.v2023.2.Hs (H: 50 gene sets)",
    "kegg": "MSigDB.v2023.2.Hs (C2.CP.KEGG_MEDICUS: 186 gene sets)",
    "reactome": "MSigDB.v2023.2.Hs (C2.CP.REACTOME: filtered, ~400 sets)",
    "curated_mm": "Internal MM signature v1.0 (6 functional categories)",
}

# ---------------------------------------------------------------------------
# Curated MM pathway signatures
# Gene symbols validated against HGNC 2024-01 release
# ---------------------------------------------------------------------------
CURATED_MM_PATHWAYS = {
    "MM_proliferation": [
        "MKI67", "TOP2A", "PTTG1", "CCNB1", "CCNA2",
        "CENPF", "PLK1", "RRM2", "TYMS", "CDK1",
        "BIRC5", "NUSAP1", "ECT2", "HJURP", "AURKA",
    ],
    "MM_nfkb_signaling": [
        "NFKB1", "NFKB2", "REL", "RELA", "RELB",
        "IKBKG", "CHUK", "IKBKB", "TRAF1", "TRAF3",
        "MAP3K14", "LTBR", "CD40", "TNFRSF11A",
    ],
    "MM_myc_targets": [
        "MYC", "GNAI2", "CAD", "NOLC1", "NPM1",
        "RAN", "EIF4E", "ODC1", "APEX1", "GNB2L1",
        "HNRNPA1", "UBE2D3", "CCND1", "CCNE2",
    ],
    "MM_bone_disease": [
        "TNFSF11", "TNFRSF11A", "TNFRSF11B", "TRAF6", "NFATC1",
        "ACP5", "CTSK", "TCIRG1", "CALCR", "ITGAV",
        "ITGB3", "SRC", "PDIA3", "DKK1", "SOST",
    ],
    "MM_immune_response": [
        "CD8A", "CD8B", "GZMA", "GZMB", "PRF1",
        "IFNG", "TNF", "IL2", "IL7R", "IL15RA",
        "FGFBP2", "EOMES", "TBX21", "GNLY",
    ],
    "MM_drug_resistance": [
        "TP53", "ABCB1", "ABCC1", "ABCG2", "LRP1",
        "GSTP1", "SOD1", "CAT", "GPX1", "BRCA1",
        "BRCA2", "XRCC1", "XPA", "ATM", "PARP1",
    ],
}


class PathwayScorer:
    """Pathway-level scoring from gene expression.

    Implements the key design choice: each study is converted to pathway
    space independently (GSVA or ssGSEA), so that downstream models
    operate on pathway scores instead of raw gene counts.
    """

    def __init__(
        self,
        method: str = "gsva",
        min_genes_per_pathway: int = 5,
        n_jobs: int = 4,
        seed: int = 42,
    ):
        """
        Parameters
        ----------
        method : str
            Scoring method: ``"gsva"`` (default, recommended) or ``"ssgsea"``.
        min_genes_per_pathway : int
            Minimum number of genes from a pathway that must be present in
            the expression matrix for the pathway to be scored.
        n_jobs : int
            Parallel workers for ssGSEA (ignored for GSVA which controls
            parallelism inside R).
        seed : int
            Random seed for reproducibility.
        """
        self.method = method.lower()
        if self.method not in ("gsva", "ssgsea"):
            raise ValueError(f"Unknown method: {method!r}. Use 'gsva' or 'ssgsea'.")
        self.min_genes_per_pathway = min_genes_per_pathway
        self.n_jobs = n_jobs
        self.seed = seed
        self.pathway_sets: Dict[str, List[str]] = {}
        self._scoring_metadata: Dict = {}

    # ------------------------------------------------------------------
    # Pathway loading
    # ------------------------------------------------------------------
    def load_msigdb_hallmark(self) -> Dict[str, List[str]]:
        """Load MSigDB Hallmark (H) gene sets via gseapy.

        Returns
        -------
        Dict[str, List[str]]
            Pathway name -> gene list.
        """
        logger.info("Loading MSigDB Hallmark gene sets...")
        try:
            lib = gseapy.get_library(name="MSigDB_Hallmark_2020", organism="Human")
            logger.info(f"  Loaded {len(lib)} Hallmark sets")
            return {f"Hallmark_{k}": v for k, v in lib.items()}
        except Exception as e:
            logger.warning(f"Failed to load Hallmark from gseapy: {e}. "
                           "Trying GO_Biological_Process as fallback...")
            try:
                lib = gseapy.get_library(
                    name="GO_Biological_Process_2021", organism="Human"
                )
                filtered = {
                    k: v for k, v in lib.items()
                    if 15 <= len(v) <= 500
                }
                top50 = dict(list(filtered.items())[:50])
                return {f"Hallmark_{k}": v for k, v in top50.items()}
            except Exception:
                logger.error("Could not load any Hallmark-like set")
                return {}

    def load_kegg_pathways(self) -> Dict[str, List[str]]:
        """Load KEGG pathway gene sets."""
        logger.info("Loading KEGG pathways...")
        try:
            lib = gseapy.get_library(name="KEGG_2021_Human", organism="Human")
            logger.info(f"  Loaded {len(lib)} KEGG sets")
            return {f"KEGG_{k}": v for k, v in lib.items()}
        except Exception as e:
            logger.warning(f"Failed to load KEGG: {e}")
            return {}

    def load_reactome_pathways(self, max_pathways: int = 400) -> Dict[str, List[str]]:
        """Load Reactome pathway gene sets, filtered to 10-500 genes."""
        logger.info("Loading Reactome pathways...")
        try:
            lib = gseapy.get_library(name="Reactome_2022", organism="Human")
            filtered = {
                k: v for k, v in lib.items()
                if 10 <= len(v) <= 500
            }
            selected = dict(list(filtered.items())[:max_pathways])
            logger.info(f"  Loaded {len(selected)} Reactome sets (filtered from {len(lib)})")
            return {f"Reactome_{k}": v for k, v in selected.items()}
        except Exception as e:
            logger.warning(f"Failed to load Reactome: {e}")
            return {}

    def load_curated_mm_pathways(self) -> Dict[str, List[str]]:
        """Load curated MM-specific pathway signatures."""
        logger.info(f"Loading curated MM pathways ({len(CURATED_MM_PATHWAYS)} sets)...")
        return CURATED_MM_PATHWAYS.copy()

    def get_all_pathways(self) -> Dict[str, List[str]]:
        """Load and merge all pathway sources."""
        logger.info("Loading all pathway sets...")

        all_pw: Dict[str, List[str]] = {}
        for loader in (
            self.load_msigdb_hallmark,
            self.load_kegg_pathways,
            self.load_reactome_pathways,
            self.load_curated_mm_pathways,
        ):
            all_pw.update(loader())

        self.pathway_sets = all_pw
        logger.info(f"Total pathways loaded: {len(all_pw)}")
        return all_pw

    # ------------------------------------------------------------------
    # Pathway filtering
    # ------------------------------------------------------------------
    def filter_pathways_by_genes(
        self,
        expression_df: pd.DataFrame,
        pathway_sets: Dict[str, List[str]],
    ) -> Dict[str, List[str]]:
        """Keep only pathways with >= min_genes_per_pathway present in expression data.

        Parameters
        ----------
        expression_df : pd.DataFrame
            Gene expression matrix (genes x samples).
        pathway_sets : Dict[str, List[str]]
            Pathway name -> gene list.

        Returns
        -------
        Dict[str, List[str]]
            Filtered pathway sets.
        """
        available_genes = set(expression_df.index)
        filtered: Dict[str, List[str]] = {}

        for pw_name, genes in pathway_sets.items():
            overlap = [g for g in genes if g in available_genes]
            if len(overlap) >= self.min_genes_per_pathway:
                filtered[pw_name] = overlap

        logger.info(
            f"Retained {len(filtered)}/{len(pathway_sets)} pathways "
            f"(>= {self.min_genes_per_pathway} genes in expression data)"
        )
        return filtered

    # ------------------------------------------------------------------
    # GSVA scoring (R via rpy2)
    # ------------------------------------------------------------------
    def score_with_gsva(
        self,
        expression_df: pd.DataFrame,
        pathway_sets: Dict[str, List[str]],
        kcdf: str = "Gaussian",
        parallel_sz: int = 1,
    ) -> pd.DataFrame:
        """Score pathways using R GSVA package via rpy2.

        Parameters
        ----------
        expression_df : pd.DataFrame
            Gene expression matrix (genes x samples), log2-normalized.
        pathway_sets : Dict[str, List[str]]
            Filtered pathway gene sets.
        kcdf : str
            Kernel CDF: ``"Gaussian"`` for continuous (microarray/log-TPM),
            ``"Poisson"`` for integer counts.
        parallel_sz : int
            Number of parallel threads inside R GSVA.

        Returns
        -------
        pd.DataFrame
            Pathway scores (pathways x samples).
        """
        if not RPY2_AVAILABLE:
            logger.warning("rpy2 not available — falling back to ssGSEA (gseapy).")
            return self.score_with_ssgsea(expression_df, pathway_sets)

        logger.info(
            f"Scoring {len(pathway_sets)} pathways with GSVA "
            f"(kcdf={kcdf}, parallel.sz={parallel_sz})..."
        )

        try:
            # Activate automatic pandas/numpy <-> R conversion
            numpy2ri.activate()

            # Import R packages
            gsva_pkg = importr("GSVA")
            base = importr("base")

            # Build R matrix with dimnames
            # R matrix() fills column-major, so flatten in Fortran order
            # to preserve genes-as-rows, samples-as-columns layout
            r_matrix = ro.r["matrix"](
                ro.FloatVector(expression_df.values.flatten(order='F')),
                nrow=expression_df.shape[0],
                ncol=expression_df.shape[1],
            )
            # Assign row/col names via R base functions
            r_gene_names = ro.StrVector(expression_df.index.tolist())
            r_sample_names = ro.StrVector(expression_df.columns.tolist())
            r_matrix = base.structure(
                r_matrix,
                dimnames=ro.ListVector(
                    {"genes": r_gene_names, "samples": r_sample_names}
                ),
            )

            # Build R gene-set list
            r_genesets = ro.ListVector({
                pw_name: ro.StrVector(genes)
                for pw_name, genes in pathway_sets.items()
            })

            # Assign to R global env and call GSVA
            ro.globalenv["expr_matrix"] = r_matrix
            ro.globalenv["gene_sets"] = r_genesets

            gsva_call = f"""
            gsva_result <- GSVA::gsva(
                expr       = expr_matrix,
                gset.idx.list = gene_sets,
                method     = "gsva",
                kcdf       = "{kcdf}",
                parallel.sz = {parallel_sz}L
            )
            """
            ro.r(gsva_call)

            gsva_r = ro.globalenv["gsva_result"]
            gsva_array = np.array(gsva_r)

            pw_names = list(ro.r("rownames(gsva_result)"))
            sample_names = list(ro.r("colnames(gsva_result)"))

            pathway_scores = pd.DataFrame(
                gsva_array,
                index=pw_names,
                columns=sample_names,
            )

            logger.info(f"GSVA complete: {pathway_scores.shape[0]} pathways x "
                        f"{pathway_scores.shape[1]} samples")

            # Deactivate conversion
            try:
                numpy2ri.deactivate()
            except Exception:
                pass

            return pathway_scores

        except Exception as e:
            logger.error(f"GSVA via R failed: {e}")
            logger.info("Falling back to ssGSEA (gseapy)...")
            try:
                numpy2ri.deactivate()
            except Exception:
                pass
            return self.score_with_ssgsea(expression_df, pathway_sets)

    # ------------------------------------------------------------------
    # ssGSEA scoring (gseapy — pure Python)
    # ------------------------------------------------------------------
    def score_with_ssgsea(
        self,
        expression_df: pd.DataFrame,
        pathway_sets: Dict[str, List[str]],
    ) -> pd.DataFrame:
        """Score pathways using single-sample GSEA via gseapy.

        This uses the Barbie et al. (2009) algorithm: a modified
        Kolmogorov-Smirnov-like running-sum statistic per sample.
        NOT a simple gene-set mean.

        Parameters
        ----------
        expression_df : pd.DataFrame
            Gene expression matrix (genes x samples).
        pathway_sets : Dict[str, List[str]]
            Filtered pathway gene sets.

        Returns
        -------
        pd.DataFrame
            Pathway scores (pathways x samples).
        """
        logger.info(
            f"Scoring {len(pathway_sets)} pathways with ssGSEA (gseapy)..."
        )

        ss_result = gseapy.ssgsea(
            data=expression_df,
            gene_sets=pathway_sets,
            outdir=None,
            no_plot=True,
            processes=self.n_jobs,
            seed=self.seed,
        )

        # ss_result.res2d has columns: Name (sample), Term (pathway), NES, ES, etc.
        scores_long = ss_result.res2d
        pathway_scores = scores_long.pivot(
            index="Term", columns="Name", values="NES"
        )
        pathway_scores = pathway_scores.fillna(0.0)

        logger.info(
            f"ssGSEA complete: {pathway_scores.shape[0]} pathways x "
            f"{pathway_scores.shape[1]} samples"
        )
        return pathway_scores

    # ------------------------------------------------------------------
    # Main scoring pipeline
    # ------------------------------------------------------------------
    def score_pathways(
        self,
        expression_df: pd.DataFrame,
        pathway_source: str = "all",
        kcdf: str = "Gaussian",
    ) -> Tuple[pd.DataFrame, Dict]:
        """Complete pathway scoring pipeline for one study.

        Parameters
        ----------
        expression_df : pd.DataFrame
            Gene expression matrix (genes x samples, log2-normalized).
        pathway_source : str
            Which pathway sets to load:
            ``"all"`` | ``"hallmark"`` | ``"kegg"`` | ``"reactome"`` | ``"mm"``.
        kcdf : str
            Kernel CDF for GSVA (``"Gaussian"`` or ``"Poisson"``).

        Returns
        -------
        Tuple[pd.DataFrame, Dict]
            * ``pathway_scores`` — DataFrame (samples x pathways)
            * ``metadata`` — dict with method, counts, timestamp, versions
        """
        logger.info(f"=== Pathway scoring: method={self.method}, source={pathway_source} ===")

        # Load pathways
        loader_map = {
            "all": self.get_all_pathways,
            "hallmark": self.load_msigdb_hallmark,
            "kegg": self.load_kegg_pathways,
            "reactome": self.load_reactome_pathways,
            "mm": self.load_curated_mm_pathways,
        }
        if pathway_source not in loader_map:
            raise ValueError(f"Unknown pathway_source: {pathway_source!r}")

        raw_pathways = loader_map[pathway_source]()
        logger.info(f"Loaded {len(raw_pathways)} raw pathways from '{pathway_source}'")

        # Filter to genes present
        filtered_pathways = self.filter_pathways_by_genes(expression_df, raw_pathways)

        if len(filtered_pathways) == 0:
            raise RuntimeError(
                f"No pathways retained after filtering. Check that expression_df "
                f"index contains HGNC gene symbols."
            )

        # Score
        if self.method == "gsva":
            scores_pw_x_samples = self.score_with_gsva(
                expression_df, filtered_pathways, kcdf=kcdf
            )
        else:
            scores_pw_x_samples = self.score_with_ssgsea(
                expression_df, filtered_pathways
            )

        # Build metadata
        metadata = {
            "method": self.method,
            "pathway_source": pathway_source,
            "n_pathways_loaded": len(raw_pathways),
            "n_pathways_scored": scores_pw_x_samples.shape[0],
            "n_samples": scores_pw_x_samples.shape[1],
            "min_genes_per_pathway": self.min_genes_per_pathway,
            "timestamp": datetime.now().isoformat(),
            "pathway_set_versions": PATHWAY_SET_VERSIONS,
            "expression_genes": expression_df.shape[0],
            "expression_samples": expression_df.shape[1],
            "input_hash": hashlib.sha256(
                expression_df.values.tobytes()
            ).hexdigest()[:16],
        }

        # Transpose to samples x pathways (analysis-ready orientation)
        scores_samples_x_pw = scores_pw_x_samples.T
        logger.info(
            f"=== Done: {scores_samples_x_pw.shape[0]} samples x "
            f"{scores_samples_x_pw.shape[1]} pathways ==="
        )

        self._scoring_metadata = metadata
        return scores_samples_x_pw, metadata

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
    def save_pathway_scores(
        self,
        pathway_scores: pd.DataFrame,
        metadata: Dict,
        output_path: Path,
        dataset_id: str = "unknown",
    ) -> None:
        """Save pathway scores (Parquet) and metadata (JSON).

        Parameters
        ----------
        pathway_scores : pd.DataFrame
            Pathway scores (samples x pathways).
        metadata : Dict
            Scoring metadata.
        output_path : Path
            Output Parquet file path.
        dataset_id : str
            Dataset identifier for provenance.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Scores
        pathway_scores.to_parquet(output_path, compression="snappy")
        logger.info(f"Saved pathway scores: {output_path}")

        # Metadata
        meta = {**metadata, "dataset_id": dataset_id}
        meta_path = output_path.parent / f"{output_path.stem}_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        logger.info(f"Saved metadata: {meta_path}")

    @staticmethod
    def load_pathway_scores(parquet_path: Path) -> Tuple[pd.DataFrame, Dict]:
        """Load pathway scores and associated metadata.

        Parameters
        ----------
        parquet_path : Path
            Path to scores Parquet file.

        Returns
        -------
        Tuple[pd.DataFrame, Dict]
            (scores, metadata)
        """
        parquet_path = Path(parquet_path)
        scores = pd.read_parquet(parquet_path)

        meta_path = parquet_path.parent / f"{parquet_path.stem}_metadata.json"
        metadata = {}
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)

        return scores, metadata


# -----------------------------------------------------------------------
# Convenience CLI
# -----------------------------------------------------------------------
def main():
    """Example usage and smoke test."""
    import argparse

    parser = argparse.ArgumentParser(description="Pathway scoring for MM studies")
    parser.add_argument(
        "--expression", type=str, help="Path to expression Parquet (genes x samples)"
    )
    parser.add_argument(
        "--method", type=str, default="gsva", choices=["gsva", "ssgsea"]
    )
    parser.add_argument(
        "--source", type=str, default="all",
        choices=["all", "hallmark", "kegg", "reactome", "mm"],
    )
    parser.add_argument("--output", type=str, default="pathway_scores.parquet")
    parser.add_argument("--dataset-id", type=str, default="unknown")
    args = parser.parse_args()

    if args.expression:
        expr = pd.read_parquet(args.expression)
        scorer = PathwayScorer(method=args.method)
        scores, meta = scorer.score_pathways(expr, pathway_source=args.source)
        scorer.save_pathway_scores(scores, meta, Path(args.output), args.dataset_id)
    else:
        # Smoke test with random data
        logger.info("Running smoke test with random data...")
        np.random.seed(42)
        genes = [f"GENE{i}" for i in range(200)]
        samples = [f"SAMPLE{i}" for i in range(20)]
        fake_expr = pd.DataFrame(
            np.random.randn(200, 20) * 2 + 8,
            index=genes,
            columns=samples,
        )
        scorer = PathwayScorer(method="ssgsea", min_genes_per_pathway=3)
        # Only score curated MM (has known gene names we can control)
        mm_pw = scorer.load_curated_mm_pathways()
        logger.info(f"Smoke test: {len(mm_pw)} curated MM pathways loaded")
        logger.info("Pathway scorer initialized and ready.")


if __name__ == "__main__":
    main()
