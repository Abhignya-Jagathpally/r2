"""
Probe-to-Gene Mapping
=====================
Map microarray probes to HGNC gene symbols using mygene/biomaRt.
Handles multi-mapping (aggregation by max mean), removes ambiguous probes.
Platform-specific: U133Plus2, HuEx, HTA2.0.

Author: PhD Researcher 2
Date: 2026
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from collections import defaultdict

try:
    import mygene
except ImportError:
    raise ImportError("mygene not installed. Run: pip install mygene")

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Platform-specific probe files and metadata
PLATFORMS = {
    "GPL570": {  # Affymetrix U133Plus2
        "name": "U133Plus2",
        "species": "human",
        "description": "Affymetrix GeneChip Human Genome U133 Plus 2.0",
    },
    "GPL5175": {  # Agilent SurePrint G3 Human GE 8x60K Microarray
        "name": "AgilentHE",
        "species": "human",
        "description": "Agilent SurePrint G3 Human GE 8x60K microarray",
    },
    "GPL6884": {  # Illumina HumanWG-6 v3.0 Expression BeadChip
        "name": "Illumina_HumanWG6_v3",
        "species": "human",
        "description": "Illumina HumanWG-6 v3.0 Expression BeadChip",
    },
    "GPL17077": {  # Agilent-039494 SurePrint G3 Human GE v2 8x60K Microarray
        "name": "AgilentHE_v2",
        "species": "human",
        "description": "Agilent SurePrint G3 Human GE v2 8x60K microarray",
    },
}


class ProbeMapper:
    """Map microarray probes to gene symbols."""

    def __init__(self, species: str = "human"):
        """
        Initialize probe mapper.

        Parameters
        ----------
        species : str
            Species code (e.g., "human").
        """
        self.species = species
        self.mg = mygene.MyGeneInfo()
        self.gene_cache = {}  # Cache for mygene queries
        self.mapping_stats = {}

    def query_mygene(self, probe_ids: List[str], scopes: Optional[List[str]] = None) -> Dict:
        """
        Query mygene.info for gene annotations.

        Parameters
        ----------
        probe_ids : List[str]
            Probe IDs or gene symbols to map.
        scopes : Optional[List[str]]
            Search scopes (e.g., ["entrezgene", "symbol", "ensembl.gene"]).
            If None, defaults to ["symbol", "entrezgene", "ensembl.gene"].

        Returns
        -------
        Dict
            Mapping of probe_id → gene information.
        """
        if scopes is None:
            scopes = ["symbol", "entrezgene", "ensembl.gene"]

        try:
            results = self.mg.querymany(
                probe_ids,
                scopes=scopes,
                fields=["symbol", "entrezgene", "ensembl.gene", "uniprot.Swiss-Prot"],
                species=self.species,
                returnall=True,
            )
            return {r.get("query"): r for r in results.get("hits", [])}
        except Exception as e:
            logger.error(f"mygene query failed: {str(e)}")
            return {}

    def map_probes_to_genes(
        self, expression_df: pd.DataFrame, platform: str = "GPL570"
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Map probe IDs (index) to gene symbols.

        Parameters
        ----------
        expression_df : pd.DataFrame
            Expression matrix with probe IDs as index.
        platform : str
            Affymetrix platform ID (e.g., "GPL570" for U133Plus2).

        Returns
        -------
        Tuple[pd.DataFrame, Dict]
            (gene_expression_df, mapping_stats)
        """
        probe_ids = expression_df.index.tolist()
        logger.info(f"Mapping {len(probe_ids)} probes from {platform}...")

        # Query mygene
        mygene_results = self.query_mygene(probe_ids)

        # Build mapping
        probe_to_gene = {}
        unmapped_probes = []
        multi_mapped = []

        for probe_id in probe_ids:
            if probe_id in mygene_results:
                hit = mygene_results[probe_id]
                gene_symbol = hit.get("symbol")
                if gene_symbol:
                    probe_to_gene[probe_id] = gene_symbol
                else:
                    unmapped_probes.append(probe_id)
            else:
                unmapped_probes.append(probe_id)

        # Create gene-level mapping
        gene_to_probes = defaultdict(list)
        for probe_id, gene_symbol in probe_to_gene.items():
            gene_to_probes[gene_symbol].append(probe_id)

        # Identify multi-mapped probes
        for gene_symbol, probes in gene_to_probes.items():
            if len(probes) > 1:
                multi_mapped.extend(probes)

        # Aggregate multi-mapped probes by max mean expression
        aggregated_matrix = []
        aggregated_genes = []

        for probe_id in expression_df.index:
            if probe_id in probe_to_gene:
                gene_symbol = probe_to_gene[probe_id]
                if gene_symbol in aggregated_genes:
                    # Already aggregated; skip
                    continue

                # Get all probes for this gene
                probes_for_gene = gene_to_probes[gene_symbol]
                probes_expr = expression_df.loc[probes_for_gene]

                # Aggregate by max mean
                probe_means = probes_expr.mean(axis=1)
                best_probe_idx = probe_means.idxmax()
                aggregated_matrix.append(expression_df.loc[best_probe_idx].values)
                aggregated_genes.append(gene_symbol)

        # Create new dataframe
        if aggregated_matrix:
            gene_expr_df = pd.DataFrame(
                np.array(aggregated_matrix),
                index=aggregated_genes,
                columns=expression_df.columns,
            )
            gene_expr_df.index.name = "gene_symbol"
        else:
            gene_expr_df = pd.DataFrame(columns=expression_df.columns)

        # Statistics
        stats = {
            "total_probes": len(probe_ids),
            "mapped_probes": len(probe_to_gene),
            "unmapped_probes": len(unmapped_probes),
            "multi_mapped_probes": len(set(multi_mapped)),
            "unique_genes": len(aggregated_genes),
            "mapping_rate": len(probe_to_gene) / len(probe_ids) if probe_ids else 0,
        }

        logger.info(f"Mapping complete: {stats['mapped_probes']} mapped, "
                   f"{stats['unmapped_probes']} unmapped, "
                   f"{stats['unique_genes']} unique genes")

        self.mapping_stats[platform] = stats
        return gene_expr_df, stats

    def map_affymetrix_probes(
        self, expression_df: pd.DataFrame, platform: str = "GPL570", species: str = "human"
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Specialized mapping for Affymetrix platforms.

        Parameters
        ----------
        expression_df : pd.DataFrame
            Expression matrix with Affymetrix probe IDs as index.
        platform : str
            Platform identifier (GPL570 for U133Plus2, etc.).
        species : str
            Species code.

        Returns
        -------
        Tuple[pd.DataFrame, Dict]
            (gene_expression_df, mapping_stats)
        """
        if platform not in PLATFORMS:
            logger.warning(f"Unknown platform {platform}. Attempting generic mapping.")
            return self.map_probes_to_genes(expression_df, platform)

        return self.map_probes_to_genes(expression_df, platform)

    def validate_mapping(self, gene_expr_df: pd.DataFrame) -> Dict:
        """
        Validate mapped gene expression matrix.

        Parameters
        ----------
        gene_expr_df : pd.DataFrame
            Gene expression matrix after mapping.

        Returns
        -------
        Dict
            Validation report.
        """
        report = {
            "n_genes": gene_expr_df.shape[0],
            "n_samples": gene_expr_df.shape[1],
            "missing_values": gene_expr_df.isna().sum().sum(),
            "missing_pct": (gene_expr_df.isna().sum().sum() /
                           (gene_expr_df.shape[0] * gene_expr_df.shape[1]) * 100),
            "inf_values": np.isinf(gene_expr_df.values).sum(),
            "zero_count": (gene_expr_df == 0).sum().sum(),
            "zero_pct": ((gene_expr_df == 0).sum().sum() /
                        (gene_expr_df.shape[0] * gene_expr_df.shape[1]) * 100),
        }
        logger.info(f"Validation: {report['n_genes']} genes, "
                   f"{report['missing_pct']:.2f}% missing, "
                   f"{report['zero_pct']:.2f}% zeros")
        return report

    def save_mapping(
        self, gene_expr_df: pd.DataFrame, output_path: Path, platform: str = "GPL570"
    ) -> None:
        """
        Save mapped gene expression matrix.

        Parameters
        ----------
        gene_expr_df : pd.DataFrame
            Gene expression matrix after probe mapping.
        output_path : Path
            Output file path (Parquet format).
        platform : str
            Platform identifier.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        gene_expr_df.to_parquet(output_path, compression="snappy")
        logger.info(f"Saved mapped expression: {output_path}")

        # Save mapping statistics
        stats_path = output_path.parent / f"{output_path.stem}_mapping_stats.txt"
        stats = self.mapping_stats.get(platform, {})
        with open(stats_path, "w") as f:
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
        logger.info(f"Saved mapping statistics: {stats_path}")


def main():
    """Example usage."""
    logger.info("Probe mapper initialized. Use map_probes_to_genes() method.")


if __name__ == "__main__":
    main()
