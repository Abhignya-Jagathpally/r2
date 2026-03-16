"""
Preprocessing module for MM risk signature pipeline.

Core components:
- download_geo: Download and parse GEO datasets
- probe_mapping: Map microarray probes to gene symbols
- normalization: Within-platform normalization and contract system
- pathway_scoring: GSVA/ssGSEA pathway scoring (per-study)
- quality_control: Sample QC, batch effects, HTML reports
- data_contract: Frozen preprocessing contracts (Karpathy autoresearch pattern)
- harmonization: Cross-study pathway alignment

Design principle: Convert each study to pathway space independently.
Avoid raw gene-level merges across array and RNA-seq.
"""

from . import download_geo
from . import probe_mapping
from . import normalization
from . import pathway_scoring
from . import quality_control
from . import data_contract
from . import harmonization

__all__ = [
    "download_geo",
    "probe_mapping",
    "normalization",
    "pathway_scoring",
    "quality_control",
    "data_contract",
    "harmonization",
]
