process compute_pathway_scores {
    tag "${dataset.name}"
    publishDir "${params.outdir}/pathways", mode: 'copy'

    input:
    tuple val(dataset), path(normalized)

    output:
    tuple val(dataset), path("${dataset.name}_pathways.parquet")

    script:
    """
    python -c "
import pandas as pd
from src.preprocessing.pathway_scoring import PathwayScorer
expr = pd.read_parquet('${normalized}')
scorer = PathwayScorer(method='ssgsea')
scores, meta = scorer.score_pathways(expr, pathway_source='all')
scores.to_parquet('${dataset.name}_pathways.parquet')
"
    """
}
