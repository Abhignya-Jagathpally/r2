process preprocess_arrays {
    tag "${dataset.name}"
    publishDir "${params.outdir}/standardized", mode: 'copy'

    input:
    tuple val(dataset), path(expression), path(phenotype)

    output:
    tuple val(dataset), path("${dataset.name}_normalized.parquet")

    script:
    """
    python -c "
import pandas as pd
from src.preprocessing.normalization import ExpressionNormalizer, NormalizationContract
contract = NormalizationContract()
normalizer = ExpressionNormalizer(contract=contract)
expr = pd.read_parquet('${expression}')
norm, _ = normalizer.normalize_pipeline(expr, platform_type='array')
norm.to_parquet('${dataset.name}_normalized.parquet')
"
    """
}

process preprocess_rnaseq {
    tag "${dataset.name}"
    publishDir "${params.outdir}/standardized", mode: 'copy'

    input:
    tuple val(dataset), path(expression), path(phenotype)

    output:
    tuple val(dataset), path("${dataset.name}_normalized.parquet")

    script:
    """
    python -c "
import pandas as pd
from src.preprocessing.normalization import ExpressionNormalizer, NormalizationContract
contract = NormalizationContract()
normalizer = ExpressionNormalizer(contract=contract)
expr = pd.read_parquet('${expression}')
norm, _ = normalizer.normalize_pipeline(expr, platform_type='rnaseq')
norm.to_parquet('${dataset.name}_normalized.parquet')
"
    """
}
