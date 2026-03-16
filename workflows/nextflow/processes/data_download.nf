process download_geo_data {
    tag "${dataset.name}"
    publishDir "${params.outdir}/raw", mode: 'copy'

    input:
    val dataset

    output:
    tuple val(dataset), path("${dataset.name}_expression.parquet"), path("${dataset.name}_phenotype.csv")

    script:
    """
    python -c "
from src.preprocessing.download_geo import GEODownloader
dl = GEODownloader(output_dir='.')
dl.download_dataset('${dataset.name}')
"
    """
}
