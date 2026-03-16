process download_geo_data {
    tag "${dataset.name}"
    publishDir "${params.outdir}/raw", mode: 'copy'

    container 'python:3.11-slim'

    cpus 2
    memory '8 GB'
    time '2h'

    input:
    val dataset

    output:
    tuple val(dataset), path("${dataset.name}_expression.parquet"), path("${dataset.name}_phenotype.csv")

    script:
    """
    python "${baseDir}/scripts/download_geo_data.py" \\
        --dataset "${dataset.name}" \\
        --output_dir "." \\
        --output_expr "${dataset.name}_expression.parquet" \\
        --output_pheno "${dataset.name}_phenotype.csv"
    """
}
