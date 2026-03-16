process preprocess_arrays {
    tag "${dataset.name}"
    publishDir "${params.outdir}/standardized", mode: 'copy'

    container 'python:3.11-slim'

    cpus 4
    memory '16 GB'
    time '3h'

    input:
    tuple val(dataset), path(expression), path(phenotype)

    output:
    tuple val(dataset), path("${dataset.name}_normalized.parquet")

    script:
    """
    python "${baseDir}/scripts/preprocess_all.py" \\
        --input_expr "${expression}" \\
        --input_pheno "${phenotype}" \\
        --platform "array" \\
        --dataset_name "${dataset.name}" \\
        --output "${dataset.name}_normalized.parquet" \\
        --config_path "${baseDir}/config/pipeline_config.yaml"
    """
}

process preprocess_rnaseq {
    tag "${dataset.name}"
    publishDir "${params.outdir}/standardized", mode: 'copy'

    container 'python:3.11-slim'

    cpus 4
    memory '16 GB'
    time '3h'

    input:
    tuple val(dataset), path(expression), path(phenotype)

    output:
    tuple val(dataset), path("${dataset.name}_normalized.parquet")

    script:
    """
    python "${baseDir}/scripts/preprocess_all.py" \\
        --input_expr "${expression}" \\
        --input_pheno "${phenotype}" \\
        --platform "rnaseq" \\
        --dataset_name "${dataset.name}" \\
        --output "${dataset.name}_normalized.parquet" \\
        --config_path "${baseDir}/config/pipeline_config.yaml"
    """
}
