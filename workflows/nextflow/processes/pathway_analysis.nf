process compute_pathway_scores {
    tag "${dataset.name}"
    publishDir "${params.outdir}/pathways", mode: 'copy'

    container 'python:3.11-slim'

    cpus 8
    memory '32 GB'
    time '4h'

    input:
    tuple val(dataset), path(normalized)

    output:
    tuple val(dataset), path("${dataset.name}_pathways.parquet")

    script:
    """
    python "${baseDir}/scripts/run_preprocessing.py" \\
        --input "${normalized}" \\
        --dataset_name "${dataset.name}" \\
        --output "${dataset.name}_pathways.parquet" \\
        --pathway_method "ssgsea" \\
        --pathway_databases "Hallmark,KEGG,Reactome,curated_MM" \\
        --config_path "${baseDir}/config/pipeline_config.yaml"
    """
}
