process generate_report {
    tag "report"
    publishDir "${params.outdir}/reports", mode: 'copy'

    container 'biocontainers/bioconda:latest'

    cpus 4
    memory '16 GB'
    time '2h'

    input:
    path cross_study_evaluation
    path risk_scores
    path evaluation_metrics

    output:
    tuple path("pipeline_report.html"), \\
          path("pipeline_report.pdf"), \\
          path("pipeline_report.md")

    script:
    """
    python "${baseDir}/scripts/generate_report.py" \\
        --input_results "${cross_study_evaluation}" \\
        --input_risk_scores "${risk_scores}" \\
        --input_metrics "${evaluation_metrics}" \\
        --output_html "pipeline_report.html" \\
        --output_pdf "pipeline_report.pdf" \\
        --output_markdown "pipeline_report.md" \\
        --pipeline_version "0.1.0" \\
        --config_path "${baseDir}/config/pipeline_config.yaml"
    """
}
