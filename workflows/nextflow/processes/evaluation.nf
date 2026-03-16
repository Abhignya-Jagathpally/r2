process evaluate_cross_study {
    tag "evaluation"
    publishDir "${params.outdir}/evaluation", mode: 'copy'

    container 'biocontainers/bioconda:latest'

    cpus 8
    memory '32 GB'
    time '6h'

    input:
    path baseline_models
    path modern_models
    path fusion_models
    path pathway_inputs

    output:
    tuple path("cross_study_evaluation.json"), \\
          path("predictions.parquet"), \\
          path("risk_scores.csv"), \\
          path("evaluation_metrics.json")

    script:
    """
    python "${baseDir}/scripts/evaluate_cross_study.py" \\
        --baseline_models "${baseline_models}" \\
        --modern_models "${modern_models}" \\
        --fusion_models "${fusion_models}" \\
        --input_dir "." \\
        --output_results "cross_study_evaluation.json" \\
        --output_predictions "predictions.parquet" \\
        --output_risk_scores "risk_scores.csv" \\
        --output_metrics "evaluation_metrics.json" \\
        --metrics "c_index,time_auc,brier_score,ici" \\
        --config_path "${baseDir}/config/pipeline_config.yaml" \\
        --mlflow_uri "${params.mlflow_uri}" \\
        --experiment_name "${params.experiment_name}"
    """
}
