process train_baselines {
    tag "baselines"
    publishDir "${params.outdir}/models", mode: 'copy'

    container 'biocontainers/bioconda:latest'

    cpus 8
    memory '32 GB'
    time '6h'

    input:
    path pathway_inputs

    output:
    tuple path("baseline_models.pkl"), path("baseline_metrics.json")

    script:
    """
    python "${baseDir}/scripts/train_baselines.py" \\
        --input_dir "." \\
        --output_models "baseline_models.pkl" \\
        --output_metrics "baseline_metrics.json" \\
        --model_types "CoxPH,RandomSurvival,XGBoost,CatBoost,RandomForest" \\
        --cross_validation_folds 5 \\
        --inner_folds 3 \\
        --config_path "${baseDir}/config/pipeline_config.yaml" \\
        --mlflow_uri "${params.mlflow_uri}" \\
        --experiment_name "${params.experiment_name}"
    """
}

process train_modern_models {
    tag "modern"
    publishDir "${params.outdir}/models", mode: 'copy'

    container 'biocontainers/bioconda:latest'

    cpus 8
    memory '32 GB'
    time '8h'
    gpus 1

    input:
    path pathway_inputs

    output:
    tuple path("modern_models.pkl"), path("modern_metrics.json")

    script:
    """
    python "${baseDir}/scripts/train_modern.py" \\
        --input_dir "." \\
        --output_models "modern_models.pkl" \\
        --output_metrics "modern_metrics.json" \\
        --model_types "PathwayAutoencoder,DomainAdversarial,TabPFN,DeepSurv" \\
        --cross_validation_folds 5 \\
        --inner_folds 3 \\
        --config_path "${baseDir}/config/pipeline_config.yaml" \\
        --mlflow_uri "${params.mlflow_uri}" \\
        --experiment_name "${params.experiment_name}"
    """
}

process train_fusion_models {
    tag "fusion"
    publishDir "${params.outdir}/models", mode: 'copy'

    container 'biocontainers/bioconda:latest'

    cpus 8
    memory '32 GB'
    time '8h'
    gpus 1

    input:
    path baseline_models
    path modern_models
    path pathway_inputs

    output:
    tuple path("fusion_models.pkl"), path("fusion_metrics.json")

    script:
    """
    python "${baseDir}/scripts/train_fusion.py" \\
        --baseline_models "${baseline_models}" \\
        --modern_models "${modern_models}" \\
        --input_dir "." \\
        --output_models "fusion_models.pkl" \\
        --output_metrics "fusion_metrics.json" \\
        --fusion_types "LateFusion,Stacking,AttentionFusion,MultimodalAttention" \\
        --cross_validation_folds 5 \\
        --config_path "${baseDir}/config/pipeline_config.yaml" \\
        --mlflow_uri "${params.mlflow_uri}" \\
        --experiment_name "${params.experiment_name}"
    """
}
