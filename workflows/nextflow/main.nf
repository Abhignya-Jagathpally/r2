#!/usr/bin/env nextflow
/**
 * MM Transcriptomics Risk Signature Pipeline
 * Nextflow DSL2 implementation
 * Author: PhD Researcher 1
 * Description: Bulk transcriptomics cross-study analysis for MM risk stratification
 */

nextflow.enable.dsl = 2

// Import modules
include { download_geo_data } from './processes/data_download.nf'
include { preprocess_arrays } from './processes/preprocessing.nf'
include { preprocess_rnaseq } from './processes/preprocessing.nf'
include { compute_pathway_scores } from './processes/pathway_analysis.nf'
include { train_baselines } from './processes/modeling.nf'
include { train_modern_models } from './processes/modeling.nf'
include { train_fusion_models } from './processes/modeling.nf'
include { evaluate_cross_study } from './processes/evaluation.nf'
include { generate_report } from './processes/reporting.nf'

// ============================================================================
// WORKFLOW DEFINITION
// ============================================================================

workflow {
    log.info """\
        ╔════════════════════════════════════════════════════════════════════╗
        ║     MM Transcriptomics Risk Signature Pipeline (Nextflow)          ║
        ║     Version: 0.1.0                                                 ║
        ║     DSL2 Implementation                                             ║
        ╚════════════════════════════════════════════════════════════════════╝
        """

    // ========================================================================
    // Step 1: Data Download & Preparation
    // ========================================================================
    log.info "Step 1: Downloading and preparing data..."

    // Define dataset channels
    datasets_ch = Channel.of(
        [name: "GSE2658", type: "microarray"],
        [name: "GSE19784", type: "microarray"],
        [name: "GSE39754", type: "microarray"],
        [name: "CoMMpass", type: "rnaseq"]
    )

    // Download GEO data (skip CoMMpass - requires credentials)
    geo_data = datasets_ch
        .filter { it.name != "CoMMpass" }
        .set { geo_datasets }

    download_ch = download_geo_data(geo_datasets)

    // ========================================================================
    // Step 2: Preprocessing
    // ========================================================================
    log.info "Step 2: Preprocessing transcriptomics data..."

    // Separate microarray and RNA-seq for appropriate preprocessing
    download_ch
        .branch {
            array: it.type == "microarray"
            rnaseq: it.type == "rnaseq"
        }
        .set { data_by_type }

    // Process microarrays
    preprocessed_arrays = preprocess_arrays(data_by_type.array)

    // Process RNA-seq
    preprocessed_rnaseq = preprocess_rnaseq(data_by_type.rnaseq)

    // Combine all preprocessed data
    all_preprocessed = preprocessed_arrays.concat(preprocessed_rnaseq)

    // ========================================================================
    // Step 3: Pathway Analysis
    // ========================================================================
    log.info "Step 3: Computing pathway scores..."

    pathway_scores = compute_pathway_scores(all_preprocessed)

    // Collect all pathway outputs into single file for model training
    pathway_inputs = pathway_scores
        .map { it[1] }
        .collect()

    // ========================================================================
    // Step 4: Model Training
    // ========================================================================
    log.info "Step 4: Training survival models..."

    // Baseline models (processes all pathway scores together)
    baselines = train_baselines(pathway_inputs)

    // Modern gradient boosting models
    modern = train_modern_models(pathway_inputs)

    // Fusion/deep learning models (depends on baseline and modern)
    fusion = train_fusion_models(baselines.out[0], modern.out[0], pathway_inputs)

    // ========================================================================
    // Step 5: Cross-Study Evaluation
    // ========================================================================
    log.info "Step 5: Evaluating across studies..."

    evaluation_results = evaluate_cross_study(
        baselines.out[0],
        modern.out[0],
        fusion.out[0],
        pathway_inputs
    )

    // ========================================================================
    // Step 6: Report Generation
    // ========================================================================
    log.info "Step 6: Generating comprehensive report..."

    evaluation_results
        .map { [it[0], it[2], it[3]] }
        .set { report_inputs }

    final_report = generate_report(report_inputs)

    log.info "Pipeline completed successfully!"
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

def log_section(String title) {
    log.info "\n" + "="*80
    log.info title
    log.info "="*80 + "\n"
}

def check_required_files(String path) {
    if (!file(path).exists()) {
        error("Required file not found: ${path}")
    }
}
