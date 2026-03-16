# MM Risk Signature Preprocessing Pipeline

## Overview

This preprocessing pipeline converts bulk transcriptomics data from multiple studies into a unified pathway-space representation for cross-study MM risk signature development.

**Critical Design Decision**: Each study is converted to pathway space **independently**. This avoids problematic raw gene-level merges across heterogeneous array and RNA-seq platforms.

### Datasets

1. **GSE19784** (320 NDMM samples, Affymetrix U133Plus2)
2. **GSE39754** (Healthy + MM, Agilent exon arrays)
3. **GSE2658** (Pre-treatment MM, Affymetrix U133Plus2)
4. **MMRF_CoMMpass_IA21** (RNA-seq, manual download from MMRF Portal)

## Pipeline Architecture

```
Raw Data (GEO/MMRF)
        ↓
[1. Download] → Raw .parquet files + phenotype CSVs
        ↓
[2. Probe Mapping] → Gene symbols (arrays only)
        ↓
[3. Normalization] → Platform-specific, frozen contracts
        ↓
[4. Quality Control] → PCA outliers, HTML reports
        ↓
[5. Pathway Scoring] → GSVA/ssGSEA per-study (CRITICAL STEP)
        ↓
[6. Harmonization] → Cross-study alignment, common pathways
        ↓
Analysis-Ready: Samples × Pathways (Parquet)
```

## Step 1: Download GEO Datasets

Downloads all GEO datasets using GEOparse. Extracts phenotype data and standardizes clinical metadata (OS, PFS, ISS, cytogenetics).

```python
from src.preprocessing.download_geo import GEODownloader

downloader = GEODownloader(output_dir="./data/raw")
results = downloader.download_all()
```

**Outputs**:
- `data/raw/GSE19784/GSE19784_expression.parquet` (genes × samples)
- `data/raw/GSE19784/GSE19784_phenotype.csv` (sample metadata)
- `data/raw/GSE19784/GSE19784_metadata.json` (study info)

## Step 2: Probe Mapping (Arrays Only)

Maps microarray probe IDs to HGNC gene symbols using mygene.info.

- **Multi-mapping strategy**: Max mean (retains highest-expression probe per gene)
- **Platforms**: U133Plus2, Agilent HE v2, Illumina HumanWG6
- **Filtering**: Removes probes mapping to multiple genes or no genes

```python
from src.preprocessing.probe_mapping import ProbeMapper

mapper = ProbeMapper(species="human")
gene_expr, stats = mapper.map_affymetrix_probes(
    expression_df,
    platform="GPL570"  # U133Plus2
)
mapper.save_mapping(gene_expr, output_path, platform="GPL570")
```

**Outputs**:
- `data/standardized/GSE19784_genes.parquet` (gene symbols × samples)
- `data/standardized/GSE19784_mapping_stats.txt`

**Statistics Example**:
```
total_probes: 54675
mapped_probes: 52341
unique_genes: 18612
unmapped_probes: 2334
mapping_rate: 95.7%
```

## Step 3: Normalization (Per-Platform)

Within-platform normalization using **frozen preprocessing contracts** (Karpathy autoresearch pattern).

### Array Normalization

- **Method**: Quantile normalization
- **Steps**:
  1. Rank genes within each sample
  2. Compute target distribution (mean ranks)
  3. Replace with target quantiles
- **QC**: Density plots, boxplots, mean-SD plots

```python
from src.preprocessing.normalization import ExpressionNormalizer, NormalizationContract

contract = NormalizationContract()
normalizer = ExpressionNormalizer(contract=contract)

norm_expr, pipeline_stats = normalizer.normalize_pipeline(
    expression_df,
    platform_type="array",
    low_expr_filter_percentile=25,
    output_dir="./data/analysis_ready/qc_plots"
)

contract.freeze()  # Prevent modifications
contract.save("./data/standardized/normalization_contract.pkl")
```

### RNA-seq Normalization

- **Method**: TMM (edgeR) + voom transformation
- **Steps**:
  1. TMM normalization factors
  2. Log-CPM calculation
  3. voom variance stabilization
  4. Low-expression filtering (bottom 25% by mean)
  5. Log2 transformation

### Contract System

Frozen contracts encode preprocessing parameters for reproducibility:

```
contract_id: 20260315_143022
is_frozen: true
created_at: 2026-03-15T14:30:22

schema:
  columns: [18612 genes]
  index_name: sample_id

low_expr_filter_percentile: 25
low_expr_filter_threshold: 4.23

normalization_method: quantile
log_transform: true
pseudocount: 0.0
```

**Outputs**:
- `data/standardized/GSE19784_normalized.parquet` (log2 expression)
- `data/standardized/GSE19784_normalization_contract.pkl` (frozen contract)
- `data/analysis_ready/qc_plots/GSE19784/` (QC plots)

## Step 4: Quality Control

Sample-level QC: PCA outliers (Mahalanobis distance), missing data, batch effects.

```python
from src.preprocessing.quality_control import QualityController

qc = QualityController(output_dir="./data/analysis_ready/qc_reports")

pca_scores, outlier_flags, stats = qc.detect_pca_outliers(
    expression_df,
    n_pcs=10,
    threshold_sd=3.0
)

missing_stats = qc.analyze_missing_data(expression_df)
batch_stats = qc.analyze_batch_effects(expression_df, "dataset_id", metadata)

report_path = qc.generate_qc_report(
    expression_df,
    metadata=metadata,
    dataset_id="GSE19784",
    batch_column="dataset_id"
)
```

**Outputs**:
- `data/analysis_ready/qc_reports/qc_report_GSE19784.html` (interactive report)
- `data/analysis_ready/qc_reports/qc_summary_GSE19784.json` (stats)

**QC Thresholds**:
- Outlier rate > 10%: Warning
- Missing data > 5%: Warning
- Batch effect F-ratio > 10: Investigate

## Step 5: Pathway Scoring (CRITICAL)

**This is the key design decision**: Score pathways independently per study, then harmonize.

```python
from src.preprocessing.pathway_scoring import PathwayScorer

scorer = PathwayScorer(method="ssgsea")  # or "gsva"

pathway_scores, metadata = scorer.score_pathways(
    log2_normalized_expression,
    pathway_source="all",  # hallmark, kegg, reactome, mm
)

scorer.save_pathway_scores(
    pathway_scores,
    metadata,
    output_path="data/analysis_ready/GSE19784_pathways.parquet",
    dataset_id="GSE19784"
)
```

### Pathway Sources

1. **Hallmark** (50 gene sets)
   - MSigDB v2023.2
   - Broad phenotypes: proliferation, immune, stress, angiogenesis, etc.

2. **KEGG** (186 pathways)
   - Metabolic, signaling, disease pathways
   - Curated from KEGG database

3. **Reactome** (~400 pathways)
   - Filtered to pathways with 10-500 genes
   - Detailed biological processes

4. **Curated MM** (6 signatures)
   - **Proliferation**: MKI67, TOP2A, PTTG1, CCNB1, CCNA2, ...
   - **NFkB Signaling**: NFKB1, NFKB2, RELA, RELB, CHUK, ...
   - **MYC Targets**: MYC, CAD, NOLC1, NPM1, RAN, ...
   - **Bone Disease**: RANKL, RANK, OPG, TRAF6, NFATc1, ...
   - **Immune Response**: CD8A, CD8B, GZMA, GZMB, PRF1, ...
   - **Drug Resistance**: TP53, ABCB1, MDR1, GST, BRCA1, ...

### Scoring Methods

- **ssGSEA** (Single Sample GSEA): Rank-based, fast, recommended for >50 pathways
- **GSVA** (Gene Set Variation Analysis): Kernel-based, more robust, computationally intensive

**Outputs**:
- `data/analysis_ready/GSE19784_pathways.parquet` (samples × pathways)
- `data/analysis_ready/GSE19784_pathways_metadata.json` (pathway info + versions)
- `data/analysis_ready/GSE19784_pathways_summary.txt`

**Example Output**:
```
Dataset: GSE19784
Scoring method: ssgsea
Pathway source: all
Pathways: 642
Samples: 320
Timestamp: 2026-03-15T14:35:00
```

## Step 6: Harmonization

Cross-study alignment of pathway scores.

```python
from src.preprocessing.harmonization import PathwayHarmonizer

harmonizer = PathwayHarmonizer()

# Load pathway scores from all studies
study_pathways = harmonizer.load_study_pathways({
    "GSE19784": Path("data/analysis_ready/GSE19784_pathways.parquet"),
    "GSE39754": Path("data/analysis_ready/GSE39754_pathways.parquet"),
    "GSE2658": Path("data/analysis_ready/GSE2658_pathways.parquet"),
})

# Find common pathways
common_pathways, study_specific = harmonizer.identify_common_pathways(study_pathways)

# Standardize scales (z-score normalization per pathway)
standardized = harmonizer.standardize_pathway_scales(study_pathways, method="zscore")

# Analyze batch effects
study_effects = harmonizer.analyze_study_effects(standardized)

# Visualize
plots = harmonizer.visualize_study_distributions(
    standardized,
    output_dir="data/analysis_ready/harmonization_plots"
)

# Create unified matrix
harmonized_pathways, harmonized_metadata = harmonizer.create_harmonized_matrix(standardized)

harmonizer.save_harmonized_data(
    harmonized_pathways,
    harmonized_metadata,
    output_dir="data/analysis_ready/harmonized"
)
```

**Outputs**:
- `data/analysis_ready/harmonized/harmonized_pathways.parquet` (samples × common pathways)
- `data/analysis_ready/harmonized/harmonized_metadata.csv` (sample + study + clinical info)
- `data/analysis_ready/harmonization_plots/pathway_distributions_*.png`

**Harmonization Statistics**:
```
Common pathways: 612 / 642 (95.3%)
Study-specific (GSE19784): 30
Study-specific (GSE39754): 25
Study-specific (GSE2658): 28

Study effects F-ratio: 2.3 (moderate batch effect)
```

## Running the Full Pipeline

### CLI Usage

```bash
# Full pipeline (all steps)
python scripts/run_preprocessing.py

# With custom paths
python scripts/run_preprocessing.py \
  --data-dir ./data \
  --output-dir ./data/analysis_ready \
  --config ./config/preprocessing_config.json

# Selective steps
python scripts/run_preprocessing.py \
  --steps normalization pathway_scoring harmonization
```

### Python API

```python
from scripts.run_preprocessing import PreprocessingPipeline

pipeline = PreprocessingPipeline(
    data_dir="./data",
    output_dir="./data/analysis_ready",
    config_file="./config/preprocessing_config.json"
)

# Run full pipeline
pipeline.run_full_pipeline()

# Or specific steps
pipeline.run_selective(["normalization", "pathway_scoring"])
```

## Testing

```bash
# Run all tests
python -m pytest tests/test_preprocessing.py -v

# Specific test class
python -m pytest tests/test_preprocessing.py::TestPathwayScoring -v

# With coverage
python -m pytest tests/test_preprocessing.py --cov=src.preprocessing --cov-report=html
```

**Test Coverage**:
- Probe mapping: Input/output shapes, statistics
- Normalization: Quantile normalization, filtering, log transform, contract freezing
- Pathway scoring: Pathway loading, filtering, scoring output shapes
- Quality control: PCA outliers, missing data analysis
- Data contracts: Schema validation, dtype validation, contract serialization
- Harmonization: Common pathway identification, scale standardization

## Configuration

Edit `config/preprocessing_config.json` to customize:

```json
{
  "preprocessing": {
    "normalization": {
      "low_expr_filter_percentile": 25,
      "use_voom": true
    },
    "pathway_scoring": {
      "method": "ssgsea",
      "sources": ["hallmark", "kegg", "reactome", "mm"],
      "min_genes_per_pathway": 3
    },
    "quality_control": {
      "pca_outlier_threshold_sd": 3.0
    }
  }
}
```

## Output Directory Structure

```
data/
├── raw/
│   ├── GSE19784/
│   │   ├── GSE19784_expression.parquet
│   │   ├── GSE19784_phenotype.csv
│   │   └── GSE19784_metadata.json
│   ├── GSE39754/
│   └── GSE2658/
├── standardized/
│   ├── GSE19784_genes.parquet (mapped probes)
│   ├── GSE19784_normalized.parquet (log2, normalized)
│   ├── GSE19784_normalization_contract.pkl (frozen)
│   └── GSE19784_mapping_stats.txt
└── analysis_ready/
    ├── qc_plots/
    │   ├── GSE19784/ (density, boxplot, mean-SD)
    │   └── ...
    ├── qc_reports/
    │   ├── qc_report_GSE19784.html
    │   └── qc_summary_GSE19784.json
    ├── GSE19784_pathways.parquet (samples × pathways)
    ├── GSE19784_pathways_metadata.json
    ├── GSE39754_pathways.parquet
    ├── GSE2658_pathways.parquet
    ├── harmonization_plots/
    │   ├── pathway_distributions_boxplot.png
    │   └── pathway_distributions_violin.png
    └── harmonized/
        ├── harmonized_pathways.parquet (FINAL)
        ├── harmonized_metadata.csv (FINAL)
        └── harmonization_summary.txt
```

## Dependencies

Core:
- pandas, numpy, pyarrow (data manipulation)
- GEOparse (GEO downloads)
- mygene (probe-to-gene mapping)
- gseapy (pathway analysis)
- rpy2 (R integration for GSVA, edgeR/limma)
- scipy, scikit-learn (statistics, PCA)
- matplotlib, seaborn (visualization)

R packages (required for voom, GSVA):
- edgeR
- limma
- GSVA

Install:
```bash
pip install pandas numpy pyarrow GEOparse mygene gseapy rpy2 scipy scikit-learn matplotlib seaborn

# R packages (in R):
Rscript -e "BiocManager::install(c('edgeR', 'limma', 'GSVA'))"
```

## Reproducibility

This pipeline implements Karpathy's autoresearch pattern:

1. **Frozen Contracts**: Preprocessing parameters are locked before model training
2. **Code Hashing**: Detect if preprocessing code has been modified
3. **Checkpoint System**: Resume from any step
4. **QC Reports**: HTML reports for each study

```python
# Verify preprocessing hasn't changed
from src.preprocessing.data_contract import PreprocessingCodeHasher

is_valid, modified = PreprocessingCodeHasher.verify_module_integrity(
    Path("src/preprocessing"),
    expected_hashes
)
```

## Key Design Decisions

1. **Pathway-Space Representation**: Avoids gene-level batch effects from mixing arrays + RNA-seq
2. **Per-Study Scoring**: Each study scored independently before harmonization
3. **Frozen Contracts**: Preprocessing parameters immutable after definition
4. **Common Pathways Only**: Cross-study analysis uses ~600 common pathways
5. **Curated MM Signatures**: Domain-specific pathways for MM biology

---

**Authors**: PhD Researcher 2
**Last Updated**: 2026-03-15
**Version**: 1.0
