# MM Transcriptomics Risk Signature Pipeline

Multimodal clinical AI pipeline for Multiple Myeloma (MM) risk stratification
from bulk transcriptomics across microarray and RNA-seq platforms.

## Architecture

```
GEO/CoMMpass Data -> Preprocessing -> Data Contract (SHA256)
                                         |
                    +--------------------+--------------------+
                    |                                         |
            Classical Baselines                     Foundation Models
        (SGL-Cox, RSF, XGBoost)              (DeepSurv, VAE, DANN)
                    |                                         |
                    +--------------------+--------------------+
                                         |
                              Multimodal Fusion
                     (Late Fusion + Cross-Attention)
                                         |
                                        HPO
                        (Autoresearch, Ray Tune)
                                         |
                           LOSO Cross-Study Eval
                      (C-index, AUC(t), IBS, CI)
                                         |
                               Reporting
```

## Quick Start

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Show pipeline diagram
python main.py --diagram

# Dry run (validate config)
python main.py --dry-run

# Run full pipeline
python main.py

# Run specific stages
python main.py --stages 3 4 5    # baselines + foundation + fusion
python main.py --stage-from 4    # from foundation onward

# Resume from checkpoint
python main.py --resume <run_id>
```

## Data Sources

| Dataset | Platform | Samples | Source |
|---------|----------|---------|--------|
| GSE2658 | Affymetrix U133+ 2.0 | 559 | GEO |
| GSE19784 | Affymetrix U133+ 2.0 | 589 | GEO |
| GSE39754 | Affymetrix U133+ 2.0 | 559 | GEO |
| CoMMpass IA21 | RNA-seq | ~1000 | MMRF |

## Pipeline Stages

1. **Preprocessing** -- Download, normalize (quantile/TMM), QC, pathway scoring (ssGSEA)
2. **Data Contract** -- Freeze preprocessing with SHA256-verified contract
3. **Classical Baselines** -- Sparse Group Lasso Cox, Elastic Net, RSF, XGBoost
4. **Foundation Models** -- DeepSurv, Pathway VAE, DANN (PyTorch Lightning)
5. **Multimodal Fusion** -- Late fusion (weighted/stacking/attention) + cross-attention
6. **HPO** -- Autoresearch agent with frozen preprocessing (Ray Tune)
7. **Evaluation** -- LOSO-CV, C-index, Uno's C-index, AUC(t), IBS, bootstrap CI
8. **Reporting** -- Publication-ready figures, pipeline diagram, research takeaways

## Key Design Decisions

- **Per-study pathway scoring**: Each study is converted to pathway space independently
  (no raw gene-level merges across microarray and RNA-seq)
- **Frozen preprocessing**: SHA256-verified contract ensures reproducibility
- **Patient-level splits**: No data leakage in cross-validation
- **Primary metric**: Concordance index (C-index) for survival discrimination

## Project Structure

```
pipeline3/
├── main.py                  # End-to-end pipeline orchestrator
├── config/                  # YAML/JSON configuration
├── src/
│   ├── preprocessing/       # Download, normalize, pathway scoring
│   ├── models/
│   │   ├── baselines/       # Classical survival models
│   │   ├── modern/          # DeepSurv, VAE, DANN, autoresearch
│   │   └── fusion/          # Late fusion, cross-attention
│   ├── evaluation/          # Metrics, splits, benchmarking
│   └── utils/               # Config, logging, visualization
├── tests/                   # Test suite
├── notebooks/               # Analysis notebooks
├── workflows/               # Nextflow & Snakemake
└── docker/                  # Containerization
```

## Requirements

- Python 3.11+
- PyTorch 2.0+
- scikit-survival, lifelines
- gseapy (pathway scoring)
- MLflow (experiment tracking)
- Ray Tune (HPO)

## License

Research use only. See institutional agreement.
