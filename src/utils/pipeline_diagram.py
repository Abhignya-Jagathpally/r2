"""
Pipeline architecture diagram for the MM Transcriptomics Risk Signature Pipeline.

Provides both text-based (for terminal/logging) and matplotlib-based
(for publication/reports) diagrams of the full pipeline architecture.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def get_pipeline_diagram() -> str:
    """
    Return the full ASCII pipeline architecture diagram.

    This diagram documents the end-to-end flow from raw GEO data
    through preprocessing, modeling, fusion, optimization, and evaluation.
    """
    return r"""
================================================================================
       MM TRANSCRIPTOMICS RISK SIGNATURE PIPELINE - ARCHITECTURE DIAGRAM
================================================================================

  DATA SOURCES (Real GEO/CoMMpass Only)
  ======================================
  +-------------+  +-------------+  +-------------+  +------------------+
  |  GSE2658    |  |  GSE19784   |  |  GSE39754   |  |  CoMMpass IA21   |
  | (559 array) |  | (589 array) |  | (559 array) |  | (~1000 RNA-seq)  |
  +------+------+  +------+------+  +------+------+  +--------+---------+
         |                |                |                   |
         +--------+-------+-------+--------+-------------------+
                  |               |
                  v               v
  STAGE 1: PREPROCESSING [Checkpoint: preprocessing]
  ===================================================
  +-----------------------------------------------------------+
  |  1a. Download         GEOparse / wget from NCBI GEO       |
  |  1b. Probe Mapping    HGNC gene symbols, collapse probes   |
  |  1c. Normalization    Quantile (array) / TMM+voom (seq)    |
  |  1d. Quality Control  PCA outlier detection, MAD filter    |
  |  1e. Pathway Scoring  ssGSEA on Hallmark/KEGG/Reactome    |
  |  1f. Harmonization    Z-score cross-study standardization  |
  +----------------------------+------------------------------+
                               |
                               v
  STAGE 2: DATA CONTRACT [Checkpoint: data_contract]
  ===================================================
  +-----------------------------------------------------------+
  |  Frozen Preprocessing Contract (SHA256 verified)           |
  |  - Schema validation   (dtypes, dimensions, ranges)       |
  |  - Code hash           (preprocessing source integrity)   |
  |  - Data hash           (input data fingerprint)           |
  |  - Contract hash  -->  Immutable reference for all models  |
  +----------------------------+------------------------------+
                               |
              +----------------+----------------+
              |                                 |
              v                                 v
  STAGE 3: CLASSICAL BASELINES          STAGE 4: FOUNDATION MODELS
  [Checkpoint: baseline_training]       [Checkpoint: foundation_training]
  ================================      ====================================
  +---------------------------+         +--------------------------------+
  | Sparse Group Lasso Cox    |         |  DeepSurv (Cox PH MLP)         |
  |   L1 + L2 pathway-aware   |         |   [256]->[128]->[64]->1        |
  | Lasso Cox / Elastic Net   |         |   BatchNorm + Dropout          |
  | Random Survival Forest    |         +--------------------------------+
  | XGBoost / CatBoost (Cox)  |         |  Pathway VAE (Unsupervised)    |
  | DE + Enrichment Baseline  |         |   Encoder: d->128->64->z(32)   |
  +-------------+-------------+         |   Decoder: z(32)->64->128->d   |
                |                       |   KL + Recon + Survival loss   |
                |                       +--------------------------------+
                |                       |  DANN (Domain Adversarial)     |
                |                       |   Shared features + GRL        |
                |                       |   Survival head + Domain disc  |
                |                       |   CORAL alignment loss         |
                |                       +---------------+----------------+
                |                                       |
                +-------------------+-------------------+
                                    |
                                    v
  STAGE 5: MULTIMODAL FUSION [Checkpoint: fusion_training]
  ==========================================================
  +-----------------------------------------------------------+
  |                                                           |
  |  Pathway Features ----+                                   |
  |                       |     Late Fusion (3 strategies)    |
  |                       +---> - Weighted (softmax weights)  |
  |                       |     - Stacking (MLP meta-learner) |
  |  Clinical Features ---+     - Attention (learned weights) |
  |                       |                                   |
  |                       +---> Cross-Attention Fusion        |
  |                             - Clinical queries pathways   |
  |                             - 4-head multi-head attention |
  |                             - Fused -> survival pred      |
  +----------------------------+------------------------------+
                               |
                               v
  STAGE 6: HYPERPARAMETER OPTIMIZATION [Checkpoint: hpo]
  ========================================================
  +-----------------------------------------------------------+
  |  Autoresearch Agent (Karpathy Paradigm)                   |
  |                                                           |
  |  FROZEN: Preprocessing contract (verified by hash)        |
  |  EDITABLE: Architecture, LR, dropout, batch_size, ...    |
  |                                                           |
  |  Budget:  20 experiments max / 3600s wall-clock           |
  |  Search:  Ray Tune + ASHA scheduler                       |
  |  Metric:  C-index (maximize)                              |
  |  Logging: MLflow per trial                                |
  +----------------------------+------------------------------+
                               |
                               v
  STAGE 7: CROSS-STUDY EVALUATION [Checkpoint: evaluation]
  ==========================================================
  +-----------------------------------------------------------+
  |  Patient-Level CV     No data leakage (group-aware)       |
  |  LOSO Validation      Leave-one-study-out                 |
  |  Metrics:                                                 |
  |    - C-index          Primary (concordance)               |
  |    - AUC(t)           Time-dependent at 1, 3, 5 years    |
  |    - IBS              Integrated Brier Score              |
  |    - Calibration      Slope, intercept, D-calibration     |
  |    - Subgroup         Per ISS stage / cytogenetics        |
  |    - Bootstrap CI     95% confidence intervals            |
  +----------------------------+------------------------------+
                               |
                               v
  STAGE 8: REPORTING [Checkpoint: reporting]
  ============================================
  +-----------------------------------------------------------+
  |  Publication-Ready Outputs:                               |
  |    - Kaplan-Meier survival curves                         |
  |    - Forest plots with CIs                                |
  |    - Model comparison tables                              |
  |    - Calibration plots                                    |
  |    - Feature importance (SHAP)                            |
  |    - Cross-study benchmark summary                        |
  |    - HTML/PDF report                                      |
  +-----------------------------------------------------------+

  RESEARCH TAKEAWAYS
  ===================
  1. Pathway-level features (ssGSEA) capture biologically meaningful signal
  2. Cross-study harmonization is critical for generalization
  3. Domain adversarial training (DANN) reduces batch effects
  4. Multimodal fusion (pathway + clinical) outperforms unimodal
  5. Frozen preprocessing contracts ensure reproducibility
  6. LOSO validation provides honest generalization estimates
  7. C-index is the primary metric for survival discrimination

================================================================================
"""


def get_model_architecture_diagram() -> str:
    """Return a detailed model architecture diagram for all models."""
    return r"""
================================================================================
                    MODEL ARCHITECTURES - DETAILED VIEW
================================================================================

  1. DeepSurv (Cox PH MLP)
  -------------------------
  Input: pathway_features [batch, d]
    |-> Linear(d, 256) -> BatchNorm -> ReLU -> Dropout(0.1)
    |-> Linear(256, 128) -> BatchNorm -> ReLU -> Dropout(0.1)
    |-> Linear(128, 64) -> BatchNorm -> ReLU -> Dropout(0.1)
    |-> Linear(64, 1) -> log_hazard [batch, 1]
  Loss: Cox Partial Likelihood (handles censoring)
  Metric: C-index

  2. Pathway VAE
  ---------------
  Encoder:
    pathway_scores [batch, d]
      |-> Linear(d, 128) -> ReLU
      |-> Linear(128, 64) -> ReLU
      |-> mu: Linear(64, z_dim)    }  Reparameterization
      |-> logvar: Linear(64, z_dim) }  z = mu + eps * exp(0.5*logvar)
  Decoder:
    z [batch, z_dim]
      |-> Linear(z_dim, 64) -> ReLU
      |-> Linear(64, 128) -> ReLU
      |-> Linear(128, d) -> reconstructed_pathways
  Loss: MSE_recon + beta_kl * KL(q||p) + beta_surv * Cox_loss

  3. DANN (Domain Adversarial Neural Network)
  --------------------------------------------
  Shared Feature Extractor:
    pathways [batch, d]
      |-> Linear(d, 256) -> BatchNorm -> ReLU -> Dropout
      |-> Linear(256, 128) -> BatchNorm -> ReLU -> Dropout
      |-> features [batch, 128]
            |                          |
            v                          v (Gradient Reversal Layer)
      Survival Head              Domain Discriminator
        |-> Linear(128, 64)       |-> Linear(128, 64)
        |-> Linear(64, 1)        |-> Linear(64, n_domains)
        |-> Cox loss              |-> CrossEntropy loss
                                  + CORAL alignment loss

  4. Multimodal Cross-Attention Fusion
  --------------------------------------
  Pathway features [batch, n_pathways, d_path]  ->  Keys, Values
  Clinical features [batch, n_clinical, d_clin] ->  Queries
            |
            v
  Multi-Head Cross-Attention (4 heads)
    Q = W_q * clinical,  K = W_k * pathway,  V = W_v * pathway
    Attention = softmax(Q @ K^T / sqrt(d_k)) @ V
            |
            v
  Fused representation [batch, d_fused]
    |-> Linear(d_fused, 64) -> ReLU -> Dropout
    |-> Linear(64, 1) -> log_hazard
  Loss: Cox Partial Likelihood

  5. Late Fusion (3 Strategies)
  ------------------------------
  Strategy A: Weighted Fusion
    risk_pathway * w1 + risk_clinical * w2  (softmax weights, learnable)

  Strategy B: Stacking Meta-Learner
    [risk_pathway, risk_clinical] -> MLP(2, 16, 1) -> fused_risk

  Strategy C: Attention Fusion
    [risk_pathway, risk_clinical] -> Attention(2, d) -> weighted_risk

================================================================================
"""


def render_pipeline_diagram_matplotlib(
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """
    Render a publication-quality pipeline diagram using matplotlib.

    Parameters
    ----------
    output_path : Path, optional
        Where to save the figure. If None, saves to outputs/figures/pipeline_diagram.png.

    Returns
    -------
    Path or None
        Path to saved figure, or None if matplotlib is unavailable.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import FancyBboxPatch
    except ImportError:
        logger.warning("matplotlib not available; skipping diagram rendering")
        return None

    if output_path is None:
        output_path = Path("outputs/figures/pipeline_diagram.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(14, 20), dpi=150)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 22)
    ax.axis("off")
    ax.set_title(
        "MM Transcriptomics Risk Signature Pipeline",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    stages = [
        (5, 20.5, "Data Sources", "GSE2658 / GSE19784 / GSE39754 / CoMMpass", "#E8F5E9"),
        (5, 18.5, "Stage 1: Preprocessing", "Download -> Probe Map -> Normalize -> QC -> Pathway -> Harmonize", "#E3F2FD"),
        (5, 16.5, "Stage 2: Data Contract", "Frozen SHA256 preprocessing contract", "#FFF3E0"),
        (3, 14.5, "Stage 3: Baselines", "SGL-Cox / Lasso / RSF / XGBoost", "#F3E5F5"),
        (7, 14.5, "Stage 4: Foundation", "DeepSurv / VAE / DANN", "#F3E5F5"),
        (5, 12.5, "Stage 5: Fusion", "Late Fusion (3) + Cross-Attention", "#E8EAF6"),
        (5, 10.5, "Stage 6: HPO", "Autoresearch (Ray Tune, 20 trials)", "#FFF9C4"),
        (5, 8.5, "Stage 7: Evaluation", "LOSO-CV / C-index / AUC(t) / IBS / Bootstrap CI", "#FFEBEE"),
        (5, 6.5, "Stage 8: Reporting", "KM curves / Forest plots / HTML report", "#E0F2F1"),
    ]

    for x, y, title, desc, color in stages:
        width = 8 if x == 5 else 3.5
        x_start = x - width / 2
        box = FancyBboxPatch(
            (x_start, y - 0.6),
            width,
            1.2,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor="#333333",
            linewidth=1.5,
        )
        ax.add_patch(box)
        ax.text(x, y + 0.15, title, ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(x, y - 0.2, desc, ha="center", va="center", fontsize=7, style="italic", color="#555555")

    # Draw arrows between stages
    arrow_props = dict(arrowstyle="->", color="#333333", lw=1.5)
    connections = [
        (5, 19.9, 5, 19.1),
        (5, 17.9, 5, 17.1),
        (5, 15.9, 3, 15.1),
        (5, 15.9, 7, 15.1),
        (3, 13.9, 5, 13.1),
        (7, 13.9, 5, 13.1),
        (5, 11.9, 5, 11.1),
        (5, 9.9, 5, 9.1),
        (5, 7.9, 5, 7.1),
    ]
    for x1, y1, x2, y2 in connections:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=arrow_props)

    # Checkpoint markers
    for i, (x, y, title, desc, color) in enumerate(stages[1:], 1):
        ax.text(
            9.2 if x == 5 else (x + 2.2),
            y,
            f"CP-{i}",
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#F44336", edgecolor="none"),
        )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Pipeline diagram saved to %s", output_path)
    return output_path
