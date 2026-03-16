# Modern ML Pipeline: Complete Implementation

## Overview

This document describes the complete production ML pipeline for bulk transcriptomics cross-study MM (Multiple Myeloma) risk-signature prediction.

**Architecture Pattern:** Classical baseline → Foundation model → Multimodal fusion (Karpathy paradigm)

**Implementation Paradigm:** PyTorch Lightning + MLflow + Ray Tune with constrained autoresearch

---

## Modules Implemented

### 1. Training Utilities (`src/models/modern/training_utils.py`)

Core loss functions and training infrastructure.

**Components:**
- **CoxPartialLikelihood**: Cox proportional hazards partial likelihood loss
  - Computes negative log likelihood for right-censored survival data
  - Numerically stable risk set computation

- **RankingLoss**: Pairwise concordance-based ranking loss for survival analysis

- **ConcordanceIndex**: C-index computation (fraction of concordant pairs)
  - Primary metric for survival model evaluation
  - Range [0.5, 1.0] where 0.5 = random, 1.0 = perfect

- **EarlyStopping**: Patience-based early stopping with best weight restoration

- **GradientClipper**: Global gradient norm clipping for training stability

- **Schedulers**: Cosine annealing with optional linear warmup

**Key Features:**
- Full type hints and docstrings
- Numerical stability (eps parameters, log-space computations)
- Compatible with PyTorch and Lightning

---

### 2. DeepSurv (`src/models/modern/deepsurv.py`)

Deep Cox Proportional Hazards model (PyTorch Lightning).

**Architecture:**
```
pathways (d) → Dense(256) + BN + ReLU + Dropout
            → Dense(128) + BN + ReLU + Dropout
            → Dense(64) + BN + ReLU + Dropout
            → Dense(1) → log hazard
```

**Features:**
- Configurable hidden dimensions
- Batch normalization and dropout for regularization
- L1 and L2 regularization options
- Cosine annealing scheduler with warmup
- Validation with C-index monitoring
- Test step with detailed metrics

**Usage:**
```python
model = DeepSurv(
    input_dim=50,  # pathway features
    hidden_dims=[256, 128, 64],
    dropout_rate=0.1,
    learning_rate=1e-3,
)
# Train with PyTorch Lightning Trainer
```

---

### 3. Pathway Autoencoder (`src/models/modern/pathway_autoencoder.py`)

Variational Autoencoder for cross-study invariant pathway representations.

**Architecture:**
```
Encoder: pathways (d) → 128 → 64 → {μ, σ}(32)
Decoder: z(32) → 64 → 128 → pathways (d)
```

**Features:**
- Reparameterization trick for VAE sampling
- Reconstruction loss (MSE) + KL divergence
- Optional survival objective via latent space Cox prediction
- Cross-study domain invariance through unsupervised learning
- Encoder/decoder access for latent representation extraction

**Loss Function:**
```
L_VAE = MSE(x, x̂) + β_KL · KL(q(z|x) || p(z))
L_total = L_VAE + β_surv · L_Cox(z)
```

**Usage:**
```python
vae = PathwayVAE(
    pathway_dim=50,
    latent_dim=32,
    beta_kl=1.0,
    beta_survival=0.5,  # Enable survival objective
)
z = vae.encode(X_pathway)  # Get latent codes
```

---

### 4. Domain Adversarial Network (`src/models/modern/domain_adversarial.py`)

DANN + CORAL for unsupervised cross-study alignment.

**Architecture:**
```
Shared Extractor: pathways → 256 → 128 → features (128)
                          ↓
        ┌───────────────────┴───────────────────┐
        ↓                                       ↓
Task Head (Cox)                        Domain Discriminator
Dense(64) → ReLU → Dense(1)           Dense(128) → ReLU → Dense(n_domains)
```

**Features:**
- Gradient reversal layer for adversarial training
- CORAL loss for correlation alignment (optional)
- Domain classification accuracy monitoring
- Feature extraction for downstream tasks
- Flexible number of domains (studies)

**Loss Function:**
```
L = L_Cox + λ_domain · L_domain + λ_coral · L_CORAL
L_domain = CrossEntropy(domain_pred, domain_labels)
L_CORAL = ||Cov_source - Cov_target||_F^2
```

**Usage:**
```python
dann = DANN(
    input_dim=50,
    num_domains=5,  # 5 studies
    lambda_domain=0.5,
    lambda_coral=0.1,
)
features = dann.extract_features(X)  # Aligned features
```

---

### 5. TabPFN Classifier (`src/models/modern/tabpfn_classifier.py`)

Wrapper for TabPFN (Prior Function Network) for tabular risk classification.

**Features:**
- Converts survival → binary risk classification
- Survival threshold dichotomization (median or custom)
- sklearn-compatible interface
- Optional feature scaling
- Sample size validation and warnings
- Probability estimates for calibration

**Usage:**
```python
classifier = TabPFNRiskClassifier(
    risk_threshold=10.0,  # Years
    use_scaling=True,
)
classifier.fit(X, y_times, event_indicators)
risk_pred = classifier.predict(X_test)
proba = classifier.predict_proba(X_test)
```

---

### 6. Late Fusion (`src/models/fusion/late_fusion.py`)

Three late fusion strategies combining pathway + clinical predictions.

**Three Strategies:**

1. **WeightedFusion**: Softmax-weighted average
   ```
   ŷ = softmax(w) · [ŷ_pathway, ŷ_clinical]
   ```

2. **StackingMetaLearner**: MLP on concatenated predictions
   ```
   ŷ = MLP([ŷ_pathway, ŷ_clinical])
   ```

3. **AttentionFusion**: Learned attention weights
   ```
   α = softmax(MLP([ŷ_pathway, ŷ_clinical]))
   ŷ = MLP(α ⊙ [ŷ_pathway, ŷ_clinical])
   ```

**Features:**
- Freezes component models during training
- Compares pathway-only vs clinical-only vs fused C-index
- Early stopping on validation C-index
- Test-time modality-specific predictions

**Usage:**
```python
fusion = LateFusion(
    pathway_model=DeepSurv(...),
    clinical_model=DeepSurv(...),
    fusion_strategy='attention',
    freeze_component_models=True,
)
pathway_pred, clinical_pred, fused_pred = fusion.predict_multimodal(
    X_pathway, X_clinical
)
```

---

### 7. Multimodal Attention Fusion (`src/models/fusion/multimodal_attention.py`)

Cross-attention fusion with clinical queries and pathway keys/values.

**Architecture:**
```
X_pathway → Dense → pathway_features (64)
                         ↓
                  CrossAttention
                    ↑     ↑
                    Q     K,V
                    |
X_clinical → Dense → clinical_features (64)
                         ↓
                  [clinical, attended_pathway]
                         ↓
                      Dense → log hazard
```

**Features:**
- Multi-head attention (4 heads default)
- Learned fine-grained pathway-clinical interactions
- Smaller model than DeepSurv but with explicit cross-attention
- Scaled dot-product attention with dropout

**Usage:**
```python
model = MultimodalAttentionSurvival(
    pathway_dim=50,
    clinical_dim=10,
    attention_hidden_dim=64,
    num_heads=4,
)
log_hazard = model(X_pathway, X_clinical)
```

---

### 8. Autoresearch Agent (`src/models/modern/autoresearch_agent.py`)

**CRITICAL IMPLEMENTATION** - Constrained autoresearch with frozen preprocessing.

**Core Principle:** Only model architecture, hyperparameters, and training config are tunable. Preprocessing is cryptographically sealed.

**Components:**

#### PreprocessingContract
```python
@dataclass
class PreprocessingContract:
    pathway_normalization: str       # Frozen: 'zscore', 'minmax', 'log1p'
    clinical_normalization: str      # Frozen
    missing_value_strategy: str      # Frozen: 'drop', 'median', 'mean'
    feature_selection_method: str    # Frozen: None, 'variance', 'mrmr'
    train_val_test_split: Tuple     # Frozen: (0.6, 0.2, 0.2)
    random_seed: int                # Frozen
    n_samples_total: int            # Frozen
    n_pathways: int                 # Frozen
    n_clinical_features: int        # Frozen
```

**Features:**
- SHA256 hash of entire contract
- Hash verification on instantiation
- Cannot be modified without invalidating hash
- Prevents silent preprocessing leakage

#### HyperparameterSpace
```python
@dataclass
class HyperparameterSpace:
    # Architecture
    hidden_dims: list
    dropout_rate: float
    use_batch_norm: bool

    # Training
    learning_rate: float
    weight_decay: float
    batch_size: int
    num_epochs: int
    warmup_epochs: int

    # Regularization
    l1_penalty: float
    gradient_clip_norm: float

    # Early stopping
    early_stopping_patience: int
    early_stopping_min_delta: float
```

#### AutoresearchAgent
Orchestrates constrained HPO with:
- Fixed experiment budget: `max_experiments` (default 20)
- Wall-clock time budget: `max_wall_clock_seconds` (default 3600s)
- Single metric: concordance index (C-index)
- Full MLflow logging per trial
- Ray Tune integration with ASHA scheduler
- Training/validation/test split from frozen contract

**MLflow Logging:**
- Preprocessing contract hash per run
- All hyperparameters
- Test C-index as primary metric
- Wall-clock elapsed time
- Trial ID and best result tracking

**Guard Rails:**
- Cannot instantiate without valid contract hash
- Preprocessing immutable during agent lifetime
- Only HyperparameterSpace fields are tunable
- Each trial logs contract hash for reproducibility

**Usage:**
```python
# Define frozen contract
contract = PreprocessingContract(
    pathway_normalization='zscore',
    clinical_normalization='minmax',
    missing_value_strategy='drop',
    feature_selection_method=None,
    feature_selection_k=None,
    train_val_test_split=(0.6, 0.2, 0.2),
    random_seed=42,
    n_samples_total=1000,
    n_pathways=50,
    n_clinical_features=10,
)
contract_hash = contract.compute_hash()

# Create agent (fails if hash doesn't match)
agent = AutoresearchAgent(
    preprocessing_contract=contract,
    preprocessing_hash=contract_hash,
    train_dataloader=...,
    val_dataloader=...,
    test_dataloader=...,
    model_factory=lambda hparams: DeepSurv(
        input_dim=50,
        hidden_dims=hparams.hidden_dims,
        dropout_rate=hparams.dropout_rate,
    ),
    max_experiments=20,
    max_wall_clock_seconds=3600.0,
)

# Run HPO (single metric: C-index)
search_space = create_search_space(base_hparams)
results = agent.search(search_space)

# Train final model
final_model = agent.finalize(results['best_hparams'])
```

---

### 9. Tests (`tests/test_modern_models.py`)

Comprehensive test suite with **37 test functions** covering:

**Test Classes:**
1. **TestTrainingUtils** - Loss functions, C-index, early stopping
2. **TestPathwayAutoencoder** - VAE forward pass, encode/decode, survival objective
3. **TestDANN** - DANN forward, feature extraction, CORAL loss
4. **TestDeepSurv** - Forward pass, hazard prediction, risk scores
5. **TestLateFusion** - All three fusion strategies
6. **TestMultimodalAttention** - Attention forward and predictions
7. **TestAutoresearchAgent** - Contract hashing, immutability, HPO constraints
8. **TestIntegration** - Full training with DataLoaders

**Coverage:**
- All model architectures
- All loss functions
- Preprocessing contract constraints
- Hyperparameter space validation
- Search space creation
- Integration with PyTorch Lightning

---

## Design Patterns

### 1. PyTorch Lightning Pattern
All models inherit from `pl.LightningModule`:
- `training_step()` - Training batch
- `validation_step()` - Validation with C-index
- `test_step()` - Test with detailed metrics
- `predict_step()` - Inference
- `configure_optimizers()` - Optimizer + scheduler

### 2. Constrained Optimization
**Autoresearch paradigm:**
- Frozen preprocessing (hash-verified)
- Editable surface: architecture, hyperparams, training config
- Fixed budget: N experiments, wall-clock time
- Single metric: C-index
- Full reproducibility via contract hash

### 3. Survival Analysis
**Cox Proportional Hazards:**
- Partial likelihood loss (handles censoring)
- C-index metric (concordance)
- Risk sets computed efficiently
- No assumptions on baseline hazard

### 4. Domain Adaptation
**DANN + CORAL:**
- Shared feature extractor
- Adversarial domain discriminator
- Correlation alignment loss
- Domain-invariant representations

### 5. Multimodal Fusion
**Three strategies:**
1. Weighted average (interpretable)
2. Stacking (flexible meta-learner)
3. Attention (fine-grained interactions)

---

## Dependencies

**Core:**
- `torch >= 2.0` - Deep learning
- `pytorch-lightning >= 2.0` - Training orchestration
- `numpy` - Numerical operations
- `scikit-learn` - Utilities, preprocessing

**Advanced:**
- `mlflow` - Experiment tracking
- `ray[tune]` - Hyperparameter optimization
- `tabpfn` - Prior function network (optional, sklearn-compatible)

**Testing:**
- `pytest` - Test framework

---

## Statistics & Metrics

**Model Summary:**
- **9 modules**: 9 Python files
- **18 classes**: Models, losses, utilities, agent
- **~2000 lines**: Production-quality code with docstrings
- **37 test functions**: Comprehensive coverage

**Loss Functions:**
1. Cox Partial Likelihood
2. Ranking Loss
3. VAE (Reconstruction + KL)
4. Domain Classification (Cross-entropy)
5. CORAL (Frobenius norm)

**Metrics:**
- Concordance Index (C-index) - primary
- Loss values (train/val/test)
- Domain classification accuracy
- Feature reconstruction error

---

## File Structure

```
src/models/
├── modern/
│   ├── __init__.py                    (imports)
│   ├── training_utils.py              (losses, metrics, schedulers)
│   ├── deepsurv.py                    (DeepSurv model)
│   ├── pathway_autoencoder.py         (VAE for pathways)
│   ├── domain_adversarial.py          (DANN for alignment)
│   ├── tabpfn_classifier.py           (TabPFN wrapper)
│   └── autoresearch_agent.py          (constrained HPO)
├── fusion/
│   ├── __init__.py                    (imports)
│   ├── late_fusion.py                 (3 fusion strategies)
│   └── multimodal_attention.py        (attention fusion)
tests/
└── test_modern_models.py              (comprehensive tests)
```

---

## Key Implementation Details

### 1. Numerical Stability
- Log-space Cox loss computation
- Epsilon clipping to prevent log(0)
- Gradient clipping for stability
- Batch normalization in networks

### 2. Reproducibility
- Fixed random seeds in fixtures
- Deterministic tensor operations
- MLflow logging of all hyperparams
- Preprocessing contract hash verification

### 3. Type Safety
- Full type hints on all functions
- `Optional` types for nullable args
- `Dict`, `List`, `Tuple` annotations
- `@dataclass` for configuration

### 4. Docstrings
- Google-style docstrings
- Argument and return value descriptions
- Example usage where relevant
- Mathematical notation in docstrings

### 5. Error Handling
- Preprocessing contract hash mismatch detection
- Sample size validation (TabPFN)
- Assertion checks for tensor shapes
- Meaningful error messages

---

## Modeling Roadmap (Usage Order)

**Phase 1: Classical Baselines**
1. DeepSurv on pathways
2. DeepSurv on clinical
3. Evaluate C-index

**Phase 2: Foundation Models**
1. Pathway VAE (unsupervised)
2. DANN for cross-study alignment
3. Extract aligned representations

**Phase 3: Multimodal Fusion**
1. Late fusion (3 strategies)
2. Multimodal attention fusion
3. Compare vs Phase 1

**Phase 4: Autoresearch Optimization**
1. Define preprocessing contract
2. Run AutoresearchAgent
3. Train final model on best hparams

---

## Production Checklist

- [x] All models PyTorch Lightning compatible
- [x] Full type hints and docstrings
- [x] Loss functions numerically stable
- [x] Preprocessing contract frozen and verified
- [x] MLflow logging for reproducibility
- [x] Ray Tune integration for HPO
- [x] Comprehensive test suite
- [x] Guard rails preventing preprocessing modification
- [x] C-index as single optimization metric
- [x] Wall-clock and experiment budget enforcement

---

## Next Steps

1. **Data preparation**: Implement DataLoader wrapping with preprocessing contract
2. **Training scripts**: Create experiment runner scripts
3. **Evaluation pipeline**: Add cross-validation and benchmarking
4. **Visualization**: Implement Kaplan-Meier curves, calibration plots
5. **Model serving**: Package trained models for inference

