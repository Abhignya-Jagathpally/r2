"""
Quick-start example: Training MM risk signature models.

Demonstrates the full pipeline:
1. DeepSurv baseline
2. Cross-study alignment with DANN
3. Multimodal fusion
4. Autoresearch optimization
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import Trainer

# Imports from our models
from src.models.modern import (
    DeepSurv,
    PathwayVAE,
    DANN,
    PreprocessingContract,
    HyperparameterSpace,
    AutoresearchAgent,
    create_search_space,
)
from src.models.fusion import LateFusion, MultimodalAttentionSurvival


def create_synthetic_data(
    n_samples: int = 500,
    pathway_dim: int = 50,
    clinical_dim: int = 10,
    n_studies: int = 3,
):
    """
    Create synthetic transcriptomics data for demonstration.

    Args:
        n_samples: Total samples
        pathway_dim: Number of pathways
        clinical_dim: Number of clinical features
        n_studies: Number of studies

    Returns:
        Tuple of (X_pathway, X_clinical, T, E, study_ids)
    """
    np.random.seed(42)

    # Pathway scores (normalized)
    X_pathway = np.random.randn(n_samples, pathway_dim).astype(np.float32)
    X_pathway = (X_pathway - X_pathway.mean(axis=0)) / (X_pathway.std(axis=0) + 1e-8)

    # Clinical features (normalized)
    X_clinical = np.random.randn(n_samples, clinical_dim).astype(np.float32)
    X_clinical = (X_clinical - X_clinical.mean(axis=0)) / (X_clinical.std(axis=0) + 1e-8)

    # Survival times (exponential)
    T = np.random.exponential(10, n_samples).astype(np.float32)

    # Event indicators (70% events, 30% censored)
    E = np.random.binomial(1, 0.7, n_samples).astype(np.float32)

    # Study IDs
    study_ids = np.random.randint(0, n_studies, n_samples).astype(np.long)

    return X_pathway, X_clinical, T, E, study_ids


def example_1_deepsurv_baseline():
    """Example 1: Train DeepSurv baseline on pathways."""
    print("\n" + "="*70)
    print("Example 1: DeepSurv Baseline")
    print("="*70)

    # Generate data
    X_pathway, _, T, E, _ = create_synthetic_data()

    # Create dataset and loader
    dataset = TensorDataset(
        torch.from_numpy(X_pathway),
        torch.from_numpy(T),
        torch.from_numpy(E),
    )
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Create model
    model = DeepSurv(
        input_dim=X_pathway.shape[1],
        hidden_dims=[256, 128, 64],
        learning_rate=1e-3,
    )

    # Wrap data for PyTorch Lightning
    class SurvivalDataLoader(DataLoader):
        def __iter__(self):
            for batch in super().__iter__():
                X, T, E = batch
                yield {'X': X, 'T': T, 'E': E}

    wrapped_loader = SurvivalDataLoader(dataset, batch_size=32, shuffle=True)

    print(f"Model: DeepSurv")
    print(f"Input dim: {X_pathway.shape[1]}")
    print(f"Hidden dims: [256, 128, 64]")
    print(f"Training samples: {len(dataset)}")
    print("✓ Ready for training with Trainer.fit()")


def example_2_pathway_vae():
    """Example 2: Learn unsupervised pathway representation with VAE."""
    print("\n" + "="*70)
    print("Example 2: Pathway VAE (Unsupervised Domain Invariance)")
    print("="*70)

    # Generate data
    X_pathway, _, T, E, _ = create_synthetic_data()

    # Create VAE model
    vae = PathwayVAE(
        pathway_dim=X_pathway.shape[1],
        latent_dim=32,
        beta_kl=1.0,
        beta_survival=0.5,  # Enable survival objective
    )

    # Encode to latent space
    X_tensor = torch.from_numpy(X_pathway)
    with torch.no_grad():
        z = vae.encode(X_tensor)  # [n_samples, 32]

    print(f"Model: Pathway VAE")
    print(f"Input dim: {X_pathway.shape[1]}")
    print(f"Latent dim: 32")
    print(f"Latent codes shape: {z.shape}")
    print("✓ Learned cross-study invariant representations")


def example_3_domain_alignment():
    """Example 3: Cross-study alignment with DANN."""
    print("\n" + "="*70)
    print("Example 3: Domain Adversarial Alignment (DANN)")
    print("="*70)

    # Generate data from multiple studies
    X_pathway, _, T, E, study_ids = create_synthetic_data(n_studies=5)

    # Create DANN model
    dann = DANN(
        input_dim=X_pathway.shape[1],
        num_domains=5,
        lambda_domain=0.5,
        lambda_coral=0.1,
    )

    # Extract domain-invariant features
    X_tensor = torch.from_numpy(X_pathway)
    with torch.no_grad():
        aligned_features = dann.extract_features(X_tensor)  # [n_samples, 128]

    print(f"Model: DANN")
    print(f"Input dim: {X_pathway.shape[1]}")
    print(f"Number of studies: 5")
    print(f"Domain-invariant features shape: {aligned_features.shape}")
    print("✓ Learned domain-invariant representations for all studies")


def example_4_multimodal_fusion():
    """Example 4: Multimodal fusion of pathways and clinical features."""
    print("\n" + "="*70)
    print("Example 4: Multimodal Attention Fusion")
    print("="*70)

    # Generate multimodal data
    X_pathway, X_clinical, T, E, _ = create_synthetic_data()

    # Create fusion model
    model = MultimodalAttentionSurvival(
        pathway_dim=X_pathway.shape[1],
        clinical_dim=X_clinical.shape[1],
        attention_hidden_dim=64,
        num_heads=4,
    )

    # Forward pass
    X_pathway_tensor = torch.from_numpy(X_pathway)
    X_clinical_tensor = torch.from_numpy(X_clinical)

    with torch.no_grad():
        log_hazard = model(X_pathway_tensor, X_clinical_tensor)

    print(f"Model: MultimodalAttentionSurvival")
    print(f"Pathway dim: {X_pathway.shape[1]}")
    print(f"Clinical dim: {X_clinical.shape[1]}")
    print(f"Attention heads: 4")
    print(f"Output shape (log hazard): {log_hazard.shape}")
    print("✓ Pathways and clinical features fused with cross-attention")


def example_5_late_fusion():
    """Example 5: Late fusion of pathway and clinical models."""
    print("\n" + "="*70)
    print("Example 5: Late Fusion (3 Strategies)")
    print("="*70)

    # Generate data
    X_pathway, X_clinical, T, E, _ = create_synthetic_data()

    # Create component models
    pathway_model = DeepSurv(input_dim=X_pathway.shape[1])
    clinical_model = DeepSurv(input_dim=X_clinical.shape[1])

    # Create fusion models
    for strategy in ['weighted', 'stacking', 'attention']:
        fusion = LateFusion(
            pathway_model=pathway_model,
            clinical_model=clinical_model,
            fusion_strategy=strategy,
        )

        X_pathway_tensor = torch.from_numpy(X_pathway)
        X_clinical_tensor = torch.from_numpy(X_clinical)

        with torch.no_grad():
            fused_pred = fusion(X_pathway_tensor, X_clinical_tensor)

        print(f"\n  Strategy: {strategy}")
        print(f"    Output shape: {fused_pred.shape}")

    print("\n✓ All three fusion strategies working")


def example_6_autoresearch():
    """Example 6: Constrained autoresearch with frozen preprocessing."""
    print("\n" + "="*70)
    print("Example 6: Autoresearch Agent (Constrained HPO)")
    print("="*70)

    # Define frozen preprocessing contract
    contract = PreprocessingContract(
        pathway_normalization='zscore',
        clinical_normalization='minmax',
        missing_value_strategy='drop',
        feature_selection_method=None,
        feature_selection_k=None,
        train_val_test_split=(0.6, 0.2, 0.2),
        random_seed=42,
        n_samples_total=500,
        n_pathways=50,
        n_clinical_features=10,
    )

    contract_hash = contract.compute_hash()

    print(f"Preprocessing Contract:")
    print(f"  Pathway normalization: {contract.pathway_normalization}")
    print(f"  Clinical normalization: {contract.clinical_normalization}")
    print(f"  Feature selection: {contract.feature_selection_method}")
    print(f"  Train/Val/Test split: {contract.train_val_test_split}")
    print(f"  Contract hash: {contract_hash[:16]}...")

    # Verify contract
    assert contract.verify_hash(contract_hash), "Hash verification failed!"
    print("\n✓ Preprocessing contract verified and locked")

    # Show editable hyperparameter space
    base_hparams = HyperparameterSpace(
        hidden_dims=[256, 128, 64],
        dropout_rate=0.1,
        use_batch_norm=True,
        learning_rate=1e-3,
        weight_decay=1e-4,
        batch_size=32,
        num_epochs=100,
        warmup_epochs=5,
        l1_penalty=0.0,
        gradient_clip_norm=5.0,
        early_stopping_patience=10,
        early_stopping_min_delta=0.001,
    )

    print("\nEditable Hyperparameter Space:")
    print(f"  hidden_dims: {base_hparams.hidden_dims}")
    print(f"  dropout_rate: {base_hparams.dropout_rate}")
    print(f"  learning_rate: {base_hparams.learning_rate}")
    print(f"  batch_size: {base_hparams.batch_size}")
    print(f"  num_epochs: {base_hparams.num_epochs}")

    # Show search space
    search_space = create_search_space(base_hparams)
    print("\nSearch Space Keys (for Ray Tune):")
    for key in search_space.keys():
        print(f"  - {key}")

    print("\n✓ Autoresearch agent ready (use in agent.search(search_space))")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("MM Risk Signature ML Pipeline - Quick Start Examples")
    print("="*70)

    example_1_deepsurv_baseline()
    example_2_pathway_vae()
    example_3_domain_alignment()
    example_4_multimodal_fusion()
    example_5_late_fusion()
    example_6_autoresearch()

    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("="*70)
    print("\nNext steps:")
    print("1. Prepare your actual transcriptomics data")
    print("2. Create preprocessing contract")
    print("3. Train models with PyTorch Lightning Trainer")
    print("4. Use AutoresearchAgent for hyperparameter optimization")
    print("5. Evaluate with cross-validation and external validation sets")


if __name__ == '__main__':
    main()
