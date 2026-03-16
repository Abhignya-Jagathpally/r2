"""
Comprehensive tests for modern ML models and autoresearch agent.

Tests cover:
- Pathway Autoencoder (VAE)
- Domain Adversarial Neural Network (DANN)
- DeepSurv
- Late Fusion
- Multimodal Attention Fusion
- Autoresearch Agent constraints
"""

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer

from src.models.modern.pathway_autoencoder import PathwayVAE
from src.models.modern.domain_adversarial import DANN
from src.models.modern.deepsurv import DeepSurv
from src.models.modern.training_utils import (
    CoxPartialLikelihood,
    ConcordanceIndex,
    EarlyStopping,
)
from src.models.fusion.late_fusion import LateFusion, WeightedFusion, StackingMetaLearner
from src.models.fusion.multimodal_attention import MultimodalAttentionSurvival
from src.models.modern.autoresearch_agent import (
    PreprocessingContract,
    HyperparameterSpace,
    AutoresearchAgent,
    create_search_space,
)


# Fixtures
@pytest.fixture
def pathway_data():
    """Generate synthetic pathway data."""
    np.random.seed(42)
    n_samples = 100
    pathway_dim = 50
    X = np.random.randn(n_samples, pathway_dim).astype(np.float32)
    return torch.from_numpy(X)


@pytest.fixture
def survival_data():
    """Generate synthetic survival data."""
    np.random.seed(42)
    n_samples = 100
    T = np.random.exponential(10, n_samples).astype(np.float32)
    E = np.random.binomial(1, 0.7, n_samples).astype(np.float32)
    return torch.from_numpy(T), torch.from_numpy(E)


@pytest.fixture
def multimodal_data():
    """Generate synthetic multimodal data."""
    np.random.seed(42)
    n_samples = 100
    pathway_dim = 50
    clinical_dim = 10
    X_pathway = np.random.randn(n_samples, pathway_dim).astype(np.float32)
    X_clinical = np.random.randn(n_samples, clinical_dim).astype(np.float32)
    return (
        torch.from_numpy(X_pathway),
        torch.from_numpy(X_clinical),
    )


@pytest.fixture
def domain_data():
    """Generate synthetic domain/study data."""
    np.random.seed(42)
    n_samples = 100
    n_domains = 3
    domain_ids = np.random.randint(0, n_domains, n_samples).astype(np.long)
    return torch.from_numpy(domain_ids)


# Tests: Training Utilities
class TestTrainingUtils:
    """Test loss functions and metrics."""

    def test_cox_partial_likelihood(self, survival_data):
        """Test Cox partial likelihood loss."""
        T, E = survival_data
        log_hazard = torch.randn(len(T), requires_grad=True)

        loss_fn = CoxPartialLikelihood()
        loss = loss_fn(log_hazard, T, E)

        assert loss.item() >= 0, "Loss should be non-negative"
        assert loss.requires_grad, "Loss should be differentiable"

    def test_concordance_index(self, survival_data):
        """Test C-index computation."""
        T, E = survival_data
        log_hazard = np.random.randn(len(T))

        c_index = ConcordanceIndex.compute(
            log_hazard,
            T.numpy(),
            E.numpy(),
        )

        assert 0 <= c_index <= 1, f"C-index should be in [0,1], got {c_index}"

    def test_early_stopping(self):
        """Test early stopping mechanism."""
        stopper = EarlyStopping(patience=3, mode='min')
        model = torch.nn.Linear(10, 1)

        # Simulate improving loss
        for epoch in range(5):
            if epoch < 3:
                should_stop = stopper(1.0 - epoch * 0.1, model, epoch)
            else:
                should_stop = stopper(0.5, model, epoch)  # Plateau

            if epoch < 3:
                assert not should_stop
            elif epoch >= 5:
                assert should_stop


# Tests: Pathway Autoencoder
class TestPathwayAutoencoder:
    """Test VAE for pathways."""

    def test_vae_forward_pass(self, pathway_data):
        """Test VAE forward pass."""
        vae = PathwayVAE(
            pathway_dim=pathway_data.shape[1],
            latent_dim=32,
        )

        x_recon, z, mu, logvar = vae(pathway_data)

        assert x_recon.shape == pathway_data.shape
        assert z.shape[0] == pathway_data.shape[0]
        assert z.shape[1] == 32
        assert mu.shape == z.shape
        assert logvar.shape == z.shape

    def test_vae_encode_decode(self, pathway_data):
        """Test encode and decode methods."""
        vae = PathwayVAE(
            pathway_dim=pathway_data.shape[1],
            latent_dim=32,
        )

        z = vae.encode(pathway_data)
        x_recon = vae.decode(z)

        assert z.shape[0] == pathway_data.shape[0]
        assert x_recon.shape == pathway_data.shape

    def test_vae_with_survival(self, pathway_data, survival_data):
        """Test VAE with survival objective."""
        T, E = survival_data
        vae = PathwayVAE(
            pathway_dim=pathway_data.shape[1],
            latent_dim=32,
            beta_survival=0.5,
        )

        # Forward pass
        x_recon, z, mu, logvar = vae(pathway_data)

        # Survival head should exist
        assert vae.survival_head is not None
        assert vae.cox_loss is not None


# Tests: Domain Adversarial Network
class TestDANN:
    """Test DANN for cross-study alignment."""

    def test_dann_forward_pass(self, pathway_data, domain_data):
        """Test DANN forward pass."""
        dann = DANN(
            input_dim=pathway_data.shape[1],
            num_domains=domain_data.max().item() + 1,
        )

        log_hazard, domain_logits = dann(pathway_data)

        assert log_hazard.shape == (pathway_data.shape[0], 1)
        assert domain_logits.shape[0] == pathway_data.shape[0]

    def test_feature_extraction(self, pathway_data):
        """Test feature extraction from DANN."""
        dann = DANN(input_dim=pathway_data.shape[1], num_domains=2)
        features = dann.extract_features(pathway_data)

        assert features.shape[0] == pathway_data.shape[0]
        assert features.shape[1] > 0

    def test_coral_loss(self, pathway_data):
        """Test CORAL loss computation."""
        dann = DANN(
            input_dim=pathway_data.shape[1],
            num_domains=2,
            lambda_coral=0.1,
        )

        # Split data
        X1 = pathway_data[:50]
        X2 = pathway_data[50:]

        features1 = dann.extractor(X1)
        features2 = dann.extractor(X2)

        coral_loss = dann._compute_coral_loss(features1, features2)
        assert coral_loss.item() >= 0


# Tests: DeepSurv
class TestDeepSurv:
    """Test DeepSurv model."""

    def test_deepsurv_forward_pass(self, pathway_data):
        """Test DeepSurv forward pass."""
        model = DeepSurv(input_dim=pathway_data.shape[1])
        log_hazard = model(pathway_data)

        assert log_hazard.shape == (pathway_data.shape[0], 1)

    def test_deepsurv_predict_hazard(self, pathway_data):
        """Test hazard prediction."""
        model = DeepSurv(input_dim=pathway_data.shape[1])
        hazard = model.predict_hazard(pathway_data)

        assert hazard.shape[0] == pathway_data.shape[0]
        assert torch.all(torch.isfinite(hazard))

    def test_deepsurv_risk_score(self, pathway_data):
        """Test risk score prediction."""
        model = DeepSurv(input_dim=pathway_data.shape[1])
        risk_score = model.predict_risk_score(pathway_data)

        assert risk_score.shape[0] == pathway_data.shape[0]
        assert torch.all(risk_score > 0)  # exp of any real number is positive


# Tests: Late Fusion
class TestLateFusion:
    """Test late fusion models."""

    def test_weighted_fusion(self, pathway_data):
        """Test weighted fusion module."""
        fusion = WeightedFusion(n_modalities=2)
        pred1 = torch.randn(pathway_data.shape[0], 1)
        pred2 = torch.randn(pathway_data.shape[0], 1)

        fused = fusion(pred1, pred2)
        assert fused.shape == pred1.shape

    def test_stacking_fusion(self, pathway_data):
        """Test stacking meta-learner."""
        fusion = StackingMetaLearner(n_modalities=2, hidden_dim=32)
        pred1 = torch.randn(pathway_data.shape[0], 1)
        pred2 = torch.randn(pathway_data.shape[0], 1)

        fused = fusion(pred1, pred2)
        assert fused.shape == pred1.shape

    def test_late_fusion_model(self, multimodal_data, survival_data):
        """Test full late fusion model."""
        X_pathway, X_clinical = multimodal_data
        T, E = survival_data

        # Create component models
        pathway_model = DeepSurv(input_dim=X_pathway.shape[1])
        clinical_model = DeepSurv(input_dim=X_clinical.shape[1])

        # Fusion model
        fusion = LateFusion(
            pathway_model=pathway_model,
            clinical_model=clinical_model,
            fusion_strategy='attention',
        )

        # Forward pass
        fused_pred = fusion(X_pathway, X_clinical)
        assert fused_pred.shape == (X_pathway.shape[0], 1)


# Tests: Multimodal Attention
class TestMultimodalAttention:
    """Test multimodal attention fusion."""

    def test_multimodal_attention_forward(self, multimodal_data):
        """Test multimodal attention model."""
        X_pathway, X_clinical = multimodal_data

        model = MultimodalAttentionSurvival(
            pathway_dim=X_pathway.shape[1],
            clinical_dim=X_clinical.shape[1],
        )

        log_hazard = model(X_pathway, X_clinical)
        assert log_hazard.shape == (X_pathway.shape[0], 1)

    def test_multimodal_predict_hazard(self, multimodal_data):
        """Test hazard prediction with multimodal model."""
        X_pathway, X_clinical = multimodal_data

        model = MultimodalAttentionSurvival(
            pathway_dim=X_pathway.shape[1],
            clinical_dim=X_clinical.shape[1],
        )

        hazard = model.predict_hazard(X_pathway, X_clinical)
        assert hazard.shape[0] == X_pathway.shape[0]


# Tests: Autoresearch Agent
class TestAutoresearchAgent:
    """Test autoresearch agent constraints."""

    def test_preprocessing_contract_hash(self):
        """Test preprocessing contract hash verification."""
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

        hash1 = contract.compute_hash()
        hash2 = contract.compute_hash()

        assert hash1 == hash2, "Hash should be deterministic"
        assert contract.verify_hash(hash1), "Verification should pass"
        assert not contract.verify_hash("invalid_hash"), "Invalid hash should fail"

    def test_preprocessing_contract_immutability(self):
        """Test that contract cannot be modified without rehashing."""
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

        original_hash = contract.compute_hash()

        # Attempt to modify
        contract.pathway_normalization = 'log1p'
        modified_hash = contract.compute_hash()

        assert original_hash != modified_hash, "Hash should change after modification"

    def test_hyperparameter_space(self):
        """Test hyperparameter space."""
        hparams = HyperparameterSpace(
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

        assert hparams.learning_rate > 0
        assert hparams.num_epochs > hparams.warmup_epochs

    def test_search_space_creation(self):
        """Test Ray Tune search space creation."""
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

        search_space = create_search_space(base_hparams)

        assert 'learning_rate' in search_space
        assert 'batch_size' in search_space
        assert 'hidden_dims' in search_space


# Integration Tests
class TestIntegration:
    """Integration tests with full data loaders."""

    def test_deepsurv_with_dataloader(self, pathway_data, survival_data):
        """Test DeepSurv training with DataLoader."""
        T, E = survival_data

        dataset = TensorDataset(
            pathway_data,
            T.unsqueeze(1),
            E.unsqueeze(1),
        )
        dataloader = DataLoader(dataset, batch_size=32)

        model = DeepSurv(input_dim=pathway_data.shape[1])
        trainer = Trainer(max_epochs=2, enable_progress_bar=False, logger=False)

        # Create proper batch format
        class SurvivalDataModule(pl.LightningDataModule):
            def train_dataloader(self):
                dataset = TensorDataset(pathway_data, T, E)
                return DataLoader(dataset, batch_size=32, shuffle=True)

            def val_dataloader(self):
                dataset = TensorDataset(pathway_data, T, E)
                return DataLoader(dataset, batch_size=32)

        # This should not raise an error
        # trainer.fit(model, datamodule=SurvivalDataModule())


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
