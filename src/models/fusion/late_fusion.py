"""
Late Fusion for Multi-Modal Survival Analysis.

Three fusion strategies for combining pathway and clinical features:
1. Weighted combination: weighted sum of risk scores
2. Stacking: meta-learner on concatenated predictions
3. Attention-based: learnable attention weights on modalities

Late fusion delays the combination until after individual model predictions.
"""

from typing import Optional, Dict, Any, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..modern.training_utils import CoxPartialLikelihood, ConcordanceIndex


class WeightedFusion(nn.Module):
    """
    Weighted combination of pathway and clinical predictions.

    Simple linear combination with learned weights.
    """

    def __init__(self, n_modalities: int = 2):
        """
        Args:
            n_modalities: Number of modalities (pathways, clinical, etc.)
        """
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n_modalities) / n_modalities)

    def forward(self, *predictions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            *predictions: List of [batch_size, 1] predictions from each modality

        Returns:
            [batch_size, 1] fused predictions
        """
        stacked = torch.stack(predictions, dim=1)  # [batch_size, n_modalities, 1]
        normalized_weights = F.softmax(self.weights, dim=0)
        fused = (stacked * normalized_weights.view(1, -1, 1)).sum(dim=1)
        return fused


class StackingMetaLearner(nn.Module):
    """
    Stacking-based fusion with meta-learner.

    Concatenates predictions from individual models and trains
    a meta-learner to combine them.
    """

    def __init__(
        self,
        n_modalities: int = 2,
        hidden_dim: int = 64,
        dropout_rate: float = 0.1,
    ):
        """
        Args:
            n_modalities: Number of input modalities
            hidden_dim: Hidden dimension of meta-learner
            dropout_rate: Dropout probability
        """
        super().__init__()
        self.meta_learner = nn.Sequential(
            nn.Linear(n_modalities, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, *predictions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            *predictions: List of [batch_size, 1] predictions

        Returns:
            [batch_size, 1] fused predictions
        """
        stacked = torch.cat(predictions, dim=1)  # [batch_size, n_modalities]
        return self.meta_learner(stacked)


class AttentionFusion(nn.Module):
    """
    Attention-based fusion with learnable modality weights.

    Uses attention mechanism to learn dynamic weights for each modality.
    """

    def __init__(
        self,
        n_modalities: int = 2,
        hidden_dim: int = 64,
        dropout_rate: float = 0.1,
    ):
        """
        Args:
            n_modalities: Number of modalities
            hidden_dim: Hidden dimension for attention
            dropout_rate: Dropout probability
        """
        super().__init__()
        self.n_modalities = n_modalities

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(n_modalities, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, n_modalities),
            nn.Softmax(dim=1),
        )

        # Output projection
        self.output_layer = nn.Linear(n_modalities, 1)

    def forward(self, *predictions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            *predictions: List of [batch_size, 1] predictions

        Returns:
            [batch_size, 1] fused predictions
        """
        stacked = torch.cat(predictions, dim=1)  # [batch_size, n_modalities]
        attention_weights = self.attention(stacked)  # [batch_size, n_modalities]
        weighted = stacked * attention_weights
        return self.output_layer(weighted)


class LateFusion(pl.LightningModule):
    """
    Late Fusion Model for Multi-Modal Survival Analysis.

    Combines predictions from pathway and clinical models using
    one of three fusion strategies.
    """

    def __init__(
        self,
        pathway_model: nn.Module,
        clinical_model: nn.Module,
        fusion_strategy: str = 'attention',
        hidden_dim: int = 64,
        dropout_rate: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        freeze_component_models: bool = True,
    ):
        """
        Args:
            pathway_model: Pre-trained pathway prediction model
            clinical_model: Pre-trained clinical prediction model
            fusion_strategy: 'weighted', 'stacking', or 'attention'
            hidden_dim: Hidden dimension for meta-learner/attention
            dropout_rate: Dropout probability
            learning_rate: Adam learning rate
            weight_decay: L2 regularization
            freeze_component_models: Whether to freeze pathway/clinical models
        """
        super().__init__()
        self.save_hyperparameters(
            ignore=['pathway_model', 'clinical_model'],
        )

        self.pathway_model = pathway_model
        self.clinical_model = clinical_model
        self.fusion_strategy = fusion_strategy
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Freeze component models
        if freeze_component_models:
            for param in self.pathway_model.parameters():
                param.requires_grad = False
            for param in self.clinical_model.parameters():
                param.requires_grad = False

        # Fusion module
        if fusion_strategy == 'weighted':
            self.fusion = WeightedFusion(n_modalities=2)
        elif fusion_strategy == 'stacking':
            self.fusion = StackingMetaLearner(
                n_modalities=2,
                hidden_dim=hidden_dim,
                dropout_rate=dropout_rate,
            )
        elif fusion_strategy == 'attention':
            self.fusion = AttentionFusion(
                n_modalities=2,
                hidden_dim=hidden_dim,
                dropout_rate=dropout_rate,
            )
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")

        self.cox_loss = CoxPartialLikelihood()

    def forward(
        self,
        X_pathway: torch.Tensor,
        X_clinical: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            X_pathway: [batch_size, pathway_dim]
            X_clinical: [batch_size, clinical_dim]

        Returns:
            [batch_size, 1] fused log hazard predictions
        """
        # Get individual predictions
        with torch.no_grad():
            pathway_pred = self.pathway_model(X_pathway)
            clinical_pred = self.clinical_model(X_clinical)

        # Fuse predictions
        fused_pred = self.fusion(pathway_pred, clinical_pred)
        return fused_pred

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Args:
            batch: Dictionary with keys:
                - 'X_pathway': [batch_size, pathway_dim]
                - 'X_clinical': [batch_size, clinical_dim]
                - 'T': [batch_size] event times
                - 'E': [batch_size] event indicators
        """
        X_pathway = batch['X_pathway']
        X_clinical = batch['X_clinical']
        event_times = batch['T']
        event_indicators = batch['E']

        fused_pred = self(X_pathway, X_clinical)
        loss = self.cox_loss(
            fused_pred.squeeze(),
            event_times,
            event_indicators,
        )

        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, float]:
        """Validation with C-index."""
        X_pathway = batch['X_pathway']
        X_clinical = batch['X_clinical']
        event_times = batch['T']
        event_indicators = batch['E']

        fused_pred = self(X_pathway, X_clinical)
        loss = self.cox_loss(
            fused_pred.squeeze(),
            event_times,
            event_indicators,
        )

        # C-index
        fused_pred_np = fused_pred.squeeze().detach().cpu().numpy()
        event_times_np = event_times.detach().cpu().numpy()
        event_indicators_np = event_indicators.detach().cpu().numpy()

        c_index = ConcordanceIndex.compute(
            fused_pred_np,
            event_times_np,
            event_indicators_np,
        )

        self.log('val_loss', loss, on_epoch=True)
        self.log('val_c_index', c_index, on_epoch=True)

        return {'val_loss': loss, 'val_c_index': c_index}

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, float]:
        """Test step with detailed metrics."""
        X_pathway = batch['X_pathway']
        X_clinical = batch['X_clinical']
        event_times = batch['T']
        event_indicators = batch['E']

        # Get individual and fused predictions
        with torch.no_grad():
            pathway_pred = self.pathway_model(X_pathway)
            clinical_pred = self.clinical_model(X_clinical)
            fused_pred = self(X_pathway, X_clinical)

        loss = self.cox_loss(
            fused_pred.squeeze(),
            event_times,
            event_indicators,
        )

        # Convert to numpy
        pathway_pred_np = pathway_pred.squeeze().detach().cpu().numpy()
        clinical_pred_np = clinical_pred.squeeze().detach().cpu().numpy()
        fused_pred_np = fused_pred.squeeze().detach().cpu().numpy()
        event_times_np = event_times.detach().cpu().numpy()
        event_indicators_np = event_indicators.detach().cpu().numpy()

        # Compute C-indices
        c_index_pathway = ConcordanceIndex.compute(
            pathway_pred_np,
            event_times_np,
            event_indicators_np,
        )
        c_index_clinical = ConcordanceIndex.compute(
            clinical_pred_np,
            event_times_np,
            event_indicators_np,
        )
        c_index_fused = ConcordanceIndex.compute(
            fused_pred_np,
            event_times_np,
            event_indicators_np,
        )

        metrics = {
            'test_loss': loss,
            'test_c_index_pathway': c_index_pathway,
            'test_c_index_clinical': c_index_clinical,
            'test_c_index_fused': c_index_fused,
        }

        for k, v in metrics.items():
            self.log(k, v, on_epoch=True)

        return metrics

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Return fused predictions."""
        X_pathway = batch['X_pathway']
        X_clinical = batch['X_clinical']
        return self(X_pathway, X_clinical).squeeze()

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and scheduler."""
        # Only optimize fusion module (component models frozen)
        optimizer = torch.optim.AdamW(
            self.fusion.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            },
        }

    @torch.no_grad()
    def predict_multimodal(
        self,
        X_pathway: torch.Tensor,
        X_clinical: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get individual and fused predictions.

        Args:
            X_pathway: [n_samples, pathway_dim]
            X_clinical: [n_samples, clinical_dim]

        Returns:
            pathway_pred: [n_samples]
            clinical_pred: [n_samples]
            fused_pred: [n_samples]
        """
        self.eval()
        pathway_pred = self.pathway_model(X_pathway).squeeze()
        clinical_pred = self.clinical_model(X_clinical).squeeze()
        fused_pred = self(X_pathway, X_clinical).squeeze()
        return pathway_pred, clinical_pred, fused_pred
