"""
Multimodal Attention-Based Fusion for Survival Analysis.

Cross-attention fusion where clinical features serve as queries
and pathway features serve as keys/values. Enables fine-grained
interaction between modalities.

Smaller model compared to DeepSurv but with explicit multimodal interaction.
"""

from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..modern.training_utils import CoxPartialLikelihood, ConcordanceIndex


class MultiheadCrossAttention(nn.Module):
    """
    Multi-head cross-attention module.

    Query from clinical, Keys/Values from pathways.
    """

    def __init__(
        self,
        query_dim: int,
        kv_dim: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
    ):
        """
        Args:
            query_dim: Dimension of query (clinical features)
            kv_dim: Dimension of keys/values (pathway features)
            hidden_dim: Hidden dimension of attention
            num_heads: Number of attention heads
            dropout_rate: Dropout probability
        """
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Linear projections
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(kv_dim, hidden_dim)
        self.value_proj = nn.Linear(kv_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout_rate)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query: [batch_size, query_dim] clinical features
            key: [batch_size, kv_dim] pathway features
            value: [batch_size, kv_dim] pathway features

        Returns:
            [batch_size, hidden_dim] attended features
        """
        batch_size = query.shape[0]

        # Project to hidden dimension
        Q = self.query_proj(query)  # [batch_size, hidden_dim]
        K = self.key_proj(key)      # [batch_size, hidden_dim]
        V = self.value_proj(value)  # [batch_size, hidden_dim]

        # Reshape for multi-head attention
        # [batch_size, num_heads, head_dim]
        Q = Q.view(batch_size, self.num_heads, self.head_dim)
        K = K.view(batch_size, self.num_heads, self.head_dim)
        V = V.view(batch_size, self.num_heads, self.head_dim)

        # Attention scores: dot product per head
        scores = torch.einsum('bhd,bhd->bh', Q, K) * self.scale
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        attended = torch.einsum('bh,bhd->bhd', attention_weights, V)

        # Concatenate heads
        attended = attended.contiguous().view(batch_size, self.hidden_dim)

        # Output projection
        output = self.output_proj(attended)

        return output


class MultimodalAttentionSurvival(pl.LightningModule):
    """
    Multimodal Attention-Based Survival Model.

    Combines pathways and clinical features using cross-attention,
    outputting survival predictions (log hazard).

    Smaller model suitable for balanced pathway+clinical integration.
    """

    def __init__(
        self,
        pathway_dim: int,
        clinical_dim: int,
        attention_hidden_dim: int = 64,
        num_heads: int = 4,
        survival_hidden_dim: int = 32,
        dropout_rate: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        use_batch_norm: bool = True,
    ):
        """
        Args:
            pathway_dim: Number of pathway features
            clinical_dim: Number of clinical features
            attention_hidden_dim: Hidden dimension for attention
            num_heads: Number of attention heads
            survival_hidden_dim: Hidden dimension for survival head
            dropout_rate: Dropout probability
            learning_rate: Adam learning rate
            weight_decay: L2 regularization
            use_batch_norm: Use batch normalization in feature projections
        """
        super().__init__()
        self.save_hyperparameters()

        self.pathway_dim = pathway_dim
        self.clinical_dim = clinical_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Feature projections (optional batch norm)
        pathway_layers = [nn.Linear(pathway_dim, attention_hidden_dim)]
        if use_batch_norm:
            pathway_layers.append(nn.BatchNorm1d(attention_hidden_dim))
        pathway_layers.append(nn.ReLU())
        pathway_layers.append(nn.Dropout(dropout_rate))
        self.pathway_projection = nn.Sequential(*pathway_layers)

        clinical_layers = [nn.Linear(clinical_dim, attention_hidden_dim)]
        if use_batch_norm:
            clinical_layers.append(nn.BatchNorm1d(attention_hidden_dim))
        clinical_layers.append(nn.ReLU())
        clinical_layers.append(nn.Dropout(dropout_rate))
        self.clinical_projection = nn.Sequential(*clinical_layers)

        # Cross-attention: clinical query, pathways key/value
        self.attention = MultiheadCrossAttention(
            query_dim=attention_hidden_dim,
            kv_dim=attention_hidden_dim,
            hidden_dim=attention_hidden_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
        )

        # Fusion: concatenate clinical + attended pathway
        fusion_input_dim = attention_hidden_dim * 2

        # Survival head
        self.survival_head = nn.Sequential(
            nn.Linear(fusion_input_dim, survival_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(survival_hidden_dim, 1),
        )

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
            [batch_size, 1] log hazard
        """
        # Project features
        pathway_feat = self.pathway_projection(X_pathway)  # [batch_size, hidden_dim]
        clinical_feat = self.clinical_projection(X_clinical)  # [batch_size, hidden_dim]

        # Cross-attention
        attended_pathway = self.attention(
            query=clinical_feat,
            key=pathway_feat,
            value=pathway_feat,
        )  # [batch_size, hidden_dim]

        # Fuse: concatenate
        fused = torch.cat([clinical_feat, attended_pathway], dim=1)

        # Survival prediction
        log_hazard = self.survival_head(fused)

        return log_hazard

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

        log_hazard = self(X_pathway, X_clinical)
        loss = self.cox_loss(
            log_hazard.squeeze(),
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

        log_hazard = self(X_pathway, X_clinical)
        loss = self.cox_loss(
            log_hazard.squeeze(),
            event_times,
            event_indicators,
        )

        # C-index
        log_hazard_np = log_hazard.squeeze().detach().cpu().numpy()
        event_times_np = event_times.detach().cpu().numpy()
        event_indicators_np = event_indicators.detach().cpu().numpy()

        c_index = ConcordanceIndex.compute(
            log_hazard_np,
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

        log_hazard = self(X_pathway, X_clinical)
        loss = self.cox_loss(
            log_hazard.squeeze(),
            event_times,
            event_indicators,
        )

        log_hazard_np = log_hazard.squeeze().detach().cpu().numpy()
        event_times_np = event_times.detach().cpu().numpy()
        event_indicators_np = event_indicators.detach().cpu().numpy()

        c_index = ConcordanceIndex.compute(
            log_hazard_np,
            event_times_np,
            event_indicators_np,
        )

        metrics = {
            'test_loss': loss,
            'test_c_index': c_index,
        }

        for k, v in metrics.items():
            self.log(k, v, on_epoch=True)

        return metrics

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Return log hazard predictions."""
        X_pathway = batch['X_pathway']
        X_clinical = batch['X_clinical']
        return self(X_pathway, X_clinical).squeeze()

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
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
    def predict_hazard(
        self,
        X_pathway: torch.Tensor,
        X_clinical: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict log hazard ratios.

        Args:
            X_pathway: [n_samples, pathway_dim]
            X_clinical: [n_samples, clinical_dim]

        Returns:
            [n_samples] log hazard
        """
        self.eval()
        return self(X_pathway, X_clinical).squeeze()
