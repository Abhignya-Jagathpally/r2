"""
DeepSurv: Deep Cox Proportional Hazards Model.

PyTorch Lightning implementation of deep neural network-based Cox PH model
for survival analysis with pathways and clinical features.

Reference: Faraggi & Simon (1995), Katzman et al. (2018)
"""

from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR

from .training_utils import (
    CoxPartialLikelihood,
    ConcordanceIndex,
    create_cosine_scheduler,
    EarlyStopping,
)


class DeepCoxMLP(nn.Module):
    """
    Deep Cox MLP for survival analysis.

    Architecture: pathway_features -> hidden layers -> 1 (log hazard)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = None,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
    ):
        """
        Args:
            input_dim: Number of pathway features
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # Output layer: single hazard score
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, input_dim] pathway features

        Returns:
            [batch_size, 1] log hazard ratios
        """
        return self.network(x)


class DeepSurv(pl.LightningModule):
    """
    DeepSurv: Deep Cox Proportional Hazards Model.

    Learns a deep representation of survival risk from high-dimensional
    pathway scores and clinical features.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = None,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        l1_penalty: float = 0.0,
    ):
        """
        Args:
            input_dim: Number of input features (pathways)
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability
            use_batch_norm: Whether to use batch normalization
            learning_rate: Initial learning rate
            weight_decay: L2 regularization (AdamW)
            l1_penalty: L1 regularization penalty on weights
        """
        super().__init__()
        self.save_hyperparameters()

        self.model = DeepCoxMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
        )

        self.cox_loss = CoxPartialLikelihood()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.l1_penalty = l1_penalty

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, input_dim]

        Returns:
            [batch_size, 1] log hazard
        """
        return self.model(x)

    def _compute_loss(
        self,
        log_hazard: torch.Tensor,
        event_times: torch.Tensor,
        event_indicators: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Cox loss with optional L1 penalty."""
        cox_loss = self.cox_loss(
            log_hazard.squeeze(),
            event_times,
            event_indicators,
        )

        # L1 regularization
        if self.l1_penalty > 0:
            l1_loss = sum(
                torch.abs(p).sum() for p in self.model.parameters()
            )
            cox_loss = cox_loss + self.l1_penalty * l1_loss

        return cox_loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Args:
            batch: Dictionary with keys:
                - 'X': [batch_size, input_dim]
                - 'T': [batch_size] event times
                - 'E': [batch_size] event indicators
        """
        X = batch['X']
        event_times = batch['T']
        event_indicators = batch['E']

        log_hazard = self(X)
        loss = self._compute_loss(log_hazard, event_times, event_indicators)

        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, float]:
        """Validation step with C-index computation."""
        X = batch['X']
        event_times = batch['T']
        event_indicators = batch['E']

        log_hazard = self(X)
        loss = self._compute_loss(log_hazard, event_times, event_indicators)

        # Compute C-index
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
        X = batch['X']
        event_times = batch['T']
        event_indicators = batch['E']

        log_hazard = self(X)
        loss = self._compute_loss(log_hazard, event_times, event_indicators)

        log_hazard_np = log_hazard.squeeze().detach().cpu().numpy()
        event_times_np = event_times.detach().cpu().numpy()
        event_indicators_np = event_indicators.detach().cpu().numpy()

        c_index = ConcordanceIndex.compute(
            log_hazard_np,
            event_times_np,
            event_indicators_np,
        )

        self.log('test_loss', loss, on_epoch=True)
        self.log('test_c_index', c_index, on_epoch=True)

        return {'test_loss': loss, 'test_c_index': c_index}

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Return log hazard predictions."""
        X = batch['X']
        return self(X).squeeze()

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
    def predict_hazard(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict log hazard ratios.

        Args:
            X: [n_samples, input_dim] features

        Returns:
            [n_samples] log hazard predictions
        """
        self.eval()
        return self(X).squeeze()

    @torch.no_grad()
    def predict_risk_score(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict risk scores (exp of log hazard).

        Args:
            X: [n_samples, input_dim]

        Returns:
            [n_samples] risk scores
        """
        return torch.exp(self.predict_hazard(X))
