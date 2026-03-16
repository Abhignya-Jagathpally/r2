"""
Domain Adversarial Neural Network (DANN) for cross-study alignment.

Implements domain adversarial training with gradient reversal layer
and CORAL (CORrelation ALignment) loss for unsupervised domain adaptation
in survival analysis across multiple studies.

Reference: Ganin & Lempitsky (2015), Sun & Saenko (2016)
"""

from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR

from .training_utils import CoxPartialLikelihood, ConcordanceIndex


class GradientReversal(torch.autograd.Function):
    """Gradient reversal layer for adversarial training."""

    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lam, None


class FeatureExtractor(nn.Module):
    """Shared feature extractor for both task and domain adaptation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = None,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability
            use_batch_norm: Use batch normalization
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

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

        self.network = nn.Sequential(*layers)
        self.output_dim = prev_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, input_dim]

        Returns:
            [batch_size, output_dim] features
        """
        return self.network(x)


class SurvivalHead(nn.Module):
    """Task head for survival analysis (Cox proportional hazards)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout_rate: float = 0.1,
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            dropout_rate: Dropout probability
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, input_dim]

        Returns:
            [batch_size, 1] log hazard
        """
        return self.network(x)


class DomainDiscriminator(nn.Module):
    """Domain discriminator for adversarial training."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_domains: int = 2,
        dropout_rate: float = 0.1,
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_domains: Number of domains (studies)
            dropout_rate: Dropout probability
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_domains),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, input_dim]

        Returns:
            [batch_size, num_domains] domain logits
        """
        return self.network(x)


class DANN(pl.LightningModule):
    """
    Domain Adversarial Neural Network for cross-study alignment.

    Uses adversarial training and/or CORAL loss to learn domain-invariant
    features for survival prediction across multiple studies.
    """

    def __init__(
        self,
        input_dim: int,
        num_domains: int,
        extractor_hidden_dims: list = None,
        survival_hidden_dim: int = 64,
        discriminator_hidden_dim: int = 128,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        lambda_domain: float = 0.5,
        lambda_coral: float = 0.0,
        gradient_reversal_weight: float = 1.0,
    ):
        """
        Args:
            input_dim: Input feature dimension
            num_domains: Number of domains (studies)
            extractor_hidden_dims: Hidden dims for feature extractor
            survival_hidden_dim: Hidden dim for survival head
            discriminator_hidden_dim: Hidden dim for domain discriminator
            dropout_rate: Dropout probability
            use_batch_norm: Use batch normalization
            learning_rate: Adam learning rate
            weight_decay: L2 regularization
            lambda_domain: Weight of adversarial domain loss
            lambda_coral: Weight of CORAL loss (0 = disabled)
            gradient_reversal_weight: Weight of gradient reversal
        """
        super().__init__()
        self.save_hyperparameters()

        self.input_dim = input_dim
        self.num_domains = num_domains
        self.lambda_domain = lambda_domain
        self.lambda_coral = lambda_coral
        self.gradient_reversal_weight = gradient_reversal_weight
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Shared feature extractor
        self.extractor = FeatureExtractor(
            input_dim=input_dim,
            hidden_dims=extractor_hidden_dims,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
        )

        # Task head (survival)
        self.survival_head = SurvivalHead(
            input_dim=self.extractor.output_dim,
            hidden_dim=survival_hidden_dim,
            dropout_rate=dropout_rate,
        )

        # Domain discriminator
        self.domain_discriminator = DomainDiscriminator(
            input_dim=self.extractor.output_dim,
            hidden_dim=discriminator_hidden_dim,
            num_domains=num_domains,
            dropout_rate=dropout_rate,
        )

        self.cox_loss = CoxPartialLikelihood()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, input_dim]

        Returns:
            log_hazard: [batch_size, 1]
            domain_logits: [batch_size, num_domains]
        """
        features = self.extractor(x)
        log_hazard = self.survival_head(features)
        domain_logits = self.domain_discriminator(features)
        return log_hazard, domain_logits

    def _compute_coral_loss(
        self,
        features_source: torch.Tensor,
        features_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        CORAL (CORrelation ALignment) loss.

        Aligns correlation matrices of source and target features.

        Args:
            features_source: [n_source, feature_dim]
            features_target: [n_target, feature_dim]

        Returns:
            CORAL loss
        """
        # Center features
        source_mean = features_source.mean(dim=0)
        target_mean = features_target.mean(dim=0)

        features_source_centered = features_source - source_mean
        features_target_centered = features_target - target_mean

        # Compute correlation matrices
        n_source = features_source.shape[0]
        n_target = features_target.shape[0]

        cov_source = (
            (features_source_centered.T @ features_source_centered)
            / (n_source - 1)
        )
        cov_target = (
            (features_target_centered.T @ features_target_centered)
            / (n_target - 1)
        )

        # CORAL loss: Frobenius norm of difference
        coral_loss = torch.norm(cov_source - cov_target, p='fro') ** 2

        return coral_loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Args:
            batch: Dictionary with keys:
                - 'X': [batch_size, input_dim]
                - 'T': [batch_size] event times
                - 'E': [batch_size] event indicators
                - 'domain': [batch_size] domain/study ID
        """
        X = batch['X']
        event_times = batch['T']
        event_indicators = batch['E']
        domain_labels = batch['domain']

        # Forward pass
        log_hazard, domain_logits = self(X)

        # Task loss (Cox partial likelihood)
        task_loss = self.cox_loss(
            log_hazard.squeeze(),
            event_times,
            event_indicators,
        )

        # Domain loss
        domain_loss = F.cross_entropy(domain_logits, domain_labels.long())

        # Total loss
        total_loss = task_loss + self.lambda_domain * domain_loss

        # Optional CORAL loss
        if self.lambda_coral > 0:
            # Need to separate source and target domains
            # Assuming first domain is source, rest are target
            source_mask = domain_labels == 0
            target_mask = ~source_mask

            if source_mask.sum() > 0 and target_mask.sum() > 0:
                features = self.extractor(X)
                coral_loss = self._compute_coral_loss(
                    features[source_mask],
                    features[target_mask],
                )
                total_loss = total_loss + self.lambda_coral * coral_loss
                self.log('train_coral_loss', coral_loss, on_epoch=True)

        self.log('train_loss', total_loss, on_epoch=True)
        self.log('train_task_loss', task_loss, on_epoch=True)
        self.log('train_domain_loss', domain_loss, on_epoch=True)

        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, float]:
        """Validation with survival and domain metrics."""
        X = batch['X']
        event_times = batch['T']
        event_indicators = batch['E']
        domain_labels = batch['domain']

        log_hazard, domain_logits = self(X)

        # Task loss
        task_loss = self.cox_loss(
            log_hazard.squeeze(),
            event_times,
            event_indicators,
        )

        # Domain loss
        domain_loss = F.cross_entropy(domain_logits, domain_labels.long())

        # C-index
        log_hazard_np = log_hazard.squeeze().detach().cpu().numpy()
        event_times_np = event_times.detach().cpu().numpy()
        event_indicators_np = event_indicators.detach().cpu().numpy()

        c_index = ConcordanceIndex.compute(
            log_hazard_np,
            event_times_np,
            event_indicators_np,
        )

        # Domain accuracy
        domain_pred = domain_logits.argmax(dim=1)
        domain_acc = (domain_pred == domain_labels).float().mean()

        metrics = {
            'val_loss': task_loss + self.lambda_domain * domain_loss,
            'val_task_loss': task_loss,
            'val_domain_loss': domain_loss,
            'val_c_index': c_index,
            'val_domain_acc': domain_acc,
        }

        for k, v in metrics.items():
            self.log(k, v, on_epoch=True)

        return metrics

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, float]:
        """Test step with detailed metrics."""
        X = batch['X']
        event_times = batch['T']
        event_indicators = batch['E']

        log_hazard, _ = self(X)

        # Task loss
        task_loss = self.cox_loss(
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

        metrics = {
            'test_loss': task_loss,
            'test_c_index': c_index,
        }

        for k, v in metrics.items():
            self.log(k, v, on_epoch=True)

        return metrics

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Return log hazard predictions."""
        X = batch['X']
        log_hazard, _ = self(X)
        return log_hazard.squeeze()

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
    def extract_features(self, X: torch.Tensor) -> torch.Tensor:
        """
        Extract learned features from shared extractor.

        Args:
            X: [n_samples, input_dim]

        Returns:
            [n_samples, feature_dim] extracted features
        """
        self.eval()
        return self.extractor(X)

    @torch.no_grad()
    def predict_hazard(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict log hazard ratios.

        Args:
            X: [n_samples, input_dim]

        Returns:
            [n_samples] log hazard
        """
        self.eval()
        log_hazard, _ = self(X)
        return log_hazard.squeeze()
