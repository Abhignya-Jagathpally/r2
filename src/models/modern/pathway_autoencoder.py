"""
Pathway Autoencoder: Variational Autoencoder on pathway scores.

VAE for learning cross-study invariant latent representations of
pathway activities with optional survival-aware objectives.

Architecture: pathway_dim -> 128 -> 64 -> latent(32) -> 64 -> 128 -> pathway_dim
Loss: Reconstruction + KL divergence + optional survival objective (DeepSurv-style)
"""

from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR

from .training_utils import CoxPartialLikelihood, ConcordanceIndex


class PathwayVAEEncoder(nn.Module):
    """VAE Encoder: pathway_dim -> latent(latent_dim)"""

    def __init__(
        self,
        pathway_dim: int,
        hidden_dims: list = None,
        latent_dim: int = 32,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.1,
    ):
        """
        Args:
            pathway_dim: Number of pathway features
            hidden_dims: List of hidden dimensions (default: [128, 64])
            latent_dim: Dimension of latent space
            use_batch_norm: Use batch normalization
            dropout_rate: Dropout probability
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers = []
        prev_dim = pathway_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Latent distribution: mean and log variance
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, pathway_dim]

        Returns:
            z: [batch_size, latent_dim] - latent samples
            mu: [batch_size, latent_dim] - mean
            logvar: [batch_size, latent_dim] - log variance
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return z, mu, logvar


class PathwayVAEDecoder(nn.Module):
    """VAE Decoder: latent(latent_dim) -> pathway_dim"""

    def __init__(
        self,
        pathway_dim: int,
        hidden_dims: list = None,
        latent_dim: int = 32,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.1,
    ):
        """
        Args:
            pathway_dim: Number of pathway features (reconstruction target)
            hidden_dims: List of hidden dimensions (default: [64, 128])
            latent_dim: Dimension of latent space
            use_batch_norm: Use batch normalization
            dropout_rate: Dropout probability
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128]

        layers = []
        prev_dim = latent_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, pathway_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [batch_size, latent_dim]

        Returns:
            [batch_size, pathway_dim] reconstructed pathways
        """
        return self.decoder(z)


class PathwayVAE(pl.LightningModule):
    """
    Pathway Variational Autoencoder with optional survival objective.

    Learns a cross-study invariant latent representation of pathway activities.
    Can optionally include a survival prediction head for DeepSurv-style objective.
    """

    def __init__(
        self,
        pathway_dim: int,
        hidden_dims: list = None,
        latent_dim: int = 32,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        beta_kl: float = 1.0,
        beta_survival: float = 0.0,
        survival_hidden_dim: int = 64,
    ):
        """
        Args:
            pathway_dim: Number of pathway features
            hidden_dims: Encoder/decoder hidden dimensions
            latent_dim: Dimension of latent space
            use_batch_norm: Use batch normalization
            dropout_rate: Dropout probability
            learning_rate: Adam learning rate
            weight_decay: L2 regularization
            beta_kl: Weight of KL divergence term
            beta_survival: Weight of survival objective (0 = disabled)
            survival_hidden_dim: Hidden dimension for survival head
        """
        super().__init__()
        self.save_hyperparameters()

        self.pathway_dim = pathway_dim
        self.latent_dim = latent_dim
        self.beta_kl = beta_kl
        self.beta_survival = beta_survival
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Encoder and decoder
        self.encoder = PathwayVAEEncoder(
            pathway_dim=pathway_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
        )

        self.decoder = PathwayVAEDecoder(
            pathway_dim=pathway_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
        )

        # Optional: Survival prediction head on latent space
        if beta_survival > 0:
            self.survival_head = nn.Sequential(
                nn.Linear(latent_dim, survival_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(survival_hidden_dim, 1),
            )
            self.cox_loss = CoxPartialLikelihood()
        else:
            self.survival_head = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, pathway_dim]

        Returns:
            x_recon: [batch_size, pathway_dim] reconstructed pathways
            z: [batch_size, latent_dim] latent codes
            mu: [batch_size, latent_dim] encoder mean
            logvar: [batch_size, latent_dim] encoder log variance
        """
        z, mu, logvar = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z, mu, logvar

    def _compute_vae_loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute VAE loss: reconstruction + KL divergence.

        Returns:
            total_loss, recon_loss, kl_loss
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')

        # KL divergence
        kl_loss = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        )

        total_loss = recon_loss + self.beta_kl * kl_loss
        return total_loss, recon_loss, kl_loss

    def _compute_survival_loss(
        self,
        z: torch.Tensor,
        event_times: torch.Tensor,
        event_indicators: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Cox partial likelihood loss on latent codes.

        Args:
            z: [batch_size, latent_dim] latent codes
            event_times: [batch_size] event/censoring times
            event_indicators: [batch_size] binary event indicator

        Returns:
            Cox loss
        """
        log_hazard = self.survival_head(z)
        cox_loss = self.cox_loss(
            log_hazard.squeeze(),
            event_times,
            event_indicators,
        )
        return cox_loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Args:
            batch: Dictionary with keys:
                - 'X': [batch_size, pathway_dim] pathways
                - 'T': [batch_size] event times (optional, for survival objective)
                - 'E': [batch_size] event indicators (optional, for survival objective)
        """
        X = batch['X']
        x_recon, z, mu, logvar = self(X)

        vae_loss, recon_loss, kl_loss = self._compute_vae_loss(X, x_recon, mu, logvar)

        total_loss = vae_loss

        # Optional survival objective
        if self.beta_survival > 0 and 'T' in batch and 'E' in batch:
            event_times = batch['T']
            event_indicators = batch['E']
            survival_loss = self._compute_survival_loss(z, event_times, event_indicators)
            total_loss = total_loss + self.beta_survival * survival_loss
            self.log('train_survival_loss', survival_loss, on_epoch=True)

        self.log('train_loss', total_loss, on_epoch=True)
        self.log('train_recon_loss', recon_loss, on_epoch=True)
        self.log('train_kl_loss', kl_loss, on_epoch=True)

        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, float]:
        """Validation with VAE metrics."""
        X = batch['X']
        x_recon, z, mu, logvar = self(X)

        vae_loss, recon_loss, kl_loss = self._compute_vae_loss(X, x_recon, mu, logvar)

        metrics = {
            'val_loss': vae_loss,
            'val_recon_loss': recon_loss,
            'val_kl_loss': kl_loss,
        }

        # Optional survival metrics
        if self.beta_survival > 0 and 'T' in batch and 'E' in batch:
            event_times = batch['T']
            event_indicators = batch['E']
            survival_loss = self._compute_survival_loss(z, event_times, event_indicators)
            metrics['val_survival_loss'] = survival_loss

            # C-index on latent codes
            log_hazard_np = self.survival_head(z).squeeze().detach().cpu().numpy()
            event_times_np = event_times.detach().cpu().numpy()
            event_indicators_np = event_indicators.detach().cpu().numpy()

            c_index = ConcordanceIndex.compute(
                log_hazard_np,
                event_times_np,
                event_indicators_np,
            )
            metrics['val_c_index'] = c_index

        for k, v in metrics.items():
            self.log(k, v, on_epoch=True)

        return metrics

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, float]:
        """Test step with detailed metrics."""
        X = batch['X']
        x_recon, z, mu, logvar = self(X)

        vae_loss, recon_loss, kl_loss = self._compute_vae_loss(X, x_recon, mu, logvar)

        metrics = {
            'test_loss': vae_loss,
            'test_recon_loss': recon_loss,
            'test_kl_loss': kl_loss,
        }

        for k, v in metrics.items():
            self.log(k, v, on_epoch=True)

        return metrics

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return latent codes and reconstructions."""
        X = batch['X']
        x_recon, z, mu, logvar = self(X)
        return z, x_recon

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
    def encode(self, X: torch.Tensor) -> torch.Tensor:
        """
        Encode pathways to latent space.

        Args:
            X: [n_samples, pathway_dim]

        Returns:
            [n_samples, latent_dim] latent codes
        """
        self.eval()
        z, _, _ = self.encoder(X)
        return z

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent codes to pathway space.

        Args:
            z: [n_samples, latent_dim]

        Returns:
            [n_samples, pathway_dim] reconstructed pathways
        """
        self.eval()
        return self.decoder(z)

    @torch.no_grad()
    def reconstruct(self, X: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct pathways through VAE.

        Args:
            X: [n_samples, pathway_dim]

        Returns:
            [n_samples, pathway_dim] reconstructed pathways
        """
        x_recon, _, _, _ = self(X)
        return x_recon
