"""
Training utilities for survival analysis and modern ML models.

Includes loss functions, learning rate schedulers, and training helpers
for Cox proportional hazards and other survival-based models.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR


class CoxPartialLikelihood(nn.Module):
    """
    Cox partial likelihood loss for survival analysis.

    Expects:
    - log_hazard: [batch_size] - log hazard ratios from model
    - event_times: [batch_size] - observed times
    - event_indicators: [batch_size] - binary event indicator (1=event, 0=censored)
    """

    def __init__(self, eps: float = 1e-8):
        """
        Args:
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps

    def forward(
        self,
        log_hazard: torch.Tensor,
        event_times: torch.Tensor,
        event_indicators: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute negative log partial likelihood (Breslow, numerically stable).

        Args:
            log_hazard: [batch_size] log hazard ratios
            event_times: [batch_size] event/censoring times
            event_indicators: [batch_size] binary event indicator

        Returns:
            Scalar loss (negative log partial likelihood)
        """
        # Sort by time DESCENDING for cumsum-based risk set
        sorted_idx = torch.argsort(event_times, descending=True)
        log_hazard = log_hazard[sorted_idx]
        event_indicators = event_indicators[sorted_idx]

        # Log-sum-exp trick for numerical stability
        max_log_h = log_hazard.max()
        risk_set_log = torch.log(
            torch.cumsum(torch.exp(log_hazard - max_log_h), dim=0)
        ) + max_log_h

        # Log partial likelihood
        log_likelihood = log_hazard - risk_set_log

        # Weight by event indicator (only count actual events)
        weighted_ll = log_likelihood * event_indicators

        n_events = event_indicators.sum()
        if n_events == 0:
            return torch.tensor(0.0, requires_grad=True, device=log_hazard.device)

        return -weighted_ll.sum() / n_events


class RankingLoss(nn.Module):
    """
    Ranking loss for survival analysis (C-index compatible).

    Encourages concordant pairs to have correct ordering.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        log_hazard: torch.Tensor,
        event_times: torch.Tensor,
        event_indicators: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pairwise ranking loss.

        Args:
            log_hazard: [batch_size] log hazard ratios
            event_times: [batch_size] event/censoring times
            event_indicators: [batch_size] binary event indicator

        Returns:
            Scalar loss
        """
        batch_size = log_hazard.shape[0]

        # Expand to all pairs
        # Shape: [batch_size, batch_size]
        hazard_pairs = log_hazard.unsqueeze(1) - log_hazard.unsqueeze(0)
        time_pairs = event_times.unsqueeze(1) - event_times.unsqueeze(0)

        # Concordance: i happens before j AND i has event
        # i.e., time[i] < time[j] AND event[i] == 1
        event_i = event_indicators.unsqueeze(1)  # [batch_size, 1]

        # Pairs where i is an event and happens before j
        concordant = (time_pairs < 0) * event_i

        # Ranking loss: log(1 + exp(-hazard_diff)) for concordant pairs
        # When hazard[i] > hazard[j], this is small
        ranking_loss = F.softplus(-hazard_pairs)

        weighted_loss = (ranking_loss * concordant).sum()
        n_pairs = concordant.sum() + self.eps

        return weighted_loss / n_pairs


class ConcordanceIndex:
    """
    Compute concordance index (C-index) for survival models.

    Measures the fraction of concordant pairs in the predictions.
    """

    @staticmethod
    def compute(
        log_hazard: np.ndarray,
        event_times: np.ndarray,
        event_indicators: np.ndarray,
    ) -> float:
        """
        Compute C-index.

        Args:
            log_hazard: [n_samples] log hazard ratios (or raw predictions)
            event_times: [n_samples] event/censoring times
            event_indicators: [n_samples] binary event indicator

        Returns:
            C-index in [0, 1]
        """
        n = len(log_hazard)
        concordant = 0
        total = 0

        for i in range(n):
            for j in range(i + 1, n):
                # Check if pair is comparable
                if event_times[i] < event_times[j] and event_indicators[i] == 1:
                    total += 1
                    # Check concordance: higher hazard for earlier event
                    if log_hazard[i] > log_hazard[j]:
                        concordant += 1
                elif event_times[j] < event_times[i] and event_indicators[j] == 1:
                    total += 1
                    if log_hazard[j] > log_hazard[i]:
                        concordant += 1

        if total == 0:
            return 0.5  # Random classifier

        return concordant / total


def create_cosine_scheduler(
    optimizer,
    num_epochs: int,
    warmup_epochs: int = 0,
    min_lr: float = 1e-6,
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Create cosine annealing scheduler with optional warmup.

    Args:
        optimizer: PyTorch optimizer
        num_epochs: Total number of epochs
        warmup_epochs: Number of warmup epochs
        min_lr: Minimum learning rate

    Returns:
        LR scheduler
    """
    if warmup_epochs > 0:
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=min_lr / optimizer.defaults['lr'],
            total_iters=warmup_epochs,
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_epochs - warmup_epochs,
            eta_min=min_lr,
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
    else:
        return CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=min_lr,
        )


class EarlyStopping:
    """
    Early stopping based on validation metric.
    """

    def __init__(
        self,
        metric_name: str = "val_loss",
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
        restore_best: bool = True,
    ):
        """
        Args:
            metric_name: Name of metric to monitor
            patience: Number of epochs with no improvement before stopping
            min_delta: Minimum change to qualify as improvement
            mode: "min" for loss, "max" for accuracy/C-index
            restore_best: Whether to restore best model weights
        """
        self.metric_name = metric_name
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best

        self.best_value = None
        self.best_epoch = 0
        self.counter = 0
        self.best_state = None

        if mode == "min":
            self.is_better = lambda new, best: new < (best - min_delta)
            self.best_value = float('inf')
        elif mode == "max":
            self.is_better = lambda new, best: new > (best + min_delta)
            self.best_value = float('-inf')
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def __call__(self, value: float, model: nn.Module, epoch: int) -> bool:
        """
        Check if training should stop.

        Args:
            value: Current metric value
            model: PyTorch model (for saving best state)
            epoch: Current epoch number

        Returns:
            True if training should stop, False otherwise
        """
        if self.is_better(value, self.best_value):
            self.best_value = value
            self.best_epoch = epoch
            self.counter = 0
            if self.restore_best:
                self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1

        return self.counter >= self.patience

    def restore_best_weights(self, model: nn.Module) -> None:
        """Restore best model weights."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class GradientClipper:
    """
    Clip gradients by norm or value.
    """

    @staticmethod
    def clip_by_norm(
        parameters: list,
        max_norm: float,
    ) -> float:
        """
        Clip gradients by global norm.

        Args:
            parameters: Model parameters
            max_norm: Maximum allowed norm

        Returns:
            Total norm before clipping
        """
        total_norm = 0.0
        for p in parameters:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in parameters:
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef)

        return total_norm


def compute_survival_metrics(
    log_hazard: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray,
) -> Dict[str, float]:
    """
    Compute multiple survival analysis metrics.

    Args:
        log_hazard: [n_samples] log hazard predictions
        event_times: [n_samples] event/censoring times
        event_indicators: [n_samples] binary event indicator

    Returns:
        Dictionary with metrics
    """
    c_index = ConcordanceIndex.compute(log_hazard, event_times, event_indicators)

    return {
        "c_index": c_index,
        "concordance": c_index,
    }
