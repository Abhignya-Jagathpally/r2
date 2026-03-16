"""
TabPFN Classifier for risk classification.

Wrapper for TabPFN (Prior Function Network for tabular data) adapted
for binary risk classification from survival data.

Reference: Hollmann et al. (2023)

Note: TabPFN requires specific sample size ranges (typically 500-10000 samples).
This wrapper handles conversion from survival data to risk classification.
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
import warnings

try:
    from tabpfn import TabPFNClassifier
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    warnings.warn("TabPFN not installed. Install with: pip install tabpfn")


class TabPFNRiskClassifier(BaseEstimator, ClassifierMixin):
    """
    Risk classification using TabPFN.

    Converts survival analysis problem to binary risk classification
    by dichotomizing at median survival time. Handles censored observations
    via exclusion or inverse probability of censoring weighting (IPCW).
    """

    def __init__(
        self,
        risk_threshold: Optional[float] = None,
        use_scaling: bool = True,
        n_samples_error_handling: str = 'warn',
        handle_censoring: str = 'exclude',
        **tabpfn_kwargs,
    ):
        """
        Args:
            risk_threshold: Threshold for high risk (default: median survival)
            use_scaling: Standardize features before fitting
            n_samples_error_handling: 'warn' or 'raise' if sample size unsuitable
            handle_censoring: How to handle censored observations:
                - 'exclude': Exclude censored patients with follow-up < threshold (default)
                - 'ipcw': Use inverse probability of censoring weighting
                - 'naive': Treat censored as 0 (legacy behavior, not recommended)
            **tabpfn_kwargs: Additional arguments for TabPFNClassifier
        """
        if not TABPFN_AVAILABLE:
            raise ImportError("TabPFN must be installed. Run: pip install tabpfn")

        if handle_censoring not in ('exclude', 'ipcw', 'naive'):
            raise ValueError(f"handle_censoring must be 'exclude', 'ipcw', or 'naive', got {handle_censoring}")

        self.risk_threshold = risk_threshold
        self.use_scaling = use_scaling
        self.n_samples_error_handling = n_samples_error_handling
        self.handle_censoring = handle_censoring
        self.tabpfn_kwargs = tabpfn_kwargs

        self.classifier = TabPFNClassifier(**tabpfn_kwargs)
        self.scaler = StandardScaler() if use_scaling else None
        self.classes_ = np.array([0, 1])
        self.sample_weights_ = None

    def fit(
        self,
        X: np.ndarray,
        y_survival: np.ndarray,
        event_indicators: Optional[np.ndarray] = None,
        risk_threshold: Optional[float] = None,
    ) -> 'TabPFNRiskClassifier':
        """
        Fit TabPFN classifier from survival data.

        Args:
            X: [n_samples, n_features] pathway features
            y_survival: [n_samples] survival/censoring times
            event_indicators: [n_samples] binary event indicator (optional)
                If provided, uses only events for threshold calculation
            risk_threshold: Override the default threshold

        Returns:
            self
        """
        # Check sample size
        n_samples = X.shape[0]
        if n_samples < 100 or n_samples > 100000:
            msg = f"TabPFN works best with 100-100k samples, got {n_samples}"
            if self.n_samples_error_handling == 'raise':
                raise ValueError(msg)
            else:
                warnings.warn(msg)

        # Determine risk threshold
        if risk_threshold is not None:
            threshold = risk_threshold
        elif self.risk_threshold is not None:
            threshold = self.risk_threshold
        else:
            # Use median survival time as threshold
            if event_indicators is not None:
                event_times = y_survival[event_indicators == 1]
                if len(event_times) > 0:
                    threshold = np.median(event_times)
                else:
                    threshold = np.median(y_survival)
            else:
                threshold = np.median(y_survival)

        self.risk_threshold = threshold

        # Handle censoring
        if event_indicators is None:
            # No censoring info: treat as all events
            event_indicators = np.ones(len(y_survival), dtype=int)

        if self.handle_censoring == 'exclude':
            # Exclude censored patients with follow-up < threshold
            mask = (event_indicators == 1) | (y_survival >= threshold)
            X_fit = X[mask]
            y_survival_fit = y_survival[mask]

            # Create binary labels: 1 if event and survival < threshold, 0 otherwise
            y_binary = np.zeros(len(y_survival_fit), dtype=int)
            y_binary[event_indicators[mask] == 1] = (y_survival_fit[event_indicators[mask] == 1] < threshold).astype(int)

            # No sample weights needed
            self.sample_weights_ = None

        elif self.handle_censoring == 'ipcw':
            # Inverse probability of censoring weighting
            X_fit = X
            y_survival_fit = y_survival

            # Estimate censoring distribution using Kaplan-Meier
            # For simplicity, use empirical censoring probabilities
            n_total = len(y_survival)
            n_censored_before_threshold = np.sum((event_indicators == 0) & (y_survival < threshold))

            # Probability of being censored at time threshold
            censoring_prob = max(0.05, n_censored_before_threshold / max(1, n_total))  # Avoid division by zero

            # Compute sample weights: down-weight censored observations
            sample_weights = np.ones(n_total)
            censored_mask = (event_indicators == 0)
            if np.any(censored_mask):
                sample_weights[censored_mask] = 1.0 / (1.0 - censoring_prob + 1e-6)

            self.sample_weights_ = sample_weights / sample_weights.sum() * len(sample_weights)

            # Create binary labels: 1 if survival < threshold, 0 otherwise (including censored)
            y_binary = (y_survival_fit < threshold).astype(int)

        else:  # 'naive'
            # Legacy behavior: treat censored < threshold as 0, events < threshold as 1
            X_fit = X
            y_binary = (y_survival < threshold).astype(int)
            self.sample_weights_ = None

        # Fit scaler on training data
        if self.scaler is not None:
            X_scaled = self.scaler.fit_transform(X_fit)
        else:
            X_scaled = X_fit

        # Fit TabPFN with optional sample weights
        if self.sample_weights_ is not None:
            self.classifier.fit(X_scaled, y_binary, sample_weight=self.sample_weights_)
        else:
            self.classifier.fit(X_scaled, y_binary)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary risk class.

        Args:
            X: [n_samples, n_features]

        Returns:
            [n_samples] binary risk predictions (0 or 1)
        """
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X

        return self.classifier.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of high risk.

        Args:
            X: [n_samples, n_features]

        Returns:
            [n_samples, 2] probability estimates
        """
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X

        return self.classifier.predict_proba(X_scaled)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy score.

        Args:
            X: [n_samples, n_features]
            y: [n_samples] true labels

        Returns:
            Accuracy
        """
        return self.classifier.score(
            self.scaler.transform(X) if self.scaler else X,
            y,
        )


def convert_survival_to_risk_classification(
    y_times: np.ndarray,
    y_events: Optional[np.ndarray] = None,
    threshold: Optional[float] = None,
    handle_censoring: str = 'exclude',
) -> Tuple[np.ndarray, float, Optional[np.ndarray]]:
    """
    Convert survival data to binary risk classification.

    Args:
        y_times: [n_samples] survival/censoring times
        y_events: [n_samples] binary event indicator (optional)
        threshold: Risk threshold (default: median survival)
        handle_censoring: How to handle censored observations:
            - 'exclude': Exclude censored with follow-up < threshold
            - 'ipcw': Use inverse probability of censoring weighting
            - 'naive': Treat censored as 0

    Returns:
        y_binary: [n_samples] binary risk labels
        threshold_used: The threshold value applied
        mask_or_weights: None for naive/exclude, or sample weights for IPCW
    """
    if threshold is None:
        if y_events is not None:
            event_times = y_times[y_events == 1]
            if len(event_times) > 0:
                threshold = np.median(event_times)
            else:
                threshold = np.median(y_times)
        else:
            threshold = np.median(y_times)

    if y_events is None:
        y_events = np.ones(len(y_times), dtype=int)

    if handle_censoring == 'exclude':
        # Return mask to exclude censored patients with follow-up < threshold
        mask = (y_events == 1) | (y_times >= threshold)
        y_binary = np.zeros(len(y_times[mask]), dtype=int)
        y_binary[y_events[mask] == 1] = (y_times[mask][y_events[mask] == 1] < threshold).astype(int)
        return y_binary, threshold, mask

    elif handle_censoring == 'ipcw':
        # Compute IPCW weights
        y_binary = (y_times < threshold).astype(int)
        n_total = len(y_times)
        n_censored_before_threshold = np.sum((y_events == 0) & (y_times < threshold))
        censoring_prob = max(0.05, n_censored_before_threshold / max(1, n_total))

        weights = np.ones(n_total)
        censored_mask = (y_events == 0)
        if np.any(censored_mask):
            weights[censored_mask] = 1.0 / (1.0 - censoring_prob + 1e-6)

        weights = weights / weights.sum() * len(weights)
        return y_binary, threshold, weights

    else:  # 'naive'
        y_binary = (y_times < threshold).astype(int)
        return y_binary, threshold, None
