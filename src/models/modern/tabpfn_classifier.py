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
    by dichotomizing at median survival time.
    """

    def __init__(
        self,
        risk_threshold: Optional[float] = None,
        use_scaling: bool = True,
        n_samples_error_handling: str = 'warn',
        **tabpfn_kwargs,
    ):
        """
        Args:
            risk_threshold: Threshold for high risk (default: median survival)
            use_scaling: Standardize features before fitting
            n_samples_error_handling: 'warn' or 'raise' if sample size unsuitable
            **tabpfn_kwargs: Additional arguments for TabPFNClassifier
        """
        if not TABPFN_AVAILABLE:
            raise ImportError("TabPFN must be installed. Run: pip install tabpfn")

        self.risk_threshold = risk_threshold
        self.use_scaling = use_scaling
        self.n_samples_error_handling = n_samples_error_handling
        self.tabpfn_kwargs = tabpfn_kwargs

        self.classifier = TabPFNClassifier(**tabpfn_kwargs)
        self.scaler = StandardScaler() if use_scaling else None
        self.classes_ = np.array([0, 1])

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

        # Convert to binary classification: 1 if survival < threshold, 0 otherwise
        y_binary = (y_survival < threshold).astype(int)

        # Fit scaler
        if self.scaler is not None:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X

        # Fit TabPFN
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
) -> Tuple[np.ndarray, float]:
    """
    Convert survival data to binary risk classification.

    Args:
        y_times: [n_samples] survival/censoring times
        y_events: [n_samples] binary event indicator (optional)
        threshold: Risk threshold (default: median survival)

    Returns:
        y_binary: [n_samples] binary risk labels
        threshold_used: The threshold value applied
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

    y_binary = (y_times < threshold).astype(int)
    return y_binary, threshold
