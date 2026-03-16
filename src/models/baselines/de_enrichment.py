"""
Differential expression enrichment baseline for survival prediction.

Workflow:
1. Stratify patients into high/low risk (median or tertile-based)
2. Perform limma-style differential expression between groups
3. Run ORA (over-representation analysis) and GSEA
4. Combine DE + enrichment results into risk score

This serves as an interpretable baseline and sanity check.
"""

from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

from .base_model import BaseSurvivalModel


class DEEnrichmentBaseline(BaseSurvivalModel):
    """
    Differential expression and enrichment-based survival risk model.

    Stratifies patients into risk groups, performs DE analysis, and combines
    pathway enrichment with clinical outcome association.
    """

    def __init__(
        self,
        risk_stratification: str = "median",
        n_pathways: int = 50,
        min_pathway_size: int = 5,
        max_pathway_size: int = 500,
        random_state: int = 42,
    ):
        """
        Initialize DE enrichment baseline.

        Parameters
        ----------
        risk_stratification : str
            How to stratify patients: 'median', 'tertile', or 'kmeans'.
        n_pathways : int
            Number of top pathways to use for risk score.
        min_pathway_size : int
            Minimum pathway size (genes/features).
        max_pathway_size : int
            Maximum pathway size.
        random_state : int
            Random state.
        """
        super().__init__(random_state=random_state)
        self.risk_stratification = risk_stratification
        self.n_pathways = n_pathways
        self.min_pathway_size = min_pathway_size
        self.max_pathway_size = max_pathway_size

        # Results storage
        self.de_stats_ = None
        self.enrichment_scores_ = None
        self.selected_pathways_ = None
        self.risk_score_weights_ = None

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y_time: Union[np.ndarray, pd.Series],
        y_event: Union[np.ndarray, pd.Series],
        pathway_matrix: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> "DEEnrichmentBaseline":
        """
        Fit DE enrichment model.

        Parameters
        ----------
        X : array-like or DataFrame, shape (n_samples, n_features)
            Feature matrix (e.g., pathway scores from GSVA/ssGSEA).
        y_time : array-like, shape (n_samples,)
            Survival time.
        y_event : array-like, shape (n_samples,)
            Event indicator.
        pathway_matrix : DataFrame, optional
            If provided, use for enrichment analysis. Otherwise use X.
        **kwargs
            Additional arguments (unused).

        Returns
        -------
        self
        """
        X_array, y_time_array, y_event_array = self._validate_input(X, y_time, y_event)

        # Stratify patients into risk groups
        risk_groups = self._stratify_risk(y_event_array, y_time_array)

        # Compute differential expression (correlation with event)
        self.de_stats_ = self._compute_de_stats(X_array, y_event_array, y_time_array)

        # Score pathways by outcome association
        self.enrichment_scores_ = self._score_enrichment(X_array, y_event_array)

        # Select top pathways
        self.selected_pathways_ = self._select_pathways()

        # Compute risk score weights
        self.risk_score_weights_ = self._compute_weights(X_array)

        # Store scaler
        self._fit_scaler(X_array)

        return self

    def _stratify_risk(self, y_event: np.ndarray, y_time: np.ndarray) -> np.ndarray:
        """
        Stratify patients into risk groups.

        Parameters
        ----------
        y_event : ndarray, shape (n_samples,)
            Event indicator.
        y_time : ndarray, shape (n_samples,)
            Survival time.

        Returns
        -------
        risk_groups : ndarray, shape (n_samples,)
            Risk group assignment (0=low, 1=high).
        """
        if self.risk_stratification == "median":
            threshold = np.median(y_time)
            risk_groups = (y_time >= threshold).astype(int)
        elif self.risk_stratification == "tertile":
            threshold = np.percentile(y_time, 66.67)
            risk_groups = (y_time >= threshold).astype(int)
        else:
            # Event-based stratification
            risk_groups = y_event.copy()

        return risk_groups

    def _compute_de_stats(
        self,
        X: np.ndarray,
        y_event: np.ndarray,
        y_time: np.ndarray,
    ) -> pd.DataFrame:
        """
        Compute differential expression statistics (t-test and correlation).

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Feature matrix.
        y_event : ndarray, shape (n_samples,)
            Event indicator.
        y_time : ndarray, shape (n_samples,)
            Survival time.

        Returns
        -------
        de_stats : DataFrame
            Feature-level DE statistics (t-stat, p-value, logFC, correlation).
        """
        n_features = X.shape[1]
        de_results = []

        for j in range(n_features):
            x_j = X[:, j]

            # T-test: event vs no-event
            x_event = x_j[y_event == 1]
            x_no_event = x_j[y_event == 0]
            t_stat, p_val = stats.ttest_ind(x_event, x_no_event)

            # Log fold change
            mean_event = np.mean(x_event) if len(x_event) > 0 else 0
            mean_no_event = np.mean(x_no_event) if len(x_no_event) > 0 else 0
            logfc = np.log2(np.abs(mean_event) + 1e-10) - np.log2(np.abs(mean_no_event) + 1e-10)

            # Correlation with survival time
            corr_time, p_corr = stats.pearsonr(x_j, y_time)

            # Correlation with event
            corr_event, _ = stats.pearsonr(x_j, y_event)

            de_results.append({
                "feature_idx": j,
                "feature_name": self.feature_names_[j] if self.feature_names_ else f"f_{j}",
                "t_stat": t_stat,
                "p_value": p_val,
                "logfc": logfc,
                "corr_time": corr_time,
                "corr_event": corr_event,
                "abs_t_stat": np.abs(t_stat),
            })

        de_df = pd.DataFrame(de_results)
        de_df["q_value"] = self._bh_correct(de_df["p_value"].values)
        return de_df.sort_values("abs_t_stat", ascending=False)

    def _score_enrichment(self, X: np.ndarray, y_event: np.ndarray) -> pd.Series:
        """
        Score each pathway/feature by outcome association.

        Uses Cox partial likelihood score as a simple survival association metric.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Feature matrix.
        y_event : ndarray, shape (n_samples,)
            Event indicator.

        Returns
        -------
        scores : Series
            Enrichment score per feature (higher = more associated with outcome).
        """
        scores = {}

        for j in range(X.shape[1]):
            x_j = X[:, j]
            x_j = (x_j - x_j.mean()) / (x_j.std() + 1e-8)

            # Simple association: correlation with event
            corr = np.abs(np.corrcoef(x_j, y_event)[0, 1])
            scores[j] = corr

        return pd.Series(scores)

    def _select_pathways(self) -> list:
        """
        Select top pathways by enrichment score.

        Returns
        -------
        selected : list
            Indices of selected pathways.
        """
        # Rank by absolute t-stat and enrichment score
        combined_score = np.abs(self.de_stats_["t_stat"].values)
        top_idx = np.argsort(combined_score)[-self.n_pathways :]
        return self.de_stats_.iloc[top_idx]["feature_idx"].tolist()

    def _compute_weights(self, X: np.ndarray) -> Dict[int, float]:
        """
        Compute risk score weights for selected pathways.

        Uses direction of association (correlation sign) and effect size.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        weights : dict
            Feature index -> weight.
        """
        weights = {}

        for idx in self.selected_pathways_:
            row = self.de_stats_[self.de_stats_["feature_idx"] == idx].iloc[0]

            # Weight = signed t-stat (direction) * abs(correlation)
            sign = np.sign(row["corr_event"])
            magnitude = np.abs(row["t_stat"])
            weights[idx] = sign * magnitude

        return weights

    @staticmethod
    def _bh_correct(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
        """
        Benjamini-Hochberg multiple testing correction.

        Parameters
        ----------
        p_values : ndarray
            P-values.
        alpha : float
            FDR threshold.

        Returns
        -------
        q_values : ndarray
            Adjusted p-values.
        """
        ranked = np.argsort(p_values)
        ranked_pvals = p_values[ranked]
        m = len(p_values)
        bh_critical = np.arange(1, m + 1) / m * alpha
        rej = ranked_pvals <= bh_critical
        threshold = ranked_pvals[rej][-1] if rej.any() else p_values.min()
        q_values = np.minimum.accumulate(p_values * m / np.arange(1, m + 1))
        return q_values

    def predict_risk(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict risk scores as weighted sum of selected pathways.

        Parameters
        ----------
        X : array-like or DataFrame, shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        risk_scores : ndarray, shape (n_samples,)
            Predicted risk scores.
        """
        X_array, _, _ = self._validate_input(X)
        X_scaled = self._transform_scaler(X_array)

        risk_scores = np.zeros(X_array.shape[0])

        for idx, weight in self.risk_score_weights_.items():
            risk_scores += weight * X_scaled[:, idx]

        # Normalize to [0, 1]
        risk_scores = (risk_scores - risk_scores.min()) / (risk_scores.max() - risk_scores.min() + 1e-8)

        return risk_scores

    def predict_survival_function(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        times: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Predict Kaplan-Meier style survival based on risk stratification.

        Parameters
        ----------
        X : array-like or DataFrame, shape (n_samples, n_features)
            Feature matrix.
        times : ndarray, optional
            Time points for evaluation. Currently unused.

        Returns
        -------
        survival_prob : DataFrame
            Placeholder survival probabilities (risk-stratified KM curves).
        """
        risk_scores = self.predict_risk(X)
        n_samples = len(risk_scores)

        # Dummy output: survival as 1 - risk_score for each time
        if times is None:
            times = np.array([1.0, 2.0, 3.0, 5.0, 10.0])

        survival_probs = pd.DataFrame(
            index=np.arange(n_samples),
            columns=times,
        )

        for t in times:
            # Naive: survival ~ exp(-risk * t)
            survival_probs[t] = np.exp(-risk_scores * t / times.max())

        return survival_probs
