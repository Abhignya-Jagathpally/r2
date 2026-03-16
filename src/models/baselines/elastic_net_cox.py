"""
Elastic Net Cox proportional hazards model.

Combines L1 (Lasso) and L2 (Ridge) penalties via scikit-survival.
Flexible interpolation between pure Lasso (l1_ratio=1.0) and Ridge (l1_ratio=0.0).
"""

from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sksurv.linear_model import CoxPHFitter
from sksurv.preprocessing import Standardizer
from sksurv.util import Surv
from sklearn.model_selection import GridSearchCV, KFold

from .base_model import BaseSurvivalModel


class ElasticNetCoxModel(BaseSurvivalModel):
    """
    Elastic Net regularized Cox proportional hazards model.

    Penalty: alpha * (l1_ratio * ||beta||_1 + (1 - l1_ratio) * ||beta||_2^2)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        fit_baseline_hazard: bool = True,
        normalize: bool = True,
        tol: float = 1e-7,
        max_iter: int = 10000,
        random_state: int = 42,
    ):
        """
        Initialize Elastic Net Cox model.

        Parameters
        ----------
        alpha : float
            Overall regularization strength.
        l1_ratio : float
            L1 fraction in [0, 1]. 0.5 = equal L1 and L2.
        fit_baseline_hazard : bool
            Whether to fit baseline hazard.
        normalize : bool
            Whether to normalize features.
        tol : float
            Convergence tolerance.
        max_iter : int
            Maximum iterations.
        random_state : int
            Random state.
        """
        super().__init__(random_state=random_state)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_baseline_hazard = fit_baseline_hazard
        self.normalize = normalize
        self.tol = tol
        self.max_iter = max_iter

        # Model components
        self.model_ = None
        self.standardizer_ = None
        self.feature_importance_ = None
        self.coef_history_ = None

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y_time: Union[np.ndarray, pd.Series],
        y_event: Union[np.ndarray, pd.Series],
        **kwargs,
    ) -> "ElasticNetCoxModel":
        """
        Fit Elastic Net Cox model.

        Parameters
        ----------
        X : array-like or DataFrame, shape (n_samples, n_features)
            Feature matrix.
        y_time : array-like, shape (n_samples,)
            Survival time.
        y_event : array-like, shape (n_samples,)
            Event indicator.
        **kwargs
            Additional arguments (unused).

        Returns
        -------
        self
        """
        X_array, y_time_array, y_event_array = self._validate_input(X, y_time, y_event)

        # Standardize features
        self.standardizer_ = Standardizer()
        X_std = self.standardizer_.fit_transform(X_array)

        # Create structured array for sksurv
        y_struct = Surv.from_arrays(y_event_array.astype(bool), y_time_array)

        # Fit Cox model with Elastic Net penalty
        self.model_ = CoxPHFitter(
            penalizer=self.alpha,
            l1_ratio=self.l1_ratio,
            fit_baseline_hazard=self.fit_baseline_hazard,
            normalize=self.normalize,
            tol=self.tol,
            max_iter=self.max_iter,
        )

        df = pd.DataFrame(X_std, columns=self.feature_names_)
        self.model_.fit(df, y_struct)

        # Extract coefficients
        self.feature_importance_ = self.model_.params_.to_dict()

        return self

    def predict_risk(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict risk scores (partial hazard).

        Parameters
        ----------
        X : array-like or DataFrame, shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        risk_scores : ndarray, shape (n_samples,)
            Risk scores.
        """
        X_array, _, _ = self._validate_input(X)
        X_std = self.standardizer_.transform(X_array)

        df = pd.DataFrame(X_std, columns=self.feature_names_)
        partial_hazard = self.model_.predict_partial_hazard(df).values

        return partial_hazard

    def predict_survival_function(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        times: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Predict survival probability.

        Parameters
        ----------
        X : array-like or DataFrame, shape (n_samples, n_features)
            Feature matrix.
        times : ndarray, optional
            Time points for evaluation.

        Returns
        -------
        survival_prob : DataFrame
            Survival probabilities.
        """
        X_array, _, _ = self._validate_input(X)
        X_std = self.standardizer_.transform(X_array)

        df = pd.DataFrame(X_std, columns=self.feature_names_)
        surv_funcs = self.model_.predict_survival_function(df)

        if times is not None:
            result = pd.DataFrame(index=df.index, columns=times)
            for idx in df.index:
                s = surv_funcs.iloc[:, idx]
                for t in times:
                    idx_closest = (s.index - t).abs().argmin()
                    result.loc[idx, t] = s.iloc[idx_closest]
            return result
        else:
            return surv_funcs.T

    def get_selected_features(self, threshold: float = 0.01) -> list:
        """
        Get features with non-zero coefficients.

        Parameters
        ----------
        threshold : float
            Absolute coefficient threshold.

        Returns
        -------
        selected : list
            Selected feature names.
        """
        selected = [
            fname for fname, coef in self.feature_importance_.items()
            if np.abs(coef) >= threshold
        ]
        return selected

    def grid_search_cv(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y_time: Union[np.ndarray, pd.Series],
        y_event: Union[np.ndarray, pd.Series],
        alpha_grid: Optional[np.ndarray] = None,
        l1_ratio_grid: Optional[np.ndarray] = None,
        cv: int = 5,
    ) -> Tuple[float, float, Dict]:
        """
        Grid search for optimal alpha and l1_ratio.

        Parameters
        ----------
        X : array-like or DataFrame
            Feature matrix.
        y_time : array-like
            Survival time.
        y_event : array-like
            Event indicator.
        alpha_grid : ndarray, optional
            Alpha values to search.
        l1_ratio_grid : ndarray, optional
            L1 ratio values to search.
        cv : int
            Cross-validation folds.

        Returns
        -------
        best_alpha : float
            Optimal alpha.
        best_l1_ratio : float
            Optimal l1_ratio.
        results : dict
            Grid search results with C-indices.
        """
        from sksurv.metrics import concordance_index_censored

        X_array, y_time_array, y_event_array = self._validate_input(X, y_time, y_event)

        if alpha_grid is None:
            alpha_grid = np.logspace(-4, 2, 15)
        if l1_ratio_grid is None:
            l1_ratio_grid = np.linspace(0.0, 1.0, 6)

        best_score = -np.inf
        best_alpha = alpha_grid[0]
        best_l1_ratio = l1_ratio_grid[0]
        results = {
            "alpha": [],
            "l1_ratio": [],
            "mean_c_index": [],
            "std_c_index": [],
        }

        splitter = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)

        for alpha in alpha_grid:
            for l1_ratio in l1_ratio_grid:
                fold_scores = []

                for train_idx, test_idx in splitter.split(X_array):
                    X_train, X_test = X_array[train_idx], X_array[test_idx]
                    y_time_train = y_time_array[train_idx]
                    y_event_train = y_event_array[train_idx]
                    y_time_test = y_time_array[test_idx]
                    y_event_test = y_event_array[test_idx]

                    # Fit model
                    model = ElasticNetCoxModel(
                        alpha=alpha,
                        l1_ratio=l1_ratio,
                        random_state=self.random_state,
                    )
                    model.fit(X_train, y_time_train, y_event_train)

                    # Evaluate
                    risk_scores = model.predict_risk(X_test)
                    c_index = concordance_index_censored(
                        y_event_test.astype(bool),
                        y_time_test,
                        risk_scores,
                    )[0]
                    fold_scores.append(c_index)

                mean_score = np.mean(fold_scores)
                std_score = np.std(fold_scores)

                results["alpha"].append(alpha)
                results["l1_ratio"].append(l1_ratio)
                results["mean_c_index"].append(mean_score)
                results["std_c_index"].append(std_score)

                if mean_score > best_score:
                    best_score = mean_score
                    best_alpha = alpha
                    best_l1_ratio = l1_ratio

        # Refit with best parameters
        self.alpha = best_alpha
        self.l1_ratio = best_l1_ratio
        self.fit(X, y_time, y_event)

        return best_alpha, best_l1_ratio, results
