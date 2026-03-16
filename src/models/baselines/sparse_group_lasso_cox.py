"""
Sparse Group Lasso Cox proportional hazards model.

Objective: min_β -ℓ_Cox(β) + λ₁‖β‖₁ + λ₂ Σ_g w_g ‖β_g‖₂

Key features:
- Pathway/gene group awareness via group regularization
- Proximal gradient descent solver
- Group weights w_g = sqrt(|group_g|)
- Nested CV for (λ₁, λ₂) selection
- Pathway selection and biological interpretation
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler as Standardizer
from sksurv.util import Surv
from sklearn.model_selection import KFold

from .base_model import BaseSurvivalModel


class SparseGroupLassoCoxModel(BaseSurvivalModel):
    """
    Sparse Group Lasso regularized Cox model.

    Combines L1 sparsity within groups and group-level L2 for pathway selection.
    """

    def __init__(
        self,
        lambda1: float = 0.1,
        lambda2: float = 0.1,
        groups: Optional[Dict[str, List[int]]] = None,
        group_weights: Optional[Dict[str, float]] = None,
        fit_baseline: bool = True,
        normalize: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-5,
        random_state: int = 42,
    ):
        """
        Initialize Sparse Group Lasso Cox model.

        Parameters
        ----------
        lambda1 : float
            L1 (Lasso) penalty weight.
        lambda2 : float
            L2 (Group Lasso) penalty weight.
        groups : dict, optional
            Mapping group name -> list of feature indices.
            If None, each feature is its own group.
        group_weights : dict, optional
            Weight per group. If None, use sqrt(group_size).
        fit_baseline : bool
            Whether to fit baseline hazard.
        normalize : bool
            Whether to normalize features.
        max_iter : int
            Maximum iterations for proximal gradient descent.
        tol : float
            Convergence tolerance.
        random_state : int
            Random state.
        """
        super().__init__(random_state=random_state)
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.groups = groups
        self.group_weights = group_weights
        self.fit_baseline = fit_baseline
        self.normalize = normalize
        self.max_iter = max_iter
        self.tol = tol

        # Model state
        self.coef_ = None
        self.standardizer_ = None
        self.baseline_hazard_ = None
        self.feature_importance_ = None
        self.selected_groups_ = None
        self.group_mapping_ = None

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y_time: Union[np.ndarray, pd.Series],
        y_event: Union[np.ndarray, pd.Series],
        groups: Optional[Dict[str, List[int]]] = None,
        **kwargs,
    ) -> "SparseGroupLassoCoxModel":
        """
        Fit Sparse Group Lasso Cox model via proximal gradient descent.

        Parameters
        ----------
        X : array-like or DataFrame, shape (n_samples, n_features)
            Feature matrix.
        y_time : array-like, shape (n_samples,)
            Survival time.
        y_event : array-like, shape (n_samples,)
            Event indicator.
        groups : dict, optional
            Override self.groups for this fit.
        **kwargs
            Additional arguments (unused).

        Returns
        -------
        self
        """
        X_array, y_time_array, y_event_array = self._validate_input(X, y_time, y_event)

        # Use provided groups or default to single-feature groups
        if groups is not None:
            self.groups = groups
        elif self.groups is None:
            self.groups = {f"f_{i}": [i] for i in range(X_array.shape[1])}

        # Set up group weights
        self._setup_group_weights(X_array.shape[1])

        # Standardize features
        self.standardizer_ = Standardizer()
        X_std = self.standardizer_.fit_transform(X_array)

        # Create structured array for Cox likelihood
        y_struct = Surv.from_arrays(y_event_array.astype(bool), y_time_array)

        # Initialize coefficients
        beta_init = np.zeros(X_array.shape[1])

        # Proximal gradient descent
        self.coef_ = self._proximal_gradient_descent(
            X_std, y_struct, beta_init
        )

        # Extract baseline hazard and feature importance
        self._compute_baseline_hazard(X_std, y_struct)
        self.feature_importance_ = {
            self.feature_names_[i]: self.coef_[i]
            for i in range(len(self.feature_names_))
        }

        # Identify selected groups
        self._select_groups()

        return self

    def _setup_group_weights(self, n_features: int) -> None:
        """
        Set up group weights (default: sqrt(group_size)).

        Parameters
        ----------
        n_features : int
            Number of features.
        """
        if self.group_weights is None:
            self.group_weights = {}
            for group_name, group_indices in self.groups.items():
                size = len(group_indices)
                self.group_weights[group_name] = np.sqrt(size)

    def _cox_log_likelihood(
        self,
        beta: np.ndarray,
        X: np.ndarray,
        y_struct: Surv,
    ) -> float:
        """
        Cox partial log-likelihood.

        Parameters
        ----------
        beta : ndarray, shape (n_features,)
            Coefficients.
        X : ndarray, shape (n_samples, n_features)
            Feature matrix.
        y_struct : Surv
            Survival structured array.

        Returns
        -------
        ll : float
            Negative log-likelihood (for minimization).
        """
        n_samples = X.shape[0]
        eta = X @ beta  # Linear predictor
        eta = np.clip(eta, -500, 500)  # Prevent exp overflow
        exp_eta = np.exp(eta)

        ll = 0.0

        for i in range(n_samples):
            if y_struct['event'][i]:
                # Risk set at time y_struct['time'][i]
                at_risk = y_struct['time'] >= y_struct['time'][i]
                risk_set_exp_eta = exp_eta[at_risk].sum()

                if risk_set_exp_eta > 0:
                    ll += eta[i] - np.log(risk_set_exp_eta)
                else:
                    ll += eta[i]

        return -ll / n_samples  # Negative (for minimization), normalized

    def _cox_gradient(
        self,
        beta: np.ndarray,
        X: np.ndarray,
        y_struct: Surv,
    ) -> np.ndarray:
        """
        Gradient of Cox partial log-likelihood.

        Parameters
        ----------
        beta : ndarray
            Coefficients.
        X : ndarray
            Feature matrix.
        y_struct : Surv
            Survival structured array.

        Returns
        -------
        grad : ndarray
            Gradient vector.
        """
        n_samples, n_features = X.shape
        eta = X @ beta
        # Clip eta to prevent overflow in exp()
        eta = np.clip(eta, -500, 500)
        exp_eta = np.exp(eta)
        grad = np.zeros(n_features)

        for i in range(n_samples):
            if y_struct['event'][i]:
                at_risk = y_struct['time'] >= y_struct['time'][i]
                risk_set_exp_eta = exp_eta[at_risk].sum()

                if risk_set_exp_eta > 0:
                    weighted_x = (X[at_risk].T * exp_eta[at_risk]).sum(axis=1)
                    grad -= (X[i] - weighted_x / risk_set_exp_eta)

        return grad / n_samples

    def _proximal_l1_operator(
        self,
        beta: np.ndarray,
        lambda1: float,
        step_size: float,
    ) -> np.ndarray:
        """
        Proximal operator for L1 (soft-thresholding).

        Parameters
        ----------
        beta : ndarray
            Current coefficients.
        lambda1 : float
            L1 penalty weight.
        step_size : float
            Step size (learning rate).

        Returns
        -------
        beta_prox : ndarray
            After soft-thresholding.
        """
        threshold = lambda1 * step_size
        return np.sign(beta) * np.maximum(np.abs(beta) - threshold, 0)

    def _proximal_group_lasso_operator(
        self,
        beta: np.ndarray,
        lambda2: float,
        step_size: float,
    ) -> np.ndarray:
        """
        Proximal operator for grouped L2 (block soft-thresholding).

        Parameters
        ----------
        beta : ndarray
            Current coefficients.
        lambda2 : float
            Group L2 penalty weight.
        step_size : float
            Step size.

        Returns
        -------
        beta_prox : ndarray
            After group soft-thresholding.
        """
        beta_prox = beta.copy()

        for group_name, group_indices in self.groups.items():
            group_beta = beta[group_indices]
            w_g = self.group_weights[group_name]
            threshold = lambda2 * w_g * step_size

            norm_g = np.linalg.norm(group_beta)
            if norm_g > threshold:
                beta_prox[group_indices] = group_beta * (1 - threshold / norm_g)
            else:
                beta_prox[group_indices] = 0

        return beta_prox

    def _proximal_gradient_descent(
        self,
        X: np.ndarray,
        y_struct: Surv,
        beta_init: np.ndarray,
        step_size: float = 0.01,
    ) -> np.ndarray:
        """
        Proximal gradient descent for sparse group lasso Cox.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Feature matrix.
        y_struct : Surv
            Survival structured array.
        beta_init : ndarray
            Initial coefficients.
        step_size : float
            Initial step size (will be adapted).

        Returns
        -------
        beta : ndarray
            Optimized coefficients.
        """
        beta = beta_init.copy()
        prev_loss = float('inf')

        for iteration in range(self.max_iter):
            # Gradient step
            grad = self._cox_gradient(beta, X, y_struct)
            beta_gd = beta - step_size * grad

            # Proximal steps
            beta_prox = self._proximal_l1_operator(beta_gd, self.lambda1, step_size)
            beta = self._proximal_group_lasso_operator(beta_prox, self.lambda2, step_size)

            # Check convergence
            curr_loss = self._cox_log_likelihood(beta, X, y_struct)
            if np.abs(curr_loss - prev_loss) < self.tol:
                break

            prev_loss = curr_loss

        return beta

    def _compute_baseline_hazard(self, X: np.ndarray, y_struct: Surv) -> None:
        """
        Estimate baseline hazard at event times (Breslow estimator).

        Parameters
        ----------
        X : ndarray
            Feature matrix.
        y_struct : Surv
            Survival structured array.
        """
        if not self.fit_baseline:
            return

        eta = X @ self.coef_
        exp_eta = np.exp(eta)
        event_times = y_struct['time'][y_struct['event']]
        baseline_hazard = {}

        for t in np.unique(event_times):
            at_event = (y_struct['time'] == t) & y_struct['event']
            at_risk = y_struct['time'] >= t

            d_t = at_event.sum()
            s_t = exp_eta[at_risk].sum()

            if s_t > 0:
                baseline_hazard[t] = d_t / s_t

        self.baseline_hazard_ = baseline_hazard

    def _select_groups(self) -> None:
        """
        Identify selected groups based on non-zero coefficients.
        """
        self.selected_groups_ = []

        for group_name, group_indices in self.groups.items():
            group_norm = np.linalg.norm(self.coef_[group_indices])
            if group_norm > 1e-8:
                self.selected_groups_.append({
                    "name": group_name,
                    "indices": group_indices,
                    "norm": group_norm,
                    "n_features": len(group_indices),
                })

        # Sort by norm
        self.selected_groups_ = sorted(
            self.selected_groups_,
            key=lambda x: x["norm"],
            reverse=True,
        )

    def predict_risk(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict risk scores (linear predictor eta = X @ beta).

        Parameters
        ----------
        X : array-like or DataFrame, shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        risk_scores : ndarray, shape (n_samples,)
            Risk scores (higher = higher risk).
        """
        X_array, _, _ = self._validate_input(X)
        X_std = self.standardizer_.transform(X_array)

        eta = X_std @ self.coef_
        return np.exp(eta)  # Partial hazard

    def predict_survival_function(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        times: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Predict survival probability S(t) based on baseline hazard.

        Parameters
        ----------
        X : array-like or DataFrame, shape (n_samples, n_features)
            Feature matrix.
        times : ndarray, optional
            Time points. If None, use observed event times.

        Returns
        -------
        survival_prob : DataFrame
            Survival probabilities. Shape: (n_samples, n_times).
        """
        X_array, _, _ = self._validate_input(X)
        X_std = self.standardizer_.transform(X_array)

        partial_hazards = np.exp(X_std @ self.coef_)
        n_samples = X_array.shape[0]

        if times is None:
            times = np.array([1.0, 2.0, 3.0, 5.0, 10.0])

        # Naive Kaplan-Meier style (without observed data)
        baseline_surv = np.exp(-np.array(list(self.baseline_hazard_.values())).cumsum())
        baseline_times = np.array(sorted(self.baseline_hazard_.keys()))

        result = pd.DataFrame(index=np.arange(n_samples), columns=times)

        for i in range(n_samples):
            for t in times:
                # Interpolate baseline
                idx_closest = (baseline_times - t).abs().argmin()
                if idx_closest < len(baseline_surv):
                    base_surv = baseline_surv[idx_closest]
                else:
                    base_surv = baseline_surv[-1]

                result.loc[i, t] = base_surv ** partial_hazards[i]

        return result

    def get_selected_pathways(self) -> pd.DataFrame:
        """
        Get selected pathways with statistics.

        Returns
        -------
        pathways : DataFrame
            Selected pathways and their properties.
        """
        if self.selected_groups_ is None:
            return pd.DataFrame()

        data = []
        for group in self.selected_groups_:
            feature_names = [self.feature_names_[i] for i in group["indices"]]
            coefs = [self.coef_[i] for i in group["indices"]]

            data.append({
                "pathway": group["name"],
                "n_features": group["n_features"],
                "group_norm": group["norm"],
                "features": ", ".join(feature_names),
                "mean_coef": np.mean(np.abs(coefs)),
            })

        return pd.DataFrame(data)

    def nested_cv_lambda_selection(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y_time: Union[np.ndarray, pd.Series],
        y_event: Union[np.ndarray, pd.Series],
        lambda1_grid: Optional[np.ndarray] = None,
        lambda2_grid: Optional[np.ndarray] = None,
        cv_inner: int = 3,
        cv_outer: int = 5,
    ) -> Tuple[float, float, float, np.ndarray]:
        """
        Nested cross-validation for (λ₁, λ₂) selection.

        Parameters
        ----------
        X : array-like or DataFrame
            Feature matrix.
        y_time : array-like
            Survival time.
        y_event : array-like
            Event indicator.
        lambda1_grid : ndarray, optional
            L1 penalty values to search.
        lambda2_grid : ndarray, optional
            L2 penalty values to search.
        cv_inner : int
            Inner CV folds.
        cv_outer : int
            Outer CV folds.

        Returns
        -------
        best_lambda1 : float
            Optimal λ₁.
        best_lambda2 : float
            Optimal λ₂.
        cv_score : float
            Outer CV C-index.
        cv_scores : ndarray
            Per-fold C-index scores.
        """
        from sksurv.metrics import concordance_index_censored

        X_array, y_time_array, y_event_array = self._validate_input(X, y_time, y_event)

        if lambda1_grid is None:
            lambda1_grid = np.logspace(-4, 0, 10)
        if lambda2_grid is None:
            lambda2_grid = np.logspace(-4, 0, 10)

        outer_cv = KFold(n_splits=cv_outer, shuffle=True, random_state=self.random_state)
        cv_scores_outer = []
        best_lambda1_values = []
        best_lambda2_values = []

        for train_idx, test_idx in outer_cv.split(X_array):
            X_train, X_test = X_array[train_idx], X_array[test_idx]
            y_time_train, y_time_test = y_time_array[train_idx], y_time_array[test_idx]
            y_event_train, y_event_test = y_event_array[train_idx], y_event_array[test_idx]

            # Inner CV for hyperparameter selection
            best_score = -np.inf
            best_l1 = lambda1_grid[0]
            best_l2 = lambda2_grid[0]

            inner_cv = KFold(n_splits=cv_inner, shuffle=True, random_state=self.random_state)

            for lambda1_val in lambda1_grid:
                for lambda2_val in lambda2_grid:
                    inner_scores = []

                    for inner_train_idx, inner_val_idx in inner_cv.split(X_train):
                        X_inner_train = X_train[inner_train_idx]
                        X_inner_val = X_train[inner_val_idx]
                        y_time_inner_train = y_time_train[inner_train_idx]
                        y_event_inner_train = y_event_train[inner_train_idx]
                        y_time_inner_val = y_time_train[inner_val_idx]
                        y_event_inner_val = y_event_train[inner_val_idx]

                        # Fit
                        model_tmp = SparseGroupLassoCoxModel(
                            lambda1=lambda1_val,
                            lambda2=lambda2_val,
                            groups=self.groups,
                            random_state=self.random_state,
                        )
                        model_tmp.fit(X_inner_train, y_time_inner_train, y_event_inner_train)

                        # Evaluate
                        risk_scores = model_tmp.predict_risk(X_inner_val)
                        c_index = concordance_index_censored(
                            y_event_inner_val.astype(bool),
                            y_time_inner_val,
                            risk_scores,
                        )[0]
                        inner_scores.append(c_index)

                    mean_inner_score = np.mean(inner_scores)

                    if mean_inner_score > best_score:
                        best_score = mean_inner_score
                        best_l1 = lambda1_val
                        best_l2 = lambda2_val

            best_lambda1_values.append(best_l1)
            best_lambda2_values.append(best_l2)

            # Fit outer model with best lambdas
            model_outer = SparseGroupLassoCoxModel(
                lambda1=best_l1,
                lambda2=best_l2,
                groups=self.groups,
                random_state=self.random_state,
            )
            model_outer.fit(X_train, y_time_train, y_event_train)

            # Evaluate on outer test
            risk_scores_test = model_outer.predict_risk(X_test)
            c_index_test = concordance_index_censored(
                y_event_test.astype(bool),
                y_time_test,
                risk_scores_test,
            )[0]
            cv_scores_outer.append(c_index_test)

        best_lambda1 = np.median(best_lambda1_values)
        best_lambda2 = np.median(best_lambda2_values)
        cv_score = np.mean(cv_scores_outer)
        cv_scores = np.array(cv_scores_outer)

        # Refit with best lambdas
        self.lambda1 = best_lambda1
        self.lambda2 = best_lambda2
        self.fit(X, y_time, y_event)

        return best_lambda1, best_lambda2, cv_score, cv_scores
