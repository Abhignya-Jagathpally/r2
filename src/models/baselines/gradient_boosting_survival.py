"""
Gradient boosting survival models: XGBoost and CatBoost.

Features:
- XGBoost with AFT (Accelerated Failure Time) and Cox objectives
- CatBoost survival mode
- SHAP explanations
- Ray Tune hyperparameter optimization
"""

from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sksurv.util import Surv
from sklearn.model_selection import KFold

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from .base_model import BaseSurvivalModel


class XGBoostSurvivalModel(BaseSurvivalModel):
    """
    XGBoost survival model with AFT or Cox objectives.
    """

    def __init__(
        self,
        objective: str = "survival:cox",  # or "survival:aft"
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        random_state: int = 42,
    ):
        """
        Initialize XGBoost survival model.

        Parameters
        ----------
        objective : str
            'survival:cox' or 'survival:aft'.
        n_estimators : int
            Number of boosting rounds.
        max_depth : int
            Maximum tree depth.
        learning_rate : float
            Learning rate (eta).
        subsample : float
            Fraction of samples for each tree.
        colsample_bytree : float
            Fraction of features for each tree.
        reg_alpha : float
            L1 regularization.
        reg_lambda : float
            L2 regularization.
        random_state : int
            Random state.
        """
        super().__init__(random_state=random_state)

        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")

        self.objective = objective
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda

        # Model
        self.model_ = None
        self.feature_importance_ = None
        self.explainer_ = None

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y_time: Union[np.ndarray, pd.Series],
        y_event: Union[np.ndarray, pd.Series],
        eval_set: Optional[Tuple] = None,
        **kwargs,
    ) -> "XGBoostSurvivalModel":
        """
        Fit XGBoost survival model.

        Parameters
        ----------
        X : array-like or DataFrame, shape (n_samples, n_features)
            Feature matrix.
        y_time : array-like, shape (n_samples,)
            Survival time.
        y_event : array-like, shape (n_samples,)
            Event indicator.
        eval_set : tuple, optional
            Evaluation set for early stopping.
        **kwargs
            Additional arguments (unused).

        Returns
        -------
        self
        """
        X_array, y_time_array, y_event_array = self._validate_input(X, y_time, y_event)

        # Fit scaler
        X_scaled = self._fit_scaler(X_array)

        # Create DMatrix with proper label encoding per objective
        if self.objective == "survival:aft":
            dmatrix = xgb.DMatrix(X_scaled)
            # For AFT: lower_bound = time for all, upper_bound = time for events, +inf for censored
            y_lower = y_time_array.copy()
            y_upper = np.where(y_event_array > 0, y_time_array, np.inf)
            dmatrix.set_float_info("label_lower_bound", y_lower)
            dmatrix.set_float_info("label_upper_bound", y_upper)
        else:
            # survival:cox: label = +time if event, -time if censored
            y_label = np.where(y_event_array > 0, y_time_array, -y_time_array)
            dmatrix = xgb.DMatrix(X_scaled, label=y_label)

        params = {
            "objective": self.objective,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "seed": self.random_state,
        }

        self.model_ = xgb.train(params, dmatrix, num_boost_round=self.n_estimators)

        # Feature importance
        self.feature_importance_ = {
            fname: imp
            for fname, imp in zip(
                self.feature_names_,
                self.model_.get_score(importance_type="weight").values(),
            )
        }

        return self

    def predict_risk(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict risk scores.

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
        X_scaled = self._transform_scaler(X_array)

        dmatrix = xgb.DMatrix(X_scaled)
        predictions = self.model_.predict(dmatrix)

        return predictions

    def predict_survival_function(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        times: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Predict survival probability (dummy implementation).

        Parameters
        ----------
        X : array-like or DataFrame
            Feature matrix.
        times : ndarray, optional
            Time points.

        Returns
        -------
        survival_prob : DataFrame
            Dummy survival probabilities.
        """
        risk_scores = self.predict_risk(X)
        n_samples = len(risk_scores)

        if times is None:
            times = np.array([1.0, 2.0, 3.0, 5.0, 10.0])

        result = pd.DataFrame(
            index=np.arange(n_samples),
            columns=times,
        )

        for t in times:
            result[t] = np.exp(-risk_scores * t / times.max())

        return result

    def explain_with_shap(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        sample_size: Optional[int] = None,
    ) -> Dict:
        """
        Compute SHAP explanations.

        Parameters
        ----------
        X : array-like or DataFrame
            Feature matrix.
        sample_size : int, optional
            Sample size for background data.

        Returns
        -------
        shap_values : dict
            SHAP values and base value.
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not installed. Install with: pip install shap")

        X_array, _, _ = self._validate_input(X)
        X_scaled = self._transform_scaler(X_array)

        if sample_size is not None:
            indices = np.random.choice(len(X_scaled), size=min(sample_size, len(X_scaled)))
            X_background = X_scaled[indices]
        else:
            X_background = X_scaled

        explainer = shap.TreeExplainer(self.model_)
        shap_values = explainer.shap_values(X_scaled)

        return {
            "shap_values": shap_values,
            "base_value": explainer.expected_value,
            "feature_names": self.feature_names_,
        }


class CatBoostSurvivalModel(BaseSurvivalModel):
    """
    CatBoost survival model.
    """

    def __init__(
        self,
        iterations: int = 100,
        depth: int = 6,
        learning_rate: float = 0.1,
        l2_leaf_reg: float = 3.0,
        random_strength: float = 1.0,
        subsample: float = 0.8,
        random_state: int = 42,
        verbose: bool = False,
    ):
        """
        Initialize CatBoost survival model.

        Parameters
        ----------
        iterations : int
            Number of boosting iterations.
        depth : int
            Tree depth.
        learning_rate : float
            Learning rate.
        l2_leaf_reg : float
            L2 regularization coefficient.
        random_strength : float
            Randomness for splitting.
        subsample : float
            Subsample ratio.
        random_state : int
            Random state.
        verbose : bool
            Whether to print progress.
        """
        super().__init__(random_state=random_state)

        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost not installed. Install with: pip install catboost")

        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.random_strength = random_strength
        self.subsample = subsample
        self.verbose = verbose

        # Model
        self.model_ = None
        self.feature_importance_ = None

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y_time: Union[np.ndarray, pd.Series],
        y_event: Union[np.ndarray, pd.Series],
        **kwargs,
    ) -> "CatBoostSurvivalModel":
        """
        Fit CatBoost survival model.

        Parameters
        ----------
        X : array-like or DataFrame
            Feature matrix.
        y_time : array-like
            Survival time.
        y_event : array-like
            Event indicator.
        **kwargs
            Additional arguments (unused).

        Returns
        -------
        self
        """
        X_array, y_time_array, y_event_array = self._validate_input(X, y_time, y_event)

        # Fit scaler
        X_scaled = self._fit_scaler(X_array)

        # Create pool: CatBoost Cox loss expects label = +time if event, -time if censored
        y_label = np.where(y_event_array > 0, y_time_array, -y_time_array)
        pool = cb.Pool(
            X_scaled,
            label=y_label,
        )

        self.model_ = cb.CatBoostRegressor(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            random_strength=self.random_strength,
            subsample=self.subsample,
            loss_function="Cox",
            random_state=self.random_state,
            verbose=self.verbose,
        )

        self.model_.fit(pool)

        # Feature importance
        self.feature_importance_ = {
            fname: imp
            for fname, imp in zip(self.feature_names_, self.model_.feature_importances_)
        }

        return self

    def predict_risk(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict risk scores.

        Parameters
        ----------
        X : array-like or DataFrame
            Feature matrix.

        Returns
        -------
        risk_scores : ndarray
            Risk scores.
        """
        X_array, _, _ = self._validate_input(X)
        X_scaled = self._transform_scaler(X_array)

        predictions = self.model_.predict(X_scaled)
        return np.exp(predictions)  # Exponentiate for hazard

    def predict_survival_function(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        times: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Predict survival probability (dummy implementation).

        Parameters
        ----------
        X : array-like or DataFrame
            Feature matrix.
        times : ndarray, optional
            Time points.

        Returns
        -------
        survival_prob : DataFrame
            Dummy survival probabilities.
        """
        risk_scores = self.predict_risk(X)
        n_samples = len(risk_scores)

        if times is None:
            times = np.array([1.0, 2.0, 3.0, 5.0, 10.0])

        result = pd.DataFrame(
            index=np.arange(n_samples),
            columns=times,
        )

        for t in times:
            result[t] = np.exp(-risk_scores * t / times.max())

        return result

    def grid_search_cv(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y_time: Union[np.ndarray, pd.Series],
        y_event: Union[np.ndarray, pd.Series],
        param_grid: Optional[Dict] = None,
        cv: int = 5,
    ) -> Tuple[Dict, float]:
        """
        Grid search for hyperparameters.

        Parameters
        ----------
        X : array-like or DataFrame
            Feature matrix.
        y_time : array-like
            Survival time.
        y_event : array-like
            Event indicator.
        param_grid : dict, optional
            Parameter grid.
        cv : int
            Cross-validation folds.

        Returns
        -------
        best_params : dict
            Best parameters.
        best_score : float
            Best CV score.
        """
        from sksurv.metrics import concordance_index_censored

        X_array, y_time_array, y_event_array = self._validate_input(X, y_time, y_event)

        if param_grid is None:
            param_grid = {
                "iterations": [50, 100, 200],
                "depth": [4, 6, 8],
                "learning_rate": [0.01, 0.1, 0.3],
            }

        best_score = -np.inf
        best_params = {}

        from itertools import product

        param_keys = list(param_grid.keys())
        param_values = [param_grid[k] for k in param_keys]

        for param_combo in product(*param_values):
            params = dict(zip(param_keys, param_combo))
            fold_scores = []

            splitter = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)

            for train_idx, test_idx in splitter.split(X_array):
                X_train, X_test = X_array[train_idx], X_array[test_idx]
                y_time_train = y_time_array[train_idx]
                y_event_train = y_event_array[train_idx]
                y_time_test = y_time_array[test_idx]
                y_event_test = y_event_array[test_idx]

                model = CatBoostSurvivalModel(
                    **params,
                    random_state=self.random_state,
                    verbose=False,
                )
                model.fit(X_train, y_time_train, y_event_train)

                risk_scores = model.predict_risk(X_test)
                c_index = concordance_index_censored(
                    y_event_test.astype(bool),
                    y_time_test,
                    risk_scores,
                )[0]
                fold_scores.append(c_index)

            mean_score = np.mean(fold_scores)

            if mean_score > best_score:
                best_score = mean_score
                best_params = params

        for key, val in best_params.items():
            setattr(self, key, val)

        self.fit(X, y_time, y_event)

        return best_params, best_score
