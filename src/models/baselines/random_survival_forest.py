"""
Random Survival Forest via scikit-survival.

Features:
- Tree-based nonlinear survival modeling
- Feature importance via permutation
- Hyperparameter tuning (n_estimators, max_depth, min_samples_leaf)
- Partial dependence plots
"""

from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.preprocessing import Standardizer
from sksurv.util import Surv
from sklearn.model_selection import GridSearchCV, KFold

from .base_model import BaseSurvivalModel


class RandomSurvivalForestModel(BaseSurvivalModel):
    """
    Random Survival Forest from scikit-survival.

    Ensemble of survival trees for nonlinear risk prediction.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 1,
        min_samples_split: int = 2,
        max_features: Union[int, str] = "sqrt",
        bootstrap: bool = True,
        random_state: int = 42,
    ):
        """
        Initialize Random Survival Forest.

        Parameters
        ----------
        n_estimators : int
            Number of trees.
        max_depth : int or None
            Maximum tree depth. None = unlimited.
        min_samples_leaf : int
            Minimum samples required to be at a leaf node.
        min_samples_split : int
            Minimum samples required to split a node.
        max_features : int or str
            Number of features to consider for best split.
            'sqrt', 'log2', or integer.
        bootstrap : bool
            Whether to use bootstrap samples.
        random_state : int
            Random state.
        """
        super().__init__(random_state=random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap

        # Model components
        self.model_ = None
        self.feature_importance_ = None
        self.permutation_importance_ = None

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y_time: Union[np.ndarray, pd.Series],
        y_event: Union[np.ndarray, pd.Series],
        **kwargs,
    ) -> "RandomSurvivalForestModel":
        """
        Fit Random Survival Forest.

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

        # Create structured array for sksurv
        y_struct = Surv.from_arrays(y_event_array.astype(bool), y_time_array)

        # Fit model
        self.model_ = RandomSurvivalForest(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            min_samples_split=self.min_samples_split,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            random_state=self.random_state,
            n_jobs=-1,
        )

        df = pd.DataFrame(X_array, columns=self.feature_names_)
        self.model_.fit(df, y_struct)

        # Extract feature importance
        self.feature_importance_ = {
            fname: imp for fname, imp in zip(self.feature_names_, self.model_.feature_importances_)
        }

        return self

    def predict_risk(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict risk scores (cumulative hazard at median follow-up).

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
        df = pd.DataFrame(X_array, columns=self.feature_names_)

        # Use cumulative hazard at median time as risk score
        cumulative_hazard = self.model_.predict_cumulative_hazard(df)
        median_time_idx = len(cumulative_hazard) // 2
        risk_scores = cumulative_hazard.iloc[median_time_idx].values

        return risk_scores

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
            Survival probabilities. Shape: (n_samples, n_times).
        """
        X_array, _, _ = self._validate_input(X)
        df = pd.DataFrame(X_array, columns=self.feature_names_)

        # Get survival function
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

    def compute_permutation_importance(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y_time: Union[np.ndarray, pd.Series],
        y_event: Union[np.ndarray, pd.Series],
        n_repeats: int = 10,
    ) -> Dict[str, Tuple[float, float]]:
        """
        Compute permutation importance.

        Parameters
        ----------
        X : array-like or DataFrame
            Feature matrix.
        y_time : array-like
            Survival time.
        y_event : array-like
            Event indicator.
        n_repeats : int
            Number of permutation repeats.

        Returns
        -------
        importance : dict
            Feature name -> (mean importance, std).
        """
        X_array, y_time_array, y_event_array = self._validate_input(X, y_time, y_event)
        df = pd.DataFrame(X_array, columns=self.feature_names_)

        y_struct = Surv.from_arrays(y_event_array.astype(bool), y_time_array)

        # Baseline score
        baseline_score = concordance_index_censored(
            y_struct.event,
            y_struct.time,
            self.predict_risk(X_array),
        )[0]

        importance_scores = {fname: [] for fname in self.feature_names_}

        for _ in range(n_repeats):
            for col_idx, fname in enumerate(self.feature_names_):
                # Permute feature
                X_perm = df.copy()
                X_perm.iloc[:, col_idx] = np.random.permutation(X_perm.iloc[:, col_idx])

                # Score with permuted feature
                risk_scores_perm = self.predict_risk(X_perm.values)
                score_perm = concordance_index_censored(
                    y_struct.event,
                    y_struct.time,
                    risk_scores_perm,
                )[0]

                # Importance = drop in score
                importance = baseline_score - score_perm
                importance_scores[fname].append(importance)

        # Average across repeats
        result = {}
        for fname in self.feature_names_:
            scores = np.array(importance_scores[fname])
            result[fname] = (np.mean(scores), np.std(scores))

        self.permutation_importance_ = result
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
            Parameter grid. If None, use default.
        cv : int
            Cross-validation folds.

        Returns
        -------
        best_params : dict
            Best hyperparameters.
        best_score : float
            Best CV score.
        """
        X_array, y_time_array, y_event_array = self._validate_input(X, y_time, y_event)

        if param_grid is None:
            param_grid = {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 20, None],
                "min_samples_leaf": [1, 5, 10],
            }

        y_struct = Surv.from_arrays(y_event_array.astype(bool), y_time_array)
        df = pd.DataFrame(X_array, columns=self.feature_names_)

        best_score = -np.inf
        best_params = {}
        results = []

        # Manual grid search (scikit-survival compatibility)
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

                # Fit with these parameters
                model = RandomSurvivalForestModel(**params, random_state=self.random_state)
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
            results.append({"params": params, "mean_score": mean_score})

            if mean_score > best_score:
                best_score = mean_score
                best_params = params

        # Update self with best parameters
        for key, val in best_params.items():
            setattr(self, key, val)

        self.fit(X, y_time, y_event)

        return best_params, best_score
