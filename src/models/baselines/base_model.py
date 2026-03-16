"""
Abstract base class for classical survival models.

Provides:
- Standardized interface: fit(), predict_risk(), predict_survival_function()
- MLflow logging and metric tracking
- Model serialization (pickle/joblib)
- Cross-validation wrapper with patient-level stratification
- Hyperparameter tracking
"""

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


class BaseSurvivalModel(ABC, BaseEstimator):
    """
    Abstract base class for survival models.

    Implements common interface for fitting, prediction, serialization, and MLflow logging.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize base survival model.

        Parameters
        ----------
        random_state : int
            Random state for reproducibility.
        """
        self.random_state = random_state
        self.scaler_ = None
        self.feature_names_ = None
        self.n_features_in_ = None
        self.model_ = None

    @abstractmethod
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y_time: Union[np.ndarray, pd.Series],
        y_event: Union[np.ndarray, pd.Series],
        **kwargs,
    ) -> "BaseSurvivalModel":
        """
        Fit the survival model.

        Parameters
        ----------
        X : array-like or DataFrame, shape (n_samples, n_features)
            Feature matrix.
        y_time : array-like, shape (n_samples,)
            Observed time-to-event or censoring time.
        y_event : array-like, shape (n_samples,)
            Event indicator (1=event, 0=censored).
        **kwargs
            Additional model-specific arguments.

        Returns
        -------
        self
        """
        pass

    @abstractmethod
    def predict_risk(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict risk scores.

        Parameters
        ----------
        X : array-like or DataFrame, shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        risk_scores : array, shape (n_samples,)
            Predicted risk scores (higher = higher risk).
        """
        pass

    @abstractmethod
    def predict_survival_function(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        times: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Predict survival probability S(t) over time points.

        Parameters
        ----------
        X : array-like or DataFrame, shape (n_samples, n_features)
            Feature matrix.
        times : array-like, optional
            Time points at which to evaluate survival. If None, use all unique event times.

        Returns
        -------
        survival_prob : DataFrame
            Survival probabilities. Shape: (n_samples, n_times).
            Index = sample indices, columns = time points.
        """
        pass

    def _validate_input(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y_time: Optional[Union[np.ndarray, pd.Series]] = None,
        y_event: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Validate and convert input to numpy arrays.

        Parameters
        ----------
        X : array-like or DataFrame
            Feature matrix.
        y_time : array-like, optional
            Time variable.
        y_event : array-like, optional
            Event indicator.

        Returns
        -------
        X_array : ndarray
            Feature matrix as array.
        y_time_array : ndarray or None
            Time variable as array.
        y_event_array : ndarray or None
            Event indicator as array.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X_array = X.values
        else:
            X_array = np.asarray(X)
            if self.feature_names_ is None:
                self.feature_names_ = [f"feature_{i}" for i in range(X_array.shape[1])]

        self.n_features_in_ = X_array.shape[1]

        y_time_array = np.asarray(y_time) if y_time is not None else None
        y_event_array = np.asarray(y_event) if y_event is not None else None

        return X_array, y_time_array, y_event_array

    def _fit_scaler(self, X: np.ndarray) -> np.ndarray:
        """
        Fit StandardScaler and transform features.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        X_scaled : ndarray
            Scaled feature matrix.
        """
        self.scaler_ = StandardScaler()
        return self.scaler_.fit_transform(X)

    def _transform_scaler(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted scaler.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        X_scaled : ndarray
            Scaled feature matrix.
        """
        if self.scaler_ is None:
            raise RuntimeError("Scaler not fitted. Call fit() first.")
        return self.scaler_.transform(X)

    def log_params_to_mlflow(self, params: Dict[str, Union[str, int, float]]) -> None:
        """
        Log hyperparameters to MLflow.

        Parameters
        ----------
        params : dict
            Hyperparameters to log.
        """
        mlflow.log_params(params)

    def log_metrics_to_mlflow(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics to MLflow.

        Parameters
        ----------
        metrics : dict
            Metrics to log.
        step : int, optional
            Step/epoch number.
        """
        mlflow.log_metrics(metrics, step=step)

    def log_model_to_mlflow(self, artifact_path: str = "model") -> None:
        """
        Log model to MLflow.

        Parameters
        ----------
        artifact_path : str
            Path within MLflow artifact store to save model.
        """
        try:
            mlflow.sklearn.log_model(self.model_, artifact_path)
        except Exception as e:
            mlflow.log_artifact(self.save(None), artifact_path)

    def save(self, path: Union[str, Path]) -> Union[str, Path]:
        """
        Serialize model to disk using joblib.

        Parameters
        ----------
        path : str or Path
            Output file path (e.g., 'model.joblib').

        Returns
        -------
        path : str or Path
            Path where model was saved.
        """
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: Union[str, Path]) -> "BaseSurvivalModel":
        """
        Load serialized model from disk.

        Parameters
        ----------
        path : str or Path
            Model file path.

        Returns
        -------
        model : BaseSurvivalModel
            Loaded model instance.
        """
        return joblib.load(path)

    def get_params(self, deep: bool = True) -> Dict:
        """
        Get parameters for this model (sklearn-compatible).

        Parameters
        ----------
        deep : bool
            If True, return parameters for estimators within this object.

        Returns
        -------
        params : dict
            Model parameters.
        """
        params = super().get_params(deep=deep)
        return params

    def set_params(self, **params) -> "BaseSurvivalModel":
        """
        Set parameters (sklearn-compatible).

        Parameters
        ----------
        **params
            Model parameters to set.

        Returns
        -------
        self
        """
        super().set_params(**params)
        return self


class SurvivalModelCV:
    """
    Cross-validation wrapper for survival models with patient-level stratification.

    Ensures no patient data leaks between folds.
    """

    def __init__(
        self,
        model: BaseSurvivalModel,
        n_splits: int = 5,
        stratify_by: Optional[pd.DataFrame] = None,
        random_state: int = 42,
    ):
        """
        Initialize CV wrapper.

        Parameters
        ----------
        model : BaseSurvivalModel
            Model to cross-validate.
        n_splits : int
            Number of cross-validation folds.
        stratify_by : DataFrame, optional
            Variables to stratify by (e.g., ISS stage, cytogenetics).
            Index should match X and y.
        random_state : int
            Random state for reproducibility.
        """
        self.model = model
        self.n_splits = n_splits
        self.stratify_by = stratify_by
        self.random_state = random_state
        self.cv_results_ = {}
        self.cv_models_ = []

    def split(
        self, X: Union[np.ndarray, pd.DataFrame], y_event: Union[np.ndarray, pd.Series]
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate patient-level CV splits.

        Parameters
        ----------
        X : array-like or DataFrame
            Feature matrix.
        y_event : array-like
            Event indicator.

        Returns
        -------
        splits : list of tuples
            Each tuple: (train_indices, test_indices).
        """
        n_samples = len(X) if isinstance(X, pd.DataFrame) else X.shape[0]

        if self.stratify_by is not None:
            # Use stratified k-fold with categorical stratification variables
            strata = self._create_strata(self.stratify_by)
            splitter = StratifiedKFold(
                n_splits=self.n_splits, shuffle=True, random_state=self.random_state
            )
            splits = list(splitter.split(X, strata))
        else:
            # Simple stratified k-fold by event indicator
            splitter = StratifiedKFold(
                n_splits=self.n_splits, shuffle=True, random_state=self.random_state
            )
            splits = list(splitter.split(X, y_event))

        return splits

    @staticmethod
    def _create_strata(df: pd.DataFrame) -> np.ndarray:
        """
        Create stratification groups from multiple variables.

        Parameters
        ----------
        df : DataFrame
            Variables to stratify by.

        Returns
        -------
        strata : ndarray
            Strata group identifiers.
        """
        strata = np.zeros(len(df), dtype=object)
        for col in df.columns:
            strata = strata.astype(str) + "_" + df[col].astype(str)
        return strata

    def fit_predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y_time: Union[np.ndarray, pd.Series],
        y_event: Union[np.ndarray, pd.Series],
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit model on each fold and predict on held-out test sets.

        Implements proper cross-validation with no data leakage:
        - Each fold gets a fresh model instance
        - Scalers are fit on training fold only
        - No information flows from test fold to training

        Parameters
        ----------
        X : array-like or DataFrame
            Feature matrix.
        y_time : array-like
            Survival time.
        y_event : array-like
            Event indicator.
        **kwargs
            Additional fit arguments.

        Returns
        -------
        y_pred_train : ndarray, shape (n_samples,)
            Risk scores on training folds.
        y_pred_test : ndarray, shape (n_samples,)
            Risk scores on test folds.
        """
        X_array, y_time_array, y_event_array = self.model._validate_input(X, y_time, y_event)

        y_pred_train = np.zeros(X_array.shape[0])
        y_pred_test = np.zeros(X_array.shape[0])

        splits = self.split(X_array, y_event_array)

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train, X_test = X_array[train_idx], X_array[test_idx]
            y_time_train = y_time_array[train_idx]
            y_event_train = y_event_array[train_idx]

            # Create fresh model for this fold
            # IMPORTANT: This ensures scaler is fit on train fold only
            model_fold = self.model.__class__(**self.model.get_params())
            model_fold.fit(X_train, y_time_train, y_event_train, **kwargs)

            # Predictions (scaler transforms test fold independently)
            y_pred_train[train_idx] = model_fold.predict_risk(X_train)
            y_pred_test[test_idx] = model_fold.predict_risk(X_test)

            self.cv_models_.append(model_fold)

        return y_pred_train, y_pred_test

    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """
        Aggregate feature importance across CV folds.

        Returns
        -------
        importance : dict
            Feature name -> importance scores across folds.
        """
        importance_dict = {}

        for fold_idx, model_fold in enumerate(self.cv_models_):
            if hasattr(model_fold, "feature_importance_"):
                for fname, imp in model_fold.feature_importance_.items():
                    if fname not in importance_dict:
                        importance_dict[fname] = []
                    importance_dict[fname].append(imp)

        # Average across folds
        for fname in importance_dict:
            importance_dict[fname] = np.mean(importance_dict[fname])

        return importance_dict
