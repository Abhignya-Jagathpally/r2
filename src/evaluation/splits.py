"""
Patient-level cross-validation splitting strategies.

Ensures no data leakage and proper stratification by clinical variables.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
    TimeSeriesSplit,
)


class PatientLevelSplitter:
    """
    Patient-level cross-validation splitter.

    Ensures each patient (group of samples) is entirely in train or test.
    """

    def __init__(
        self,
        n_splits: int = 5,
        stratify_by: Optional[pd.DataFrame] = None,
        random_state: int = 42,
    ):
        """
        Initialize patient-level splitter.

        Parameters
        ----------
        n_splits : int
            Number of CV folds.
        stratify_by : DataFrame, optional
            Variables to stratify by (ISS stage, cytogenetics, etc.).
        random_state : int
            Random state.
        """
        self.n_splits = n_splits
        self.stratify_by = stratify_by
        self.random_state = random_state

    def split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        groups: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate patient-level CV folds.

        Parameters
        ----------
        X : array-like or DataFrame
            Feature matrix (samples x features).
        y : array-like, optional
            Target variable (for stratification).
        groups : array-like, optional
            Group labels (patient IDs). If None, assume one-to-one mapping.

        Returns
        -------
        splits : list of tuples
            Each tuple: (train_indices, test_indices).
        """
        n_samples = len(X) if isinstance(X, pd.DataFrame) else X.shape[0]

        if groups is None:
            groups = np.arange(n_samples)

        # Get unique groups
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)

        # Stratified or regular split?
        if self.stratify_by is not None:
            strata = self._create_strata_from_groups(groups, self.stratify_by)
            splitter = StratifiedKFold(
                n_splits=min(self.n_splits, n_groups // 2),
                shuffle=True,
                random_state=self.random_state,
            )
            splits = list(splitter.split(unique_groups, strata))
        else:
            splitter = StratifiedKFold(
                n_splits=min(self.n_splits, n_groups // 2),
                shuffle=True,
                random_state=self.random_state,
            )
            split_strata = np.zeros(n_groups)
            if y is not None:
                # Stratify by event
                split_strata = np.array([y[groups == g].iloc[0] if isinstance(y, pd.Series) else y[groups == g][0]
                                        for g in unique_groups])
            splits = list(splitter.split(unique_groups, split_strata))

        # Convert group indices to sample indices
        result_splits = []
        for train_group_idx, test_group_idx in splits:
            train_groups = unique_groups[train_group_idx]
            test_groups = unique_groups[test_group_idx]

            train_samples = np.where(np.isin(groups, train_groups))[0]
            test_samples = np.where(np.isin(groups, test_groups))[0]

            result_splits.append((train_samples, test_samples))

        return result_splits

    @staticmethod
    def _create_strata_from_groups(
        groups: np.ndarray,
        stratify_by: pd.DataFrame,
    ) -> np.ndarray:
        """
        Create stratification labels from groups.

        Parameters
        ----------
        groups : ndarray
            Group labels (patient IDs).
        stratify_by : DataFrame
            Variables to stratify by.

        Returns
        -------
        strata : ndarray
            Strata per group.
        """
        unique_groups = np.unique(groups)
        strata = []

        for group in unique_groups:
            # Get first sample of this group
            group_samples = np.where(groups == group)[0]
            first_sample = group_samples[0]

            # Get strata value
            strata_str = "_".join(
                str(stratify_by.iloc[first_sample, col])
                for col in range(stratify_by.shape[1])
            )
            strata.append(strata_str)

        return np.array(strata)


class TimeAwareSplitter:
    """
    Time-aware CV: splits by follow-up time window.

    Useful for validating models on specific follow-up periods.
    """

    def __init__(
        self,
        time_windows: Optional[List[Tuple[float, float]]] = None,
        random_state: int = 42,
    ):
        """
        Initialize time-aware splitter.

        Parameters
        ----------
        time_windows : list of tuples, optional
            Time windows (min_time, max_time). If None, create tertiles.
        random_state : int
            Random state.
        """
        self.time_windows = time_windows
        self.random_state = random_state

    def split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y_time: Union[np.ndarray, pd.Series],
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate time-aware CV splits.

        Parameters
        ----------
        X : array-like or DataFrame
            Feature matrix.
        y_time : array-like
            Survival times.

        Returns
        -------
        splits : list of tuples
            Train/test splits by time window.
        """
        y_time = np.asarray(y_time)
        n_samples = len(y_time)

        if self.time_windows is None:
            # Create tertiles
            t_lower = np.percentile(y_time, 33)
            t_middle = np.percentile(y_time, 67)
            self.time_windows = [
                (0, t_lower),
                (t_lower, t_middle),
                (t_middle, np.inf),
            ]

        splits = []

        for min_t, max_t in self.time_windows:
            # Test set: samples in time window
            test_mask = (y_time >= min_t) & (y_time < max_t)
            test_idx = np.where(test_mask)[0]

            # Train set: samples outside window
            train_idx = np.where(~test_mask)[0]

            if len(test_idx) > 0 and len(train_idx) > 0:
                splits.append((train_idx, test_idx))

        return splits


class StratifiedTimeAwareSplitter:
    """
    Stratified time-aware CV: stratify by ISS/cytogenetics within time windows.
    """

    def __init__(
        self,
        n_splits: int = 5,
        stratify_cols: Optional[List[str]] = None,
        random_state: int = 42,
    ):
        """
        Initialize stratified time-aware splitter.

        Parameters
        ----------
        n_splits : int
            Number of CV folds.
        stratify_cols : list, optional
            Columns to stratify by (e.g., ['ISS_stage', 'cytogenetics']).
        random_state : int
            Random state.
        """
        self.n_splits = n_splits
        self.stratify_cols = stratify_cols or []
        self.random_state = random_state

    def split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: pd.DataFrame,  # Should have time, event, and stratify columns
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate stratified time-aware CV splits.

        Parameters
        ----------
        X : array-like or DataFrame
            Feature matrix.
        y : DataFrame
            Targets with columns: 'time', 'event', and stratify_cols.

        Returns
        -------
        splits : list of tuples
            Train/test splits.
        """
        n_samples = len(X) if isinstance(X, pd.DataFrame) else X.shape[0]

        # Create strata
        strata_str = np.zeros(n_samples, dtype=object)
        for col in self.stratify_cols:
            strata_str = strata_str.astype(str) + "_" + y[col].astype(str).values

        # Stratified KFold
        splitter = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        return list(splitter.split(X, strata_str))


class LeaveOneStudyOutSplitter:
    """
    Leave-one-study-out (LOSO) cross-validation.

    For multi-cohort datasets, validate on held-out cohorts.
    """

    def __init__(self):
        """Initialize LOSO splitter."""
        self.studies = None

    def split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        study_ids: Union[np.ndarray, pd.Series],
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate LOSO splits.

        Parameters
        ----------
        X : array-like or DataFrame
            Feature matrix.
        study_ids : array-like
            Study ID per sample (e.g., 'CoMMpass', 'GSE24080').

        Returns
        -------
        splits : list of tuples
            Train on all studies except one, test on held-out study.
        """
        study_ids = np.asarray(study_ids)
        self.studies = np.unique(study_ids)
        splits = []

        for test_study in self.studies:
            train_idx = np.where(study_ids != test_study)[0]
            test_idx = np.where(study_ids == test_study)[0]

            if len(test_idx) > 0:
                splits.append((train_idx, test_idx))

        return splits

    def get_study_names(self) -> List[str]:
        """Get unique study names."""
        return list(self.studies) if self.studies is not None else []


class NestedCVSplitter:
    """
    Nested cross-validation: outer CV for model evaluation, inner CV for hyperparameter tuning.
    """

    def __init__(
        self,
        n_splits_outer: int = 5,
        n_splits_inner: int = 3,
        stratify_by: Optional[pd.DataFrame] = None,
        random_state: int = 42,
    ):
        """
        Initialize nested CV splitter.

        Parameters
        ----------
        n_splits_outer : int
            Outer CV folds.
        n_splits_inner : int
            Inner CV folds.
        stratify_by : DataFrame, optional
            Variables for stratification.
        random_state : int
            Random state.
        """
        self.n_splits_outer = n_splits_outer
        self.n_splits_inner = n_splits_inner
        self.stratify_by = stratify_by
        self.random_state = random_state

    def split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate outer CV splits.

        Parameters
        ----------
        X : array-like or DataFrame
            Feature matrix.
        y : array-like, optional
            Target for stratification.

        Returns
        -------
        splits : list of tuples
            Outer fold splits.
        """
        splitter = PatientLevelSplitter(
            n_splits=self.n_splits_outer,
            stratify_by=self.stratify_by,
            random_state=self.random_state,
        )
        return splitter.split(X, y)

    def inner_split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate inner CV splits.

        Parameters
        ----------
        X : array-like or DataFrame
            Feature matrix (training set from outer fold).
        y : array-like, optional
            Target.

        Returns
        -------
        splits : list of tuples
            Inner fold splits.
        """
        splitter = PatientLevelSplitter(
            n_splits=self.n_splits_inner,
            stratify_by=None,  # Don't stratify inner
            random_state=self.random_state,
        )
        return splitter.split(X, y)
