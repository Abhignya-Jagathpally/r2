"""
Comprehensive survival model evaluation metrics.

Includes:
- C-index (concordance)
- Uno's C-index (IPCW-weighted)
- Time-dependent AUC
- Integrated Brier Score
- Integrated Calibration Index (ICI)
- Calibration metrics
- Subgroup robustness
- Bootstrap confidence intervals
- Pairwise model comparison

Censoring Assumptions
---------------------
All metrics in this module assume:
1. Non-informative (independent) censoring: the censoring mechanism is
   independent of the event process, conditional on covariates.
2. Right censoring only: observations are censored from the right
   (we know at least how long a patient survived, but not the exact event time).
3. Type I censoring for time-dependent metrics: administrative censoring
   at a fixed study end time.

When censoring may be informative, use Uno's C-index (IPCW-weighted) instead
of Harrell's C-index, as it accounts for the censoring distribution.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sksurv.metrics import (
    brier_score,
    concordance_index_censored,
    cumulative_dynamic_auc,
    integrated_brier_score,
)


class SurvivalMetrics:
    """
    Compute comprehensive survival evaluation metrics.
    """

    @staticmethod
    def concordance_index(
        y_event: np.ndarray,
        y_time: np.ndarray,
        predictions: np.ndarray,
    ) -> float:
        """
        Harrell's C-index.

        Parameters
        ----------
        y_event : ndarray
            Event indicator.
        y_time : ndarray
            Survival time.
        predictions : ndarray
            Predicted risk scores.

        Returns
        -------
        c_index : float
            C-index in [0, 1]. 0.5 = random, 1.0 = perfect.
        """
        c_index, _, _, _, _ = concordance_index_censored(
            y_event.astype(bool),
            y_time,
            predictions,
        )
        return c_index

    @staticmethod
    def time_dependent_auc(
        y_event: np.ndarray,
        y_time: np.ndarray,
        predictions: np.ndarray,
        times: Optional[np.ndarray] = None,
    ) -> Dict[float, float]:
        """
        Time-dependent AUC (dynamic AUC).

        Parameters
        ----------
        y_event : ndarray
            Event indicator.
        y_time : ndarray
            Survival time.
        predictions : ndarray
            Predicted risk scores.
        times : ndarray, optional
            Time points. If None, use quantiles of event times.

        Returns
        -------
        auc_dict : dict
            Time -> AUC value.
        """
        if times is None:
            event_times = y_time[y_event.astype(bool)]
            times = np.percentile(event_times, [25, 50, 75])

        auc_dict = {}

        for t in times:
            try:
                auc, _, _ = cumulative_dynamic_auc(
                    y_event.astype(bool),
                    y_time,
                    predictions,
                    times=np.array([t]),
                )
                auc_dict[t] = auc[0]
            except Exception:
                auc_dict[t] = np.nan

        return auc_dict

    @staticmethod
    def brier_score(
        y_event: np.ndarray,
        y_time: np.ndarray,
        survival_probs: np.ndarray,
        times: Optional[np.ndarray] = None,
    ) -> Dict[float, float]:
        """
        Brier Score (mean squared error of survival probability).

        Parameters
        ----------
        y_event : ndarray
            Event indicator.
        y_time : ndarray
            Survival time.
        survival_probs : ndarray
            Predicted survival probabilities. Shape: (n_samples, n_times).
        times : ndarray, optional
            Time points.

        Returns
        -------
        brier_dict : dict
            Time -> Brier Score.
        """
        if times is None:
            times = np.percentile(y_time[y_event.astype(bool)], [25, 50, 75])

        brier_dict = {}

        for i, t in enumerate(times):
            if i < survival_probs.shape[1]:
                try:
                    bs = brier_score(
                        y_event.astype(bool),
                        y_time,
                        survival_probs[:, i],
                        times=np.array([t]),
                    )
                    brier_dict[t] = bs[0]
                except Exception:
                    brier_dict[t] = np.nan

        return brier_dict

    @staticmethod
    def integrated_brier_score(
        y_event: np.ndarray,
        y_time: np.ndarray,
        survival_probs: np.ndarray,
        times: Optional[np.ndarray] = None,
    ) -> float:
        """
        Integrated Brier Score over time.

        Parameters
        ----------
        y_event : ndarray
            Event indicator.
        y_time : ndarray
            Survival time.
        survival_probs : ndarray
            Predicted survival probabilities. Shape: (n_samples, n_times).
        times : ndarray, optional
            Time points.

        Returns
        -------
        ibs : float
            Integrated Brier Score.
        """
        if times is None:
            times = np.percentile(y_time[y_event.astype(bool)], [25, 50, 75])

        try:
            ibs = integrated_brier_score(
                y_event.astype(bool),
                y_time,
                survival_probs,
                times=times,
            )
            return ibs
        except Exception:
            return np.nan

    @staticmethod
    def calibration_metrics(
        y_event: np.ndarray,
        y_time: np.ndarray,
        predictions: np.ndarray,
    ) -> Dict[str, float]:
        """
        Calibration: slope, intercept, and D-calibration.

        Compares predicted vs observed risk in risk groups.

        Parameters
        ----------
        y_event : ndarray
            Event indicator.
        y_time : ndarray
            Survival time.
        predictions : ndarray
            Predicted risk scores.

        Returns
        -------
        metrics : dict
            Calibration slope, intercept, and D-calibration.
        """
        # Sort by predicted risk
        sort_idx = np.argsort(predictions)
        y_event_sorted = y_event[sort_idx]
        y_time_sorted = y_time[sort_idx]
        pred_sorted = predictions[sort_idx]

        # Divide into risk groups (deciles)
        n_groups = 10
        group_size = len(y_time) // n_groups
        metrics_dict = {}

        observed_events = []
        predicted_risks = []

        for i in range(n_groups):
            start = i * group_size
            end = start + group_size if i < n_groups - 1 else len(y_time)

            group_y_event = y_event_sorted[start:end]
            group_y_time = y_time_sorted[start:end]
            group_pred = pred_sorted[start:end]

            # Observed event rate in this group
            obs_event_rate = group_y_event.sum() / len(group_y_event) if len(group_y_event) > 0 else 0

            # Mean predicted risk
            mean_pred = group_pred.mean()

            observed_events.append(obs_event_rate)
            predicted_risks.append(mean_pred)

        observed_events = np.array(observed_events)
        predicted_risks = np.array(predicted_risks)

        # Calibration slope and intercept
        valid_idx = ~(np.isnan(observed_events) | np.isnan(predicted_risks))
        if valid_idx.sum() > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                predicted_risks[valid_idx],
                observed_events[valid_idx],
            )
            metrics_dict["calibration_slope"] = slope
            metrics_dict["calibration_intercept"] = intercept
            metrics_dict["calibration_r2"] = r_value ** 2
        else:
            metrics_dict["calibration_slope"] = np.nan
            metrics_dict["calibration_intercept"] = np.nan
            metrics_dict["calibration_r2"] = np.nan

        # D-calibration (mean squared error between observed and predicted)
        mse = np.mean((observed_events - predicted_risks) ** 2)
        metrics_dict["d_calibration"] = mse

        return metrics_dict

    @staticmethod
    def subgroup_robustness(
        y_event: np.ndarray,
        y_time: np.ndarray,
        predictions: np.ndarray,
        subgroups: np.ndarray,
    ) -> Dict[Union[int, str], float]:
        """
        C-index per subgroup.

        Assesses model robustness across patient subgroups (e.g., ISS stages).

        Parameters
        ----------
        y_event : ndarray
            Event indicator.
        y_time : ndarray
            Survival time.
        predictions : ndarray
            Predicted risk scores.
        subgroups : ndarray
            Subgroup label per sample (e.g., ISS stage).

        Returns
        -------
        subgroup_scores : dict
            Subgroup -> C-index.
        """
        subgroup_scores = {}
        unique_subgroups = np.unique(subgroups)

        for subgroup in unique_subgroups:
            mask = subgroups == subgroup
            y_event_sub = y_event[mask]
            y_time_sub = y_time[mask]
            pred_sub = predictions[mask]

            if len(y_event_sub) > 1 and y_event_sub.sum() > 0:
                c_index = concordance_index_censored(
                    y_event_sub.astype(bool),
                    y_time_sub,
                    pred_sub,
                )[0]
                subgroup_scores[subgroup] = c_index
            else:
                subgroup_scores[subgroup] = np.nan

        return subgroup_scores

    @staticmethod
    def bootstrap_ci(
        y_event: np.ndarray,
        y_time: np.ndarray,
        predictions: np.ndarray,
        metric_func,
        n_bootstraps: int = 1000,
        ci: float = 0.95,
    ) -> Tuple[float, float, float]:
        """
        Bootstrap confidence intervals for a metric.

        Parameters
        ----------
        y_event : ndarray
            Event indicator.
        y_time : ndarray
            Survival time.
        predictions : ndarray
            Predicted risk scores.
        metric_func : callable
            Metric function to evaluate. Signature: metric_func(y_event, y_time, predictions).
        n_bootstraps : int
            Number of bootstrap samples.
        ci : float
            Confidence interval level (e.g., 0.95 for 95% CI).

        Returns
        -------
        point_estimate : float
            Point estimate on full data.
        ci_lower : float
            Lower CI bound.
        ci_upper : float
            Upper CI bound.
        """
        n_samples = len(y_time)
        point_estimate = metric_func(y_event, y_time, predictions)

        bootstrap_scores = []

        for _ in range(n_bootstraps):
            idx = np.random.choice(n_samples, size=n_samples, replace=True)
            y_event_boot = y_event[idx]
            y_time_boot = y_time[idx]
            pred_boot = predictions[idx]

            try:
                score = metric_func(y_event_boot, y_time_boot, pred_boot)
                bootstrap_scores.append(score)
            except Exception:
                pass

        bootstrap_scores = np.array(bootstrap_scores)
        bootstrap_scores = bootstrap_scores[~np.isnan(bootstrap_scores)]

        alpha = 1 - ci
        ci_lower = np.percentile(bootstrap_scores, alpha / 2 * 100)
        ci_upper = np.percentile(bootstrap_scores, (1 - alpha / 2) * 100)

        return point_estimate, ci_lower, ci_upper

    @staticmethod
    def unos_concordance_index(
        y_event: np.ndarray,
        y_time: np.ndarray,
        predictions: np.ndarray,
        tau: Optional[float] = None,
    ) -> float:
        """
        Uno's C-index (inverse-probability-of-censoring weighted).

        More robust than Harrell's C-index when censoring is informative.
        Uses IPCW to account for censoring distribution.

        Parameters
        ----------
        y_event : ndarray
            Event indicator.
        y_time : ndarray
            Survival time.
        predictions : ndarray
            Predicted risk scores.
        tau : float, optional
            Truncation time. If None, use max observed event time.

        Returns
        -------
        c_index : float
            Uno's C-index.
        """
        from sksurv.metrics import concordance_index_ipcw

        # Create structured arrays for sksurv
        y_train = np.array(
            [(bool(e), t) for e, t in zip(y_event, y_time)],
            dtype=[('event', bool), ('time', float)]
        )

        if tau is None:
            tau = y_time[y_event.astype(bool)].max()

        try:
            c_index, _, _, _, _ = concordance_index_ipcw(
                y_train, y_train, predictions, tau=tau
            )
            return c_index
        except Exception:
            return np.nan

    @staticmethod
    def integrated_calibration_index(
        y_event: np.ndarray,
        y_time: np.ndarray,
        predicted_survival_probs: np.ndarray,
        times: np.ndarray,
    ) -> float:
        """
        Integrated Calibration Index (ICI) for time-dependent calibration.

        Measures the mean absolute difference between predicted and observed
        survival probabilities across the full range of predictions.

        Parameters
        ----------
        y_event : ndarray
            Event indicator.
        y_time : ndarray
            Survival time.
        predicted_survival_probs : ndarray, shape (n_samples,)
            Predicted survival probabilities at a specific time point.
        times : ndarray
            Time points for evaluation.

        Returns
        -------
        ici : float
            Integrated Calibration Index (lower is better).
        """
        from sklearn.isotonic import IsotonicRegression

        ici_values = []
        for t_idx, t in enumerate(times):
            if t_idx >= predicted_survival_probs.shape[1] if predicted_survival_probs.ndim > 1 else True:
                break

            pred_surv = predicted_survival_probs[:, t_idx] if predicted_survival_probs.ndim > 1 else predicted_survival_probs

            # Observed: did the patient survive past time t?
            observed = (y_time > t).astype(float)
            # Only use uncensored or censored after t
            usable = (y_event.astype(bool)) | (y_time > t)

            if usable.sum() < 10:
                continue

            pred_sub = pred_surv[usable]
            obs_sub = observed[usable]

            # Fit isotonic regression (calibration curve)
            ir = IsotonicRegression(out_of_bounds='clip')
            calibrated = ir.fit_transform(pred_sub, obs_sub)

            # ICI = mean |predicted - calibrated|
            ici_t = np.mean(np.abs(pred_sub - calibrated))
            ici_values.append(ici_t)

        return np.mean(ici_values) if ici_values else np.nan

    @staticmethod
    def pairwise_model_comparison(
        y_event: np.ndarray,
        y_time: np.ndarray,
        predictions_model1: np.ndarray,
        predictions_model2: np.ndarray,
        metric: str = "c_index",
    ) -> Dict[str, Union[float, str]]:
        """
        Statistical comparison of two models.

        Parameters
        ----------
        y_event : ndarray
            Event indicator.
        y_time : ndarray
            Survival time.
        predictions_model1 : ndarray
            Predictions from model 1.
        predictions_model2 : ndarray
            Predictions from model 2.
        metric : str
            Metric for comparison: 'c_index', 'auc', 'brier'.

        Returns
        -------
        comparison : dict
            Comparison statistics.
        """
        if metric == "c_index":
            metric_func = SurvivalMetrics.concordance_index
        elif metric == "auc":
            metric_func = lambda ye, yt, p: np.mean(
                list(SurvivalMetrics.time_dependent_auc(ye, yt, p).values())
            )
        else:
            raise ValueError(f"Unknown metric: {metric}")

        score1 = metric_func(y_event, y_time, predictions_model1)
        score2 = metric_func(y_event, y_time, predictions_model2)

        # Bootstrap CI for difference
        n_bootstraps = 1000
        n_samples = len(y_time)
        diffs = []

        for _ in range(n_bootstraps):
            idx = np.random.choice(n_samples, size=n_samples, replace=True)
            y_event_boot = y_event[idx]
            y_time_boot = y_time[idx]
            pred1_boot = predictions_model1[idx]
            pred2_boot = predictions_model2[idx]

            try:
                s1 = metric_func(y_event_boot, y_time_boot, pred1_boot)
                s2 = metric_func(y_event_boot, y_time_boot, pred2_boot)
                diffs.append(s1 - s2)
            except Exception:
                pass

        diffs = np.array(diffs)
        diffs = diffs[~np.isnan(diffs)]

        # P-value: proportion of bootstrap differences crossing zero
        p_value = np.mean(diffs * np.sign(score1 - score2) < 0) if len(diffs) > 0 else np.nan

        return {
            "model1_score": score1,
            "model2_score": score2,
            "difference": score1 - score2,
            "difference_ci_lower": np.percentile(diffs, 2.5),
            "difference_ci_upper": np.percentile(diffs, 97.5),
            "p_value": p_value,
            "winner": "Model 1" if score1 > score2 else ("Model 2" if score2 > score1 else "Tie"),
        }

    @staticmethod
    def evaluate_summary(
        y_event: np.ndarray,
        y_time: np.ndarray,
        predictions: np.ndarray,
        survival_probs: Optional[np.ndarray] = None,
        subgroups: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Comprehensive evaluation summary.

        Parameters
        ----------
        y_event : ndarray
            Event indicator.
        y_time : ndarray
            Survival time.
        predictions : ndarray
            Predicted risk scores.
        survival_probs : ndarray, optional
            Predicted survival probabilities.
        subgroups : ndarray, optional
            Subgroup labels.

        Returns
        -------
        summary : DataFrame
            Evaluation summary with all metrics.
        """
        results = {}

        # C-index
        results["c_index"] = SurvivalMetrics.concordance_index(y_event, y_time, predictions)

        # Uno's C-index (IPCW)
        results["unos_c_index"] = SurvivalMetrics.unos_concordance_index(y_event, y_time, predictions)

        # Time-dependent AUC
        auc_dict = SurvivalMetrics.time_dependent_auc(y_event, y_time, predictions)
        results["mean_auc"] = np.mean(list(auc_dict.values()))

        # Calibration
        cal_metrics = SurvivalMetrics.calibration_metrics(y_event, y_time, predictions)
        results.update(cal_metrics)

        # IBS if survival probs provided
        if survival_probs is not None:
            results["ibs"] = SurvivalMetrics.integrated_brier_score(
                y_event, y_time, survival_probs
            )

        # Subgroup robustness if provided
        if subgroups is not None:
            subgroup_dict = SurvivalMetrics.subgroup_robustness(
                y_event, y_time, predictions, subgroups
            )
            results["subgroup_c_index_mean"] = np.nanmean(list(subgroup_dict.values()))
            results["subgroup_c_index_std"] = np.nanstd(list(subgroup_dict.values()))

        return pd.DataFrame([results])
