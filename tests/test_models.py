"""
Unit tests for survival models and evaluation pipeline.

Tests use synthetic survival data with known properties.
"""

import numpy as np
import pandas as pd
import pytest
from sksurv.util import Surv

from src.models.baselines.base_model import BaseSurvivalModel, SurvivalModelCV
from src.models.baselines.de_enrichment import DEEnrichmentBaseline
from src.models.baselines.elastic_net_cox import ElasticNetCoxModel
from src.models.baselines.lasso_cox import LassoCoxModel
from src.models.baselines.random_survival_forest import RandomSurvivalForestModel
from src.models.baselines.sparse_group_lasso_cox import SparseGroupLassoCoxModel
from src.evaluation.metrics import SurvivalMetrics
from src.evaluation.splits import (
    LeaveOneStudyOutSplitter,
    NestedCVSplitter,
    PatientLevelSplitter,
    StratifiedTimeAwareSplitter,
    TimeAwareSplitter,
)


@pytest.fixture
def synthetic_survival_data():
    """Generate synthetic survival data."""
    np.random.seed(42)
    n_samples = 200
    n_features = 50

    # Feature matrix
    X = np.random.randn(n_samples, n_features)

    # Generate survival times and events
    # Linear risk: eta = X @ beta
    true_beta = np.zeros(n_features)
    true_beta[:5] = [0.5, -0.3, 0.4, -0.2, 0.3]  # Only 5 features matter

    eta = X @ true_beta
    # Weibull survival
    lambda_param = 0.1 * np.exp(eta)
    k = 1.5
    U = np.random.uniform(0, 1, n_samples)
    y_time = (-np.log(U) / lambda_param) ** (1 / k)

    # Censoring
    censoring_rate = 0.3
    y_event = np.random.binomial(1, 1 - censoring_rate, n_samples)

    # Cap times
    y_time = np.minimum(y_time, 10.0)

    return {
        "X": X,
        "y_time": y_time,
        "y_event": y_event,
        "feature_names": [f"feature_{i}" for i in range(n_features)],
    }


@pytest.fixture
def synthetic_grouped_data(synthetic_survival_data):
    """Create grouped/pathway structure."""
    data = synthetic_survival_data.copy()
    n_features = data["X"].shape[1]

    # Create 5 pathway groups
    groups = {}
    features_per_group = n_features // 5
    for i in range(5):
        start = i * features_per_group
        end = start + features_per_group if i < 4 else n_features
        groups[f"pathway_{i}"] = list(range(start, end))

    data["groups"] = groups
    return data


# ============================================================================
# Base Model Tests
# ============================================================================


class TestBaseSurvivalModel:
    """Test abstract base model."""

    def test_input_validation(self, synthetic_survival_data):
        """Test input validation."""
        data = synthetic_survival_data
        model = DEEnrichmentBaseline()

        X_array, y_time_array, y_event_array = model._validate_input(
            data["X"], data["y_time"], data["y_event"]
        )

        assert X_array.shape == data["X"].shape
        assert len(y_time_array) == len(data["y_time"])
        assert len(y_event_array) == len(data["y_event"])

    def test_scaler(self, synthetic_survival_data):
        """Test feature scaling."""
        data = synthetic_survival_data
        model = DEEnrichmentBaseline()

        X_scaled = model._fit_scaler(data["X"])
        assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-10)
        assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-10)

        X_transformed = model._transform_scaler(data["X"])
        assert X_transformed.shape == data["X"].shape


# ============================================================================
# Model Fit and Predict Tests
# ============================================================================


class TestDEEnrichmentBaseline:
    """Test DE enrichment baseline."""

    def test_fit_predict(self, synthetic_survival_data):
        """Test fitting and prediction."""
        data = synthetic_survival_data
        model = DEEnrichmentBaseline(n_pathways=10)

        model.fit(data["X"], data["y_time"], data["y_event"])

        assert model.coef_ is not None
        assert model.feature_names_ == data["feature_names"]

        # Predict
        predictions = model.predict_risk(data["X"])
        assert len(predictions) == len(data["X"])
        assert np.all(predictions >= 0)  # Risk scores should be non-negative

        # Survival function
        surv_probs = model.predict_survival_function(data["X"][:10])
        assert surv_probs.shape[0] == 10
        assert np.all((surv_probs >= 0) & (surv_probs <= 1))

    def test_selected_pathways(self, synthetic_survival_data):
        """Test pathway selection."""
        data = synthetic_survival_data
        model = DEEnrichmentBaseline(n_pathways=20)

        model.fit(data["X"], data["y_time"], data["y_event"])

        assert len(model.selected_pathways_) <= 20
        assert all(isinstance(idx, (int, np.integer)) for idx in model.selected_pathways_)


class TestLassoCoxModel:
    """Test Lasso Cox model."""

    def test_fit_predict(self, synthetic_survival_data):
        """Test Lasso Cox fit and predict."""
        data = synthetic_survival_data
        model = LassoCoxModel(alpha=0.1, l1_ratio=1.0)

        model.fit(data["X"], data["y_time"], data["y_event"])

        predictions = model.predict_risk(data["X"])
        assert len(predictions) == len(data["X"])
        assert np.all(np.isfinite(predictions))

        # Test survival function
        surv_probs = model.predict_survival_function(data["X"][:10])
        assert surv_probs.shape[0] == 10

    def test_nested_cv(self, synthetic_survival_data):
        """Test nested CV lambda selection."""
        data = synthetic_survival_data
        model = LassoCoxModel()

        best_alpha, cv_score, cv_scores = model.fit_nested_cv(
            data["X"], data["y_time"], data["y_event"],
            alpha_grid=np.logspace(-2, 1, 3),
            cv_inner=2,
            cv_outer=2,
        )

        assert best_alpha > 0
        assert 0 <= cv_score <= 1
        assert len(cv_scores) == 2

    def test_feature_selection(self, synthetic_survival_data):
        """Test feature selection."""
        data = synthetic_survival_data
        model = LassoCoxModel(alpha=1.0)

        model.fit(data["X"], data["y_time"], data["y_event"])

        selected = model.get_selected_features(threshold=0.01)
        assert len(selected) <= 50  # Can't select more than n_features


class TestElasticNetCoxModel:
    """Test Elastic Net Cox model."""

    def test_fit_predict(self, synthetic_survival_data):
        """Test Elastic Net fit and predict."""
        data = synthetic_survival_data
        model = ElasticNetCoxModel(alpha=0.1, l1_ratio=0.5)

        model.fit(data["X"], data["y_time"], data["y_event"])

        predictions = model.predict_risk(data["X"])
        assert len(predictions) == len(data["X"])

    def test_grid_search(self, synthetic_survival_data):
        """Test grid search for hyperparameters."""
        data = synthetic_survival_data
        model = ElasticNetCoxModel()

        best_alpha, best_l1_ratio, results = model.grid_search_cv(
            data["X"], data["y_time"], data["y_event"],
            alpha_grid=np.logspace(-1, 1, 3),
            l1_ratio_grid=[0.3, 0.7],
            cv=2,
        )

        assert best_alpha > 0
        assert 0 <= best_l1_ratio <= 1
        assert len(results["alpha"]) == 6  # 3 alphas x 2 l1_ratios


class TestSparseGroupLassoCoxModel:
    """Test sparse group lasso Cox model."""

    def test_fit_predict(self, synthetic_grouped_data):
        """Test sparse group lasso fit and predict."""
        data = synthetic_grouped_data
        model = SparseGroupLassoCoxModel(
            lambda1=0.1,
            lambda2=0.1,
            groups=data["groups"],
        )

        model.fit(data["X"], data["y_time"], data["y_event"], groups=data["groups"])

        predictions = model.predict_risk(data["X"])
        assert len(predictions) == len(data["X"])

        # Test pathway selection
        pathways = model.get_selected_pathways()
        assert len(pathways) >= 0
        assert "pathway" in pathways["pathway"].values[0] if len(pathways) > 0 else True

    def test_nested_cv_lambda(self, synthetic_grouped_data):
        """Test nested CV for lambda selection."""
        data = synthetic_grouped_data
        model = SparseGroupLassoCoxModel(groups=data["groups"])

        best_l1, best_l2, cv_score, cv_scores = model.nested_cv_lambda_selection(
            data["X"], data["y_time"], data["y_event"],
            lambda1_grid=np.logspace(-2, 0, 2),
            lambda2_grid=np.logspace(-2, 0, 2),
            cv_inner=2,
            cv_outer=2,
        )

        assert best_l1 > 0
        assert best_l2 > 0
        assert 0 <= cv_score <= 1


class TestRandomSurvivalForest:
    """Test Random Survival Forest."""

    def test_fit_predict(self, synthetic_survival_data):
        """Test RSF fit and predict."""
        data = synthetic_survival_data
        model = RandomSurvivalForestModel(n_estimators=10)

        model.fit(data["X"], data["y_time"], data["y_event"])

        predictions = model.predict_risk(data["X"])
        assert len(predictions) == len(data["X"])

    def test_feature_importance(self, synthetic_survival_data):
        """Test feature importance."""
        data = synthetic_survival_data
        model = RandomSurvivalForestModel(n_estimators=10)

        model.fit(data["X"], data["y_time"], data["y_event"])

        importance = model.feature_importance_
        assert len(importance) == 50  # n_features
        assert all(v >= 0 for v in importance.values())

    def test_grid_search(self, synthetic_survival_data):
        """Test grid search."""
        data = synthetic_survival_data
        model = RandomSurvivalForestModel()

        best_params, best_score = model.grid_search_cv(
            data["X"], data["y_time"], data["y_event"],
            param_grid={
                "n_estimators": [5, 10],
                "max_depth": [5, 10],
            },
            cv=2,
        )

        assert best_params["n_estimators"] in [5, 10]
        assert best_score > 0


# ============================================================================
# Cross-Validation Tests
# ============================================================================


class TestPatientLevelSplitter:
    """Test patient-level splitting."""

    def test_basic_split(self, synthetic_survival_data):
        """Test basic patient-level split."""
        data = synthetic_survival_data
        splitter = PatientLevelSplitter(n_splits=3)

        splits = splitter.split(data["X"], data["y_event"])

        assert len(splits) == 3
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            assert len(np.intersect1d(train_idx, test_idx)) == 0  # No overlap

    def test_stratified_split(self, synthetic_survival_data):
        """Test stratified split."""
        data = synthetic_survival_data
        stratify_df = pd.DataFrame({
            "stage": np.random.choice([1, 2, 3], len(data["X"])),
        })

        splitter = PatientLevelSplitter(
            n_splits=3,
            stratify_by=stratify_df,
        )

        splits = splitter.split(data["X"], data["y_event"])
        assert len(splits) == 3


class TestTimeAwareSplitter:
    """Test time-aware splitting."""

    def test_split_by_time_windows(self, synthetic_survival_data):
        """Test time window splits."""
        data = synthetic_survival_data
        splitter = TimeAwareSplitter(
            time_windows=[(0, 3), (3, 6), (6, 10)],
        )

        splits = splitter.split(data["X"], data["y_time"])

        assert len(splits) > 0
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0


class TestLeaveOneStudyOutSplitter:
    """Test LOSO splitter."""

    def test_loso_split(self, synthetic_survival_data):
        """Test LOSO splitting."""
        data = synthetic_survival_data
        study_ids = np.random.choice(["study1", "study2", "study3"], len(data["X"]))

        splitter = LeaveOneStudyOutSplitter()
        splits = splitter.split(data["X"], study_ids)

        assert len(splits) == 3  # 3 studies


class TestNestedCVSplitter:
    """Test nested CV splitter."""

    def test_nested_cv(self, synthetic_survival_data):
        """Test nested CV splits."""
        data = synthetic_survival_data
        splitter = NestedCVSplitter(n_splits_outer=2, n_splits_inner=2)

        outer_splits = splitter.split(data["X"], data["y_event"])

        assert len(outer_splits) == 2
        for train_idx, test_idx in outer_splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0


# ============================================================================
# Evaluation Metrics Tests
# ============================================================================


class TestSurvivalMetrics:
    """Test evaluation metrics."""

    def test_concordance_index(self, synthetic_survival_data):
        """Test C-index computation."""
        data = synthetic_survival_data
        predictions = np.random.rand(len(data["X"]))

        c_index = SurvivalMetrics.concordance_index(
            data["y_event"],
            data["y_time"],
            predictions,
        )

        assert 0 <= c_index <= 1

    def test_time_dependent_auc(self, synthetic_survival_data):
        """Test time-dependent AUC."""
        data = synthetic_survival_data
        predictions = np.random.rand(len(data["X"]))

        auc_dict = SurvivalMetrics.time_dependent_auc(
            data["y_event"],
            data["y_time"],
            predictions,
        )

        assert len(auc_dict) > 0
        for t, auc in auc_dict.items():
            assert 0 <= auc <= 1 or np.isnan(auc)

    def test_calibration_metrics(self, synthetic_survival_data):
        """Test calibration metrics."""
        data = synthetic_survival_data
        predictions = np.random.rand(len(data["X"]))

        cal_metrics = SurvivalMetrics.calibration_metrics(
            data["y_event"],
            data["y_time"],
            predictions,
        )

        assert "calibration_slope" in cal_metrics
        assert "calibration_intercept" in cal_metrics
        assert "d_calibration" in cal_metrics

    def test_subgroup_robustness(self, synthetic_survival_data):
        """Test subgroup robustness."""
        data = synthetic_survival_data
        predictions = np.random.rand(len(data["X"]))
        subgroups = np.random.choice([1, 2, 3], len(data["X"]))

        subgroup_scores = SurvivalMetrics.subgroup_robustness(
            data["y_event"],
            data["y_time"],
            predictions,
            subgroups,
        )

        assert len(subgroup_scores) == 3

    def test_bootstrap_ci(self, synthetic_survival_data):
        """Test bootstrap confidence intervals."""
        data = synthetic_survival_data
        predictions = np.random.rand(len(data["X"]))

        point_est, ci_lower, ci_upper = SurvivalMetrics.bootstrap_ci(
            data["y_event"],
            data["y_time"],
            predictions,
            SurvivalMetrics.concordance_index,
            n_bootstraps=10,
        )

        assert 0 <= point_est <= 1
        assert ci_lower <= point_est <= ci_upper

    def test_pairwise_comparison(self, synthetic_survival_data):
        """Test pairwise model comparison."""
        data = synthetic_survival_data
        pred1 = np.random.rand(len(data["X"]))
        pred2 = np.random.rand(len(data["X"]))

        comparison = SurvivalMetrics.pairwise_model_comparison(
            data["y_event"],
            data["y_time"],
            pred1,
            pred2,
        )

        assert "model1_score" in comparison
        assert "model2_score" in comparison
        assert "winner" in comparison


# ============================================================================
# Integration Tests
# ============================================================================


class TestCVWrapper:
    """Test cross-validation wrapper."""

    def test_cv_fit_predict(self, synthetic_survival_data):
        """Test CV fit and predict."""
        data = synthetic_survival_data
        model = DEEnrichmentBaseline()
        cv_wrapper = SurvivalModelCV(model, n_splits=2)

        y_pred_train, y_pred_test = cv_wrapper.fit_predict(
            data["X"],
            data["y_time"],
            data["y_event"],
        )

        assert len(y_pred_train) == len(data["X"])
        assert len(y_pred_test) == len(data["X"])


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
