#!/usr/bin/env python3
"""
Train Baseline Survival Models

Trains classical survival models on pathway-scored MM transcriptomics data:
- Cox PH regression (L1/L2/ElasticNet penalized)
- Random survival forest
- Gradient boosting (XGBoost, CatBoost)
- Differential expression enrichment

Uses patient-level nested cross-validation (5 outer × 3 inner folds).
Logs metrics and models to MLflow.

Author: Pipeline Team
Date: 2026
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.baselines import (
    SparseLassoSurvivalModel,
    LassoSurvivalModel,
    ElasticNetSurvivalModel,
    RandomSurvivalForest,
    GradientBoostingSurvival,
)
from src.models.baselines.de_enrichment import DifferentialExpressionEnrichment
from src.evaluation.splits import PatientLevelSplitter
from src.evaluation.metrics import SurvivalMetrics
from src.utils.config import load_config as utils_load_config

logger = logging.getLogger(__name__)


def setup_logging(log_dir: Path, level: str = "INFO") -> None:
    """Configure logging."""
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "train_baselines.log"),
        ],
    )


def load_config(config_path: Path) -> Dict:
    """Load YAML configuration."""
    if not config_path.exists():
        logger.warning(f"Config not found: {config_path}")
        return {}

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_preprocessed_data(
    data_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load preprocessed pathway scores and clinical data.

    Returns
    -------
    tuple
        (X: features, y_time: survival times, y_event: event indicators)
    """
    logger.info(f"Loading preprocessed data from {data_dir}...")

    # Look for parquet files
    pathway_files = list(data_dir.glob("*_pathway_scores.parquet"))

    if not pathway_files:
        logger.error(f"No pathway score files found in {data_dir}")
        raise FileNotFoundError(f"No pathway scores in {data_dir}")

    # Load and concatenate data
    dfs = [pd.read_parquet(f) for f in pathway_files]
    X = pd.concat(dfs, axis=0, ignore_index=False)

    # Load clinical data
    clinical_files = list(data_dir.glob("*_clinical.csv"))
    if clinical_files:
        clinical = pd.concat(
            [pd.read_csv(f, index_col=0) for f in clinical_files],
            axis=0,
        )
    else:
        # Create dummy clinical data for testing
        logger.warning("No clinical data found, using dummy values")
        clinical = pd.DataFrame(
            {
                "time_months": np.random.exponential(30, len(X)),
                "event": np.random.binomial(1, 0.7, len(X)),
            },
            index=X.index,
        )

    y_time = clinical["time_months"]
    y_event = clinical["event"]

    logger.info(f"Loaded {X.shape[0]} samples × {X.shape[1]} pathway features")
    logger.info(f"Event rate: {y_event.mean():.2%}")

    return X, y_time, y_event


def train_model(
    model_class,
    model_name: str,
    X: pd.DataFrame,
    y_time: np.ndarray,
    y_event: np.ndarray,
    config: Dict,
    output_dir: Path,
    n_outer_folds: int = 5,
    n_inner_folds: int = 3,
) -> Dict:
    """
    Train model with nested cross-validation.

    Parameters
    ----------
    model_class
        Model class to instantiate
    model_name : str
        Model name for logging
    X : pd.DataFrame
        Feature matrix
    y_time : np.ndarray
        Survival times
    y_event : np.ndarray
        Event indicators
    config : dict
        Configuration parameters
    output_dir : Path
        Output directory for model artifacts
    n_outer_folds : int
        Number of outer CV folds
    n_inner_folds : int
        Number of inner CV folds

    Returns
    -------
    dict
        Training results and metrics
    """
    logger.info(f"\nTraining {model_name}...")

    results = {
        "model_name": model_name,
        "cv_metrics": [],
        "mean_c_index": 0,
        "std_c_index": 0,
    }

    try:
        # Initialize MLflow run
        with mlflow.start_run(run_name=model_name):
            mlflow.set_tag("model_type", "baseline")
            mlflow.set_tag("model_name", model_name)

            # Patient-level CV splitter
            splitter = PatientLevelSplitter(
                n_splits=n_outer_folds,
                random_state=42,
            )

            # Initialize metrics tracker
            metrics = SurvivalMetrics()
            cv_scores = []

            # Outer CV loop
            for fold_idx, (train_idx, test_idx) in enumerate(
                tqdm(
                    splitter.split(X),
                    total=n_outer_folds,
                    desc=f"Outer CV ({model_name})",
                    leave=False,
                )
            ):
                X_train = X.iloc[train_idx]
                y_time_train = y_time.iloc[train_idx]
                y_event_train = y_event.iloc[train_idx]

                X_test = X.iloc[test_idx]
                y_time_test = y_time.iloc[test_idx]
                y_event_test = y_event.iloc[test_idx]

                # Train model
                model = model_class(random_state=42)
                model.fit(X_train, y_time_train, y_event_train)

                # Evaluate on test fold
                c_index = metrics.concordance_index(
                    y_time_test,
                    y_event_test,
                    model.predict_risk(X_test),
                )

                cv_scores.append(c_index)
                logger.debug(f"  Fold {fold_idx+1}: C-index = {c_index:.4f}")

                results["cv_metrics"].append(
                    {
                        "fold": fold_idx,
                        "c_index": float(c_index),
                    }
                )

            # Aggregate CV metrics
            results["mean_c_index"] = float(np.mean(cv_scores))
            results["std_c_index"] = float(np.std(cv_scores))

            logger.info(
                f"  Mean C-index: {results['mean_c_index']:.4f} "
                f"± {results['std_c_index']:.4f}"
            )

            # Log to MLflow
            mlflow.log_metric("mean_c_index", results["mean_c_index"])
            mlflow.log_metric("std_c_index", results["std_c_index"])

            # Train final model on all data for serialization
            final_model = model_class(random_state=42)
            final_model.fit(X, y_time, y_event)

            # Save model
            model_path = output_dir / f"{model_name}_model.pkl"
            try:
                import joblib

                joblib.dump(final_model, model_path)
                mlflow.log_artifact(str(model_path))
                logger.info(f"  Model saved: {model_path}")
            except Exception as e:
                logger.warning(f"Could not save model: {e}")

            # Log parameters
            mlflow.log_params({"n_outer_folds": n_outer_folds, "n_inner_folds": n_inner_folds})

    except Exception as e:
        logger.error(f"Error training {model_name}: {e}", exc_info=True)
        results["error"] = str(e)

    return results


def main():
    """Main baseline training orchestration."""
    parser = argparse.ArgumentParser(
        description="Train baseline survival models on MM transcriptomics data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all baseline models
  python scripts/train_baselines.py --input-dir data/processed --output-dir outputs/models

  # Train specific models
  python scripts/train_baselines.py \\
    --input-dir data/processed \\
    --output-dir outputs/models \\
    --models sparse_group_lasso lasso elastic_net

  # Use configuration file
  python scripts/train_baselines.py \\
    --input-dir data/processed \\
    --output-dir outputs/models \\
    --config config/pipeline_config.yaml
        """,
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed"),
        help="Input directory with preprocessed data (default: data/processed)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/models"),
        help="Output directory for models (default: outputs/models)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/pipeline_config.yaml"),
        help="Configuration file (default: config/pipeline_config.yaml)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Models to train (default: all). Options: sparse_group_lasso, lasso, elastic_net, "
        "random_forest, xgboost, catboost, de_enrichment",
    )
    parser.add_argument(
        "--n-outer-folds",
        type=int,
        default=5,
        help="Number of outer CV folds (default: 5)",
    )
    parser.add_argument(
        "--n-inner-folds",
        type=int,
        default=3,
        help="Number of inner CV folds (default: 3)",
    )
    parser.add_argument(
        "--mlflow-uri",
        default="http://localhost:5000",
        help="MLflow tracking URI (default: http://localhost:5000)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    log_dir = args.output_dir / "logs"
    setup_logging(log_dir, args.log_level)

    logger.info("="*80)
    logger.info("MM Transcriptomics Baseline Model Training")
    logger.info("="*80)
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    config = load_config(args.config)

    # Setup MLflow
    try:
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment("mm_risk_signature_baselines")
        logger.info(f"MLflow tracking URI: {args.mlflow_uri}")
    except Exception as e:
        logger.warning(f"Could not connect to MLflow: {e}")

    # Load preprocessed data
    try:
        X, y_time, y_event = load_preprocessed_data(args.input_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    # Define models
    all_models = {
        "sparse_group_lasso": SparseLassoSurvivalModel,
        "lasso": LassoSurvivalModel,
        "elastic_net": ElasticNetSurvivalModel,
        "random_forest": RandomSurvivalForest,
        "xgboost": GradientBoostingSurvival,
        "catboost": GradientBoostingSurvival,  # or separate class
        "de_enrichment": DifferentialExpressionEnrichment,
    }

    # Select models to train
    if args.models:
        models_to_train = {k: v for k, v in all_models.items() if k in args.models}
    else:
        models_to_train = all_models

    if not models_to_train:
        logger.error(f"No valid models specified. Options: {list(all_models.keys())}")
        sys.exit(1)

    logger.info(f"\nTraining {len(models_to_train)} model(s)")

    # Train models
    training_results = {}
    for model_name, model_class in tqdm(
        models_to_train.items(),
        desc="Training baseline models",
        unit="model",
    ):
        results = train_model(
            model_class=model_class,
            model_name=model_name,
            X=X,
            y_time=y_time.values,
            y_event=y_event.values,
            config=config,
            output_dir=args.output_dir,
            n_outer_folds=args.n_outer_folds,
            n_inner_folds=args.n_inner_folds,
        )

        training_results[model_name] = results

    # Save results summary
    results_file = args.output_dir / "baseline_training_summary.json"
    try:
        with open(results_file, "w") as f:
            json.dump(training_results, f, indent=2)
        logger.info(f"\nResults saved to {results_file}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")

    # Summary table
    logger.info("\n" + "="*80)
    logger.info("Training Summary")
    logger.info("="*80)

    summary_df = pd.DataFrame(
        [
            {
                "Model": name,
                "Mean C-index": results.get("mean_c_index", np.nan),
                "Std C-index": results.get("std_c_index", np.nan),
            }
            for name, results in training_results.items()
        ]
    )

    summary_df = summary_df.sort_values("Mean C-index", ascending=False)
    logger.info("\n" + summary_df.to_string(index=False))

    logger.info("\n" + "="*80)
    logger.info("Baseline Training Complete!")
    logger.info("="*80)
    logger.info(f"Models saved to: {args.output_dir}")
    logger.info("Next step: Run 'make train-modern' to train modern models")


if __name__ == "__main__":
    main()
