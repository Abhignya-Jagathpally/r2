#!/usr/bin/env python3
"""
Train Fusion/Ensemble Models

Trains multi-modal fusion models combining baseline and modern predictions:
- Late fusion (weighted average of model outputs)
- Stacking (meta-learner over base model predictions)
- Attention-based fusion (learned attention weights)
- Multimodal attention (fuses different data modalities)

Loads pretrained baseline and modern models, trains fusion layer.
Uses patient-level cross-validation.

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
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.fusion.late_fusion import LateFusionModel
from src.models.fusion.multimodal_attention import MultimodalAttentionModel
from src.evaluation.splits import PatientLevelSplitter
from src.evaluation.metrics import SurvivalMetrics

logger = logging.getLogger(__name__)


def setup_logging(log_dir: Path, level: str = "INFO") -> None:
    """Configure logging."""
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "train_fusion.log"),
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

    # Load and concatenate
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


def load_baseline_models(baseline_dir: Path) -> Dict:
    """
    Load trained baseline models.

    Parameters
    ----------
    baseline_dir : Path
        Directory containing baseline model artifacts

    Returns
    -------
    dict
        Dictionary mapping model names to loaded models
    """
    logger.info(f"Loading baseline models from {baseline_dir}...")

    models = {}
    model_files = list(baseline_dir.glob("*_model.pkl"))

    if not model_files:
        logger.warning(f"No baseline models found in {baseline_dir}")
        return models

    try:
        import joblib

        for model_file in model_files:
            model_name = model_file.stem.replace("_model", "")
            model = joblib.load(model_file)
            models[model_name] = model
            logger.info(f"  Loaded {model_name}")

    except Exception as e:
        logger.error(f"Failed to load baseline models: {e}")

    return models


def load_modern_models(modern_dir: Path) -> Dict:
    """
    Load trained modern models.

    Parameters
    ----------
    modern_dir : Path
        Directory containing modern model artifacts

    Returns
    -------
    dict
        Dictionary mapping model names to loaded models
    """
    logger.info(f"Loading modern models from {modern_dir}...")

    models = {}
    model_files = list(modern_dir.glob("*_model.pt"))

    if not model_files:
        logger.warning(f"No modern models found in {modern_dir}")
        return models

    # Note: Loading PyTorch models requires specific implementation
    # This is a placeholder for the actual loading logic
    logger.info(f"Found {len(model_files)} modern model(s)")

    return models


def collect_base_predictions(
    baseline_models: Dict,
    modern_models: Dict,
    X: pd.DataFrame,
    y_time: np.ndarray,
    y_event: np.ndarray,
) -> pd.DataFrame:
    """
    Generate predictions from all base models for fusion training.

    Parameters
    ----------
    baseline_models : dict
        Baseline model objects
    modern_models : dict
        Modern model objects
    X : pd.DataFrame
        Feature matrix
    y_time : np.ndarray
        Survival times
    y_event : np.ndarray
        Event indicators

    Returns
    -------
    pd.DataFrame
        DataFrame with predictions from all base models
    """
    logger.info("Collecting base model predictions...")

    predictions = pd.DataFrame(index=X.index)

    # Get baseline predictions
    for model_name, model in tqdm(
        baseline_models.items(),
        desc="Baseline predictions",
        leave=False,
    ):
        try:
            pred = model.predict_risk(X)
            predictions[f"{model_name}_risk"] = pred
        except Exception as e:
            logger.warning(f"Could not predict with {model_name}: {e}")

    # Get modern predictions
    for model_name, model in tqdm(
        modern_models.items(),
        desc="Modern predictions",
        leave=False,
    ):
        try:
            pred = model.predict_risk(X)
            predictions[f"{model_name}_risk"] = pred
        except Exception as e:
            logger.warning(f"Could not predict with {model_name}: {e}")

    logger.info(f"Collected {predictions.shape[1]} base model predictions")

    return predictions


def train_fusion_model(
    fusion_type: str,
    X: pd.DataFrame,
    y_time: np.ndarray,
    y_event: np.ndarray,
    baseline_models: Dict,
    modern_models: Dict,
    config: Dict,
    output_dir: Path,
    n_outer_folds: int = 5,
) -> Dict:
    """
    Train fusion model.

    Parameters
    ----------
    fusion_type : str
        Type of fusion: 'late_fusion' or 'multimodal_attention'
    X : pd.DataFrame
        Feature matrix
    y_time : np.ndarray
        Survival times
    y_event : np.ndarray
        Event indicators
    baseline_models : dict
        Baseline models
    modern_models : dict
        Modern models
    config : dict
        Configuration
    output_dir : Path
        Output directory
    n_outer_folds : int
        Number of CV folds

    Returns
    -------
    dict
        Training results
    """
    logger.info(f"\nTraining {fusion_type} model...")

    results = {
        "fusion_type": fusion_type,
        "cv_metrics": [],
        "mean_c_index": 0,
        "std_c_index": 0,
    }

    try:
        with mlflow.start_run(run_name=fusion_type):
            mlflow.set_tag("model_type", "fusion")
            mlflow.set_tag("fusion_type", fusion_type)

            # Collect base predictions
            base_predictions = collect_base_predictions(
                baseline_models=baseline_models,
                modern_models=modern_models,
                X=X,
                y_time=y_time,
                y_event=y_event,
            )

            # Patient-level CV
            splitter = PatientLevelSplitter(
                n_splits=n_outer_folds,
                random_state=42,
            )

            metrics = SurvivalMetrics()
            cv_scores = []

            # Outer CV loop
            for fold_idx, (train_idx, test_idx) in enumerate(
                tqdm(
                    splitter.split(base_predictions),
                    total=n_outer_folds,
                    desc=f"Fusion CV ({fusion_type})",
                    leave=False,
                )
            ):
                X_train = base_predictions.iloc[train_idx]
                y_time_train = y_time[train_idx]
                y_event_train = y_event[train_idx]

                X_test = base_predictions.iloc[test_idx]
                y_time_test = y_time[test_idx]
                y_event_test = y_event[test_idx]

                # Select fusion model
                if fusion_type == "late_fusion":
                    model = LateFusionModel()
                elif fusion_type == "multimodal_attention":
                    model = MultimodalAttentionModel()
                else:
                    raise ValueError(f"Unknown fusion type: {fusion_type}")

                # Train fusion model
                model.fit(X_train, y_time_train, y_event_train)

                # Evaluate
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

            # Aggregate metrics
            results["mean_c_index"] = float(np.mean(cv_scores))
            results["std_c_index"] = float(np.std(cv_scores))

            logger.info(
                f"  Mean C-index: {results['mean_c_index']:.4f} "
                f"± {results['std_c_index']:.4f}"
            )

            # MLflow logging
            mlflow.log_metric("mean_c_index", results["mean_c_index"])
            mlflow.log_metric("std_c_index", results["std_c_index"])
            mlflow.log_metric("n_base_models", len(base_predictions.columns))

            # Save final model
            model_path = output_dir / f"{fusion_type}_model.pt"
            try:
                model.save(str(model_path))
                mlflow.log_artifact(str(model_path))
                logger.info(f"  Model saved: {model_path}")
            except Exception as e:
                logger.warning(f"Could not save model: {e}")

    except Exception as e:
        logger.error(f"Error training {fusion_type}: {e}", exc_info=True)
        results["error"] = str(e)

    return results


def main():
    """Main fusion model training orchestration."""
    parser = argparse.ArgumentParser(
        description="Train fusion/ensemble survival models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all fusion models
  python scripts/train_fusion.py \\
    --data-dir data/processed \\
    --baseline-dir outputs/models \\
    --modern-dir outputs/models \\
    --output-dir outputs/models

  # Train specific fusion type
  python scripts/train_fusion.py \\
    --data-dir data/processed \\
    --baseline-dir outputs/models \\
    --modern-dir outputs/models \\
    --output-dir outputs/models \\
    --fusion-types late_fusion
        """,
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory with preprocessed data (default: data/processed)",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path("outputs/models"),
        help="Directory with baseline models (default: outputs/models)",
    )
    parser.add_argument(
        "--modern-dir",
        type=Path,
        default=Path("outputs/models"),
        help="Directory with modern models (default: outputs/models)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/models"),
        help="Output directory for fusion models (default: outputs/models)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/pipeline_config.yaml"),
        help="Configuration file (default: config/pipeline_config.yaml)",
    )
    parser.add_argument(
        "--fusion-types",
        nargs="+",
        default=None,
        help="Fusion types to train. Options: late_fusion, multimodal_attention",
    )
    parser.add_argument(
        "--n-outer-folds",
        type=int,
        default=5,
        help="Number of outer CV folds (default: 5)",
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
    logger.info("MM Transcriptomics Fusion Model Training")
    logger.info("="*80)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Baseline models: {args.baseline_dir}")
    logger.info(f"Modern models: {args.modern_dir}")
    logger.info(f"Output directory: {args.output_dir}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    config = load_config(args.config)

    # Setup MLflow
    try:
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment("mm_risk_signature_fusion")
        logger.info(f"MLflow tracking URI: {args.mlflow_uri}")
    except Exception as e:
        logger.warning(f"Could not connect to MLflow: {e}")

    # Load data
    try:
        X, y_time, y_event = load_preprocessed_data(args.data_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    # Load base models
    baseline_models = load_baseline_models(args.baseline_dir)
    modern_models = load_modern_models(args.modern_dir)

    if not baseline_models and not modern_models:
        logger.warning("No base models found. Train baseline/modern models first.")

    # Define fusion types
    all_fusion_types = ["late_fusion", "multimodal_attention"]

    # Select fusion types
    if args.fusion_types:
        fusion_types = [f for f in args.fusion_types if f in all_fusion_types]
    else:
        fusion_types = all_fusion_types

    if not fusion_types:
        logger.error(f"No valid fusion types. Options: {all_fusion_types}")
        sys.exit(1)

    logger.info(f"\nTraining {len(fusion_types)} fusion model(s)")

    # Train fusion models
    training_results = {}
    for fusion_type in tqdm(fusion_types, desc="Training fusion models", unit="model"):
        results = train_fusion_model(
            fusion_type=fusion_type,
            X=X,
            y_time=y_time.values,
            y_event=y_event.values,
            baseline_models=baseline_models,
            modern_models=modern_models,
            config=config,
            output_dir=args.output_dir,
            n_outer_folds=args.n_outer_folds,
        )

        training_results[fusion_type] = results

    # Save results
    results_file = args.output_dir / "fusion_training_summary.json"
    try:
        with open(results_file, "w") as f:
            json.dump(training_results, f, indent=2)
        logger.info(f"\nResults saved to {results_file}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")

    # Summary
    logger.info("\n" + "="*80)
    logger.info("Fusion Training Summary")
    logger.info("="*80)

    summary_df = pd.DataFrame(
        [
            {
                "Fusion Type": name,
                "Mean C-index": results.get("mean_c_index", np.nan),
                "Std C-index": results.get("std_c_index", np.nan),
            }
            for name, results in training_results.items()
        ]
    )

    summary_df = summary_df.sort_values("Mean C-index", ascending=False)
    logger.info("\n" + summary_df.to_string(index=False))

    logger.info("\n" + "="*80)
    logger.info("Fusion Model Training Complete!")
    logger.info("="*80)
    logger.info(f"Models saved to: {args.output_dir}")
    logger.info("Next step: Run 'make evaluate' for cross-study validation")


if __name__ == "__main__":
    main()
