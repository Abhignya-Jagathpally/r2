#!/usr/bin/env python3
"""
Train Modern Deep Learning Models

Trains state-of-the-art neural network survival models:
- Pathway Autoencoder (unsupervised feature learning)
- Domain Adversarial Network (cross-study adaptation)
- TabPFN Classifier (transformer-based)
- DeepSurv (Cox partial likelihood neural network)

Uses autoresearch agent for hyperparameter optimization with fixed budget.
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
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.modern.pathway_autoencoder import PathwayAutoencoder
from src.models.modern.domain_adversarial import DomainAdversarialNetwork
from src.models.modern.tabpfn_classifier import TabPFNClassifier
from src.models.modern.deepsurv import DeepSurvivalModel
from src.models.modern.autoresearch_agent import AutoresearchAgent
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
            logging.FileHandler(log_dir / "train_modern.log"),
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


def hyperparameter_search(
    model_name: str,
    X: pd.DataFrame,
    y_time: np.ndarray,
    y_event: np.ndarray,
    config: Dict,
    budget: int = 100,
) -> Dict:
    """
    Run hyperparameter search using autoresearch agent.

    Parameters
    ----------
    model_name : str
        Model name
    X : pd.DataFrame
        Feature matrix
    y_time : np.ndarray
        Survival times
    y_event : np.ndarray
        Event indicators
    config : dict
        Configuration
    budget : int
        Number of configurations to evaluate

    Returns
    -------
    dict
        Best configuration and metrics
    """
    logger.info(f"\nHyperparameter search for {model_name}...")

    try:
        agent = AutoresearchAgent(
            model_name=model_name,
            fixed_budget=budget,
            config=config,
        )

        best_config, best_score = agent.search(
            X=X,
            y_time=y_time,
            y_event=y_event,
        )

        logger.info(f"  Best score: {best_score:.4f}")
        logger.info(f"  Best config: {best_config}")

        return {
            "best_config": best_config,
            "best_score": float(best_score),
            "budget_used": budget,
        }

    except Exception as e:
        logger.error(f"Hyperparameter search failed for {model_name}: {e}", exc_info=True)
        return {"error": str(e), "budget_used": 0}


def train_model(
    model_name: str,
    X: pd.DataFrame,
    y_time: np.ndarray,
    y_event: np.ndarray,
    config: Dict,
    output_dir: Path,
    n_outer_folds: int = 5,
    use_hpo: bool = True,
) -> Dict:
    """
    Train modern neural network model with optional HPO.

    Parameters
    ----------
    model_name : str
        Model name
    X : pd.DataFrame
        Feature matrix
    y_time : np.ndarray
        Survival times
    y_event : np.ndarray
        Event indicators
    config : dict
        Configuration
    output_dir : Path
        Output directory
    n_outer_folds : int
        Number of CV folds
    use_hpo : bool
        Use hyperparameter optimization

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
        with mlflow.start_run(run_name=model_name):
            mlflow.set_tag("model_type", "modern")
            mlflow.set_tag("model_name", model_name)

            # Hyperparameter optimization
            hpo_results = None
            if use_hpo:
                hpo_budget = config.get("experiment", {}).get("hyperparameter_optimization", {}).get("fixed_search_budget", 50)
                hpo_results = hyperparameter_search(
                    model_name=model_name,
                    X=X,
                    y_time=y_time,
                    y_event=y_event,
                    config=config,
                    budget=hpo_budget,
                )

                if "error" not in hpo_results:
                    mlflow.log_dict(hpo_results["best_config"], "hpo_best_config.json")
                    mlflow.log_metric("hpo_best_score", hpo_results["best_score"])

            # Select model class
            model_classes = {
                "pathway_autoencoder": PathwayAutoencoder,
                "domain_adversarial": DomainAdversarialNetwork,
                "tabpfn": TabPFNClassifier,
                "deepsurv": DeepSurvivalModel,
            }

            model_class = model_classes.get(model_name.lower())
            if not model_class:
                raise ValueError(f"Unknown model: {model_name}")

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
                    splitter.split(X),
                    total=n_outer_folds,
                    desc=f"Outer CV ({model_name})",
                    leave=False,
                )
            ):
                X_train = X.iloc[train_idx]
                y_time_train = y_time[train_idx]
                y_event_train = y_event[train_idx]

                X_test = X.iloc[test_idx]
                y_time_test = y_time[test_idx]
                y_event_test = y_event[test_idx]

                # Initialize and train
                model = model_class()

                # Use HPO config if available
                if hpo_results and "best_config" in hpo_results:
                    model.set_params(**hpo_results["best_config"])

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

            # Save final model
            model_path = output_dir / f"{model_name}_model.pt"
            try:
                model.save(str(model_path))
                mlflow.log_artifact(str(model_path))
                logger.info(f"  Model saved: {model_path}")
            except Exception as e:
                logger.warning(f"Could not save model: {e}")

    except Exception as e:
        logger.error(f"Error training {model_name}: {e}", exc_info=True)
        results["error"] = str(e)

    return results


def main():
    """Main modern model training orchestration."""
    parser = argparse.ArgumentParser(
        description="Train modern deep learning survival models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all modern models with HPO
  python scripts/train_modern.py --input-dir data/processed --output-dir outputs/models

  # Train specific models
  python scripts/train_modern.py \\
    --input-dir data/processed \\
    --output-dir outputs/models \\
    --models pathway_autoencoder deepsurv

  # Skip HPO for faster training
  python scripts/train_modern.py \\
    --input-dir data/processed \\
    --output-dir outputs/models \\
    --skip-hpo
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
        help="Models to train. Options: pathway_autoencoder, domain_adversarial, "
        "tabpfn, deepsurv",
    )
    parser.add_argument(
        "--skip-hpo",
        action="store_true",
        help="Skip hyperparameter optimization",
    )
    parser.add_argument(
        "--hpo-budget",
        type=int,
        default=100,
        help="HPO evaluation budget (default: 100)",
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
    logger.info("MM Transcriptomics Modern Model Training")
    logger.info("="*80)
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"HPO enabled: {not args.skip_hpo}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    config = load_config(args.config)

    # Setup MLflow
    try:
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment("mm_risk_signature_modern")
        logger.info(f"MLflow tracking URI: {args.mlflow_uri}")
    except Exception as e:
        logger.warning(f"Could not connect to MLflow: {e}")

    # Load data
    try:
        X, y_time, y_event = load_preprocessed_data(args.input_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    # Define models
    all_models = [
        "pathway_autoencoder",
        "domain_adversarial",
        "tabpfn",
        "deepsurv",
    ]

    # Select models
    if args.models:
        models_to_train = [m for m in args.models if m in all_models]
    else:
        models_to_train = all_models

    if not models_to_train:
        logger.error(f"No valid models. Options: {all_models}")
        sys.exit(1)

    logger.info(f"\nTraining {len(models_to_train)} model(s)")

    # Train models
    training_results = {}
    for model_name in tqdm(models_to_train, desc="Training modern models", unit="model"):
        results = train_model(
            model_name=model_name,
            X=X,
            y_time=y_time.values,
            y_event=y_event.values,
            config=config,
            output_dir=args.output_dir,
            n_outer_folds=args.n_outer_folds,
            use_hpo=not args.skip_hpo,
        )

        training_results[model_name] = results

    # Save results
    results_file = args.output_dir / "modern_training_summary.json"
    try:
        with open(results_file, "w") as f:
            json.dump(training_results, f, indent=2)
        logger.info(f"\nResults saved to {results_file}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")

    # Summary
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
    logger.info("Modern Model Training Complete!")
    logger.info("="*80)
    logger.info(f"Models saved to: {args.output_dir}")
    logger.info("Next step: Run 'make train-fusion' to train fusion models")


if __name__ == "__main__":
    main()
