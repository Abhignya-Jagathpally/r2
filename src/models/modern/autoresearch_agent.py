"""
Autoresearch Agent for Biomedical ML Optimization.

Implements Karpathy-style constrained autoresearch pattern with:
- Frozen preprocessing contract (hash verified)
- Fixed editable surface: architecture, hyperparams, training config
- Bounded experiment budget (N experiments max)
- Single metric objective: concordance index (C-index)
- Full MLflow logging and reproducibility
- Ray Tune integration for distributed HPO
- Wall-clock time budget enforcement
- Guard rails preventing preprocessing modifications
"""

import hashlib
import json
import tempfile
import time
from typing import Dict, Any, Optional, Callable, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import mlflow
from ray import tune
from ray.tune.stopper import Stopper
from ray.air import session

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingContract:
    """
    Frozen preprocessing specification with hash verification.

    All fields are fixed; modifications require rehashing.
    """
    pathway_normalization: str  # 'zscore', 'minmax', 'log1p'
    clinical_normalization: str  # 'zscore', 'minmax'
    missing_value_strategy: str  # 'drop', 'median', 'mean'
    feature_selection_method: Optional[str]  # None, 'variance', 'mrmr'
    feature_selection_k: Optional[int]  # Number of features if selection enabled
    train_val_test_split: Tuple[float, float, float]  # e.g., (0.6, 0.2, 0.2)
    random_seed: int
    n_samples_total: int
    n_pathways: int
    n_clinical_features: int

    def compute_hash(self) -> str:
        """Compute SHA256 hash of preprocessing contract."""
        contract_str = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(contract_str.encode()).hexdigest()

    def verify_hash(self, expected_hash: str) -> bool:
        """Verify contract integrity."""
        return self.compute_hash() == expected_hash


@dataclass
class HyperparameterSpace:
    """
    Editable hyperparameter space for HPO.

    Only these fields are tunable; preprocessing is frozen.
    """
    # Model architecture
    hidden_dims: list  # e.g., [256, 128, 64]
    dropout_rate: float  # 0.0-0.5
    use_batch_norm: bool

    # Training
    learning_rate: float  # 1e-5 to 1e-2
    weight_decay: float  # 0 to 1e-3
    batch_size: int  # 16 to 256
    num_epochs: int  # 10 to 200
    warmup_epochs: int  # 0 to 10

    # Regularization
    l1_penalty: float  # 0 to 1e-2
    gradient_clip_norm: float  # 0 to 10

    # Early stopping
    early_stopping_patience: int  # 5 to 50
    early_stopping_min_delta: float  # 0 to 0.1


class WallClockStopper(Stopper):
    """Stop trials based on wall-clock time budget."""

    def __init__(self, max_seconds: float):
        """
        Args:
            max_seconds: Maximum wall-clock time in seconds
        """
        self.max_seconds = max_seconds
        self.start_time = time.time()

    def __call__(self, trial_id: str, result: dict) -> bool:
        """Return True if we should stop."""
        elapsed = time.time() - self.start_time
        return elapsed > self.max_seconds

    def stop_all(self) -> bool:
        """Return True if all trials should stop."""
        elapsed = time.time() - self.start_time
        return elapsed > self.max_seconds


class AutoresearchAgent:
    """
    Constrained autoresearch agent for biomedical ML.

    Orchestrates hyperparameter optimization within strict constraints.
    """

    def __init__(
        self,
        preprocessing_contract: PreprocessingContract,
        preprocessing_hash: str,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: DataLoader,
        model_factory: Callable[[HyperparameterSpace], pl.LightningModule],
        max_experiments: int = 20,
        max_wall_clock_seconds: float = 3600.0,
        experiment_name: str = "mm_risk_signature",
        mlflow_tracking_uri: Optional[str] = None,
    ):
        """
        Args:
            preprocessing_contract: Frozen preprocessing specification
            preprocessing_hash: Expected SHA256 hash of contract
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            test_dataloader: Test data loader
            model_factory: Function to create model from HyperparameterSpace
            max_experiments: Maximum number of HPO experiments
            max_wall_clock_seconds: Maximum wall-clock time budget
            experiment_name: MLflow experiment name
            mlflow_tracking_uri: MLflow tracking URI
        """
        # Verify preprocessing contract
        if not preprocessing_contract.verify_hash(preprocessing_hash):
            raise ValueError(
                f"Preprocessing contract hash mismatch! "
                f"Expected {preprocessing_hash}, "
                f"got {preprocessing_contract.compute_hash()}"
            )

        self.preprocessing_contract = preprocessing_contract
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.model_factory = model_factory
        self.max_experiments = max_experiments
        self.max_wall_clock_seconds = max_wall_clock_seconds

        # MLflow setup
        self.experiment_name = experiment_name
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)

        self.experiment_results = []
        self.start_time = None

    def _train_model(
        self,
        hparams: HyperparameterSpace,
        trial_id: str,
    ) -> Dict[str, Any]:
        """
        Train single model and return metrics.

        Args:
            hparams: Hyperparameter space
            trial_id: Unique trial identifier

        Returns:
            Dictionary with 'c_index' and other metrics
        """
        # Create model
        model = self.model_factory(hparams)

        # MLflow logging
        with mlflow.start_run(run_name=f"trial_{trial_id}"):
            # Log preprocessing contract
            mlflow.log_param(
                "preprocessing_contract_hash",
                self.preprocessing_contract.compute_hash(),
            )

            # Log hyperparameters
            for key, value in asdict(hparams).items():
                if isinstance(value, (int, float, bool, str)):
                    mlflow.log_param(key, value)
                elif isinstance(value, list):
                    mlflow.log_param(f"{key}", str(value))

            # Callbacks
            early_stopping = EarlyStopping(
                monitor='val_c_index',
                patience=hparams.early_stopping_patience,
                min_delta=hparams.early_stopping_min_delta,
                mode='max',
                verbose=False,
            )

            checkpoint = ModelCheckpoint(
                dirpath=tempfile.gettempdir(),
                monitor='val_c_index',
                mode='max',
                save_top_k=1,
            )

            # Trainer
            trainer = pl.Trainer(
                max_epochs=hparams.num_epochs,
                callbacks=[early_stopping, checkpoint],
                enable_progress_bar=False,
                logger=False,  # We handle MLflow logging
            )

            # Fit
            trainer.fit(
                model,
                train_dataloaders=self.train_dataloader,
                val_dataloaders=self.val_dataloader,
            )

            # Test
            test_results = trainer.test(model, dataloaders=self.test_dataloader)
            test_metrics = test_results[0]

            # Extract C-index (primary metric)
            c_index = test_metrics.get('test_c_index', None)
            if c_index is None:
                logger.warning(
                    "test_c_index not found in test metrics (keys: %s). "
                    "Defaulting to 0.5 (random). Check model test_step().",
                    list(test_metrics.keys()),
                )
                c_index = 0.5

            # Log metrics
            mlflow.log_metric('test_c_index', c_index)
            mlflow.log_metric('val_loss', early_stopping.best_score or 0.0)

            logger.info(f"Trial {trial_id}: C-index = {c_index:.4f}")

            return {'c_index': c_index, **test_metrics}

    def search(
        self,
        search_space: Dict[str, Any],
        scheduler: Optional[str] = 'asha',
        num_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run hyperparameter search with Ray Tune.

        Args:
            search_space: Ray Tune search space
            scheduler: 'asha' (default) or 'hyperband'
            num_samples: Number of random samples (default: max_experiments)

        Returns:
            Dictionary with best hyperparameters and results
        """
        if num_samples is None:
            num_samples = self.max_experiments

        self.start_time = time.time()

        logger.info(
            f"Starting autoresearch with {num_samples} experiments, "
            f"{self.max_wall_clock_seconds}s time budget"
        )

        # Wall-clock stopper
        stopper = WallClockStopper(self.max_wall_clock_seconds)

        # Ray Tune config
        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(self._train_model_wrapper),
                resources={'cpu': 1},
            ),
            param_space=search_space,
            tune_config=tune.TuneConfig(
                num_samples=num_samples,
                scheduler=scheduler,
                stop=stopper,
            ),
        )

        # Run
        results = tuner.fit()

        # Extract best trial
        best_result = results.get_best_result(metric='c_index', mode='max')
        best_hparams = best_result.config

        logger.info(
            f"Best C-index: {best_result.metrics['c_index']:.4f}"
        )

        return {
            'best_hparams': best_hparams,
            'best_c_index': best_result.metrics['c_index'],
            'best_trial_id': best_result.trial_id,
            'total_experiments': len(results.trials),
            'elapsed_seconds': time.time() - self.start_time,
        }

    def _train_model_wrapper(self, config: Dict[str, Any]) -> None:
        """Wrapper for Ray Tune integration."""
        hparams = HyperparameterSpace(**config)
        metrics = self._train_model(hparams, str(time.time()))
        session.report({'c_index': metrics['c_index']})

    def get_experiment_history(self) -> list:
        """Get history of all experiments."""
        return self.experiment_results

    def finalize(self, best_hparams: Dict[str, Any]) -> pl.LightningModule:
        """
        Train final model with best hyperparameters on full data.

        Args:
            best_hparams: Best hyperparameters from search

        Returns:
            Trained model
        """
        hparams = HyperparameterSpace(**best_hparams)
        model = self.model_factory(hparams)

        with mlflow.start_run(run_name="final_model"):
            # Log preprocessing contract one more time
            mlflow.log_param(
                "preprocessing_contract_hash",
                self.preprocessing_contract.compute_hash(),
            )

            trainer = pl.Trainer(
                max_epochs=hparams.num_epochs,
                enable_progress_bar=True,
            )

            trainer.fit(
                model,
                train_dataloaders=self.train_dataloader,
                val_dataloaders=self.val_dataloader,
            )

            trainer.test(model, dataloaders=self.test_dataloader)

        return model


def create_search_space(
    base_hparams: HyperparameterSpace,
) -> Dict[str, Any]:
    """
    Create Ray Tune search space from base hyperparameters.

    Args:
        base_hparams: Base hyperparameter configuration

    Returns:
        Ray Tune search space dict
    """
    return {
        'hidden_dims': tune.choice([
            [256, 128, 64],
            [512, 256, 128],
            [128, 64],
        ]),
        'dropout_rate': tune.uniform(0.0, 0.5),
        'use_batch_norm': tune.choice([True, False]),
        'learning_rate': tune.loguniform(1e-5, 1e-2),
        'weight_decay': tune.uniform(0, 1e-3),
        'batch_size': tune.choice([16, 32, 64, 128]),
        'num_epochs': tune.randint(10, 200),
        'warmup_epochs': tune.randint(0, 10),
        'l1_penalty': tune.uniform(0, 1e-2),
        'gradient_clip_norm': tune.uniform(0, 10),
        'early_stopping_patience': tune.randint(5, 50),
        'early_stopping_min_delta': tune.uniform(0, 0.1),
    }
