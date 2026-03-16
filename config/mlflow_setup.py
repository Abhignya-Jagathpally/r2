"""
MLflow experiment setup and initialization script
Configures tracking URI, creates experiments, and sets up model registry
"""

import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import Param, Metric, RunTag

# ============================================================================
# CONFIGURATION
# ============================================================================

# MLflow tracking configuration
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
BACKEND_STORE_URI = os.getenv("MLFLOW_BACKEND_STORE_URI", "file:./mlruns")
ARTIFACT_ROOT = os.getenv("MLFLOW_ARTIFACT_ROOT", "./mlruns/artifacts")

# Experiment configuration
EXPERIMENT_NAME = "mm_risk_signature_v1"
EXPERIMENT_DESCRIPTION = "Bulk transcriptomics cross-study MM risk-signature pipeline"

# Model registry configuration
MODEL_NAME = "mm_risk_signature"
MODEL_DESCRIPTION = "Multiple myeloma risk signature model based on transcriptomics"
STAGES = ["Staging", "Production", "Archived"]

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MLflowSetup:
    """Initialize and configure MLflow for the pipeline"""

    def __init__(
        self,
        tracking_uri: str = TRACKING_URI,
        backend_store_uri: str = BACKEND_STORE_URI,
        artifact_root: str = ARTIFACT_ROOT,
    ):
        """
        Initialize MLflow setup

        Parameters
        ----------
        tracking_uri : str
            MLflow tracking server URI
        backend_store_uri : str
            Backend store URI for storing experiment metadata
        artifact_root : str
            Root directory for artifact storage
        """
        self.tracking_uri = tracking_uri
        self.backend_store_uri = backend_store_uri
        self.artifact_root = artifact_root
        self.client: Optional[MlflowClient] = None

    def setup_tracking(self) -> None:
        """Set up MLflow tracking backend"""
        logger.info(f"Setting up MLflow tracking URI: {self.tracking_uri}")

        # Create artifact directories if using local storage
        if self.artifact_root.startswith("file:"):
            artifact_path = self.artifact_root.replace("file:", "")
            Path(artifact_path).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created artifact directory: {artifact_path}")

        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(self.tracking_uri)

        logger.info("MLflow tracking setup complete")

    def create_experiment(self) -> str:
        """
        Create or get experiment

        Returns
        -------
        str
            Experiment ID
        """
        logger.info(f"Creating/getting experiment: {EXPERIMENT_NAME}")

        try:
            # Try to get existing experiment
            experiment = self.client.get_experiment_by_name(EXPERIMENT_NAME)
            if experiment:
                logger.info(f"Experiment exists with ID: {experiment.experiment_id}")
                return experiment.experiment_id
        except Exception as e:
            logger.debug(f"Experiment not found: {e}")

        # Create new experiment
        experiment_id = self.client.create_experiment(
            name=EXPERIMENT_NAME,
            artifact_location=self.artifact_root,
            tags={
                "description": EXPERIMENT_DESCRIPTION,
                "project": "mm_transcriptomics",
                "phase": "development",
                "version": "0.1.0",
            },
        )

        logger.info(f"Created experiment with ID: {experiment_id}")
        return experiment_id

    def setup_model_registry(self) -> None:
        """Setup model registry and registered model"""
        logger.info(f"Setting up model registry for: {MODEL_NAME}")

        try:
            # Check if model exists
            self.client.get_registered_model(MODEL_NAME)
            logger.info(f"Registered model exists: {MODEL_NAME}")
        except Exception as e:
            logger.info(f"Creating registered model: {MODEL_NAME}")
            self.client.create_registered_model(
                name=MODEL_NAME,
                tags={"description": MODEL_DESCRIPTION, "project": "mm_transcriptomics"},
            )

        logger.info("Model registry setup complete")

    def create_tags(self) -> Dict[str, str]:
        """
        Create standard tags for experiments

        Returns
        -------
        dict
            Tags dictionary
        """
        return {
            "project": "mm_transcriptomics",
            "pipeline": "risk_signature",
            "phase": "development",
            "version": "0.1.0",
            "environment": os.getenv("ENV", "local"),
        }

    def log_pipeline_config(
        self, experiment_id: str, config: Dict[str, Any]
    ) -> None:
        """
        Log pipeline configuration to MLflow

        Parameters
        ----------
        experiment_id : str
            MLflow experiment ID
        config : dict
            Pipeline configuration dictionary
        """
        logger.info("Logging pipeline configuration")

        mlflow.set_experiment_id(experiment_id)

        with mlflow.start_run(run_name="pipeline_config") as run:
            # Log configuration as parameters
            for key, value in self._flatten_dict(config).items():
                if isinstance(value, (str, int, float, bool)):
                    try:
                        mlflow.log_param(key, value)
                    except Exception as e:
                        logger.warning(f"Failed to log parameter {key}: {e}")

            # Log tags
            for key, value in self.create_tags().items():
                mlflow.set_tag(key, value)

            logger.info(f"Logged pipeline config in run: {run.info.run_id}")

    @staticmethod
    def _flatten_dict(
        d: Dict[str, Any], parent_key: str = "", sep: str = "_"
    ) -> Dict[str, Any]:
        """
        Flatten nested dictionary

        Parameters
        ----------
        d : dict
            Dictionary to flatten
        parent_key : str
            Parent key prefix
        sep : str
            Separator for nested keys

        Returns
        -------
        dict
            Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(MLflowSetup._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, (list, tuple)):
                # Convert lists to string representation
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)

    def setup(self, config: Optional[Dict[str, Any]] = None) -> str:
        """
        Complete MLflow setup

        Parameters
        ----------
        config : dict, optional
            Pipeline configuration to log

        Returns
        -------
        str
            Experiment ID
        """
        logger.info("Starting MLflow setup")

        # Setup tracking
        self.setup_tracking()

        # Create/get experiment
        experiment_id = self.create_experiment()

        # Setup model registry
        self.setup_model_registry()

        # Log configuration if provided
        if config:
            self.log_pipeline_config(experiment_id, config)

        logger.info(f"MLflow setup complete. Experiment ID: {experiment_id}")
        return experiment_id


def load_pipeline_config(config_path: str = "config/pipeline_config.yaml") -> Dict[str, Any]:
    """
    Load pipeline configuration from YAML file

    Parameters
    ----------
    config_path : str
        Path to configuration file

    Returns
    -------
    dict
        Configuration dictionary
    """
    import yaml

    logger.info(f"Loading configuration from: {config_path}")

    if not os.path.exists(config_path):
        logger.warning(f"Configuration file not found: {config_path}")
        return {}

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration with {len(config)} sections")
    return config


def main():
    """Main entry point for MLflow setup"""

    logger.info("=" * 80)
    logger.info("MLflow Setup Script")
    logger.info("=" * 80)

    try:
        # Load configuration
        config = load_pipeline_config()

        # Initialize MLflow setup
        mlflow_setup = MLflowSetup(
            tracking_uri=TRACKING_URI,
            backend_store_uri=BACKEND_STORE_URI,
            artifact_root=ARTIFACT_ROOT,
        )

        # Run setup
        experiment_id = mlflow_setup.setup(config=config)

        logger.info("=" * 80)
        logger.info("MLflow setup completed successfully!")
        logger.info(f"Tracking URI: {TRACKING_URI}")
        logger.info(f"Experiment: {EXPERIMENT_NAME} (ID: {experiment_id})")
        logger.info(f"Model Registry: {MODEL_NAME}")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"MLflow setup failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
