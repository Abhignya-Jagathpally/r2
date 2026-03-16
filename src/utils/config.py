"""
Configuration management utilities using OmegaConf and Pydantic

Provides utilities for loading, validating, and managing pipeline configurations
with support for environment variable override and dynamic schema validation.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field, field_validator, ConfigDict

# ============================================================================
# PYDANTIC MODELS FOR CONFIGURATION VALIDATION
# ============================================================================


class DatasetConfig(BaseModel):
    """Configuration for a single dataset"""

    name: str
    identifier: str
    type: str  # microarray or rnaseq
    platform: str
    samples: int
    source: str
    accession: Optional[str] = None
    processing_notes: Optional[str] = None

    model_config = ConfigDict(str_strip_whitespace=True)


class PreprocessingConfig(BaseModel):
    """Preprocessing configuration"""

    probe_mapping: Dict[str, Any]
    array: Dict[str, Any]
    rnaseq: Dict[str, Any]
    low_expression_filter: Dict[str, Any]
    quality_control: Dict[str, Any]
    scaling: Dict[str, Any]

    model_config = ConfigDict(str_strip_whitespace=True)


class PathwayConfig(BaseModel):
    """Pathway analysis configuration"""

    databases: Dict[str, Any]
    methods: Dict[str, Any]
    filtering: Dict[str, Any]

    model_config = ConfigDict(str_strip_whitespace=True)


class ModelConfig(BaseModel):
    """Model configuration"""

    target: str
    event_column: str
    time_column: str
    baselines: List[Dict[str, Any]]
    modern: List[Dict[str, Any]]
    fusion: List[Dict[str, Any]]

    model_config = ConfigDict(str_strip_whitespace=True)


class EvaluationConfig(BaseModel):
    """Evaluation configuration"""

    cross_validation: Dict[str, Any]
    metrics: Dict[str, Any]
    risk_stratification: Dict[str, Any]
    validation: Dict[str, Any]

    model_config = ConfigDict(str_strip_whitespace=True)


class ExperimentConfig(BaseModel):
    """Experiment tracking configuration"""

    hyperparameter_optimization: Dict[str, Any]
    early_stopping: Dict[str, Any]
    tracking: Dict[str, Any]
    reproducibility: Dict[str, Any]

    model_config = ConfigDict(str_strip_whitespace=True)


class MLflowConfig(BaseModel):
    """MLflow configuration"""

    tracking_uri: str
    backend_store_uri: str
    artifact_root: str
    experiment: Dict[str, Any]
    model_registry: Dict[str, Any]

    model_config = ConfigDict(str_strip_whitespace=True)


class PipelineConfig(BaseModel):
    """Main pipeline configuration"""

    pipeline: Dict[str, str]
    datasets: Dict[str, Any]
    data_directories: Dict[str, str]
    preprocessing: PreprocessingConfig
    pathway: PathwayConfig
    modeling: ModelConfig
    evaluation: EvaluationConfig
    experiment: ExperimentConfig
    mlflow: MLflowConfig
    logging: Dict[str, Any]
    output: Dict[str, str]
    compute: Dict[str, Any]
    slurm: Optional[Dict[str, Any]] = None
    wandb: Optional[Dict[str, Any]] = None
    dvc: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(str_strip_whitespace=True, extra="allow")

    @field_validator("pipeline")
    @classmethod
    def validate_pipeline(cls, v: Dict[str, str]) -> Dict[str, str]:
        """Validate required pipeline fields"""
        required = ["name", "version", "description"]
        for field in required:
            if field not in v:
                raise ValueError(f"Missing required pipeline field: {field}")
        return v

    @field_validator("datasets")
    @classmethod
    def validate_datasets(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate datasets configuration"""
        if "training_studies" not in v or not v["training_studies"]:
            raise ValueError("Must define at least one training study")
        return v


# ============================================================================
# CONFIGURATION LOADER
# ============================================================================


class ConfigLoader:
    """
    Load and validate pipeline configurations

    Supports YAML files, environment variable overrides, and Pydantic validation.
    """

    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize configuration loader

        Parameters
        ----------
        config_path : str or Path
            Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config_dict: Optional[Dict[str, Any]] = None
        self.config: Optional[DictConfig] = None
        self.validated_config: Optional[PipelineConfig] = None

    def load(self) -> DictConfig:
        """
        Load configuration from YAML file

        Returns
        -------
        DictConfig
            Configuration as OmegaConf DictConfig

        Raises
        ------
        FileNotFoundError
            If config file doesn't exist
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            self.config_dict = yaml.safe_load(f)

        # Convert to OmegaConf for structured access
        self.config = OmegaConf.create(self.config_dict)

        return self.config

    def override_with_env(self) -> DictConfig:
        """
        Override configuration values with environment variables

        Environment variables should be prefixed with 'PIPELINE_' and use
        double underscores to denote nested keys.

        Examples
        --------
        - PIPELINE_PREPROCESSING__NORMALIZE_METHOD="loess"
        - PIPELINE_COMPUTE__N_JOBS=8
        - PIPELINE_MLflow__TRACKING_URI="http://remote:5000"

        Returns
        -------
        DictConfig
            Configuration with environment overrides applied
        """
        prefix = "PIPELINE_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and convert to lowercase
                config_key = key[len(prefix):].lower()
                # Convert double underscores to nested keys
                parts = config_key.split("__")

                # Attempt to set in config
                try:
                    current = self.config
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    current[parts[-1]] = self._parse_value(value)
                except Exception as e:
                    print(f"Warning: Failed to set config key {config_key}: {e}")

        return self.config

    def validate(self) -> PipelineConfig:
        """
        Validate configuration against Pydantic schema

        Returns
        -------
        PipelineConfig
            Validated configuration object

        Raises
        ------
        ValueError
            If configuration doesn't match schema
        """
        if self.config is None:
            raise RuntimeError("Configuration not loaded. Call load() first.")

        # Convert DictConfig to dict for Pydantic
        config_dict = OmegaConf.to_container(self.config, resolve=True)

        try:
            self.validated_config = PipelineConfig(**config_dict)
            return self.validated_config
        except Exception as e:
            raise ValueError(f"Configuration validation failed: {e}")

    def get(
        self,
        key: str,
        default: Any = None,
        required: bool = False,
    ) -> Any:
        """
        Get configuration value by key

        Supports dot-notation for nested access.

        Parameters
        ----------
        key : str
            Configuration key (e.g., 'preprocessing.normalize_method')
        default : Any, optional
            Default value if key not found
        required : bool, default False
            Raise error if key not found

        Returns
        -------
        Any
            Configuration value

        Raises
        ------
        KeyError
            If key not found and required=True
        """
        if self.config is None:
            raise RuntimeError("Configuration not loaded. Call load() first.")

        try:
            value = OmegaConf.select(self.config, key)
            if value is None and required:
                raise KeyError(f"Required configuration key not found: {key}")
            return value if value is not None else default
        except Exception as e:
            if required:
                raise KeyError(f"Error accessing configuration key '{key}': {e}")
            return default

    def save(self, output_path: Union[str, Path]) -> None:
        """
        Save current configuration to file

        Parameters
        ----------
        output_path : str or Path
            Path to save configuration
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            OmegaConf.save(self.config, f)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary

        Returns
        -------
        dict
            Configuration as dictionary
        """
        if self.config is None:
            raise RuntimeError("Configuration not loaded. Call load() first.")

        return OmegaConf.to_container(self.config, resolve=True)

    @staticmethod
    def _parse_value(value: str) -> Any:
        """
        Parse environment variable value to appropriate type

        Parameters
        ----------
        value : str
            String value from environment variable

        Returns
        -------
        Any
            Parsed value (bool, int, float, or str)
        """
        # Try boolean
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False

        # Try integer
        try:
            return int(value)
        except ValueError:
            pass

        # Try float
        try:
            return float(value)
        except ValueError:
            pass

        # Return as string
        return value


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def load_config(
    config_path: Union[str, Path],
    validate: bool = True,
    env_override: bool = True,
) -> Union[DictConfig, PipelineConfig]:
    """
    Convenience function to load and validate configuration

    Parameters
    ----------
    config_path : str or Path
        Path to configuration file
    validate : bool, default True
        Whether to validate against Pydantic schema
    env_override : bool, default True
        Whether to apply environment variable overrides

    Returns
    -------
    DictConfig or PipelineConfig
        Loaded configuration

    Examples
    --------
    >>> config = load_config("config/pipeline_config.yaml")
    >>> normalize_method = config.preprocessing.array.normalization_method
    """
    loader = ConfigLoader(config_path)
    loader.load()

    if env_override:
        loader.override_with_env()

    if validate:
        return loader.validate()

    return loader.config


def get_config_value(
    config: Union[DictConfig, PipelineConfig],
    key: str,
    default: Any = None,
) -> Any:
    """
    Get value from configuration using dot notation

    Parameters
    ----------
    config : DictConfig or PipelineConfig
        Configuration object
    key : str
        Configuration key with dot notation
    default : Any, optional
        Default value if key not found

    Returns
    -------
    Any
        Configuration value
    """
    if isinstance(config, PipelineConfig):
        # Convert Pydantic model to dict first
        config = OmegaConf.create(config.model_dump())

    try:
        value = OmegaConf.select(config, key)
        return value if value is not None else default
    except Exception:
        return default


def merge_configs(
    base_config: Union[str, Path, DictConfig],
    override_config: Union[str, Path, DictConfig],
) -> DictConfig:
    """
    Merge two configurations with override taking precedence

    Parameters
    ----------
    base_config : str, Path, or DictConfig
        Base configuration
    override_config : str, Path, or DictConfig
        Override configuration

    Returns
    -------
    DictConfig
        Merged configuration
    """
    # Load base config if path
    if isinstance(base_config, (str, Path)):
        base = OmegaConf.load(base_config)
    else:
        base = base_config

    # Load override config if path
    if isinstance(override_config, (str, Path)):
        override = OmegaConf.load(override_config)
    else:
        override = override_config

    # Merge with override taking precedence
    return OmegaConf.merge(base, override)


# ============================================================================
# CONTEXT MANAGER FOR TEMPORARY CONFIG OVERRIDES
# ============================================================================


class TemporaryConfigOverride:
    """Context manager for temporary configuration overrides"""

    def __init__(
        self,
        config: DictConfig,
        overrides: Dict[str, Any],
    ):
        """
        Initialize temporary override context

        Parameters
        ----------
        config : DictConfig
            Configuration to override
        overrides : dict
            Dictionary of key-value overrides
        """
        self.config = config
        self.overrides = overrides
        self.original_values: Dict[str, Any] = {}

    def __enter__(self) -> DictConfig:
        """Enter context and apply overrides"""
        for key, value in self.overrides.items():
            self.original_values[key] = OmegaConf.select(self.config, key)
            OmegaConf.update(self.config, key, value, merge=False)
        return self.config

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context and restore original values"""
        for key, value in self.original_values.items():
            if value is None:
                OmegaConf.update(self.config, key, None, merge=False)
            else:
                OmegaConf.update(self.config, key, value, merge=False)
