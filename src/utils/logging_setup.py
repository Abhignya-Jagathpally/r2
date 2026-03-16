"""
Logging setup and utilities with MLflow integration

Provides structured logging configuration with support for file handlers,
console output, and MLflow artifact logging.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import mlflow
from omegaconf import DictConfig, OmegaConf

# ============================================================================
# LOGGING CONFIGURATION CONSTANTS
# ============================================================================

LOG_FORMAT_STANDARD = (
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

LOG_FORMAT_DETAILED = (
    "%(asctime)s - %(name)s - %(levelname)s - "
    "[%(filename)s:%(lineno)d] - %(message)s"
)

LOG_FORMAT_SIMPLE = "%(levelname)s - %(message)s"

LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

# ============================================================================
# LOGGING SETUP
# ============================================================================


class LoggingSetup:
    """Configure structured logging with optional MLflow integration"""

    def __init__(
        self,
        name: str = "mm_pipeline",
        level: str = "INFO",
        format_type: str = "standard",
        log_dir: Union[str, Path] = "logs",
        log_file: str = "pipeline.log",
        use_mlflow: bool = True,
    ):
        """
        Initialize logging setup

        Parameters
        ----------
        name : str, default "mm_pipeline"
            Logger name
        level : str, default "INFO"
            Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type : str, default "standard"
            Log format type (simple, standard, detailed)
        log_dir : str or Path, default "logs"
            Directory for log files
        log_file : str, default "pipeline.log"
            Main log file name
        use_mlflow : bool, default True
            Whether to integrate with MLflow
        """
        self.name = name
        self.level = LOG_LEVELS.get(level.upper(), logging.INFO)
        self.format_type = format_type
        self.log_dir = Path(log_dir)
        self.log_file = log_file
        self.use_mlflow = use_mlflow
        self.logger: Optional[logging.Logger] = None
        self.handlers: List[logging.Handler] = []

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def setup(self) -> logging.Logger:
        """
        Setup logger with handlers

        Returns
        -------
        logging.Logger
            Configured logger
        """
        # Create logger
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(self.level)
        self.logger.propagate = False

        # Clear existing handlers
        self.logger.handlers.clear()

        # Create formatters
        formatter = self._get_formatter()

        # Add console handler
        self._add_console_handler(formatter)

        # Add file handler
        self._add_file_handler(formatter)

        # Add MLflow handler if enabled
        if self.use_mlflow:
            self._add_mlflow_handler()

        return self.logger

    def _get_formatter(self) -> logging.Formatter:
        """Get appropriate formatter based on format_type"""
        if self.format_type == "simple":
            fmt = LOG_FORMAT_SIMPLE
        elif self.format_type == "detailed":
            fmt = LOG_FORMAT_DETAILED
        else:
            fmt = LOG_FORMAT_STANDARD

        return logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    def _add_console_handler(self, formatter: logging.Formatter) -> None:
        """Add console (stdout) handler"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        self.handlers.append(console_handler)

    def _add_file_handler(self, formatter: logging.Formatter) -> None:
        """Add rotating file handler"""
        log_path = self.log_dir / self.log_file

        # Create rotating file handler (10MB default, keep 5 backups)
        file_handler = logging.handlers.RotatingFileHandler(
            str(log_path),
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
        )
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.handlers.append(file_handler)

    def _add_mlflow_handler(self) -> None:
        """Add MLflow handler for logging to experiments"""
        try:
            mlflow_handler = MLflowLoggingHandler(logger=self.logger)
            self.logger.addHandler(mlflow_handler)
            self.handlers.append(mlflow_handler)
        except Exception as e:
            self.logger.warning(f"Failed to add MLflow handler: {e}")

    def close(self) -> None:
        """Close all handlers"""
        for handler in self.handlers:
            handler.close()
        self.logger.handlers.clear()


class MLflowLoggingHandler(logging.Handler):
    """Custom logging handler that logs to MLflow"""

    def __init__(self, level=logging.INFO, logger: Optional[logging.Logger] = None):
        """
        Initialize MLflow logging handler

        Parameters
        ----------
        level : int, default logging.INFO
            Minimum log level to handle
        logger : logging.Logger, optional
            Logger instance to avoid circular logging
        """
        super().__init__(level)
        self.logger = logger
        self.buffer: List[str] = []
        self.buffer_size = 100  # Flush every 100 records

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record to MLflow

        Parameters
        ----------
        record : logging.LogRecord
            Log record to emit
        """
        try:
            # Format message
            msg = self.format(record)

            # Buffer messages for batch logging
            self.buffer.append(msg)

            # Flush buffer if size reached
            if len(self.buffer) >= self.buffer_size:
                self.flush_buffer()

            # Also log errors immediately
            if record.levelno >= logging.ERROR:
                self.flush_buffer()

        except Exception:
            self.handleError(record)

    def flush_buffer(self) -> None:
        """Flush buffered messages to MLflow"""
        if not self.buffer:
            return

        try:
            # Join messages and log to MLflow
            messages = "\n".join(self.buffer)
            if mlflow.active_run():
                mlflow.log_text(messages, artifact_file="pipeline.log")
        except Exception as e:
            if self.logger:
                self.logger.debug(f"Failed to log to MLflow: {e}")

        self.buffer.clear()

    def close(self) -> None:
        """Close handler and flush remaining messages"""
        self.flush_buffer()
        super().close()


# ============================================================================
# CONFIGURATION-BASED SETUP
# ============================================================================


def setup_logging_from_config(
    config: Union[DictConfig, Dict[str, Any]],
    name: str = "mm_pipeline",
) -> logging.Logger:
    """
    Setup logging from OmegaConf or dict configuration

    Parameters
    ----------
    config : DictConfig or dict
        Configuration dictionary with logging settings
    name : str, default "mm_pipeline"
        Logger name

    Returns
    -------
    logging.Logger
        Configured logger

    Examples
    --------
    >>> from src.utils.config import load_config
    >>> config = load_config("config/pipeline_config.yaml")
    >>> logger = setup_logging_from_config(config)
    """
    # Extract logging config
    if isinstance(config, DictConfig):
        log_config = OmegaConf.to_container(config.get("logging", {}))
    else:
        log_config = config.get("logging", {})

    # Setup logging
    setup = LoggingSetup(
        name=name,
        level=log_config.get("level", "INFO"),
        format_type="standard",
        log_dir=log_config.get("file", "logs/pipeline.log").rsplit("/", 1)[0],
        log_file=log_config.get("file", "logs/pipeline.log").rsplit("/", 1)[1],
        use_mlflow=True,
    )

    return setup.setup()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def get_logger(
    name: str = "mm_pipeline",
    level: str = "INFO",
) -> logging.Logger:
    """
    Get or create logger with basic configuration

    Parameters
    ----------
    name : str, default "mm_pipeline"
        Logger name
    level : str, default "INFO"
        Log level

    Returns
    -------
    logging.Logger
        Configured logger
    """
    logger = logging.getLogger(name)

    # Only configure if not already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(LOG_LEVELS.get(level.upper(), logging.INFO))

    return logger


def log_to_mlflow(
    log_dict: Dict[str, Any],
    prefix: str = "",
) -> None:
    """
    Log dictionary values to MLflow

    Parameters
    ----------
    log_dict : dict
        Dictionary of values to log
    prefix : str, optional
        Prefix for parameter names
    """
    if not mlflow.active_run():
        return

    for key, value in log_dict.items():
        param_name = f"{prefix}_{key}" if prefix else key

        try:
            if isinstance(value, (int, float, str, bool)):
                mlflow.log_param(param_name, value)
            elif isinstance(value, dict):
                # Recursively log nested dict
                log_to_mlflow(value, prefix=param_name)
            else:
                # Log as string representation
                mlflow.log_param(param_name, str(value))
        except Exception as e:
            logging.debug(f"Failed to log parameter {param_name}: {e}")


def log_metrics_batch(
    metrics: Dict[str, float],
    step: int,
) -> None:
    """
    Log multiple metrics to MLflow in a batch

    Parameters
    ----------
    metrics : dict
        Dictionary of metric_name: value pairs
    step : int
        Epoch/step number
    """
    if not mlflow.active_run():
        return

    for metric_name, value in metrics.items():
        try:
            if isinstance(value, (int, float)):
                mlflow.log_metric(metric_name, value, step=step)
        except Exception as e:
            logging.debug(f"Failed to log metric {metric_name}: {e}")


def log_artifacts(
    artifact_dir: Union[str, Path],
    artifact_path: Optional[str] = None,
) -> None:
    """
    Log artifacts directory to MLflow

    Parameters
    ----------
    artifact_dir : str or Path
        Directory containing artifacts
    artifact_path : str, optional
        MLflow artifact path (defaults to directory name)
    """
    if not mlflow.active_run():
        return

    artifact_dir = Path(artifact_dir)
    if not artifact_dir.exists():
        return

    if artifact_path is None:
        artifact_path = artifact_dir.name

    try:
        mlflow.log_artifacts(str(artifact_dir), artifact_path=artifact_path)
    except Exception as e:
        logging.debug(f"Failed to log artifacts from {artifact_dir}: {e}")


# ============================================================================
# CONTEXT MANAGERS FOR TEMPORARY LOGGING
# ============================================================================


class LogLevel:
    """Context manager to temporarily change log level"""

    def __init__(self, logger: logging.Logger, level: Union[int, str]):
        """
        Initialize log level context

        Parameters
        ----------
        logger : logging.Logger
            Logger instance
        level : int or str
            New log level
        """
        self.logger = logger
        self.new_level = (
            LOG_LEVELS.get(level.upper(), level)
            if isinstance(level, str)
            else level
        )
        self.previous_level = logger.level

    def __enter__(self):
        """Set new log level"""
        self.logger.setLevel(self.new_level)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore previous log level"""
        self.logger.setLevel(self.previous_level)


class MLflowLogger:
    """Context manager for MLflow run logging"""

    def __init__(
        self,
        experiment_name: str,
        run_name: str,
        tags: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize MLflow logger context

        Parameters
        ----------
        experiment_name : str
            MLflow experiment name
        run_name : str
            MLflow run name
        tags : dict, optional
            Tags to set for run
        params : dict, optional
            Parameters to log
        """
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tags = tags or {}
        self.params = params or {}
        self.run = None

    def __enter__(self):
        """Start MLflow run"""
        mlflow.set_experiment(self.experiment_name)
        self.run = mlflow.start_run(run_name=self.run_name)

        # Log tags and parameters
        for key, value in self.tags.items():
            mlflow.set_tag(key, value)

        for key, value in self.params.items():
            try:
                mlflow.log_param(key, value)
            except Exception:
                mlflow.log_param(key, str(value))

        return self.run

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End MLflow run"""
        if exc_type is not None:
            mlflow.set_tag("error", str(exc_val))

        mlflow.end_run()
