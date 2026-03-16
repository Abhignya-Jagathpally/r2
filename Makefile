.PHONY: help setup install install-dev clean download-data preprocess train-baselines train-modern train-fusion evaluate report docker-build docker-push docker-run lint test coverage docs mlflow-start mlflow-ui slurm-submit logs

# ============================================================================
# VARIABLES
# ============================================================================

SHELL := /bin/bash
PYTHON := python3
PIP := pip
CONDA := conda
DOCKER := docker
APPTAINER := apptainer

# Environment
VENV_NAME ?= venv
PYTHON_VERSION := 3.11
CONDA_ENV := mmpipeline

# Directories
DATA_RAW := data/raw
DATA_PROCESSED := data/processed
MODELS_DIR := outputs/models
RESULTS_DIR := outputs/results
REPORTS_DIR := outputs/reports
LOGS_DIR := logs
MLRUNS_DIR := mlruns

# Docker
DOCKER_IMAGE_NAME := mm-transcriptomics-risk-signature
DOCKER_REGISTRY := docker.io
DOCKER_TAG := latest

# Nextflow
NEXTFLOW := nextflow
NEXTFLOW_PROFILE := local
NEXTFLOW_CONFIG := workflows/nextflow/nextflow.config

# Snakemake
SNAKEMAKE := snakemake
SNAKEMAKE_CORES := 4
SNAKEMAKE_JOBS := 4

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# ============================================================================
# HELP
# ============================================================================

help:
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║  MM Transcriptomics Risk Signature Pipeline                         ║$(NC)"
	@echo "$(BLUE)║  Makefile Commands                                                  ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(YELLOW)Setup & Installation:$(NC)"
	@echo "  $(GREEN)make setup$(NC)              Setup development environment (conda/venv)"
	@echo "  $(GREEN)make install$(NC)            Install package in editable mode"
	@echo "  $(GREEN)make install-dev$(NC)        Install with development dependencies"
	@echo ""
	@echo "$(YELLOW)Data & Preprocessing:$(NC)"
	@echo "  $(GREEN)make download-data$(NC)      Download GEO datasets"
	@echo "  $(GREEN)make preprocess$(NC)         Run preprocessing on all datasets"
	@echo ""
	@echo "$(YELLOW)Model Training:$(NC)"
	@echo "  $(GREEN)make train-baselines$(NC)    Train baseline models (CoxPH, KM, RF)"
	@echo "  $(GREEN)make train-modern$(NC)       Train modern models (XGBoost, CatBoost)"
	@echo "  $(GREEN)make train-fusion$(NC)       Train fusion models (DeepSurv, DeepHit)"
	@echo "  $(GREEN)make train-all$(NC)          Train all models"
	@echo ""
	@echo "$(YELLOW)Evaluation & Reporting:$(NC)"
	@echo "  $(GREEN)make evaluate$(NC)           Run cross-study evaluation"
	@echo "  $(GREEN)make report$(NC)             Generate final HTML report"
	@echo ""
	@echo "$(YELLOW)Docker & Containers:$(NC)"
	@echo "  $(GREEN)make docker-build$(NC)       Build Docker images (all stages)"
	@echo "  $(GREEN)make docker-push$(NC)        Push Docker images to registry"
	@echo "  $(GREEN)make docker-run$(NC)         Run pipeline in Docker container"
	@echo "  $(GREEN)make apptainer-build$(NC)    Build Apptainer image for HPC"
	@echo ""
	@echo "$(YELLOW)Workflow Orchestration:$(NC)"
	@echo "  $(GREEN)make nextflow-run$(NC)       Run pipeline with Nextflow"
	@echo "  $(GREEN)make snakemake-run$(NC)      Run pipeline with Snakemake"
	@echo "  $(GREEN)make slurm-submit$(NC)       Submit pipeline to SLURM cluster"
	@echo ""
	@echo "$(YELLOW)Code Quality & Testing:$(NC)"
	@echo "  $(GREEN)make lint$(NC)               Run code linting (black, isort, flake8)"
	@echo "  $(GREEN)make format$(NC)             Auto-format code (black, isort)"
	@echo "  $(GREEN)make test$(NC)               Run unit tests"
	@echo "  $(GREEN)make test-integration$(NC)   Run integration tests"
	@echo "  $(GREEN)make coverage$(NC)           Generate test coverage report"
	@echo ""
	@echo "$(YELLOW)Documentation & Logging:$(NC)"
	@echo "  $(GREEN)make docs$(NC)               Build Sphinx documentation"
	@echo "  $(GREEN)make mlflow-start$(NC)       Start MLflow tracking server"
	@echo "  $(GREEN)make mlflow-ui$(NC)          Open MLflow UI in browser"
	@echo "  $(GREEN)make logs$(NC)               View pipeline logs"
	@echo ""
	@echo "$(YELLOW)Cleanup:$(NC)"
	@echo "  $(GREEN)make clean$(NC)              Remove generated files"
	@echo "  $(GREEN)make clean-all$(NC)          Remove all including raw data"
	@echo "  $(GREEN)make clean-docker$(NC)       Remove Docker images"
	@echo ""

# ============================================================================
# SETUP & INSTALLATION
# ============================================================================

setup:
	@echo "$(BLUE)Setting up development environment...$(NC)"
	@command -v conda >/dev/null 2>&1 && \
		echo "$(GREEN)✓ Conda found$(NC)" && \
		$(CONDA) create -n $(CONDA_ENV) python=$(PYTHON_VERSION) -y && \
		$(CONDA) run -n $(CONDA_ENV) pip install --upgrade pip || \
		(echo "$(YELLOW)Conda not found, creating venv...$(NC)" && \
		$(PYTHON) -m venv $(VENV_NAME) && \
		source $(VENV_NAME)/bin/activate && \
		pip install --upgrade pip)
	@echo "$(GREEN)✓ Environment setup complete$(NC)"

install:
	@echo "$(BLUE)Installing pipeline...$(NC)"
	$(PIP) install -e .
	@echo "$(GREEN)✓ Installation complete$(NC)"

install-dev:
	@echo "$(BLUE)Installing pipeline with development dependencies...$(NC)"
	$(PIP) install -e ".[dev,docs]"
	@echo "$(GREEN)✓ Installation complete$(NC)"

# ============================================================================
# DATA & PREPROCESSING
# ============================================================================

download-data:
	@echo "$(BLUE)Downloading GEO datasets...$(NC)"
	@mkdir -p $(DATA_RAW) $(LOGS_DIR)
	$(PYTHON) scripts/download_geo_data.py --output-dir $(DATA_RAW) \
		2>&1 | tee $(LOGS_DIR)/download.log
	@echo "$(GREEN)✓ Data download complete$(NC)"

preprocess:
	@echo "$(BLUE)Running preprocessing...$(NC)"
	@mkdir -p $(DATA_PROCESSED) $(LOGS_DIR)
	$(PYTHON) scripts/preprocess_all.py \
		--input-dir $(DATA_RAW) \
		--output-dir $(DATA_PROCESSED) \
		--config config/pipeline_config.yaml \
		2>&1 | tee $(LOGS_DIR)/preprocess.log
	@echo "$(GREEN)✓ Preprocessing complete$(NC)"

# ============================================================================
# MODEL TRAINING
# ============================================================================

train-baselines:
	@echo "$(BLUE)Training baseline models...$(NC)"
	@mkdir -p $(MODELS_DIR) $(LOGS_DIR)
	$(PYTHON) scripts/train_baselines.py \
		--input-dir $(DATA_PROCESSED) \
		--output-dir $(MODELS_DIR) \
		--config config/pipeline_config.yaml \
		2>&1 | tee $(LOGS_DIR)/train_baselines.log
	@echo "$(GREEN)✓ Baseline training complete$(NC)"

train-modern:
	@echo "$(BLUE)Training modern models (XGBoost, CatBoost, RF)...$(NC)"
	@mkdir -p $(MODELS_DIR) $(LOGS_DIR)
	$(PYTHON) scripts/train_modern_models.py \
		--input-dir $(DATA_PROCESSED) \
		--output-dir $(MODELS_DIR) \
		--config config/pipeline_config.yaml \
		--n-jobs -1 \
		2>&1 | tee $(LOGS_DIR)/train_modern.log
	@echo "$(GREEN)✓ Modern model training complete$(NC)"

train-fusion:
	@echo "$(BLUE)Training fusion models (DeepSurv, DeepHit)...$(NC)"
	@mkdir -p $(MODELS_DIR) $(LOGS_DIR)
	$(PYTHON) scripts/train_fusion_models.py \
		--input-dir $(DATA_PROCESSED) \
		--output-dir $(MODELS_DIR) \
		--config config/pipeline_config.yaml \
		--gpu-device 0 \
		2>&1 | tee $(LOGS_DIR)/train_fusion.log
	@echo "$(GREEN)✓ Fusion model training complete$(NC)"

train-all: train-baselines train-modern train-fusion
	@echo "$(GREEN)✓ All models trained$(NC)"

# ============================================================================
# EVALUATION & REPORTING
# ============================================================================

evaluate:
	@echo "$(BLUE)Running cross-study evaluation...$(NC)"
	@mkdir -p $(RESULTS_DIR) $(LOGS_DIR)
	$(PYTHON) scripts/evaluate_cross_study.py \
		--models-dir $(MODELS_DIR) \
		--data-dir $(DATA_PROCESSED) \
		--output-dir $(RESULTS_DIR) \
		--config config/pipeline_config.yaml \
		2>&1 | tee $(LOGS_DIR)/evaluate.log
	@echo "$(GREEN)✓ Evaluation complete$(NC)"

report:
	@echo "$(BLUE)Generating final report...$(NC)"
	@mkdir -p $(REPORTS_DIR) $(LOGS_DIR)
	$(PYTHON) scripts/generate_report.py \
		--results-dir $(RESULTS_DIR) \
		--output-dir $(REPORTS_DIR) \
		--config config/pipeline_config.yaml \
		2>&1 | tee $(LOGS_DIR)/report.log
	@echo "$(GREEN)✓ Report generated: $(REPORTS_DIR)/index.html$(NC)"

# ============================================================================
# DOCKER OPERATIONS
# ============================================================================

docker-build:
	@echo "$(BLUE)Building Docker images...$(NC)"
	@mkdir -p $(LOGS_DIR)
	for stage in application development production; do \
		echo "Building $$stage..."; \
		$(DOCKER) build \
			--target $$stage \
			-t $(DOCKER_IMAGE_NAME):$$stage-$(DOCKER_TAG) \
			-f docker/Dockerfile . \
			2>&1 | tee $(LOGS_DIR)/docker_build_$$stage.log || exit 1; \
	done
	@echo "$(GREEN)✓ Docker build complete$(NC)"

docker-push:
	@echo "$(BLUE)Pushing Docker images to registry...$(NC)"
	@for stage in application development production; do \
		echo "Pushing $$stage..."; \
		$(DOCKER) push $(DOCKER_REGISTRY)/$(DOCKER_IMAGE_NAME):$$stage-$(DOCKER_TAG) || exit 1; \
	done
	@echo "$(GREEN)✓ Docker push complete$(NC)"

docker-run-dev:
	@echo "$(BLUE)Running pipeline in Docker (development)...$(NC)"
	$(DOCKER) run -it --rm \
		-v $(PWD):/app \
		-w /app \
		-e PYTHONUNBUFFERED=1 \
		$(DOCKER_IMAGE_NAME):development-$(DOCKER_TAG) \
		bash

docker-run-prod:
	@echo "$(BLUE)Running pipeline in Docker (production)...$(NC)"
	$(DOCKER) run -it --rm \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/outputs:/app/outputs \
		-p 5000:5000 \
		$(DOCKER_IMAGE_NAME):production-$(DOCKER_TAG)

apptainer-build:
	@echo "$(BLUE)Building Apptainer image for HPC...$(NC)"
	@mkdir -p $(LOGS_DIR)
	sudo $(APPTAINER) build mmpipeline.sif docker/apptainer.def \
		2>&1 | tee $(LOGS_DIR)/apptainer_build.log
	@echo "$(GREEN)✓ Apptainer build complete$(NC)"

# ============================================================================
# WORKFLOW ORCHESTRATION
# ============================================================================

nextflow-run:
	@echo "$(BLUE)Running pipeline with Nextflow...$(NC)"
	@mkdir -p $(LOGS_DIR)
	$(NEXTFLOW) run workflows/nextflow/main.nf \
		-profile $(NEXTFLOW_PROFILE) \
		-c $(NEXTFLOW_CONFIG) \
		-resume \
		2>&1 | tee $(LOGS_DIR)/nextflow.log
	@echo "$(GREEN)✓ Nextflow execution complete$(NC)"

nextflow-trace:
	@echo "$(BLUE)Viewing Nextflow execution trace...$(NC)"
	@find . -name "trace.txt" -mtime -1 -print -exec head -20 {} \;

snakemake-run:
	@echo "$(BLUE)Running pipeline with Snakemake...$(NC)"
	@mkdir -p $(LOGS_DIR)
	$(SNAKEMAKE) -s workflows/snakemake/Snakefile \
		-j $(SNAKEMAKE_JOBS) \
		--use-conda \
		--conda-frontend mamba \
		-p \
		2>&1 | tee $(LOGS_DIR)/snakemake.log
	@echo "$(GREEN)✓ Snakemake execution complete$(NC)"

snakemake-dryrun:
	@echo "$(BLUE)Snakemake dry-run (preview)...$(NC)"
	$(SNAKEMAKE) -s workflows/snakemake/Snakefile \
		-j $(SNAKEMAKE_JOBS) \
		-n -p

slurm-submit:
	@echo "$(BLUE)Submitting pipeline to SLURM...$(NC)"
	@mkdir -p $(LOGS_DIR)
	$(NEXTFLOW) run workflows/nextflow/main.nf \
		-profile slurm \
		-c $(NEXTFLOW_CONFIG) \
		2>&1 | tee $(LOGS_DIR)/slurm_submit.log
	@echo "$(GREEN)✓ SLURM job submitted$(NC)"

# ============================================================================
# CODE QUALITY & TESTING
# ============================================================================

lint:
	@echo "$(BLUE)Running code linting...$(NC)"
	black --check src tests
	isort --check-only src tests
	flake8 src tests --max-line-length=100
	@echo "$(GREEN)✓ Linting complete$(NC)"

format:
	@echo "$(BLUE)Auto-formatting code...$(NC)"
	black src tests
	isort src tests
	@echo "$(GREEN)✓ Formatting complete$(NC)"

test:
	@echo "$(BLUE)Running unit tests...$(NC)"
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing -m "not slow and not gpu"
	@echo "$(GREEN)✓ Tests complete$(NC)"

test-integration:
	@echo "$(BLUE)Running integration tests...$(NC)"
	pytest tests/ -v -m "integration" --tb=short
	@echo "$(GREEN)✓ Integration tests complete$(NC)"

coverage:
	@echo "$(BLUE)Generating coverage report...$(NC)"
	pytest tests/ \
		--cov=src \
		--cov-report=html \
		--cov-report=term-missing \
		--cov-report=xml
	@echo "$(GREEN)✓ Coverage report: htmlcov/index.html$(NC)"

# ============================================================================
# DOCUMENTATION
# ============================================================================

docs:
	@echo "$(BLUE)Building documentation...$(NC)"
	@mkdir -p $(LOGS_DIR)
	cd docs && make html 2>&1 | tee ../$(LOGS_DIR)/docs.log || \
		echo "$(YELLOW)Documentation build not configured$(NC)"
	@echo "$(GREEN)✓ Documentation: docs/_build/html/index.html$(NC)"

# ============================================================================
# MLFLOW OPERATIONS
# ============================================================================

mlflow-setup:
	@echo "$(BLUE)Setting up MLflow...$(NC)"
	$(PYTHON) config/mlflow_setup.py
	@echo "$(GREEN)✓ MLflow setup complete$(NC)"

mlflow-start:
	@echo "$(BLUE)Starting MLflow tracking server...$(NC)"
	@mkdir -p $(MLRUNS_DIR) $(LOGS_DIR)
	mlflow server \
		--backend-store-uri sqlite:///$(MLRUNS_DIR)/mlflow.db \
		--default-artifact-root ./$(MLRUNS_DIR)/artifacts \
		--host 0.0.0.0 \
		--port 5000 \
		2>&1 | tee $(LOGS_DIR)/mlflow.log &
	@echo "$(GREEN)✓ MLflow server started (PID: $$!)$(NC)"
	@echo "$(GREEN)Access UI at: http://localhost:5000$(NC)"

mlflow-ui:
	@echo "$(BLUE)Opening MLflow UI...$(NC)"
	@command -v xdg-open >/dev/null 2>&1 && \
		xdg-open http://localhost:5000 || \
		open http://localhost:5000 || \
		echo "$(YELLOW)Please open http://localhost:5000 in your browser$(NC)"

mlflow-stop:
	@echo "$(BLUE)Stopping MLflow server...$(NC)"
	pkill -f "mlflow server" || echo "$(YELLOW)No MLflow server found$(NC)"

# ============================================================================
# LOGGING & MONITORING
# ============================================================================

logs:
	@echo "$(BLUE)Recent pipeline logs:$(NC)"
	@tail -100 $(LOGS_DIR)/*.log 2>/dev/null | head -50 || \
		echo "$(YELLOW)No logs found$(NC)"

logs-watch:
	@echo "$(BLUE)Watching pipeline logs (Ctrl+C to stop)...$(NC)"
	watch -n 1 'ls -la $(LOGS_DIR)/'

watch-outputs:
	@echo "$(BLUE)Watching output directory...$(NC)"
	watch -n 2 'find $(RESULTS_DIR) -type f -mmin -5'

# ============================================================================
# CLEANUP
# ============================================================================

clean:
	@echo "$(BLUE)Cleaning generated files...$(NC)"
	rm -rf $(DATA_PROCESSED) $(MODELS_DIR) $(RESULTS_DIR) $(REPORTS_DIR)
	rm -rf work/ .snakemake/ .nextflow/ .nextflow.log* work/
	rm -rf .pytest_cache/ .coverage htmlcov/
	rm -rf **/__pycache__ **/*.pyc **/*.pyo
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

clean-all: clean
	@echo "$(BLUE)Removing all data...$(NC)"
	rm -rf $(DATA_RAW)
	@echo "$(GREEN)✓ Full cleanup complete$(NC)"

clean-docker:
	@echo "$(BLUE)Removing Docker images...$(NC)"
	$(DOCKER) rmi $(DOCKER_IMAGE_NAME):*-$(DOCKER_TAG) 2>/dev/null || true
	@echo "$(GREEN)✓ Docker cleanup complete$(NC)"

clean-cache:
	@echo "$(BLUE)Clearing cache directories...$(NC)"
	rm -rf .dvc/cache/ ~/.cache/pip ~/.cache/snakemake
	@echo "$(GREEN)✓ Cache cleanup complete$(NC)"

# ============================================================================
# DEVELOPMENT UTILITIES
# ============================================================================

.DEFAULT_GOAL := help
