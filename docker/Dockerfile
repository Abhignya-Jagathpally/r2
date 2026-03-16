# Multi-stage Dockerfile for MM transcriptomics risk-signature pipeline
# Stage 1: Base conda environment with R and essential packages
FROM continuumio/miniconda3:latest AS base

LABEL maintainer="PhD Researcher 1 <researcher@institution.edu>"
LABEL description="Bulk transcriptomics cross-study MM risk-signature pipeline"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    CONDA_AUTO_UPDATE_CONDA=false \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/opt/conda/bin:$PATH

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    git \
    wget \
    curl \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Create conda environment with core Python and R packages
RUN conda config --system --prepend channels conda-forge && \
    conda config --system --set auto_update_conda false && \
    conda create -y -q -n mmpipeline \
    python=3.11 \
    numpy \
    scipy \
    pandas \
    scikit-learn \
    matplotlib \
    seaborn \
    r-base \
    r-essentials \
    && conda clean --all -f -y

# Activate environment
SHELL ["conda", "run", "-n", "mmpipeline", "/bin/bash", "-c"]

# Install R packages for pathway analysis
RUN R --slave -e "install.packages(c('devtools', 'tidyverse', 'igraph'), repos='http://cran.r-project.org', quiet=TRUE)" && \
    R --slave -e "if (!require('BiocManager', quietly=TRUE)) install.packages('BiocManager', repos='http://cran.r-project.org'); BiocManager::install(c('GSVA', 'singscore', 'Biobase'), update=FALSE, ask=FALSE, quiet=TRUE)"

# Stage 2: Python application layer
FROM base AS application

# Copy application code
WORKDIR /app
COPY pyproject.toml setup.py* README.md* ./
COPY src ./src
COPY config ./config
COPY workflows ./workflows

# Install Python dependencies via pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e ".[dev]"

# Create necessary directories
RUN mkdir -p /app/data \
    /app/models \
    /app/outputs \
    /app/logs \
    /app/mlruns

# Expose MLflow tracking server port
EXPOSE 5000

# Set entrypoint
ENTRYPOINT ["conda", "run", "-n", "mmpipeline"]
CMD ["/bin/bash"]

# Stage 3: Development image with Jupyter and testing tools
FROM application AS development

RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    ipykernel \
    ipywidgets \
    pytest-watch \
    black \
    isort \
    flake8 \
    mypy

# Expose Jupyter port
EXPOSE 8888

# Stage 4: Production image (slim)
FROM base AS production

WORKDIR /app
COPY pyproject.toml setup.py* README.md* ./
COPY src ./src
COPY config ./config
COPY workflows ./workflows

# Install only required dependencies (no dev)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir .

# Create necessary directories
RUN mkdir -p /app/data \
    /app/models \
    /app/outputs \
    /app/logs \
    /app/mlruns

# Non-root user for security
RUN useradd -m -u 1000 mmpipeline && \
    chown -R mmpipeline:mmpipeline /app
USER mmpipeline

EXPOSE 5000
ENTRYPOINT ["conda", "run", "-n", "mmpipeline"]
CMD ["/bin/bash"]
