FROM python:3.11-slim AS base

LABEL maintainer="PhD Researcher (UNT)"
LABEL description="MM Transcriptomics Risk Signature Pipeline"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git wget curl \
    libcurl4-openssl-dev libssl-dev libxml2-dev \
    gfortran libopenblas-dev liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY main.py Makefile ./

# Install package in editable mode
RUN pip install -e .

RUN mkdir -p data outputs logs mlruns

EXPOSE 5000

# Development stage
FROM base AS dev
COPY requirements_preprocessing.txt ./
RUN pip install -r requirements_preprocessing.txt && \
    pip install jupyter jupyterlab pytest pytest-cov black isort
COPY tests/ ./tests/
COPY notebooks/ ./notebooks/
COPY examples/ ./examples/
EXPOSE 8888
CMD ["bash"]

# Production stage
FROM base AS production
RUN useradd -m -u 1000 mmpipeline && chown -R mmpipeline:mmpipeline /app
USER mmpipeline
EXPOSE 5000
ENTRYPOINT ["python", "main.py"]
