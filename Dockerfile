# Multi-stage build for Haerae Evaluation Toolkit
# Stage 1: Build stage
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first for better caching
COPY requirements-docker.txt /tmp/requirements-docker.txt
COPY pyproject.toml /tmp/pyproject.toml

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r /tmp/requirements-docker.txt

# Stage 2: Runtime stage
FROM python:3.11-slim as runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    HRET_HOME="/app"

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create app directory
WORKDIR /app

# Create non-root user
RUN groupadd -r hret && useradd -r -g hret hret

# Copy application code
COPY --chown=hret:hret . .

# Install the package in development mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/configs && \
    chown -R hret:hret /app

# Switch to non-root user
USER hret

# Default command - can be overridden
CMD ["python", "run_eval.py", "--help"]