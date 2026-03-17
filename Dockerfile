FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
COPY domain_schemas/ ./domain_schemas/

# Install the package with all dependencies
RUN pip install --no-cache-dir ".[all]"

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

# Default port for the API server
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/api/v1/health'); r.raise_for_status()" || exit 1

# Default: run the API server
CMD ["python", "-m", "ontology_engine.api.server", "--host", "0.0.0.0", "--port", "8000"]
