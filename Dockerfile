# ── Stage 1: dependency resolver ─────────────────────────────────────────────
FROM python:3.12-slim AS builder

# Install uv (fast Python package installer / resolver)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy only the project manifest first to leverage Docker layer caching.
# Dependencies are reinstalled only when pyproject.toml changes.
COPY pyproject.toml .

# Install dependencies into an isolated virtual environment inside the image.
# --no-install-project skips installing the project itself at this stage.
RUN uv sync --no-install-project --no-cache

# ── Stage 2: runtime image ───────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy the pre-built virtual environment from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy application source
COPY app/ app/
COPY main.py .

# Copy the dataset directory (archive.zip must be present at build time,
# or mount it at runtime via a volume — see below)
COPY data/ data/

# Persistent stores (ChromaDB + feature vectors) are written here at runtime.
# Mount a named volume to persist them across container restarts:
#   docker run -v similarity-stores:/app/stores ...
ENV CHROMA_DB_PATH=/app/stores/chroma_db

# Ensure the virtual environment is on PATH
ENV PATH="/app/.venv/bin:$PATH"

# Expose the FastAPI port
EXPOSE 8000

# Launch with uvicorn; use --workers 1 because the in-process store build
# must only run once (increase workers after the first cold start completes)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
