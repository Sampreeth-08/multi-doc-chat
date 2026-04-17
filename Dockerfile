# ── Stage 1: dependency resolver ─────────────────────────────────────────────
# Use uv's official image to install deps into an isolated venv.
# Keeping this as a separate stage means the final image does not
# need uv itself — only the installed packages are copied over.
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Copy lockfile and project metadata first so this layer is cached
# as long as dependencies don't change.
COPY pyproject.toml uv.lock ./

# Install production deps only (no pytest / httpx etc.)
RUN uv sync --frozen --no-dev --no-install-project

# Copy source and install the project package itself
COPY multi_doc_chat/ ./multi_doc_chat/
RUN uv sync --frozen --no-dev


# ── Stage 2: runtime image ───────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy the fully-built venv from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy application code and assets
COPY multi_doc_chat/ ./multi_doc_chat/
COPY main.py ./
COPY templates/ ./templates/
COPY static/ ./static/

# Create runtime directories the app writes to at startup
RUN mkdir -p sessions data logs

# Make the venv's bin the first entry on PATH so `python` and
# installed console scripts resolve to the venv, not the system.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

CMD ["/app/.venv/bin/uvicorn", "multi_doc_chat.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
