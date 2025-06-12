FROM python:3.12-slim

# Install security updates and curl for health checks
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -m -g appuser appuser

# Install uv for dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app
RUN chown -R appuser:appuser /app

# Copy project configuration
COPY --chown=appuser:appuser pyproject.toml uv.lock ./

USER appuser
RUN uv sync --frozen

USER root
COPY --chown=appuser:appuser src src

USER appuser
EXPOSE 8001

ENV ENVIRONMENT=production
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["uv", "run", "--", "uvicorn", "research.api.app:app", "--host", "0.0.0.0", "--port", "8001"]
