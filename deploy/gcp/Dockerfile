# Container image for running `planetx simgen run` on Google Batch.
# Build from the repo root: docker build -f deploy/gcp/Dockerfile -t <image> .

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src/ src/
COPY configs/ configs/

# --frozen: use uv.lock exactly as committed, no re-resolution.
RUN uv sync --frozen --no-dev

COPY deploy/gcp/run_shard.sh /app/run_shard.sh
RUN chmod +x /app/run_shard.sh

ENV PATH="/app/.venv/bin:${PATH}"
ENTRYPOINT ["/app/run_shard.sh"]
