# ---------------------------------------------------------------------------
# Thorn Gateway — multi-stage Docker build
#
# Build:   docker build -t thorn-gateway .
# Run:     docker run --env-file .env -v /path/to/.thorn:/workspace/.thorn thorn-gateway
# ---------------------------------------------------------------------------

# ---- uv binary (fast Python package installer) ----------------------------
FROM ghcr.io/astral-sh/uv:0.7 AS uv

# ---- builder: install Python deps into a portable venv --------------------
FROM python:3.12-slim AS builder

COPY --from=uv /uv /usr/local/bin/uv

WORKDIR /build
COPY pyproject.toml ./
COPY src/ src/

RUN uv venv /opt/thorn-venv \
 && VIRTUAL_ENV=/opt/thorn-venv uv pip install ".[github,gitlab]"

# ---- runtime: slim image with only what the gateway needs -----------------
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/thorn-venv /opt/thorn-venv
ENV PATH="/opt/thorn-venv/bin:$PATH"

RUN useradd --create-home thorn
USER thorn

WORKDIR /workspace

COPY deploy/docker/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
