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

# ---- runtime: development-ready image for the gateway agent ---------------
FROM python:3.12-slim

# System packages: version control, C/C++ toolchain, and Node.js (LTS via
# NodeSource).  Rust is installed per-user below via rustup.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        cmake \
        pkg-config \
        curl \
        ca-certificates \
        gnupg \
 && mkdir -p /etc/apt/keyrings \
 && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
        | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
 && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_22.x nodistro main" \
        > /etc/apt/sources.list.d/nodesource.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends nodejs \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/thorn-venv /opt/thorn-venv
ENV PATH="/opt/thorn-venv/bin:$PATH"

RUN useradd --create-home thorn
USER thorn

# Rust toolchain (installed as the unprivileged thorn user).
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain stable --profile default
ENV PATH="/home/thorn/.cargo/bin:$PATH"

# TypeScript support (global install into the user's npm prefix).
RUN npm config set prefix /home/thorn/.npm-global \
 && npm install -g typescript ts-node
ENV PATH="/home/thorn/.npm-global/bin:$PATH"

WORKDIR /workspace

COPY deploy/docker/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
