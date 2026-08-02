#!/usr/bin/env bash
#
# Brev setup script for Thorn gateway.
#
# Runs automatically on the Brev instance after creation when passed via
# --setup-script or --setup-repo/--setup-path. Can also be executed
# manually after SSH-ing in.
#
# The script is idempotent: re-running it is safe.
#
set -euo pipefail

WORKSPACE="/home/ubuntu/workspace"
THORN_DIR="${WORKSPACE}/thorn"

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

log() { echo "==> $*"; }

ensure_uv() {
    if ! command -v curl &>/dev/null; then
        log "curl not found -- installing via apt"
        sudo apt-get update -qq
        sudo apt-get install -y -qq curl
    fi

    if ! command -v uv &>/dev/null; then
        log "uv not found -- installing from Astral's public installer"
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="${HOME}/.local/bin:${PATH}"
    fi

    command -v uv &>/dev/null || {
        echo "ERROR: uv installation did not place uv on PATH" >&2
        exit 1
    }
    log "$(uv --version)"
}

ensure_git() {
    if ! command -v git &>/dev/null; then
        log "git not found -- installing via apt"
        sudo apt-get update -qq
        sudo apt-get install -y -qq git
    fi
}

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

log "Thorn Brev setup starting"

ensure_git
ensure_uv

# If Thorn repo is already present (e.g. brev start cloned it), use that.
# Otherwise check for a THORN_REPO_URL env var to clone from.
if [ -d "$THORN_DIR" ]; then
    log "Thorn repo already present at $THORN_DIR"
elif [ -n "${THORN_REPO_URL:-}" ]; then
    log "Cloning Thorn from $THORN_REPO_URL"
    git clone "$THORN_REPO_URL" "$THORN_DIR"
else
    log "WARNING: Thorn repo not found at $THORN_DIR and THORN_REPO_URL is not set."
    log "You will need to clone or copy the repo manually before running thorn serve."
    exit 0
fi

cd "$THORN_DIR"

log "Synchronizing Thorn's locked environment"
uv sync --all-extras --locked

uv run thorn --help >/dev/null
log "Thorn CLI ready"

log "Thorn Brev setup complete"
log ""
log "Next steps:"
log "  1. Supply required secrets explicitly; see deploy/brev/README.md"
log "  2. Set up the agency: copy a .thorn/ directory or run uv run thorn serve bootstrap"
log "  3. Validate it: cd $THORN_DIR && uv run thorn serve --agency .thorn preflight"
log "  4. Start it: cd $THORN_DIR && uv run thorn serve --agency .thorn"
