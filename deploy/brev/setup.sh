#!/usr/bin/env bash
#
# Brev setup script for Thorn gateway.
#
# Runs automatically on the Brev instance after creation when passed via
# --setup-script or --setup-repo/--setup-path.  Can also be executed
# manually after SSH-ing in.
#
# The script is idempotent: re-running it is safe.
#
set -euo pipefail

WORKSPACE="/home/ubuntu/workspace"
THORN_DIR="${WORKSPACE}/thorn"
THORN_EXTRAS="github,gitlab"

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

log() { echo "==> $*"; }

ensure_python3() {
    if ! command -v python3 &>/dev/null; then
        log "python3 not found -- installing via apt"
        sudo apt-get update -qq
        sudo apt-get install -y -qq python3 python3-pip python3-venv
    fi

    local py_version
    py_version="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    local required="3.11"
    if [ "$(printf '%s\n' "$required" "$py_version" | sort -V | head -n1)" != "$required" ]; then
        echo "ERROR: Thorn requires Python >= $required but found $py_version" >&2
        exit 1
    fi
    log "Python $py_version OK"
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
ensure_python3

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

log "Installing Thorn (editable) with extras: $THORN_EXTRAS"
pip install -e ".[$THORN_EXTRAS]"

# Verify the thorn CLI is available
if command -v thorn &>/dev/null; then
    log "thorn CLI installed: $(thorn --version 2>/dev/null || echo 'ok')"
else
    log "WARNING: 'thorn' not on PATH. You may need to activate the venv or adjust PATH."
fi

log "Thorn Brev setup complete"
log ""
log "Next steps:"
log "  1. Ensure environment variables are set (see deploy/brev/env.template)"
log "  2. Set up the agency:  copy a .thorn/ directory or run thorn serve bootstrap"
log "  3. Start the gateway:  cd $THORN_DIR && thorn serve"
