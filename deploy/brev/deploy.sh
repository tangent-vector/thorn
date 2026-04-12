#!/usr/bin/env bash
#
# Deploy Thorn gateway to an NVIDIA Brev instance.
#
# Run this from your dev machine (Linux or WSL) from the thorn repo root:
#
#   ./deploy/brev/deploy.sh [OPTIONS]
#
# Options (all optional -- sensible defaults are used):
#
#   --name NAME         Brev instance name      (default: thorn-gateway)
#   --repo URL          Git clone URL for Thorn  (default: current remote origin)
#   --gpu GPU_TYPE      Brev GPU spec            (omit for cheapest default)
#   --env-file PATH     Copy this .env file to the instance (optional; not defaulted)
#   --agency-dir PATH   .thorn/ directory to copy (skips bootstrap if provided)
#   --no-start          Don't start thorn serve after setup
#
set -euo pipefail

# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------

INSTANCE_NAME="thorn-gateway"
REPO_URL=""
GPU_TYPE=""
ENV_FILE=""
AGENCY_DIR=""
AUTO_START="true"

REMOTE_WORKSPACE="/home/ubuntu/workspace"
REMOTE_THORN="${REMOTE_WORKSPACE}/thorn"

# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)       INSTANCE_NAME="$2"; shift 2 ;;
        --repo)       REPO_URL="$2";      shift 2 ;;
        --gpu)        GPU_TYPE="$2";      shift 2 ;;
        --env-file)   ENV_FILE="$2";      shift 2 ;;
        --agency-dir) AGENCY_DIR="$2";    shift 2 ;;
        --no-start)   AUTO_START="false"; shift ;;
        -h|--help)
            sed -n '2,/^set -euo/{ /^#/s/^# \{0,1\}//p }' "$0"
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

log()  { echo "==> $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }

require_cmd() {
    command -v "$1" &>/dev/null || die "'$1' is not installed or not on PATH."
}

# --------------------------------------------------------------------------
# Preflight checks
# --------------------------------------------------------------------------

require_cmd brev

# Verify brev is authenticated (brev list succeeds for logged-in users)
if ! brev list &>/dev/null; then
    die "Brev CLI is not authenticated. Run 'brev login' first."
fi

# Resolve repo URL from git remote if not explicitly provided
if [ -z "$REPO_URL" ]; then
    if git remote get-url origin &>/dev/null; then
        REPO_URL="$(git remote get-url origin)"
        log "Using repo URL from git remote origin: $REPO_URL"
    else
        die "No --repo URL given and no git remote origin found. Provide --repo."
    fi
fi

# --------------------------------------------------------------------------
# Step 1: Create the Brev instance
# --------------------------------------------------------------------------

log "Creating Brev instance '$INSTANCE_NAME' from $REPO_URL"

CREATE_ARGS=(
    --name "$INSTANCE_NAME"
    --setup-repo .
    --setup-path deploy/brev/setup.sh
)
if [ -n "$GPU_TYPE" ]; then
    CREATE_ARGS+=(--gpu "$GPU_TYPE")
fi

brev start "$REPO_URL" "${CREATE_ARGS[@]}"

log "Waiting for instance to be ready ..."
# brev start blocks until the instance is running, but the setup script may
# still be executing.  Give it a moment, then poll for the thorn CLI.
sleep 10

RETRIES=30
DELAY=10
for ((i=1; i<=RETRIES; i++)); do
    if brev shell "$INSTANCE_NAME" -- command -v thorn &>/dev/null; then
        log "thorn CLI detected on instance"
        break
    fi
    if [ "$i" -eq "$RETRIES" ]; then
        log "WARNING: thorn CLI not yet available after $((RETRIES * DELAY))s."
        log "The setup script may still be running.  Check with:"
        log "  brev shell $INSTANCE_NAME"
        break
    fi
    log "  Setup still running ... ($((i * DELAY))s elapsed)"
    sleep "$DELAY"
done

# --------------------------------------------------------------------------
# Step 2: Copy .env file (if provided)
# --------------------------------------------------------------------------

if [ -n "$ENV_FILE" ]; then
    log "Copying $ENV_FILE to instance"
    brev copy "$ENV_FILE" "${INSTANCE_NAME}:${REMOTE_THORN}/.env"
fi

# --------------------------------------------------------------------------
# Step 3: Copy agency directory (if provided)
# --------------------------------------------------------------------------

if [ -n "$AGENCY_DIR" ]; then
    if [ ! -d "$AGENCY_DIR" ]; then
        die "Agency directory not found: $AGENCY_DIR"
    fi
    log "Copying agency directory $AGENCY_DIR to instance"
    brev copy -r "$AGENCY_DIR" "${INSTANCE_NAME}:${REMOTE_THORN}/.thorn"
fi

# --------------------------------------------------------------------------
# Step 4: Start the gateway (or print instructions)
# --------------------------------------------------------------------------

if [ "$AUTO_START" = "true" ]; then
    log "Starting thorn serve on the instance (in a detached tmux session)"
    brev shell "$INSTANCE_NAME" -- bash -c "
        cd ${REMOTE_THORN}
        tmux new-session -d -s thorn 'thorn serve 2>&1 | tee /tmp/thorn-serve.log'
    "
    log ""
    log "Gateway is running in a tmux session on the instance."
    log "  Attach:  brev shell $INSTANCE_NAME -- tmux attach -t thorn"
    log "  Logs:    brev shell $INSTANCE_NAME -- tail -f /tmp/thorn-serve.log"
    log "  Stop:    brev shell $INSTANCE_NAME -- tmux kill-session -t thorn"
else
    log ""
    log "Instance is ready. To start the gateway manually:"
    log "  brev shell $INSTANCE_NAME"
    log "  cd ${REMOTE_THORN}"
    log "  thorn serve"
fi

log ""
log "Instance management:"
log "  Shell:   brev shell $INSTANCE_NAME"
log "  Stop:    brev stop $INSTANCE_NAME"
log "  Delete:  brev delete $INSTANCE_NAME"
