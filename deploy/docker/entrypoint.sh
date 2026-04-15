#!/usr/bin/env bash
# Docker entrypoint for the Thorn gateway.
#
# Responsibilities:
#   1. Configure git identity from GIT_AUTHOR_NAME / GIT_COMMITTER_NAME
#      (or sensible defaults) so agent git operations work.
#   2. Exec into `thorn serve` (or whatever command the user passed).
set -euo pipefail

# -- Tool paths --------------------------------------------------------------
# Rust (rustup) and npm global installs live under the thorn user's home.
# These are also set via ENV in the Dockerfile, but we re-export here so
# that interactive `docker exec` sessions pick them up too.
export PATH="$HOME/.cargo/bin:$HOME/.npm-global/bin:$PATH"

# -- Git identity -----------------------------------------------------------
# Git reads GIT_AUTHOR_NAME / GIT_COMMITTER_NAME from the environment for
# commits, but `git config` is still needed for operations that consult the
# config directly (e.g. `git clone` credential helpers).
GIT_AUTHOR_NAME="${GIT_AUTHOR_NAME:-thorn-bot}"
GIT_COMMITTER_NAME="${GIT_COMMITTER_NAME:-${GIT_AUTHOR_NAME}}"
GIT_AUTHOR_EMAIL="${GIT_AUTHOR_EMAIL:-${GIT_AUTHOR_NAME}@users.noreply.github.com}"
GIT_COMMITTER_EMAIL="${GIT_COMMITTER_EMAIL:-${GIT_AUTHOR_EMAIL}}"

export GIT_AUTHOR_NAME GIT_COMMITTER_NAME GIT_AUTHOR_EMAIL GIT_COMMITTER_EMAIL

git config --global user.name  "${GIT_AUTHOR_NAME}"
git config --global user.email "${GIT_AUTHOR_EMAIL}"

# -- Run --------------------------------------------------------------------
# If the user passed explicit arguments (e.g. `docker run ... thorn serve
# bootstrap ...`), run those.  Otherwise default to `thorn serve`.
if [ $# -gt 0 ]; then
    exec "$@"
else
    exec thorn serve --workspace /workspace
fi
