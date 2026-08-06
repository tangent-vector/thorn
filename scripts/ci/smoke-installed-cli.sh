#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <thorn-executable> <smoke-root>" >&2
    exit 2
fi

readonly thorn_executable=$1
readonly smoke_root=$2
readonly agency_home="$smoke_root/agency"
readonly agency_workspace="$smoke_root/workspace"

if [[ ! -x "$thorn_executable" ]]; then
    echo "thorn executable is not runnable: $thorn_executable" >&2
    exit 1
fi

"$thorn_executable" --version
"$thorn_executable" --help >/dev/null
"$thorn_executable" agency init "$agency_home" --workspace "$agency_workspace"
"$thorn_executable" agency check --agency "$agency_home"
"$thorn_executable" agency show --agency "$agency_home" --json >/dev/null
