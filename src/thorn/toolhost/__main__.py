"""``python -m thorn.toolhost`` -- the per-agent tool-execution daemon CLI.

Argument parsing only; all of the work lives in
:mod:`thorn.toolhost._server`.  Keeping this file dumb means tests can
exercise the server directly without poking through ``argparse``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from thorn.toolhost._server import ToolhostConfig, run


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m thorn.toolhost",
        description=(
            "Per-agent tool-execution daemon. Binds a Unix-domain socket "
            "and serves sandbox-venue tool calls from the brain."
        ),
    )
    parser.add_argument(
        "--socket",
        required=True,
        type=Path,
        help="Path to the Unix-domain socket to bind.",
    )
    parser.add_argument(
        "--agent-id",
        required=True,
        help="Identifier of the agent this daemon serves.",
    )
    parser.add_argument(
        "--home",
        type=Path,
        default=None,
        help=(
            "Path to the agent's home directory. Used to resolve '~' in tool "
            "arguments and as the journal root."
        ),
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=None,
        help=(
            "Agent workspace mount point. Per-call requests may name a "
            "subdirectory under this path."
        ),
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Path to the daemon's log file. Defaults to stderr only.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=8,
        help="Maximum concurrent tool calls (semaphore size).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose logging and traceback-bearing error responses.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    config = ToolhostConfig(
        socket_path=args.socket,
        agent_id=args.agent_id,
        home_path=args.home,
        workspace_root=args.workspace_root,
        log_path=args.log_file,
        max_concurrency=args.max_concurrency,
        debug=args.debug,
    )
    run(config)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
