"""Run the Thorn workflow inside a prepared workspace via ``thorn run``.

Invokes ``thorn run`` as a subprocess with ``--result-file`` so that
outcome, timing, and token usage are captured as structured JSON without
needing to import any thorn internals.

Usage::

    python run_workflow.py --workspace-dir <path> --prompt "..." --result-file <path>
    python run_workflow.py --workspace-dir <path> --prompt "..." --result-file <path> --quiet
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _empty_token_usage() -> dict[str, int]:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _fallback_result(error: str, trace_file: str | None = None) -> dict[str, Any]:
    return {
        "outcome": "agent_error",
        "duration_s": 0.0,
        "token_usage": _empty_token_usage(),
        "error": error,
        "trace_file": trace_file,
    }


def run_workflow(
    workspace_dir: Path,
    prompt: str,
    result_file: Path,
    quiet: bool = False,
) -> None:
    """Run the Thorn workflow and write results to *result_file*."""
    workspace_dir = workspace_dir.resolve()
    result_file = result_file.resolve()
    result_file.parent.mkdir(parents=True, exist_ok=True)

    trace_path = result_file.parent / "trace.jsonl"

    # thorn run writes its result JSON to a temporary location so we can
    # detect whether it actually ran to completion.
    thorn_result_path = result_file.parent / "_thorn_result.json"

    cmd = [
        sys.executable, "-m", "thorn", "run", prompt,
        "--workspace", str(workspace_dir),
        "--trace", str(trace_path),
        "--result-file", str(thorn_result_path),
    ]
    if quiet:
        cmd.append("--quiet")

    try:
        proc = subprocess.run(cmd, cwd=str(workspace_dir))
    except Exception as exc:
        result = _fallback_result(str(exc), trace_file=trace_path.name)
        result_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return

    if thorn_result_path.exists():
        shutil.move(str(thorn_result_path), str(result_file))
    else:
        result = _fallback_result(
            f"thorn subprocess exited {proc.returncode} without writing a result file",
            trace_file=trace_path.name if trace_path.exists() else None,
        )
        result_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Thorn coordinator workflow in a workspace.",
    )
    parser.add_argument(
        "--workspace-dir",
        required=True,
        help="Path to the prepared trial workspace",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="The task prompt to pass to the coordinate tool",
    )
    parser.add_argument(
        "--result-file",
        required=True,
        help="Path to write the JSON workflow result",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output during the workflow run",
    )
    args = parser.parse_args()

    run_workflow(
        workspace_dir=Path(args.workspace_dir),
        prompt=args.prompt,
        result_file=Path(args.result_file),
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
