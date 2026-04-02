"""Run a Claude Code CLI workflow inside a prepared workspace.

Invokes ``claude -p`` as a subprocess with ``--output-format json`` and
auto mode (``--enable-auto-mode --permission-mode auto``) so that the
agent can work autonomously within the workspace.  Writes a JSON result
file with outcome, timing, and token usage in the same schema as the
``thorn_cpp`` workflow.

We use auto mode rather than ``--allowedTools`` because enterprise
managed settings (server-managed policies) take precedence over CLI
flags and can silently block ``--allowedTools`` and even
``--dangerously-skip-permissions``.  Auto mode uses a background
classifier to approve tool calls and is not subject to the same
override, making it the most reliable path for non-sandboxed
enterprise environments.

Usage::

    python run_workflow.py --workspace-dir <path> --prompt "..." --result-file <path>
    python run_workflow.py --workspace-dir <path> --prompt "..." --result-file <path> --quiet
    python run_workflow.py --workspace-dir <path> --prompt "..." --result-file <path> --model opus
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_MAX_TURNS = 50

TRACE_FILENAME = "claude_output.json"


def _empty_token_usage() -> dict[str, int]:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _fallback_result(
    error: str,
    *,
    outcome: str = "agent_error",
    duration_s: float = 0.0,
) -> dict[str, Any]:
    return {
        "outcome": outcome,
        "duration_s": round(duration_s, 2),
        "token_usage": _empty_token_usage(),
        "error": error,
        "trace_file": None,
    }


def _parse_claude_json(raw: str) -> dict[str, Any]:
    """Extract the workflow result fields from ``claude -p --output-format json`` output."""
    data = json.loads(raw)

    is_error = data.get("is_error", False)

    usage = data.get("usage", {})
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

    duration_ms = data.get("duration_ms", 0)

    return {
        "outcome": "agent_error" if is_error else "success",
        "duration_s": round(duration_ms / 1000.0, 2),
        "token_usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
        "error": data.get("result") if is_error else None,
        "trace_file": TRACE_FILENAME,
        "cost_usd": data.get("cost_usd"),
        "session_id": data.get("session_id"),
    }


def _find_claude() -> str:
    """Locate the ``claude`` CLI binary on PATH."""
    path = shutil.which("claude")
    if path is None:
        raise FileNotFoundError(
            "Could not find 'claude' on PATH. "
            "Install Claude Code: https://docs.claude.com/en/docs/claude-code/quickstart"
        )
    return path


def run_workflow(
    workspace_dir: Path,
    prompt: str,
    result_file: Path,
    quiet: bool = False,
    model: str = "opus",
    max_turns: int = DEFAULT_MAX_TURNS,
    bypass_permissions: bool = False,
) -> None:
    """Run the Claude CLI workflow and write results to *result_file*."""
    workspace_dir = workspace_dir.resolve()
    result_file = result_file.resolve()
    result_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        claude_bin = _find_claude()
    except FileNotFoundError as exc:
        result = _fallback_result(str(exc))
        result_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return

    cmd = [
        claude_bin,
        "-p", prompt,
        "--output-format", "json",
        "--model", model,
        "--max-turns", str(max_turns),
    ]

    if bypass_permissions:
        cmd.append("--dangerously-skip-permissions")
    else:
        cmd.extend(["--enable-auto-mode", "--permission-mode", "auto"])

    trace_path = result_file.parent / TRACE_FILENAME

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(workspace_dir),
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        duration_s = time.monotonic() - t0
        result = _fallback_result(
            "trial timed out", outcome="timeout", duration_s=duration_s,
        )
        result_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return
    except Exception as exc:
        duration_s = time.monotonic() - t0
        result = _fallback_result(str(exc), duration_s=duration_s)
        result_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return

    if proc.stdout:
        trace_path.write_text(proc.stdout, encoding="utf-8")

    if not quiet and proc.stderr:
        sys.stderr.write(proc.stderr)

    if not proc.stdout.strip():
        duration_s = time.monotonic() - t0
        error = f"claude exited {proc.returncode} with no output"
        if proc.stderr:
            error += f": {proc.stderr[:500]}"
        result = _fallback_result(error, duration_s=duration_s)
        result_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return

    try:
        result = _parse_claude_json(proc.stdout)
    except (json.JSONDecodeError, KeyError) as exc:
        duration_s = time.monotonic() - t0
        result = _fallback_result(
            f"failed to parse claude output: {exc}", duration_s=duration_s,
        )

    result_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Claude Code CLI workflow in a workspace.",
    )
    parser.add_argument(
        "--workspace-dir",
        required=True,
        help="Path to the prepared trial workspace",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="The task prompt to pass to Claude",
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
    parser.add_argument(
        "--model",
        default="opus",
        help="Claude model to use (default: opus)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=DEFAULT_MAX_TURNS,
        help=f"Max agentic turns (default: {DEFAULT_MAX_TURNS})",
    )
    parser.add_argument(
        "--dangerously-skip-permissions",
        action="store_true",
        dest="bypass_permissions",
        help="Skip permission prompts (only for sandboxed environments)",
    )
    args = parser.parse_args()

    run_workflow(
        workspace_dir=Path(args.workspace_dir),
        prompt=args.prompt,
        result_file=Path(args.result_file),
        quiet=args.quiet,
        model=args.model,
        max_turns=args.max_turns,
        bypass_permissions=args.bypass_permissions,
    )


if __name__ == "__main__":
    main()
