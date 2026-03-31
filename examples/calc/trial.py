"""Run a single trial of the calc workflow and write structured results.

Bootstraps the calc project into an isolated temporary directory, runs the
agentic workflow via ``coordinate()``, and writes a JSON result file with
outcome, duration, token usage, and module inventory.

Usage::

    python trial.py
    python trial.py --output-dir ./my_trials --task "implement a calculator"
    python trial.py --trial-id custom_run_01 --quiet
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CALC_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = CALC_DIR / "template"
THORN_DIR = CALC_DIR / ".thorn"

DEFAULT_TASK = "implement a calculator"

VALIDATION_CASES: list[dict[str, Any]] = [
    {"name": "addition",  "input": "2 + 3\n",        "expected": ["5"]},
    {"name": "division",  "input": "10 / 4\n",       "expected": ["2.5"]},
    {"name": "variables", "input": "x = 7\nx * 3\n", "expected": ["7", "21"]},
    {"name": "sin",       "input": "sin(0)\n",        "expected": ["0"]},
    {"name": "sqrt",      "input": "sqrt(16)\n",      "expected": ["4"]},
]


def _trial_id_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def _bootstrap(target: Path) -> None:
    """Copy template files and .thorn/ directory into *target*."""
    for src_path in TEMPLATE_DIR.rglob("*"):
        if src_path.is_file():
            rel = src_path.relative_to(TEMPLATE_DIR)
            dst = target / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst)

    shutil.copytree(
        THORN_DIR,
        target / ".thorn",
        ignore=shutil.ignore_patterns("__pycache__"),
    )


def _values_match(expected: str, actual: str) -> bool:
    """Flexible numeric comparison between an expected and actual value."""
    e, a = expected.strip(), actual.strip()
    if e == a:
        return True
    try:
        return abs(float(e) - float(a)) < 1e-6
    except ValueError:
        return False


def _find_calc_binary(work_dir: Path) -> Path | None:
    """Locate the built calc executable under *work_dir*/build."""
    build_dir = work_dir / "build"
    if not build_dir.exists():
        return None
    for pattern in ("calc.exe", "calc"):
        for candidate in build_dir.rglob(pattern):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate
    return None


def validate_binary(work_dir: Path) -> dict[str, Any]:
    """Run canned expressions through the calc binary and check outputs.

    Returns ``{"cases": [...], "score": <float 0-1>}``.
    Each case has *name*, *passed*, and either *expected*/*actual* or *error*.
    """
    exe = _find_calc_binary(work_dir)
    if exe is None:
        return {
            "cases": [
                {"name": c["name"], "passed": False, "error": "binary not found"}
                for c in VALIDATION_CASES
            ],
            "score": 0.0,
        }

    cases: list[dict[str, Any]] = []
    for case in VALIDATION_CASES:
        try:
            proc = subprocess.run(
                [str(exe)],
                input=case["input"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            output_lines = [
                l.strip() for l in (proc.stdout or "").splitlines() if l.strip()
            ]

            matched: list[bool] = []
            oi = 0
            for exp in case["expected"]:
                found = False
                while oi < len(output_lines):
                    line = output_lines[oi]
                    oi += 1
                    if _values_match(exp, line):
                        found = True
                        break
                    tokens = line.split()
                    if tokens and _values_match(exp, tokens[-1]):
                        found = True
                        break
                matched.append(found)

            cases.append({
                "name": case["name"],
                "passed": all(matched),
                "expected": case["expected"],
                "actual": output_lines,
            })
        except subprocess.TimeoutExpired:
            cases.append({"name": case["name"], "passed": False, "error": "timeout"})
        except Exception as exc:
            cases.append({"name": case["name"], "passed": False, "error": str(exc)})

    passed_count = sum(1 for c in cases if c["passed"])
    return {"cases": cases, "score": passed_count / len(cases) if cases else 0.0}


def _check_build(work_dir: Path) -> bool:
    """Run cmake configure + build and return True if both succeed."""
    build_dir = work_dir / "build"
    if not build_dir.exists():
        cfg = subprocess.run(
            ["cmake", "-S", ".", "-B", "build"],
            capture_output=True,
            cwd=work_dir,
        )
        if cfg.returncode != 0:
            return False
    result = subprocess.run(
        ["cmake", "--build", "build"],
        capture_output=True,
        cwd=work_dir,
    )
    return result.returncode == 0


def _collect_modules(work_dir: Path) -> list[str]:
    """Return sorted module names based on header files in src/."""
    src = work_dir / "src"
    if not src.is_dir():
        return []
    headers = sorted(src.rglob("*.h"))
    return [h.stem for h in headers]


async def _run_trial(
    work_dir: Path,
    trace_path: Path,
    task: str,
    quiet: bool,
) -> dict[str, Any]:
    """Execute the workflow inside *work_dir* and return a result dict."""
    from thorn import (
        ConsoleEventSink,
        ExecutionContext,
        NullEventSink,
        Verbosity,
        discover_tools,
        load_provider_from_env,
        set_context,
    )
    from thorn._context import reset_context
    from thorn._file_access import load_global_ignores
    from thorn._trace import CompositeEventSink, JsonLinesSink
    from thorn.errors import SkillError

    provider = load_provider_from_env()

    trace_fh = open(trace_path, "w", encoding="utf-8")
    try:
        json_sink = JsonLinesSink(trace_fh)
        if quiet:
            sink = json_sink
        else:
            console_sink = ConsoleEventSink(verbosity=Verbosity.NORMAL)
            sink = CompositeEventSink([console_sink, json_sink])

        ws_root = work_dir.resolve()
        global_ignores = load_global_ignores(ws_root)

        ctx = ExecutionContext(
            provider=provider,
            event_sink=sink,
            workspace_root=ws_root,
            global_ignores=global_ignores,
        )

        token = set_context(ctx)
        try:
            tools = discover_tools(start=work_dir)

            coordinate_fn = None
            for fn in tools:
                if getattr(fn, "__name__", None) == "coordinate":
                    coordinate_fn = fn
                    break

            if coordinate_fn is None:
                return {
                    "outcome": "agent_error",
                    "error": "coordinate tool not found after discovery",
                    "duration_s": 0.0,
                }

            outcome = "success"
            error_msg: str | None = None
            t0 = time.monotonic()

            try:
                await coordinate_fn(task)
            except SkillError as exc:
                outcome = "agent_error"
                error_msg = exc.detail
            except TimeoutError:
                outcome = "timeout"
                error_msg = "trial timed out"
            except Exception as exc:
                outcome = "agent_error"
                error_msg = str(exc)

            duration_s = time.monotonic() - t0

            build_ok = _check_build(work_dir)
            if outcome == "success" and not build_ok:
                outcome = "build_failure"

            if build_ok:
                validation = validate_binary(work_dir)
            else:
                validation = {"cases": [], "score": 0.0}

            modules = _collect_modules(work_dir)

            return {
                "outcome": outcome,
                "duration_s": round(duration_s, 2),
                "token_usage": {
                    "prompt_tokens": ctx.usage.prompt_tokens,
                    "completion_tokens": ctx.usage.completion_tokens,
                    "total_tokens": ctx.usage.total_tokens,
                },
                "modules_created": modules,
                "build_ok": build_ok,
                "test_ok": None,
                "binary_runs": validation["cases"],
                "validation_score": validation["score"],
                "error": error_msg,
            }
        finally:
            reset_context(token)
    finally:
        trace_fh.close()


def run_trial(
    output_dir: str = "trials",
    trial_id: str | None = None,
    task: str = DEFAULT_TASK,
    quiet: bool = False,
) -> Path:
    """Run a single trial and write results.  Returns the result directory."""
    if trial_id is None:
        trial_id = _trial_id_now()

    result_dir = Path(output_dir).resolve() / trial_id
    result_dir.mkdir(parents=True, exist_ok=True)

    work_dir = Path(tempfile.mkdtemp(prefix="thorn_trial_"))
    orig_cwd = Path.cwd()
    trace_path = result_dir / "trace.jsonl"

    try:
        _bootstrap(work_dir)
        os.chdir(work_dir)

        result = asyncio.run(
            _run_trial(work_dir, trace_path, task, quiet)
        )

        result["trial_id"] = trial_id
        result["timestamp"] = datetime.now(timezone.utc).isoformat()
        result["task"] = task
        result["trace_file"] = "trace.jsonl"

        result_json = result_dir / "result.json"
        result_json.write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8",
        )

        src_dir = work_dir / "src"
        if src_dir.is_dir():
            shutil.copytree(src_dir, result_dir / "src")

        print(f"\nTrial {trial_id}: {result['outcome']}")
        print(f"  duration: {result['duration_s']}s")
        if result["token_usage"]["total_tokens"]:
            print(f"  tokens:   {result['token_usage']['total_tokens']}")
        print(f"  build_ok: {result['build_ok']}")
        score = result.get("validation_score", 0.0)
        passed = sum(1 for c in result.get("binary_runs", []) if c.get("passed"))
        total = len(result.get("binary_runs", []))
        print(f"  binary:   {passed}/{total} passed (score {score:.2f})")
        print(f"  modules:  {result['modules_created']}")
        if result["error"]:
            print(f"  error:    {result['error']}")
        print(f"  results:  {result_json}")

        return result_dir

    finally:
        os.chdir(orig_cwd)
        shutil.rmtree(work_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a single trial of the calc agentic workflow.",
    )
    parser.add_argument(
        "--output-dir",
        default="trials",
        help="Base directory for trial results (default: ./trials)",
    )
    parser.add_argument(
        "--trial-id",
        default=None,
        help="Identifier for this trial (default: UTC timestamp)",
    )
    parser.add_argument(
        "--task",
        default=DEFAULT_TASK,
        help=f"Prompt to pass to coordinate (default: {DEFAULT_TASK!r})",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output during the workflow run",
    )
    args = parser.parse_args()

    run_trial(
        output_dir=args.output_dir,
        trial_id=args.trial_id,
        task=args.task,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
