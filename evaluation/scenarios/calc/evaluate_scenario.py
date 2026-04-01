"""Evaluate the final state of a calc workspace after a workflow has run.

Performs CMake build, binary validation against canned test cases, and
module inventory.  Writes a JSON result file with build status,
per-case pass/fail, a 0-1 validation score, and the list of modules
created.

Usage::

    python evaluate_scenario.py --workspace-dir <path> --result-file <path>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

VALIDATION_CASES: list[dict[str, Any]] = [
    {"name": "addition",  "input": "2 + 3\n",        "expected": ["5"]},
    {"name": "division",  "input": "10 / 4\n",       "expected": ["2.5"]},
    {"name": "variables", "input": "x = 7\nx * 3\n", "expected": ["7", "21"]},
    {"name": "sin",       "input": "sin(0)\n",        "expected": ["0"]},
    {"name": "sqrt",      "input": "sqrt(16)\n",      "expected": ["4"]},
]


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


def _validate_binary(work_dir: Path) -> dict[str, Any]:
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
                line.strip()
                for line in (proc.stdout or "").splitlines()
                if line.strip()
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


def _collect_modules(work_dir: Path) -> list[str]:
    """Return sorted module names based on header files in src/."""
    src = work_dir / "src"
    if not src.is_dir():
        return []
    headers = sorted(src.rglob("*.h"))
    return [h.stem for h in headers]


def evaluate(work_dir: Path) -> dict[str, Any]:
    """Run full evaluation on a completed workspace and return results."""
    build_ok = _check_build(work_dir)

    if build_ok:
        validation = _validate_binary(work_dir)
    else:
        validation = {"cases": [], "score": 0.0}

    modules = _collect_modules(work_dir)

    build_dir = work_dir / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir, ignore_errors=True)

    return {
        "build_ok": build_ok,
        "validation_score": validation["score"],
        "cases": validation["cases"],
        "modules_created": modules,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a calc workspace after a workflow run.",
    )
    parser.add_argument(
        "--workspace-dir",
        required=True,
        help="Path to the trial workspace directory",
    )
    parser.add_argument(
        "--result-file",
        required=True,
        help="Path to write the JSON evaluation result",
    )
    args = parser.parse_args()

    work_dir = Path(args.workspace_dir).resolve()
    result = evaluate(work_dir)

    result_path = Path(args.result_file).resolve()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
