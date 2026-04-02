"""Run a single evaluation trial parameterized on workflow and scenario.

Bootstraps a workspace from the scenario and workflow templates, invokes
the workflow's ``run_workflow.py`` script, then invokes the scenario's
``evaluate_scenario.py`` script.  Merges the results into a single
``result.json`` under the trial output directory.

Usage::

    python trial.py --workflow thorn_cpp --scenario calc
    python trial.py --workflow thorn_cpp --scenario calc --output-dir ./my_trials
    python trial.py --workflow thorn_cpp --scenario calc --task "build a calculator"
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import Any

EVALUATION_DIR = Path(__file__).resolve().parent
WORKFLOWS_DIR = EVALUATION_DIR / "workflows"
SCENARIOS_DIR = EVALUATION_DIR / "scenarios"


class Verbosity(IntEnum):
    QUIET = 0
    VERBOSE = 1
    TRACE = 2


def _trial_id_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def _resolve_workflow(name: str) -> Path:
    """Return the workflow directory, raising if it doesn't exist."""
    workflow_dir = WORKFLOWS_DIR / name
    if not workflow_dir.is_dir():
        available = [p.name for p in WORKFLOWS_DIR.iterdir() if p.is_dir()]
        raise FileNotFoundError(
            f"Workflow {name!r} not found at {workflow_dir}. "
            f"Available: {available}"
        )
    return workflow_dir


def _resolve_scenario(name: str) -> Path:
    """Return the scenario directory, raising if it doesn't exist."""
    scenario_dir = SCENARIOS_DIR / name
    if not scenario_dir.is_dir():
        available = [p.name for p in SCENARIOS_DIR.iterdir() if p.is_dir()]
        raise FileNotFoundError(
            f"Scenario {name!r} not found at {scenario_dir}. "
            f"Available: {available}"
        )
    return scenario_dir


def _load_prompt(scenario_dir: Path) -> str:
    """Read the scenario's prompt.md file."""
    prompt_path = scenario_dir / "prompt.md"
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"No prompt.md found in scenario directory {scenario_dir}"
        )
    return prompt_path.read_text(encoding="utf-8").strip()


def _overlay_template(template_dir: Path, target: Path) -> None:
    """Copy all files from *template_dir* into *target*, preserving structure."""
    if not template_dir.is_dir():
        return
    for src_path in template_dir.rglob("*"):
        if src_path.is_file():
            rel = src_path.relative_to(template_dir)
            dst = target / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst)


def _bootstrap(
    scenario_dir: Path,
    workflow_dir: Path,
    target: Path,
) -> None:
    """Populate the workspace by layering scenario then workflow templates."""
    scenario_template = scenario_dir / "template"
    workflow_template = workflow_dir / "template"

    _overlay_template(scenario_template, target)
    _overlay_template(workflow_template, target)


def run_trial(
    workflow: str,
    scenario: str,
    output_dir: str = "trials",
    trial_id: str | None = None,
    task: str | None = None,
    verbose: Verbosity = Verbosity.VERBOSE,
) -> Path:
    """Run a single trial and write results.  Returns the result directory."""
    workflow_dir = _resolve_workflow(workflow)
    scenario_dir = _resolve_scenario(scenario)

    if task is None:
        task = _load_prompt(scenario_dir)

    if trial_id is None:
        trial_id = _trial_id_now()

    result_dir = Path(output_dir).resolve() / trial_id
    result_dir.mkdir(parents=True, exist_ok=True)

    work_dir = result_dir / "workspace"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir()

    _bootstrap(scenario_dir, workflow_dir, work_dir)

    workflow_result_file = result_dir / "workflow_result.json"
    eval_result_file = result_dir / "eval_result.json"

    workflow_script = workflow_dir / "run_workflow.py"
    workflow_cmd = [
        sys.executable,
        str(workflow_script),
        "--workspace-dir", str(work_dir),
        "--prompt", task,
        "--result-file", str(workflow_result_file),
    ]
    if verbose <= Verbosity.QUIET:
        workflow_cmd.append("--quiet")

    subprocess.run(workflow_cmd)

    if workflow_result_file.exists():
        workflow_result: dict[str, Any] = json.loads(
            workflow_result_file.read_text(encoding="utf-8")
        )
    else:
        workflow_result = {
            "outcome": "agent_error",
            "duration_s": 0.0,
            "token_usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "error": "workflow did not produce a result file",
            "trace_file": None,
        }

    eval_script = scenario_dir / "evaluate_scenario.py"
    eval_cmd = [
        sys.executable,
        str(eval_script),
        "--workspace-dir", str(work_dir),
        "--result-file", str(eval_result_file),
    ]
    if verbose >= Verbosity.TRACE:
        eval_cmd.append("--verbose")
    subprocess.run(eval_cmd)

    if eval_result_file.exists():
        eval_result: dict[str, Any] = json.loads(
            eval_result_file.read_text(encoding="utf-8")
        )
    else:
        eval_result = {
            "build_ok": False,
            "validation_score": 0.0,
            "cases": [],
            "modules_created": [],
        }

    outcome = workflow_result.get("outcome", "agent_error")
    if outcome == "success" and not eval_result.get("build_ok", False):
        outcome = "build_failure"

    result: dict[str, Any] = {
        "trial_id": trial_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "workflow": workflow,
        "scenario": scenario,
        "task": task,
        "outcome": outcome,
        "duration_s": workflow_result.get("duration_s", 0.0),
        "token_usage": workflow_result.get("token_usage", {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }),
        "error": workflow_result.get("error"),
        "trace_file": workflow_result.get("trace_file"),
        "build_ok": eval_result.get("build_ok", False),
        "validation_score": eval_result.get("validation_score", 0.0),
        "binary_runs": eval_result.get("cases", []),
        "modules_created": eval_result.get("modules_created", []),
        "test_ok": None,
    }

    result_json = result_dir / "result.json"
    result_json.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8",
    )

    print(f"\nTrial {trial_id}: {result['outcome']}")
    print(f"  workflow: {workflow}")
    print(f"  scenario: {scenario}")
    print(f"  duration: {result['duration_s']}s")
    if result["token_usage"].get("total_tokens"):
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a single evaluation trial.",
    )
    parser.add_argument(
        "--workflow",
        required=True,
        help="Name of the workflow to use (e.g. 'thorn_cpp')",
    )
    parser.add_argument(
        "--scenario",
        required=True,
        help="Name of the scenario to evaluate (e.g. 'calc')",
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
        default=None,
        help="Override the scenario's default prompt",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output during the workflow run",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="count",
        default=None,
        help="Increase verbosity (-v: show workflow output, -vv: also show eval output)",
    )
    args = parser.parse_args()

    if args.verbose is not None:
        verbose = Verbosity(min(args.verbose, Verbosity.TRACE))
    elif args.quiet:
        verbose = Verbosity.QUIET
    else:
        verbose = Verbosity.VERBOSE

    run_trial(
        workflow=args.workflow,
        scenario=args.scenario,
        output_dir=args.output_dir,
        trial_id=args.trial_id,
        task=args.task,
        verbose=verbose,
    )


if __name__ == "__main__":
    main()
