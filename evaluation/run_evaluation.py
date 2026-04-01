"""Matrix evaluation runner.

Runs ``run_trials.py`` for every (workflow, scenario) pair in the
evaluation suite and writes a combined ``evaluation.json`` report.

Workflows and scenarios are discovered automatically from the
``workflows/`` and ``scenarios/`` directories, or can be restricted
via command-line options.

Usage::

    python run_evaluation.py
    python run_evaluation.py -w thorn_cpp
    python run_evaluation.py -s calc -n 5
    python run_evaluation.py -w thorn_cpp -s calc -n 3 -p 2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EVALUATION_DIR = Path(__file__).resolve().parent
WORKFLOWS_DIR = EVALUATION_DIR / "workflows"
SCENARIOS_DIR = EVALUATION_DIR / "scenarios"
RUN_TRIALS_SCRIPT = EVALUATION_DIR / "run_trials.py"


def _discover_workflows() -> list[str]:
    """Return sorted names of valid workflow directories."""
    return sorted(
        p.name for p in WORKFLOWS_DIR.iterdir()
        if p.is_dir() and (p / "run_workflow.py").is_file()
    )


def _discover_scenarios() -> list[str]:
    """Return sorted names of valid scenario directories."""
    return sorted(
        p.name for p in SCENARIOS_DIR.iterdir()
        if p.is_dir() and (p / "evaluate_scenario.py").is_file()
    )


def _run_pair(
    workflow: str,
    scenario: str,
    num_trials: int,
    output_dir: str,
    task: str | None,
    parallel: int,
) -> dict[str, Any]:
    """Run ``run_trials.py`` for a single (workflow, scenario) pair.

    Returns the summary dict read back from the generated
    ``summary.json``, or a stub on failure.
    """
    cmd = [
        sys.executable,
        str(RUN_TRIALS_SCRIPT),
        "--workflow", workflow,
        "--scenario", scenario,
        "--num-trials", str(num_trials),
        "--output-dir", output_dir,
        "--parallel", str(parallel),
    ]
    if task is not None:
        cmd.extend(["--task", task])

    proc = subprocess.run(cmd, capture_output=True, text=True)

    summary_file = Path(output_dir) / "summary.json"
    if summary_file.exists():
        return json.loads(summary_file.read_text(encoding="utf-8"))

    return {
        "workflow": workflow,
        "scenario": scenario,
        "total_trials": num_trials,
        "outcomes": {"launch_error": num_trials},
        "success_rate": 0.0,
        "duration_stats": {"mean": None, "median": None, "stdev": None},
        "token_stats": {"mean": None, "median": None, "stdev": None},
        "validation_score_stats": {"mean": None, "median": None, "stdev": None},
        "trials": [],
        "error": f"run_trials exited {proc.returncode}: {(proc.stderr or '')[:500]}",
    }


def _build_overall(
    pair_summaries: list[dict[str, Any]],
    workflows: list[str],
    scenarios: list[str],
) -> dict[str, Any]:
    """Compute aggregate statistics across all pair summaries."""
    total_pairs = len(pair_summaries)
    total_trials = sum(s["total_trials"] for s in pair_summaries)

    all_successes = 0
    all_trials = 0
    for s in pair_summaries:
        all_trials += s["total_trials"]
        all_successes += s.get("outcomes", {}).get("success", 0)

    per_workflow: dict[str, dict[str, Any]] = {}
    for wf in workflows:
        wf_summaries = [s for s in pair_summaries if s["workflow"] == wf]
        wf_trials = sum(s["total_trials"] for s in wf_summaries)
        wf_successes = sum(
            s.get("outcomes", {}).get("success", 0) for s in wf_summaries
        )
        wf_scores = [
            s["validation_score_stats"]["mean"]
            for s in wf_summaries
            if s.get("validation_score_stats", {}).get("mean") is not None
        ]
        per_workflow[wf] = {
            "success_rate": round(wf_successes / wf_trials, 3) if wf_trials else 0.0,
            "mean_validation_score": (
                round(sum(wf_scores) / len(wf_scores), 3) if wf_scores else None
            ),
        }

    per_scenario: dict[str, dict[str, Any]] = {}
    for sc in scenarios:
        sc_summaries = [s for s in pair_summaries if s["scenario"] == sc]
        sc_trials = sum(s["total_trials"] for s in sc_summaries)
        sc_successes = sum(
            s.get("outcomes", {}).get("success", 0) for s in sc_summaries
        )
        sc_scores = [
            s["validation_score_stats"]["mean"]
            for s in sc_summaries
            if s.get("validation_score_stats", {}).get("mean") is not None
        ]
        per_scenario[sc] = {
            "success_rate": round(sc_successes / sc_trials, 3) if sc_trials else 0.0,
            "mean_validation_score": (
                round(sum(sc_scores) / len(sc_scores), 3) if sc_scores else None
            ),
        }

    return {
        "total_pairs": total_pairs,
        "total_trials": total_trials,
        "aggregate_success_rate": (
            round(all_successes / all_trials, 3) if all_trials else 0.0
        ),
        "per_workflow": per_workflow,
        "per_scenario": per_scenario,
    }


def _print_matrix(
    pair_summaries: list[dict[str, Any]],
    workflows: list[str],
    scenarios: list[str],
    overall: dict[str, Any],
) -> None:
    """Print a compact matrix of success rates."""
    col_width = max((len(sc) for sc in scenarios), default=8)
    col_width = max(col_width, 8)
    wf_width = max((len(wf) for wf in workflows), default=10)

    header = " " * (wf_width + 2) + "  ".join(sc.center(col_width) for sc in scenarios)
    separator = "=" * len(header)

    print(f"\n{separator}")
    print("  Evaluation Summary")
    print(separator)
    print(header)
    print("-" * len(header))

    lookup: dict[tuple[str, str], dict[str, Any]] = {
        (s["workflow"], s["scenario"]): s for s in pair_summaries
    }

    for wf in workflows:
        cells: list[str] = []
        for sc in scenarios:
            summary = lookup.get((wf, sc))
            if summary is None:
                cells.append("--".center(col_width))
                continue
            rate = summary["success_rate"]
            total = summary["total_trials"]
            successes = summary.get("outcomes", {}).get("success", 0)
            cells.append(f"{successes}/{total} ({rate:.0%})".center(col_width))
        print(f"{wf:<{wf_width}}  {'  '.join(cells)}")

    print("-" * len(header))

    agg = overall["aggregate_success_rate"]
    total_t = overall["total_trials"]
    total_p = overall["total_pairs"]
    print(
        f"  Overall: {agg:.1%} success across "
        f"{total_t} trial{'s' if total_t != 1 else ''} "
        f"in {total_p} pair{'s' if total_p != 1 else ''}"
    )
    print(separator)


def run_evaluation(
    workflow: str | None = None,
    scenario: str | None = None,
    num_trials: int = 1,
    output_dir: str = "trials",
    task: str | None = None,
    parallel: int = 1,
) -> Path:
    """Run the full evaluation matrix and write ``evaluation.json``.

    Returns the resolved output directory.
    """
    workflows = [workflow] if workflow else _discover_workflows()
    scenarios = [scenario] if scenario else _discover_scenarios()

    if not workflows:
        raise FileNotFoundError(
            f"No valid workflows found in {WORKFLOWS_DIR}. "
            "Each workflow directory must contain a run_workflow.py."
        )
    if not scenarios:
        raise FileNotFoundError(
            f"No valid scenarios found in {SCENARIOS_DIR}. "
            "Each scenario directory must contain an evaluate_scenario.py."
        )

    pairs = [(wf, sc) for wf in workflows for sc in scenarios]

    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    print(f"Evaluation: {len(workflows)} workflow(s) x {len(scenarios)} scenario(s) "
          f"= {len(pairs)} pair(s), {num_trials} trial(s) each")
    print(f"  Workflows: {workflows}")
    print(f"  Scenarios: {scenarios}")

    if task is not None and len(scenarios) > 1:
        print(
            "\n  WARNING: --task overrides the prompt for ALL scenarios. "
            "This is probably not what you want when evaluating multiple scenarios."
        )

    pair_summaries: list[dict[str, Any]] = []
    wall_start = time.monotonic()

    for i, (wf, sc) in enumerate(pairs, 1):
        pair_dir = out / wf / sc
        pair_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{i}/{len(pairs)}] {wf} / {sc}  ({num_trials} trial(s)) ...")
        pair_start = time.monotonic()

        summary = _run_pair(
            workflow=wf,
            scenario=sc,
            num_trials=num_trials,
            output_dir=str(pair_dir),
            task=task,
            parallel=parallel,
        )

        pair_elapsed = time.monotonic() - pair_start
        rate = summary.get("success_rate", 0.0)
        print(f"  -> {wf}/{sc}: {rate:.0%} success in {pair_elapsed:.1f}s")

        summary.setdefault("workflow", wf)
        summary.setdefault("scenario", sc)
        pair_summaries.append(summary)

    wall_elapsed = time.monotonic() - wall_start

    overall = _build_overall(pair_summaries, workflows, scenarios)

    report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "wall_time_s": round(wall_elapsed, 2),
        "num_trials_per_pair": num_trials,
        "workflows": workflows,
        "scenarios": scenarios,
        "pairs": [
            {
                "workflow": s["workflow"],
                "scenario": s["scenario"],
                "success_rate": s["success_rate"],
                "total_trials": s["total_trials"],
                "outcomes": s.get("outcomes", {}),
                "duration_stats": s.get("duration_stats"),
                "token_stats": s.get("token_stats"),
                "validation_score_stats": s.get("validation_score_stats"),
                "summary_path": f"{s['workflow']}/{s['scenario']}/summary.json",
            }
            for s in pair_summaries
        ],
        "overall": overall,
    }

    report_path = out / "evaluation.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    _print_matrix(pair_summaries, workflows, scenarios, overall)
    print(f"\n  Report: {report_path}")

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the evaluation matrix across workflows and scenarios. "
            "Discovers all available workflows/scenarios by default."
        ),
    )
    parser.add_argument(
        "--workflow", "-w",
        default=None,
        help="Restrict to a single workflow (default: all discovered)",
    )
    parser.add_argument(
        "--scenario", "-s",
        default=None,
        help="Restrict to a single scenario (default: all discovered)",
    )
    parser.add_argument(
        "--num-trials", "-n",
        type=int,
        default=1,
        help="Number of trials per workflow/scenario pair (default: 1)",
    )
    parser.add_argument(
        "--output-dir",
        default="trials",
        help="Base directory for all output (default: ./trials)",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Override the scenario's default prompt (use with --scenario)",
    )
    parser.add_argument(
        "--parallel", "-p",
        type=int,
        default=1,
        help="Max concurrent trials within each pair (default: 1)",
    )
    args = parser.parse_args()

    run_evaluation(
        workflow=args.workflow,
        scenario=args.scenario,
        num_trials=args.num_trials,
        output_dir=args.output_dir,
        task=args.task,
        parallel=args.parallel,
    )


if __name__ == "__main__":
    main()
