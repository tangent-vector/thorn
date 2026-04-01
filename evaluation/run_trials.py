"""Batch runner for evaluation trials.

Runs N independent trials of a given workflow/scenario combination,
collects structured results, and produces summary statistics.

Usage::

    python run_trials.py --workflow thorn_cpp --scenario calc --num-trials 5
    python run_trials.py -w thorn_cpp -s calc -n 10 -p 3 --output-dir ./results
    python run_trials.py -w thorn_cpp -s calc -n 1 --task "implement a scientific calculator"
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

TRIAL_SCRIPT = Path(__file__).resolve().parent / "trial.py"


def _run_one(
    workflow: str,
    scenario: str,
    output_dir: str,
    trial_id: str,
    task: str | None,
) -> dict[str, Any]:
    """Run a single trial as a subprocess and return its result dict."""
    cmd = [
        sys.executable,
        str(TRIAL_SCRIPT),
        "--workflow", workflow,
        "--scenario", scenario,
        "--output-dir", output_dir,
        "--trial-id", trial_id,
        "--quiet",
    ]
    if task is not None:
        cmd.extend(["--task", task])

    proc = subprocess.run(cmd, capture_output=True, text=True)

    result_file = Path(output_dir) / trial_id / "result.json"
    if result_file.exists():
        return json.loads(result_file.read_text(encoding="utf-8"))

    return {
        "trial_id": trial_id,
        "outcome": "agent_error",
        "error": f"subprocess exited {proc.returncode}: {(proc.stderr or '')[:500]}",
        "duration_s": 0.0,
        "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "build_ok": False,
        "binary_runs": [],
        "validation_score": 0.0,
    }


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "stdev": None}
    return {
        "mean": round(statistics.mean(values), 2),
        "median": round(statistics.median(values), 2),
        "stdev": round(statistics.stdev(values), 2) if len(values) > 1 else 0.0,
    }


def _compute_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate statistics from a list of trial result dicts."""
    total = len(results)

    outcomes: dict[str, int] = {}
    for r in results:
        o = r["outcome"]
        outcomes[o] = outcomes.get(o, 0) + 1

    successes = [r for r in results if r["outcome"] == "success"]
    durations = [r["duration_s"] for r in successes]
    tokens = [
        r["token_usage"]["total_tokens"]
        for r in successes
        if r.get("token_usage", {}).get("total_tokens")
    ]
    scores = [
        r["validation_score"]
        for r in results
        if r.get("validation_score") is not None
    ]

    return {
        "total_trials": total,
        "outcomes": outcomes,
        "success_rate": round(len(successes) / total, 3) if total else 0.0,
        "duration_stats": _stats(durations),
        "token_stats": _stats(tokens),
        "validation_score_stats": _stats(scores),
        "trials": [
            {
                "trial_id": r.get("trial_id"),
                "outcome": r.get("outcome"),
                "duration_s": r.get("duration_s"),
                "total_tokens": r.get("token_usage", {}).get("total_tokens"),
                "build_ok": r.get("build_ok"),
                "validation_score": r.get("validation_score"),
            }
            for r in results
        ],
    }


def _print_summary(summary: dict[str, Any]) -> None:
    total = summary["total_trials"]
    outcomes = summary["outcomes"]
    successes = outcomes.get("success", 0)

    print(f"\n{'=' * 55}")
    print(f"  Trial Summary  ({total} trial{'s' if total != 1 else ''})")
    print(f"{'=' * 55}")
    print(f"  Success rate: {successes}/{total} ({summary['success_rate']:.1%})")
    for outcome, count in sorted(outcomes.items()):
        print(f"    {outcome}: {count}")

    def _show(label: str, stats: dict[str, float | None]) -> None:
        if stats["mean"] is None:
            print(f"\n  {label}: (no data)")
            return
        print(f"\n  {label}:")
        print(f"    Mean:   {stats['mean']}")
        print(f"    Median: {stats['median']}")
        print(f"    Stdev:  {stats['stdev']}")

    _show("Duration (s)", summary["duration_stats"])
    _show("Token usage", summary["token_stats"])
    _show("Validation score", summary["validation_score_stats"])

    print(f"\n  Per-trial breakdown:")
    for t in summary["trials"]:
        score = t.get("validation_score")
        score_str = f"  score={score:.2f}" if score is not None else ""
        tokens = t.get("total_tokens") or 0
        dur = t.get("duration_s") or 0
        print(f"    {t['trial_id']}: {t['outcome']}  "
              f"{dur:.1f}s  {tokens} tok{score_str}")

    print(f"{'=' * 55}")


def run_trials(
    workflow: str,
    scenario: str,
    num_trials: int = 3,
    output_dir: str = "trials",
    task: str | None = None,
    parallel: int = 1,
) -> dict[str, Any]:
    """Run *num_trials* independent trials and write a summary.

    Returns the summary dict (also written to ``summary.json``).
    """
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    trial_ids = [f"trial_{i:03d}" for i in range(1, num_trials + 1)]

    results: list[dict[str, Any]] = []

    if parallel <= 1:
        for i, tid in enumerate(trial_ids, 1):
            print(f"[{i}/{num_trials}] Running {tid} ...")
            results.append(_run_one(workflow, scenario, str(out), tid, task))
    else:
        print(f"Running {num_trials} trials ({parallel} concurrent) ...")
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {
                pool.submit(
                    _run_one, workflow, scenario, str(out), tid, task,
                ): tid
                for tid in trial_ids
            }
            for future in as_completed(futures):
                tid = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    results.append({
                        "trial_id": tid,
                        "outcome": "agent_error",
                        "error": str(exc),
                        "duration_s": 0.0,
                        "token_usage": {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                        },
                        "build_ok": False,
                        "binary_runs": [],
                        "validation_score": 0.0,
                    })
                print(f"  Completed {tid}: {results[-1]['outcome']}")
        results.sort(key=lambda r: r.get("trial_id", ""))

    summary = _compute_summary(results)
    summary["workflow"] = workflow
    summary["scenario"] = scenario
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    _print_summary(summary)
    print(f"\n  Results in: {out}")
    print(f"  Summary:    {summary_path}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multiple evaluation trials.",
    )
    parser.add_argument(
        "--workflow", "-w",
        required=True,
        help="Name of the workflow to evaluate (e.g. 'thorn_cpp')",
    )
    parser.add_argument(
        "--scenario", "-s",
        required=True,
        help="Name of the scenario (e.g. 'calc')",
    )
    parser.add_argument(
        "--num-trials", "-n",
        type=int,
        default=3,
        help="Number of trials to run (default: 3)",
    )
    parser.add_argument(
        "--output-dir",
        default="trials",
        help="Base directory for trial results (default: ./trials)",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Override the scenario's default prompt",
    )
    parser.add_argument(
        "--parallel", "-p",
        type=int,
        default=1,
        help="Max concurrent trials (default: 1, sequential)",
    )
    args = parser.parse_args()

    run_trials(
        workflow=args.workflow,
        scenario=args.scenario,
        num_trials=args.num_trials,
        output_dir=args.output_dir,
        task=args.task,
        parallel=args.parallel,
    )


if __name__ == "__main__":
    main()
