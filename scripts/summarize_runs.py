#!/usr/bin/env python3
"""
Summarize Paper 2 SWE-bench experiment results.

Parses results directories + SWE-bench evaluation reports and outputs:
  - Per-task accuracy (resolved / runs)
  - Per-task mean steps + CV
  - Per-model aggregates
  - p90/p95 steps

Usage:
    python scripts/summarize_runs.py

    # Custom directories:
    python scripts/summarize_runs.py \
        --models claude:swe-bench/results_claude_10:swe-bench/claude.claude-local-run{run}.json \
        --models gpt5:swe-bench/results_gpt5_snowflake:swe-bench/gpt5.gpt5-local-run{run}.json
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


# ============================================================================
# Data Loading
# ============================================================================

def load_results(results_dir: Path) -> dict:
    """Load all task result JSONs from a directory. Returns {task_id: data}."""
    results = {}
    for f in sorted(results_dir.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
        results[data["task_id"]] = data
    return results


def load_eval_reports(report_pattern: str, n_runs: int = 5) -> dict:
    """
    Load SWE-bench evaluation reports.
    
    Args:
        report_pattern: Path pattern with {run} placeholder,
                        e.g. "swe-bench/claude.claude-local-run{run}.json"
        n_runs: Number of runs to load
    
    Returns:
        {task_id: [bool, bool, ...]} — per-run resolved status
    """
    resolved_per_task = defaultdict(lambda: [False] * n_runs)

    for run_idx in range(n_runs):
        run_num = run_idx + 1
        report_path = Path(report_pattern.format(run=run_num))
        if not report_path.exists():
            print(f"  WARNING: {report_path} not found, skipping run {run_num}")
            continue
        with open(report_path) as f:
            report = json.load(f)
        for task_id in report.get("resolved_ids", report.get("resolved", [])):
            resolved_per_task[task_id][run_idx] = True

    return dict(resolved_per_task)


# ============================================================================
# Metrics
# ============================================================================

def compute_task_metrics(task_data: dict, resolved_list: list) -> dict:
    """Compute metrics for a single task."""
    runs = task_data["runs"]
    n_runs = len(runs)
    
    # Step counts
    step_counts = [r["n_steps"] for r in runs]
    mean_steps = np.mean(step_counts)
    std_steps = np.std(step_counts, ddof=1) if n_runs > 1 else 0.0
    cv = (std_steps / mean_steps * 100) if mean_steps > 0 else 0.0
    
    # Accuracy from eval reports
    resolved_count = sum(resolved_list)
    accuracy = resolved_count / n_runs
    
    # Valid patches (submitted)
    valid_patches = sum(1 for r in runs if r.get("exit_status") == "Submitted")
    
    # Cost
    costs = [r.get("total_cost", 0.0) for r in runs]
    mean_cost = np.mean(costs)
    
    return {
        "task_id": task_data["task_id"],
        "n_runs": n_runs,
        "step_counts": step_counts,
        "mean_steps": float(mean_steps),
        "std_steps": float(std_steps),
        "cv": float(cv),
        "resolved_count": resolved_count,
        "accuracy": float(accuracy),
        "valid_patches": valid_patches,
        "mean_cost": float(mean_cost),
    }


def compute_model_aggregates(task_metrics: list) -> dict:
    """Compute model-level aggregates from per-task metrics."""
    all_steps = []
    for tm in task_metrics:
        all_steps.extend(tm["step_counts"])
    
    all_steps_arr = np.array(all_steps)
    cvs = [tm["cv"] for tm in task_metrics]
    
    total_resolved = sum(tm["resolved_count"] for tm in task_metrics)
    total_runs = sum(tm["n_runs"] for tm in task_metrics)
    total_valid = sum(tm["valid_patches"] for tm in task_metrics)
    
    return {
        "n_tasks": len(task_metrics),
        "total_runs": total_runs,
        "accuracy": total_resolved / total_runs if total_runs > 0 else 0.0,
        "accuracy_str": f"{total_resolved}/{total_runs}",
        "valid_patches": f"{total_valid}/{total_runs}",
        "mean_steps": float(np.mean(all_steps_arr)),
        "median_steps": float(np.median(all_steps_arr)),
        "p90_steps": float(np.percentile(all_steps_arr, 90)),
        "p95_steps": float(np.percentile(all_steps_arr, 95)),
        "mean_cv": float(np.mean(cvs)),
        "std_cv": float(np.std(cvs, ddof=1)) if len(cvs) > 1 else 0.0,
        "mean_cost": float(np.mean([tm["mean_cost"] for tm in task_metrics])),
    }


# ============================================================================
# Display
# ============================================================================

def short_task_id(task_id: str) -> str:
    """astropy__astropy-12907 → 12907"""
    return task_id.split("-")[-1]


def print_per_task_table(model_name: str, task_metrics: list):
    """Print per-task breakdown."""
    print(f"\n{'─' * 65}")
    print(f"  {model_name} — Per-Task Results")
    print(f"{'─' * 65}")
    print(f"  {'Task':>8}  {'Accuracy':>10}  {'Mean Steps':>11}  {'CV (%)':>8}  {'Cost':>7}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*11}  {'─'*8}  {'─'*7}")
    
    for tm in task_metrics:
        tid = short_task_id(tm["task_id"])
        acc = f"{tm['resolved_count']}/{tm['n_runs']}"
        print(f"  {tid:>8}  {acc:>10}  {tm['mean_steps']:>11.1f}  {tm['cv']:>8.1f}  ${tm['mean_cost']:>6.2f}")


def print_model_summary(model_name: str, agg: dict):
    """Print model-level summary."""
    print(f"\n{'━' * 65}")
    print(f"  {model_name} — Aggregates")
    print(f"{'━' * 65}")
    print(f"  Tasks:           {agg['n_tasks']}")
    print(f"  Total runs:      {agg['total_runs']}")
    print(f"  Accuracy:        {agg['accuracy_str']}  ({agg['accuracy']:.0%})")
    print(f"  Valid patches:   {agg['valid_patches']}")
    print(f"  Mean steps:      {agg['mean_steps']:.1f}")
    print(f"  Median steps:    {agg['median_steps']:.1f}")
    print(f"  p90 steps:       {agg['p90_steps']:.1f}")
    print(f"  p95 steps:       {agg['p95_steps']:.1f}")
    print(f"  Mean CV:         {agg['mean_cv']:.1f}%")
    print(f"  Mean cost/run:   ${agg['mean_cost']:.2f}")


def print_cross_model_comparison(all_aggs: dict):
    """Print side-by-side comparison table."""
    models = list(all_aggs.keys())
    
    print(f"\n{'━' * 75}")
    print(f"  CROSS-MODEL COMPARISON")
    print(f"{'━' * 75}")
    
    header = f"  {'Metric':<20}"
    for m in models:
        header += f"  {m:>14}"
    print(header)
    print(f"  {'─'*20}" + f"  {'─'*14}" * len(models))
    
    rows = [
        ("Accuracy", lambda a: f"{a['accuracy']:.0%} ({a['accuracy_str']})"),
        ("Mean CV (%)", lambda a: f"{a['mean_cv']:.1f}%"),
        ("Mean Steps", lambda a: f"{a['mean_steps']:.1f}"),
        ("Median Steps", lambda a: f"{a['median_steps']:.1f}"),
        ("p90 Steps", lambda a: f"{a['p90_steps']:.1f}"),
        ("p95 Steps", lambda a: f"{a['p95_steps']:.1f}"),
        ("Valid Patches", lambda a: a['valid_patches']),
        ("Cost/Run", lambda a: f"${a['mean_cost']:.2f}"),
    ]
    
    for label, fmt_fn in rows:
        row = f"  {label:<20}"
        for m in models:
            row += f"  {fmt_fn(all_aggs[m]):>14}"
        print(row)


# ============================================================================
# Default Paper 2 Configuration
# ============================================================================

DEFAULT_MODELS = {
    "Claude": {
        "results_dir": "swe-bench/results_claude_10",
        "eval_pattern": "swe-bench/claude.claude-local-run{run}.json",
    },
    "GPT-5": {
        "results_dir": "swe-bench/results_gpt5_snowflake",
        "eval_pattern": "swe-bench/gpt5.gpt5-local-run{run}.json",
    },
    "Llama": {
        "results_dir": "swe-bench/results_llama_10",
        "eval_pattern": "swe-bench/llama.llama-local-run{run}.json",
    },
}


# ============================================================================
# Main
# ============================================================================

def summarize_model(model_name: str, results_dir: str, eval_pattern: str,
                    base_dir: Path, n_runs: int = 5) -> tuple:
    """Summarize a single model. Returns (task_metrics, aggregates)."""
    rdir = base_dir / results_dir
    epattern = str(base_dir / eval_pattern)
    
    print(f"\nLoading {model_name}...")
    print(f"  Results dir:  {rdir}")
    print(f"  Eval pattern: {epattern}")
    
    results = load_results(rdir)
    resolved_map = load_eval_reports(epattern, n_runs=n_runs)
    
    task_metrics = []
    for task_id in sorted(results.keys()):
        resolved_list = resolved_map.get(task_id, [False] * n_runs)
        tm = compute_task_metrics(results[task_id], resolved_list)
        task_metrics.append(tm)
    
    agg = compute_model_aggregates(task_metrics)
    
    print_per_task_table(model_name, task_metrics)
    print_model_summary(model_name, agg)
    
    return task_metrics, agg


def main():
    parser = argparse.ArgumentParser(description="Summarize SWE-bench consistency experiment results")
    parser.add_argument("--base-dir", type=str, default=None,
                        help="Base directory (default: repo root)")
    parser.add_argument("--models", action="append", default=None,
                        help="Model spec: name:results_dir:eval_pattern (can repeat)")
    parser.add_argument("--n-runs", type=int, default=5, help="Number of runs per task")
    args = parser.parse_args()
    
    # Determine base directory
    if args.base_dir:
        base_dir = Path(args.base_dir)
    else:
        # Walk up from script location to find repo root
        base_dir = Path(__file__).resolve().parent.parent
    
    # Parse model specs
    if args.models:
        models = {}
        for spec in args.models:
            parts = spec.split(":")
            if len(parts) != 3:
                parser.error(f"Invalid model spec: {spec}. Expected name:results_dir:eval_pattern")
            models[parts[0]] = {"results_dir": parts[1], "eval_pattern": parts[2]}
    else:
        models = DEFAULT_MODELS
    
    print("=" * 75)
    print("  SWE-BENCH CONSISTENCY — RESULTS SUMMARY")
    print("=" * 75)
    print(f"  Base dir: {base_dir}")
    print(f"  Models:   {', '.join(models.keys())}")
    print(f"  Runs:     {args.n_runs} per task")
    
    all_aggs = {}
    for model_name, config in models.items():
        _, agg = summarize_model(
            model_name,
            config["results_dir"],
            config["eval_pattern"],
            base_dir,
            n_runs=args.n_runs,
        )
        all_aggs[model_name] = agg
    
    # Cross-model comparison
    if len(all_aggs) > 1:
        print_cross_model_comparison(all_aggs)
    
    print(f"\n{'=' * 75}")
    print("  DONE")
    print(f"{'=' * 75}")


if __name__ == "__main__":
    main()
