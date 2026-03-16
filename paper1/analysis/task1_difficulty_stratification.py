"""
Task 1: Difficulty Stratification Analysis

Tests whether the consistency-correctness correlation survives controlling
for task difficulty (the key confound critique).

Outputs: task1_results.json
"""

import json
import sys
import glob
import statistics
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "hotpotqa"))

from analysis import is_correct, count_unique_sequences, load_results

# Model name -> list of result directories (combined for 200 questions)
MODEL_DIRS = {
    "Llama 3.1 70B": ["hotpotqa/results_llama", "hotpotqa/results_cortex_llama"],
    "Claude Sonnet 4.5": ["hotpotqa/results_claude", "hotpotqa/results_cortex_claude"],
    "GPT-4o": ["hotpotqa/results_gpt4o"],
    "GPT-5": ["hotpotqa/results_gpt5", "hotpotqa/results_cortex_gpt5"],
    "Gemini 3 Pro": ["hotpotqa/results_gemini", "hotpotqa/results_cortex_gemini"],
}

BASE = Path(__file__).resolve().parent.parent.parent


def load_all_model_data():
    """Load all result files for all models, keyed by (model, task_id)."""
    model_data = {}
    for model_name, dirs in MODEL_DIRS.items():
        model_data[model_name] = {}
        for d in dirs:
            full_path = BASE / d
            for f in sorted(full_path.glob("*.json")):
                with open(f) as fh:
                    data = json.load(fh)
                task_id = data["task_id"]
                model_data[model_name][task_id] = data
    return model_data


def compute_difficulty_proxy(model_data):
    """
    For each task_id, compute difficulty as average correctness across ALL
    models and ALL runs. This is model-agnostic.
    """
    # Collect all task_ids
    all_task_ids = set()
    for model_name, tasks in model_data.items():
        all_task_ids.update(tasks.keys())

    difficulty = {}
    for task_id in all_task_ids:
        correct = 0
        total = 0
        for model_name, tasks in model_data.items():
            if task_id in tasks:
                task = tasks[task_id]
                gt = task["answer"]
                for run in task.get("runs", []):
                    total += 1
                    if is_correct(run, gt):
                        correct += 1
        difficulty[task_id] = correct / total if total > 0 else 0.0

    return difficulty


def stratify(difficulty):
    """Bin tasks into Easy (>=0.80), Medium (0.40-0.79), Hard (<0.40)."""
    strata = {"Easy": [], "Medium": [], "Hard": []}
    for task_id, avg_corr in difficulty.items():
        if avg_corr >= 0.80:
            strata["Easy"].append(task_id)
        elif avg_corr >= 0.40:
            strata["Medium"].append(task_id)
        else:
            strata["Hard"].append(task_id)
    return strata


def compute_gap(model_tasks, task_ids, ground_truths):
    """
    For a set of task_ids, compute accuracy gap between consistent
    (<=2 unique sequences) and inconsistent (>=4 unique sequences) tasks.
    """
    consistent_acc = []
    inconsistent_acc = []

    for tid in task_ids:
        if tid not in model_tasks:
            continue
        task = model_tasks[tid]
        gt = ground_truths[tid]
        n_unique = count_unique_sequences(task)
        runs = task.get("runs", [])
        if not runs:
            continue
        acc = sum(1 for r in runs if is_correct(r, gt)) / len(runs)

        if n_unique <= 2:
            consistent_acc.append(acc)
        elif n_unique >= 4:
            inconsistent_acc.append(acc)

    gap = None
    mean_consistent = None
    mean_inconsistent = None

    if consistent_acc:
        mean_consistent = statistics.mean(consistent_acc)
    if inconsistent_acc:
        mean_inconsistent = statistics.mean(inconsistent_acc)
    if mean_consistent is not None and mean_inconsistent is not None:
        gap = mean_consistent - mean_inconsistent

    return {
        "n_consistent": len(consistent_acc),
        "n_inconsistent": len(inconsistent_acc),
        "mean_consistent_acc": mean_consistent,
        "mean_inconsistent_acc": mean_inconsistent,
        "gap_pp": round(gap * 100, 1) if gap is not None else None,
    }


def compute_partial_correlation(model_tasks, difficulty, ground_truths):
    """
    Partial correlation: corr(unique_sequences, accuracy | difficulty).
    Control variable = difficulty proxy (continuous).
    """
    unique_seqs = []
    accuracies = []
    diff_vals = []

    for tid, task in model_tasks.items():
        if tid not in difficulty:
            continue
        gt = ground_truths[tid]
        runs = task.get("runs", [])
        if not runs:
            continue
        n_unique = count_unique_sequences(task)
        acc = sum(1 for r in runs if is_correct(r, gt)) / len(runs)

        unique_seqs.append(n_unique)
        accuracies.append(acc)
        diff_vals.append(difficulty[tid])

    if len(unique_seqs) < 10:
        return {"partial_r": None, "partial_p": None, "n": len(unique_seqs)}

    # Partial correlation: residualize both X and Y on Z
    unique_seqs = np.array(unique_seqs)
    accuracies = np.array(accuracies)
    diff_vals = np.array(diff_vals)

    # Regress unique_seqs on difficulty
    slope_x, intercept_x, _, _, _ = stats.linregress(diff_vals, unique_seqs)
    resid_x = unique_seqs - (slope_x * diff_vals + intercept_x)

    # Regress accuracy on difficulty
    slope_y, intercept_y, _, _, _ = stats.linregress(diff_vals, accuracies)
    resid_y = accuracies - (slope_y * diff_vals + intercept_y)

    # Correlate residuals
    r, p = stats.pearsonr(resid_x, resid_y)

    return {"partial_r": round(r, 3), "partial_p": round(p, 6), "n": len(unique_seqs)}


def main():
    print("=" * 70)
    print("TASK 1: DIFFICULTY STRATIFICATION ANALYSIS")
    print("=" * 70)

    model_data = load_all_model_data()

    # Collect ground truths
    ground_truths = {}
    for model_name, tasks in model_data.items():
        for tid, task in tasks.items():
            ground_truths[tid] = task["answer"]

    # Compute difficulty proxy
    difficulty = compute_difficulty_proxy(model_data)

    # Stratify
    strata = stratify(difficulty)
    print(f"\nDifficulty strata:")
    for s, tids in strata.items():
        print(f"  {s}: {len(tids)} questions")

    # Compute gaps per stratum per model
    results = {"strata": {}, "partial_correlations": {}}

    print(f"\n{'Stratum':<10} {'n':>4}", end="")
    for model in MODEL_DIRS:
        short = model.split()[0] if "Sonnet" not in model else "Claude"
        print(f" | {short:>10}", end="")
    print()
    print("-" * (16 + 13 * len(MODEL_DIRS)))

    for stratum_name in ["Easy", "Medium", "Hard"]:
        task_ids = strata[stratum_name]
        results["strata"][stratum_name] = {"n_questions": len(task_ids), "models": {}}
        print(f"{stratum_name:<10} {len(task_ids):>4}", end="")

        for model_name in MODEL_DIRS:
            gap_info = compute_gap(
                model_data[model_name], task_ids, ground_truths
            )
            results["strata"][stratum_name]["models"][model_name] = gap_info
            gap_str = f"{gap_info['gap_pp']}pp" if gap_info["gap_pp"] is not None else "N/A"
            print(f" | {gap_str:>10}", end="")
        print()

    # Partial correlations
    print(f"\nPartial correlations (unique_seqs vs accuracy | difficulty):")
    for model_name in MODEL_DIRS:
        pc = compute_partial_correlation(
            model_data[model_name], difficulty, ground_truths
        )
        results["partial_correlations"][model_name] = pc
        if pc["partial_r"] is not None:
            sig = "***" if pc["partial_p"] < 0.001 else "**" if pc["partial_p"] < 0.01 else "*" if pc["partial_p"] < 0.05 else "ns"
            print(f"  {model_name}: r={pc['partial_r']:.3f}, p={pc['partial_p']:.4f} {sig} (n={pc['n']})")
        else:
            print(f"  {model_name}: insufficient data (n={pc['n']})")

    # Save
    output_path = Path(__file__).parent / "task1_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
