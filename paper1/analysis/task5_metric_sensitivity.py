"""
Task 5: Correctness Metric Sensitivity Analysis

Tests whether main results hold under three different correctness metrics:
fuzzy match (current), exact match (EM), and token F1.

Outputs: task5_results.json
"""

import json
import re
import sys
import string
import statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "hotpotqa"))

from analysis import count_unique_sequences, is_correct as fuzzy_correct

MODEL_DIRS = {
    "Llama 3.1 70B": ["hotpotqa/results_llama", "hotpotqa/results_cortex_llama"],
    "Claude Sonnet 4.5": ["hotpotqa/results_claude", "hotpotqa/results_cortex_claude"],
    "GPT-4o": ["hotpotqa/results_gpt4o"],
    "GPT-5": ["hotpotqa/results_gpt5", "hotpotqa/results_cortex_gpt5"],
    "Gemini 3 Pro": ["hotpotqa/results_gemini", "hotpotqa/results_cortex_gemini"],
}

BASE = Path(__file__).resolve().parent.parent.parent


def normalize_hotpotqa(text):
    """Standard HotpotQA normalization: lowercase, strip articles/punct/whitespace."""
    if text is None:
        return ""
    text = str(text).lower()
    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Collapse whitespace
    text = ' '.join(text.split())
    return text.strip()


def exact_match(run, ground_truth):
    """Standard HotpotQA exact match after normalization."""
    answer = run.get("final_answer", "")
    if not answer:
        return False
    return normalize_hotpotqa(answer) == normalize_hotpotqa(ground_truth)


def token_f1(run, ground_truth, threshold=0.5):
    """Standard HotpotQA token F1. Returns True if F1 > threshold."""
    answer = run.get("final_answer", "")
    if not answer:
        return False

    pred_tokens = normalize_hotpotqa(answer).split()
    gold_tokens = normalize_hotpotqa(ground_truth).split()

    if not pred_tokens or not gold_tokens:
        return False

    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return False

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1 > threshold


def load_all_model_data():
    model_data = {}
    for model_name, dirs in MODEL_DIRS.items():
        model_data[model_name] = {}
        for d in dirs:
            full_path = BASE / d
            for f in sorted(full_path.glob("*.json")):
                with open(f) as fh:
                    data = json.load(fh)
                model_data[model_name][data["task_id"]] = data
    return model_data


METRICS = {
    "fuzzy": fuzzy_correct,
    "exact_match": exact_match,
    "token_f1": token_f1,
}


def compute_table1(model_data, metric_fn):
    """Compute Table 1 equivalent under a given correctness metric."""
    rows = {}
    for model_name, tasks in model_data.items():
        all_acc = []
        all_unique_seqs = []
        all_mean_steps = []
        all_step_var = []

        for tid, task in tasks.items():
            gt = task["answer"]
            runs = task.get("runs", [])
            if not runs:
                continue

            # Accuracy under this metric
            correct = sum(1 for r in runs if metric_fn(r, gt))
            acc = correct / len(runs)
            all_acc.append(acc)

            # Consistency metrics (same regardless of correctness metric)
            n_unique = count_unique_sequences(task)
            all_unique_seqs.append(n_unique)

            step_counts = [len(r.get("steps", [])) for r in runs]
            all_mean_steps.append(statistics.mean(step_counts))
            mean_s = statistics.mean(step_counts)
            if mean_s > 0:
                var_ratio = (max(step_counts) - min(step_counts)) / mean_s
            else:
                var_ratio = 0.0
            all_step_var.append(var_ratio)

        rows[model_name] = {
            "n_questions": len(all_acc),
            "accuracy": round(statistics.mean(all_acc) * 100, 1) if all_acc else 0,
            "mean_unique_seqs": round(statistics.mean(all_unique_seqs), 1) if all_unique_seqs else 0,
            "mean_steps": round(statistics.mean(all_mean_steps), 1) if all_mean_steps else 0,
            "mean_step_var": round(statistics.mean(all_step_var) * 100, 1) if all_step_var else 0,
        }

    return rows


def compute_consistency_gap(model_data, metric_fn):
    """Compute accuracy gap between consistent and inconsistent tasks."""
    gaps = {}
    for model_name, tasks in model_data.items():
        consistent_acc = []
        inconsistent_acc = []

        for tid, task in tasks.items():
            gt = task["answer"]
            runs = task.get("runs", [])
            if not runs:
                continue

            n_unique = count_unique_sequences(task)
            acc = sum(1 for r in runs if metric_fn(r, gt)) / len(runs)

            if n_unique <= 2:
                consistent_acc.append(acc)
            elif n_unique >= 4:
                inconsistent_acc.append(acc)

        gap = None
        if consistent_acc and inconsistent_acc:
            gap = statistics.mean(consistent_acc) - statistics.mean(inconsistent_acc)

        gaps[model_name] = {
            "consistent_acc": round(statistics.mean(consistent_acc) * 100, 1) if consistent_acc else None,
            "inconsistent_acc": round(statistics.mean(inconsistent_acc) * 100, 1) if inconsistent_acc else None,
            "gap_pp": round(gap * 100, 1) if gap is not None else None,
            "n_consistent": len(consistent_acc),
            "n_inconsistent": len(inconsistent_acc),
        }

    return gaps


def main():
    print("=" * 70)
    print("TASK 5: CORRECTNESS METRIC SENSITIVITY ANALYSIS")
    print("=" * 70)

    model_data = load_all_model_data()

    results = {"table1": {}, "consistency_gaps": {}}

    for metric_name, metric_fn in METRICS.items():
        print(f"\n{'='*50}")
        print(f"Metric: {metric_name}")
        print(f"{'='*50}")

        table1 = compute_table1(model_data, metric_fn)
        results["table1"][metric_name] = table1

        print(f"\n{'Model':<22} {'Acc%':>5} {'UniSeq':>6} {'Steps':>5} {'Var%':>5}")
        print("-" * 50)
        for model_name, row in table1.items():
            print(f"{model_name:<22} {row['accuracy']:>5.1f} {row['mean_unique_seqs']:>6.1f} "
                  f"{row['mean_steps']:>5.1f} {row['mean_step_var']:>5.1f}")

        gaps = compute_consistency_gap(model_data, metric_fn)
        results["consistency_gaps"][metric_name] = gaps

        print(f"\nConsistency-correctness gap:")
        for model_name, gap in gaps.items():
            gap_str = f"{gap['gap_pp']}pp" if gap['gap_pp'] is not None else "N/A"
            print(f"  {model_name}: {gap_str} "
                  f"(consistent={gap['consistent_acc']}%, inconsistent={gap['inconsistent_acc']}%)")

    # Cross-metric comparison
    print(f"\n{'='*70}")
    print("MODEL RANKING COMPARISON ACROSS METRICS")
    print(f"{'='*70}")
    for metric_name in METRICS:
        table1 = results["table1"][metric_name]
        ranked = sorted(table1.items(), key=lambda x: -x[1]["accuracy"])
        ranking = [m.split()[0] if "Sonnet" not in m else "Claude" for m, _ in ranked]
        print(f"  {metric_name:<15}: {' > '.join(ranking)}")

    # Save
    output_path = Path(__file__).parent / "task5_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
