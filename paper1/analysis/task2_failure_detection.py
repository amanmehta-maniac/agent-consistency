"""
Task 2: Failure Detection Experiment

Frames consistency metrics as binary classifiers for detecting incorrect
majority answers. Computes precision, recall, F1, and AUROC.

Outputs: task2_results.json
"""

import json
import sys
import math
import statistics
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "hotpotqa"))

from analysis import is_correct, count_unique_sequences, first_divergence_point

# Model name -> list of result directories
MODEL_DIRS = {
    "Llama 3.1 70B": ["hotpotqa/results_llama", "hotpotqa/results_cortex_llama"],
    "Claude Sonnet 4.5": ["hotpotqa/results_claude", "hotpotqa/results_cortex_claude"],
    "GPT-4o": ["hotpotqa/results_gpt4o"],
    "GPT-5": ["hotpotqa/results_gpt5", "hotpotqa/results_cortex_gpt5"],
    "Gemini 3 Pro": ["hotpotqa/results_gemini", "hotpotqa/results_cortex_gemini"],
}

BASE = Path(__file__).resolve().parent.parent.parent


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


def majority_answer_correct(task_data):
    """Check if the majority answer across runs is correct."""
    gt = task_data["answer"]
    runs = task_data.get("runs", [])
    if not runs:
        return False

    answers = []
    for r in runs:
        ans = r.get("final_answer", "")
        if ans:
            answers.append(str(ans).lower().strip())
        else:
            answers.append("")

    if not answers:
        return False

    counter = Counter(answers)
    majority_ans = counter.most_common(1)[0][0]

    # Check correctness using fuzzy match
    gt_norm = str(gt).lower().strip().rstrip(".,!?")
    majority_norm = majority_ans.rstrip(".,!?")
    return gt_norm in majority_norm or majority_norm in gt_norm


def compute_features(task_data):
    """Compute all consistency features for a task."""
    runs = task_data.get("runs", [])
    if not runs:
        return None

    # 1. unique_sequences
    n_unique = count_unique_sequences(task_data)

    # 2. step_variance_ratio
    step_counts = [len(r.get("steps", [])) for r in runs]
    mean_steps = statistics.mean(step_counts) if step_counts else 0
    if mean_steps > 0:
        step_var_ratio = (max(step_counts) - min(step_counts)) / mean_steps
    else:
        step_var_ratio = 0.0

    # 3. first_divergence_step
    div = first_divergence_point(runs)
    div_step = div.get("divergence_step")
    if div_step is None:
        div_step = 999  # no divergence = very high

    # 4. answer_entropy
    answers = []
    for r in runs:
        ans = r.get("final_answer", "")
        answers.append(str(ans).lower().strip() if ans else "")
    counter = Counter(answers)
    total = len(answers)
    entropy = 0.0
    for count in counter.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)

    return {
        "unique_sequences": n_unique,
        "step_variance_ratio": step_var_ratio,
        "first_divergence_step": div_step,
        "answer_entropy": entropy,
    }


def evaluate_threshold(labels, feature_values, threshold, higher_is_flagged=True):
    """Evaluate a threshold classifier."""
    if higher_is_flagged:
        predictions = [1 if v > threshold else 0 for v in feature_values]
    else:
        predictions = [1 if v <= threshold else 0 for v in feature_values]

    tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "n_flagged": sum(predictions),
    }


def main():
    print("=" * 70)
    print("TASK 2: FAILURE DETECTION EXPERIMENT")
    print("=" * 70)

    model_data = load_all_model_data()

    feature_configs = {
        "unique_sequences": {
            "thresholds": [2, 3, 4, 5],
            "higher_is_flagged": True,
        },
        "step_variance_ratio": {
            "thresholds": [0.3, 0.5, 0.7, 1.0],
            "higher_is_flagged": True,
        },
        "first_divergence_step": {
            "thresholds": [2],
            "higher_is_flagged": False,  # flag if divergence <= 2
        },
        "answer_entropy": {
            "thresholds": [0.5, 1.0, 1.5, 2.0],
            "higher_is_flagged": True,
        },
    }

    results = {}

    for model_name in MODEL_DIRS:
        tasks = model_data[model_name]
        labels = []
        features_dict = {k: [] for k in feature_configs}

        for tid, task in tasks.items():
            feats = compute_features(task)
            if feats is None:
                continue
            # Label: 1 = majority answer is INCORRECT (failure)
            is_failure = 0 if majority_answer_correct(task) else 1
            labels.append(is_failure)
            for feat_name in feature_configs:
                features_dict[feat_name].append(feats[feat_name])

        n_total = len(labels)
        n_incorrect = sum(labels)
        baseline_precision = n_incorrect / n_total if n_total > 0 else 0

        print(f"\n{'='*50}")
        print(f"Model: {model_name}")
        print(f"  Tasks: {n_total}, Incorrect: {n_incorrect} ({100*baseline_precision:.1f}%)")
        print(f"  Random baseline precision: {baseline_precision:.3f}")

        model_results = {
            "n_tasks": n_total,
            "n_incorrect": n_incorrect,
            "baseline_precision": round(baseline_precision, 3),
            "features": {},
        }

        for feat_name, config in feature_configs.items():
            feat_vals = features_dict[feat_name]
            best_f1 = -1
            best_result = None

            # AUROC
            try:
                if feat_name == "first_divergence_step":
                    # Lower divergence step = more inconsistent = more likely failure
                    auroc = roc_auc_score(labels, [-v for v in feat_vals])
                else:
                    auroc = roc_auc_score(labels, feat_vals)
            except ValueError:
                auroc = None

            for threshold in config["thresholds"]:
                result = evaluate_threshold(
                    labels, feat_vals, threshold, config["higher_is_flagged"]
                )
                if result["f1"] > best_f1:
                    best_f1 = result["f1"]
                    best_result = {**result, "threshold": threshold}

            if best_result:
                best_result["auroc"] = round(auroc, 3) if auroc is not None else None
                model_results["features"][feat_name] = best_result
                print(f"  {feat_name} (threshold>{best_result['threshold']}): "
                      f"P={best_result['precision']:.3f} R={best_result['recall']:.3f} "
                      f"F1={best_result['f1']:.3f} AUROC={best_result['auroc']}")

        results[model_name] = model_results

    # Summary table for paper (Llama primary)
    print(f"\n{'='*70}")
    print("SUMMARY TABLE (for paper — Llama 3.1 70B primary)")
    print(f"{'='*70}")
    print(f"{'Feature':<25} {'Prec':>6} {'Recall':>6} {'F1':>6} {'AUROC':>6}")
    print("-" * 55)
    if "Llama 3.1 70B" in results:
        for feat_name in feature_configs:
            r = results["Llama 3.1 70B"]["features"].get(feat_name, {})
            print(f"{feat_name:<25} {r.get('precision',0):>6.3f} {r.get('recall',0):>6.3f} "
                  f"{r.get('f1',0):>6.3f} {r.get('auroc','N/A'):>6}")

    # Save
    output_path = Path(__file__).parent / "task2_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
