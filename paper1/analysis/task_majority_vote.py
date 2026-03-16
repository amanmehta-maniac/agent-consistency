"""
Task A: Majority-Vote Intervention Analysis

Simulates majority-vote intervention at various budgets (k=1,3,5,7,10)
using existing 10 runs per question. Also evaluates selective prediction
(abstain when runs disagree).

Uses common/analysis.py for is_correct() and load_results().
"""

import sys
import json
import random
import statistics
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from common.analysis import load_results, is_correct

MODELS = {
    "Llama 3.1 70B": ["hotpotqa/results_llama", "hotpotqa/results_cortex_llama"],
    "Claude Sonnet 4.5": ["hotpotqa/results_claude", "hotpotqa/results_cortex_claude"],
    "GPT-5": ["hotpotqa/results_gpt5", "hotpotqa/results_cortex_gpt5"],
    "GPT-4o": ["hotpotqa/results_gpt4o"],
    "Gemini 3 Pro": ["hotpotqa/results_gemini", "hotpotqa/results_cortex_gemini"],
}

# Difficulty strata from task1 results
DIFFICULTY_THRESHOLDS = {"Easy": 0.80, "Medium": 0.40}  # Easy >= 0.80, Medium >= 0.40, Hard < 0.40

BASE_DIR = Path(__file__).resolve().parent.parent.parent

random.seed(42)
np.random.seed(42)


def get_majority_answer(answers):
    """Return the majority (most common) answer from a list."""
    if not answers:
        return None
    counter = Counter(answers)
    return counter.most_common(1)[0][0]


def normalize_for_voting(answer):
    """Normalize answer for voting comparison."""
    if answer is None:
        return ""
    return str(answer).lower().strip().rstrip(".,!?")


def load_all_model_data():
    """Load data for all models."""
    all_data = {}
    for model_name, dirs in MODELS.items():
        tasks = []
        seen_ids = set()
        for d in dirs:
            full_path = BASE_DIR / d
            if full_path.exists():
                for task in load_results(str(full_path)):
                    tid = task.get("task_id", "")
                    if tid not in seen_ids:
                        seen_ids.add(tid)
                        tasks.append(task)
        all_data[model_name] = tasks
        print(f"  {model_name}: {len(tasks)} tasks loaded")
    return all_data


def compute_difficulty_strata(all_data):
    """Compute per-question difficulty proxy (mean correctness across all models)."""
    # Collect all task_ids
    task_correctness = {}  # task_id -> list of per-run correctness values
    
    for model_name, tasks in all_data.items():
        for task in tasks:
            tid = task.get("task_id", "")
            gt = task.get("answer", "")
            runs = task.get("runs", [])
            
            if tid not in task_correctness:
                task_correctness[tid] = []
            
            for run in runs:
                task_correctness[tid].append(1.0 if is_correct(run, gt) else 0.0)
    
    # Compute mean correctness per task
    difficulty = {}
    for tid, scores in task_correctness.items():
        mean_corr = statistics.mean(scores) if scores else 0.0
        if mean_corr >= DIFFICULTY_THRESHOLDS["Easy"]:
            difficulty[tid] = "Easy"
        elif mean_corr >= DIFFICULTY_THRESHOLDS["Medium"]:
            difficulty[tid] = "Medium"
        else:
            difficulty[tid] = "Hard"
    
    return difficulty


def majority_vote_accuracy(tasks, k, n_bootstrap=500):
    """
    Compute majority-vote accuracy for budget k.
    
    For each task, subsample k runs, take majority answer, check correctness.
    Bootstrap over n_bootstrap iterations (except k=10 which is deterministic).
    
    Returns: (mean_accuracy, ci_low, ci_high)
    """
    if k == 10:
        # Deterministic: use all 10 runs
        correct = 0
        total = 0
        for task in tasks:
            gt = task.get("answer", "")
            runs = task.get("runs", [])
            if len(runs) < 10:
                continue
            
            # Get all answers, normalize for voting
            answers = [normalize_for_voting(r.get("final_answer", "")) for r in runs]
            majority = get_majority_answer(answers)
            
            # Check if majority answer is correct (using is_correct logic)
            gt_norm = normalize_for_voting(gt)
            is_corr = gt_norm in majority or majority in gt_norm if majority else False
            
            correct += int(is_corr)
            total += 1
        
        acc = correct / total if total > 0 else 0.0
        return acc, acc, acc, total  # No CI for deterministic
    
    accuracies = []
    n_tasks = 0
    
    for _ in range(n_bootstrap):
        correct = 0
        total = 0
        for task in tasks:
            gt = task.get("answer", "")
            runs = task.get("runs", [])
            if len(runs) < k:
                continue
            
            # Subsample k runs
            sampled = random.sample(runs, k)
            answers = [normalize_for_voting(r.get("final_answer", "")) for r in sampled]
            majority = get_majority_answer(answers)
            
            gt_norm = normalize_for_voting(gt)
            is_corr = gt_norm in majority or majority in gt_norm if majority else False
            
            correct += int(is_corr)
            total += 1
        
        acc = correct / total if total > 0 else 0.0
        accuracies.append(acc)
        n_tasks = total
    
    mean_acc = statistics.mean(accuracies)
    ci_low = np.percentile(accuracies, 2.5)
    ci_high = np.percentile(accuracies, 97.5)
    
    return mean_acc, ci_low, ci_high, n_tasks


def selective_prediction(tasks, k, threshold):
    """
    Selective prediction: only answer when >= threshold of k runs agree.
    
    Returns: (accuracy_on_answered, coverage, n_answered, n_total)
    """
    correct = 0
    answered = 0
    total = 0
    
    # Use bootstrap to get stable estimates for k < 10
    if k < 10:
        n_bootstrap = 500
        all_correct = []
        all_answered = []
        all_total = []
        
        for _ in range(n_bootstrap):
            b_correct = 0
            b_answered = 0
            b_total = 0
            
            for task in tasks:
                gt = task.get("answer", "")
                runs = task.get("runs", [])
                if len(runs) < k:
                    continue
                
                sampled = random.sample(runs, k)
                answers = [normalize_for_voting(r.get("final_answer", "")) for r in sampled]
                counter = Counter(answers)
                most_common, count = counter.most_common(1)[0]
                
                b_total += 1
                if count >= threshold:
                    b_answered += 1
                    gt_norm = normalize_for_voting(gt)
                    is_corr = gt_norm in most_common or most_common in gt_norm if most_common else False
                    b_correct += int(is_corr)
            
            all_correct.append(b_correct)
            all_answered.append(b_answered)
            all_total.append(b_total)
        
        mean_correct = statistics.mean(all_correct)
        mean_answered = statistics.mean(all_answered)
        mean_total = statistics.mean(all_total)
        
        acc = mean_correct / mean_answered if mean_answered > 0 else 0.0
        coverage = mean_answered / mean_total if mean_total > 0 else 0.0
        
        return acc, coverage, mean_answered, mean_total
    else:
        # k=10, deterministic
        for task in tasks:
            gt = task.get("answer", "")
            runs = task.get("runs", [])
            if len(runs) < k:
                continue
            
            answers = [normalize_for_voting(r.get("final_answer", "")) for r in runs]
            counter = Counter(answers)
            most_common, count = counter.most_common(1)[0]
            
            total += 1
            if count >= threshold:
                answered += 1
                gt_norm = normalize_for_voting(gt)
                is_corr = gt_norm in most_common or most_common in gt_norm if most_common else False
                correct += int(is_corr)
        
        acc = correct / answered if answered > 0 else 0.0
        coverage = answered / total if total > 0 else 0.0
        return acc, coverage, answered, total


def main():
    print("=" * 70)
    print("MAJORITY VOTE INTERVENTION ANALYSIS")
    print("=" * 70)
    
    print("\nLoading data...")
    all_data = load_all_model_data()
    
    print("\nComputing difficulty strata...")
    difficulty = compute_difficulty_strata(all_data)
    strata_counts = Counter(difficulty.values())
    print(f"  Easy: {strata_counts.get('Easy', 0)}, Medium: {strata_counts.get('Medium', 0)}, Hard: {strata_counts.get('Hard', 0)}")
    
    results = {}
    
    for model_name, tasks in all_data.items():
        print(f"\n{'=' * 60}")
        print(f"Model: {model_name} ({len(tasks)} tasks)")
        print(f"{'=' * 60}")
        
        model_results = {"n_tasks": len(tasks)}
        
        # === Majority Vote ===
        print("\n--- Majority Vote Accuracy ---")
        vote_results = {}
        k1_acc = None
        
        for k in [1, 3, 5, 7, 10]:
            n_boot = 1000 if k == 1 else 500
            acc, ci_low, ci_high, n = majority_vote_accuracy(tasks, k, n_bootstrap=n_boot)
            
            if k == 1:
                k1_acc = acc
            
            gain = (acc - k1_acc) * 100 if k1_acc is not None else 0.0
            
            vote_results[f"k={k}"] = {
                "accuracy": round(acc * 100, 1),
                "ci_low": round(ci_low * 100, 1),
                "ci_high": round(ci_high * 100, 1),
                "gain_pp": round(gain, 1),
                "n_tasks": n,
            }
            
            if k == 10:
                print(f"  k={k:2d}: {acc*100:.1f}% (deterministic)  gain: +{gain:.1f}pp  (n={n})")
            else:
                print(f"  k={k:2d}: {acc*100:.1f}% [{ci_low*100:.1f}-{ci_high*100:.1f}%]  gain: +{gain:.1f}pp  (n={n})")
        
        model_results["majority_vote"] = vote_results
        
        # === Selective Prediction ===
        print("\n--- Selective Prediction ---")
        selective_results = {}
        
        # k=3
        for label, thresh in [("unanimous_3/3", 3), ("majority_2/3", 2)]:
            acc, cov, n_ans, n_tot = selective_prediction(tasks, k=3, threshold=thresh)
            selective_results[label] = {
                "accuracy": round(acc * 100, 1),
                "coverage": round(cov * 100, 1),
                "effective_accuracy": round(acc * cov * 100, 1),
                "n_answered": round(n_ans, 1),
                "n_total": round(n_tot, 1),
            }
            print(f"  k=3, {label}: acc={acc*100:.1f}%, coverage={cov*100:.1f}%, effective={acc*cov*100:.1f}%")
        
        # k=5
        for label, thresh in [("unanimous_5/5", 5), ("4/5", 4), ("majority_3/5", 3)]:
            acc, cov, n_ans, n_tot = selective_prediction(tasks, k=5, threshold=thresh)
            selective_results[label] = {
                "accuracy": round(acc * 100, 1),
                "coverage": round(cov * 100, 1),
                "effective_accuracy": round(acc * cov * 100, 1),
                "n_answered": round(n_ans, 1),
                "n_total": round(n_tot, 1),
            }
            print(f"  k=5, {label}: acc={acc*100:.1f}%, coverage={cov*100:.1f}%, effective={acc*cov*100:.1f}%")
        
        model_results["selective_prediction"] = selective_results
        
        # === Per-Stratum Analysis (k=3 gain) ===
        print("\n--- Gain by Difficulty Stratum (k=3) ---")
        stratum_results = {}
        
        for stratum in ["Easy", "Medium", "Hard"]:
            stratum_tasks = [t for t in tasks if difficulty.get(t.get("task_id", ""), "") == stratum]
            if len(stratum_tasks) < 3:
                print(f"  {stratum}: insufficient data ({len(stratum_tasks)} tasks)")
                stratum_results[stratum] = {"n_tasks": len(stratum_tasks), "gain_pp": None}
                continue
            
            acc_k1, _, _, _ = majority_vote_accuracy(stratum_tasks, k=1, n_bootstrap=500)
            acc_k3, ci_low, ci_high, n = majority_vote_accuracy(stratum_tasks, k=3, n_bootstrap=500)
            gain = (acc_k3 - acc_k1) * 100
            
            stratum_results[stratum] = {
                "n_tasks": len(stratum_tasks),
                "k1_accuracy": round(acc_k1 * 100, 1),
                "k3_accuracy": round(acc_k3 * 100, 1),
                "gain_pp": round(gain, 1),
            }
            print(f"  {stratum} ({len(stratum_tasks)} tasks): k=1 {acc_k1*100:.1f}% → k=3 {acc_k3*100:.1f}% (gain: {gain:+.1f}pp)")
        
        model_results["per_stratum"] = stratum_results
        results[model_name] = model_results
    
    # === Cross-model summary ===
    print("\n" + "=" * 70)
    print("CROSS-MODEL SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Model':<20s} {'k=1':>8s} {'k=3':>8s} {'k=5':>8s} {'Gain(3)':>8s} {'Gain(5)':>8s}")
    print("-" * 60)
    for model_name in MODELS:
        r = results[model_name]["majority_vote"]
        print(f"{model_name:<20s} {r['k=1']['accuracy']:>7.1f}% {r['k=3']['accuracy']:>7.1f}% {r['k=5']['accuracy']:>7.1f}% {r['k=3']['gain_pp']:>+7.1f} {r['k=5']['gain_pp']:>+7.1f}")
    
    # Average gain
    avg_gain_k3 = statistics.mean(results[m]["majority_vote"]["k=3"]["gain_pp"] for m in MODELS)
    avg_gain_k5 = statistics.mean(results[m]["majority_vote"]["k=5"]["gain_pp"] for m in MODELS)
    print(f"\n  Average gain k=3: +{avg_gain_k3:.1f}pp")
    print(f"  Average gain k=5: +{avg_gain_k5:.1f}pp")
    
    # Selective prediction summary
    print(f"\n{'Model':<20s} {'Unan.3 Acc':>10s} {'Cov':>6s} {'Maj.2/3 Acc':>11s} {'Cov':>6s}")
    print("-" * 60)
    for model_name in MODELS:
        sp = results[model_name]["selective_prediction"]
        u3 = sp["unanimous_3/3"]
        m3 = sp["majority_2/3"]
        print(f"{model_name:<20s} {u3['accuracy']:>9.1f}% {u3['coverage']:>5.1f}% {m3['accuracy']:>10.1f}% {m3['coverage']:>5.1f}%")
    
    # Save results
    output_path = Path(__file__).resolve().parent / "task_majority_vote_results.json"
    
    # Add summary
    results["_summary"] = {
        "avg_gain_k3_pp": round(avg_gain_k3, 1),
        "avg_gain_k5_pp": round(avg_gain_k5, 1),
    }
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
