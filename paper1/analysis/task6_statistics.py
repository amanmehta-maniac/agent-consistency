"""
Task 6: Statistical Reporting

Computes effect sizes, confidence intervals, and statistical tests for all
key claims in the paper. Ensures every number has proper statistical backing.

Outputs: task6_results.json
"""

import json
import sys
import math
import statistics
from pathlib import Path
from collections import Counter

import numpy as np
from scipy import stats

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


# ── Section 1: Consistency-Correctness Gap (Mann-Whitney U) ──────────────

def consistency_correctness_tests(model_data):
    """
    For each model, split tasks into consistent (<=2 unique seqs) vs
    inconsistent (>=4 unique seqs). Run Mann-Whitney U, compute
    rank-biserial effect size.
    """
    results = {}

    for model_name, tasks in model_data.items():
        consistent_acc = []
        inconsistent_acc = []

        for tid, task in tasks.items():
            gt = task["answer"]
            runs = task.get("runs", [])
            if not runs:
                continue
            n_unique = count_unique_sequences(task)
            acc = sum(1 for r in runs if is_correct(r, gt)) / len(runs)

            if n_unique <= 2:
                consistent_acc.append(acc)
            elif n_unique >= 4:
                inconsistent_acc.append(acc)

        n1 = len(consistent_acc)
        n2 = len(inconsistent_acc)

        result = {
            "n_consistent": n1,
            "n_inconsistent": n2,
            "mean_consistent_acc": round(np.mean(consistent_acc), 4) if consistent_acc else None,
            "mean_inconsistent_acc": round(np.mean(inconsistent_acc), 4) if inconsistent_acc else None,
        }

        if n1 >= 5 and n2 >= 5:
            U, p = stats.mannwhitneyu(
                consistent_acc, inconsistent_acc, alternative="greater"
            )
            # Rank-biserial correlation: r = 1 - (2U)/(n1*n2)
            r_rb = 1 - (2 * U) / (n1 * n2)
            result["mann_whitney_U"] = float(U)
            result["mann_whitney_p"] = float(p)
            result["rank_biserial_r"] = round(r_rb, 3)
            result["small_sample_caveat"] = n1 < 10 or n2 < 10
        else:
            result["mann_whitney_U"] = None
            result["mann_whitney_p"] = None
            result["rank_biserial_r"] = None
            result["small_sample_caveat"] = True

        results[model_name] = result

    return results


# ── Section 2: Ranking Instability CI ────────────────────────────────────

def ranking_instability_ci(model_data):
    """
    Reproduce the bootstrap ranking analysis from rankings_analysis.py,
    then compute 95% CI around the instability proportion.
    """
    # Use all 5 models, find common questions
    model_names = list(MODEL_DIRS.keys())

    # Build correctness arrays per model
    all_task_ids = set()
    for m in model_names:
        all_task_ids.update(model_data[m].keys())

    # Find questions available in ALL models
    common_ids = None
    for m in model_names:
        m_ids = set(model_data[m].keys())
        if common_ids is None:
            common_ids = m_ids
        else:
            common_ids = common_ids & m_ids

    question_ids = sorted(common_ids)
    n_questions = len(question_ids)

    # Build correctness arrays: correctness[model][question_idx][run_idx] = bool
    correctness = {}
    for m in model_names:
        arr = []
        for qid in question_ids:
            task = model_data[m][qid]
            gt = task["answer"]
            runs = task.get("runs", [])
            run_correct = [is_correct(r, gt) for r in runs]
            arr.append(run_correct)
        correctness[m] = arr

    # Multi-run accuracy (ground truth ranking)
    multi_run_acc = {}
    for m in model_names:
        total_correct = sum(sum(q) for q in correctness[m])
        total_runs = sum(len(q) for q in correctness[m])
        multi_run_acc[m] = total_correct / total_runs if total_runs > 0 else 0

    ground_truth_ranking = tuple(sorted(model_names, key=lambda m: -multi_run_acc[m]))

    # Bootstrap
    N_BOOTSTRAP = 10000
    rng = np.random.RandomState(42)
    n_match = 0

    for _ in range(N_BOOTSTRAP):
        sample_acc = {}
        for m in model_names:
            correct_count = 0
            for q_idx in range(n_questions):
                runs_for_q = correctness[m][q_idx]
                if runs_for_q:
                    chosen = rng.randint(0, len(runs_for_q))
                    if runs_for_q[chosen]:
                        correct_count += 1
            sample_acc[m] = correct_count / n_questions
        ranked = tuple(sorted(model_names, key=lambda m: -sample_acc[m]))
        if ranked == ground_truth_ranking:
            n_match += 1

    n_differ = N_BOOTSTRAP - n_match
    instability_pct = n_differ / N_BOOTSTRAP * 100

    # 95% CI via Wilson score interval for a proportion
    p_hat = n_differ / N_BOOTSTRAP
    z = 1.96
    denominator = 1 + z**2 / N_BOOTSTRAP
    centre = (p_hat + z**2 / (2 * N_BOOTSTRAP)) / denominator
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * N_BOOTSTRAP)) / N_BOOTSTRAP) / denominator
    ci_low = max(0, (centre - margin)) * 100
    ci_high = min(1, (centre + margin)) * 100

    return {
        "n_models": len(model_names),
        "n_common_questions": n_questions,
        "ground_truth_ranking": list(ground_truth_ranking),
        "multi_run_acc": {m: round(v * 100, 1) for m, v in multi_run_acc.items()},
        "n_bootstrap": N_BOOTSTRAP,
        "n_ranking_matches": n_match,
        "n_ranking_differs": n_differ,
        "instability_pct": round(instability_pct, 1),
        "ci_95_low": round(ci_low, 1),
        "ci_95_high": round(ci_high, 1),
    }


# ── Section 3: Step-2 Divergence CI ─────────────────────────────────────

def step2_divergence_ci(model_data):
    """
    Compute the proportion of tasks where first divergence occurs at step <= 2,
    plus binomial 95% CI (Wilson score).
    """
    results = {}

    for model_name, tasks in model_data.items():
        n_tasks = 0
        n_early_div = 0

        for tid, task in tasks.items():
            runs = task.get("runs", [])
            if len(runs) < 2:
                continue
            n_tasks += 1
            div = first_divergence_point(runs)
            div_step = div.get("divergence_step")
            if div_step is not None and div_step <= 2:
                n_early_div += 1

        if n_tasks == 0:
            results[model_name] = {"n": 0, "pct": None, "ci_low": None, "ci_high": None}
            continue

        p_hat = n_early_div / n_tasks

        # Wilson score interval
        z = 1.96
        denom = 1 + z**2 / n_tasks
        centre = (p_hat + z**2 / (2 * n_tasks)) / denom
        margin = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n_tasks)) / n_tasks) / denom
        ci_low = max(0, centre - margin) * 100
        ci_high = min(1, centre + margin) * 100

        results[model_name] = {
            "n": n_tasks,
            "n_early_divergence": n_early_div,
            "pct": round(p_hat * 100, 1),
            "ci_95_low": round(ci_low, 1),
            "ci_95_high": round(ci_high, 1),
        }

    # Pooled across all models
    total_tasks = sum(r["n"] for r in results.values())
    total_early = sum(r.get("n_early_divergence", 0) for r in results.values())
    if total_tasks > 0:
        p_hat = total_early / total_tasks
        z = 1.96
        denom = 1 + z**2 / total_tasks
        centre = (p_hat + z**2 / (2 * total_tasks)) / denom
        margin = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * total_tasks)) / total_tasks) / denom
        results["pooled"] = {
            "n": total_tasks,
            "n_early_divergence": total_early,
            "pct": round(p_hat * 100, 1),
            "ci_95_low": round(max(0, centre - margin) * 100, 1),
            "ci_95_high": round(min(1, centre + margin) * 100, 1),
        }

    return results


# ── Section 4: Path-Length Correlation ───────────────────────────────────

def path_length_correlation(model_data):
    """
    Correlation between unique_sequences (consistency) and mean step count
    (path length). Reports Spearman r and p-value per model.
    """
    results = {}

    for model_name, tasks in model_data.items():
        unique_seqs = []
        mean_steps = []

        for tid, task in tasks.items():
            runs = task.get("runs", [])
            if not runs:
                continue
            n_unique = count_unique_sequences(task)
            step_counts = [len(r.get("steps", [])) for r in runs]
            avg_steps = statistics.mean(step_counts) if step_counts else 0

            unique_seqs.append(n_unique)
            mean_steps.append(avg_steps)

        n = len(unique_seqs)
        if n < 10:
            results[model_name] = {"n": n, "spearman_r": None, "p_value": None}
            continue

        r, p = stats.spearmanr(unique_seqs, mean_steps)
        results[model_name] = {
            "n": n,
            "spearman_r": round(r, 3),
            "p_value": round(p, 6),
        }

    # Pooled: concatenate all models
    all_seqs = []
    all_steps = []
    for model_name, tasks in model_data.items():
        for tid, task in tasks.items():
            runs = task.get("runs", [])
            if not runs:
                continue
            all_seqs.append(count_unique_sequences(task))
            step_counts = [len(r.get("steps", [])) for r in runs]
            all_steps.append(statistics.mean(step_counts) if step_counts else 0)

    if len(all_seqs) >= 10:
        r, p = stats.spearmanr(all_seqs, all_steps)
        results["pooled"] = {
            "n": len(all_seqs),
            "spearman_r": round(r, 3),
            "p_value": round(p, 6),
        }

    return results


def main():
    print("=" * 70)
    print("TASK 6: STATISTICAL REPORTING")
    print("=" * 70)

    model_data = load_all_model_data()

    all_results = {}

    # Section 1: Consistency-Correctness Gap
    print("\n" + "=" * 50)
    print("1. CONSISTENCY-CORRECTNESS GAP (Mann-Whitney U)")
    print("=" * 50)

    cc_results = consistency_correctness_tests(model_data)
    all_results["consistency_correctness_gap"] = cc_results

    print(f"\n{'Model':<22} {'n_con':>5} {'n_inc':>5} {'U':>8} {'p':>10} {'r_rb':>6} {'caveat':>7}")
    print("-" * 70)
    for model_name, r in cc_results.items():
        U_str = f"{r['mann_whitney_U']:.0f}" if r["mann_whitney_U"] is not None else "N/A"
        p_str = f"{r['mann_whitney_p']:.4f}" if r["mann_whitney_p"] is not None else "N/A"
        r_str = f"{r['rank_biserial_r']:.3f}" if r["rank_biserial_r"] is not None else "N/A"
        caveat = "YES" if r["small_sample_caveat"] else ""
        print(f"{model_name:<22} {r['n_consistent']:>5} {r['n_inconsistent']:>5} "
              f"{U_str:>8} {p_str:>10} {r_str:>6} {caveat:>7}")

    # Section 2: Ranking Instability
    print("\n" + "=" * 50)
    print("2. RANKING INSTABILITY (Bootstrap CI)")
    print("=" * 50)

    ri_results = ranking_instability_ci(model_data)
    all_results["ranking_instability"] = ri_results

    print(f"\n  Models: {ri_results['n_models']}")
    print(f"  Common questions: {ri_results['n_common_questions']}")
    print(f"  Ground truth ranking: {' > '.join(ri_results['ground_truth_ranking'])}")
    print(f"  Multi-run accuracy: {ri_results['multi_run_acc']}")
    print(f"  Instability: {ri_results['instability_pct']}% "
          f"[95% CI: {ri_results['ci_95_low']}% - {ri_results['ci_95_high']}%]")

    # Section 3: Step-2 Divergence
    print("\n" + "=" * 50)
    print("3. STEP-2 DIVERGENCE (Binomial CI)")
    print("=" * 50)

    div_results = step2_divergence_ci(model_data)
    all_results["step2_divergence"] = div_results

    for name, r in div_results.items():
        if r.get("pct") is not None:
            print(f"  {name:<22}: {r['pct']}% [{r['ci_95_low']}% - {r['ci_95_high']}%] (n={r['n']})")

    # Section 4: Path-Length Correlation
    print("\n" + "=" * 50)
    print("4. PATH-LENGTH CORRELATION (Spearman)")
    print("=" * 50)

    pl_results = path_length_correlation(model_data)
    all_results["path_length_correlation"] = pl_results

    for name, r in pl_results.items():
        if r.get("spearman_r") is not None:
            sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else "ns"
            print(f"  {name:<22}: r={r['spearman_r']:.3f}, p={r['p_value']:.6f} {sig} (n={r['n']})")

    # Save
    output_path = Path(__file__).parent / "task6_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
