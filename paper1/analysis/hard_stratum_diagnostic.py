"""
Deep diagnostic for the Hard stratum reversal:
Why do inconsistent agents outperform consistent agents on hard tasks?

Key questions:
1. What does "consistent" mean on hard tasks? (consistently wrong?)
2. What does "inconsistent" mean? (occasionally stumbling onto right answer?)
3. What is the distribution of per-task accuracy in each group?
4. How many correct runs do consistent vs inconsistent tasks have?
5. Is the sample size too small to draw conclusions?
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
import statistics

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "hotpotqa"))

from analysis import is_correct, count_unique_sequences

BASE = Path(__file__).resolve().parent.parent.parent

MODEL_DIRS = {
    "Llama 3.1 70B": ["hotpotqa/results_llama", "hotpotqa/results_cortex_llama"],
    "Claude Sonnet 4.5": ["hotpotqa/results_claude", "hotpotqa/results_cortex_claude"],
    "GPT-4o": ["hotpotqa/results_gpt4o"],
    "GPT-5": ["hotpotqa/results_gpt5", "hotpotqa/results_cortex_gpt5"],
    "Gemini 3 Pro": ["hotpotqa/results_gemini", "hotpotqa/results_cortex_gemini"],
}


def load_all():
    model_data = {}
    for model_name, dirs in MODEL_DIRS.items():
        model_data[model_name] = {}
        for d in dirs:
            fp = BASE / d
            for f in sorted(fp.glob("*.json")):
                with open(f) as fh:
                    data = json.load(fh)
                model_data[model_name][data["task_id"]] = data
    return model_data


def compute_difficulty(model_data):
    all_tasks = set()
    for tasks in model_data.values():
        all_tasks.update(tasks.keys())

    difficulty = {}
    for tid in all_tasks:
        correct = total = 0
        for tasks in model_data.values():
            if tid in tasks:
                gt = tasks[tid]["answer"]
                for run in tasks[tid].get("runs", []):
                    total += 1
                    if is_correct(run, gt):
                        correct += 1
        difficulty[tid] = correct / total if total > 0 else 0.0
    return difficulty


def main():
    model_data = load_all()
    difficulty = compute_difficulty(model_data)

    # Get hard tasks
    hard_tasks = [tid for tid, d in difficulty.items() if d < 0.40]
    hard_tasks.sort(key=lambda t: difficulty[t])

    print("=" * 80)
    print(f"HARD STRATUM DIAGNOSTIC ({len(hard_tasks)} questions)")
    print("=" * 80)

    print(f"\nDifficulty distribution of Hard tasks:")
    diff_vals = [difficulty[t] for t in hard_tasks]
    print(f"  Mean: {np.mean(diff_vals):.3f}")
    print(f"  Median: {np.median(diff_vals):.3f}")
    print(f"  Min: {min(diff_vals):.3f}, Max: {max(diff_vals):.3f}")
    zero_diff = sum(1 for d in diff_vals if d == 0.0)
    print(f"  Tasks with 0% avg correctness across all models: {zero_diff}/{len(hard_tasks)}")

    for model_name in MODEL_DIRS:
        print(f"\n{'='*80}")
        print(f"MODEL: {model_name}")
        print(f"{'='*80}")

        tasks = model_data[model_name]
        consistent_tasks = []  # <=2 unique seqs
        inconsistent_tasks = []  # >=4 unique seqs
        middle_tasks = []  # 3 unique seqs

        for tid in hard_tasks:
            if tid not in tasks:
                continue
            task = tasks[tid]
            gt = task["answer"]
            n_unique = count_unique_sequences(task)
            runs = task.get("runs", [])
            n_correct = sum(1 for r in runs if is_correct(r, gt))
            acc = n_correct / len(runs) if runs else 0

            entry = {
                "task_id": tid,
                "n_unique_seqs": n_unique,
                "n_correct": n_correct,
                "n_runs": len(runs),
                "accuracy": acc,
                "difficulty": difficulty[tid],
                "gold_answer": gt,
                "answers": [(r.get("final_answer") or "N/A")[:60] for r in runs],
            }

            if n_unique <= 2:
                consistent_tasks.append(entry)
            elif n_unique >= 4:
                inconsistent_tasks.append(entry)
            else:
                middle_tasks.append(entry)

        print(f"\n  Consistent (≤2 seqs): n={len(consistent_tasks)}")
        print(f"  Middle (3 seqs):      n={len(middle_tasks)}")
        print(f"  Inconsistent (≥4):    n={len(inconsistent_tasks)}")

        # --- Consistent tasks analysis ---
        if consistent_tasks:
            accs = [t["accuracy"] for t in consistent_tasks]
            print(f"\n  --- CONSISTENT TASKS (≤2 seqs) ---")
            print(f"  Mean accuracy: {np.mean(accs)*100:.1f}%")
            print(f"  # with 0% accuracy: {sum(1 for a in accs if a == 0)}/{len(accs)}")
            print(f"  # with >0% accuracy: {sum(1 for a in accs if a > 0)}/{len(accs)}")

            # Show each task
            for t in sorted(consistent_tasks, key=lambda x: x["accuracy"]):
                print(f"    Task {t['task_id'][:12]}...: "
                      f"unique_seqs={t['n_unique_seqs']}, "
                      f"correct={t['n_correct']}/{t['n_runs']}, "
                      f"difficulty={t['difficulty']:.3f}")
                print(f"      Gold: {t['gold_answer'][:60]}")
                # Show answer distribution
                answer_counts = Counter(t["answers"])
                for ans, cnt in answer_counts.most_common(3):
                    is_right = "✓" if is_correct({"final_answer": ans}, t["gold_answer"]) else "✗"
                    print(f"      {is_right} '{ans}' x{cnt}")

        # --- Inconsistent tasks analysis ---
        if inconsistent_tasks:
            accs = [t["accuracy"] for t in inconsistent_tasks]
            print(f"\n  --- INCONSISTENT TASKS (≥4 seqs) ---")
            print(f"  Mean accuracy: {np.mean(accs)*100:.1f}%")
            print(f"  # with 0% accuracy: {sum(1 for a in accs if a == 0)}/{len(accs)}")
            print(f"  # with >0% accuracy: {sum(1 for a in accs if a > 0)}/{len(accs)}")

            for t in sorted(inconsistent_tasks, key=lambda x: x["accuracy"], reverse=True):
                print(f"    Task {t['task_id'][:12]}...: "
                      f"unique_seqs={t['n_unique_seqs']}, "
                      f"correct={t['n_correct']}/{t['n_runs']}, "
                      f"difficulty={t['difficulty']:.3f}")
                print(f"      Gold: {t['gold_answer'][:60]}")
                answer_counts = Counter(t["answers"])
                for ans, cnt in answer_counts.most_common(3):
                    is_right = "✓" if is_correct({"final_answer": ans}, t["gold_answer"]) else "✗"
                    print(f"      {is_right} '{ans}' x{cnt}")

    # --- Cross-model summary ---
    print(f"\n{'='*80}")
    print("CROSS-MODEL SUMMARY: Why does the gap reverse?")
    print(f"{'='*80}")

    print("\nHypothesis: On hard tasks, 'consistent' = consistently WRONG.")
    print("The model confidently picks a wrong answer every time (low exploration).")
    print("'Inconsistent' = tries many paths, occasionally stumbles on correct one.\n")

    for model_name in MODEL_DIRS:
        tasks = model_data[model_name]
        cons_zero = 0
        cons_total = 0
        incons_some_correct = 0
        incons_total = 0

        for tid in hard_tasks:
            if tid not in tasks:
                continue
            task = tasks[tid]
            gt = task["answer"]
            n_unique = count_unique_sequences(task)
            runs = task.get("runs", [])
            n_correct = sum(1 for r in runs if is_correct(r, gt))

            if n_unique <= 2:
                cons_total += 1
                if n_correct == 0:
                    cons_zero += 1
            elif n_unique >= 4:
                incons_total += 1
                if n_correct > 0:
                    incons_some_correct += 1

        if cons_total > 0 and incons_total > 0:
            print(f"  {model_name}:")
            print(f"    Consistent:   {cons_zero}/{cons_total} have 0% accuracy "
                  f"(= consistently wrong)")
            print(f"    Inconsistent: {incons_some_correct}/{incons_total} have >0% accuracy "
                  f"(= occasionally correct)")

    # --- Statistical test: is the reversal significant? ---
    print(f"\n{'='*80}")
    print("STATISTICAL SIGNIFICANCE OF REVERSAL")
    print(f"{'='*80}")

    for model_name in MODEL_DIRS:
        tasks = model_data[model_name]
        cons_accs = []
        incons_accs = []

        for tid in hard_tasks:
            if tid not in tasks:
                continue
            task = tasks[tid]
            gt = task["answer"]
            n_unique = count_unique_sequences(task)
            runs = task.get("runs", [])
            acc = sum(1 for r in runs if is_correct(r, gt)) / len(runs) if runs else 0

            if n_unique <= 2:
                cons_accs.append(acc)
            elif n_unique >= 4:
                incons_accs.append(acc)

        if len(cons_accs) >= 3 and len(incons_accs) >= 3:
            u_stat, p_val = stats.mannwhitneyu(cons_accs, incons_accs, alternative='two-sided')
            print(f"  {model_name}: n_cons={len(cons_accs)}, n_incons={len(incons_accs)}, "
                  f"mean_cons={np.mean(cons_accs)*100:.1f}%, mean_incons={np.mean(incons_accs)*100:.1f}%, "
                  f"U={u_stat:.1f}, p={p_val:.4f} {'*' if p_val < 0.05 else 'ns'}")
        else:
            print(f"  {model_name}: insufficient data (n_cons={len(cons_accs)}, n_incons={len(incons_accs)})")

    # --- Alternative analysis: what if we define "consistent" differently? ---
    print(f"\n{'='*80}")
    print("ALTERNATIVE: What if 'consistent' = same ANSWER (not same action sequence)?")
    print(f"{'='*80}")
    print("Action sequence diversity may not be the right grouping variable on hard tasks.")
    print("A model could take different paths but give the same (wrong) answer.\n")

    for model_name in MODEL_DIRS:
        tasks = model_data[model_name]
        answer_cons_accs = []  # high answer consistency
        answer_incons_accs = []  # low answer consistency

        for tid in hard_tasks:
            if tid not in tasks:
                continue
            task = tasks[tid]
            gt = task["answer"]
            runs = task.get("runs", [])
            if not runs:
                continue

            # Answer consistency = fraction of runs with majority answer
            answers = [r.get("final_answer", "") for r in runs]
            most_common_count = Counter(answers).most_common(1)[0][1]
            ans_consistency = most_common_count / len(runs)
            acc = sum(1 for r in runs if is_correct(r, gt)) / len(runs)

            if ans_consistency >= 0.7:  # high answer agreement
                answer_cons_accs.append(acc)
            elif ans_consistency < 0.5:  # low answer agreement
                answer_incons_accs.append(acc)

        if answer_cons_accs and answer_incons_accs:
            print(f"  {model_name}: "
                  f"high_ans_cons (≥70%): n={len(answer_cons_accs)}, "
                  f"acc={np.mean(answer_cons_accs)*100:.1f}% | "
                  f"low_ans_cons (<50%): n={len(answer_incons_accs)}, "
                  f"acc={np.mean(answer_incons_accs)*100:.1f}%")


if __name__ == "__main__":
    main()
