"""
Metrics for analyzing agent consistency across runs.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter
import statistics


def load_results(results_dir: str = "results") -> List[Dict[str, Any]]:
    """Load all result files from directory."""
    results_dir = Path(results_dir)
    results = []
    for f in sorted(results_dir.glob("*.json")):
        with open(f) as file:
            results.append(json.load(file))
    return results


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if answer is None:
        return ""
    answer = str(answer).lower().strip()
    # Remove common prefixes
    for prefix in ["yes, ", "no, ", "the answer is ", "the "]:
        if answer.startswith(prefix):
            answer = answer[len(prefix):]
    # Remove trailing punctuation
    answer = answer.rstrip(".,!?")
    return answer


def extract_core_answer(answer: str, ground_truth: str) -> str:
    """Extract the core answer, using ground truth as hint."""
    if answer is None:
        return ""
    answer_lower = answer.lower()
    gt_lower = ground_truth.lower()
    
    # Check if ground truth appears in answer
    if gt_lower in answer_lower:
        return gt_lower
    
    # For yes/no questions
    if gt_lower in ["yes", "no"]:
        if "yes" in answer_lower[:20]:
            return "yes"
        if "no" in answer_lower[:20]:
            return "no"
    
    return normalize_answer(answer)


def answer_consistency(runs: List[Dict], ground_truth: str) -> Dict[str, Any]:
    """
    Compute answer consistency metrics for a set of runs.
    
    Returns:
        - exact_match_ratio: fraction of runs with identical answers
        - semantic_match_ratio: fraction with same core answer
        - unique_answers: number of distinct answers
        - most_common_answer: the plurality answer
        - correct_ratio: fraction matching ground truth
    """
    answers = [r.get("final_answer", "") for r in runs]
    normalized = [normalize_answer(a) for a in answers]
    core_answers = [extract_core_answer(a, ground_truth) for a in answers]
    
    n_runs = len(runs)
    
    # Exact match: how many unique raw answers?
    unique_raw = len(set(answers))
    
    # Normalized match
    unique_normalized = len(set(normalized))
    
    # Core answer match
    unique_core = len(set(core_answers))
    core_counter = Counter(core_answers)
    most_common_core, most_common_count = core_counter.most_common(1)[0]
    
    # Correctness
    gt_normalized = ground_truth.lower().strip()
    correct_count = sum(1 for c in core_answers if gt_normalized in c or c in gt_normalized)
    
    return {
        "unique_raw_answers": unique_raw,
        "unique_normalized_answers": unique_normalized,
        "unique_core_answers": unique_core,
        "most_common_answer": most_common_core,
        "most_common_count": most_common_count,
        "exact_consistency": 1.0 if unique_raw == 1 else 0.0,
        "core_consistency": most_common_count / n_runs,
        "correct_ratio": correct_count / n_runs,
        "all_correct": correct_count == n_runs,
        "all_consistent": unique_core == 1,
    }


def step_count_metrics(runs: List[Dict]) -> Dict[str, Any]:
    """
    Compute step count metrics for a set of runs.
    """
    step_counts = [len(r.get("steps", [])) for r in runs]
    
    if not step_counts:
        return {}
    
    return {
        "step_counts": step_counts,
        "mean_steps": statistics.mean(step_counts),
        "min_steps": min(step_counts),
        "max_steps": max(step_counts),
        "step_range": max(step_counts) - min(step_counts),
        "step_stdev": statistics.stdev(step_counts) if len(step_counts) > 1 else 0,
        "step_variance_ratio": (max(step_counts) - min(step_counts)) / statistics.mean(step_counts) if statistics.mean(step_counts) > 0 else 0,
    }


def action_sequence(run: Dict) -> List[str]:
    """Extract action sequence from a run."""
    return [step.get("action", "") for step in run.get("steps", [])]


def action_sequence_similarity(runs: List[Dict]) -> Dict[str, Any]:
    """
    Compute action sequence similarity metrics.
    Uses edit distance (Levenshtein) between action sequences.
    """
    sequences = [action_sequence(r) for r in runs]
    
    if len(sequences) < 2:
        return {"identical_sequences": True, "unique_sequences": 1}
    
    # Count unique sequences
    seq_strings = [",".join(s) for s in sequences]
    unique_seqs = len(set(seq_strings))
    
    # Find most common sequence
    seq_counter = Counter(seq_strings)
    most_common_seq, most_common_count = seq_counter.most_common(1)[0]
    
    # Compute pairwise edit distances
    def edit_distance(s1: List[str], s2: List[str]) -> int:
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        return dp[m][n]
    
    distances = []
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            distances.append(edit_distance(sequences[i], sequences[j]))
    
    avg_distance = statistics.mean(distances) if distances else 0
    
    return {
        "unique_sequences": unique_seqs,
        "identical_sequences": unique_seqs == 1,
        "most_common_sequence": most_common_seq.split(","),
        "most_common_sequence_count": most_common_count,
        "sequence_consistency": most_common_count / len(runs),
        "avg_edit_distance": avg_distance,
        "max_edit_distance": max(distances) if distances else 0,
    }


def first_divergence_point(runs: List[Dict]) -> Dict[str, Any]:
    """
    Find at which step runs first diverge in their actions.
    """
    sequences = [action_sequence(r) for r in runs]
    
    if not sequences:
        return {"divergence_step": 0}
    
    min_len = min(len(s) for s in sequences)
    
    divergence_step = None
    for step_idx in range(min_len):
        actions_at_step = set(s[step_idx] for s in sequences)
        if len(actions_at_step) > 1:
            divergence_step = step_idx + 1  # 1-indexed
            break
    
    # If no divergence in common prefix, divergence is at the length difference
    if divergence_step is None:
        if all(len(s) == len(sequences[0]) for s in sequences):
            divergence_step = None  # Fully identical
        else:
            divergence_step = min_len + 1
    
    return {
        "divergence_step": divergence_step,
        "all_identical_actions": divergence_step is None,
    }


def analyze_task(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Full analysis of a single task (question with multiple runs).
    """
    runs = task_data.get("runs", [])
    ground_truth = task_data.get("answer", "")
    
    return {
        "task_id": task_data.get("task_id", ""),
        "question": task_data.get("question", ""),
        "ground_truth": ground_truth,
        "n_runs": len(runs),
        "answer_metrics": answer_consistency(runs, ground_truth),
        "step_metrics": step_count_metrics(runs),
        "sequence_metrics": action_sequence_similarity(runs),
        "divergence_metrics": first_divergence_point(runs),
    }


def analyze_all(results_dir: str = "results") -> List[Dict[str, Any]]:
    """Analyze all tasks in results directory."""
    results = load_results(results_dir)
    return [analyze_task(r) for r in results]


def print_summary(analyses: List[Dict[str, Any]]):
    """Print human-readable summary of analyses."""
    print("=" * 70)
    print("AGENT CONSISTENCY ANALYSIS")
    print("=" * 70)
    
    for a in analyses:
        print(f"\nQuestion: {a['question'][:65]}...")
        print(f"Ground truth: {a['ground_truth']}")
        print()
        
        ans = a['answer_metrics']
        print(f"  ANSWERS:")
        print(f"    Unique answers: {ans['unique_raw_answers']}/{a['n_runs']}")
        print(f"    Core consistency: {ans['core_consistency']:.0%}")
        print(f"    Correct: {ans['correct_ratio']:.0%}")
        print(f"    Most common: \"{ans['most_common_answer']}\" ({ans['most_common_count']}/{a['n_runs']})")
        
        steps = a['step_metrics']
        if steps:
            print(f"  STEPS:")
            print(f"    Range: {steps['min_steps']}-{steps['max_steps']} (mean: {steps['mean_steps']:.1f})")
            print(f"    Variance ratio: {steps['step_variance_ratio']:.0%}")
        else:
            print(f"  STEPS: No steps recorded")
        
        seq = a['sequence_metrics']
        print(f"  ACTION SEQUENCES:")
        print(f"    Unique sequences: {seq['unique_sequences']}/{a['n_runs']}")
        print(f"    Avg edit distance: {seq['avg_edit_distance']:.1f}")
        
        div = a['divergence_metrics']
        if div['all_identical_actions']:
            print(f"    Divergence: None (all identical)")
        else:
            print(f"    First divergence: Step {div['divergence_step']}")
        
        print("-" * 70)
    
    # Aggregate stats
    print("\nAGGREGATE STATISTICS")
    print("-" * 70)
    
    total_runs = sum(a['n_runs'] for a in analyses)
    avg_correct = statistics.mean(a['answer_metrics']['correct_ratio'] for a in analyses)
    avg_consistency = statistics.mean(a['answer_metrics']['core_consistency'] for a in analyses)
    avg_step_variance = statistics.mean(a['step_metrics'].get('step_variance_ratio', 0) for a in analyses if a['step_metrics'])
    avg_unique_seqs = statistics.mean(a['sequence_metrics']['unique_sequences'] for a in analyses)
    
    print(f"  Total tasks: {len(analyses)}")
    print(f"  Total runs: {total_runs}")
    print(f"  Avg correctness: {avg_correct:.0%}")
    print(f"  Avg answer consistency: {avg_consistency:.0%}")
    print(f"  Avg step variance ratio: {avg_step_variance:.0%}")
    print(f"  Avg unique action sequences per task: {avg_unique_seqs:.1f}")


if __name__ == "__main__":
    analyses = analyze_all()
    print_summary(analyses)