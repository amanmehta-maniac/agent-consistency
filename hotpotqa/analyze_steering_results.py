"""
Analyze activation steering experiment results.
Computes:
- CV comparison (baseline vs steered at each scale)
- Answer consistency and accuracy
- Dose-response curves

Generates plots:
- cv_comparison.png: Bar chart comparing CV across conditions
- accuracy_comparison.png: Accuracy by condition
- dose_response.png: Step count CV vs steering scale
- summary.md: Markdown summary of results
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import Counter
import argparse

# Configuration
RESULTS_DIR = Path("/Users/amehta/research/agent-consistency/hotpotqa/steering_results")
STEERING_SCALES = [0.5, 1.0, 2.0]


def load_results() -> Dict[str, Any]:
    """Load all experiment results."""
    combined_file = RESULTS_DIR / "all_results.json"
    if combined_file.exists():
        with open(combined_file) as f:
            return json.load(f)
    
    # Fall back to loading individual files
    results = {}
    for f in RESULTS_DIR.glob("*_results.json"):
        if f.name != "all_results.json":
            qid = f.stem.replace("_results", "")
            with open(f) as fp:
                results[qid] = json.load(fp)
    return results


def extract_metrics(runs: List[Dict]) -> Dict[str, Any]:
    """Extract metrics from a list of runs."""
    step_counts = []
    answers = []
    successes = []
    
    for run in runs:
        if run.get("error"):
            continue
        
        steps = run.get("steps", [])
        step_counts.append(len(steps))
        
        final_answer = run.get("final_answer")
        if final_answer:
            answers.append(str(final_answer).strip().lower())
        
        successes.append(run.get("success", False))
    
    if not step_counts:
        return {
            "n_runs": 0,
            "mean_steps": 0,
            "std_steps": 0,
            "cv_steps": 0,
            "unique_answers": 0,
            "success_rate": 0,
            "modal_answer": None,
            "modal_answer_freq": 0,
        }
    
    mean_steps = np.mean(step_counts)
    std_steps = np.std(step_counts)
    cv_steps = std_steps / mean_steps if mean_steps > 0 else 0
    
    answer_counts = Counter(answers)
    modal_answer, modal_freq = answer_counts.most_common(1)[0] if answer_counts else (None, 0)
    
    return {
        "n_runs": len(step_counts),
        "mean_steps": float(mean_steps),
        "std_steps": float(std_steps),
        "cv_steps": float(cv_steps),
        "step_counts": step_counts,
        "unique_answers": len(set(answers)),
        "success_rate": sum(successes) / len(successes) if successes else 0,
        "modal_answer": modal_answer,
        "modal_answer_freq": modal_freq / len(answers) if answers else 0,
        "answers": answers,
    }


def compute_accuracy(runs: List[Dict], ground_truth: str) -> float:
    """Compute accuracy against ground truth."""
    correct = 0
    total = 0
    
    gt_lower = ground_truth.strip().lower()
    
    for run in runs:
        if run.get("error"):
            continue
        
        final_answer = run.get("final_answer")
        if final_answer:
            total += 1
            answer_lower = str(final_answer).strip().lower()
            # Check if ground truth is contained in answer or vice versa
            if gt_lower in answer_lower or answer_lower in gt_lower:
                correct += 1
    
    return correct / total if total > 0 else 0


def analyze_question(qid: str, data: Dict) -> Dict[str, Any]:
    """Analyze results for a single question."""
    ground_truth = data.get("ground_truth", "")
    
    # Baseline metrics
    baseline_metrics = extract_metrics(data.get("baseline_runs", []))
    baseline_accuracy = compute_accuracy(data.get("baseline_runs", []), ground_truth)
    
    # Steered metrics at each scale
    steered_metrics = {}
    steered_accuracy = {}
    
    for scale in STEERING_SCALES:
        scale_key = str(scale)
        runs = data.get("steered_runs", {}).get(scale_key, [])
        steered_metrics[scale_key] = extract_metrics(runs)
        steered_accuracy[scale_key] = compute_accuracy(runs, ground_truth)
    
    return {
        "question_id": qid,
        "question": data.get("question", ""),
        "ground_truth": ground_truth,
        "cv_original": data.get("cv_original", 0),
        "baseline": {
            **baseline_metrics,
            "accuracy": baseline_accuracy,
        },
        "steered": {
            scale: {
                **steered_metrics[scale],
                "accuracy": steered_accuracy[scale],
            }
            for scale in [str(s) for s in STEERING_SCALES]
        },
    }


def plot_cv_comparison(analysis: Dict[str, Dict], output_path: Path):
    """Create bar chart comparing CV across conditions."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    questions = list(analysis.keys())
    n_questions = len(questions)
    n_conditions = 1 + len(STEERING_SCALES)  # baseline + steered scales
    
    bar_width = 0.8 / n_conditions
    x = np.arange(n_questions)
    
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    labels = ['Baseline'] + [f'Scale {s}' for s in STEERING_SCALES]
    
    for i, (label, color) in enumerate(zip(labels, colors)):
        if i == 0:  # Baseline
            cvs = [analysis[qid]["baseline"]["cv_steps"] for qid in questions]
        else:
            scale = str(STEERING_SCALES[i-1])
            cvs = [analysis[qid]["steered"][scale]["cv_steps"] for qid in questions]
        
        offset = (i - n_conditions/2 + 0.5) * bar_width
        ax.bar(x + offset, cvs, bar_width, label=label, color=color, alpha=0.8)
    
    ax.set_xlabel('Question ID', fontsize=12)
    ax.set_ylabel('Coefficient of Variation (CV)', fontsize=12)
    ax.set_title('Step Count Consistency: Baseline vs Steered', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([qid[:8] + '...' for qid in questions], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_accuracy_comparison(analysis: Dict[str, Dict], output_path: Path):
    """Create bar chart comparing accuracy across conditions."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    questions = list(analysis.keys())
    n_questions = len(questions)
    n_conditions = 1 + len(STEERING_SCALES)
    
    bar_width = 0.8 / n_conditions
    x = np.arange(n_questions)
    
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    labels = ['Baseline'] + [f'Scale {s}' for s in STEERING_SCALES]
    
    for i, (label, color) in enumerate(zip(labels, colors)):
        if i == 0:
            accs = [analysis[qid]["baseline"]["accuracy"] for qid in questions]
        else:
            scale = str(STEERING_SCALES[i-1])
            accs = [analysis[qid]["steered"][scale]["accuracy"] for qid in questions]
        
        offset = (i - n_conditions/2 + 0.5) * bar_width
        ax.bar(x + offset, accs, bar_width, label=label, color=color, alpha=0.8)
    
    ax.set_xlabel('Question ID', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Answer Accuracy: Baseline vs Steered', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([qid[:8] + '...' for qid in questions], rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_dose_response(analysis: Dict[str, Dict], output_path: Path):
    """Create dose-response plot showing CV vs steering scale."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scales = [0] + STEERING_SCALES  # 0 = baseline
    colors = plt.cm.Set2(np.linspace(0, 1, len(analysis)))
    
    for i, (qid, data) in enumerate(analysis.items()):
        cvs = [data["baseline"]["cv_steps"]]
        for scale in STEERING_SCALES:
            cvs.append(data["steered"][str(scale)]["cv_steps"])
        
        ax.plot(scales, cvs, 'o-', color=colors[i], linewidth=2, markersize=8,
                label=f'{qid[:8]}... (orig CV={data["cv_original"]:.2f})')
    
    ax.set_xlabel('Steering Scale (0 = baseline)', fontsize=12)
    ax.set_ylabel('Coefficient of Variation (CV)', fontsize=12)
    ax.set_title('Dose-Response: CV vs Steering Scale', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(alpha=0.3)
    ax.set_xticks(scales)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_summary_md(analysis: Dict[str, Dict], output_path: Path):
    """Generate markdown summary of results."""
    lines = [
        "# Activation Steering Experiment Results",
        "",
        "## Overview",
        "",
        f"- **Questions tested**: {len(analysis)}",
        f"- **Steering scales**: {STEERING_SCALES}",
        f"- **Intervention**: Layer 72, Step 4",
        "",
        "## Summary Table",
        "",
        "| Question ID | Orig CV | Baseline CV | Scale 0.5 CV | Scale 1.0 CV | Scale 2.0 CV | Baseline Acc | Best Steered Acc |",
        "|-------------|---------|-------------|--------------|--------------|--------------|--------------|------------------|",
    ]
    
    for qid, data in analysis.items():
        baseline_cv = data["baseline"]["cv_steps"]
        baseline_acc = data["baseline"]["accuracy"]
        orig_cv = data["cv_original"]
        
        steered_cvs = [data["steered"][str(s)]["cv_steps"] for s in STEERING_SCALES]
        steered_accs = [data["steered"][str(s)]["accuracy"] for s in STEERING_SCALES]
        best_steered_acc = max(steered_accs)
        
        lines.append(
            f"| {qid[:12]}... | {orig_cv:.3f} | {baseline_cv:.3f} | "
            f"{steered_cvs[0]:.3f} | {steered_cvs[1]:.3f} | {steered_cvs[2]:.3f} | "
            f"{baseline_acc:.2f} | {best_steered_acc:.2f} |"
        )
    
    # Aggregate statistics
    all_baseline_cvs = [data["baseline"]["cv_steps"] for data in analysis.values()]
    all_steered_cvs = {
        str(s): [data["steered"][str(s)]["cv_steps"] for data in analysis.values()]
        for s in STEERING_SCALES
    }
    
    lines.extend([
        "",
        "## Aggregate Statistics",
        "",
        f"- **Mean baseline CV**: {np.mean(all_baseline_cvs):.3f} (std={np.std(all_baseline_cvs):.3f})",
    ])
    
    for scale in STEERING_SCALES:
        cvs = all_steered_cvs[str(scale)]
        lines.append(f"- **Mean CV at scale {scale}**: {np.mean(cvs):.3f} (std={np.std(cvs):.3f})")
    
    # CV reduction analysis
    lines.extend([
        "",
        "## CV Change Analysis",
        "",
        "| Question | Best Scale | CV Reduction | % Change |",
        "|----------|------------|--------------|----------|",
    ])
    
    for qid, data in analysis.items():
        baseline_cv = data["baseline"]["cv_steps"]
        
        best_scale = None
        best_cv = baseline_cv
        for scale in STEERING_SCALES:
            cv = data["steered"][str(scale)]["cv_steps"]
            if cv < best_cv:
                best_cv = cv
                best_scale = scale
        
        if best_scale:
            reduction = baseline_cv - best_cv
            pct_change = (reduction / baseline_cv * 100) if baseline_cv > 0 else 0
            lines.append(f"| {qid[:12]}... | {best_scale} | {reduction:.3f} | {pct_change:.1f}% |")
        else:
            lines.append(f"| {qid[:12]}... | None | (no improvement) | - |")
    
    lines.extend([
        "",
        "## Key Findings",
        "",
        "1. **Consistency Impact**: [To be filled based on results]",
        "2. **Accuracy Impact**: [To be filled based on results]", 
        "3. **Optimal Scale**: [To be filled based on results]",
        "",
        "## Plots",
        "",
        "- `cv_comparison.png`: CV across conditions",
        "- `accuracy_comparison.png`: Accuracy across conditions",
        "- `dose_response.png`: CV vs steering scale",
    ])
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze steering experiment results")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR), help="Results directory")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    print("Loading results...")
    results = load_results()
    
    if not results:
        print("No results found!")
        return
    
    print(f"Found results for {len(results)} questions")
    
    # Analyze each question
    print("\nAnalyzing results...")
    analysis = {}
    for qid, data in results.items():
        analysis[qid] = analyze_question(qid, data)
    
    # Save analysis
    analysis_file = results_dir / "analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"Saved analysis to: {analysis_file}")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_cv_comparison(analysis, results_dir / "cv_comparison.png")
    plot_accuracy_comparison(analysis, results_dir / "accuracy_comparison.png")
    plot_dose_response(analysis, results_dir / "dose_response.png")
    
    # Generate summary
    print("\nGenerating summary...")
    generate_summary_md(analysis, results_dir / "summary.md")
    
    # Print quick summary
    print("\n" + "="*60)
    print("QUICK SUMMARY")
    print("="*60)
    
    for qid, data in analysis.items():
        print(f"\n{qid[:20]}...")
        print(f"  Original CV: {data['cv_original']:.3f}")
        print(f"  Baseline CV: {data['baseline']['cv_steps']:.3f}")
        for scale in STEERING_SCALES:
            cv = data['steered'][str(scale)]['cv_steps']
            acc = data['steered'][str(scale)]['accuracy']
            print(f"  Scale {scale}: CV={cv:.3f}, Acc={acc:.2f}")


if __name__ == "__main__":
    main()
