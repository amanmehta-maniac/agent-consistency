"""
Generate main paper figures with all 5 models.
Figure 1: Histogram of unique action sequences per task
Figure 2: Correctness comparison (consistent vs inconsistent tasks)
"""

import sys
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

# Add parent for common imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.analysis import count_unique_sequences, is_correct, load_results

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Model configs: (display_name, [data_dirs], color)
MODELS = {
    "Llama 3.1 70B": {
        "dirs": ["hotpotqa/results_llama", "hotpotqa/results_cortex_llama"],
        "color": "#e74c3c",
    },
    "Claude Sonnet 4.5": {
        "dirs": ["hotpotqa/results_claude", "hotpotqa/results_cortex_claude"],
        "color": "#3498db",
    },
    "GPT-5": {
        "dirs": ["hotpotqa/results_gpt5", "hotpotqa/results_cortex_gpt5"],
        "color": "#2ecc71",
    },
    "GPT-4o": {
        "dirs": ["hotpotqa/results_gpt4o"],
        "color": "#9b59b6",
    },
    "Gemini 3 Pro": {
        "dirs": ["hotpotqa/results_gemini", "hotpotqa/results_cortex_gemini"],
        "color": "#f39c12",
    },
}

BASE = Path(__file__).resolve().parent.parent


def load_all_tasks(model_name):
    """Load all task data for a model from its directories."""
    cfg = MODELS[model_name]
    tasks = []
    for d in cfg["dirs"]:
        full_path = BASE / d
        if full_path.exists():
            tasks.extend(load_results(str(full_path)))
    return tasks


def compute_unique_seqs(tasks):
    """Compute unique sequence counts for all tasks."""
    return [count_unique_sequences(t) for t in tasks]


def compute_consistency_gap(tasks):
    """Compute accuracy for consistent (<=2 unique seqs) and inconsistent (>=8) tasks."""
    consistent_accs = []
    inconsistent_accs = []
    
    for t in tasks:
        n_unique = count_unique_sequences(t)
        gt = t.get("answer", "")
        runs = t.get("runs", [])
        if not runs:
            continue
        acc = sum(1 for r in runs if is_correct(r, gt)) / len(runs)
        
        if n_unique <= 2:
            consistent_accs.append(acc)
        elif n_unique >= 4:
            inconsistent_accs.append(acc)
    
    return {
        "consistent_acc": np.mean(consistent_accs) * 100 if consistent_accs else 0,
        "inconsistent_acc": np.mean(inconsistent_accs) * 100 if inconsistent_accs else 0,
        "n_consistent": len(consistent_accs),
        "n_inconsistent": len(inconsistent_accs),
    }


def figure1_histogram(output_path):
    """Generate histogram of unique action sequences per task for all 5 models."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    all_data = {}
    for name in MODELS:
        tasks = load_all_tasks(name)
        seqs = compute_unique_seqs(tasks)
        all_data[name] = seqs
        print(f"  {name}: {len(tasks)} tasks, mean unique seqs = {np.mean(seqs):.2f}")
    
    # Bin edges: 1, 2, 3, ..., 10
    bins = np.arange(0.5, 11.5, 1)
    
    for name, seqs in all_data.items():
        # Clip to 10 for display
        clipped = [min(s, 10) for s in seqs]
        ax.hist(clipped, bins=bins, alpha=0.45, label=name,
                color=MODELS[name]["color"], edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel("Unique Action Sequences per Task", fontsize=12)
    ax.set_ylabel("Number of Tasks", fontsize=12)
    ax.set_title("Distribution of Behavioral Variability Across Models", fontsize=13)
    ax.set_xticks(range(1, 11))
    ax.set_xticklabels([str(i) if i < 10 else "10+" for i in range(1, 11)])
    ax.legend(fontsize=9, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def figure2_correctness(output_path):
    """Generate grouped bar chart: consistent vs inconsistent accuracy for all 5 models."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    model_names = list(MODELS.keys())
    consistent_accs = []
    inconsistent_accs = []
    n_consistent = []
    n_inconsistent = []
    
    for name in model_names:
        tasks = load_all_tasks(name)
        gap = compute_consistency_gap(tasks)
        consistent_accs.append(gap["consistent_acc"])
        inconsistent_accs.append(gap["inconsistent_acc"])
        n_consistent.append(gap["n_consistent"])
        n_inconsistent.append(gap["n_inconsistent"])
        print(f"  {name}: consistent={gap['consistent_acc']:.1f}% (n={gap['n_consistent']}), "
              f"inconsistent={gap['inconsistent_acc']:.1f}% (n={gap['n_inconsistent']})")
    
    x = np.arange(len(model_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, consistent_accs, width, label='Consistent (≤2 unique seqs)',
                   color='#27ae60', edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, inconsistent_accs, width, label='Inconsistent (≥4 unique seqs)',
                   color='#e74c3c', edgecolor='white', linewidth=0.5)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    # Add sample size annotations below bars
    for i, (nc, ni) in enumerate(zip(n_consistent, n_inconsistent)):
        ax.annotate(f'n={nc}', xy=(x[i] - width/2, -3), fontsize=7,
                    ha='center', va='top', color='#27ae60')
        ax.annotate(f'n={ni}', xy=(x[i] + width/2, -3), fontsize=7,
                    ha='center', va='top', color='#e74c3c')
    
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Accuracy Gap: Consistent vs. Inconsistent Tasks", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" ", "\n") for n in model_names], fontsize=9)
    ax.legend(fontsize=9, loc='upper right')
    ax.set_ylim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


if __name__ == "__main__":
    print("Generating Figure 1: Unique Sequences Histogram...")
    figure1_histogram(str(BASE / "paper1" / "unique_sequences_histogram.png"))
    
    print("\nGenerating Figure 2: Correctness Comparison...")
    figure2_correctness(str(BASE / "paper1" / "correctness_comparison.png"))
    
    print("\nDone!")
