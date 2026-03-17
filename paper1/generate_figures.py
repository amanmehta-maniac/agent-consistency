"""
Generate main paper figures with 4 primary models (GPT-4o moved to appendix).
Figure 1: Histogram of unique action sequences per task
Figure 2: Correctness comparison (consistent vs inconsistent tasks)
Figure 3: Temperature ablation (4 models x 3 temperatures)
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
from matplotlib.backends.backend_pdf import PdfPages

# Model configs: (display_name, [data_dirs], color)
# GPT-4o moved to appendix; primary analysis uses 4 models x 200 questions each
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
    """Generate grouped bar chart of unique action sequences per task for 4 primary models."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    all_data = {}
    for name in MODELS:
        tasks = load_all_tasks(name)
        seqs = compute_unique_seqs(tasks)
        all_data[name] = seqs
        print(f"  {name}: {len(tasks)} tasks, mean unique seqs = {np.mean(seqs):.2f}")
    
    # Count occurrences per bin (1-9, 10+)
    bin_labels = [str(i) for i in range(1, 10)] + ["10+"]
    n_bins = len(bin_labels)
    model_names = list(all_data.keys())
    n_models = len(model_names)
    
    counts = {}
    for name, seqs in all_data.items():
        hist = [0] * n_bins
        for s in seqs:
            if s >= 10:
                hist[9] += 1
            else:
                hist[s - 1] += 1
        counts[name] = hist
    
    # Grouped bars
    x = np.arange(n_bins)
    total_width = 0.8
    bar_width = total_width / n_models
    
    for i, name in enumerate(model_names):
        offset = (i - n_models / 2 + 0.5) * bar_width
        ax.bar(x + offset, counts[name], bar_width,
               label=name, color=MODELS[name]["color"],
               edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel("Unique Action Sequences per Task", fontsize=12)
    ax.set_ylabel("Number of Tasks", fontsize=12)
    ax.set_title("Distribution of Behavioral Variability Across Models", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)
    ax.legend(fontsize=9, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def figure2_correctness(output_path):
    """Generate grouped bar chart: consistent vs inconsistent accuracy for 4 primary models."""
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


def save_fig(fig, base_path):
    """Save figure as both PNG and PDF."""
    fig.savefig(base_path + ".png", dpi=300, bbox_inches='tight')
    fig.savefig(base_path + ".pdf", bbox_inches='tight')
    print(f"  Saved: {base_path}.png and .pdf")


def figure3_temperature(output_base):
    """Generate temperature ablation line plot for 4 primary models."""
    temps = [0.0, 0.3, 0.7]

    # Temperature ablation directories per model
    TEMP_DIRS = {
        "Llama 3.1 70B": {
            0.0: "hotpotqa/results_llama_temp0",
            0.3: "hotpotqa/results_llama_temp03",
            0.7: "hotpotqa/results_llama",
        },
        "Claude Sonnet 4.5": {
            0.0: "hotpotqa/results_claude_temp0",
            0.3: "hotpotqa/results_claude_temp03",
            0.7: "hotpotqa/results_claude",
        },
        "GPT-5": {
            0.0: "hotpotqa/results_gpt5_temp0",
            0.3: "hotpotqa/results_gpt5_temp03",
            0.7: "hotpotqa/results_gpt5",
        },
        "Gemini 3 Pro": {
            0.0: "hotpotqa/results_gemini_temp0",
            0.3: "hotpotqa/results_gemini_temp03",
            0.7: "hotpotqa/results_gemini",
        },
    }

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax2 = ax1.twinx()

    for model_name, temp_dirs in TEMP_DIRS.items():
        color = MODELS[model_name]["color"]
        unique_vals = []
        acc_vals = []

        # Get the 20 question IDs from T=0.0 directory
        t0_path = BASE / temp_dirs[0.0]
        t0_ids = set(f.stem for f in t0_path.iterdir()
                     if f.suffix == '.json') if t0_path.exists() else set()

        for t in temps:
            d = BASE / temp_dirs[t]
            if not d.exists():
                unique_vals.append(np.nan)
                acc_vals.append(np.nan)
                continue

            tasks = load_results(str(d))
            # For T=0.7, filter to only the ablation questions and truncate to 5 runs
            if t == 0.7 and t0_ids:
                tasks = [tk for tk in tasks if tk.get('task_id', '') in t0_ids]
                # Ablation uses 5 runs; truncate T=0.7 to match
                for tk in tasks:
                    tk['runs'] = tk.get('runs', [])[:5]

            if not tasks:
                unique_vals.append(np.nan)
                acc_vals.append(np.nan)
                continue

            seqs = [count_unique_sequences(tk) for tk in tasks]
            accs = []
            for tk in tasks:
                gt = tk.get('answer', '')
                runs = tk.get('runs', [])
                if runs:
                    accs.append(sum(1 for r in runs if is_correct(r, gt)) / len(runs))

            unique_vals.append(np.mean(seqs))
            acc_vals.append(np.mean(accs) * 100)

        print(f"  {model_name}: unique={unique_vals}, acc={acc_vals}")

        ax1.plot(temps, unique_vals, 'o-', color=color, label=model_name, linewidth=2, markersize=6)
        ax2.plot(temps, acc_vals, 's--', color=color, linewidth=1.5, markersize=5, alpha=0.6)

    ax1.set_xlabel("Temperature", fontsize=12)
    ax1.set_ylabel("Mean Unique Sequences", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", fontsize=12, color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax1.set_xticks(temps)

    # Legend
    ax1.legend(fontsize=9, loc='upper left')
    # Add annotation for line styles
    ax1.annotate('Solid: unique seqs | Dashed: accuracy',
                 xy=(0.5, -0.12), xycoords='axes fraction',
                 ha='center', fontsize=8, color='gray')

    ax1.spines['top'].set_visible(False)
    ax1.set_title("Temperature Ablation: Consistency vs. Accuracy", fontsize=13)

    plt.tight_layout()
    save_fig(fig, output_base)
    plt.close()


if __name__ == "__main__":
    out_dir = str(BASE / "paper1")

    print("Generating Figure 1: Unique Sequences Histogram...")
    figure1_histogram(out_dir + "/unique_sequences_histogram.png")
    # Also save PDF
    figure1_histogram(out_dir + "/unique_sequences_histogram.pdf")

    print("\nGenerating Figure 2: Correctness Comparison...")
    figure2_correctness(out_dir + "/correctness_comparison.png")
    figure2_correctness(out_dir + "/correctness_comparison.pdf")

    print("\nGenerating Figure 3: Temperature Ablation...")
    figure3_temperature(out_dir + "/temperature_ablation")

    print("\nDone!")
