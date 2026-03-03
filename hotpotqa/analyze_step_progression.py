#!/usr/bin/env python3
"""
Analyze consistency signal progression across steps.
For each step, compute:
1. Mean pairwise cosine similarity at layer 32 across runs
2. Correlation between activation similarity and behavioral consistency
"""

import json
import numpy as np
from pathlib import Path
from itertools import combinations
from scipy import stats
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt

DATA_DIR = Path("pilot_hidden_states_70b")
RESULTS_DIR = Path("analysis_results")
LAYER = 32  # Best layer from previous analysis
MAX_STEP = 5  # Analyze steps 1-5

def load_hidden_states(run_dir, step_num):
    """Load hidden states for a specific step."""
    hs_file = run_dir / f"hidden_states_step_{step_num}.npy"
    if hs_file.exists():
        return np.load(hs_file)
    return None

def compute_pairwise_similarity(hidden_states_list, layer):
    """Compute mean pairwise cosine similarity for a layer."""
    layer_activations = [hs[layer] for hs in hidden_states_list if hs is not None]
    if len(layer_activations) < 2:
        return None
    
    similarities = []
    for (a, b) in combinations(layer_activations, 2):
        sim = 1 - cosine(a, b)
        similarities.append(sim)
    
    return np.mean(similarities)

def load_behavioral_consistency(question_dir):
    """Compute behavioral consistency (1/CV of step counts) for a question."""
    step_counts = []
    for run_dir in sorted(question_dir.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
            continue
        meta_file = run_dir / "metadata.json"
        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)
                step_counts.append(meta.get("total_steps", 0))
    
    if len(step_counts) < 2:
        return None
    
    mean_steps = np.mean(step_counts)
    std_steps = np.std(step_counts)
    
    if mean_steps == 0:
        return None
    
    cv = std_steps / mean_steps
    # Return 1/CV as consistency (higher = more consistent)
    # Handle CV=0 case (perfect consistency)
    if cv == 0:
        return 10.0  # Cap at high value
    return 1.0 / cv

def analyze_step(step_num):
    """Analyze a single step across all questions."""
    similarities = []
    consistencies = []
    question_ids = []
    
    for question_dir in sorted(DATA_DIR.iterdir()):
        if not question_dir.is_dir():
            continue
        
        # Load hidden states for all runs at this step
        hidden_states_list = []
        valid_runs = 0
        for run_dir in sorted(question_dir.iterdir()):
            if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                continue
            hs = load_hidden_states(run_dir, step_num)
            if hs is not None:
                hidden_states_list.append(hs)
                valid_runs += 1
        
        # Need at least 2 runs to compute similarity
        if len(hidden_states_list) < 2:
            continue
        
        # Compute similarity at layer 32
        sim = compute_pairwise_similarity(hidden_states_list, LAYER)
        if sim is None:
            continue
        
        # Get behavioral consistency
        consistency = load_behavioral_consistency(question_dir)
        if consistency is None:
            continue
        
        similarities.append(sim)
        consistencies.append(consistency)
        question_ids.append(question_dir.name)
    
    return {
        "step": step_num,
        "n_questions": len(similarities),
        "similarities": similarities,
        "consistencies": consistencies,
        "question_ids": question_ids,
        "mean_similarity": np.mean(similarities) if similarities else None,
        "std_similarity": np.std(similarities) if similarities else None,
    }

def main():
    print("=" * 60)
    print("STEP-BY-STEP CONSISTENCY SIGNAL ANALYSIS")
    print(f"Layer: {LAYER}")
    print("=" * 60)
    
    results = []
    
    for step_num in range(1, MAX_STEP + 1):
        print(f"\nAnalyzing step {step_num}...")
        step_result = analyze_step(step_num)
        
        if step_result["n_questions"] < 3:
            print(f"  Skipping - only {step_result['n_questions']} questions have this step")
            continue
        
        # Compute correlation
        sims = step_result["similarities"]
        cons = step_result["consistencies"]
        
        # Handle constant input (step 1 has identical similarities)
        if np.std(sims) < 1e-10:
            r, p = np.nan, 1.0
        else:
            r, p = stats.pearsonr(sims, cons)
        step_result["correlation_r"] = float(r)
        step_result["correlation_p"] = float(p)
        
        print(f"  Questions: {step_result['n_questions']}")
        print(f"  Mean similarity: {step_result['mean_similarity']:.6f} ± {step_result['std_similarity']:.6f}")
        print(f"  Correlation: r={r:.4f}, p={p:.4f}")
        
        results.append(step_result)
    
    # Create summary table
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Step':<6} {'N':<4} {'Mean Sim':<12} {'Std Sim':<12} {'r':<10} {'p-value':<12} {'Sig?':<6}")
    print("-" * 60)
    
    for r in results:
        r_val = r["correlation_r"]
        if np.isnan(r_val):
            sig = "N/A"
            r_str = "nan"
        else:
            sig = "YES" if r["correlation_p"] < 0.05 else "no"
            r_str = f"{r_val:.4f}"
        print(f"{r['step']:<6} {r['n_questions']:<4} {r['mean_similarity']:.6f}     {r['std_similarity']:.6f}     {r_str:<10} {r['correlation_p']:<12.4f} {sig:<6}")
    
    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    steps = [r["step"] for r in results]
    mean_sims = [r["mean_similarity"] for r in results]
    std_sims = [r["std_similarity"] for r in results]
    correlations = [r["correlation_r"] if not np.isnan(r["correlation_r"]) else 0 for r in results]
    p_values = [r["correlation_p"] for r in results]
    has_signal = [not np.isnan(r["correlation_r"]) for r in results]
    
    # Plot 1: Similarity decay curve
    ax1 = axes[0]
    ax1.errorbar(steps, mean_sims, yerr=std_sims, marker='o', markersize=10, 
                 linewidth=2, capsize=5, color='steelblue', label='Mean ± Std')
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect similarity')
    ax1.set_xlabel('Step Number', fontsize=12)
    ax1.set_ylabel('Mean Pairwise Cosine Similarity', fontsize=12)
    ax1.set_title(f'Similarity Decay Curve (Layer {LAYER})', fontsize=14)
    ax1.set_ylim(0.9, 1.01)
    ax1.set_xticks(steps)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Annotate with values
    for i, (s, sim) in enumerate(zip(steps, mean_sims)):
        ax1.annotate(f'{sim:.4f}', (s, sim), textcoords="offset points", 
                     xytext=(0, 10), ha='center', fontsize=9)
    
    # Plot 2: Signal emergence curve
    ax2 = axes[1]
    colors = []
    for i, (p, has_sig) in enumerate(zip(p_values, has_signal)):
        if not has_sig:
            colors.append('lightgray')  # No signal (NaN)
        elif p < 0.05:
            colors.append('green')  # Significant
        else:
            colors.append('gray')  # Not significant
    ax2.bar(steps, correlations, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Step Number', fontsize=12)
    ax2.set_ylabel('Correlation (r) with Behavioral Consistency', fontsize=12)
    ax2.set_title(f'Signal Emergence Curve (Layer {LAYER})', fontsize=14)
    ax2.set_xticks(steps)
    ax2.set_ylim(-0.8, 0.3)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add significance markers
    for i, (s, r_val, p_val, has_sig) in enumerate(zip(steps, correlations, p_values, has_signal)):
        if not has_sig:
            label = 'N/A\n(identical)'
        else:
            label = f'r={r_val:.2f}\np={p_val:.3f}'
            if p_val < 0.05:
                label += '\n*'
        ax2.annotate(label, (s, r_val), textcoords="offset points",
                     xytext=(0, -50 if r_val < 0 else 10), ha='center', fontsize=9)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', alpha=0.7, edgecolor='black', label='p < 0.05'),
                       Patch(facecolor='gray', alpha=0.7, edgecolor='black', label='p ≥ 0.05'),
                       Patch(facecolor='lightgray', alpha=0.7, edgecolor='black', label='N/A (identical)')]
    ax2.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    # Save plot
    RESULTS_DIR.mkdir(exist_ok=True)
    plot_path = RESULTS_DIR / "step_progression.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {plot_path}")
    
    # Save numerical results
    json_results = []
    for r in results:
        json_results.append({
            "step": int(r["step"]),
            "n_questions": int(r["n_questions"]),
            "mean_similarity": float(r["mean_similarity"]) if r["mean_similarity"] is not None else None,
            "std_similarity": float(r["std_similarity"]) if r["std_similarity"] is not None else None,
            "correlation_r": float(r["correlation_r"]) if not np.isnan(r["correlation_r"]) else None,
            "correlation_p": float(r["correlation_p"]),
        })
    
    json_path = RESULTS_DIR / "step_progression.json"
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"Saved data to {json_path}")
    
    plt.show()

if __name__ == "__main__":
    main()
