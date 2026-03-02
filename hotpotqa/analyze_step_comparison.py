#!/usr/bin/env python3
"""
Compare step 1 (pre-first-action) vs step 3 (post-divergence) correlation analysis.
Tests if consistency signal exists before the agent takes any action.
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
from scipy import stats

DATA_DIR = Path("/Users/amehta/research/agent-consistency/hotpotqa/pilot_hidden_states_70b")
OUTPUT_DIR = Path("/Users/amehta/research/agent-consistency/hotpotqa/analysis_results")


def load_question_data(question_dir, step_num):
    """Load all runs for a question at a specific step."""
    runs = []
    for run_dir in sorted(question_dir.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
            continue
        
        metadata_file = run_dir / "metadata.json"
        if not metadata_file.exists():
            continue
            
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        # Load specific step's hidden states
        hs_file = run_dir / f"hidden_states_step_{step_num}.npy"
        hidden_states = None
        if hs_file.exists():
            hidden_states = np.load(hs_file)
        
        runs.append({
            "run_id": run_dir.name,
            "metadata": metadata,
            "hidden_states": hidden_states
        })
    
    return runs


def compute_behavioral_consistency(runs):
    """Compute behavioral consistency (1/CV of step counts)."""
    step_counts = [r["metadata"]["total_steps"] for r in runs]
    
    if np.std(step_counts) > 0:
        cv = np.std(step_counts) / np.mean(step_counts)
        return 1 / (cv + 0.01)
    return 100  # Max consistency when no variance


def compute_activation_similarity(runs, layer_idx):
    """Compute mean pairwise cosine similarity at a given layer."""
    hidden_states = [r["hidden_states"] for r in runs if r["hidden_states"] is not None]
    
    if len(hidden_states) < 2:
        return None
    
    layer_activations = [hs[layer_idx] for hs in hidden_states]
    
    n = len(layer_activations)
    similarities = []
    
    for i in range(n):
        for j in range(i + 1, n):
            a = layer_activations[i]
            b = layer_activations[j]
            
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a > 0 and norm_b > 0:
                sim = np.dot(a, b) / (norm_a * norm_b)
                similarities.append(sim)
    
    return np.mean(similarities) if similarities else None


def analyze_step(step_num, step_label):
    """Run correlation analysis for a specific step."""
    question_dirs = [d for d in DATA_DIR.iterdir() 
                     if d.is_dir() and not d.name.startswith('.')]
    
    print(f"\n{'='*70}")
    print(f"Analyzing {step_label} (step {step_num})")
    print(f"{'='*70}")
    
    results = []
    
    for q_dir in sorted(question_dirs):
        runs = load_question_data(q_dir, step_num)
        
        if not runs:
            continue
        
        # Check if we have hidden states
        hs_count = sum(1 for r in runs if r["hidden_states"] is not None)
        if hs_count < 2:
            continue
        
        consistency = compute_behavioral_consistency(runs)
        
        # Compute similarity at all layers
        layer_sims = []
        for layer_idx in range(81):
            sim = compute_activation_similarity(runs, layer_idx)
            layer_sims.append(sim)
        
        results.append({
            "question_id": q_dir.name,
            "consistency": consistency,
            "layer_similarities": layer_sims
        })
    
    print(f"Questions with valid data: {len(results)}")
    
    # Compute layer-wise correlations
    layer_correlations = []
    layer_p_values = []
    
    consistencies = [r["consistency"] for r in results]
    
    for layer_idx in range(81):
        layer_sims = [r["layer_similarities"][layer_idx] for r in results 
                      if r["layer_similarities"][layer_idx] is not None]
        
        if len(layer_sims) == len(consistencies) and len(layer_sims) >= 3:
            corr, p_val = stats.pearsonr(layer_sims, consistencies)
            layer_correlations.append(corr)
            layer_p_values.append(p_val)
        else:
            layer_correlations.append(0)
            layer_p_values.append(1)
    
    # Find best layer
    best_layer = np.argmax(np.abs(layer_correlations))
    best_corr = layer_correlations[best_layer]
    best_p = layer_p_values[best_layer]
    
    # Count significant layers
    sig_layers = [i for i, p in enumerate(layer_p_values) if p < 0.05]
    
    print(f"\nBest predictive layer: {best_layer}")
    print(f"  Correlation: r = {best_corr:.4f}")
    print(f"  P-value: p = {best_p:.6f}")
    print(f"Significant layers (p < 0.05): {len(sig_layers)}")
    if sig_layers:
        print(f"  Layers: {min(sig_layers)}-{max(sig_layers)}")
    
    # Check similarity variance
    all_sims = []
    for r in results:
        for sim in r["layer_similarities"]:
            if sim is not None:
                all_sims.append(sim)
    print(f"\nSimilarity range: {min(all_sims):.6f} - {max(all_sims):.6f}")
    print(f"Similarity std: {np.std(all_sims):.6f}")
    
    return {
        "step_num": step_num,
        "step_label": step_label,
        "n_questions": len(results),
        "layer_correlations": layer_correlations,
        "layer_p_values": layer_p_values,
        "best_layer": int(best_layer),
        "best_correlation": float(best_corr),
        "best_p_value": float(best_p),
        "significant_layers": sig_layers,
        "similarity_range": (float(min(all_sims)), float(max(all_sims))),
        "similarity_std": float(np.std(all_sims))
    }


def main():
    print("="*70)
    print("STEP COMPARISON: Pre-action (Step 1) vs Post-divergence (Step 3)")
    print("="*70)
    
    # Analyze both steps
    step1_results = analyze_step(1, "Pre-first-action")
    step3_results = analyze_step(3, "Post-divergence")
    
    # Summary comparison
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    print(f"\n{'Metric':<30} {'Step 1':>15} {'Step 3':>15}")
    print("-"*60)
    print(f"{'Best layer':<30} {step1_results['best_layer']:>15} {step3_results['best_layer']:>15}")
    print(f"{'Best correlation (r)':<30} {step1_results['best_correlation']:>15.4f} {step3_results['best_correlation']:>15.4f}")
    print(f"{'Best p-value':<30} {step1_results['best_p_value']:>15.6f} {step3_results['best_p_value']:>15.6f}")
    print(f"{'Significant layers':<30} {len(step1_results['significant_layers']):>15} {len(step3_results['significant_layers']):>15}")
    print(f"{'Similarity variance':<30} {step1_results['similarity_std']:>15.6f} {step3_results['similarity_std']:>15.6f}")
    
    # Generate comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Step 1 correlations
    ax1 = axes[0, 0]
    ax1.plot(range(81), step1_results['layer_correlations'], 'b-', linewidth=2)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.fill_between(range(81), step1_results['layer_correlations'], alpha=0.3)
    sig1 = step1_results['significant_layers']
    if sig1:
        ax1.scatter(sig1, [step1_results['layer_correlations'][i] for i in sig1], 
                   c='red', s=30, zorder=5, label=f'p<0.05 ({len(sig1)})')
        ax1.legend()
    ax1.set_title(f"Step 1 (Pre-action): Best r={step1_results['best_correlation']:.3f} @ Layer {step1_results['best_layer']}")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Correlation (r)")
    ax1.set_xlim(0, 80)
    
    # Plot 2: Step 3 correlations
    ax2 = axes[0, 1]
    ax2.plot(range(81), step3_results['layer_correlations'], 'g-', linewidth=2)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.fill_between(range(81), step3_results['layer_correlations'], alpha=0.3, color='green')
    sig3 = step3_results['significant_layers']
    if sig3:
        ax2.scatter(sig3, [step3_results['layer_correlations'][i] for i in sig3], 
                   c='red', s=30, zorder=5, label=f'p<0.05 ({len(sig3)})')
        ax2.legend()
    ax2.set_title(f"Step 3 (Post-divergence): Best r={step3_results['best_correlation']:.3f} @ Layer {step3_results['best_layer']}")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Correlation (r)")
    ax2.set_xlim(0, 80)
    
    # Plot 3: Overlay comparison
    ax3 = axes[1, 0]
    ax3.plot(range(81), step1_results['layer_correlations'], 'b-', linewidth=2, label='Step 1 (pre-action)', alpha=0.8)
    ax3.plot(range(81), step3_results['layer_correlations'], 'g-', linewidth=2, label='Step 3 (post-divergence)', alpha=0.8)
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_title("Comparison: Step 1 vs Step 3")
    ax3.set_xlabel("Layer")
    ax3.set_ylabel("Correlation (r)")
    ax3.legend()
    ax3.set_xlim(0, 80)
    
    # Plot 4: P-values comparison
    ax4 = axes[1, 1]
    ax4.semilogy(range(81), step1_results['layer_p_values'], 'b-', linewidth=2, label='Step 1', alpha=0.8)
    ax4.semilogy(range(81), step3_results['layer_p_values'], 'g-', linewidth=2, label='Step 3', alpha=0.8)
    ax4.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='p=0.05')
    ax4.set_title("P-values by Layer")
    ax4.set_xlabel("Layer")
    ax4.set_ylabel("P-value (log scale)")
    ax4.legend()
    ax4.set_xlim(0, 80)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "step_comparison.png", dpi=150)
    plt.close()
    
    # Save results
    comparison = {
        "step1": step1_results,
        "step3": step3_results
    }
    with open(OUTPUT_DIR / "step_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\nResults saved to {OUTPUT_DIR}/step_comparison.png")
    print(f"Data saved to {OUTPUT_DIR}/step_comparison.json")
    
    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    if step1_results['best_p_value'] < 0.05:
        print("\n✓ Step 1 shows SIGNIFICANT correlation!")
        print("  → Consistency signal exists BEFORE the agent takes any action")
        print("  → Can predict behavioral consistency from initial representation alone")
        if step1_results['best_p_value'] < step3_results['best_p_value']:
            print("  → Step 1 is actually MORE predictive than Step 3!")
    else:
        print("\n✗ Step 1 does NOT show significant correlation")
        print("  → Consistency signal only emerges after trajectory divergence")
        print("  → Step 3 activations needed for prediction")


if __name__ == "__main__":
    main()
