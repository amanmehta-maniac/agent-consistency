"""
Generate figures for agent consistency analysis.
Creates bar charts and histograms comparing Llama, GPT-4o, and Claude results.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.analysis import (
    analyze_task_for_figures as analyze_task,
    categorize_tasks_for_figures as categorize_tasks,
)


def load_results(results_dir: str) -> List[Dict[str, Any]]:
    """Load all result files from directory."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        print(f"Warning: {results_dir} does not exist")
        return []
    
    results = []
    for f in sorted(results_dir.glob("*.json")):
        try:
            with open(f) as file:
                results.append(json.load(file))
        except Exception as e:
            print(f"Error loading {f}: {e}")
            continue
    return results


# analyze_task and categorize_tasks are imported from common.analysis


def create_bar_chart(
    llama_consistent: List[Dict],
    llama_inconsistent: List[Dict],
    gpt4o_consistent: List[Dict],
    gpt4o_inconsistent: List[Dict],
    claude_consistent: List[Dict],
    claude_inconsistent: List[Dict],
    output_dir: str = None,
):
    """Create bar chart comparing correctness for consistent vs inconsistent tasks."""
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "hotpotqa" / "figures"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate average correctness for each category
    llama_cons_correct = np.mean([a["correctness"] for a in llama_consistent]) if llama_consistent else 0.0
    llama_incons_correct = np.mean([a["correctness"] for a in llama_inconsistent]) if llama_inconsistent else 0.0
    gpt4o_cons_correct = np.mean([a["correctness"] for a in gpt4o_consistent]) if gpt4o_consistent else 0.0
    gpt4o_incons_correct = np.mean([a["correctness"] for a in gpt4o_inconsistent]) if gpt4o_inconsistent else 0.0
    claude_cons_correct = np.mean([a["correctness"] for a in claude_consistent]) if claude_consistent else 0.0
    claude_incons_correct = np.mean([a["correctness"] for a in claude_inconsistent]) if claude_inconsistent else 0.0
    
    # Prepare data
    categories = ["Consistent\n(≤2 unique seqs)", "Inconsistent\n(≥8 unique seqs)"]
    llama_values = [llama_cons_correct, llama_incons_correct]
    gpt4o_values = [gpt4o_cons_correct, gpt4o_incons_correct]
    claude_values = [claude_cons_correct, claude_incons_correct]
    
    x = np.arange(len(categories))
    width = 0.25  # Adjusted for 3 bars
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, llama_values, width, label="Llama 3.1 70B", color="#2E86AB", alpha=0.8)  # Blue
    bars2 = ax.bar(x, gpt4o_values, width, label="GPT-4o", color="#A23B72", alpha=0.8)  # Pink
    bars3 = ax.bar(x + width, claude_values, width, label="Claude Sonnet 4.5", color="#FF6B35", alpha=0.8)  # Orange
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2%}',
                       ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel("Average Correctness", fontsize=12)
    ax.set_title("Correctness: Consistent vs Inconsistent Action Sequences", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add sample size annotations
    llama_cons_n = len(llama_consistent)
    llama_incons_n = len(llama_inconsistent)
    gpt4o_cons_n = len(gpt4o_consistent)
    gpt4o_incons_n = len(gpt4o_inconsistent)
    claude_cons_n = len(claude_consistent)
    claude_incons_n = len(claude_inconsistent)
    
    ax.text(0, -0.1, f'n={llama_cons_n}', ha='center', va='top', fontsize=8, color='gray', transform=ax.get_xaxis_transform())
    ax.text(1, -0.1, f'n={llama_incons_n}', ha='center', va='top', fontsize=8, color='gray', transform=ax.get_xaxis_transform())
    ax.text(0, -0.15, f'n={gpt4o_cons_n}', ha='center', va='top', fontsize=8, color='gray', transform=ax.get_xaxis_transform())
    ax.text(1, -0.15, f'n={gpt4o_incons_n}', ha='center', va='top', fontsize=8, color='gray', transform=ax.get_xaxis_transform())
    ax.text(0, -0.2, f'n={claude_cons_n}', ha='center', va='top', fontsize=8, color='gray', transform=ax.get_xaxis_transform())
    ax.text(1, -0.2, f'n={claude_incons_n}', ha='center', va='top', fontsize=8, color='gray', transform=ax.get_xaxis_transform())
    
    plt.tight_layout()
    
    # Save as PNG and PDF
    png_path = output_dir / "correctness_comparison.png"
    pdf_path = output_dir / "correctness_comparison.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved bar chart: {png_path}, {pdf_path}")
    plt.close()


def create_histogram(
    llama_analyses: List[Dict],
    gpt4o_analyses: List[Dict],
    claude_analyses: List[Dict],
    output_dir: str = None,
):
    """Create histogram of unique action sequences distribution."""
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "hotpotqa" / "figures"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    llama_unique_seqs = [a["unique_sequences"] for a in llama_analyses]
    gpt4o_unique_seqs = [a["unique_sequences"] for a in gpt4o_analyses]
    claude_unique_seqs = [a["unique_sequences"] for a in claude_analyses]
    
    # Determine bin range
    all_seqs = llama_unique_seqs + gpt4o_unique_seqs + claude_unique_seqs
    max_seqs = max(all_seqs) if all_seqs else 10
    bins = np.arange(0, max_seqs + 2, 1)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create histogram
    ax.hist(
        llama_unique_seqs,
        bins=bins,
        alpha=0.6,
        label=f"Llama 3.1 70B (n={len(llama_analyses)})",
        color="#2E86AB",  # Blue
        edgecolor='black',
        linewidth=0.5,
    )
    ax.hist(
        gpt4o_unique_seqs,
        bins=bins,
        alpha=0.6,
        label=f"GPT-4o (n={len(gpt4o_analyses)})",
        color="#A23B72",  # Pink
        edgecolor='black',
        linewidth=0.5,
    )
    ax.hist(
        claude_unique_seqs,
        bins=bins,
        alpha=0.6,
        label=f"Claude Sonnet 4.5 (n={len(claude_analyses)})",
        color="#FF6B35",  # Orange
        edgecolor='black',
        linewidth=0.5,
    )
    
    ax.set_xlabel("Number of Unique Action Sequences per Task", fontsize=12)
    ax.set_ylabel("Number of Tasks", fontsize=12)
    ax.set_title("Distribution of Unique Action Sequences", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_xticks(bins[:-1] + 0.5)
    ax.set_xticklabels([int(b) for b in bins[:-1]])
    
    plt.tight_layout()
    
    # Save as PNG and PDF
    png_path = output_dir / "unique_sequences_histogram.png"
    pdf_path = output_dir / "unique_sequences_histogram.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved histogram: {png_path}, {pdf_path}")
    plt.close()


def main():
    """Main function to generate all figures."""
    print("Loading results...")
    
    # Load results from all three directories (assuming running from hotpotqa/)
    base_dir = Path(__file__).parent.parent / "hotpotqa"
    llama_results = load_results(str(base_dir / "results_llama"))
    gpt4o_results = load_results(str(base_dir / "results_gpt4o"))
    claude_results = load_results(str(base_dir / "results_claude"))
    
    print(f"Loaded {len(llama_results)} Llama tasks")
    print(f"Loaded {len(gpt4o_results)} GPT-4o tasks")
    print(f"Loaded {len(claude_results)} Claude tasks")
    
    if not llama_results and not gpt4o_results and not claude_results:
        print("Error: No results found in any directory")
        return
    
    # Analyze tasks
    print("\nAnalyzing tasks...")
    llama_analyses = [analyze_task(task) for task in llama_results]
    gpt4o_analyses = [analyze_task(task) for task in gpt4o_results]
    claude_analyses = [analyze_task(task) for task in claude_results]
    
    # Categorize
    llama_consistent, llama_inconsistent = categorize_tasks(llama_analyses)
    gpt4o_consistent, gpt4o_inconsistent = categorize_tasks(gpt4o_analyses)
    claude_consistent, claude_inconsistent = categorize_tasks(claude_analyses)
    
    print(f"\nLlama:")
    print(f"  Consistent (≤2 unique seqs): {len(llama_consistent)} tasks")
    print(f"  Inconsistent (≥8 unique seqs): {len(llama_inconsistent)} tasks")
    print(f"\nGPT-4o:")
    print(f"  Consistent (≤2 unique seqs): {len(gpt4o_consistent)} tasks")
    print(f"  Inconsistent (≥8 unique seqs): {len(gpt4o_inconsistent)} tasks")
    print(f"\nClaude:")
    print(f"  Consistent (≤2 unique seqs): {len(claude_consistent)} tasks")
    print(f"  Inconsistent (≥8 unique seqs): {len(claude_inconsistent)} tasks")
    
    # Create figures
    print("\nGenerating figures...")
    create_bar_chart(
        llama_consistent,
        llama_inconsistent,
        gpt4o_consistent,
        gpt4o_inconsistent,
        claude_consistent,
        claude_inconsistent,
    )
    create_histogram(llama_analyses, gpt4o_analyses, claude_analyses)
    
    print("\n✓ All figures generated successfully!")


if __name__ == "__main__":
    main()
