"""
Combined 40-question analysis for Paper 3 pilot.
Analyzes 20 original "hard" questions + 20 easier questions.

Outputs:
- step_progression_40q.png
- layerwise_correlation_step3_40q.png  
- consistent_vs_inconsistent_similarity.png
- threshold_classifier_roc.png
- summary_table.md
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy import stats
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Directories
PILOT_DIR = Path("pilot_hidden_states_70b")
EASIER_DIR = Path("results_easier")
QUESTIONS_FILE = Path("pilot_questions.json")
EASIER_QUESTIONS_FILE = Path("easier_questions_selection.json")
OUTPUT_DIR = Path("analysis_results/combined_40q")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_answer(text):
    """Extract yes/no from answer text."""
    if not text:
        return None
    text = str(text).lower().strip()
    if "yes" in text and "no" not in text:
        return "yes"
    elif "no" in text:
        return "no"
    elif text.startswith("yes"):
        return "yes"
    return None

def compute_pairwise_cosine_similarity(vectors):
    """Compute mean pairwise cosine similarity."""
    if len(vectors) < 2:
        return np.nan
    arr = np.array(vectors)
    sim_matrix = cosine_similarity(arr)
    # Get upper triangle (excluding diagonal)
    n = len(vectors)
    upper_tri = sim_matrix[np.triu_indices(n, k=1)]
    return np.mean(upper_tri)

def load_pilot_questions():
    """Load pilot questions metadata."""
    questions = {}
    with open(QUESTIONS_FILE) as f:
        for q in json.load(f):
            questions[q["id"]] = q
    return questions

def load_easier_questions():
    """Load easier questions metadata."""
    questions = {}
    with open(EASIER_QUESTIONS_FILE) as f:
        for q in json.load(f):
            questions[q["id"]] = q
    return questions

def load_pilot_data(pilot_questions):
    """Load original 20 questions from pilot_hidden_states_70b (npy format)."""
    print("Loading original 20 questions (npy format)...")
    results = {}
    
    for qdir in PILOT_DIR.iterdir():
        if not qdir.is_dir():
            continue
        qid = qdir.name
        
        runs_data = []
        for run_dir in sorted(qdir.glob("run_*")):
            meta_file = run_dir / "metadata.json"
            traj_file = run_dir / "trajectory.json"
            
            if not meta_file.exists() or not traj_file.exists():
                continue
            
            with open(meta_file) as f:
                meta = json.load(f)
            with open(traj_file) as f:
                traj = json.load(f)
            
            # Load hidden states for each step
            hidden_states = {}
            for hs_file in run_dir.glob("hidden_states_step_*.npy"):
                step_num = int(hs_file.stem.split("_")[-1])
                hs = np.load(hs_file)  # Shape: (81 layers, 8192 dims)
                hidden_states[step_num] = hs
            
            runs_data.append({
                "run_id": run_dir.name,
                "final_answer": traj.get("final_answer") or meta.get("agent_answer"),
                "step_count": len(traj.get("steps", [])),
                "hidden_states": hidden_states,
                "correct": meta.get("correct")
            })
        
        if runs_data and qid in pilot_questions:
            results[qid] = {
                "question": pilot_questions[qid],
                "runs": runs_data,
                "source": "hard"
            }
            print(f"  Loaded {qid}: {len(runs_data)} runs")
    
    return results

def load_easier_data(easier_questions):
    """Load 20 easier questions from results_easier (json format)."""
    print("\nLoading 20 easier questions (json format)...")
    results = {}
    
    for f in sorted(EASIER_DIR.glob("*.json")):
        qid = f.stem
        if qid not in easier_questions:
            continue
        
        print(f"  Loading {qid}...", end=" ", flush=True)
        with open(f) as fp:
            data = json.load(fp)
        
        runs_data = []
        for run in data.get("runs", []):
            # Extract hidden states from steps
            hidden_states = {}
            for step in run.get("steps", []):
                step_num = step.get("step_number", 0)
                hs_data = step.get("hidden_states", {})
                layers = hs_data.get("layers", {})
                
                if layers:
                    # Convert dict of layers to numpy array (81 layers x dims)
                    layer_vecs = []
                    for i in range(81):
                        layer_key = f"layer_{i}"
                        if layer_key in layers and layers[layer_key]:
                            layer_vecs.append(layers[layer_key])
                        else:
                            layer_vecs.append([0] * 8192)  # Placeholder
                    hidden_states[step_num] = np.array(layer_vecs)
            
            runs_data.append({
                "run_id": run.get("run_id"),
                "final_answer": run.get("final_answer"),
                "step_count": len(run.get("steps", [])),
                "hidden_states": hidden_states,
                "correct": None  # Will compute based on expected answer
            })
        
        if runs_data:
            results[qid] = {
                "question": easier_questions[qid],
                "runs": runs_data,
                "source": "easier"
            }
            print(f"{len(runs_data)} runs")
    
    return results

def compute_question_metrics(qid, entry):
    """Compute behavioral consistency and similarity metrics for a question."""
    runs = entry["runs"]
    question = entry["question"]
    expected = question.get("answer", "").lower().strip()
    
    # Step counts and CV
    step_counts = [r["step_count"] for r in runs]
    cv = np.std(step_counts) / np.mean(step_counts) if np.mean(step_counts) > 0 else 0
    
    # Answer correctness
    correct_count = 0
    answers = []
    for run in runs:
        ans = extract_answer(run["final_answer"])
        if ans:
            answers.append(ans)
            if ans == expected:
                correct_count += 1
    
    correct_rate = correct_count / len(runs) if runs else 0
    
    # Determine consistency category
    if cv < 0.15 and correct_rate >= 0.8:
        category = "consistent-correct"
    elif cv < 0.15 and correct_rate < 0.2:
        category = "consistent-wrong"
    else:
        category = "inconsistent"
    
    # Compute pairwise cosine similarity at each step/layer
    # Focus on steps 1-5 and sample layers (0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80)
    similarity_by_step_layer = {}
    
    for step in range(1, 6):
        similarity_by_step_layer[step] = {}
        
        for layer in [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80]:
            vectors = []
            for run in runs:
                hs = run["hidden_states"].get(step)
                if hs is not None and len(hs) > layer:
                    vectors.append(hs[layer])
            
            if len(vectors) >= 2:
                sim = compute_pairwise_cosine_similarity(vectors)
                similarity_by_step_layer[step][layer] = sim
    
    return {
        "qid": qid,
        "source": entry["source"],
        "cv": cv,
        "correct_rate": correct_rate,
        "category": category,
        "step_counts": step_counts,
        "similarity_by_step_layer": similarity_by_step_layer
    }

def run_analysis():
    """Run the complete 40-question analysis."""
    print("=" * 70)
    print("COMBINED 40-QUESTION ANALYSIS")
    print("=" * 70)
    
    # Load questions metadata
    pilot_questions = load_pilot_questions()
    easier_questions = load_easier_questions()
    
    print(f"\nPilot questions loaded: {len(pilot_questions)}")
    print(f"Easier questions loaded: {len(easier_questions)}")
    
    # Load data
    pilot_data = load_pilot_data(pilot_questions)
    easier_data = load_easier_data(easier_questions)
    
    # Combine
    all_data = {**pilot_data, **easier_data}
    print(f"\nTotal questions loaded: {len(all_data)}")
    
    # Compute metrics for each question
    print("\nComputing metrics for each question...")
    metrics = []
    for qid, entry in all_data.items():
        m = compute_question_metrics(qid, entry)
        metrics.append(m)
        print(f"  {qid[:20]}... CV={m['cv']:.3f}, Category={m['category']}")
    
    # ==========================================
    # ANALYSIS 1: Behavioral Consistency Summary
    # ==========================================
    print("\n" + "=" * 70)
    print("ANALYSIS 1: BEHAVIORAL CONSISTENCY SUMMARY")
    print("=" * 70)
    
    categories = defaultdict(list)
    for m in metrics:
        categories[m["category"]].append(m)
    
    print(f"\nCategory breakdown (n={len(metrics)}):")
    for cat in ["consistent-correct", "consistent-wrong", "inconsistent"]:
        items = categories[cat]
        hard = sum(1 for m in items if m["source"] == "hard")
        easy = sum(1 for m in items if m["source"] == "easier")
        print(f"  {cat}: {len(items)} ({hard} hard, {easy} easier)")
    
    # ==========================================
    # ANALYSIS 2: Step Progression at Layer 32
    # ==========================================
    print("\n" + "=" * 70)
    print("ANALYSIS 2: STEP PROGRESSION (Layer 32)")
    print("=" * 70)
    
    step_correlations = {}
    step_data_for_plot = {}
    
    for step in range(1, 6):
        cvs = []
        sims = []
        
        for m in metrics:
            if step in m["similarity_by_step_layer"] and 32 in m["similarity_by_step_layer"][step]:
                sim = m["similarity_by_step_layer"][step][32]
                if not np.isnan(sim):
                    cvs.append(m["cv"])
                    sims.append(sim)
        
        if len(cvs) >= 5:
            corr, pval = stats.pearsonr(sims, cvs)
            step_correlations[step] = {"r": corr, "p": pval, "n": len(cvs)}
            step_data_for_plot[step] = (sims, cvs)
            print(f"  Step {step}: r={corr:.4f}, p={pval:.4f}, n={len(cvs)}")
    
    # Find peak
    if step_correlations:
        # Looking for negative correlation (higher sim = lower CV = more consistent)
        peak_step = min(step_correlations.keys(), key=lambda s: step_correlations[s]["r"])
        print(f"\n  Peak correlation step: {peak_step} (r={step_correlations[peak_step]['r']:.4f})")
    
    # Plot step progression
    fig, ax = plt.subplots(figsize=(10, 6))
    steps = sorted(step_correlations.keys())
    rs = [step_correlations[s]["r"] for s in steps]
    ps = [step_correlations[s]["p"] for s in steps]
    
    bars = ax.bar(steps, rs, color=['green' if p < 0.05 else 'gray' for p in ps])
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel("Step Number", fontsize=12)
    ax.set_ylabel("Pearson r (Similarity vs CV)", fontsize=12)
    ax.set_title(f"Step Progression: Activation Similarity vs Behavioral Consistency\n(Layer 32, n={len(metrics)} questions)", fontsize=14)
    
    for i, (step, r, p) in enumerate(zip(steps, rs, ps)):
        sig = "*" if p < 0.05 else ""
        ax.text(step, r + 0.02 if r >= 0 else r - 0.05, f"{r:.3f}{sig}", ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "step_progression_40q.png", dpi=150)
    plt.close()
    print(f"\n  Saved: step_progression_40q.png")
    
    # ==========================================
    # ANALYSIS 3: Layer-wise Correlation at Step 3
    # ==========================================
    print("\n" + "=" * 70)
    print("ANALYSIS 3: LAYER-WISE CORRELATION (Step 3)")
    print("=" * 70)
    
    layer_correlations = {}
    
    for layer in [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80]:
        cvs = []
        sims = []
        
        for m in metrics:
            if 3 in m["similarity_by_step_layer"] and layer in m["similarity_by_step_layer"][3]:
                sim = m["similarity_by_step_layer"][3][layer]
                if not np.isnan(sim):
                    cvs.append(m["cv"])
                    sims.append(sim)
        
        if len(cvs) >= 5:
            corr, pval = stats.pearsonr(sims, cvs)
            layer_correlations[layer] = {"r": corr, "p": pval, "n": len(cvs)}
            print(f"  Layer {layer:2d}: r={corr:.4f}, p={pval:.4f}, n={len(cvs)}")
    
    # Plot layer-wise
    fig, ax = plt.subplots(figsize=(12, 6))
    layers = sorted(layer_correlations.keys())
    rs = [layer_correlations[l]["r"] for l in layers]
    ps = [layer_correlations[l]["p"] for l in layers]
    
    bars = ax.bar(layers, rs, width=6, color=['green' if p < 0.05 else 'gray' for p in ps])
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel("Layer Number", fontsize=12)
    ax.set_ylabel("Pearson r (Similarity vs CV)", fontsize=12)
    ax.set_title(f"Layer-wise Correlation at Step 3\n(n={len(metrics)} questions)", fontsize=14)
    ax.set_xticks(layers)
    
    for layer, r, p in zip(layers, rs, ps):
        sig = "*" if p < 0.05 else ""
        ax.text(layer, r + 0.02 if r >= 0 else r - 0.05, f"{r:.2f}{sig}", ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "layerwise_correlation_step3_40q.png", dpi=150)
    plt.close()
    print(f"\n  Saved: layerwise_correlation_step3_40q.png")
    
    # ==========================================
    # ANALYSIS 4: Consistent-Correct vs Inconsistent
    # ==========================================
    print("\n" + "=" * 70)
    print("ANALYSIS 4: CONSISTENT-CORRECT vs INCONSISTENT")
    print("=" * 70)
    
    cc_sims = []
    inc_sims = []
    
    for m in metrics:
        if 3 in m["similarity_by_step_layer"] and 32 in m["similarity_by_step_layer"][3]:
            sim = m["similarity_by_step_layer"][3][32]
            if not np.isnan(sim):
                if m["category"] == "consistent-correct":
                    cc_sims.append(sim)
                elif m["category"] == "inconsistent":
                    inc_sims.append(sim)
    
    print(f"\n  Consistent-correct: n={len(cc_sims)}, mean_sim={np.mean(cc_sims):.4f} (±{np.std(cc_sims):.4f})")
    print(f"  Inconsistent:       n={len(inc_sims)}, mean_sim={np.mean(inc_sims):.4f} (±{np.std(inc_sims):.4f})")
    
    if len(cc_sims) >= 2 and len(inc_sims) >= 2:
        t_stat, t_pval = stats.ttest_ind(cc_sims, inc_sims)
        print(f"\n  T-test: t={t_stat:.4f}, p={t_pval:.4f}")
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(cc_sims) + np.var(inc_sims)) / 2)
        cohens_d = (np.mean(cc_sims) - np.mean(inc_sims)) / pooled_std if pooled_std > 0 else 0
        print(f"  Effect size (Cohen's d): {cohens_d:.4f}")
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    
    positions = [1, 2]
    data = [cc_sims, inc_sims]
    labels = [f"Consistent-Correct\n(n={len(cc_sims)})", f"Inconsistent\n(n={len(inc_sims)})"]
    
    bp = ax.boxplot(data, positions=positions, widths=0.5, patch_artist=True)
    bp['boxes'][0].set_facecolor('green')
    bp['boxes'][1].set_facecolor('orange')
    
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Activation Similarity (Step 3, Layer 32)", fontsize=12)
    ax.set_title(f"Activation Similarity by Consistency Category\n(p={t_pval:.4f}, d={cohens_d:.2f})", fontsize=14)
    
    # Add means as points
    ax.scatter([1], [np.mean(cc_sims)], color='darkgreen', s=100, zorder=5, marker='D', label='Mean')
    ax.scatter([2], [np.mean(inc_sims)], color='darkorange', s=100, zorder=5, marker='D')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "consistent_vs_inconsistent_similarity.png", dpi=150)
    plt.close()
    print(f"\n  Saved: consistent_vs_inconsistent_similarity.png")
    
    # ==========================================
    # ANALYSIS 5: Threshold Classifier
    # ==========================================
    print("\n" + "=" * 70)
    print("ANALYSIS 5: THRESHOLD CLASSIFIER")
    print("=" * 70)
    
    # Prepare data: similarity predicting consistency (consistent=1, inconsistent=0)
    all_sims = []
    all_labels = []  # 1 = consistent, 0 = inconsistent
    
    for m in metrics:
        if 3 in m["similarity_by_step_layer"] and 32 in m["similarity_by_step_layer"][3]:
            sim = m["similarity_by_step_layer"][3][32]
            if not np.isnan(sim):
                all_sims.append(sim)
                all_labels.append(1 if m["category"] == "consistent-correct" else 0)
    
    all_sims = np.array(all_sims)
    all_labels = np.array(all_labels)
    
    print(f"\n  Total samples: {len(all_sims)}")
    print(f"  Positive (consistent-correct): {sum(all_labels)}")
    print(f"  Negative (inconsistent/wrong): {len(all_labels) - sum(all_labels)}")
    
    if len(all_sims) >= 5 and len(set(all_labels)) > 1:
        # ROC curve
        fpr, tpr, thresholds = roc_curve(all_labels, all_sims)
        roc_auc = auc(fpr, tpr)
        
        # Find optimal threshold (Youden's J)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        # Compute metrics at optimal threshold
        predictions = (all_sims >= optimal_threshold).astype(int)
        accuracy = np.mean(predictions == all_labels)
        
        tp = np.sum((predictions == 1) & (all_labels == 1))
        fp = np.sum((predictions == 1) & (all_labels == 0))
        fn = np.sum((predictions == 0) & (all_labels == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n  ROC AUC: {roc_auc:.4f}")
        print(f"  Optimal threshold: {optimal_threshold:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1: {f1:.4f}")
        
        # Plot ROC curve
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, label='Random')
        ax.scatter([fpr[optimal_idx]], [tpr[optimal_idx]], color='red', s=100, zorder=5,
                   label=f'Optimal (thresh={optimal_threshold:.3f})')
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title(f"ROC Curve: Predicting Consistency from Activation Similarity\n(Step 3, Layer 32)", fontsize=14)
        ax.legend(loc='lower right')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "threshold_classifier_roc.png", dpi=150)
        plt.close()
        print(f"\n  Saved: threshold_classifier_roc.png")
    
    # ==========================================
    # SUMMARY TABLE
    # ==========================================
    print("\n" + "=" * 70)
    print("GENERATING SUMMARY TABLE")
    print("=" * 70)
    
    summary_md = f"""# Combined 40-Question Analysis Summary

## Dataset
- **Total questions**: {len(metrics)}
- **Hard questions**: {sum(1 for m in metrics if m['source'] == 'hard')}
- **Easier questions**: {sum(1 for m in metrics if m['source'] == 'easier')}

## Behavioral Consistency Categories

| Category | Count | Hard | Easier |
|----------|-------|------|--------|
| Consistent-Correct | {len(categories['consistent-correct'])} | {sum(1 for m in categories['consistent-correct'] if m['source'] == 'hard')} | {sum(1 for m in categories['consistent-correct'] if m['source'] == 'easier')} |
| Consistent-Wrong | {len(categories['consistent-wrong'])} | {sum(1 for m in categories['consistent-wrong'] if m['source'] == 'hard')} | {sum(1 for m in categories['consistent-wrong'] if m['source'] == 'easier')} |
| Inconsistent | {len(categories['inconsistent'])} | {sum(1 for m in categories['inconsistent'] if m['source'] == 'hard')} | {sum(1 for m in categories['inconsistent'] if m['source'] == 'easier')} |

## Step Progression (Layer 32)

| Step | Pearson r | p-value | n |
|------|-----------|---------|---|
"""
    for step in sorted(step_correlations.keys()):
        sc = step_correlations[step]
        sig = "**" if sc["p"] < 0.05 else ""
        summary_md += f"| {step} | {sc['r']:.4f}{sig} | {sc['p']:.4f} | {sc['n']} |\n"
    
    summary_md += f"""
## Layer-wise Correlation at Step 3

| Layer | Pearson r | p-value | n |
|-------|-----------|---------|---|
"""
    for layer in sorted(layer_correlations.keys()):
        lc = layer_correlations[layer]
        sig = "**" if lc["p"] < 0.05 else ""
        summary_md += f"| {layer} | {lc['r']:.4f}{sig} | {lc['p']:.4f} | {lc['n']} |\n"
    
    summary_md += f"""
## Consistent-Correct vs Inconsistent Comparison

| Metric | Value |
|--------|-------|
| CC Mean Similarity | {np.mean(cc_sims):.4f} ± {np.std(cc_sims):.4f} |
| Inc Mean Similarity | {np.mean(inc_sims):.4f} ± {np.std(inc_sims):.4f} |
| T-statistic | {t_stat:.4f} |
| P-value | {t_pval:.4f} |
| Cohen's d | {cohens_d:.4f} |

## Threshold Classifier Performance

| Metric | Value |
|--------|-------|
| ROC AUC | {roc_auc:.4f} |
| Optimal Threshold | {optimal_threshold:.4f} |
| Accuracy | {accuracy:.4f} |
| Precision | {precision:.4f} |
| Recall | {recall:.4f} |
| F1 Score | {f1:.4f} |

## Key Findings

1. **Step 3 peak**: {"Confirmed" if step_correlations.get(3, {}).get("p", 1) < 0.05 else "Not significant"} (r={step_correlations.get(3, {}).get('r', 0):.3f})
2. **Consistent-correct vs inconsistent**: {"Significant difference" if t_pval < 0.05 else "No significant difference"} (p={t_pval:.4f})
3. **Classifier performance**: AUC={roc_auc:.3f}, suggesting {"good" if roc_auc > 0.7 else "moderate" if roc_auc > 0.5 else "poor"} predictive ability

---
*Generated from combined analysis of {len(metrics)} HotpotQA questions*
"""
    
    with open(OUTPUT_DIR / "summary_table.md", "w") as f:
        f.write(summary_md)
    print(f"  Saved: summary_table.md")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_analysis()
