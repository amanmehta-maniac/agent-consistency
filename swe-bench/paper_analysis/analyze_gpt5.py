#!/usr/bin/env python3
"""
Comprehensive Analysis for GPT-5 (via Snowflake Cortex)
Matches methodology from Claude/Llama analysis for Paper 2: "Behavioral Consistency in Code Agents"
"""

import json
import numpy as np
import pandas
import pandas as pd
from pathlib import Path
from scipy import stats
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: matplotlib/seaborn not available, skipping visualizations")

# Paths
RESULTS_DIR = Path(__file__).parent.parent
GPT5_DIR = RESULTS_DIR / "results_gpt5_snowflake"
CLAUDE_DIR = RESULTS_DIR / "results_claude_10"
LLAMA_DIR = RESULTS_DIR / "results_llama_10"
OUTPUT_DIR = Path(__file__).parent
FIGURES_DIR = OUTPUT_DIR / "figures"


def load_results(results_dir):
    """Load all results from a directory"""
    data = []
    for json_file in sorted(results_dir.glob("*.json")):
        result = json.loads(json_file.read_text())
        task_id = result.get("task_id", json_file.stem)
        short_id = task_id.split("__")[1] if "__" in task_id else task_id
        
        runs = result.get("runs", [])
        steps_list = [r.get("n_steps", len(r.get("steps", []))) for r in runs]
        
        # Get action sequences from steps
        action_seqs = []
        for r in runs:
            if "action_sequence" in r:
                action_seqs.append(r["action_sequence"])
            elif "steps" in r:
                # Extract actions from steps
                actions = [s.get("action", "") for s in r.get("steps", [])]
                action_seqs.append(actions)
            else:
                action_seqs.append([])
        
        # Get resolved status for accuracy
        resolved_list = [r.get("resolved", False) for r in runs]
        
        if len(steps_list) >= 2:
            mean_steps = np.mean(steps_list)
            std_steps = np.std(steps_list, ddof=1)
            cv = (std_steps / mean_steps * 100) if mean_steps > 0 else 0
        else:
            mean_steps = steps_list[0] if steps_list else 0
            std_steps = 0
            cv = 0
        
        unique_seqs = len(set(tuple(seq) for seq in action_seqs))
        
        data.append({
            "task_id": short_id,
            "full_task_id": task_id,
            "steps": steps_list,
            "mean_steps": mean_steps,
            "std_steps": std_steps,
            "cv": cv,
            "unique_sequences": unique_seqs,
            "n_runs": len(runs),
            "action_sequences": action_seqs,
            "first_actions": [seq[0] if seq else "" for seq in action_seqs],
            "resolved": resolved_list,
            "accuracy": sum(resolved_list) / len(resolved_list) if resolved_list else 0
        })
    
    return data


def categorize_action(action):
    """Categorize an action into a phase/type"""
    if not action:
        return "other"
    action = action.strip().lower()
    
    # EXPLORE phase
    if action.startswith("ls"):
        return "ls"
    elif action.startswith("find"):
        return "find"
    elif action.startswith("grep") or action.startswith("rg"):
        return "grep"
    elif action.startswith("cd"):
        return "cd"
    
    # UNDERSTAND phase
    elif action.startswith("cat") or action.startswith("head") or action.startswith("tail") or action.startswith("less"):
        return "cat/read"
    
    # EDIT phase
    elif action.startswith("sed") or action.startswith("awk"):
        return "sed/edit"
    elif action.startswith("echo") and (">" in action or ">>" in action):
        return "echo/write"
    elif action.startswith("patch"):
        return "patch"
    
    # VERIFY phase
    elif action.startswith("python"):
        return "python"
    elif action.startswith("pytest"):
        return "pytest"
    
    else:
        return "other"


def categorize_phase(action):
    """Map action to phase: EXPLORE, UNDERSTAND, EDIT, VERIFY"""
    cat = categorize_action(action)
    if cat in ["ls", "find", "grep", "cd"]:
        return "EXPLORE"
    elif cat in ["cat/read"]:
        return "UNDERSTAND"
    elif cat in ["sed/edit", "echo/write", "patch"]:
        return "EDIT"
    elif cat in ["python", "pytest"]:
        return "VERIFY"
    else:
        return "OTHER"


def analyze_phase_decomposition(data):
    """Analyze action distribution across phases"""
    phase_counts = {"EXPLORE": 0, "UNDERSTAND": 0, "EDIT": 0, "VERIFY": 0, "OTHER": 0}
    action_type_counts = Counter()
    
    for task in data:
        for seq in task["action_sequences"]:
            for action in seq:
                phase = categorize_phase(action)
                phase_counts[phase] += 1
                action_type_counts[categorize_action(action)] += 1
    
    total = sum(phase_counts.values())
    phase_pcts = {k: v/total*100 if total > 0 else 0 for k, v in phase_counts.items()}
    
    return {
        "phase_counts": phase_counts,
        "phase_percentages": phase_pcts,
        "action_type_counts": dict(action_type_counts),
        "total_actions": total
    }


def classify_failure_mode(task_data):
    """Classify failure modes for unsuccessful runs"""
    failures = []
    for i, resolved in enumerate(task_data["resolved"]):
        if not resolved:
            steps = task_data["steps"][i] if i < len(task_data["steps"]) else 0
            seq = task_data["action_sequences"][i] if i < len(task_data["action_sequences"]) else []
            
            # Check for loop death (repeated actions)
            if len(seq) > 10:
                last_5 = seq[-5:]
                if len(set(last_5)) <= 2:
                    failures.append("LOOP_DEATH")
                    continue
            
            # Check for empty patch (no edit actions)
            has_edit = any(categorize_phase(a) == "EDIT" for a in seq)
            if not has_edit:
                failures.append("EMPTY_PATCH")
                continue
            
            # Default: wrong fix
            failures.append("WRONG_FIX")
    
    return Counter(failures)


def find_divergence_point(sequences):
    """Find at which step runs diverge"""
    if len(sequences) < 2:
        return len(sequences[0]) if sequences else 0
    
    min_len = min(len(s) for s in sequences)
    for step in range(min_len):
        actions_at_step = set(categorize_action(s[step]) for s in sequences)
        if len(actions_at_step) > 1:
            return step + 1  # 1-indexed
    return min_len


def compute_sequence_similarity(sequences):
    """Compute pairwise Jaccard similarity between action sequences"""
    if len(sequences) < 2:
        return 1.0
    
    similarities = []
    for i in range(len(sequences)):
        for j in range(i+1, len(sequences)):
            set1 = set(sequences[i])
            set2 = set(sequences[j])
            if len(set1) == 0 and len(set2) == 0:
                sim = 1.0
            elif len(set1 | set2) == 0:
                sim = 0.0
            else:
                sim = len(set1 & set2) / len(set1 | set2)
            similarities.append(sim)
    
    return np.mean(similarities) if similarities else 0.0


def run_full_analysis(gpt5_data, claude_data=None, llama_data=None):
    """Run the complete analysis pipeline"""
    results = {}
    
    # ==== 1. BASIC METRICS ====
    all_steps = [s for d in gpt5_data for s in d["steps"]]
    all_cvs = [d["cv"] for d in gpt5_data]
    all_accuracies = [d["accuracy"] for d in gpt5_data]
    
    results["basic_metrics"] = {
        "model": "GPT-5 (Snowflake Cortex)",
        "n_tasks": len(gpt5_data),
        "n_runs_per_task": gpt5_data[0]["n_runs"] if gpt5_data else 0,
        "total_runs": sum(d["n_runs"] for d in gpt5_data),
        "steps": {
            "mean": np.mean(all_steps),
            "median": np.median(all_steps),
            "std": np.std(all_steps, ddof=1),
            "min": min(all_steps),
            "max": max(all_steps)
        },
        "cv": {
            "mean": np.mean(all_cvs),
            "median": np.median(all_cvs),
            "std": np.std(all_cvs, ddof=1),
            "ci_95": stats.t.interval(0.95, len(all_cvs)-1,
                                       loc=np.mean(all_cvs),
                                       scale=stats.sem(all_cvs)) if len(all_cvs) > 1 else (0, 0)
        },
        "accuracy": {
            "mean": np.mean(all_accuracies) * 100,
            "per_task": {d["task_id"]: d["accuracy"]*100 for d in gpt5_data}
        },
        "unique_sequences": {
            "mean": np.mean([d["unique_sequences"] for d in gpt5_data]),
            "all_unique_rate": sum(d["unique_sequences"] == d["n_runs"] for d in gpt5_data) / len(gpt5_data)
        }
    }
    
    # ==== 2. PER-TASK BREAKDOWN (Table 8) ====
    results["per_task"] = []
    for d in gpt5_data:
        results["per_task"].append({
            "task_id": d["task_id"],
            "steps": d["steps"],
            "mean_steps": d["mean_steps"],
            "std_steps": d["std_steps"],
            "cv": d["cv"],
            "accuracy": d["accuracy"] * 100,
            "unique_sequences": d["unique_sequences"],
            "resolved": d["resolved"]
        })
    
    # ==== 3. PHASE DECOMPOSITION (Table 4) ====
    results["phase_decomposition"] = analyze_phase_decomposition(gpt5_data)
    
    # ==== 4. FAILURE MODE CLASSIFICATION (Table 5) ====
    all_failures = Counter()
    for d in gpt5_data:
        all_failures.update(classify_failure_mode(d))
    
    total_failures = sum(all_failures.values())
    results["failure_modes"] = {
        "counts": dict(all_failures),
        "percentages": {k: v/total_failures*100 if total_failures > 0 else 0 
                       for k, v in all_failures.items()},
        "total_failures": total_failures
    }
    
    # ==== 5. DIVERGENCE ANALYSIS (Table 6) ====
    divergence_points = []
    similarities = []
    for d in gpt5_data:
        dp = find_divergence_point(d["action_sequences"])
        divergence_points.append(dp)
        sim = compute_sequence_similarity(d["action_sequences"])
        similarities.append(sim)
    
    results["divergence"] = {
        "divergence_point": {
            "mean": np.mean(divergence_points),
            "median": np.median(divergence_points),
            "per_task": {gpt5_data[i]["task_id"]: divergence_points[i] for i in range(len(gpt5_data))}
        },
        "sequence_similarity": {
            "mean": np.mean(similarities),
            "std": np.std(similarities, ddof=1),
            "per_task": {gpt5_data[i]["task_id"]: similarities[i] for i in range(len(gpt5_data))}
        }
    }
    
    # ==== 6. FIRST ACTION ANALYSIS (Table 7) ====
    first_actions = [categorize_action(a) for d in gpt5_data for a in d["first_actions"] if a]
    first_action_counts = Counter(first_actions)
    
    results["first_action"] = {
        "distribution": dict(first_action_counts),
        "total": len(first_actions),
        "most_common": first_action_counts.most_common(3)
    }
    
    # ==== 7. STATISTICAL COMPARISONS (Appendix C) ====
    if claude_data and llama_data:
        claude_cvs = [d["cv"] for d in claude_data]
        llama_cvs = [d["cv"] for d in llama_data]
        gpt5_cvs = all_cvs
        
        # GPT-5 vs Claude
        t_gpt5_claude, p_gpt5_claude = stats.ttest_ind(gpt5_cvs, claude_cvs)
        u_gpt5_claude, p_u_gpt5_claude = stats.mannwhitneyu(gpt5_cvs, claude_cvs, alternative='two-sided')
        
        # GPT-5 vs Llama
        t_gpt5_llama, p_gpt5_llama = stats.ttest_ind(gpt5_cvs, llama_cvs)
        u_gpt5_llama, p_u_gpt5_llama = stats.mannwhitneyu(gpt5_cvs, llama_cvs, alternative='two-sided')
        
        # Cohen's d calculations
        def cohens_d(g1, g2):
            pooled_std = np.sqrt(((len(g1)-1)*np.var(g1, ddof=1) + 
                                  (len(g2)-1)*np.var(g2, ddof=1)) / 
                                 (len(g1) + len(g2) - 2))
            return (np.mean(g1) - np.mean(g2)) / pooled_std if pooled_std > 0 else 0
        
        d_gpt5_claude = cohens_d(gpt5_cvs, claude_cvs)
        d_gpt5_llama = cohens_d(gpt5_cvs, llama_cvs)
        
        results["statistical_comparisons"] = {
            "gpt5_vs_claude": {
                "t_test": {"t": t_gpt5_claude, "p": p_gpt5_claude},
                "mann_whitney": {"u": u_gpt5_claude, "p": p_u_gpt5_claude},
                "cohens_d": d_gpt5_claude,
                "cv_ratio": np.mean(gpt5_cvs) / np.mean(claude_cvs) if np.mean(claude_cvs) > 0 else float('inf')
            },
            "gpt5_vs_llama": {
                "t_test": {"t": t_gpt5_llama, "p": p_gpt5_llama},
                "mann_whitney": {"u": u_gpt5_llama, "p": p_u_gpt5_llama},
                "cohens_d": d_gpt5_llama,
                "cv_ratio": np.mean(gpt5_cvs) / np.mean(llama_cvs) if np.mean(llama_cvs) > 0 else float('inf')
            },
            "summary": {
                "claude_cv": np.mean(claude_cvs),
                "llama_cv": np.mean(llama_cvs),
                "gpt5_cv": np.mean(gpt5_cvs)
            }
        }
    
    return results


def generate_markdown_report(results, claude_data=None, llama_data=None):
    """Generate comprehensive markdown report"""
    bm = results["basic_metrics"]
    
    report = f"""# GPT-5 (Snowflake Cortex) Analysis Report
## Behavioral Consistency in Code Agents

---

## Executive Summary

**Model**: {bm['model']}
**Tasks**: {bm['n_tasks']} astropy tasks from SWE-bench Verified
**Runs per task**: {bm['n_runs_per_task']}
**Total runs**: {bm['total_runs']}
**Temperature**: 0.5

### Key Metrics

| Metric | Value |
|--------|-------|
| Mean CV (Consistency) | {bm['cv']['mean']:.1f}% |
| Mean Steps | {bm['steps']['mean']:.1f} |
| Mean Accuracy | {bm['accuracy']['mean']:.1f}% |
| Unique Sequence Rate | {bm['unique_sequences']['all_unique_rate']*100:.0f}% |

---

## 1. Basic Metrics

### Step Count Statistics
| Statistic | Value |
|-----------|-------|
| Mean | {bm['steps']['mean']:.2f} |
| Median | {bm['steps']['median']:.2f} |
| Std Dev | {bm['steps']['std']:.2f} |
| Min | {bm['steps']['min']} |
| Max | {bm['steps']['max']} |

### Coefficient of Variation (CV)
| Statistic | Value |
|-----------|-------|
| Mean | {bm['cv']['mean']:.2f}% |
| Median | {bm['cv']['median']:.2f}% |
| Std Dev | {bm['cv']['std']:.2f}% |
| 95% CI | ({bm['cv']['ci_95'][0]:.2f}%, {bm['cv']['ci_95'][1]:.2f}%) |

---

## 2. Per-Task Breakdown (Appendix Table 8)

| Task ID | Steps (mean±std) | CV (%) | Accuracy | Unique Seqs |
|---------|------------------|--------|----------|-------------|
"""
    
    for t in results["per_task"]:
        report += f"| {t['task_id']} | {t['mean_steps']:.1f}±{t['std_steps']:.1f} | {t['cv']:.1f} | {t['accuracy']:.0f}% | {t['unique_sequences']}/5 |\n"
    
    # Phase decomposition
    pd = results["phase_decomposition"]
    report += f"""
---

## 3. Phase Decomposition (Table 4)

Total actions analyzed: {pd['total_actions']}

| Phase | Count | Percentage |
|-------|-------|------------|
| EXPLORE | {pd['phase_counts']['EXPLORE']} | {pd['phase_percentages']['EXPLORE']:.1f}% |
| UNDERSTAND | {pd['phase_counts']['UNDERSTAND']} | {pd['phase_percentages']['UNDERSTAND']:.1f}% |
| EDIT | {pd['phase_counts']['EDIT']} | {pd['phase_percentages']['EDIT']:.1f}% |
| VERIFY | {pd['phase_counts']['VERIFY']} | {pd['phase_percentages']['VERIFY']:.1f}% |
| OTHER | {pd['phase_counts']['OTHER']} | {pd['phase_percentages']['OTHER']:.1f}% |

### Action Type Distribution
"""
    for action, count in sorted(pd['action_type_counts'].items(), key=lambda x: -x[1]):
        pct = count / pd['total_actions'] * 100 if pd['total_actions'] > 0 else 0
        report += f"- **{action}**: {count} ({pct:.1f}%)\n"
    
    # Failure modes
    fm = results["failure_modes"]
    report += f"""
---

## 4. Failure Mode Classification (Table 5)

Total failures: {fm['total_failures']}

| Failure Mode | Count | Percentage |
|--------------|-------|------------|
"""
    for mode in ["WRONG_FIX", "EMPTY_PATCH", "LOOP_DEATH"]:
        count = fm['counts'].get(mode, 0)
        pct = fm['percentages'].get(mode, 0)
        report += f"| {mode} | {count} | {pct:.1f}% |\n"
    
    # Divergence analysis
    div = results["divergence"]
    report += f"""
---

## 5. Divergence Analysis (Table 6)

### Divergence Point (step where runs start differing)
| Statistic | Value |
|-----------|-------|
| Mean | {div['divergence_point']['mean']:.2f} |
| Median | {div['divergence_point']['median']:.2f} |

### Per-Task Divergence Points
| Task ID | Divergence Step |
|---------|-----------------|
"""
    for task_id, dp in div['divergence_point']['per_task'].items():
        report += f"| {task_id} | {dp} |\n"
    
    report += f"""
### Sequence Similarity (Jaccard)
| Statistic | Value |
|-----------|-------|
| Mean | {div['sequence_similarity']['mean']:.3f} |
| Std Dev | {div['sequence_similarity']['std']:.3f} |

---

## 6. First Action Analysis (Table 7)

Total first actions: {results['first_action']['total']}

| Action Type | Count | Percentage |
|-------------|-------|------------|
"""
    for action, count in sorted(results['first_action']['distribution'].items(), key=lambda x: -x[1]):
        pct = count / results['first_action']['total'] * 100
        report += f"| {action} | {count} | {pct:.1f}% |\n"
    
    # Statistical comparisons
    if "statistical_comparisons" in results:
        sc = results["statistical_comparisons"]
        report += f"""
---

## 7. Statistical Comparisons (Appendix C)

### Cross-Model CV Comparison
| Model | Mean CV |
|-------|---------|
| Claude 4.5 Sonnet | {sc['summary']['claude_cv']:.1f}% |
| Llama-3.1-70B | {sc['summary']['llama_cv']:.1f}% |
| **GPT-5 (Snowflake)** | **{sc['summary']['gpt5_cv']:.1f}%** |

### GPT-5 vs Claude
| Test | Statistic | p-value | Significant (α=0.05) |
|------|-----------|---------|----------------------|
| Independent t-test | t={sc['gpt5_vs_claude']['t_test']['t']:.3f} | {sc['gpt5_vs_claude']['t_test']['p']:.4f} | {'Yes' if sc['gpt5_vs_claude']['t_test']['p'] < 0.05 else 'No'} |
| Mann-Whitney U | U={sc['gpt5_vs_claude']['mann_whitney']['u']:.1f} | {sc['gpt5_vs_claude']['mann_whitney']['p']:.4f} | {'Yes' if sc['gpt5_vs_claude']['mann_whitney']['p'] < 0.05 else 'No'} |

- **Cohen's d**: {sc['gpt5_vs_claude']['cohens_d']:.3f}
- **CV Ratio (GPT-5/Claude)**: {sc['gpt5_vs_claude']['cv_ratio']:.2f}x

### GPT-5 vs Llama
| Test | Statistic | p-value | Significant (α=0.05) |
|------|-----------|---------|----------------------|
| Independent t-test | t={sc['gpt5_vs_llama']['t_test']['t']:.3f} | {sc['gpt5_vs_llama']['t_test']['p']:.4f} | {'Yes' if sc['gpt5_vs_llama']['t_test']['p'] < 0.05 else 'No'} |
| Mann-Whitney U | U={sc['gpt5_vs_llama']['mann_whitney']['u']:.1f} | {sc['gpt5_vs_llama']['mann_whitney']['p']:.4f} | {'Yes' if sc['gpt5_vs_llama']['mann_whitney']['p'] < 0.05 else 'No'} |

- **Cohen's d**: {sc['gpt5_vs_llama']['cohens_d']:.3f}
- **CV Ratio (GPT-5/Llama)**: {sc['gpt5_vs_llama']['cv_ratio']:.2f}x
"""
    
    report += """
---

## 8. Key Findings

"""
    # Generate key findings based on results
    cv_mean = bm['cv']['mean']
    if "statistical_comparisons" in results:
        sc = results["statistical_comparisons"]
        if cv_mean < sc['summary']['claude_cv']:
            report += f"1. **GPT-5 is MORE consistent than Claude**: CV {cv_mean:.1f}% vs {sc['summary']['claude_cv']:.1f}%\n"
        else:
            report += f"1. **GPT-5 is LESS consistent than Claude**: CV {cv_mean:.1f}% vs {sc['summary']['claude_cv']:.1f}%\n"
        
        if cv_mean < sc['summary']['llama_cv']:
            report += f"2. **GPT-5 is MORE consistent than Llama**: CV {cv_mean:.1f}% vs {sc['summary']['llama_cv']:.1f}%\n"
        else:
            report += f"2. **GPT-5 is LESS consistent than Llama**: CV {cv_mean:.1f}% vs {sc['summary']['llama_cv']:.1f}%\n"
    
    report += f"3. **Accuracy**: {bm['accuracy']['mean']:.1f}% of runs produced correct fixes\n"
    report += f"4. **Divergence**: Runs diverge on average at step {div['divergence_point']['mean']:.1f}\n"
    report += f"5. **All runs produce unique sequences**: {bm['unique_sequences']['all_unique_rate']*100:.0f}% of tasks have 5/5 unique sequences\n"
    
    report += f"""
---

*Analysis generated: {pandas.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
*Model: openai-gpt-5 via Snowflake Cortex*
*Provider: Snowflake Account AKB73862*
"""
    
    return report


def generate_figures(gpt5_data, claude_data=None, llama_data=None, output_dir=None):
    """Generate visualization figures"""
    if not HAS_PLOTTING:
        print("Skipping figures - matplotlib not available")
        return
    
    if output_dir is None:
        output_dir = FIGURES_DIR
    output_dir.mkdir(exist_ok=True)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Figure 1: CV Distribution comparison (3 models)
    if claude_data and llama_data:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        claude_cvs = [d["cv"] for d in claude_data]
        llama_cvs = [d["cv"] for d in llama_data]
        gpt5_cvs = [d["cv"] for d in gpt5_data]
        
        data = pd.DataFrame({
            "CV (%)": claude_cvs + llama_cvs + gpt5_cvs,
            "Model": ["Claude"] * len(claude_cvs) + ["Llama"] * len(llama_cvs) + ["GPT-5"] * len(gpt5_cvs)
        })
        
        sns.violinplot(data=data, x="Model", y="CV (%)", ax=ax, inner=None, alpha=0.7)
        sns.stripplot(data=data, x="Model", y="CV (%)", ax=ax, color="black", alpha=0.5, size=8)
        
        ax.set_title("Coefficient of Variation Distribution by Model", fontsize=14, fontweight='bold')
        ax.set_ylabel("Coefficient of Variation (%)", fontsize=12)
        
        plt.tight_layout()
        fig.savefig(output_dir / "fig_cv_3models.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 2: Step count comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        claude_steps = [s for d in claude_data for s in d["steps"]]
        llama_steps = [s for d in llama_data for s in d["steps"]]
        gpt5_steps = [s for d in gpt5_data for s in d["steps"]]
        
        ax.hist(claude_steps, bins=15, alpha=0.5, label=f"Claude (μ={np.mean(claude_steps):.1f})", color='blue')
        ax.hist(llama_steps, bins=15, alpha=0.5, label=f"Llama (μ={np.mean(llama_steps):.1f})", color='orange')
        ax.hist(gpt5_steps, bins=15, alpha=0.5, label=f"GPT-5 (μ={np.mean(gpt5_steps):.1f})", color='green')
        
        ax.set_xlabel("Step Count", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title("Step Count Distributions", fontsize=14, fontweight='bold')
        ax.legend()
        
        plt.tight_layout()
        fig.savefig(output_dir / "fig_steps_3models.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Figure 3: GPT-5 specific heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    steps_matrix = np.array([d["steps"] for d in gpt5_data])
    task_labels = [d["task_id"].replace("astropy-", "") for d in gpt5_data]
    
    sns.heatmap(steps_matrix, ax=ax, cmap="YlOrRd", annot=True, fmt="d",
                xticklabels=[f"Run {i+1}" for i in range(5)],
                yticklabels=task_labels)
    ax.set_title("GPT-5 Step Counts Across Runs", fontsize=14, fontweight='bold')
    ax.set_ylabel("Task", fontsize=12)
    
    plt.tight_layout()
    fig.savefig(output_dir / "fig_gpt5_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Figures saved to {output_dir}")


if __name__ == "__main__":
    print("=" * 60)
    print("GPT-5 (Snowflake Cortex) Full Analysis Pipeline")
    print("=" * 60)
    
    # Load data
    print("\nLoading results...")
    gpt5_data = load_results(GPT5_DIR)
    print(f"  GPT-5: {len(gpt5_data)} tasks loaded")
    
    claude_data = None
    llama_data = None
    
    if CLAUDE_DIR.exists():
        claude_data = load_results(CLAUDE_DIR)
        print(f"  Claude: {len(claude_data)} tasks loaded")
    
    if LLAMA_DIR.exists():
        llama_data = load_results(LLAMA_DIR)
        print(f"  Llama: {len(llama_data)} tasks loaded")
    
    # Run analysis
    print("\nRunning full analysis...")
    results = run_full_analysis(gpt5_data, claude_data, llama_data)
    
    # Save JSON results
    json_path = OUTPUT_DIR / "gpt5_analysis_results.json"
    with open(json_path, 'w') as f:
        # Convert numpy types for JSON serialization
        def convert(o):
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            raise TypeError
        json.dump(results, f, indent=2, default=convert)
    print(f"\n  JSON results saved to: {json_path}")
    
    # Generate markdown report
    print("\nGenerating markdown report...")
    report = generate_markdown_report(results, claude_data, llama_data)
    report_path = OUTPUT_DIR / "gpt5_analysis_report.md"
    report_path.write_text(report)
    print(f"  Report saved to: {report_path}")
    
    # Generate figures
    print("\nGenerating figures...")
    generate_figures(gpt5_data, claude_data, llama_data)
    
    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE - KEY FINDINGS")
    print("=" * 60)
    bm = results["basic_metrics"]
    print(f"\n  GPT-5 Mean CV: {bm['cv']['mean']:.1f}%")
    print(f"  GPT-5 Mean Steps: {bm['steps']['mean']:.1f}")
    print(f"  GPT-5 Accuracy: {bm['accuracy']['mean']:.1f}%")
    
    if "statistical_comparisons" in results:
        sc = results["statistical_comparisons"]
        print(f"\n  Claude Mean CV: {sc['summary']['claude_cv']:.1f}%")
        print(f"  Llama Mean CV: {sc['summary']['llama_cv']:.1f}%")
        print(f"\n  GPT-5 vs Claude: p={sc['gpt5_vs_claude']['t_test']['p']:.4f}, d={sc['gpt5_vs_claude']['cohens_d']:.3f}")
        print(f"  GPT-5 vs Llama: p={sc['gpt5_vs_llama']['t_test']['p']:.4f}, d={sc['gpt5_vs_llama']['cohens_d']:.3f}")
