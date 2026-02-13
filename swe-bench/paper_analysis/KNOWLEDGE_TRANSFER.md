# Knowledge Transfer: Paper Analysis Pipeline

This document explains how all analysis in `paper_analysis/` was produced, including procedural steps and methodology.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Data Collection](#2-data-collection)
3. [Core Analysis (run_analysis.py)](#3-core-analysis)
4. [Accuracy Evaluation](#4-accuracy-evaluation)
5. [Divergence Analysis](#5-divergence-analysis)
6. [First Action Analysis](#6-first-action-analysis)
7. [Qualitative Case Studies](#7-qualitative-case-studies)
8. [Sanity Check & Verification](#8-sanity-check--verification)
9. [File Reference](#9-file-reference)

---

## 1. Project Overview

### Goal
Measure **behavioral consistency** of LLM code agents: Do they take the same actions when given the same task multiple times?

### Experiment Design
- **Models**: Claude Sonnet 4.5, Llama-3.1-70B-Instruct
- **Tasks**: 10 astropy issues from SWE-bench
- **Runs**: 5 runs per task per model = 100 total runs
- **Agent**: mini-swe-agent with Docker environment
- **Max steps**: 250
- **Temperature**: 0.5

### Key Metrics
| Metric | Definition |
|--------|------------|
| **CV (Coefficient of Variation)** | `std(steps) / mean(steps) * 100%` - lower = more consistent |
| **Accuracy** | Runs where SWE-bench tests pass |
| **Divergence Step** | First step where runs differ (by action category) |

---

## 2. Data Collection

### Running Experiments

```bash
# Claude experiments
python3.11 runner.py \
    --model claude-sonnet-4-5-20250929 \
    --provider anthropic \
    --swebench \
    --n-tasks 10 \
    --n-runs 5 \
    --max-steps 250 \
    --temperature 0.5 \
    --results-dir results_claude_10

# Llama experiments
python3.11 runner.py \
    --model meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo \
    --provider together \
    --swebench \
    --n-tasks 10 \
    --n-runs 5 \
    --max-steps 250 \
    --temperature 0.5 \
    --results-dir results_llama_10
```

### Output Structure
```
results_claude_10/
├── astropy__astropy-12907.json
├── astropy__astropy-13033.json
├── ... (10 files total)

results_llama_10/
├── astropy__astropy-12907.json
├── ... (10 files total)
```

### Result JSON Schema
```json
{
  "task_id": "astropy__astropy-12907",
  "runs": [
    {
      "run_id": 1,
      "n_steps": 45,
      "action_sequence": ["find ...", "ls ...", "cat ..."],
      "exit_status": "Submitted",
      "success": true,
      "final_output": "<patch content>"
    }
  ],
  "unique_sequences": 5,
  "avg_steps": 46.1
}
```

---

## 3. Core Analysis (run_analysis.py)

### Location
`paper_analysis/run_analysis.py` (881 lines)

### How to Run
```bash
cd swe-bench/paper_analysis
python3.11 run_analysis.py
```

### What It Produces

#### Figures (`figures/`)
| Figure | Description |
|--------|-------------|
| `fig1_cv_distribution.png` | Box plot comparing CV per model |
| `fig2_step_heatmap.png` | Heatmap of step counts (10 tasks × 5 runs × 2 models) |
| `fig3_step_distributions.png` | Histogram of all step counts |
| `fig4_tradeoff_scatter.png` | Scatter: mean steps vs CV per task |
| `fig5_per_task_bars.png` | Grouped bar chart: mean steps per task |
| `fig6_accuracy.png` | Accuracy comparison |
| `fig7_consistency_accuracy.png` | CV vs accuracy scatter |
| `fig8_phase_decomposition.png` | Action type breakdown |
| `fig10_failure_modes.png` | Failure mode distribution |

#### Tables (`tables/`)
| Table | Description |
|-------|-------------|
| `table1_overall.tex` | Overall results (steps, CV, accuracy) |
| `table2_per_task.tex` | Per-task breakdown |
| `table3_statistics.tex` | Statistical tests (t-test, Mann-Whitney, Cohen's d) |
| `table4_accuracy.tex` | Accuracy per task |
| `table5_phase_variance.tex` | Phase-level variance |
| `table6_failure_taxonomy.tex` | Failure categorization |

#### Data Files
| File | Description |
|------|-------------|
| `raw_metrics.json` | All computed metrics |
| `accuracy_data.json` | Per-run accuracy from SWE-bench |
| `phase_data.json` | Action phase breakdown |
| `failure_data.json` | Failure categorization |

### Key Algorithms

#### CV Calculation
```python
for task in tasks:
    steps = [run['n_steps'] for run in task['runs']]
    mean = np.mean(steps)
    std = np.std(steps, ddof=1)
    cv = (std / mean) * 100  # as percentage
```

#### Action Categorization
```python
def categorize_action(action):
    if action.startswith(('ls', 'find', 'cd', 'tree')):
        return 'EXPLORE'
    if action.startswith(('cat', 'head', 'tail', 'grep', 'less')):
        return 'UNDERSTAND'
    if action.startswith(('sed', 'echo')) or '>>' in action:
        return 'EDIT'
    if action.startswith(('python', 'pytest')):
        return 'VERIFY'
    return 'OTHER'
```

#### Statistical Tests
```python
from scipy import stats

# t-test on CV distributions
t_stat, p_value = stats.ttest_ind(claude_cvs, llama_cvs)

# Mann-Whitney U (non-parametric)
u_stat, p_mw = stats.mannwhitneyu(claude_cvs, llama_cvs)

# Cohen's d effect size
pooled_std = np.sqrt((np.var(claude_cvs) + np.var(llama_cvs)) / 2)
cohens_d = (np.mean(llama_cvs) - np.mean(claude_cvs)) / pooled_std
```

---

## 4. Accuracy Evaluation

### Step 1: Generate Prediction Files

```python
# For each run, extract the patch and create JSONL
for run in range(1, 6):
    predictions = []
    for task_file in results_dir.glob("*.json"):
        data = json.loads(task_file.read_text())
        patch = data["runs"][run-1]["final_output"]
        predictions.append({
            "instance_id": data["task_id"],
            "model_patch": patch,
            "model_name_or_path": "claude"
        })
    
    with open(f"preds_claude_run{run}.jsonl", "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")
```

### Step 2: Run SWE-bench Evaluation

```bash
# Local evaluation (all tasks)
python -m swebench.harness.run_evaluation \
    --predictions_path preds_claude_run1.jsonl \
    --max_workers 4 \
    --dataset_name princeton-nlp/SWE-bench \
    --run_id claude-local-run1
```

### Step 3: Parse Results

```python
# Evaluation produces: claude.claude-local-run1.json
report = json.loads(Path("claude.claude-local-run1.json").read_text())
resolved_ids = report["resolved_ids"]  # Tasks that passed tests
```

### Important Distinction
| Field | Location | Meaning |
|-------|----------|---------|
| `success` | Agent result JSON | Agent submitted something |
| `resolved` | SWE-bench report | Tests actually passed |

---

## 5. Divergence Analysis

### Location
`paper_analysis/divergence/`

### Methodology

**Definition**: Divergence step = first step where not all 5 runs have the same action CATEGORY.

```python
def find_divergence_step(runs):
    """
    runs: list of 5 action sequences
    returns: first step where actions differ (1-indexed)
    """
    for step in range(min(len(r) for r in runs)):
        categories = set(categorize_action(r[step]) for r in runs)
        if len(categories) > 1:
            return step + 1  # 1-indexed
    return None
```

### Action Categories Used
- **EXPLORE**: ls, find, cd, tree, pwd
- **UNDERSTAND**: cat, head, tail, grep, less
- **EDIT**: sed, echo, cat <<, >>
- **VERIFY**: python, pytest
- **OTHER**: everything else

### Output Files
| File | Description |
|------|-------------|
| `divergence_summary.csv` | Per-task divergence step |
| `aggregate_stats.json` | Mean, median divergence stats |
| `statistical_tests.json` | t-test comparing models |
| `divergence_report.md` | Full narrative report |
| `figures/fig_divergence_*.png` | Visualizations |

### Key Finding
- **Claude**: Mean divergence at step 3.1
- **Llama**: Mean divergence at step 1.4
- Claude is 2.2x more consistent in early steps

---

## 6. First Action Analysis

### Location
`paper_analysis/first_action/`

### Methodology

Extract the first action from each of 100 runs and analyze:

```python
# Extract first actions
first_actions = []
for task_file in results_dir.glob("*.json"):
    for run in data["runs"]:
        action = run["action_sequence"][0]
        cmd = action.split()[0]  # e.g., "find", "ls", "grep"
        first_actions.append(cmd)

# Chi-square test
from scipy.stats import chi2_contingency
contingency = [[claude_find, claude_ls], [llama_find, llama_ls]]
chi2, p, dof, expected = chi2_contingency(contingency)
```

### Output Files
| File | Description |
|------|-------------|
| `first_action_summary.csv` | All 100 first actions |
| `chi_square_results.json` | Statistical tests |
| `first_action_report.md` | Narrative summary |
| `figures/fig_first_action_*.png` | Visualizations |

### Key Finding
- Claude: 68% start with `find`, 26% with `ls`
- Llama: 54% start with `ls`, 24% with `grep`
- First action predicts MODEL, not success (p < 0.0001)

---

## 7. Qualitative Case Studies

### Location
Various `.md` files in `paper_analysis/`

### Llama Success Analysis (`llama_success_analysis.md`)

**Procedure**:
1. Identify Llama's 2 successful runs (out of 50)
2. Extract full action sequences
3. Compare to failed runs on same tasks
4. Identify what made them succeed

**Command to regenerate**:
```python
# Find successful Llama runs from evaluation reports
for run in range(1, 6):
    report = json.loads(Path(f"llama.llama-local-run{run}.json").read_text())
    for task_id in report["resolved_ids"]:
        print(f"Run {run}: {task_id}")
```

### Claude Failure Analysis (`claude_13236_failure_analysis.md`)

**Procedure**:
1. Load `results_claude_10/astropy__astropy-13236.json`
2. For each of 5 runs, extract the approach taken
3. Compare to Llama's successful run on same task
4. Identify where Claude went wrong

### Claude Failure Patterns (`claude_failure_patterns.md`)

**Procedure**:
1. Categorize all 21 Claude failures
2. Group by pattern: CONSISTENT_WRONG, RANDOM_FAIL, CLOSE_BUT_WRONG
3. Count occurrences of each pattern

### Step 5 Divergence (`step5_divergence_analysis.md`)

**Procedure**:
1. Load astropy-13236 for both models
2. Extract step 5 action and reasoning
3. Compare thought processes side-by-side

---

## 8. Sanity Check & Verification

### Correctness Verification (`correctness_verification/`)

**Purpose**: Confirm that accuracy numbers are real (tests actually ran)

**Procedure**:
```python
# Check evaluation logs exist
logs = Path("logs/run_evaluation/claude-local-run1/claude/")
for task_log in logs.glob("*/run_instance.log"):
    content = task_log.read_text()
    # Verify test runtime (should be 100+ seconds)
    assert "Test runtime:" in content
```

**Evidence collected**:
- Test runtime logs (110+ seconds per task)
- Docker container names
- FAIL_TO_PASS / PASS_TO_PASS test results
- Git diffs showing patches applied

### Sanity Check (`sanity_check/`)

**Purpose**: Verify all paper claims against raw JSON files

**Procedure**:
```python
# Verify basic counts
assert len(list(Path("results_claude_10").glob("*.json"))) == 10
assert len(list(Path("results_llama_10").glob("*.json"))) == 10

for f in Path("results_claude_10").glob("*.json"):
    data = json.loads(f.read_text())
    assert len(data["runs"]) == 5  # 5 runs per task

# Verify accuracy
claude_success = sum(sum(resolved[task]) for task in tasks)
assert claude_success == 29  # 58%
```

**Output**:
- `verification_report.md` - Full verification results
- `discrepancies.txt` - Any differences found
- `raw_counts.json` - All raw numbers

---

## 9. File Reference

### Directory Structure
```
paper_analysis/
├── run_analysis.py           # Main analysis script
├── analysis_report.md        # Generated narrative report
├── project_summary.md        # High-level summary
├── paper_claims.md           # Paper-ready claims
│
├── figures/                  # All visualizations (PNG + PDF)
│   ├── fig1_cv_distribution.png
│   ├── fig2_step_heatmap.png
│   └── ...
│
├── tables/                   # LaTeX tables
│   ├── table1_overall.tex
│   ├── table2_per_task.tex
│   └── ...
│
├── divergence/               # Divergence step analysis
│   ├── divergence_report.md
│   ├── divergence_summary.csv
│   └── figures/
│
├── first_action/             # First action analysis
│   ├── first_action_report.md
│   ├── chi_square_results.json
│   └── figures/
│
├── sanity_check/             # Verification of claims
│   ├── verification_report.md
│   └── raw_counts.json
│
├── correctness_verification/ # SWE-bench evaluation proof
│   ├── conclusion.md
│   └── evaluation_evidence.md
│
├── qualitative/              # Qualitative analysis
│   ├── variance_source_analysis.md
│   └── failure_analysis.md
│
├── case_studies/             # Detailed case studies
│   ├── case1_claude_success_vs_failure.md
│   └── ...
│
├── llama_success_analysis.md     # Llama's 2 successes
├── claude_13236_failure_analysis.md  # Claude's failure on 13236
├── claude_failure_patterns.md    # Pattern analysis
├── step5_divergence_analysis.md  # Step 5 comparison
│
├── raw_metrics.json          # All computed metrics
├── accuracy_data.json        # SWE-bench accuracy
├── phase_data.json           # Phase breakdown
└── failure_data.json         # Failure categorization
```

### Raw Data Location
```
swe-bench/
├── results_claude_10/        # Claude experiment results
├── results_llama_10/         # Llama experiment results
├── claude.claude-local-run*.json  # SWE-bench evaluation reports
├── llama.llama-local-run*.json    # SWE-bench evaluation reports
└── logs/run_evaluation/      # Test execution logs
```

---

## Regenerating All Analysis

To regenerate everything from scratch:

```bash
cd swe-bench/paper_analysis

# 1. Core analysis (figures, tables, metrics)
python3.11 run_analysis.py

# 2. Divergence analysis (run inline Python)
# See divergence/ generation code in advisor responses

# 3. First action analysis (run inline Python)
# See first_action/ generation code in advisor responses

# 4. Verify claims against raw data
# See sanity_check/ generation code
```

---

## Key Findings Summary

| Finding | Claude | Llama | Significance |
|---------|--------|-------|--------------|
| Accuracy | 58% | 4% | p < 0.001 |
| Mean Steps | 46.1 | 17.0 | Claude 2.7x slower |
| Mean CV | 13.6% | 42.0% | Claude 3x more consistent |
| Divergence Step | 3.1 | 1.4 | Claude 2.2x later |
| First Action | 68% find | 54% ls | Different strategies |

---

## Contact

For questions about this analysis, refer to:
- `advisor_response*.md` files (local, not in git)
- GitHub: https://github.com/amanmehta-maniac/agent-consistency
