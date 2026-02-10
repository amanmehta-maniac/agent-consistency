# SWE-bench Consistency Experiment Results

## Expanded Experiment (10 Tasks × 5 Runs)

**Date**: Feb 4-5, 2026  
**Temperature**: 0.5  
**Max Steps**: 250  
**Tasks**: 10 (astropy repository)  
**Runs per task**: 5  
**Total runs per model**: 50

### Summary Comparison

| Metric | Llama-3.1-70B | Claude 4.5 Sonnet |
|--------|---------------|-------------------|
| **Avg Steps** | **17.0** 🏆 | 46.1 |
| **Avg CV (consistency)** | 47.0% | **15.2%** 🏆 |
| **Unique Sequences** | 5.0/5 (100%) | 5.0/5 (100%) |
| **Success Rate** | **100%** | **100%** |
| **Avg Cost/Run** | **$0.11** 🏆 | $2.78 |
| **Step Range** | 7-55 | 26-64 |

> **Note**: Initial Claude results showed 70% success due to $3 cost_limit. After increasing to $10, Claude achieved 100% success.

### Per-Task Results: Llama-3.1-70B

| Task | Steps [Run 1-5] | Mean | CV | Success |
|------|-----------------|------|-----|---------|
| astropy-12907 | [35, 13, 18, 7, 10] | 16.6 | 67% | 100% |
| astropy-13033 | [10, 36, 18, 28, 20] | 22.4 | 44% | 100% |
| astropy-13236 | [7, 21, 11, 14, 13] | 13.2 | 39% | 100% |
| astropy-13398 | [17, 13, 12, 8, 27] | 15.4 | 47% | 100% |
| astropy-13453 | [12, 16, 10, 9, 8] | 11.0 | 29% | 100% |
| astropy-13579 | [9, 13, 22, 21, 7] | 14.4 | 48% | 100% |
| astropy-13977 | [19, 10, 11, 18, 8] | 13.2 | 38% | 100% |
| astropy-14096 | [30, 17, 15, 20, 29] | 22.2 | 31% | 100% |
| astropy-14182 | [18, 21, 14, 55, 26] | 26.8 | 61% | 100% |
| astropy-14309 | [11, 31, 16, 8, 7] | 14.6 | 67% | 100% |

### Per-Task Results: Claude 4.5 Sonnet (with $10 cost limit)

| Task | Steps [Run 1-5] | Mean | CV | Success |
|------|-----------------|------|-----|---------|
| astropy-12907 | [26, 35, 43, 46, 41] | 38.2 | 21% | 100% |
| astropy-13033 | [35, 46, 38, 44, 50] | 42.6 | 14% | 100% |
| astropy-13236 | [31, 56, 29, 43, 44] | 40.6 | 27% | 100% |
| astropy-13398 | [48, 46, 55, 45, 45] | 47.8 | 9% | 100% |
| astropy-13453 | [38, 42, 48, 45, 33] | 41.2 | 14% | 100% |
| astropy-13579 | [52, 59, 44, 55, 54] | 52.8 | 10% | 100% |
| astropy-13977 | [41, 53, 48, 59, 44] | 49.0 | 15% | 100% |
| astropy-14096 | [40, 58, 53, 44, 40] | 47.0 | 17% | 100% |
| astropy-14182 | [49, 64, 54, 52, 47] | 53.2 | 12% | 100% |
| astropy-14309 | [57, 51, 45, 47, 42] | 48.4 | 12% | 100% |

### Key Findings from Expanded Experiment

1. **Both achieve 100% success** when given adequate budget
   - Initial Claude results showed 70% due to $3 cost limit
   - After increasing to $10, Claude achieved 100% success

2. **Claude is 3x more consistent** (CV 15.2% vs 47%)
   - Step counts within narrow ranges (e.g., 45-55 vs 7-55)
   - Predictable behavior across runs

3. **Llama is 2.7x faster but more variable**
   - 17 vs 46 steps on average
   - High variance indicates exploratory/opportunistic approach

4. **Claude costs 25x more** ($2.78 vs $0.11 per run)
   - Important for budget-constrained experiments
   - Cost limit can cause artificial "failures"

5. **Both produce 100% unique action sequences**
   - Every run takes a different path
   - Temperature 0.5 doesn't eliminate variability

### Raw Data Location

| Model | Results Directory |
|-------|-------------------|
| Claude 4.5 Sonnet | `results_claude_10/` |
| Llama-3.1-70B | `results_llama_10/` |

---

## Paper 2 Pilot Experiment (Format Error Fix Applied)

**Date**: Feb 3, 2026  
**Temperature**: 0.5  
**Max Steps**: 250  
**Format Error Fix**: ✅ Applied (submission blocked during format error recovery)

### Summary Table

| Model | Tasks | Runs/Task | Unique Seqs | Avg Steps | Step Variance (CV) | Format Errors |
|-------|-------|-----------|-------------|-----------|-------------------|---------------|
| **Llama-3.1-70B-Instruct** | 3 | 3 | 3.0 | 26.7 | 90% | 0 |
| **Claude 4.5 Sonnet** | 3 | 3 | 3.0 | 43.0 | 9% | 0 |
| **GPT-4o-mini** | 1* | 3 | 3.0 | 7.3 | 21% | 0 |

*GPT-4o-mini only completed 1 task due to OpenAI API issues - repeatedly hung on Task 2 despite timeout/retry settings. Llama (Together API) and Claude (Anthropic API) had no such issues.

---

### Per-Task Results

#### Task 1: `astropy__astropy-12907`
**Description:** Modeling's `separability_matrix` does not compute separability correctly for nested CompoundModels

| Model | Run 1 | Run 2 | Run 3 | Avg | Stdev | CV |
|-------|-------|-------|-------|-----|-------|-----|
| Llama-3.1-70B | 25 | 7 | 12 | 14.7 | 9.3 | 63% |
| Claude 4.5 Sonnet | 31 | 29 | 35 | 31.7 | 3.1 | **10%** |
| GPT-4o-mini | 9 | 7 | 6 | 7.3 | 1.5 | 21% |

#### Task 2: `astropy__astropy-13033`
**Description:** TimeSeries: misleading exception when required column check fails

| Model | Run 1 | Run 2 | Run 3 | Avg | Stdev | CV |
|-------|-------|-------|-------|-----|-------|-----|
| Llama-3.1-70B | 31 | 17 | 5 | 17.7 | 13.0 | 74% |
| Claude 4.5 Sonnet | 38 | 43 | 41 | 40.7 | 2.5 | **6%** |
| GPT-4o-mini | - | - | - | - | - | - |

#### Task 3: `astropy__astropy-13236`
**Description:** Consider removing auto-transform of structured column into NdarrayMixin

| Model | Run 1 | Run 2 | Run 3 | Avg | Stdev | CV |
|-------|-------|-------|-------|-----|-------|-----|
| Llama-3.1-70B | 120 | 17 | 6 | 47.7 | 62.9 | **132%** |
| Claude 4.5 Sonnet | 64 | 55 | 51 | 56.7 | 6.7 | **12%** |
| GPT-4o-mini | - | - | - | - | - | - |

---

### First 3 Actions Comparison (Where Runs Diverge)

#### Llama-3.1-70B-Instruct
```
Task: astropy-12907
  Run 1: ['grep -r "def separability_matrix" .', 'nl -ba ./astropy/modeling/separable.py...', 'sed -i ...']
  Run 2: ['ls -la', 'cd astropy', 'grep -r "separability_matrix" .']
  Run 3: ['ls -la', 'find astropy/modeling/separable -type f...', 'find astropy/modeling -type f...']

Task: astropy-13236 (MOST VARIABLE - 120 vs 6 steps!)
  Run 1: ['grep -r "data = data.view(NdarrayMixin)" .', 'sed -i...', 'python -m compileall...']  → 120 steps
  Run 2: ['grep -ri "data.view(NdarrayMixin)" .', 'sed -i...', 'cat <<EOF > test_script.py']  → 17 steps
  Run 3: ['grep -r "data.view(NdarrayMixin)" .', 'cat astropy/table/table.py...', 'cat...']  → 6 steps
```

#### Claude 4.5 Sonnet (temp 0.5)
```
Task: astropy-12907
  Run 1: ['ls -la', 'find astropy -name "*separab*"...', 'cat astropy/modeling/separable.py']
  Run 2: ['find /testbed -type f -name "*.py"...', 'cat /testbed/astropy/modeling/separable.py', 'cat <<EOF...']
  Run 3: ['find /testbed -type f -name "*.py"...', 'cat /testbed/astropy/modeling/separable.py', 'cat <<EOF...']

Task: astropy-13236
  Run 1: ['find /testbed -type f -name "*.py"...', 'find /testbed/astropy/table...', 'ls -la /testbed/astropy/table/']
  Run 2: ['find /testbed -type f -name "*.py"...', 'find /testbed/astropy/table...', 'ls -la /testbed/astropy/table/*.py']
  Run 3: ['find /testbed -type f -name "*.py"...', 'find /testbed/astropy/table...', 'ls -la /testbed/astropy/table/*.py']
```

---

## Key Findings

### ✅ 1. FORMAT ERROR FIX SUCCESSFUL
- **Zero format errors** across all models and all runs
- Previous experiment had format errors causing premature 1-step submissions
- The fix (blocking submission during format error recovery) works

### 📊 2. Claude Shows Highest Consistency (Verified ✓)
- **Coefficient of Variation (CV)**: Claude 6-12% vs Llama 63-132%
- Claude takes similar number of steps across runs (very consistent)
- Llama shows extreme variability (120 steps vs 6 steps on same task)

**CV Calculation Verification:**
| Model | Task | Steps | Mean | Stdev | CV |
|-------|------|-------|------|-------|-----|
| Llama | 12907 | [25, 7, 12] | 14.7 | 9.3 | 63% ✓ |
| Llama | 13033 | [31, 17, 5] | 17.7 | 13.0 | 74% ✓ |
| Llama | 13236 | [120, 17, 6] | 47.7 | 62.9 | 132% ✓ |
| Claude | 12907 | [31, 29, 35] | 31.7 | 3.1 | 10% ✓ |
| Claude | 13033 | [38, 43, 41] | 40.7 | 2.5 | 6% ✓ |
| Claude | 13236 | [64, 55, 51] | 56.7 | 6.7 | 12% ✓ |

**Why Llama's 120-step run is real (not a bug):**
- Run 1 got stuck in a **debugging loop**
- Ran `python -m compileall` **48 times** trying to compile after edits
- Kept trying `sed` commands to fix indentation errors
- This is inefficient but legitimate problem-solving behavior
- Run 3 found a direct solution in just 6 steps (lucky/efficient path)

### 🔄 3. All Models Produce Unique Action Sequences
- Every run produced a unique action sequence (3/3 per task)
- Even at temperature 0.5, models don't follow the same path
- Claude's runs converge toward similar patterns after first action

### ⚡ 4. Step Efficiency vs Consistency Tradeoff
| Model | Avg Steps | Consistency (CV) |
|-------|-----------|------------------|
| GPT-4o-mini | 7.3 | 21% |
| Llama-3.1-70B | 26.7 | 90% |
| Claude 4.5 Sonnet | 43.0 | **9%** |

- GPT-4o-mini is fastest but incomplete
- Claude is slowest but most consistent
- Llama varies wildly (exploration-heavy)

### 🎯 5. Divergence Pattern
- **First action differs**: All models start with exploration (ls, find, grep)
- **Llama**: Often jumps straight to `sed -i` edits (risky)
- **Claude**: Reads files before editing, uses test scripts

---

## Raw Data Location

| Model | Results Directory |
|-------|-------------------|
| Claude 4.5 Sonnet (temp 0.5) | `results_claude_temp05_fixed/` |
| Llama-3.1-70B-Instruct | `results_llama/` |
| GPT-4o-mini | `results_gpt4o_mini/` |

---

## Experiment Configuration

- **Dataset**: SWE-bench Verified
- **Tasks**: 3 (astropy repository issues)
- **Runs per task**: 3
- **Temperature**: 0.5 (reduced from 0.7)
- **Max steps**: 250
- **Environment**: Docker (per-task SWE-bench images)
- **Format Error Fix**: Applied (submission blocked during format error recovery)

---

## Previous Experiment Results (Pre-Fix)

*Note: These results had format error issues causing anomalous 1-step runs*

| Model | Tasks | Runs | Success Rate | Avg Unique Seqs | Format Errors |
|-------|-------|------|--------------|-----------------|---------------|
| Claude 4.5 Sonnet (temp 0.7) | 3 | 9 | 100% | 3.0 | 2 (caused 1-step runs) |
| GPT-4o | 3 | 9 | 44% | 2.7 | 0 (rate limit issues) |

---

## Model Information

| Experiment | Model | Model ID |
|------------|-------|----------|
| Paper 2 Pilot | Claude 4.5 Sonnet | `claude-sonnet-4-5-20250929` |
| Paper 2 Pilot | Llama-3.1-70B | `meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo` |
| Paper 2 Pilot | GPT-4o-mini | `gpt-4o-mini` |
