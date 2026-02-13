# Advisor Response: GPT-5 (Snowflake Cortex) Experiment Results

**Date**: February 13, 2026  
**Experiment**: GPT-5 via Snowflake Cortex on SWE-bench Verified (Astropy subset)

---

## Executive Summary

We successfully ran GPT-5 (openai-gpt-5) through Snowflake Cortex on 10 astropy tasks from SWE-bench Verified with 5 runs each (50 total runs). The key finding is that **GPT-5's behavioral consistency falls between Claude and Llama**, with a mean CV of **32.2%** compared to Claude's 15.2% and Llama's 47.0%.

### Consistency Ranking (Most to Least Consistent)
1. **Claude 4.5 Sonnet**: CV 15.2%
2. **GPT-5 (Snowflake)**: CV 32.2%  
3. **Llama-3.1-70B**: CV 47.0%

---

## Key Findings

### 1. Consistency Metrics

| Model | Mean CV | Interpretation |
|-------|---------|----------------|
| Claude 4.5 Sonnet | 15.2% | Most consistent |
| **GPT-5 (Snowflake)** | **32.2%** | **Intermediate** |
| Llama-3.1-70B | 47.0% | Least consistent |

**Statistical Significance:**
- GPT-5 vs Claude: **p = 0.018** (significant), Cohen's d = 1.16 (large effect)
- GPT-5 vs Llama: **p = 0.071** (not significant), Cohen's d = -0.86 (large effect)

**Interpretation**: GPT-5 is significantly less consistent than Claude but not significantly different from Llama at α=0.05.

### 2. Step Count Comparison

| Model | Mean Steps | Median | Range |
|-------|------------|--------|-------|
| Claude | 26.0 | 24.0 | 6-82 |
| **GPT-5** | **9.9** | **8.0** | **4-29** |
| Llama | 9.6 | 7.0 | 4-44 |

**Key Insight**: GPT-5 takes significantly fewer steps than Claude (similar to Llama), suggesting it may be making quicker but potentially less thorough attempts.

### 3. Divergence Analysis

| Model | Mean Divergence Step |
|-------|---------------------|
| Claude | 3.1 |
| **GPT-5** | **3.1** |
| Llama | 1.4 |

GPT-5's divergence point (step 3.1) matches Claude's, meaning runs stay similar for the first ~3 steps before diverging. This is notably better than Llama's immediate divergence at step 1.4.

### 4. Phase Decomposition

GPT-5's action distribution:
- **VERIFY (32.3%)**: Heavy reliance on running Python
- **OTHER (28.0%)**: Many unclassified actions
- **EXPLORE (23.9%)**: Directory/file exploration
- **EDIT (14.7%)**: Code modifications
- **UNDERSTAND (1.1%)**: Very little file reading

**Comparison to Claude/Llama**: GPT-5 shows a different pattern - more verification-focused with less explicit understanding phase.

### 5. First Action Analysis

| Model | Most Common First Action |
|-------|-------------------------|
| Claude | find (56%), ls (40%) |
| **GPT-5** | **ls (100%)** |
| Llama | grep (42%), ls (32%) |

**Striking Finding**: GPT-5 **always** starts with `ls` - 100% consistency in first action choice across all 50 runs. This deterministic first action contrasts sharply with Claude and Llama's varied approaches.

### 6. Failure Modes

All 50 GPT-5 runs failed to resolve the issues:
- **WRONG_FIX**: 58% (produced a patch, but incorrect)
- **EMPTY_PATCH**: 42% (failed to produce a valid patch)
- **LOOP_DEATH**: 0% (no infinite loops)

*Note: Accuracy requires separate SWE-bench evaluation; the `resolved` field in results shows no successful fixes.*

---

## Implications for Paper

### For "Behavioral Consistency in Code Agents"

1. **Three-way comparison strengthens findings**: The gradient from Claude (15.2%) → GPT-5 (32.2%) → Llama (47.0%) suggests consistency is a differentiating characteristic across frontier models.

2. **Model family patterns**: 
   - Claude: Most consistent, most steps, thorough exploration
   - GPT-5: Intermediate consistency, fast execution, deterministic start
   - Llama: Least consistent, fast but erratic

3. **Unique Sequence Rate**: All three models produce 100% unique action sequences across runs, confirming that behavioral variance is universal regardless of consistency level.

4. **First Action Determinism**: GPT-5's 100% `ls` first action is notable - suggests stronger prompt adherence or less exploration of alternatives.

---

## Experimental Setup

- **Model**: openai-gpt-5 via Snowflake Cortex COMPLETE()
- **Provider**: Snowflake (Account AKB73862)
- **Temperature**: 0.5
- **Max Steps**: 250
- **Tasks**: 10 astropy issues from SWE-bench Verified
- **Runs per Task**: 5
- **Total Runs**: 50

---

## Files Generated

1. **Analysis Report**: `paper_analysis/gpt5_analysis_report.md`
2. **JSON Results**: `paper_analysis/gpt5_analysis_results.json`
3. **Figures**:
   - `figures/fig_cv_3models.png` - CV distribution (3 models)
   - `figures/fig_steps_3models.png` - Step count histogram (3 models)
   - `figures/fig_gpt5_heatmap.png` - GPT-5 step counts heatmap

---

## Next Steps

1. **Run SWE-bench evaluation**: Execute the evaluation harness to get actual `resolved` status for accuracy metrics
2. **Update paper tables**: Add GPT-5 row to Tables 4-8
3. **Statistical analysis**: Include GPT-5 in correlation analyses
4. **Consider additional models**: Test other Snowflake Cortex models (Claude, Mistral) for cross-platform comparison

---

## Raw Data Summary

### Per-Task Results

| Task | Steps (mean±std) | CV% |
|------|------------------|-----|
| astropy-12907 | 11.0±3.4 | 30.8% |
| astropy-13033 | 9.2±6.1 | 66.7% |
| astropy-13236 | 7.0±2.4 | 35.0% |
| astropy-13398 | 9.4±1.9 | 20.7% |
| astropy-13453 | 16.6±6.5 | 39.2% |
| astropy-13579 | 8.4±1.1 | 13.6% |
| astropy-13977 | 7.0±0.0 | 0.0% |
| astropy-14096 | 16.2±9.0 | 55.6% |
| astropy-14182 | 7.2±1.3 | 18.1% |
| astropy-14309 | 7.2±3.0 | 42.1% |

**Notable**: Task 13977 achieved CV=0% (perfect consistency across 5 runs, all completing in exactly 7 steps).

---

*Generated by Cortex Code analysis pipeline*
*Experiment completed: February 13, 2026 03:19 PST*
