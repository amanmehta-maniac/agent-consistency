# Comprehensive Analysis Report
## Paper 2: "Behavioral Consistency in Code Agents"

---

## Executive Summary

This analysis compares behavioral consistency between Claude 4.5 Sonnet and Llama-3.1-70B-Instruct
across 10 SWE-bench tasks with 5 runs each (50 runs per model).

### Key Findings

- **Consistency (CV)**: Claude avg 15.2% vs Llama avg 47.0%
- **Speed**: Claude avg 46.1 steps vs Llama avg 17.0 steps
- **Accuracy**: Claude 58% vs Llama 4% (SWE-bench evaluation)
- **Statistical Significance**: p = 0.0000
- **Effect Size**: Cohen's d = -3.020 (large)

---

## 1. Statistical Analysis

### 1.1 Per-Model Summary

#### Claude 4.5 Sonnet

| Metric | Steps | CV (%) |
|--------|-------|--------|
| Mean | 46.08 | 15.19 |
| Median | 45.50 | 14.27 |
| Std Dev | 8.08 | 5.35 |
| Range | 26-64 | - |
| 95% CI for CV | (11.37, 19.02) | - |

#### Llama-3.1-70B-Instruct

| Metric | Steps | CV (%) |
|--------|-------|--------|
| Mean | 16.98 | 47.00 |
| Median | 14.50 | 45.67 |
| Std Dev | 9.40 | 13.90 |
| Range | 7-55 | - |
| 95% CI for CV | (37.06, 56.94) | - |

### 1.2 Statistical Tests

| Test | Statistic | p-value | Interpretation |
|------|-----------|---------|----------------|
| Independent t-test | t = -6.7535 | 0.0000 | Significant at α=0.05 |
| Mann-Whitney U | U = 0.0 | 0.0002 | Significant at α=0.05 |
| Cohen's d | -3.0203 | - | Large effect |

---

## 2. Per-Task Results

| Task ID | Claude Steps (mean±std) | Claude CV (%) | Llama Steps (mean±std) | Llama CV (%) |
|---------|------------------------|---------------|------------------------|--------------|
| astropy-12907 | 38.2±7.9 | 20.7 | 16.6±11.1 | 66.6 |
| astropy-13033 | 42.6±6.1 | 14.2 | 22.4±9.9 | 44.4 |
| astropy-13236 | 40.6±11.0 | 27.0 | 13.2±5.1 | 38.8 |
| astropy-13398 | 47.8±4.2 | 8.8 | 15.4±7.2 | 47.0 |
| astropy-13453 | 41.2±5.9 | 14.3 | 11.0±3.2 | 28.7 |
| astropy-13579 | 52.8±5.5 | 10.5 | 14.4±6.8 | 47.5 |
| astropy-13977 | 49.0±7.2 | 14.6 | 13.2±5.0 | 37.7 |
| astropy-14096 | 47.0±8.1 | 17.3 | 22.2±6.9 | 31.1 |
| astropy-14182 | 53.2±6.6 | 12.4 | 26.8±16.4 | 61.1 |
| astropy-14309 | 48.4±5.8 | 12.0 | 14.6±9.8 | 67.2 |

---

## 3. Divergence Analysis

### 3.1 First Action Distribution

| Action Type | Claude | Llama |
|-------------|--------|-------|
| cat/read | 2 | 1 |
| find | 34 | 10 |
| grep | 1 | 12 |
| ls | 13 | 27 |

**Chi-square test**: χ² = 27.63, p = 0.0000 (significant)

### 3.2 Intra-Task Similarity (Jaccard)

| Model | Mean Similarity | Std Dev |
|-------|----------------|---------|
| Claude | 0.069 | 0.041 |
| Llama | 0.030 | 0.021 |

### 3.3 Divergence Point

| Model | Mean Step | Median Step |
|-------|-----------|-------------|
| Claude | 0.3 | 0.0 |
| Llama | 0.1 | 0.0 |

---

## 4. Correlation Analysis

Does harder task → more variance?

| Model | Correlation (r) | p-value | Interpretation |
|-------|----------------|---------|----------------|
| Claude | -0.504 | 0.1376 | Not significant |
| Llama | 0.247 | 0.4906 | Not significant |

---

## 5. Claims Verification

### Claude 3X More Consistent
- **Status**: ✅ VERIFIED
- **Interpretation**: Claude is 3.1x more consistent (CV ratio)

### Llama 2 7X Faster
- **Status**: ✅ VERIFIED
- **Interpretation**: Claude takes 2.7x more steps than Llama

### 100 Percent Unique Sequences
- **Status**: ✅ VERIFIED
- **Interpretation**: Claude: 100% unique, Llama: 100% unique

### Large Effect Size
- **Status**: ✅ VERIFIED
- **Interpretation**: Cohen's d = -3.020 (large)

### Statistically Significant
- **Status**: ✅ VERIFIED
- **Interpretation**: p = 0.0000 (significant at α=0.01)

---

## 6. Accuracy Analysis (SWE-bench Evaluation)

### 6.1 Overall Accuracy

| Model | Total Resolved | Total Runs | Accuracy |
|-------|---------------|------------|----------|
| Claude 4.5 Sonnet | 29 | 50 | **58%** |
| Llama-3.1-70B | 2 | 50 | **4%** |

### 6.2 Accuracy Per Run

| Run | Claude | Llama |
|-----|--------|-------|
| Run 1 | 5/10 (50%) | 0/10 (0%) |
| Run 2 | 6/10 (60%) | 0/10 (0%) |
| Run 3 | 7/10 (70%) | 0/10 (0%) |
| Run 4 | 5/10 (50%) | 1/10 (10%) |
| Run 5 | 6/10 (60%) | 1/10 (10%) |

### 6.3 Per-Task Accuracy

| Task ID | Claude (resolved/runs) | Llama (resolved/runs) |
|---------|------------------------|----------------------|
| astropy-12907 | 5/5 (100%) | 0/5 (0%) |
| astropy-13033 | 3/5 (60%) | 0/3 (0%) |
| astropy-13236 | 0/5 (0%) | 1/4 (25%) |
| astropy-13398 | 0/5 (0%) | 0/1 (0%) |
| astropy-13453 | 5/5 (100%) | 0/5 (0%) |
| astropy-13579 | 5/5 (100%) | 0/5 (0%) |
| astropy-13977 | 0/5 (0%) | 0/4 (0%) |
| astropy-14096 | 5/5 (100%) | 0/5 (0%) |
| astropy-14182 | 1/5 (20%) | 0/5 (0%) |
| astropy-14309 | 5/5 (100%) | 1/3 (33%) |

### 6.4 Accuracy Consistency

| Metric | Claude | Llama |
|--------|--------|-------|
| Tasks with 100% accuracy | 5/10 | 0/10 |
| Tasks with 0% accuracy | 3/10 | 8/10 |
| Accuracy variance across runs | 50-70% | 0-10% |
| Accuracy Coefficient of Variation | 14.9% | 173.2% |

### 6.5 Key Insights

1. **Claude is 14.5x more accurate** (58% vs 4%)
2. **Claude solves 5 tasks perfectly** (100% success across all runs)
3. **Both models fail on some tasks consistently** (astropy-13398, astropy-13977)
4. **Llama's accuracy is highly inconsistent** - most runs resolve nothing
5. **Submission ≠ Correctness** - Llama submits 100% but only 4% are correct

---

## 7. Key Takeaways

1. **Claude is significantly more consistent** in its approach to solving tasks
2. **Llama is faster** but with higher behavioral variance
3. **Both models produce unique sequences** every run - no deterministic paths
4. **The difference is statistically significant** with a large effect size
5. **Runs diverge early** - typically within the first few steps

---

*Report generated automatically for Paper 2 analysis*
