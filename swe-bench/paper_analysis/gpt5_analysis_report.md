# GPT-5 (Snowflake Cortex) Analysis Report
## Behavioral Consistency in Code Agents

---

## Executive Summary

**Model**: GPT-5 (Snowflake Cortex)
**Tasks**: 10 astropy tasks from SWE-bench Verified
**Runs per task**: 5
**Total runs**: 50
**Temperature**: 0.5

### Key Metrics

| Metric | Value |
|--------|-------|
| Mean CV (Consistency) | 32.2% |
| Mean Steps | 9.9 |
| Mean Accuracy | 0.0% |
| Unique Sequence Rate | 100% |

---

## 1. Basic Metrics

### Step Count Statistics
| Statistic | Value |
|-----------|-------|
| Mean | 9.92 |
| Median | 8.00 |
| Std Dev | 5.31 |
| Min | 4 |
| Max | 29 |

### Coefficient of Variation (CV)
| Statistic | Value |
|-----------|-------|
| Mean | 32.19% |
| Median | 32.91% |
| Std Dev | 20.03% |
| 95% CI | (17.86%, 46.52%) |

---

## 2. Per-Task Breakdown (Appendix Table 8)

| Task ID | Steps (mean±std) | CV (%) | Accuracy | Unique Seqs |
|---------|------------------|--------|----------|-------------|
| astropy-12907 | 11.0±3.4 | 30.8 | 0% | 5/5 |
| astropy-13033 | 9.2±6.1 | 66.7 | 0% | 5/5 |
| astropy-13236 | 7.0±2.4 | 35.0 | 0% | 5/5 |
| astropy-13398 | 9.4±1.9 | 20.7 | 0% | 5/5 |
| astropy-13453 | 16.6±6.5 | 39.2 | 0% | 5/5 |
| astropy-13579 | 8.4±1.1 | 13.6 | 0% | 5/5 |
| astropy-13977 | 7.0±0.0 | 0.0 | 0% | 5/5 |
| astropy-14096 | 16.2±9.0 | 55.6 | 0% | 5/5 |
| astropy-14182 | 7.2±1.3 | 18.1 | 0% | 5/5 |
| astropy-14309 | 7.2±3.0 | 42.1 | 0% | 5/5 |

---

## 3. Phase Decomposition (Table 4)

Total actions analyzed: 468

| Phase | Count | Percentage |
|-------|-------|------------|
| EXPLORE | 112 | 23.9% |
| UNDERSTAND | 5 | 1.1% |
| EDIT | 69 | 14.7% |
| VERIFY | 151 | 32.3% |
| OTHER | 131 | 28.0% |

### Action Type Distribution
- **python**: 150 (32.1%)
- **other**: 131 (28.0%)
- **sed/edit**: 69 (14.7%)
- **ls**: 60 (12.8%)
- **grep**: 51 (10.9%)
- **cat/read**: 5 (1.1%)
- **pytest**: 1 (0.2%)
- **find**: 1 (0.2%)

---

## 4. Failure Mode Classification (Table 5)

Total failures: 50

| Failure Mode | Count | Percentage |
|--------------|-------|------------|
| WRONG_FIX | 29 | 58.0% |
| EMPTY_PATCH | 21 | 42.0% |
| LOOP_DEATH | 0 | 0.0% |

---

## 5. Divergence Analysis (Table 6)

### Divergence Point (step where runs start differing)
| Statistic | Value |
|-----------|-------|
| Mean | 3.10 |
| Median | 2.50 |

### Per-Task Divergence Points
| Task ID | Divergence Step |
|---------|-----------------|
| astropy-12907 | 6 |
| astropy-13033 | 2 |
| astropy-13236 | 4 |
| astropy-13398 | 2 |
| astropy-13453 | 3 |
| astropy-13579 | 2 |
| astropy-13977 | 5 |
| astropy-14096 | 2 |
| astropy-14182 | 3 |
| astropy-14309 | 2 |

### Sequence Similarity (Jaccard)
| Statistic | Value |
|-----------|-------|
| Mean | 0.052 |
| Std Dev | 0.032 |

---

## 6. First Action Analysis (Table 7)

Total first actions: 50

| Action Type | Count | Percentage |
|-------------|-------|------------|
| ls | 50 | 100.0% |

---

## 7. Statistical Comparisons (Appendix C)

### Cross-Model CV Comparison
| Model | Mean CV |
|-------|---------|
| Claude 4.5 Sonnet | 15.2% |
| Llama-3.1-70B | 47.0% |
| **GPT-5 (Snowflake)** | **32.2%** |

### GPT-5 vs Claude
| Test | Statistic | p-value | Significant (α=0.05) |
|------|-----------|---------|----------------------|
| Independent t-test | t=2.592 | 0.0184 | Yes |
| Mann-Whitney U | U=81.0 | 0.0211 | Yes |

- **Cohen's d**: 1.159
- **CV Ratio (GPT-5/Claude)**: 2.12x

### GPT-5 vs Llama
| Test | Statistic | p-value | Significant (α=0.05) |
|------|-----------|---------|----------------------|
| Independent t-test | t=-1.921 | 0.0708 | No |
| Mann-Whitney U | U=27.0 | 0.0890 | No |

- **Cohen's d**: -0.859
- **CV Ratio (GPT-5/Llama)**: 0.68x

---

## 8. Key Findings

1. **GPT-5 is LESS consistent than Claude**: CV 32.2% vs 15.2%
2. **GPT-5 is MORE consistent than Llama**: CV 32.2% vs 47.0%
3. **Accuracy**: 0.0% of runs produced correct fixes
4. **Divergence**: Runs diverge on average at step 3.1
5. **All runs produce unique sequences**: 100% of tasks have 5/5 unique sequences

---

*Analysis generated: 2026-02-13 09:47:35*
*Model: openai-gpt-5 via Snowflake Cortex*
*Provider: Snowflake Account AKB73862*
