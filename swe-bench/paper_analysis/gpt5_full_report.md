# GPT-5 Full Analysis Report
## Matching Claude/Llama Paper Pipeline

---

## GPT-5 Results Summary

```
Accuracy:       16/50 (32%)
Mean CV:        32.2%
Mean Steps:     9.9 (SD: 5.3)
Valid Patches:  48/50
Cost/Run:       $0.53
```

---

## 3-Model Comparison (Table 1)

| Metric | Claude 4.5 | GPT-5 | Llama 70B |
|--------|------------|-------|-----------|
| **Accuracy** | **29/50 (58%)** | **16/50 (32%)** | **2/50 (4%)** |
| Mean Steps | 46.1 | **9.9** | 17.0 |
| Mean CV | **13.6%** | 32.2% | 42.0% |
| Median CV | 12.1% | 32.9% | 38.2% |
| Div. Step | 3.2 | 3.4 | 1.4 |
| First Action | 68% find | 100% ls | 54% ls |
| Cost/Run | ~$1.50 | $0.53 | ~$0.10 |

---

## Per-Task Breakdown (Appendix Table 8)

| Task | Claude Steps | Claude CV | Claude Acc | GPT-5 Steps | GPT-5 CV | GPT-5 Acc | Llama Steps | Llama CV | Llama Acc |
|------|-------------|-----------|------------|-------------|----------|-----------|-------------|----------|-----------|
| 12907 | 45±8 | 18.5% | 5/5 | 11±3 | 30.8% | 2/5 | 18±11 | 59.6% | 0/5 |
| 13033 | 48±6 | 12.7% | 3/5 | 9±6 | 66.7% | 0/5 | 16±6 | 39.7% | 0/5 |
| 13236 | 41±10 | 24.2% | 0/5 | 7±2 | 35.0% | 0/5 | 15±5 | 34.7% | 1/5 |
| 13398 | 49±4 | 7.9% | 0/5 | 9±2 | 20.7% | 0/5 | 18±8 | 42.0% | 0/5 |
| 13453 | 45±6 | 12.8% | 5/5 | 17±7 | 39.2% | 3/5 | 16±4 | 25.7% | 0/5 |
| 13579 | 49±5 | 9.4% | 5/5 | 8±1 | 13.6% | 3/5 | 18±8 | 42.5% | 0/5 |
| 13977 | 46±6 | 13.1% | 0/5 | 7±0 | 0.0% | 0/5 | 18±6 | 33.7% | 0/5 |
| 14096 | 41±6 | 15.5% | 5/5 | 16±9 | 55.6% | 3/5 | 14±4 | 27.8% | 0/5 |
| 14182 | 48±5 | 11.1% | 1/5 | 7±1 | 18.1% | 0/5 | 24±13 | 54.6% | 0/5 |
| 14309 | 49±5 | 10.7% | 5/5 | 7±3 | 42.1% | 5/5 | 13±8 | 60.1% | 1/5 |

---

## Phase Decomposition (Table 4)

| Phase | Claude | GPT-5 | Llama |
|-------|--------|-------|-------|
| EXPLORE | 17.8% | 13.0% | 28.1% |
| UNDERSTAND | 41.2% | 31.4% | 30.5% |
| EDIT | 14.5% | 12.0% | 11.2% |
| VERIFY | 19.3% | 32.3% | 18.9% |
| OTHER | 7.2% | 11.3% | 11.3% |

**GPT-5 spends the most time on VERIFY (32.3%)** — heavy use of `python` to test.

### GPT-5 Action Distribution
| Command | Count | % |
|---------|-------|---|
| python | 149 | 31.8% |
| nl | 91 | 19.4% |
| ls | 60 | 12.8% |
| sed | 56 | 12.0% |
| grep | 51 | 10.9% |
| applypatch | 38 | 8.1% |

---

## Failure Mode Classification (Table 5)

| Failure Mode | Claude (21 fails) | GPT-5 (34 fails) | Llama (48 fails) |
|--------------|-------------------|-------------------|-------------------|
| WRONG_FIX | 21 (100%) | 32 (94%) | 38 (79%) |
| EMPTY_PATCH | 0 (0%) | 2 (6%) | 10 (21%) |
| LOOP_DEATH | 0 (0%) | 0 (0%) | 0 (0%) |

**All three models primarily fail by submitting wrong fixes, not by getting stuck.**

---

## Divergence Analysis (Table 6)

| Task | Claude Div. | GPT-5 Div. | Llama Div. |
|------|-------------|------------|------------|
| 12907 | 5 | 5 | 1 |
| 13033 | 3 | 3 | 2 |
| 13236 | 3 | 4 | 1 |
| 13398 | 8 | 2 | 2 |
| 13453 | 2 | 3 | 1 |
| 13579 | 2 | 4 | 1 |
| 13977 | 2 | 4 | 2 |
| 14096 | 2 | 4 | 1 |
| 14182 | 2 | 3 | 2 |
| 14309 | 3 | 2 | 1 |

| Statistic | Claude | GPT-5 | Llama |
|-----------|--------|-------|-------|
| Mean Div. Step | 3.2 | 3.4 | 1.4 |
| Median | 3.0 | 3.5 | 1.0 |
| % at Step 1 | 0% | 0% | 60% |
| % by Step 5 | 90% | 100% | 100% |

**GPT-5 and Claude diverge at similar rates (~step 3), while Llama diverges immediately (60% at step 1).**

---

## First Action Analysis (Table 7)

| First Action | Claude | GPT-5 | Llama |
|--------------|--------|-------|-------|
| find | 34 (68%) | 0 (0%) | 10 (20%) |
| ls | 13 (26%) | **50 (100%)** | 27 (54%) |
| grep | 1 (2%) | 0 (0%) | 12 (24%) |
| cat | 2 (4%) | 0 (0%) | 1 (2%) |

**GPT-5 always starts with `ls` (100%) — the most predictable first action of any model.**

| First Action × Success | Claude | GPT-5 | Llama |
|------------------------|--------|-------|-------|
| ls → Success | 85% (11/13) | 32% (16/50) | 4% (1/27) |
| find → Success | 50% (17/34) | N/A | 0% (0/10) |

---

## Statistical Tests (Appendix C)

### CV Comparisons
| Comparison | t-stat | p-value | Cohen's d | 95% CI (diff) | Significant |
|------------|--------|---------|-----------|---------------|-------------|
| GPT-5 vs Claude | 2.592 | **0.018** | 1.159 | [4.1, 29.8] | **Yes** |
| GPT-5 vs Llama | -1.921 | 0.071 | -0.859 | [-29.9, 0.3] | No |
| Claude vs Llama | -6.754 | **<0.001** | -3.020 | [-41.0, -22.6] | **Yes** |

### Accuracy Comparisons (Fisher's Exact)
| Comparison | Odds Ratio | p-value | Significant |
|------------|------------|---------|-------------|
| GPT-5 vs Claude | 0.341 | **0.015** | **Yes** |
| GPT-5 vs Llama | 11.294 | **<0.001** | **Yes** |
| Claude vs Llama | 33.143 | **<0.001** | **Yes** |

### Divergence Comparisons
| Comparison | t-stat | p-value | Significant |
|------------|--------|---------|-------------|
| GPT-5 vs Claude | 0.254 | 0.803 | No |
| GPT-5 vs Llama | 5.774 | **<0.001** | **Yes** |

### Summary of Significance
```
Consistency (CV):  Claude << GPT-5 ≈ Llama
Accuracy:          Claude >> GPT-5 >> Llama  (all significant)
Divergence:        Claude ≈ GPT-5 >> Llama
```

---

## Key Findings

### 1. GPT-5 is the "Fast Middle Ground"
- **4.7x faster than Claude** (9.9 vs 46.1 steps)
- **Mid-range accuracy** (32% vs Claude 58%, Llama 4%)
- **Mid-range consistency** (CV 32.2% vs Claude 13.6%, Llama 42.0%)

### 2. Consistency Correlates with Accuracy
```
Claude:  CV=13.6%  →  Accuracy=58%
GPT-5:   CV=32.2%  →  Accuracy=32%
Llama:   CV=42.0%  →  Accuracy=4%
```
Spearman correlation: r = -1.0 (perfect negative — lower CV = higher accuracy)

### 3. GPT-5's Unique Behavior Pattern
- **100% `ls` first action** — most predictable opening
- **32% VERIFY phase** — tests aggressively with `python`
- **Only 13% EXPLORE** — less exploration than Claude (18%) or Llama (28%)
- Uses `nl` (numbered line output) heavily (19.4%) — unique to GPT-5

### 4. Divergence Pattern
- GPT-5 and Claude share similar divergence timing (~step 3)
- Both are significantly later than Llama (step 1.4)
- Despite similar divergence, Claude is 2x more consistent overall

### 5. All Models Produce Unique Sequences
- 100% of tasks have 5/5 unique sequences for all 3 models
- Even GPT-5's perfectly consistent first action doesn't prevent divergence

---

## Updated Paper Claims

1. ✅ **"Consistency predicts accuracy across models"** — Perfect rank correlation (Claude > GPT-5 > Llama)
2. ✅ **"Claude is 2.4x more consistent than GPT-5"** (CV: 13.6% vs 32.2%, p=0.018)
3. ✅ **"GPT-5 is significantly more accurate than Llama"** (32% vs 4%, p<0.001)
4. ✅ **"Speed doesn't equal accuracy"** — GPT-5 is fastest but not most accurate
5. ✅ **"Early strategy agreement doesn't guarantee consistency"** — GPT-5 diverges at similar step as Claude but has 2x higher CV

---

*Analysis verified via SWE-bench evaluation harness (Docker-based test execution)*
*GPT-5 model: openai-gpt-5 via Snowflake Cortex*
*All 5 runs × 10 tasks evaluated independently*
