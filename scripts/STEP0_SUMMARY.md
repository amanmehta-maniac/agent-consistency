# Paper 3 — STEP 0: Baseline Sanity Check

**Date**: 2026-02-24  
**Purpose**: Verify Paper 2 numbers from raw data before implementing any intervention.

---

## 1. Infrastructure Located

| Component | Path |
|-----------|------|
| Experiment runner | `swe-bench/runner.py` |
| Agent (TrackedAgent) | `swe-bench/runner.py` (wraps mini-swe-agent `DefaultAgent`) |
| Agent framework | `swe-bench/mini-swe-agent/` |
| Claude results | `swe-bench/results_claude_10/` (10 task JSONs, 5 runs each) |
| GPT-5 results | `swe-bench/results_gpt5_snowflake/` (10 task JSONs, 5 runs each) |
| Llama results | `swe-bench/results_llama_10/` (10 task JSONs, 5 runs each) |
| SWE-bench eval reports | `swe-bench/{model}.{model}-local-run{1-5}.json` |
| Summary script (new) | `scripts/summarize_runs.py` |

---

## 2. Cross-Model Comparison (from raw data)

| Metric | Claude | GPT-5 | Llama |
|--------|--------|-------|-------|
| **Accuracy** | **29/50 (58%)** | 16/50 (32%) | 2/50 (4%) |
| **Mean CV** | **15.2%** | 32.2% | 47.0% |
| **Mean Steps** | 46.1 | **9.9** | 17.0 |
| **Median Steps** | 45.5 | 8.0 | 14.5 |
| **p90 Steps** | 56.1 | 17.3 | 29.1 |
| **p95 Steps** | 58.5 | 20.5 | 33.2 |
| **Valid Patches** | 50/50 | 48/50 | 40/50 |
| **Cost/Run** | $2.78 | $0.53 | $0.11 |

---

## 3. Per-Task Accuracy (from SWE-bench evaluation reports)

| Task | Claude | GPT-5 | Llama |
|------|--------|-------|-------|
| 12907 | 5/5 | 2/5 | 0/5 |
| 13033 | 3/5 | 0/5 | 0/5 |
| 13236 | 0/5 | 0/5 | 1/5 |
| 13398 | 0/5 | 0/5 | 0/5 |
| 13453 | 5/5 | 3/5 | 0/5 |
| 13579 | 5/5 | 3/5 | 0/5 |
| 13977 | 0/5 | 0/5 | 0/5 |
| 14096 | 5/5 | 3/5 | 0/5 |
| 14182 | 1/5 | 0/5 | 0/5 |
| 14309 | 5/5 | 5/5 | 1/5 |
| **Total** | **29/50** | **16/50** | **2/50** |

---

## 4. Per-Task Steps & CV (from raw data)

### Claude
| Task | Mean Steps | CV (%) |
|------|-----------|--------|
| 12907 | 38.2 | 20.7 |
| 13033 | 42.6 | 14.2 |
| 13236 | 40.6 | 27.0 |
| 13398 | 47.8 | 8.8 |
| 13453 | 41.2 | 14.3 |
| 13579 | 52.8 | 10.5 |
| 13977 | 49.0 | 14.6 |
| 14096 | 47.0 | 17.3 |
| 14182 | 53.2 | 12.4 |
| 14309 | 48.4 | 12.0 |
| **Mean** | **46.1** | **15.2** |

### GPT-5
| Task | Mean Steps | CV (%) |
|------|-----------|--------|
| 12907 | 11.0 | 30.8 |
| 13033 | 9.2 | 66.7 |
| 13236 | 7.0 | 35.0 |
| 13398 | 9.4 | 20.7 |
| 13453 | 16.6 | 39.2 |
| 13579 | 8.4 | 13.6 |
| 13977 | 7.0 | 0.0 |
| 14096 | 16.2 | 55.6 |
| 14182 | 7.2 | 18.1 |
| 14309 | 7.2 | 42.1 |
| **Mean** | **9.9** | **32.2** |

### Llama
| Task | Mean Steps | CV (%) |
|------|-----------|--------|
| 12907 | 16.6 | 66.6 |
| 13033 | 22.4 | 44.4 |
| 13236 | 13.2 | 38.8 |
| 13398 | 15.4 | 47.0 |
| 13453 | 11.0 | 28.7 |
| 13579 | 14.4 | 47.5 |
| 13977 | 13.2 | 37.7 |
| 14096 | 22.2 | 31.1 |
| 14182 | 26.8 | 61.1 |
| 14309 | 14.6 | 67.2 |
| **Mean** | **17.0** | **47.0** |

---

## 5. Discrepancies vs. Paper 2

### 5a. Per-task step counts & CVs in Paper 2 Table 9

**Affected models**: Claude and Llama (GPT-5 is correct)

Paper 2's Table 9 has per-task step values that **do not match the raw data**. Example:

| Task | Claude (raw data) | Claude (Paper Table 9) |
|------|------------------|----------------------|
| 12907 | 38.2 steps, CV=20.7% | 45 steps, CV=18.5% |
| 13033 | 42.6 steps, CV=14.2% | 48 steps, CV=12.7% |
| 13579 | 52.8 steps, CV=10.5% | 49 steps, CV=9.4% |

The auto-generated `paper_analysis/tables/table2_per_task.tex` confirms the raw data is correct (38.2, 42.6, etc.). The paper's Table 9 values appear to have been manually entered from an earlier draft or different computation.

**Aggregate impact**:
- Overall mean steps: both give **46.1** (correct by coincidence of rounding)
- Mean CV: raw data = **15.2%**, paper = **13.6%** → paper underreports Claude's variance
- Llama mean CV: raw data = **47.0%**, paper = **42.0%** → paper underreports Llama's variance

**Action needed**: Correct Table 9 in Paper 2 before arXiv submission.

### 5b. GPT-5 valid patches: 48/50 ✅ Confirmed

Two GPT-5 runs (task 13579, runs 2 and 3) submitted empty patches (`final_output` is empty string). The agent framework records `exit_status: "Submitted"` and `success: True`, but the patches contain no diff. Both runs failed SWE-bench evaluation. Paper's claim of "48/50 valid patches" is **correct**.

### 5c. Llama valid patches: paper says 40/50

Llama has **10 empty patches** across all 50 runs. Paper reports 40/50 valid patches (21% empty). Confirmed from raw data: **correct**.

---

## 6. Verified Claims (all confirmed ✅)

| Claim | Paper 2 | Raw Data | Status |
|-------|---------|----------|--------|
| Claude accuracy | 58% (29/50) | 29/50 | ✅ |
| GPT-5 accuracy | 32% (16/50) | 16/50 | ✅ |
| Llama accuracy | 4% (2/50) | 2/50 | ✅ |
| Claude mean steps | 46.1 | 46.1 | ✅ |
| GPT-5 mean steps | 9.9 | 9.9 | ✅ |
| Llama mean steps | 17.0 | 17.0 | ✅ |
| GPT-5 valid patches | 48/50 | 48/50 | ✅ |
| Llama valid patches | 40/50 | 40/50 | ✅ |
| Claude mean CV | 13.6% | **15.2%** | ⚠️ Table 9 error |
| Llama mean CV | 42.0% | **47.0%** | ⚠️ Table 9 error |

---

## 7. Recommendation

**Baseline is solid for Paper 3.** Accuracy numbers, step means, and patch validity are all correct. The per-task breakdown in Table 9 needs correction, but this doesn't affect the experimental setup for Paper 3.

**Ready to proceed to STEP 1** (Interpretation Guard implementation) using:
- `swe-bench/runner.py` as the base
- `results_claude_10/` and `results_gpt5_snowflake/` as baselines
- `scripts/summarize_runs.py` for automated comparison

---

## 8. Reproducing This Summary

```bash
cd /path/to/agent-consistency
python3 scripts/summarize_runs.py
```
