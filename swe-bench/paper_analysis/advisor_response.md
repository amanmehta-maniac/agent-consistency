# Paper 2: Qualitative Analysis Complete

Hi [Advisor],

I've completed the full qualitative analysis you requested. All files are pushed to GitHub. Here's a summary:

---

## ✅ RQ1: Does Consistency Predict Accuracy?

**Finding**: Within-model, CV doesn't predict accuracy (p > 0.05). But **cross-model, the more consistent model wins 88% of tasks** (7/8).

| Deliverable | GitHub URL |
|-------------|------------|
| CV-Accuracy Table | [table4_cv_accuracy_correlation.tex](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/tables/table4_cv_accuracy_correlation.tex) |
| Correlation Data | [correlation_data.json](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/correlation_data.json) |

**Key stats**: Claude r = -0.10 (p = 0.78), Llama r = 0.30 (p = 0.40) — no within-model correlation, but Claude (lower CV) beats Llama on 7/8 tasks.

---

## ✅ RQ2: Where Does Variance Originate?

**Finding**: Variance originates in **EXPLORE and UNDERSTAND phases**, not EDIT. Claude spends 10× more time exploring.

| Deliverable | GitHub URL |
|-------------|------------|
| Phase Decomposition Figure | [fig8_phase_decomposition.png](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/figures/fig8_phase_decomposition.png) |
| Phase Variance Table | [table5_phase_variance.tex](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/tables/table5_phase_variance.tex) |
| Variance Source Analysis | [variance_source_analysis.md](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/qualitative/variance_source_analysis.md) |

**Key stats**:
- EXPLORE: Claude 34% (15 steps) vs Llama 9% (1.5 steps) — **10× difference**
- UNDERSTAND: Claude 30% (14 steps) vs Llama 18% (3 steps) — **4.7× difference**
- EXPLORE CV: Claude 42% vs Llama 123% — Llama's exploration is erratic

---

## ✅ RQ3: What Are the Failure Modes?

**Finding**: Claude fails due to **wrong logic** (100% WRONG_FIX). Llama fails due to **insufficient effort** (21% EMPTY_PATCH).

| Deliverable | GitHub URL |
|-------------|------------|
| Failure Modes Figure | [fig10_failure_modes.png](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/figures/fig10_failure_modes.png) |
| Failure Taxonomy Table | [table6_failure_taxonomy.tex](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/tables/table6_failure_taxonomy.tex) |
| Failure Analysis | [failure_analysis.md](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/qualitative/failure_analysis.md) |

**Key stats**:

| Mode | Claude | Llama |
|------|--------|-------|
| WRONG_FIX | 21 (100%) | 37 (77%) |
| EMPTY_PATCH | 0 (0%) | 10 (21%) |
| LOOP_DEATH | 0 (0%) | 1 (2%) |

---

## ✅ Case Studies

| Case Study | GitHub URL | Description |
|------------|------------|-------------|
| Case 1 | [case1_claude_success_vs_failure.md](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/case_studies/case1_claude_success_vs_failure.md) | Same task, Claude succeeds vs fails — difference is understanding time |
| Case 2 | [case2_llama_death_spiral.md](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/case_studies/case2_llama_death_spiral.md) | Llama stuck in edit-test-edit loop (55 steps) |
| Case 3 | [case3_llama_lucky_success.md](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/case_studies/case3_llama_lucky_success.md) | One of Llama's 2 successes — lucky file find |
| Case 4 | [case4_both_succeed_different_paths.md](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/case_studies/case4_both_succeed_different_paths.md) | Same task, both succeed — Claude 45 steps, Llama 8 steps |

---

## ✅ Paper Claims (Verified)

**Full document**: [paper_claims.md](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/paper_claims.md)

| Claim | Evidence |
|-------|----------|
| **CLAIM 1**: Claude is 3.1× more consistent | CV 15.2% vs 47.0%, t = -6.75, p < 0.0001, Cohen's d = -3.02 |
| **CLAIM 2**: Lower CV model wins 88% of tasks | Cross-model comparison (7/8 tasks) |
| **CLAIM 3**: Variance from EXPLORE/UNDERSTAND | Phase decomposition: Claude 64% vs Llama 27% |
| **CLAIM 4**: Different failure modes | Claude: quality failures, Llama: effort failures |
| **CLAIM 5**: Llama speed from skipping exploration | 24.5 steps saved, but 96% failure rate |
| **CLAIM 6**: Claude is 14.5× more accurate | 58% vs 4% (SWE-bench evaluated) |

---

## 📊 Summary Table (LaTeX)

[table7_summary.tex](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/tables/table7_summary.tex)

---

## 📁 Complete Directory Structure

```
swe-bench/paper_analysis/
├── figures/
│   ├── fig1_cv_distribution.png      # CV comparison box plot
│   ├── fig2_step_heatmap.png         # Steps per task/run heatmap
│   ├── fig3_step_distributions.png   # Step count histograms
│   ├── fig4_tradeoff_scatter.png     # CV vs mean steps
│   ├── fig5_per_task_bars.png        # Per-task step comparison
│   ├── fig6_accuracy.png             # Accuracy comparison
│   ├── fig7_consistency_accuracy.png # CV vs accuracy scatter
│   ├── fig8_phase_decomposition.png  # Phase allocation (NEW)
│   └── fig10_failure_modes.png       # Failure taxonomy (NEW)
├── tables/
│   ├── table1_overall.tex            # Overall results
│   ├── table2_per_task.tex           # Per-task breakdown
│   ├── table3_statistics.tex         # Statistical tests
│   ├── table4_cv_accuracy_correlation.tex  # RQ1 (NEW)
│   ├── table5_phase_variance.tex     # RQ2 (NEW)
│   ├── table6_failure_taxonomy.tex   # RQ3 (NEW)
│   └── table7_summary.tex            # Final summary (NEW)
├── qualitative/
│   ├── failure_analysis.md           # RQ3 narrative
│   └── variance_source_analysis.md   # RQ2 narrative
├── case_studies/
│   ├── case1-4 (annotated examples)
├── paper_claims.md                   # Verified claims
├── analysis_report.md                # Full quantitative report
└── raw_metrics.json                  # All computed metrics
```

---

## Raw Data (for verification)

| Data | GitHub URL |
|------|------------|
| Claude results (10×5) | [results_claude_10/](https://github.com/amanmehta-maniac/agent-consistency/tree/main/swe-bench/results_claude_10) |
| Llama results (10×5) | [results_llama_10/](https://github.com/amanmehta-maniac/agent-consistency/tree/main/swe-bench/results_llama_10) |
| SWE-bench eval reports | [claude.claude-local-run1.json](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/claude.claude-local-run1.json) (×5 runs each model) |

---

## All Figures (Direct Links)

| Figure | Description | Link |
|--------|-------------|------|
| Fig 1 | CV Distribution | [fig1_cv_distribution.png](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/figures/fig1_cv_distribution.png) |
| Fig 2 | Step Heatmap | [fig2_step_heatmap.png](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/figures/fig2_step_heatmap.png) |
| Fig 3 | Step Distributions | [fig3_step_distributions.png](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/figures/fig3_step_distributions.png) |
| Fig 4 | Tradeoff Scatter | [fig4_tradeoff_scatter.png](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/figures/fig4_tradeoff_scatter.png) |
| Fig 5 | Per-Task Bars | [fig5_per_task_bars.png](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/figures/fig5_per_task_bars.png) |
| Fig 6 | Accuracy | [fig6_accuracy.png](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/figures/fig6_accuracy.png) |
| Fig 7 | Consistency vs Accuracy | [fig7_consistency_accuracy.png](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/figures/fig7_consistency_accuracy.png) |
| Fig 8 | Phase Decomposition | [fig8_phase_decomposition.png](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/figures/fig8_phase_decomposition.png) |
| Fig 10 | Failure Modes | [fig10_failure_modes.png](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/figures/fig10_failure_modes.png) |

---

## All Tables (Direct Links)

| Table | Description | Link |
|-------|-------------|------|
| Table 1 | Overall Results | [table1_overall.tex](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/tables/table1_overall.tex) |
| Table 2 | Per-Task Results | [table2_per_task.tex](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/tables/table2_per_task.tex) |
| Table 3 | Statistical Tests | [table3_statistics.tex](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/tables/table3_statistics.tex) |
| Table 4 | CV-Accuracy Correlation | [table4_cv_accuracy_correlation.tex](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/tables/table4_cv_accuracy_correlation.tex) |
| Table 5 | Phase Variance | [table5_phase_variance.tex](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/tables/table5_phase_variance.tex) |
| Table 6 | Failure Taxonomy | [table6_failure_taxonomy.tex](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/tables/table6_failure_taxonomy.tex) |
| Table 7 | Summary | [table7_summary.tex](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/tables/table7_summary.tex) |

---

Let me know if you'd like me to expand any section or run additional analysis!
