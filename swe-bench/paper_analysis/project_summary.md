# Project Summary: Behavioral Consistency in Code Agents

## Goal
Measure and analyze **behavioral consistency** of LLM-based code agents on SWE-bench tasks. Key question: *Do agents take the same actions when given the same task multiple times?*

## Experiment Setup
- **Models**: Claude Sonnet 4.5, Llama-3.1-70B
- **Tasks**: 10 astropy issues from SWE-bench
- **Runs**: 5 runs per task per model (100 total runs)
- **Agent**: mini-swe-agent with Docker environment

## Key Findings

| Metric | Claude | Llama |
|--------|--------|-------|
| Accuracy | **58%** (29/50) | **4%** (2/50) |
| Mean Steps | 46.1 | 17.0 |
| Mean CV (consistency) | **13.6%** | 42.0% |
| First Action | 68% `find` | 54% `ls` |

**Claude is 3x more consistent and 14x more accurate than Llama.**

## Analysis Completed
1. ✅ Step count statistics (mean, std, CV per task)
2. ✅ Accuracy via SWE-bench evaluation harness
3. ✅ First action analysis (Chi-square tests)
4. ✅ Divergence analysis (at which step do runs diverge)
5. ✅ Failure mode categorization
6. ✅ Case studies (Llama successes, Claude failures on astropy-13236)
7. ✅ Sanity check (verified all claims against raw JSON)

## Key Files
- Raw data: `results_claude_10/`, `results_llama_10/`
- Analysis: `paper_analysis/` (figures, tables, reports)
- GitHub: https://github.com/amanmehta-maniac/agent-consistency

## Paper-Ready Claims
1. First action predicts model, not success
2. Divergence happens at Step 3 (Claude) vs Step 1 (Llama)
3. Claude's consistency correlates with higher accuracy
4. Llama's 2 successes were "lucky" (minimal exploration, direct fix)
