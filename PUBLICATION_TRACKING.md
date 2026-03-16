# Publication Tracking -- Agent Consistency Research

**Last updated**: March 15, 2026

---

## Paper 1: "When Agents Disagree With Themselves: Measuring Behavioral Consistency in LLM-Based Agents"
**Status**: Published on arXiv (2602.11619)
**Current scope**: Observational study of multi-run behavioral consistency on HotpotQA (GPT-4o, Claude Sonnet 4.5, Llama 3.1 70B). 100 questions x 10 runs x 3 models = 3,000 runs.
**Source file**: `paper1/paper1.tex`
**Target venue**: ICML 2026 Workshop (agent reliability / benchmarking)
**Deadline**: ~June 2026 (workshop deadlines TBD)
**Goal**: 90%+ acceptance candidate

### Expansion Plan

**Why expand**: The paper currently "documents a phenomenon" -- valuable but workshop reviewers want at least one additional contribution beyond observation. Two targeted additions lift it from observation to actionable insight.

#### A. Add SWE-bench data (cross-benchmark validation)
- **What**: Add a new section showing consistency metrics on SWE-bench (10 tasks x 5 runs x 3 models: Claude 4.5 Sonnet, GPT-5, Llama 70B).
- **Data source**: Already collected -- `swe-bench/analysis_results.md`, `paper2_overleaf/main.tex`.
- **Key numbers**: Claude CV 15.2%/acc 58%, GPT-5 CV 32.2%/acc 32%, Llama CV 47.0%/acc 4%.
- **Work required**: ~1 day. Add summary table + 1 figure.

#### B. Temperature ablation (already partially in paper)
- Already in Section 4.5 of paper1.tex. Enhancement: expand if time permits.
- **Priority**: LOW.

#### C. "Rankings change" analysis (unique contribution)
- **What**: Bootstrap resample per-question accuracy, show model rankings change in X% of samples.
- **Data source**: `hotpotqa/results_claude/`, `hotpotqa/results_gpt4o/`, `hotpotqa/results_llama/`.
- **Work required**: ~1 day.

### Key narrative
> "We show this inconsistency is benchmark-general (HotpotQA + SWE-bench), model-general (5 models), and practically consequential (model rankings change across runs)."

---

## Paper 2: "Consistency Amplifies: How Behavioral Variance Shapes Agent Accuracy"
**Status**: Draft complete
**Source file**: `paper2_overleaf/main.tex`
**Current scope**: 3-model comparison on SWE-bench. 10 tasks x 5 runs x 3 models.
**Target venue**: COLM 2026 main track (preferred) or NeurIPS 2026
**Deadline**: Flexible -- quality over speed. March 31 (COLM) or May 2026 (NeurIPS).
**Goal**: 95%+ acceptance candidate

### Expansion Plan (E1-E6)
- **E1**: Scale 10 -> 50-100 tasks, multiple repos (MUST DO)
- **E2**: Add 1-2 more models (HIGH)
- **E3**: Within-model consistency-accuracy analysis at n=50+ (HIGH)
- **E4**: Cross-repository breakdown (MEDIUM)
- **E5**: Interpretation guard intervention (MEDIUM-HIGH)
- **E6**: Cost-accuracy-consistency Pareto frontier (LOW)

### Delineation from Paper 1
- Paper 1 = "what" (phenomenon exists, rankings unstable)
- Paper 2 = "why and so what" (amplification mechanism, interpretation bottleneck, intervention)

---

## Paper 3: "Representational Commitment: Hidden States Predict Agent Consistency"
**Status**: Draft complete → upgrading to COLM main track
**Source file**: `paper3_prep/main.tex`
**Target venue**: COLM 2026 main track
**Deadline**: March 31, 2026
**Goal**: 75-80% acceptance probability (strong accept)
**Improvements plan**: `paper3_improvements.md`

### Key numbers (current)
- Llama: r=-0.35 at L40, partial r=-0.45, CI [-0.59, -0.29]
- Qwen (n=100): r=-0.65 at L64, CI [-0.74, -0.55]
- Quartile effect: 4.1x, Cohen's d=1.01

### Expansion Plan (10 tasks, see paper3_improvements.md)
- **Task 1**: Hard question theoretical resolution (MUST, no compute)
- **Task 2**: Add Phi-3-Medium-14B as 3rd model (SHOULD, GPU)
- **Task 3**: Prompting intervention at step 3 (MUST, GPU)
- **Task 4**: Activation steering at layer 40 (STRETCH, GPU)
- **Task 5**: StrategyQA cross-benchmark (SHOULD, GPU)
- **Task 6**: Runtime monitor evaluation (MUST, no compute)
- **Task 7**: Anonymization and self-citations (MUST, writing)
- **Task 8**: Overclaiming language fixes (MUST, writing)
- **Task 9**: Final assembly, figures, page budget (MUST, writing)
- **Task 10**: COLM formatting and submission prep (MUST, writing)

### Delineation from Paper 1 and Paper 2
- Paper 1 = "what" (phenomenon exists, rankings unstable)
- Paper 2 = "why and so what" (amplification mechanism, intervention)
- Paper 3 = "where inside the model" (hidden state geometry, mechanistic)

---

## Paper 4: "Do LLM Agents Follow Their Own Plans?"
**Status**: Draft complete
**Source file**: `paper4_workshop/main.tex`
**Target venue**: ICML/COLM Workshop
**Assessment**: ~80% acceptance

---

## Paper 5 (Planned): Extended Paper 3 -> NeurIPS 2026 Main Track
**Status**: Superseded — Paper 3 is now targeting COLM main track directly
**Note**: The expansion planned for Paper 5 (cross-benchmark, 3rd model, causal evidence) has been folded into Paper 3's COLM expansion plan (Tasks 1-10 in paper3_improvements.md). If Paper 3 is accepted at COLM, Paper 5 becomes an extended journal version. If Paper 3 is rejected, the expanded version can target NeurIPS 2026.

---

## Timeline Summary

| Paper | Target Venue | Deadline | Status | Priority |
|-------|-------------|----------|--------|----------|
| Paper 2 | COLM 2026 main / NeurIPS | Mar 31 / May 2026 | Expanding | HIGHEST |
| Paper 3 | COLM 2026 main track | Mar 31, 2026 | Expanding | HIGHEST |
| Paper 1 | ICML 2026 Workshop | ~June 2026 | Needs A+C | HIGH |
| Paper 4 | ICML/COLM Workshop | ~June 2026 | Draft done | MEDIUM |
| Paper 5 | Superseded by Paper 3 COLM | — | Folded into Paper 3 | — |

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-13 | Paper 2: Full expansion for COLM main track | Best candidate for top venue |
| 2026-03-13 | Paper 1: ICML workshop with A+C additions | A (SWE-bench) + C (rankings change) = strong workshop paper |
| 2026-03-13 | Paper 1 and Paper 2 remain separate | Different scopes: "what" vs "why and so what" |
| 2026-03-13 | Paper 2 deadline flexible | Quality over speed |
| 2026-03-14 | Paper 3: ICML workshop (not COLM main) | Cross-benchmark risk; workshop first, extend if validates |
| 2026-03-14 | Paper 5 = Extended Paper 3 -> NeurIPS | Go/no-go after ALFWorld pilot |
