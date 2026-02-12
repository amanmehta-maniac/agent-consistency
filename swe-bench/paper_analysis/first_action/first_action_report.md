# First Action Analysis

## Summary

**Key Finding**: First action category is strongly dependent on model, but the MODEL determines success, not the first action.

---

## Table A: First Action Distribution

| First Action | Claude (n=50) | Llama (n=50) |
|--------------|---------------|--------------|
| find | 34 (68%) | 10 (20%) |
| ls | 13 (26%) | 27 (54%) |
| grep | 1 (2%) | 12 (24%) |
| cat | 2 (4%) | 1 (2%) |

**Key insight**: Claude strongly prefers `find` (68%), while Llama prefers `ls` (54%).

---

## Table B: First Action Category by Model

| Category | Claude | Llama |
|----------|--------|-------|
| EXPLORE (ls, cd) | 13 (26%) | 27 (54%) |
| FIND | 34 (68%) | 10 (20%) |
| GREP | 1 (2%) | 12 (24%) |
| UNDERSTAND | 2 (4%) | 1 (2%) |

**Claude's strategy**: Start with targeted `find` to locate relevant files.
**Llama's strategy**: Start with broad `ls` to explore directory structure.

---

## Table C: Success Rate by First Action Category

| Category | Overall | Claude | Llama |
|----------|---------|--------|-------|
| EXPLORE | 30% | **85%** (11/13) | 4% (1/27) |
| FIND | 39% | **50%** (17/34) | 0% (0/10) |
| GREP | 8% | 0% (0/1) | 8% (1/12) |
| UNDERSTAND | 33% | 50% (1/2) | 0% (0/1) |

**Critical insight**: Same first action, vastly different outcomes!
- EXPLORE leads to 85% success for Claude, but only 4% for Llama.

---

## Statistical Tests

### Test 1: Is First Action Category Independent of Model?

**Chi-square**: χ² = 27.63, p = 0.000004

**Result**: **Highly significant** — First action IS dependent on model.

### Test 2: Is First Action Category Independent of Success?

**Chi-square**: χ² = 4.53, p = 0.2098

**Result**: **Not significant** — First action alone doesn't predict success.

### Test 3: Does Starting with FIND Predict Success?

**Fisher's exact**: odds ratio = 1.89, p = 0.1916

**Result**: **Not significant** — FIND doesn't guarantee success.

---

## Key Findings

### 1. Claude and Llama start differently (p < 0.00001)
- Claude: 68% start with `find` (targeted search)
- Llama: 54% start with `ls` (broad exploration)

### 2. First action alone doesn't predict success
- Same first action → different outcomes for different models
- The model's overall approach matters, not just the first step

### 3. The shocking comparison
| Scenario | Success Rate |
|----------|--------------|
| Claude + EXPLORE | **85%** |
| Llama + EXPLORE | **4%** |

Starting with `ls` works for Claude but not for Llama!

### 4. Why first action doesn't predict success
- Claude's first action is part of a **systematic strategy**
- Llama's first action is often **disconnected from success**
- What matters is **what comes after** the first action

---

## Implications for Paper

> **First action predicts model, not success.**
>
> Claude's consistent use of `find` reflects a methodical approach,
> but it's the full trajectory — not the first step — that determines success.

This supports the divergence analysis: runs diverge at Step 3 (Claude) or Step 1 (Llama),
meaning the first action is typically consistent, but later actions determine outcomes.

---

## Figures

- `fig_first_action_heatmap.png`: First action × Model × Outcome matrix
- `fig_first_action_flow.png`: Stacked bar chart of outcomes by first action
