# Updated Figures: 3-Model Comparison (Claude, GPT-5, Llama)

All figures verified against raw JSON data. Located in `paper_analysis/figures2/`.

---

## First Action Figures

### fig_first_action_distribution
**[View on GitHub](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/figures2/fig_first_action_distribution.png)**

Grouped bar chart showing first action command distribution across models.
- Claude: 68% `find`, 26% `ls`
- GPT-5: 100% `ls`
- Llama: 54% `ls`, 24% `grep`, 20% `find`

**Key insight**: Each model has a distinctive opening strategy.

---

### fig_first_action_outcome
**[View on GitHub](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/figures2/fig_first_action_outcome.png)**

Three-panel stacked bar: For each model, shows success/fail counts per first action command. Success rates labeled on top.
- Claude `ls` → 85% success, `find` → 50%
- GPT-5 `ls` → 32% success
- Llama `ls` → 4%, `grep` → 8%

**Key insight**: Same first action, vastly different outcomes. The model matters, not the first move.

---

### fig_first_action_heatmap
**[View on GitHub](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/figures2/fig_first_action_heatmap.png)**

Matrix showing First Action Category (EXPLORE/UNDERSTAND) × Model × Outcome (success/fail). Color intensity = count.

**Key insight**: Claude has 28 EXPLORE-success vs Llama's 1 — same category, 28x difference.

---

## Divergence Figures

### fig_divergence_per_task
**[View on GitHub](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/figures2/fig_divergence_per_task.png)**

Grouped bar chart: For each of 10 tasks, shows the divergence step for all 3 models. Dashed lines at model means.
- Claude mean: 3.2 (range 1–9)
- GPT-5 mean: 3.4 (range 2–5)
- Llama mean: 1.4 (range 1–2)

**Key insight**: Claude and GPT-5 share similar divergence timing (~step 3), while Llama diverges immediately.

---

### fig_divergence_heatmap
**[View on GitHub](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/figures2/fig_divergence_heatmap.png)**

Three side-by-side heatmaps (10 tasks × 10 steps): Shows how many runs (0–4) have diverged from the majority action at each step.
- Claude: sparse early, fills later → consistent early strategy
- GPT-5: moderate divergence pattern
- Llama: fills immediately → no shared strategy

**Key insight**: Visually shows Claude's "shared opening" vs Llama's "chaos from step 1".

---

### fig_divergence_distribution
**[View on GitHub](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/figures2/fig_divergence_distribution.png)**

Histogram of divergence step per model with mean lines.
- Llama peaks at step 1 (6/10 tasks diverge immediately)
- Claude/GPT-5 spread across steps 1–9

**Key insight**: Two-tier pattern — Claude/GPT-5 vs Llama.

---

### fig_sequence_alignment (astropy-13236)
**[View on GitHub](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/figures2/fig_sequence_alignment.png)**

Color-coded action sequences for all 5 runs × 3 models on astropy-13236 (the task Llama won).
- Claude: 0/5 resolved, long sequences (~40 steps), heavy EDIT
- GPT-5: 0/5 resolved, short sequences (4–9 steps)
- Llama: 1/5 resolved (Run 5), UNDERSTAND→EDIT pattern

**Key insight**: Our case study task — shows how Llama's shorter, direct approach occasionally wins.

---

### fig_sequence_alignment_14309 (astropy-14309)
**[View on GitHub](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/figures2/fig_sequence_alignment_14309.png)**

Same format for astropy-14309 — a task where Claude (5/5) and GPT-5 (5/5) both ace it.
- Claude: All ✓, longer but diverse paths
- GPT-5: All ✓, short and VERIFY-heavy
- Llama: 1/5 ✓, most runs wander through EXPLORE

**Key insight**: On "solvable" tasks, GPT-5 is dramatically faster (4–10 steps vs Claude's 40+).
