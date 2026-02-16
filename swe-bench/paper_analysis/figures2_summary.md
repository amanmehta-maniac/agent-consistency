# Complete Figure Set: 3-Model Comparison (Claude, GPT-5, Llama)

All 17 figures verified against raw JSON data. Located in `paper_analysis/figures2/`.
Every figure includes all 3 models (Claude 4.5 Sonnet, GPT-5 Snowflake, Llama-3.1-70B).

---

## Core Metrics (Figures 1–3)

### fig_cv_distribution
**[View on GitHub](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/figures2/fig_cv_distribution.png)**

Violin plot with overlaid data points showing CV distribution per model.
- Claude: μ=13.6% (tight cluster, low variance)
- GPT-5: μ=32.2% (moderate spread)
- Llama: μ=42.0% (wide spread)

**Key insight**: Claude is 2.4x more consistent than GPT-5 and 3.1x more than Llama.

---

### fig_step_heatmap
**[View on GitHub](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/figures2/fig_step_heatmap.png)**

Three side-by-side heatmaps (10 tasks × 5 runs) showing raw step counts per run.
- Claude: uniformly high (40–60 range), consistent coloring
- GPT-5: low counts (4–29), some hot spots
- Llama: moderate (5–40), patchy/variable

**Key insight**: Visual proof of Claude's consistency (uniform color) vs Llama's chaos (scattered).

---

### fig_step_distributions
**[View on GitHub](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/figures2/fig_step_distributions.png)**

Overlaid histograms of all step counts with mean lines.
- Claude: tight peak at ~45 steps
- GPT-5: left-skewed, peak at ~7 steps
- Llama: spread across 5–40 steps

**Key insight**: GPT-5 is 4.7x faster than Claude but the distributions barely overlap.

---

## Tradeoff Analysis (Figures 4–5)

### fig_tradeoff_scatter
**[View on GitHub](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/figures2/fig_tradeoff_scatter.png)**

Scatter plot: X = mean steps per task, Y = CV per task. Large markers = model means.
- Claude cluster: high steps, low CV (upper-right → consistent but slow)
- GPT-5 cluster: low steps, mid CV (lower-left → fast but variable)
- Llama cluster: mid steps, high CV (scattered)

**Key insight**: Clear separation between model clusters reveals the speed-consistency tradeoff.

---

### fig_per_task_steps
**[View on GitHub](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/figures2/fig_per_task_steps.png)**

Grouped bar chart with error bars: mean ± std steps per task for all 3 models.

**Key insight**: Claude takes ~45 steps on EVERY task. GPT-5 varies (7–17). Llama varies (13–24).

---

## Accuracy (Figures 6–7)

### fig_accuracy
**[View on GitHub](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/figures2/fig_accuracy.png)**

Per-task accuracy bars: resolved runs out of 5 for each model.
- Claude 29/50, GPT-5 16/50, Llama 2/50
- Task 14309: All 3 solve it (Claude 5/5, GPT-5 5/5, Llama 1/5)
- Task 13236: Only Llama solves it (1/5)

**Key insight**: Claude dominates on most tasks; GPT-5 is competitive on ~6 tasks.

---

### fig_cv_vs_accuracy
**[View on GitHub](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/figures2/fig_cv_vs_accuracy.png)**

Scatter: X = CV, Y = accuracy. Per-task points (small) and model means (large).

**Key insight**: Perfect negative correlation — lower CV → higher accuracy across all 3 models.

---

## Behavioral Analysis (Figures 8–9)

### fig_phase_decomposition
**[View on GitHub](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/figures2/fig_phase_decomposition.png)**

Three-panel horizontal bars: % of actions in each phase (EXPLORE/UNDERSTAND/EDIT/VERIFY/OTHER).
- Claude: 42% UNDERSTAND, 18% EXPLORE — reads a lot before acting
- GPT-5: 32% VERIFY, 31% UNDERSTAND — tests aggressively
- Llama: 31% UNDERSTAND, 28% EXPLORE — explores broadly

**Key insight**: GPT-5's heavy VERIFY (32%) explains its speed — it tests immediately rather than reading extensively.

---

### fig_failure_modes
**[View on GitHub](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/figures2/fig_failure_modes.png)**

Grouped bar chart of failure types: WRONG_FIX / EMPTY_PATCH / LOOP_DEATH.
- Claude: 21 failures, all WRONG_FIX (0 empty, 0 loop)
- GPT-5: 34 failures, 32 WRONG_FIX + 2 EMPTY_PATCH
- Llama: 48 failures, 38 WRONG_FIX + 10 EMPTY_PATCH

**Key insight**: All models fail by submitting wrong code, not by getting stuck. Llama uniquely has 10 empty patches.

---

## First Action Analysis (Figures 10–12)

### fig_first_action_distribution
**[View on GitHub](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/figures2/fig_first_action_distribution.png)**

Grouped bar chart: first command distribution (find/ls/grep/cat) per model.
- Claude: 68% `find`, 26% `ls`
- GPT-5: 100% `ls`
- Llama: 54% `ls`, 24% `grep`, 20% `find`

**Key insight**: Each model has a distinctive opening strategy. GPT-5 is the most predictable.

---

### fig_first_action_outcome
**[View on GitHub](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/figures2/fig_first_action_outcome.png)**

Three-panel stacked bars: success/fail counts by first action, with success rate labels.
- Claude `ls` → 85% success, `find` → 50%
- GPT-5 `ls` → 32% success
- Llama `ls` → 4%, `grep` → 8%

**Key insight**: Same first action (`ls`), vastly different outcomes (85% vs 32% vs 4%). The model matters, not the opening move.

---

### fig_first_action_heatmap
**[View on GitHub](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/figures2/fig_first_action_heatmap.png)**

Matrix: First Action Category (EXPLORE/UNDERSTAND) × Model × Outcome (✓/✗). Color = count.
- Claude EXPLORE-success: 28
- GPT-5 EXPLORE-success: 16
- Llama EXPLORE-success: 1

**Key insight**: Claude converts EXPLORE starts to success 28x more often than Llama.

---

## Divergence Analysis (Figures 13–17)

### fig_divergence_per_task
**[View on GitHub](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/figures2/fig_divergence_per_task.png)**

Grouped bar chart: divergence step for each of 10 tasks, all 3 models. Dashed mean lines.
- Claude: μ=3.2 (range 1–9)
- GPT-5: μ=3.4 (range 2–5)
- Llama: μ=1.4 (range 1–2)

**Key insight**: Two tiers — Claude/GPT-5 (~step 3) vs Llama (step 1). Claude has highest variance (astropy-13398 = step 9).

---

### fig_divergence_heatmap
**[View on GitHub](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/figures2/fig_divergence_heatmap.png)**

Three side-by-side heatmaps (10 tasks × 10 steps): runs diverged from majority at each step.
- Claude: zeros in early columns → consistent early strategy
- GPT-5: gradual fill → moderate early agreement
- Llama: filled from step 1 → immediate divergence

**Key insight**: Visually shows Claude's "shared opening" vs Llama's "chaos from step 1".

---

### fig_divergence_distribution
**[View on GitHub](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/figures2/fig_divergence_distribution.png)**

Histogram of divergence steps with mean vertical lines.
- Llama peaks at step 1 (6/10 tasks)
- Claude/GPT-5 spread across steps 1–9

**Key insight**: Llama lacks any shared strategy; Claude and GPT-5 agree on early steps.

---

### fig_sequence_alignment (astropy-13236)
**[View on GitHub](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/figures2/fig_sequence_alignment.png)**

Color-coded action sequences for all 5 runs × 3 models. Task where Llama won, Claude lost.
- Claude: 0/5 ✗, long sequences (~40 steps), heavy EDIT/UNDERSTAND
- GPT-5: 0/5 ✗, very short (4–9 steps), VERIFY-heavy
- Llama: 1/5 ✓ (Run 5), direct UNDERSTAND→EDIT pattern

**Key insight**: Case study showing Llama's rare "lucky direct approach" succeeding where Claude over-engineers.

---

### fig_sequence_alignment_14309 (astropy-14309)
**[View on GitHub](https://github.com/amanmehta-maniac/agent-consistency/blob/main/swe-bench/paper_analysis/figures2/fig_sequence_alignment_14309.png)**

Same format for a task where Claude (5/5) and GPT-5 (5/5) both excel.
- Claude: All ✓, 40+ steps, diverse EXPLORE/UNDERSTAND/EDIT paths
- GPT-5: All ✓, 4–10 steps, quick EDIT→VERIFY
- Llama: 1/5 ✓, most runs stuck in EXPLORE

**Key insight**: GPT-5 solves it in 4 steps where Claude takes 40+ — dramatic speed difference on "solvable" tasks.
