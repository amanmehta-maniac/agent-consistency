# Paper 1 — Revision Plan for ICML Workshop 2026

**Paper**: "When Agents Disagree With Themselves: Measuring Behavioral Consistency in LLM-Based Agents"  
**Target**: ICML 2026 Workshop (agent evaluation / reliability)  
**Goal**: Transform from borderline to clear accept (95%+ acceptance)  
**Output**: `paper1_v2.tex` — keep `paper1.tex` unchanged

---

## ⚠️ CRITICAL: ALL COMPUTE VIA SNOWFLAKE CORTEX

**DO NOT use any direct API keys.** No `OPENAI_API_KEY`, `TOGETHER_API_KEY`, 
`ANTHROPIC_API_KEY`, or `GOOGLE_API_KEY`. ALL LLM inference MUST go through 
Snowflake Cortex's `COMPLETE()` / `AI_COMPLETE()` function.

### Snowflake Cortex model IDs for this project:

| Paper Model Name    | Cortex Model ID       | Provider  |
|--------------------|-----------------------|-----------|
| Claude Sonnet 4.5  | `claude-sonnet-4-5`   | Anthropic |
| GPT-5              | `openai-gpt-5`        | OpenAI    |
| GPT-4o             | `openai-gpt-4.1`      | OpenAI    |
| Llama 3.1 70B      | `llama3.1-70b`        | Meta      |
| Gemini 3 Pro       | `gemini-3-pro`        | Google    |

You can verify what the actual cortex model ID is by looking at documentation or just flag me if any doesn't work

### Snowflake Cortex connection:

Use `snowflake-connector-python` with named connections. This is preferred over 
the legacy `snow sql` CLI approach (Pattern A in `hotpotqa/agent.py`) because it 
provides parameterized queries (no SQL injection risk), connection reuse across 
calls, and proper Python error handling.

```python
import os
import json
import snowflake.connector

conn = snowflake.connector.connect(
    connection_name=os.getenv("SNOWFLAKE_CONNECTION_NAME") or "ML_DATA",
)
cursor = conn.cursor()
cursor.execute(
    "SELECT AI_COMPLETE(%s, PARSE_JSON(%s), PARSE_JSON(%s)) AS response",
    (model_id, messages_json, options_json),
)
result = cursor.fetchone()
response_data = json.loads(result[0])
text = response_data['choices'][0]['messages']
```

Uses named connections from `~/.snowflake/connections.toml` (default: `ML_DATA`). 
Override at runtime: `SNOWFLAKE_CONNECTION_NAME=other_conn python runner.py ...`

Reference implementation:
`swe-bench/mini-swe-agent/src/minisweagent/models/snowflake_cortex_model.py`

### NOTE on existing data:
The existing results in `results_llama/`, `results_claude/`, `results_gpt4o/` 
were collected earlier using direct API keys. That data is valid and should be 
reused. All NEW experiments (GPT-5, Gemini 3 Pro, additional 100 questions, 
temperature ablation) MUST use Snowflake Cortex.

---

## CONTEXT & EXISTING DATA

### Repository layout
```
agent-consistency/
├── hotpotqa/
│   ├── agent.py          # ReActAgent — supports openai, together, anthropic, 
│   │                     #   llama_spcs, llama_k8s providers
│   │                     #   MUST ADD: "snowflake_cortex" provider for new experiments
│   ├── runner.py          # HotpotQARunner — CLI entrypoint for experiments
│   │                     #   MUST ADD: --model flag and "snowflake_cortex" provider
│   ├── tools.py           # create_search_fn, create_retrieve_fn
│   ├── results_llama/     # 100 JSON files, 10 runs each (Llama 3.1 70B, temp 0.7)
│   ├── results_claude/    # 100 JSON files, 10 runs each (Claude Sonnet 4.5, temp 0.7)
│   ├── results_gpt4o/     # 100 JSON files, 10 runs each (GPT-4o, temp 0.7)
│   ├── results_llama_temp0/  # 20 JSON files (Llama 3.1 70B, temp 0.0)
│   └── figures/           # existing plots
├── swe-bench/
│   ├── analysis_results.md   # SWE-bench pilot results (10 tasks × 5 runs)
│   ├── gpt5.gpt5-local-run{1-5}.json
│   ├── claude.claude-local-run{1-5}.json
│   ├── llama.llama-local-run{1-5}.json
│   └── mini-swe-agent/src/minisweagent/models/snowflake_cortex_model.py
│       # ↑ Reference implementation of Snowflake Cortex model class
├── common/
│   ├── analysis.py        # Metrics: answer_consistency, step_count_metrics, 
│   │                      #   action_sequence_similarity, first_divergence_point,
│   │                      #   is_correct, count_unique_sequences, analyze_task
│   └── figures.py         # Figure generation (histogram, bar chart)
└── paper1/
    ├── paper1.tex         # Current paper (DO NOT MODIFY)
    ├── paper1.pdf
    ├── references.bib
    ├── rankings_analysis.py
    ├── rankings_instability.png/pdf
    ├── unique_sequences_histogram.png/pdf
    ├── correctness_comparison.png/pdf
    ├── icml2026.sty / icml2026.bst
    └── paper1_improvements.md  (this file)
```

### Data format (each JSON file in results_*/):
```json
{
  "task_id": "5a713ea95542994082a3e6e4",
  "question": "...",
  "answer": "Apalachees",        // gold answer
  "type": "bridge",              // or "comparison"
  "level": "hard",
  "n_runs": 10,
  "runs": [
    {
      "run_id": "...",
      "final_answer": "...",
      "steps": [
        {
          "step_number": 1,
          "action": "Search",
          "action_input": "{'query': '...'}",
          "observation": "..."
        },
        ...
      ]
    },
    ...  // 10 runs total
  ]
}
```

### Current paper numbers (from paper1.tex):
- 3 models: Llama 3.1 70B, GPT-4o, Claude Sonnet 4.5
- 100 questions × 10 runs × 3 models = 3,000 runs on HotpotQA
- SWE-bench pilot: 10 tasks × 5 runs × 3 models (Claude, GPT-5, Llama) = 150 runs
- Temperature ablation: Llama only, temp 0.0 vs 0.7, 20 questions
- Correctness metric: fuzzy match (contains/is-contained-in)

### How to run HotpotQA experiments (via Snowflake Cortex):

First, add a `"snowflake_cortex"` provider to `agent.py` and `runner.py`. 
Use the `SnowflakeCortexModel` class from 
`swe-bench/mini-swe-agent/src/minisweagent/models/snowflake_cortex_model.py` 
as reference, or implement a simpler version using `snowflake-connector-python` 
directly. The key is calling `AI_COMPLETE(model, messages, options)`.

Then add a `--model` flag to `runner.py` so the Cortex model ID can be specified.

```bash
cd hotpotqa/

# Llama 3.1 70B (via Snowflake Cortex):
python runner.py \
  --provider snowflake_cortex \
  --model llama3.1-70b \
  --results-dir results_llama \
  --n-questions 100 --n-runs-per-question 10 \
  --temperature 0.7

# GPT-4o equivalent (via Snowflake Cortex — GPT-4.1 on Cortex):
python runner.py \
  --provider snowflake_cortex \
  --model openai-gpt-4.1 \
  --results-dir results_gpt4o \
  --n-questions 100 --n-runs-per-question 10 \
  --temperature 0.7

# Claude Sonnet 4.5 (via Snowflake Cortex):
python runner.py \
  --provider snowflake_cortex \
  --model claude-sonnet-4-5 \
  --results-dir results_claude \
  --n-questions 100 --n-runs-per-question 10 \
  --temperature 0.7

# GPT-5 (via Snowflake Cortex):
python runner.py \
  --provider snowflake_cortex \
  --model openai-gpt-5 \
  --results-dir results_gpt5 \
  --n-questions 100 --n-runs-per-question 10 \
  --temperature 0.7

# Gemini 3 Pro (via Snowflake Cortex):
python runner.py \
  --provider snowflake_cortex \
  --model gemini-3-pro \
  --results-dir results_gemini \
  --n-questions 100 --n-runs-per-question 10 \
  --temperature 0.7
```

The runner loads HotpotQA validation set (distractor config) from HuggingFace.
The first 100 questions are the existing dataset. Questions 100-199 are the
second batch for scaling (see Task 0).

---

## PHASE 0 — NEW EXPERIMENTS (run before any analysis)

### Task 0-PREREQ: Add Snowflake Cortex provider to HotpotQA agent

Before running any new experiments, modify the HotpotQA agent infrastructure:

1. **Add `snowflake_cortex` provider to `agent.py`**:
   - Add `"snowflake_cortex"` to the `provider` Literal type
   - Implement `_call_snowflake_cortex()` method in `ReActAgent._call_llm()`
   - Use `snowflake-connector-python` (Pattern B above) for the implementation
   - Reference: `swe-bench/mini-swe-agent/src/minisweagent/models/snowflake_cortex_model.py`
   - The method receives `messages` (list of dicts) and must:
     a. Connect to Snowflake (use PAT token auth or connections.toml)
     b. Call `AI_COMPLETE(model, messages_json, options_json)`
     c. Parse the response and return the text content
   - Store the model ID (e.g., `claude-sonnet-4-5`, `openai-gpt-5`, 
     `gemini-3-pro`, `llama3.1-70b`) as `self.model`

2. **Add `--model` flag to `runner.py`**:
   - Currently the model is auto-selected based on provider. Add explicit 
     `--model` override so you can specify any Cortex model ID.
   - Add `"snowflake_cortex"` to the provider choices.
   - When provider is `snowflake_cortex`, use the `--model` value directly 
     as the Cortex model ID.

3. **Test with a single question** before running at scale:
   ```bash
   python runner.py \
     --provider snowflake_cortex \
     --model llama3.1-70b \
     --results-dir results_test_cortex \
     --n-questions 1 --n-runs-per-question 1 \
     --temperature 0.7
   ```
   Verify the output JSON has the same format as existing results.

### Task 0A: Add GPT-5 to HotpotQA (100 existing questions)

**Why**: The SWE-bench pilot uses GPT-5, but HotpotQA uses GPT-4o. Adding GPT-5
to HotpotQA fixes the model mismatch for cross-benchmark comparison and adds a 
4th model (GPT-4o stays as well — more models = stronger claims).

**Steps**:
1. Run GPT-5 on the same 100 questions, 10 runs each, temperature 0.7, 
   via Snowflake Cortex:
   ```bash
   python runner.py \
     --provider snowflake_cortex \
     --model openai-gpt-5 \
     --results-dir results_gpt5 \
     --n-questions 100 --n-runs-per-question 10 \
     --temperature 0.7
   ```
   Save results to `results_gpt5/`. Each file should have the same task_id as 
   the corresponding file in `results_llama/`, `results_claude/`, `results_gpt4o/`.

2. Total new runs: 100 × 10 = **1,000 runs**.

### Task 0B: Add Gemini 3 Pro to HotpotQA (100 existing questions)

**Why**: Adds Google as a 5th provider. With 5 models across 4 providers 
(Meta, OpenAI, Anthropic, Google), the "across models" claim becomes 
much stronger. Gemini 3 Pro is available on Snowflake Cortex as `gemini-3-pro`.

**Steps**:
1. Run Gemini 3 Pro on the same 100 questions, 10 runs each, temperature 0.7, 
   via Snowflake Cortex:
   ```bash
   python runner.py \
     --provider snowflake_cortex \
     --model gemini-3-pro \
     --results-dir results_gemini \
     --n-questions 100 --n-runs-per-question 10 \
     --temperature 0.7
   ```

2. **NOTE**: Gemini may format its ReAct responses differently from 
   Claude/GPT/Llama. Test on 1-2 questions first and verify that the 
   Thought/Action/Action Input parsing works correctly. If Gemini uses a 
   different format, add a model-specific parser in `agent.py`.

3. Total new runs: 100 × 10 = **1,000 runs**.

### Task 0C: Scale HotpotQA to 200 questions (all 5 models)

**Why**: Doubles statistical power. More questions per difficulty stratum 
(Task 1), more data for failure detection (Task 2), more convincing correlations.

**Steps**:
1. The existing 100 questions are `validation[0:100]` from HotpotQA distractor.
   The next 100 questions are `validation[100:200]`.
   
2. Run ALL 5 models on questions 100-199, 10 runs each, temperature 0.7.
   All via Snowflake Cortex:
   ```bash
   # For each model, run on questions 100-199:
   for MODEL in llama3.1-70b openai-gpt-4.1 openai-gpt-5 claude-sonnet-4-5 gemini-3-pro; do
     RESULTS_DIR="results_$(echo $MODEL | sed 's/openai-gpt-4.1/gpt4o/;s/openai-gpt-5/gpt5/;s/claude-sonnet-4-5/claude/;s/gemini-3-pro/gemini/;s/llama3.1-70b/llama/')"
     python runner.py \
       --provider snowflake_cortex \
       --model $MODEL \
       --results-dir $RESULTS_DIR \
       --n-questions 200 --n-runs-per-question 10 \
       --temperature 0.7
   done
   ```
   
   **IMPORTANT**: Check that the runner skips existing result files. If it 
   overwrites, modify it to skip files that already exist in results_dir.
   Alternatively, add a `--start-idx` flag, or have the runner check for 
   existing `{task_id}.json` files before running.

3. Total new runs: 100 new questions × 10 runs × 5 models = **5,000 runs**
   (Plus 2,000 from 0A+0B on existing 100 questions = **7,000 total new runs**)

### Task 0D: Temperature ablation expansion

**Why**: Current ablation is Llama-only at {0.0, 0.7}. Expand to all 5 models 
at {0.0, 0.3, 0.7} on 20 questions.

**Steps**:
1. Use the same 20 questions as the existing temp 0.0 ablation 
   (check `results_llama_temp0/` for the 20 task_ids — these are the first 20 
   questions from the validation set, matching `pilot_questions.json`).

2. Run each model at each temperature, 5 runs per question, all via Cortex:
   ```bash
   # For each model × temperature combination:
   python runner.py \
     --provider snowflake_cortex \
     --model [CORTEX_MODEL_ID] \
     --results-dir results_[model]_temp[T] \
     --n-questions 20 --n-runs-per-question 5 \
     --temperature [T]
   ```
   
   Matrix (skip combinations that already exist):
   | Model | Cortex ID | temp 0.0 | temp 0.3 | temp 0.7 |
   |-------|-----------|----------|----------|----------|
   | Llama 70B | `llama3.1-70b` | EXISTS (10 runs) | NEW (5 runs) | EXISTS (from main) |
   | GPT-4o | `openai-gpt-4.1` | NEW | NEW | EXISTS (from main) |
   | GPT-5 | `openai-gpt-5` | NEW | NEW | from Task 0A |
   | Claude 4.5 | `claude-sonnet-4-5` | NEW | NEW | EXISTS (from main) |
   | Gemini 3 | `gemini-3-pro` | NEW | NEW | from Task 0B |
   
   For temp 0.7 results: extract the matching 20 questions from the main 
   100-question results (use only 5 of the 10 runs for fair comparison).
   
3. Total new runs: ~5 models × 2 new temps × 20 questions × 5 runs = 
   **~1,000 runs** (minus existing Llama temp 0.0).

### Summary of Phase 0

| Task | New Runs | Models Added | Compute |
|------|----------|-------------|---------|
| 0-PREREQ: Add Snowflake Cortex provider | 0 | — | Code only |
| 0A: GPT-5 on 100 existing questions | 1,000 | GPT-5 | Cortex |
| 0B: Gemini 3 Pro on 100 existing questions | 1,000 | Gemini 3 Pro | Cortex |
| 0C: All 5 models on 100 new questions | 5,000 | — | Cortex |
| 0D: Temperature ablation expansion | ~1,000 | All 5 | Cortex |
| **Total** | **~8,000** | **5 models, 4 providers** | **All Cortex** |

**After Phase 0, the dataset is:**
- HotpotQA main: 200 questions × 10 runs × 5 models = **10,000 runs**
- Temperature ablation: 20 questions × 5 runs × 5 models × 3 temps = **1,500 data points**
- SWE-bench pilot: unchanged (10 tasks × 5 runs × 3 models = 150 runs)

---

## PHASE 1 — ANALYSIS (on expanded dataset)

All analysis tasks below use the full expanded dataset (200 questions, 5 models).
Create all analysis scripts in `paper1/analysis/` directory.

### Task 1: Fix the fatal confound — difficulty stratification

**Why**: The central claim "consistency predicts correctness" is vulnerable to 
the critique that task difficulty is a latent confounder: harder tasks produce 
both lower consistency AND lower accuracy, making the observed correlation 
uninformative.

**Steps**:

1. Create `paper1/analysis/task1_difficulty_stratification.py`:
   - Load ALL result files from all 5 model directories (results_llama, 
     results_gpt4o, results_gpt5, results_claude, results_gemini)
   - For each of the 200 questions, compute a **difficulty proxy**: average 
     correctness across ALL runs from ALL models. This is model-agnostic.
     Use the `is_correct()` function from `common/analysis.py`.
   - Bin questions into three difficulty strata:
     - **Easy**: avg correctness >= 0.80
     - **Medium**: avg correctness 0.40–0.79
     - **Hard**: avg correctness < 0.40
   - Report how many questions fall into each bin.

2. Within each stratum, for each model, compute:
   - Mean accuracy for **consistent** tasks (≤2 unique action sequences)
   - Mean accuracy for **inconsistent** tasks (≥4 unique sequences)
   - Note: use ≥4 (not ≥6) to ensure sufficient n within strata
   - The **gap** between consistent and inconsistent accuracy

3. Produce a table (printed + saved as JSON):
   ```
   Stratum | n_questions | Llama_gap | GPT4o_gap | GPT5_gap | Claude_gap | Gemini_gap
   Easy    | XX          | XXpp      | XXpp      | XXpp     | XXpp       | XXpp
   Medium  | XX          | XXpp      | XXpp      | XXpp     | XXpp       | XXpp
   Hard    | XX          | XXpp      | XXpp      | XXpp     | XXpp       | XXpp
   ```

4. Compute **partial correlation**: corr(unique_sequences, accuracy | difficulty)
   using scipy.stats or pingouin. Control variable = difficulty proxy (continuous, 
   not binned). Report partial r and p-value for each model.

5. Save all results to `paper1/analysis/task1_results.json`.

**How to interpret**: If the gap persists within strata (even among tasks of 
similar difficulty, consistent runs outperform inconsistent runs), the confound 
critique is refuted. If it vanishes, the claim must be softened.

### Task 2: Failure detection experiment

**Why**: The Discussion proposes consistency as a "runtime signal" for error 
detection but provides no experimental validation. This must become a real result.

**Steps**:

1. Create `paper1/analysis/task2_failure_detection.py`:
   - Frame as binary classification: for each task-model pair, predict whether 
     the majority answer is incorrect, using only consistency metrics.

2. Features (each as a standalone threshold classifier):
   - **unique_sequences**: number of unique action sequences across 10 runs. 
     Threshold: flag if > K for K ∈ {2, 3, 4, 5}
   - **step_variance_ratio**: (max - min) / mean of step counts. 
     Threshold: flag if > K for K ∈ {0.3, 0.5, 0.7, 1.0}
   - **first_divergence_step**: flag if divergence occurs at step ≤ 2
   - **answer_entropy**: H = -Σ(p_i × log₂(p_i)) over the answer distribution 
     across 10 runs. Higher entropy = more disagreement.
     Threshold: flag if > K for K ∈ {0.5, 1.0, 1.5, 2.0}

3. Ground truth label: the task is "incorrect" if the majority answer 
   (most common answer across 10 runs) does not match the gold answer 
   (using the `is_correct()` function from `common/analysis.py`).

4. For each feature × threshold × model, compute:
   - Precision: of flagged tasks, what fraction are actually incorrect?
   - Recall: of actually incorrect tasks, what fraction are flagged?
   - F1 score
   - Also compute AUROC using the continuous feature value (not thresholded) 
     via `sklearn.metrics.roc_auc_score`

5. Compute **no-signal baseline**: random flagging at the same positive rate. 
   Baseline precision = (n_incorrect / n_total).

6. Produce output:
   - Table: best precision/recall/F1/AUROC per feature, per model
   - Save to `paper1/analysis/task2_results.json`

7. **Primary model for paper**: Use Llama 3.1 70B (highest variance, most signal). 
   Report Claude, GPT-4o, GPT-5, Gemini results briefly.

### Task 5: Correctness metric sensitivity analysis

**Why**: The paper uses fuzzy matching without justification. Reviewers will 
question whether results hold under standard HotpotQA metrics (EM, F1).

**Steps**:

1. Create `paper1/analysis/task5_metric_sensitivity.py`:

2. Implement three correctness metrics:
   - **Fuzzy match** (current): answer contains gold or gold contains answer 
     (case-insensitive). This is `is_correct()` from `common/analysis.py`.
   - **Exact match (EM)**: standard HotpotQA metric. Normalize both answer 
     and gold (lowercase, strip articles "a", "an", "the", strip punctuation, 
     collapse whitespace), then check equality.
   - **Token F1**: standard HotpotQA metric. Tokenize (split on whitespace) 
     normalized answer and gold. Compute precision = |overlap|/|pred_tokens|, 
     recall = |overlap|/|gold_tokens|, F1 = 2PR/(P+R). A run is "correct" 
     if F1 > 0.5 (or report with threshold 0.5).

3. Re-compute the main Table 1 (overall model comparison) under all three metrics:
   - For each model: overall accuracy, mean unique sequences, mean steps, variance
   - Check that model rankings and consistency-correctness gaps are robust across 
     all three metrics.

4. Save to `paper1/analysis/task5_results.json`.

5. Output: a comparison table showing all three metrics side by side for each model.

### Task 6: Statistical reporting

**Why**: Only Llama has statistical tests. All models need effect sizes and CIs.

**Steps**:

1. Create `paper1/analysis/task6_statistics.py`:

2. For the consistency-correctness gap (from the main analysis), for EACH model:
   - Mann-Whitney U test p-value (consistent vs inconsistent tasks)
   - Effect size: rank-biserial correlation r = 1 - (2U)/(n1×n2)
   - Report n_consistent and n_inconsistent
   - If any group has n < 10, flag this as a small-sample caveat

3. For the ranking instability (17.8% from rankings_analysis.py):
   - Compute 95% CI around this proportion from the 10,000 bootstrap samples.
   - Simply take the 2.5th and 97.5th percentiles of a binary indicator 
     (does this bootstrap sample match the ground truth ranking?).
   - Report as: "17.8% [95% CI: X%–Y%]"

4. For step-2 divergence (69%):
   - Compute binomial 95% CI: this is now across ALL 5 models' tasks.
   - Use `scipy.stats.binom.interval()` or the Wilson score interval.

5. For path-length correlation (r = -0.34):
   - Report p-value and n (now 200 questions per model, or pooled)

6. Save all to `paper1/analysis/task6_results.json`.

---

## PHASE 2 — PAPER WRITING

Create `paper1/paper1_v2.tex` as a copy of `paper1.tex`, then apply all edits 
to the v2 file. Do NOT modify `paper1.tex`.

### Task 3: Fix SWE-bench section

The SWE-bench section uses GPT-5 (not GPT-4o), which creates a model mismatch 
with HotpotQA. Now that we've added GPT-5 to HotpotQA (Task 0A), the mismatch
is resolved differently: GPT-5 appears in BOTH benchmarks. GPT-4o also appears 
in HotpotQA, giving us 5 HotpotQA models.

**Steps**:

1. **SWE-bench model alignment**: The SWE-bench pilot already has GPT-5 data.
   Now that GPT-5 is also in HotpotQA, the cross-benchmark comparison works.
   Update the SWE-bench section to explicitly note this:
   - "We compare three of our five HotpotQA models — Claude Sonnet 4.5, GPT-5, 
     and Llama 3.1 70B — on SWE-bench..."

2. **Remove the self-citation**: Replace `\citep{mehta2026swebench}` with an 
   inline description. Change:
   > "All models use an identical bash-only agent scaffold \citep{mehta2026swebench}."
   
   to:
   > "All models use an identical bash-only agent scaffold: a ReAct loop that 
   > issues bash commands, observes stdout/stderr, and iterates until submitting 
   > a patch or reaching the step limit."

3. **Overclaiming fixes** — Find and replace throughout `paper1_v2.tex`:
   - "confirming that behavioral inconsistency is a general property" 
     → "consistent with the hypothesis that behavioral inconsistency generalizes"
   - "The HotpotQA consistency hierarchy replicates exactly" 
     → "The HotpotQA consistency hierarchy is preserved"
   - "cross-benchmark validation" (in section titles and body) 
     → "cross-benchmark pilot study"
   - "100% unique action sequences, confirming that..." 
     → "100% unique action sequences, consistent with the hypothesis that..."

4. **Add SWE-bench limitation sentence**: In the SWE-bench paragraph, after the 
   existing results, add:
   > "Given the small sample (10 tasks, single repository), these results should 
   > be interpreted as a preliminary replication rather than definitive 
   > cross-benchmark validation. We selected astropy tasks as they span diverse 
   > issue types (modeling, I/O, formatting) within a well-maintained scientific 
   > computing library. Expansion to additional repositories is planned for 
   > future work."

5. **Define CV on first use**: The CV metric first appears in the SWE-bench 
   table. Add a definition before or in the table caption:
   > "CV (coefficient of variation) = standard deviation / mean of step counts 
   > across runs, expressed as a percentage."

### Task 4: Expand temperature ablation in paper

**Steps**:
1. Replace the current Table 5 (2-row, Llama-only) with an inline reference 
   to a line plot and an appendix table.

2. Create a **line plot** (Figure Z): 
   - x-axis: temperature {0.0, 0.3, 0.7}
   - y-axis (left): mean unique sequences per task
   - y-axis (right): mean accuracy
   - One line per model (5 lines)
   - Save as `paper1/temperature_ablation.png` and `.pdf`

3. In the main body, keep one short paragraph + the figure. Move the full 
   5-model × 3-temp table to the appendix.

4. Update the Discussion recommendation from "lower temperature settings may 
   be preferable" to a model-specific observation: "The accuracy-temperature 
   tradeoff varies by model: [describe based on data]."

### Task 7: Fix overclaiming language throughout

Go through `paper1_v2.tex` section by section and apply these rules:

**RULE A — No causal language without causal identification:**
- "consistency predicts correctness" → "consistency correlates with correctness"
- "variance predicts failure" → "variance is associated with failure"
- "predicts" → "is associated with" (everywhere in observational context)
- Apply in: abstract, intro contribution list, section 4.2 header, conclusion

**RULE B — No "general property" claims from limited benchmarks:**
- Any "general property of LLM agents" → "across the benchmarks studied"
- Any "confirming that" (for observational evidence) → "consistent with"

**RULE C — Model recommendation must be hedged:**
- Replace: "Claude Sonnet 4.5 may be preferable for applications requiring reliability"
- With: "In our evaluation, Claude Sonnet 4.5 exhibited both the highest 
  accuracy and highest consistency, though we cannot disentangle whether this 
  reflects model capability, training methodology, or other factors."

**RULE D — Bimodal finding:**
- The claim "58% of tasks achieve 100% correctness... while 22% achieve below 50%"
  appears once and is never used again. ADD a small histogram figure showing 
  the bimodal distribution of per-task accuracy in the appendix, and reference 
  it inline: "The distribution of per-task correctness is notably bimodal 
  (Appendix Figure X)." If page budget is tight, remove the sentence entirely.

### Task 8: Final paper assembly

After completing all analysis and writing tasks:

1. **Update abstract** to reflect:
   - 5 models (not 3), 10,000 HotpotQA runs (not 3,000), 200 questions (not 100)
   - Difficulty-stratified result (Task 1 finding)
   - Failure detection precision/recall (Task 2 finding)
   - Temperature ablation across all models (Task 4)
   - Softened SWE-bench claim (Task 3)
   - Replace "predicts" with "correlates with" (Task 7)

2. **Update contributions list** (Section 1) to include:
   - Updated scale: "10,000 runs across 200 questions and 5 models"
   - New contribution: "Within difficulty strata, the consistency-correctness 
     gap [persists / shrinks], [supporting / qualifying] a difficulty-independent 
     consistency effect."
   - New contribution: "Consistency metrics detect task failure with [X]% 
     precision and [Y]% recall (AUROC: [Z]), outperforming random baseline 
     by [W]pp."
   - Updated: "Cross-benchmark pilot study on SWE-bench..."

3. **Add new subsections**:
   - Section 4.2 (after current consistency-correctness): Add Task 1 paragraph 
     + compact table (or inline numbers)
   - Section 4.X (new): "Consistency as a Runtime Failure Detector" — Task 2 
     results with a compact table for Llama + one sentence on other models

4. **Check number consistency**: Every number in the abstract must appear in 
   a table or figure in the body. Run a manual check.

5. **Model names**: Pick one convention and apply throughout:
   - "Claude Sonnet 4.5" (not "Claude 4.5 Sonnet")
   - "GPT-4o" (always — even though Cortex calls it gpt-4.1, the paper uses GPT-4o 
     since existing data was from the GPT-4o API)
   - "GPT-5" (always)
   - "Llama 3.1 70B" (always)
   - "Gemini 3 Pro" (always)

6. **Update Table 1** (overall model comparison): Now 5 models, 200 questions.

7. **Update all figures**: Regenerate:
   - `unique_sequences_histogram.png`: now with 5 models
   - `correctness_comparison.png`: now with 5 models
   - `rankings_instability.png`: now with 5 models (re-run `rankings_analysis.py` 
     with the expanded data)
   - NEW: `temperature_ablation.png` (Task 4)

8. **Impact Statement**: Replace boilerplate with:
   > "Our findings have direct implications for practitioners who rely on 
   > single-run benchmark scores to select agent models: we show that such 
   > evaluations produce incorrect rankings [X]% of the time. For high-stakes 
   > agent deployments, monitoring behavioral consistency across parallel 
   > executions could serve as a lightweight failure detection mechanism 
   > with [precision]% precision. We do not foresee negative societal 
   > consequences from this work."

9. **Statistical note**: Add to Section 3 (Methodology):
   > "All statistical tests are two-sided. We report raw p-values without 
   > multiple-comparison correction given the exploratory nature of this study; 
   > findings should be interpreted accordingly."

10. **Metric justification**: Add to Section 3.3 (Correctness definition):
    > "We use fuzzy matching because HotpotQA answers often include articles 
    > or minor phrasing variations; exact match would penalize correct answers 
    > with surface-form differences. Results under exact match and token F1 
    > are reported in Appendix A."

### Task 9: Page budget

**TARGET**: 6 pages main body (workshop limit) + appendix (unlimited).

**Main body — include only:**
- Task 1 result: one short paragraph + compact 2-column table 
  (model | gap_within_easy | gap_within_hard | partial_r)
  — show only 3 representative models (Llama, Claude, Gemini) if 5 is too wide
- Task 2 result: one 4-row table (feature | precision | recall | AUROC) for 
  Llama only, with one sentence noting other models
- Task 4 result: one line plot figure only (temperature vs metrics, 5 models); 
  full table in appendix
- All text fixes (Tasks 3, 7) are space-neutral rewrites

**Appendix — move these:**
- Full 5-model difficulty stratification table
- Full precision/recall sweep across all features × thresholds × models
- Full temperature ablation table (5 models × 3 temps × all metrics)
- Metric sensitivity analysis (EM / F1 / fuzzy comparison)
- SWE-bench cross-benchmark comparison table (current Table 4)
- Question-type analysis (current Table 5) if over page limit
- Bimodal distribution figure (Task 7 Rule D)

**After all edits, compile and verify:**
- [ ] `pdflatex paper1_v2.tex` compiles cleanly
- [ ] No undefined references, missing figures, or LaTeX warnings
- [ ] Page count ≤ 6 main body (excluding references and appendix)
- [ ] All new results cited in main body even if tables are in appendix
- [ ] Appendix is self-contained with section labels and captions

---

## EXECUTION ORDER

```
Phase 0 (experiments — can run in parallel, ALL VIA SNOWFLAKE CORTEX):
  0-PREREQ: Add snowflake_cortex provider to agent.py + runner.py
  0A: GPT-5 on 100 questions (1,000 runs via Cortex)
  0B: Gemini 3 Pro on 100 questions (1,000 runs via Cortex)
  0C: All 5 models on questions 100-199 (5,000 runs via Cortex)
  0D: Temperature ablation expansion (1,000 runs via Cortex)

Phase 1 (analysis — sequential, after Phase 0):
  1: Difficulty stratification
  2: Failure detection
  5: Metric sensitivity
  6: Statistical reporting

Phase 2 (writing — after Phase 1):
  Copy paper1.tex → paper1_v2.tex
  3: SWE-bench fixes
  4: Temperature ablation writeup + figure
  7: Overclaiming language fixes
  8: Final assembly (abstract, contributions, figures, tables)
  9: Page budget check + appendix
  Compile and verify
```

---

## SUCCESS CRITERIA

After all tasks are complete, the paper should satisfy ALL of the following:

- [ ] **NO direct API keys used** — all new compute via Snowflake Cortex
- [ ] 5 models across 4 providers (Llama, GPT-4o, GPT-5, Claude, Gemini 3 Pro)
- [ ] 200 questions × 10 runs × 5 models = 10,000 HotpotQA runs
- [ ] Consistency-correctness claim shown to hold within difficulty strata 
      (or explicitly qualified if it does not)
- [ ] Partial correlation r and p-value reported
- [ ] Precision/recall table validates consistency as a failure detector
- [ ] Temperature ablation covers all 5 models at 3 temperatures
- [ ] SWE-bench section uses honest "pilot study" language
- [ ] SWE-bench self-citation removed, replaced with inline description
- [ ] CV metric defined on first use
- [ ] All major comparisons have effect sizes and confidence intervals
- [ ] No causal language without causal identification
- [ ] Correctness metric sensitivity analysis in appendix
- [ ] Abstract accurately reflects the body of the paper
- [ ] All 5 models named consistently throughout
- [ ] The paper compiles cleanly at ≤ 6 pages + appendix
- [ ] `paper1.tex` is unchanged
- [ ] `paper1_v2.tex` contains all improvements
