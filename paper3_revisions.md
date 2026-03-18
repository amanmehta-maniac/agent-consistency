# Paper 3 Revisions: Post-Review Action Items

**Source**: Reviewer agent feedback (March 17, 2026)
**Paper**: `paper3_prep/main.tex`
**Current estimated acceptance**: 65–75%
**Target after revisions**: 80–85%

---

## Architecture: Parallel Execution Groups

All tasks are organized into four groups. **Groups A, B, and C can run
fully in parallel.** Group D depends on outputs from A, B, C.

```
┌─────────────────────────────────────────────────────────┐
│                  RUN IN PARALLEL                        │
│                                                         │
│  GROUP A (text edits)    GROUP B (analyses)   GROUP C   │
│  ┌──────────────────┐   ┌────────────────┐   ┌───────┐ │
│  │ A1: Steering     │   │ B1: t-SNE/PCA  │   │ C1:   │ │
│  │ A2: Layer-0      │   │ B2: Phi-3 step │   │ Token │ │
│  │ A3: CIs          │   │ B3: Figure 3   │   │ count │ │
│  │ A4: n=99         │   │     update     │   │ ctrl  │ │
│  │ A5: StrategyQA   │   └────────────────┘   │ expt  │ │
│  │ A6: Self-citation│                        └───────┘ │
│  │ A7: Bib entry    │                                   │
│  │ A8: Phi-3 text   │                                   │
│  └──────────────────┘                                   │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  GROUP D (integration) │
              │  D1: Intervention tbl  │
              │  D2: Abstract update   │
              │  D3: Limitations       │
              │  D4–D6: Add figures    │
              │  D7: Final compile     │
              └───────────────────────┘
```

---

## GROUP A — Text Edits (all parallelizable, no compute needed)

All A tasks edit `paper3_prep/main.tex` (or `references.bib`). They
touch different sections and can be done by separate agents in parallel.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### A1 — Reframe steering section mechanistically
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Section**: 4.8 "Preliminary Steering Experiment" (lines 199–203)

**Problem**: The current text says "the commitment signal is distributed
across many layers, making single-layer interventions crude." This is a
weak, hand-wavy explanation. A reviewer will ask: "If steering doesn't
work, does that weaken the causal interpretation?" The current text
leaves this question unanswered.

**Fix**: Replace the final two sentences of the steering paragraph
(starting "We interpret this as expected...") with a mechanistic
explanation for why prompting works but steering doesn't:

```latex
We interpret this as reflecting a fundamental asymmetry between
prompting and activation interventions in the multi-step setting.
The prompting intervention succeeds because it changes what the model
\emph{generates} at step~3: the explicit commitment text becomes part
of the context window, shaping how hidden states evolve across
\emph{all} layers in subsequent forward passes. Steering at a single
layer at step~4 intervenes too late and too locally---it statically
shifts representations at one point in the network after the
commitment juncture has already passed, rather than changing the
information flow through the full forward pass during the critical
step. This suggests that representational commitment is not a
localized activation pattern amenable to single-site perturbation,
but a distributed computational process that unfolds across layers
and steps. Multi-layer, multi-step steering \citep{zou2023repe}
or representation finetuning may be required to achieve
activation-level consistency improvements.
```

**Success criteria**:
- [ ] Section 4.8 rewritten with mechanistic explanation
- [ ] Reads as "the failure is theoretically informative" not "we tried
      and failed"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### A2 — Explain the layer-0 positive correlation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Section**: 4.5 "Cross-Model Validation" (near line 138) or 4.1
(near line 89)

**Problem**: Table `tab:crossmodel_full` (Appendix C, line 298) shows:
- Llama layer 0: r = +0.47, p < .001
- Qwen layer 0: r = +0.49, p < .001

Both are highly significant **positive** correlations — the opposite
sign of the main finding. The paper never mentions or explains them.
A reviewer will ask: "Does this confound the interpretation at deeper
layers?"

**Fix**: Add a footnote at the first mention of the cross-model
layer-wise results. Attach it to the sentence on line 138 that begins
"All sampled layers 4--36 are significant..." or to the cross-model
table reference.

```latex
\footnote{The positive correlation at layer~0 ($r = +0.47$ for Llama,
$+0.49$ for Qwen) reflects input-level similarity rather than
representational commitment. At the embedding layer, hidden states
are determined almost entirely by the input tokens, which are
identical across runs of the same question. Higher embedding-layer
similarity therefore indexes questions whose token distributions
are more concentrated (typically shorter, simpler questions), which
also tend to produce more consistent behavior. This positive
correlation is an expected artifact of measuring similarity at the
input layer and does not confound the negative correlations at deeper
layers, where representations have been transformed by the model's
computations. Phi-3 does not show this pattern ($r = -0.05$,
$p = .63$), likely due to its different tokenizer and embedding
structure.}
```

**Success criteria**:
- [ ] Layer-0 positive correlation explicitly acknowledged and explained
- [ ] Explanation clearly distinguishes it from the deep-layer signal
- [ ] Phi-3's absence of this artifact noted

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### A3 — Add 95% CIs to all correlations in contributions list
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Section**: Introduction, contributions list (lines 47–55)

**Problem**: Contributions item 4 reports Phi-3 correlations without CIs:
`r = -0.36 at step~4, r = -0.58 at step~5`

But items 1 and 5 include CIs. Selective CI reporting looks like
cherry-picking. The CIs already exist in the abstract (line 33).

**Fix**: Update contribution item 4 (line 51) to:

```latex
\item Cross-model validation on Qwen~2.5~72B ($r = -0.65$, 95\% CI
$[-0.74, -0.55]$) and Phi-3-Medium-14B ($r = -0.36$, 95\% CI
$[-0.52, -0.17]$ at step~4; $r = -0.58$, 95\% CI $[-0.72, -0.41]$
at step~5), a structurally different 14B model with 40 layers and
$d{=}5120$, confirms the signal generalizes across model families
and scales.
```

Also check contributions items 6 and 7 — if any correlation is stated
without a CI, add it. Every `r = X` in the contributions list must have
a CI.

**Success criteria**:
- [ ] All r values in contributions list have 95% CIs
- [ ] Notation is consistent across all items

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### A4 — Clarify n=99 vs. n=100 discrepancy
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Sections**: Figure 1 caption (line 94), Methods (line 71)

**Problem**: Figure 1 caption says n=99. Methods says 100 questions, 988
trajectories, 12 excluded. The n=99 is presumably one question whose
runs all terminated before step 4. This is never explained. A reviewer
will notice.

**Fix 1** — Update Figure 1 caption (line 94):

```latex
\caption{Pearson $r$ between activation similarity and behavioral CV
across steps and layers ($n = 99$; one question excluded at step~4 as
all runs terminated before this step). Gold borders indicate
$p < 0.05$. The signal concentrates at step~4 across layers 32--80.}
```

**Fix 2** — Add clarification to Methods section (end of paragraph on
line 71), after "yielding 988 total trajectories":

```latex
At step~4 (the primary analysis step), 99 of 100 questions have
sufficient data; one question's runs all terminated by step~3.
```

**Success criteria**:
- [ ] n=99 explained in Figure 1 caption
- [ ] n=99 explained in Methods
- [ ] No unexplained discrepancy between "100 questions" and "n=99"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### A5 — Address StrategyQA accuracy confound explicitly
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Section**: 4.6 "Cross-Benchmark Generalization: StrategyQA" (line 153)

**Problem**: The reviewer asks: "The signal is dramatically stronger on
StrategyQA (−0.83 vs. −0.35). If the uniform high accuracy on StrategyQA
(93.2%) explains the stronger signal, that's just measuring difficulty,
not commitment."

The paper already has the answer — the partial correlation controlling
for accuracy is unchanged (r = −0.83 → r = −0.83) — but it's stated
blandly without connecting it to this obvious concern.

**Fix**: After the sentence "The partial correlation controlling for
accuracy is virtually unchanged ($r = -0.83$, $p < 10^{-13}$), as
expected given the high overall accuracy (93.2\%)." add:

```latex
This rules out the concern that the stronger StrategyQA signal is
merely an artifact of higher accuracy: if accuracy were driving the
correlation, controlling for it would attenuate the signal, as it does
on HotpotQA (raw $r = -0.35 \to$ partial $r = -0.45$). The stability
of the partial correlation ($r = -0.83 \to -0.83$) indicates that the
stronger signal reflects a genuine difference in how commitment
manifests on shorter reasoning chains---the compressed trajectory means
the commitment juncture is sharper and less noisy, producing a cleaner
signal rather than an accuracy artifact.
```

**Success criteria**:
- [ ] StrategyQA accuracy confound explicitly addressed
- [ ] Partial correlation stability framed as evidence against the
      confound
- [ ] Mechanistic explanation for why shorter chains → stronger signal

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### A6 — Restructure self-citations in Introduction
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Section**: Introduction (lines 40–44)

**Problem**: The reviewer notes: "The paper cites anonymous2026a and
anonymous2026b extensively as the motivating foundation. COLM reviewers
are well aware of the de-anonymization risk when a single author cites
two 'anonymous' papers about the same niche topic." The Introduction
currently opens with these citations as established fact.

**Fix**: Rewrite the first two paragraphs of the Introduction to
establish behavioral inconsistency from first principles, citing
anonymous works only for specific quantitative claims rather than
as the conceptual foundation.

Replace lines 40–42 with:

```latex
LLM-based agents---systems combining language models with tool use
and multi-step reasoning \citep{react, cot}---face a critical
reliability concern: \emph{behavioral inconsistency}. When given
the same task multiple times with non-zero sampling temperature,
agents produce different action sequences, explore different
reasoning paths, and often reach different conclusions. This is not
merely a theoretical concern: production agent systems are typically
evaluated on single runs, yet the same system can produce correct
answers on one run and incorrect answers on another for the same
input. Inconsistency complicates deployment, debugging, and
benchmarking.

Recent work has begun to characterize this phenomenon. Multi-run
evaluation reveals that the majority of agent divergence occurs at
early decision points such as the first search query
\citep{anonymous2026a}, and that behavioral consistency amplifies
outcomes---both positive and negative---rather than guaranteeing
correctness \citep{anonymous2026b}. But these studies measure what
agents \emph{do} without examining what happens \emph{inside} the
model. A deeper question remains: does behavioral consistency have
a measurable signature in the model's internal representations?
```

This version establishes the inconsistency problem from first
principles (observable from any agent deployment), then cites the
anonymous works for specific findings. A reviewer who has never read
Papers 1 or 2 can follow the motivation.

**Also check**: Related Work section (lines 61–65). Same principle:
attribute specific numbers to the citations, but don't frame them as
the conceptual foundation of Paper 3.

**Success criteria**:
- [ ] Introduction establishes behavioral inconsistency from first
      principles without depending on anonymous citations
- [ ] anonymous2026a and anonymous2026b cited only for specific
      quantitative findings, not conceptual framing
- [ ] A reviewer unfamiliar with Papers 1 and 2 can fully understand
      the motivation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### A7 — Fix Paper 1 bib entry: add arXiv ID
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**File**: `paper3_prep/references.bib` (lines 1–6)

**Problem**: The original improvement plan (Task 7) specified that
Paper 1 is already public on arXiv at arxiv.org/abs/2602.11619. The
current bib entry says `journal={Manuscript under review}` with no
arXiv ID. COLM allows citing your own anonymous arXiv paper with the
arXiv ID — this is standard practice and explicitly permitted.

**Fix**: Update the bib entry (lines 1–6 of `references.bib`):

```bibtex
@article{anonymous2026a,
  title={When Agents Disagree With Themselves: Measuring Behavioral
         Consistency in {LLM}-Based Agents},
  author={Anonymous},
  journal={arXiv preprint arXiv:2602.11619},
  year={2026}
}
```

Leave `anonymous2026b` as-is ("Manuscript in preparation") since
Paper 2 is not yet public.

**Success criteria**:
- [ ] Paper 1 bib entry includes arXiv ID
- [ ] Paper 2 bib entry left as manuscript in preparation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### A8 — Strengthen Phi-3 step-5 disambiguation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Section**: 4.5 "Cross-Model Validation" (line 138) and Discussion
(line 215)

**Problem**: The reviewer asks: "Is the step 5 peak for Phi-3 (r=−0.58)
a replication of the commitment signal at a different step, or a
different phenomenon? How do you distinguish?"

The paper currently says "the one-step delay plausibly reflecting the
smaller model needing additional evidence before committing" but doesn't
provide evidence distinguishing it from an alternative explanation.

**Fix**: Add one sentence after the Phi-3 discussion in Section 4.5
(after "the one-step delay plausibly reflecting..."):

```latex
Two pieces of evidence support this as a delayed commitment signal
rather than a distinct phenomenon: (1)~the step-4-to-step-5 shift
mirrors the step-3-to-step-4 shift seen between StrategyQA and
HotpotQA for Llama, where the commitment juncture adjusts to the
amount of evidence needed; and (2)~the layer profile at Phi-3's
step~5 ($r = -0.58$ at layer~4, all layers 4--32 significant)
exhibits the same monotonic deepening pattern as Llama's step~4,
suggesting the same underlying process operating one step later.
```

In the Discussion section (line 215), after "Phi-3's one-step-later
peak ($r = -0.58$ at step~5 vs.\ step~4 for the 70B models) may
reflect smaller models requiring additional evidence accumulation
before committing," add:

```latex
We note that this interpretation is consistent with, but not proven
by, the current data; a definitive test would require varying the
amount of evidence available at each step while controlling for model
scale.
```

**Success criteria**:
- [ ] Phi-3 step-5 peak explicitly argued as delayed commitment
- [ ] Evidence cited (parallel to StrategyQA shift, similar layer
      profile)
- [ ] Limitation of the interpretation honestly noted

---

## GROUP B — Analyses on Existing Data (all parallelizable, no new experiments)

All B tasks use data that already exists. They can run in parallel with
each other and with Groups A and C.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### B1 — t-SNE/PCA visualization of commitment categories
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Reviewer quote**: "Should add before submission... activation
difference analysis... e.g., t-SNE or PCA of step-4 hidden states
colored by commitment category. This directly supports the
'correctness-agnostic commitment' claim."

The reviewer says this is what pushes acceptance to 85%+.

**Data source**: Existing Llama-3.1-70B hidden states.
- 100 questions, ~988 trajectories
- Hidden states at step 4, layer 40
- Data locations: `hotpotqa/pilot_hidden_states_70b/` (20q),
  `hotpotqa/results_easier/` (20q),
  `hotpotqa/experiment_60q_results/` (60q)

**Analysis steps**:

1. For each question, load hidden states at step 4, layer 40 for all
   available runs. Compute the mean hidden state across runs → one
   vector per question ∈ R^8192.

2. Classify each question into commitment categories (use the same
   thresholds as Section 4.4 of the paper):
   - **Committed-correct**: accuracy ≥ 0.8
   - **Committed-wrong**: accuracy ≤ 0.2 AND CV ≤ 0.15
   - **Uncommitted-wrong**: accuracy ≤ 0.2 AND CV > 0.15
   - **Mixed**: everything else

3. Dimensionality reduction:
   - PCA to 50 dimensions first (standard preprocessing for t-SNE)
   - t-SNE to 2D: try perplexity ∈ {10, 15, 20, 30}, pick the one
     that produces the clearest separation
   - Also try UMAP (n_neighbors=15, min_dist=0.1) as an alternative
   - Produce both plots, pick the more informative one

4. Color points by category:
   - Committed-correct: green (#2ecc71)
   - Committed-wrong: red (#e74c3c)
   - Uncommitted-wrong: orange (#f39c12)
   - Mixed: gray (#95a5a6)
   Use filled circles, size proportional to number of questions in
   each category. Add legend.

5. **Repeat for Qwen** data (`hotpotqa/qwen_cross_model_100q/`) to
   show cross-model consistency of the visualization. Produce a
   two-panel figure (Llama | Qwen) if both show the pattern.

6. **Key test**: Do committed-correct and committed-wrong overlap in
   the projection, distinct from uncommitted-wrong?

7. **If the pattern is clear**: This is a main-body figure. Add as
   a panel alongside the existing commitment categories boxplot
   (Figure 4) or as a new figure.

8. **If the pattern is noisy**: Put in appendix with a note that
   statistical tests (Section 4.4) provide more rigorous evidence
   than 2D projections, which lose high-dimensional structure.

**Output**: Save figure to `paper3_prep/figures/commitment_tsne.pdf`
and `paper3_prep/figures/commitment_tsne.png`.

**Script location**: `hotpotqa/commitment_tsne.py`

**Success criteria**:
- [ ] t-SNE/PCA/UMAP plot produced for Llama
- [ ] t-SNE/PCA/UMAP plot produced for Qwen
- [ ] Visualization confirms or qualifies correctness-agnostic claim
- [ ] Figure saved in paper3_prep/figures/
- [ ] Script saved for reproducibility

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### B2 — Phi-3 step-by-step progression figure
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Reviewer quote**: "Full Phi-3 step-by-step progression figure
(analogous to Figure 2 for Llama), showing when the signal peaks at
step 5 for Phi-3 — this would make the 'one-step delay' claim visually
compelling."

**Data source**: Existing Phi-3 hidden states.
- 100 questions × 10 runs = 1,000 trajectories
- Analysis script: `hotpotqa/phi3_analysis.py`
- Data likely in a Phi-3 results directory (check
  `hotpotqa/phi3_test_results/` and any other phi3 directories)

**Analysis steps**:

1. For each step (1 through 8), compute the correlation between
   activation similarity and behavioral CV at Phi-3's peak layer
   (layer 16). Also compute at layers 4, 8, 12, 20, 24, 28, 32.

2. Plot step progression analogous to Figure 2:
   - x-axis: step number (1–8)
   - y-axis: Pearson r
   - Add bootstrap 95% CI error bars (1,000 iterations)
   - Horizontal dashed line at r = 0

3. **Two-panel or overlay version**: Create a figure with both Llama
   (at layer 40) and Phi-3 (at layer 16) on the same axes:
   - Llama: solid blue line
   - Phi-3: dashed red line
   - Annotate the one-step shift: Llama peaks at step 4, Phi-3 at
     step 5
   - This directly visualizes the delayed commitment claim

4. If there's room, also add a Qwen line (at layer 64) to make it
   a three-model comparison.

**Output**: Save to `paper3_prep/figures/step_progression_phi3.pdf`
and a combined version as
`paper3_prep/figures/step_progression_multi_model.pdf`.

**Script location**: Add to `hotpotqa/phi3_analysis.py` or create
`hotpotqa/phi3_step_progression.py`.

**Success criteria**:
- [ ] Phi-3 step progression plot produced
- [ ] One-step delay visually clear
- [ ] Combined multi-model plot produced
- [ ] Figure saved in paper3_prep/figures/

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### B3 — Update Figure 3 to include Phi-3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Problem**: The current Figure 3 (`cross_model_layer_comparison.png`,
referenced at line 142) shows only Llama and Qwen. The paper discusses
three models but the cross-model figure only shows two.

**Fix**: Regenerate Figure 3 with all three models:
- Llama 3.1 70B: solid blue line
- Qwen 2.5 72B: solid orange line
- Phi-3 14B: dashed green line (dashed because different layer count)

For Phi-3, the x-axis challenge is that it has 40 layers vs. 80 for the
others. Options:
1. Use **proportional depth** on x-axis (0%–100%) instead of absolute
   layer numbers. This is the cleanest approach and directly supports
   the "architecture-dependent peak depth" claim.
2. Plot Phi-3 on a secondary x-axis.

Option 1 (proportional depth) is preferred. Update x-axis label to
"Relative depth (%)" and map:
- Llama layers {0,8,...,80} → {0%, 10%, ..., 100%}
- Qwen layers {0,8,...,80} → {0%, 10%, ..., 100%}
- Phi-3 layers {0,4,8,...,40} → {0%, 10%, ..., 100%}

Add:
- 95% CI shaded bands for each model
- Vertical dashed lines at each model's peak relative depth
- Horizontal dashed line at r = 0
- Stars for p < 0.05

Update the Figure 3 caption to mention all three models:

```latex
\caption{Layer-wise correlation between activation similarity and
behavioral CV at step~4 across three models, plotted by proportional
depth. All models show the negative correlation characteristic of
representational commitment, with architecture-dependent peak depths
(Llama: 50\%, Qwen: 80\%, Phi-3: 40\%). $^*p < 0.05$.}
```

**Data source**: Table `tab:crossmodel_full` already has all the
numbers. The existing figure script is likely in
`hotpotqa/paper3_100q_analysis.py` or `hotpotqa/analyze_cross_model.py`.

**Output**: Save to
`paper3_prep/figures/cross_model_layer_comparison.png` (overwrite) and
`.pdf` version.

**Success criteria**:
- [ ] Figure 3 regenerated with all three models
- [ ] Proportional depth x-axis used
- [ ] Caption updated
- [ ] 95% CI bands included

---

## GROUP C — New Experiment (runs in parallel with A and B)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### C1 — Token-count control experiment for prompting intervention
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Reviewer quote**: *"The most likely reason [for rejection] is the
prompting intervention's token-count confound. A reviewer will write:
'The authors claim that inducing representational commitment improves
consistency, but the control condition differs from the intervention in
both framing and token count. Without a matched-length control, it is
impossible to attribute the CV reduction to the commitment framing rather
than to the additional context.'"*

This is the single highest-impact addition. Without it, the causal
claim is fatally confounded.

### Step 1: Prepare the filler prompt

The existing commitment prompt (from `run_intervention_experiment.py`,
line 47–52):

```python
COMMITMENT_PROMPT = (
    "\n\n[IMPORTANT: Based on the evidence you have gathered so far, commit to a "
    "specific reasoning strategy for solving this question. State your committed "
    "strategy clearly in your next Thought, then follow through with it. Do not "
    "change strategies or start over — build on what you have learned.]"
)
```

Create a semantically neutral filler prompt matched in token count:

```python
FILLER_PROMPT = (
    "\n\n[NOTE: Please continue with the task as you normally would. "
    "Take the time you need to work through the problem. Consider the "
    "information you have gathered and proceed with your next step in "
    "whatever way seems most appropriate to you. There is no particular "
    "urgency — work at your own pace and follow your reasoning.]"
)
```

**Before launching**: Verify token counts are within ±5 tokens by
running both through the Llama-3.1 tokenizer:

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B-Instruct")
print(len(tok.encode(COMMITMENT_PROMPT)))  # should be ~55-65 tokens
print(len(tok.encode(FILLER_PROMPT)))       # must be within ±5
```

Adjust the filler text if needed to match within ±5 tokens.

The filler prompt must:
- NOT mention commitment, strategy, or reasoning direction
- NOT tell the model to change its approach
- Be generically encouraging/neutral
- Be the same length as the commitment prompt

### Step 2: Implement the filler condition

In `hotpotqa/run_intervention_experiment.py`:

1. Add `FILLER_PROMPT` constant after `COMMITMENT_PROMPT` (around
   line 52)

2. Update the condition logic at line 278–281:

```python
if step_num == self.intervention_step and action != "Finish":
    if self.condition == "commitment":
        observation = observation + COMMITMENT_PROMPT
    elif self.condition == "filler":
        observation = observation + FILLER_PROMPT
```

3. Update the argparse `--condition` choices to include `"filler"`.

### Step 3: Run the experiment

```bash
python run_intervention_experiment.py \
    --questions /data/qwen_100q_full.json \
    --output /results/intervention_experiment_filler \
    --condition filler
```

- Same 100 HotpotQA questions as existing intervention experiment
- Same model: Llama-3.1-70B
- Same temperature: T=0.5
- Same scaffold: ReAct agent, same tools
- 10 runs per question = 1,000 trajectories
- Extract hidden states identically to control and commitment conditions
- Deploy on 8×H200 GPUs via K8s (same config as existing intervention)
- Store results in `hotpotqa/intervention_experiment_filler/`

### Step 4: Analysis

Create `hotpotqa/analyze_filler_intervention.py` (or extend
`hotpotqa/analyze_intervention.py`).

Compute for all three conditions (control, filler, commitment):

1. **Per-question metrics**:
   - Behavioral CV (step count variance across 10 runs)
   - Action sequence diversity (unique action sequences / 10)
   - Accuracy (fraction of 10 runs correct)
   - Activation similarity at step 4, layer 40

2. **Three-way paired comparisons** (100 paired observations each):
   - Control vs. Filler: paired t-test + Wilcoxon for CV, diversity, acc
   - Control vs. Commitment: same (replicates existing results)
   - **Filler vs. Commitment**: same (THE KEY TEST)
   - Report: mean difference, Cohen's d, p-value for each pair

3. **Critical question**: Does the commitment prompt reduce CV
   significantly more than the filler prompt?
   - If yes (commitment CV < filler CV, p < 0.05): The commitment
     framing does real work beyond token count. This is the ideal result.
   - If marginal (p = 0.05–0.15): Report as suggestive evidence.
     Consider a one-sided test since we have a directional hypothesis.
   - If no (commitment ≈ filler): Report honestly. The CV reduction is
     from additional context, not commitment framing. This changes the
     causal interpretation but is still a finding.

4. **Also compute**: Does filler itself reduce CV vs. control?
   - If filler also reduces CV (but less than commitment): Both token
     count and framing matter. Report both effects.
   - If filler does not reduce CV: Clean result — commitment framing is
     entirely responsible.

5. **Stratified analysis**: Repeat the three-way comparison within
   CV tertiles (same tertile boundaries as existing analysis).
   Does the commitment-vs-filler difference concentrate in the
   high-CV tertile?

### Step 5: Paper updates (these go in Group D)

See D1, D2, D3 below.

**Success criteria**:
- [ ] Filler prompt verified to be ±5 tokens of commitment prompt
- [ ] Filler condition implemented in run_intervention_experiment.py
- [ ] 1,000 filler trajectories complete with hidden states
- [ ] Three-way statistical comparison complete (control/filler/commit)
- [ ] Stratified three-way analysis complete
- [ ] Results reported honestly regardless of outcome

---

## GROUP D — Paper Integration (depends on A, B, C outputs)

These tasks integrate results from Groups A–C into the paper. They
should run after the relevant upstream tasks complete. D1–D3 depend on
C1. D4–D6 depend on B1–B3. D7 depends on everything.

**D1–D3 can run in parallel. D4–D6 can run in parallel. D7 runs last.**

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### D1 — Update intervention table and text with three-way results
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Depends on**: C1 (filler experiment complete)

**Section**: 4.7 "Intervention: Inducing Representational Commitment"
(lines 172–197)

**Updates required**:

1. Replace Table `tab:intervention` (lines 179–193) with a three-way
   table:

```latex
\begin{table}[h]
\centering
\caption{Prompting intervention results ($n = 100$ questions, 10 runs
each). The commitment prompt reduces CV beyond what matched-length
filler text achieves, isolating the commitment framing from token-count
effects.}
\label{tab:intervention}
\small
\begin{tabular}{lccccc}
\toprule
\textbf{Metric} & \textbf{Control} & \textbf{Filler} &
\textbf{Commitment} & \textbf{$\Delta_{\text{C-F}}$} & \textbf{$p$} \\
\midrule
Behavioral CV & 0.112 & [?] & 0.094 & [?] & [?] \\
Action seq.\ diversity & 0.250 & [?] & 0.223 & [?] & [?] \\
Accuracy & 0.835 & [?] & 0.823 & [?] & [?] \\
\bottomrule
\end{tabular}
\end{table}
```

Fill in [?] values from C1 analysis. The Δ column should be
Commitment minus Filler (the key comparison). The p column should be
for the Commitment vs. Filler paired test.

2. Add one sentence after the table:

```latex
A matched-length filler control (``Please continue with the task as
you normally would...''), appended at the same step with the same
token count ($\pm$5 tokens), [achieves/does not achieve] significant
CV reduction relative to unmodified control ([filler stats]).
The commitment prompt reduces CV [significantly/marginally] more
than filler ($\Delta = $ [value], $p = $ [value]), [confirming/
suggesting] that the commitment framing, not additional context
length, drives the consistency improvement.
```

3. Update the stratified analysis paragraph (line 197) to include the
   three-way stratified comparison if the results support it.

**If filler ≈ commitment**: Rewrite the intervention section to reframe
the contribution. Change "inducing representational commitment" to
"providing additional context at the commitment juncture." Update the
discussion accordingly. This changes the causal story but is still a
valid finding.

**Success criteria**:
- [ ] Three-way table in paper with all values filled
- [ ] Filler control described and results reported
- [ ] Causal interpretation calibrated to the actual results

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### D2 — Update abstract with filler control result
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Depends on**: C1, D1

**Section**: Abstract (lines 33)

Currently the abstract says:

> "A prompting intervention that induces explicit commitment at step~3
> reduces behavioral CV by 16% ($p = .037$, $d = 0.21$) without
> affecting accuracy"

Update to include the filler control result. If commitment > filler:

```latex
A prompting intervention that induces explicit commitment at step~3
reduces behavioral CV by 16\% ($p = .037$, $d = 0.21$) relative to
unmodified control and by [X]\% relative to a matched-length filler
control ($p = $ [Y]), confirming that the commitment framing---not
additional context---drives the effect.
```

If commitment ≈ filler, adjust language accordingly to avoid
overclaiming.

**Success criteria**:
- [ ] Abstract accurately reflects three-way results
- [ ] No overclaiming about commitment vs. context effects
- [ ] Every number in abstract appears in a table in the body

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### D3 — Update Limitations section
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Depends on**: C1

**Section**: Discussion, Limitations paragraph (line 217)

Currently says:

> "The prompting intervention is confounded by additional tokens in the
> context window; activation-level steering would provide cleaner causal
> evidence but remains technically challenging (Section~4.8)."

**If commitment > filler**: Remove this sentence entirely — the confound
has been addressed. Replace with:

```latex
While a matched-length filler control rules out token-count confounds
for the prompting intervention, the intervention still modifies the
context window rather than directly manipulating activations;
activation-level steering remains an open challenge (Section~4.8).
```

**If commitment ≈ filler**: Update to reflect the finding:

```latex
A matched-length filler control shows that the CV reduction from the
prompting intervention is partially attributable to additional context
tokens rather than the commitment framing specifically
(Section~\ref{sec:intervention}); the precise contribution of
semantic content vs.\ context length remains an open question.
```

**Success criteria**:
- [ ] Limitations section accurately reflects filler control results
- [ ] Token-count confound either resolved or honestly characterized

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### D4 — Add t-SNE/PCA figure to paper
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Depends on**: B1

Add the t-SNE/PCA figure produced by B1 to the paper.

**If clear clustering**: Add as a panel in Figure 4 or a new main-body
figure. Add to Section 4.4 after the commitment categories boxplot:

```latex
Figure~\ref{fig:commitment_tsne} provides complementary evidence via
dimensionality reduction: committed-correct and committed-wrong
questions occupy overlapping regions in the t-SNE projection of
step-4 hidden states, while uncommitted-wrong questions cluster
separately, visualizing the correctness-agnostic nature of
representational commitment.
```

**If noisy**: Add to appendix with a note:

```latex
Appendix~\ref{app:tsne} shows a t-SNE projection of step-4 hidden
states; while the statistical tests in Section~4.4 provide more
rigorous evidence, the projection shows a qualitative trend consistent
with correctness-agnostic commitment.
```

**Success criteria**:
- [ ] Figure integrated into paper at appropriate location
- [ ] Caption written
- [ ] Reference added to Section 4.4 text

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### D5 — Add Phi-3 step progression figure to paper
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Depends on**: B2

Add the Phi-3 step progression figure to the paper.

**Preferred placement**: As a panel in Figure 2 or in the appendix
with a reference from Section 4.5.

If adding to the appendix, create a new appendix section:

```latex
\section{Phi-3 Step Progression}
\label{app:phi3_steps}

Figure~\ref{fig:phi3_steps} shows the step-by-step correlation profile
for Phi-3-Medium-14B at layer~16 (peak layer). The signal peaks at
step~5 rather than step~4, confirming the one-step delay relative to
the 70B models.

\begin{figure}[h]
\centering
\includegraphics[width=0.7\columnwidth]{figures/step_progression_multi_model.pdf}
\caption{Step-by-step correlation between activation similarity and
behavioral CV at each model's peak layer. Phi-3 (dashed) peaks one step
later than Llama (solid), consistent with smaller models requiring
additional evidence before committing.}
\label{fig:phi3_steps}
\end{figure}
```

Add reference in Section 4.5 (around line 147):

```latex
The one-step delay in Phi-3 is visible in the step-by-step
correlation profile (Appendix~\ref{app:phi3_steps}).
```

**Success criteria**:
- [ ] Figure integrated into paper (main body or appendix)
- [ ] Caption written
- [ ] Cross-reference from Section 4.5

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### D6 — Update Figure 3 in paper
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Depends on**: B3

Replace the existing Figure 3 with the updated three-model version.

1. Replace the `\includegraphics` reference (line 142) — the file is
   already overwritten by B3.

2. Update the caption (lines 143–144):

```latex
\caption{Layer-wise correlation between activation similarity and
behavioral CV at step~4 across three models, plotted by proportional
depth. All models show the negative correlation characteristic of
representational commitment, with architecture-dependent peak depths
(Llama: 50\%, Qwen: 80\%, Phi-3: 40\%). Shaded bands show 95\% CIs.
$^*p < 0.05$.}
```

3. If using proportional depth x-axis, update the filename if needed
   (or keep as `cross_model_layer_comparison.png` since B3 overwrites
   it).

**Success criteria**:
- [ ] Figure 3 shows all three models
- [ ] Caption updated
- [ ] Proportional depth axis makes the comparison intuitive

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### D7 — Final compile, page check, and consistency audit
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Depends on**: All of D1–D6, all of A1–A8

This is the final integration pass. Run after all other tasks complete.

**Compile check**:
```bash
cd paper3_prep && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```
- Zero errors
- Zero warnings (or only harmless ones like hbox warnings)

**Page count**: Must be ≤ 9 pages for main body (before references).
If over, move material to appendix in this priority order:
1. Detailed steering experiment (keep one paragraph summary)
2. Full cross-model layer-wise table (keep Figure 3)
3. StrategyQA detailed heatmap (keep summary paragraph)

**Consistency audit**:
- [ ] Every number in the abstract appears in a table or figure
- [ ] Every r value in contributions list has a 95% CI
- [ ] n is reported in every table and figure caption
- [ ] All p-values are two-sided
- [ ] No author-identifying text (search for "Mehta", "Snowflake",
      "aman", any personal GitHub URLs)
- [ ] `anonymous2026a` bib entry has arXiv ID
- [ ] Layer-0 positive correlation explained (footnote present)
- [ ] Steering section has mechanistic explanation
- [ ] Filler control results in intervention table
- [ ] StrategyQA accuracy confound addressed
- [ ] Phi-3 step-5 disambiguation text present
- [ ] n=99 explained in Figure 1 caption and Methods
- [ ] Limitations section reflects current state of evidence
- [ ] LLM usage disclosure present (line 229)

**Success criteria**:
- [ ] Paper compiles cleanly
- [ ] ≤ 9 pages main body
- [ ] All audit items pass

---

## Summary: What each task preempts

| Reviewer attack | Blocked by |
|----------------|------------|
| "Token-count confound in intervention" | C1, D1, D2, D3 |
| "Steering failure undermines causal claim" | A1 |
| "Layer-0 positive correlation is a confound" | A2 |
| "Selective CI reporting" | A3 |
| "n=99 vs n=100 inconsistency" | A4 |
| "StrategyQA r=−0.83 is just accuracy" | A5 |
| "Self-citation de-anonymization risk" | A6, A7 |
| "Phi-3 step-5 might be a different phenomenon" | A8, B2, D5 |
| "Show me the hidden states, not just statistics" | B1, D4 |
| "Figure 3 doesn't show Phi-3" | B3, D6 |
