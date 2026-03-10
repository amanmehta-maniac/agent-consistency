# Agent Consistency Research — Knowledge Share

## 1. Research Goal

We are investigating **behavioral consistency in LLM-based agents** — specifically, why the same agent given the same question produces different reasoning paths, step counts, and final answers across runs. The ultimate goal is to understand whether internal model representations (hidden states) can **predict** and **control** this variability.

The core metric is **CV (Coefficient of Variation)** = std(steps) / mean(steps) across repeated runs of the same question. High CV = inconsistent behavior.

---

## 2. Research Arc (Chronological)

### Phase 1: Observational Study (Papers 1 & 2)

Ran ReAct agents (GPT-5, Claude, Llama-3.1-70B) on HotpotQA multi-hop questions, 10 runs per question. Measured consistency metrics (CV, accuracy, unique answers). Key finding: agents exhibit significant run-to-run variability even with temperature=0 (due to sampling in tool use, context sensitivity, etc.).

Figures and analysis in `figures2/`, papers in `paper/` and `paper2_overleaf/`.

### Phase 2: Hidden State Extraction (Paper 3 Infrastructure)

Deployed Llama-3.1-70B-Instruct on an internal K8s cluster with a custom inference server that extracts hidden states at every layer during generation. This lets us observe the model's internal representations as it reasons through each step of the ReAct loop.

**Key infrastructure:**
- **K8s pod**: `llama-70b-hidden-states-0-jgglf` on cluster `sfc-or-dev-meta-k8s-2`, namespace `mltraining-dev`
- **Model**: 80 layers, hidden_dim=8192, pipeline-parallel across GPUs with `device_map="sequential"`
- **Server**: `llama-hidden-states/src/server.py` (771 lines) — FastAPI server with `/v1/chat/completions` (normal + hidden state extraction) and `/v1/chat/completions/steered` (activation steering)
- **Access**: Port-forward to `localhost:8000` (port-forward is fragile and dies during long experiments — always restart before running)

### Phase 3: Pilot Analysis — Can Hidden States Predict Consistency?

Ran 10 pilot runs per question on 40 HotpotQA questions (20 "hard" + 20 "easier"), extracting hidden states at each step. Computed pairwise cosine similarity between runs at each layer/step combination.

**Key findings** (`hotpotqa/analysis_results/combined_40q/summary_table.md`):
- **Step 4 is the critical divergence point**: Correlation between hidden-state similarity and behavioral consistency peaks at step 4 (r=-0.436, p=0.005). Earlier steps show no signal.
- **Later layers carry more signal**: Layers 64-80 show significant correlation (r=-0.32 to -0.36), while early layers do not.
- **But raw similarity is not a good classifier**: ROC AUC = 0.30 for predicting consistent vs inconsistent from hidden-state similarity alone. The signal exists but is not cleanly separable.

Question categories (reclassified for 70B):
- **Inconsistent**: CV > 0.25 (high run-to-run variability)
- **Consistent-correct**: CV <= 0.25, accuracy >= 60%
- **Consistent-wrong**: CV <= 0.25, accuracy <= 10%

### Phase 4: Activation Steering — Can We Control Consistency?

This is the current and most exciting phase. Instead of just predicting consistency, we now **intervene** on the model's hidden states during generation to steer behavior.

#### 4a. Steering Mechanism

For each question, we compute a **centroid** — the mean hidden state at layer 72, step 4, across 10 pilot runs. This represents the "average internal state" the model reaches at the critical divergence point.

Three steering modes (implemented as a PyTorch forward hook on layer 72):
- **Push**: `new_state = state + scale * (state - centroid)` — amplifies deviation from average, should increase variability
- **Pull**: `new_state = state + scale * (centroid - state)` — pulls toward average, should decrease variability  
- **Add**: `new_state = state + scale * vector` — direct vector addition (legacy mode)

The hook fires on **every token generation**, modifying the last-token hidden state at the target layer.

Steering vectors are stored in `hotpotqa/steering_vectors/<question_id>/`:
- `centroid.npy` — mean hidden state (shape: [8192])
- `run_states.npy` — per-run states
- `steering_vectors.npy` — per-run deviation vectors
- `steering_vectors_normalized.npy` — unit-norm versions

Computed by `hotpotqa/compute_steering_vectors.py` from pilot hidden state data.

#### 4b. Experiment 1: Scale Sweep on Single Question (80 runs)

Script: `hotpotqa/run_refined_steering.py`
Results: `hotpotqa/steering_results_refined/5ab3b0bf5542992ade7c6e39_refined.json`

Tested push and pull at scales [0.01, 0.02, 0.05, 0.1] on question `5ab3b0bf` ("What year did Guns N Roses perform a promo..."). This identified the best scales:
- **Push 0.05**: good CV-increasing scale
- **Pull 0.1**: good regularizing scale

#### 4c. Experiment 2: Replication Across 5 Questions (150 runs)

Script: `hotpotqa/run_replication_steering.py`
Results: `hotpotqa/steering_results_replication/replication_results.json`

5 questions (2 inconsistent, 2 consistent-correct, 1 consistent-wrong) x 3 conditions (baseline, push 0.05, pull 0.1) x 10 runs each.

**Results:**

| Question | Category | Baseline CV | Push CV | Pull CV | Baseline Acc | Push Acc | Pull Acc |
|----------|----------|-------------|---------|---------|--------------|----------|----------|
| 5ab3e456 | inconsistent | 0.174 | **0.246** | **0.105** | 0% | 0% | 0% |
| 5a85ea09 | inconsistent | 0.209 | 0.166 | 0.169 | 0% | 0% | 0% |
| 5a877e5d | consistent-correct | 0.233 | **0.272** | 0.258 | 50% | 50% | 40% |
| 5a7bbb64 | consistent-correct | 0.240 | **0.272** | **0.196** | 30% | **50%** | 30% |
| 5a87ab90 | consistent-wrong | 0.211 | 0.209 | **0.372** | 0% | 0% | 0% |
| **Average** | | **0.214** | **0.233** | **0.220** | **16%** | **20%** | **14%** |

**Key findings:**
1. **Push reliably increases CV on 3/5 questions** (avg +9%), strongest on inconsistent Q1 (+41%)
2. **Pull strongly regularizes inconsistent Q1** (CV: 0.174 -> 0.105, a 40% reduction)
3. **Pull backfires on consistent-wrong Q5** — CV jumps from 0.211 to 0.372. The centroid encodes the wrong attractor, so pulling toward it creates interference rather than regularization.
4. **Push unexpectedly improves accuracy** on consistent-correct Q4 (30% -> 50%), possibly by exploring more diverse reasoning paths.
5. **Inconsistent Q2 is resistant to steering** — neither push nor pull significantly changes CV. Some questions may have deeper sources of variability.

---

## 3. Repository Structure

```
agent-consistency/
├── hotpotqa/
│   ├── agent.py                          # ReAct agent implementation
│   ├── tools.py                          # Wikipedia search/lookup tools
│   ├── run_pilot_70b.py                  # 70B pilot experiment runner
│   ├── compute_steering_vectors.py       # Computes centroids from hidden states
│   ├── run_refined_steering.py           # Scale sweep experiment (80 runs)
│   ├── run_replication_steering.py       # 5-question replication (150 runs)
│   ├── steering_vectors/                 # Per-question centroid.npy files
│   ├── steering_results/                 # Initial 5-question steering (large scales)
│   ├── steering_results_refined/         # Scale sweep results
│   ├── steering_results_replication/     # 5-question replication results
│   ├── analysis_results/                 # Pilot analysis outputs
│   │   └── combined_40q/                 # 40-question combined analysis
│   └── hidden_states_*/                  # Raw hidden state .npy files (gitignored)
├── llama-hidden-states/
│   ├── src/server.py                     # K8s inference server with steering
│   └── k8s/                              # K8s deployment manifests
├── paper/                                # Paper 1
├── paper2_overleaf/                      # Paper 2
├── figures2/                             # 3-model comparison figures
└── swe-bench/                            # SWE-bench consistency experiments (separate track)
```

---

## 4. Infrastructure & Operational Notes

### K8s Pod Access
```bash
# Port-forward (must be running for any experiment)
kubectl port-forward llama-70b-hidden-states-0-jgglf 8000:8000 \
  --context sfc-or-dev-meta-k8s-2 -n mltraining-dev

# Health check
curl http://localhost:8000/health
```

**Critical**: Port-forward dies during long experiments. The replication script has `--resume` support to skip conditions that already have valid data (>= 5/10 successful runs). Always restart port-forward before resuming.

### Running Experiments
```bash
cd /Users/amehta/research/agent-consistency

# Compute centroid for a new question (needs hidden_states_step_4.npy in pilot data)
.venv/bin/python3 hotpotqa/compute_steering_vectors.py --question-id <qid>

# Run replication with resume support
.venv/bin/python3 hotpotqa/run_replication_steering.py --resume
```

### API Formats
- **Normal endpoint** (`/v1/chat/completions`): Returns `{choices: [{message: {content: "..."}}]}`
- **Steered endpoint** (`/v1/chat/completions/steered`): Returns flat `{content: "..."}`
  - The agent class `RefinedSteeringAgent` in the experiment scripts handles both formats.

### ConfigMap
The server code lives in K8s ConfigMap `llama-server-code`. To update:
```bash
kubectl create configmap llama-server-code --from-file=server.py=llama-hidden-states/src/server.py \
  --context sfc-or-dev-meta-k8s-2 -n mltraining-dev --dry-run=client -o yaml | kubectl apply -f -
# Then restart the pod
```

---

## 5. Open Questions & Next Steps

1. **Why does Q2 resist steering?** Some questions may have variability sources beyond what layer-72 step-4 captures. Investigate whether Q2's divergence happens at a different step or layer.

2. **Pull backfiring on consistent-wrong**: The centroid for a consistently-wrong question encodes the wrong reasoning attractor. Could we compute a "correct-answer centroid" from a different question to pull toward instead?

3. **Statistical power**: 10 runs per condition is enough for directional signals but not for significance testing. Consider 20-30 runs per condition for key questions.

4. **More question types**: We have centroids for 8 questions but only tested 6 (1 in refined + 5 in replication). Expand to more questions, especially more consistent-wrong ones (currently only 1).

5. **Layer/step sensitivity**: We fixed layer=72, step=4 based on pilot analysis. Is this optimal for all questions, or would per-question tuning help?

6. **Scaling to other models**: The infrastructure is Llama-70B-specific. The conceptual approach (centroid computation -> push/pull steering) could apply to any transformer with hook access.

---

## 6. GitHub

Repository: https://github.com/amanmehta-maniac/agent-consistency

Key commits:
- `c6b575c` — Replication: 5-question steering (latest)
- `2f51f59` — Refined steering: push/pull scale sweep
- `e5e8706` — Initial steering experiment
- `fa5fbbd` — 40-question pilot analysis
