# Agent Consistency Research

Research project analyzing consistency of LLM-based agents across multiple runs.

## Overview

This repository contains code and experiments for studying agent consistency across two benchmarks:

1. **HotpotQA** (`hotpotqa/`) - Multi-hop question answering with ReAct agents
2. **SWE-bench** (`swe-bench/`) - Software engineering tasks (coming soon)

## Research Question

How consistent are LLM-based agents in their behavior, and where does variance originate?

## Structure

```
agent-consistency/
├── README.md                 # This file
├── hotpotqa/                 # Paper 1: HotpotQA experiments
│   ├── agent.py              # ReAct agent implementation
│   ├── runner.py             # Experiment runner
│   ├── tools.py              # HotpotQA Search/Retrieve tools
│   ├── run_experiment.py     # Quick experiment script
│   ├── results_llama/        # Llama 3.1 70B results
│   ├── results_gpt4o/        # GPT-4o results
│   ├── results_claude/       # Claude Sonnet 4.5 results
│   ├── figures/              # Generated visualizations
│   └── paper/                # LaTeX paper
├── swe-bench/                # Paper 2: SWE-bench experiments
│   ├── agent.py
│   ├── runner.py
│   └── paper/
└── common/                    # Shared utilities
    ├── analysis.py           # Consistency metrics
    └── figures.py            # Figure generation
```

## Quick Start

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set API keys (add to ~/.zshrc for persistence)
export OPENAI_API_KEY="your_key"
export TOGETHER_API_KEY="your_key"
export ANTHROPIC_API_KEY="your_key"
```

### Run HotpotQA Experiments

```bash
cd hotpotqa

# Pilot: 5 questions × 5 runs
python runner.py --provider together --n-questions 5

# Full: 100 questions × 10 runs
python runner.py --provider openai --results-dir results_gpt4o --n-questions 100 --n-runs-per-question 10
```

### Generate Figures

```bash
cd hotpotqa
python -m common.figures
# Or from root:
python -m common.figures
```

## Models Supported

- **OpenAI**: GPT-4o
- **Together AI**: Llama 3.1 70B Instruct Turbo
- **Anthropic**: Claude Sonnet 4.5

## Metrics

1. **Outcome consistency**: % same final answer across runs
2. **Action sequence similarity**: Edit distance between action traces
3. **Reasoning trace similarity**: Semantic similarity of CoT (planned)
4. **First divergence point**: Which step do runs diverge?
5. **Failure mode distribution**: Categorize errors

## Results

Current experiments:
- **HotpotQA**: 100 validation questions × 10 runs each
- **Models**: Llama 3.1 70B, GPT-4o, Claude Sonnet 4.5
- **Temperature**: 0.7 (default), 0.0 (deterministic tests)

See `hotpotqa/figures/` for visualizations comparing consistency across models.

## Key Findings

> **TL;DR:** Behavioral consistency strongly predicts agent correctness. 
> Consistent agents hit 80–92% accuracy. Inconsistent ones: 25–60%.

1. **Consistency predicts correctness** — Tasks where the agent produces 
   ≤2 unique action sequences across runs: 80–92% accuracy. Tasks with 
   ≥6 unique sequences: 25–60%. A 32–55 percentage point gap.

2. **Divergence happens early** — 69% of behavioral divergence occurs at 
   step 2 (the first search query). Get the first tool call right and 
   downstream runs converge.

3. **Path length signals failure** — Consistent tasks average 3.4 steps 
   (85.7% accuracy). Inconsistent tasks average 7.8 steps (43% accuracy).

4. **Models vary significantly** — Llama 3.1 70B produces 4.2 unique 
   action sequences per task on average. Claude Sonnet: 2.0. GPT-4o: 2.4.

**Practical takeaway:** Run your agent 3–5x in parallel. If trajectories 
agree, trust the answer. If they scatter, flag for review.

## License

Research code - see individual papers for licensing.


## Papers

**Paper 1: HotpotQA** — [When Agents Disagree With Themselves: Measuring Behavioral Consistency in LLM-Based Agents](https://arxiv.org/abs/2602.11619) (arXiv, Feb 2026)

**Paper 2: SWE-bench** — Coming soon

### Citation
```bibtex
@article{mehta2026agents,
  title={When Agents Disagree With Themselves: Measuring Behavioral Consistency in LLM-Based Agents},
  author={Mehta, Aman},
  journal={arXiv preprint arXiv:2602.11619},
  year={2026}
}
```
