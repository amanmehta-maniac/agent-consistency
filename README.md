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
