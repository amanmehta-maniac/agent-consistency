"""
Experiment runner for HotpotQA consistency research.
Loads HotpotQA data, runs ReActAgent N times per question, saves results.

Supports Together AI (Llama), OpenAI (GPT-4o), and Anthropic Claude via CLI flags.
"""

import json
import os
import asyncio
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Literal
from dataclasses import asdict
from tqdm import tqdm

from datasets import load_dataset
from agent import ReActAgent, AgentRun, run_agent_n_times
from tools import create_search_fn, create_retrieve_fn


class HotpotQARunner:
    """Runner for HotpotQA experiments with real Search/Retrieve tools."""
    
    def __init__(
        self,
        model: str,
        provider: Literal["openai", "together", "anthropic"],
        openai_api_key: Optional[str] = None,
        together_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        results_dir: str = "results_llama",
        max_steps: int = 15,
        temperature: float = 0.7,
    ):
        """
        Initialize the HotpotQA runner.
        
        Args:
            model: Model name
            provider: "openai", "together", or "anthropic"
            openai_api_key: OpenAI API key
            together_api_key: Together AI API key
            results_dir: Directory to save results
            max_steps: Maximum agent steps
            temperature: Sampling temperature
        """
        self.model = model
        self.provider = provider
        self.openai_api_key = openai_api_key
        self.together_api_key = together_api_key
        self.anthropic_api_key = anthropic_api_key
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.max_steps = max_steps
        self.temperature = temperature
        
        # Will be set when loading data
        self.dataset = None
        self.current_context = None
        self.current_titles = None
    
    def load_hotpotqa(
        self,
        split: str = "validation",
        n_examples: int = 100,
        config: str = "distractor",
    ):
        """
        Load HotpotQA dataset from HuggingFace.
        
        Args:
            split: Dataset split ("train" or "validation")
            n_examples: Number of examples to load (first N)
            config: Dataset config: "distractor" (10 paras/question) or "fullwiki"
        """
        print(f"Loading HotpotQA {config} {split} set (first {n_examples} examples)...")
        dataset = load_dataset(
            "hotpotqa/hotpot_qa",
            config,
            split=f"{split}[:{n_examples}]",
        )
        self.dataset = dataset
        print(f"Loaded {len(dataset)} examples")
        return dataset
    
    
    async def run_single_task(
        self,
        example: Dict[str, Any],
        n_runs: int = 10,
        task_id: Optional[str] = None,
    ) -> List[AgentRun]:
        """
        Run the agent N times on a single HotpotQA question.
        
        Args:
            example: HotpotQA example dict
            n_runs: Number of runs per question
            task_id: Optional task identifier (defaults to example id)
            
        Returns:
            List of AgentRun objects
        """
        question = example["question"]
        context = example["context"]
        task_id = task_id or example["id"]
        
        # Create Search/Retrieve functions for this example
        search_fn = create_search_fn(context)
        retrieve_fn = create_retrieve_fn(context)
        
        # Store context for reference
        self.current_context = context
        self.current_titles = context.get("title", [])
        
        # Run agent N times (optionally throttled for OpenAI)
        if self.provider == "openai":
            # Conservative throttling to respect TPM: small batches + delay
            runs = await run_agent_n_times(
                question=question,
                task_id=task_id,
                n=n_runs,
                model=self.model,
                provider=self.provider,
                openai_api_key=self.openai_api_key,
                together_api_key=self.together_api_key,
                anthropic_api_key=self.anthropic_api_key,
                run_id_prefix=f"{task_id}",
                max_steps=self.max_steps,
                temperature=self.temperature,
                search_fn=search_fn,
                retrieve_fn=retrieve_fn,
                batch_size=2,
                batch_delay=2.0,
            )
        else:
            runs = await run_agent_n_times(
                question=question,
                task_id=task_id,
                n=n_runs,
                model=self.model,
                provider=self.provider,
                openai_api_key=self.openai_api_key,
                together_api_key=self.together_api_key,
                anthropic_api_key=self.anthropic_api_key,
                run_id_prefix=f"{task_id}",
                max_steps=self.max_steps,
                temperature=self.temperature,
                search_fn=search_fn,
                retrieve_fn=retrieve_fn,
            )
        
        return runs
    
    def save_task_results(
        self,
        example: Dict[str, Any],
        runs: List[AgentRun],
        task_id: Optional[str] = None,
    ):
        """
        Save all runs for a task to a JSON file.
        
        Args:
            example: Original HotpotQA example
            runs: List of AgentRun objects
            task_id: Task identifier
        """
        task_id = task_id or example["id"]
        output_file = self.results_dir / f"{task_id}.json"
        
        # Prepare output structure
        output = {
            "task_id": task_id,
            "question": example["question"],
            "answer": example.get("answer", ""),
            "type": example.get("type", ""),
            "level": example.get("level", ""),
            "supporting_facts": example.get("supporting_facts", {}),
            "context_titles": example["context"].get("title", []),
            "model": self.model,
            "provider": self.provider,
            "n_runs": len(runs),
            "runs": [asdict(run) for run in runs],
        }
        
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
    
    async def run_experiment(
        self,
        n_questions: int = 100,
        n_runs_per_question: int = 10,
        split: str = "validation",
        start_idx: int = 0,
    ):
        """
        Run full experiment: load data, run agent on each question N times.
        
        Args:
            n_questions: Number of questions to process
            n_runs_per_question: Number of runs per question
            split: Dataset split
            start_idx: Starting index in dataset
        """
        # Load dataset if not already loaded
        if self.dataset is None:
            self.load_hotpotqa(split=split, n_examples=n_questions + start_idx)
        
        # Process questions
        total_runs = 0
        
        print(f"\nRunning experiment:")
        print(f"  Questions: {n_questions}")
        print(f"  Runs per question: {n_runs_per_question}")
        print(f"  Total runs: {n_questions * n_runs_per_question}")
        print(f"  Model: {self.model} ({self.provider})")
        print()
        
        # Process with progress bar (dataset[i] is a row dict)
        for i in tqdm(range(n_questions), desc="Questions"):
            example = self.dataset[start_idx + i]
            try:
                runs = await self.run_single_task(
                    example=example,
                    n_runs=n_runs_per_question,
                )
                self.save_task_results(example, runs)
                total_runs += len(runs)
            except Exception as e:
                print(f"\nError processing task {example.get('id', 'unknown')}: {e}")
                continue
        
        print(f"\n✓ Experiment complete!")
        print(f"  Total runs completed: {total_runs}")
        print(f"  Results saved to: {self.results_dir}")


async def main():
    """
    CLI entrypoint.

    Defaults: pilot experiment (5 questions × 5 runs).
    Override with flags, e.g.:
      - 100 Q × 10 runs, GPT-4o:
          python runner.py --provider openai --results-dir results_gpt4o \\
              --n-questions 100 --n-runs-per-question 10
      - 100 Q × 10 runs, Llama (Together):
          python runner.py --provider together --results-dir results_llama \\
              --n-questions 100 --n-runs-per-question 10
    """
    parser = argparse.ArgumentParser(
        description="Run HotpotQA ReAct experiments with OpenAI or Together AI."
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "together", "anthropic"],
        default=None,
        help="LLM provider to use (default: anthropic if ANTHROPIC_API_KEY set, else together, else openai).",
    )
    parser.add_argument(
        "--results-dir",
        default="results_llama",
        help="Directory to save results JSON files (default: results_llama).",
    )
    parser.add_argument(
        "--n-questions",
        type=int,
        default=5,
        help="Number of HotpotQA questions to run (default: 5).",
    )
    parser.add_argument(
        "--n-runs-per-question",
        type=int,
        default=5,
        help="Number of runs per question (default: 5).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for the LLM (default: 0.7).",
    )
    args = parser.parse_args()

    # Get API keys from environment
    openai_key = os.getenv("OPENAI_API_KEY")
    together_key = os.getenv("TOGETHER_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    # Choose provider (default priority: anthropic > together > openai)
    if args.provider is not None:
        provider: Literal["openai", "together", "anthropic"] = args.provider  # type: ignore[assignment]
    else:
        if anthropic_key:
            provider = "anthropic"
        elif together_key:
            provider = "together"
        else:
            provider = "openai"

    if provider == "together":
        model = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
        api_key = together_key
    elif provider == "anthropic":
        model = "claude-sonnet-4-5-20250929"
        api_key = anthropic_key
    else:
        model = "gpt-4o"
        api_key = openai_key

    if not api_key:
        env_name = (
            "TOGETHER_API_KEY" if provider == "together"
            else "ANTHROPIC_API_KEY" if provider == "anthropic"
            else "OPENAI_API_KEY"
        )
        raise ValueError(f"{env_name} not set in environment for provider '{provider}'")

    total_runs = args.n_questions * args.n_runs_per_question

    print("=" * 60)
    print("HotpotQA Consistency Research - Experiment")
    print("=" * 60)
    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print(f"Temperature: {args.temperature}")
    print(f"Questions: {args.n_questions}")
    print(f"Runs per question: {args.n_runs_per_question}")
    print(f"Total runs: {total_runs}")
    print(f"Results dir: {args.results_dir}")
    print("=" * 60)
    print()

    # Initialize runner
    runner = HotpotQARunner(
        model=model,
        provider=provider,
        openai_api_key=openai_key if provider == "openai" else None,
        together_api_key=together_key if provider == "together" else None,
        anthropic_api_key=anthropic_key if provider == "anthropic" else None,
        results_dir=args.results_dir,
        max_steps=15,
        temperature=args.temperature,
    )

    # Run experiment
    await runner.run_experiment(
        n_questions=args.n_questions,
        n_runs_per_question=args.n_runs_per_question,
        split="validation",
        start_idx=0,
    )


if __name__ == "__main__":
    asyncio.run(main())
