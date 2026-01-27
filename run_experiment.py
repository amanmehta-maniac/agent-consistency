"""
Run a larger HotpotQA experiment: 100 questions × 10 runs.
"""
import asyncio
import os
from runner import HotpotQARunner

async def main():
    """Run 100 questions × 10 runs = 1000 total runs."""
    # Get API keys from environment
    openai_key = os.getenv("OPENAI_API_KEY")
    together_key = os.getenv("TOGETHER_API_KEY")
    
    # Choose provider (default to Together AI for cost)
    provider = "together" if together_key else "openai"
    model = (
        "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
        if provider == "together"
        else "gpt-4o"
    )
    api_key = together_key if provider == "together" else openai_key
    
    if not api_key:
        raise ValueError(f"{provider.upper()}_API_KEY not set in environment")
    
    print("=" * 70)
    print("HotpotQA Consistency Research - Full Experiment")
    print("=" * 70)
    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print(f"Questions: 100")
    print(f"Runs per question: 10")
    print(f"Total runs: 1000")
    print("=" * 70)
    print()
    
    # Initialize runner
    runner = HotpotQARunner(
        model=model,
        provider=provider,
        openai_api_key=openai_key if provider == "openai" else None,
        together_api_key=together_key if provider == "together" else None,
        results_dir="results",
        max_steps=15,
        temperature=0.7,
    )
    
    # Run experiment
    await runner.run_experiment(
        n_questions=100,
        n_runs_per_question=10,
        split="validation",
        start_idx=0,
    )

if __name__ == "__main__":
    asyncio.run(main())
