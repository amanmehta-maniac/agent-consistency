"""
Activation steering experiment for Paper 3.
Runs 5 most inconsistent questions with baseline and steered conditions.

For each question:
- 10 baseline runs (no intervention)
- 10 steered runs at scale 0.5
- 10 steered runs at scale 1.0
- 10 steered runs at scale 2.0

Total: 5 questions × 40 runs = 200 agent runs
"""

import json
import os
import asyncio
import argparse
import numpy as np
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any, Literal
from dataclasses import dataclass, asdict, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from datasets import load_dataset
from agent import AgentStep, AgentRun, ReActAgent
from tools import create_search_fn, create_retrieve_fn


# Configuration
STEERING_VECTORS_DIR = Path("/Users/amehta/research/agent-consistency/hotpotqa/steering_vectors")
RESULTS_DIR = Path("/Users/amehta/research/agent-consistency/hotpotqa/steering_results")
STEERING_SUMMARY = STEERING_VECTORS_DIR / "steering_summary.json"

# Experiment parameters
INTERVENTION_STEP = 4  # Apply steering at step 4
INTERVENTION_LAYER = 72  # Layer 72 (0-indexed)
N_BASELINE_RUNS = 10
STEERING_SCALES = [0.5, 1.0, 2.0]
N_STEERED_RUNS_PER_SCALE = 10
MAX_STEPS = 25
TEMPERATURE = 0.5


class SteeredReActAgent(ReActAgent):
    """
    ReActAgent with activation steering support.
    At the specified intervention step, calls /v1/chat/completions_steered
    with a steering vector at the target layer.
    """
    
    def __init__(
        self,
        steering_vector: Optional[np.ndarray] = None,
        steering_scale: float = 1.0,
        intervention_step: int = INTERVENTION_STEP,
        intervention_layer: int = INTERVENTION_LAYER,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.steering_vector = steering_vector
        self.steering_scale = steering_scale
        self.intervention_step = intervention_step
        self.intervention_layer = intervention_layer
        self._current_step = 0  # Track current step during run
    
    async def _call_llm(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
    ) -> str:
        """
        Call the LLM, applying steering at the intervention step.
        """
        self._current_step += 1
        
        # Check if we should apply steering
        should_steer = (
            self.steering_vector is not None and
            self._current_step == self.intervention_step and
            self.provider == "llama_k8s"
        )
        
        if should_steer:
            return await self._call_llm_steered(messages, system_prompt)
        else:
            return await super()._call_llm(messages, system_prompt)
    
    async def _call_llm_steered(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
    ) -> str:
        """Call the steered endpoint with the steering vector."""
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        
        loop = asyncio.get_event_loop()
        
        def call_k8s_steered():
            # Scale the steering vector
            scaled_vector = self.steering_vector * self.steering_scale
            
            # Build request payload with steering
            payload = {
                "messages": [{"role": m["role"], "content": m["content"]} for m in full_messages],
                "max_new_tokens": 512,
                "temperature": self.temperature,
                "stop_sequences": ["Observation:"],
                "return_hidden_states": True,
                "hidden_state_pooling": "last",
                "steering_vector": scaled_vector.tolist(),
                "steering_layer": self.intervention_layer,
            }
            
            response = requests.post(
                f"{self.k8s_endpoint}/v1/chat/completions_steered",
                json=payload,
                timeout=300
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"K8s steered endpoint failed: {response.status_code} {response.text}")
            
            response_data = response.json()
            self._last_hidden_states = response_data.get("hidden_states")
            
            return response_data.get("content", "")
        
        return await loop.run_in_executor(self.executor, call_k8s_steered)
    
    async def run(
        self,
        question: str,
        task_id: str,
        run_id: Optional[str] = None,
    ) -> AgentRun:
        """Run the agent, resetting step counter."""
        self._current_step = 0  # Reset step counter for each run
        return await super().run(question, task_id, run_id)


def load_steering_data() -> Dict[str, Dict]:
    """Load steering summary and vectors for each question."""
    with open(STEERING_SUMMARY) as f:
        summary = json.load(f)
    
    data = {}
    for qid, info in summary.items():
        vec_dir = STEERING_VECTORS_DIR / qid
        vectors = np.load(vec_dir / "steering_vectors_normalized.npy")
        centroid = np.load(vec_dir / "centroid.npy")
        
        data[qid] = {
            **info,
            "steering_vectors": vectors,  # (n_runs, 8192)
            "centroid": centroid,
        }
    
    return data


def get_hotpotqa_question(qid: str) -> Dict[str, Any]:
    """Load a specific HotpotQA question by ID."""
    # Load from validation set (where our pilot questions are from)
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    
    for example in dataset:
        if example["id"] == qid:
            return example
    
    raise ValueError(f"Question {qid} not found in HotpotQA validation set")


async def run_single_agent(
    question: str,
    context: Dict,
    task_id: str,
    run_id: str,
    k8s_endpoint: str,
    steering_vector: Optional[np.ndarray] = None,
    steering_scale: float = 1.0,
) -> AgentRun:
    """Run a single agent with optional steering."""
    search_fn = create_search_fn(context)
    retrieve_fn = create_retrieve_fn(context)
    
    agent = SteeredReActAgent(
        model="llama-70b",
        provider="llama_k8s",
        k8s_endpoint=k8s_endpoint,
        max_steps=MAX_STEPS,
        temperature=TEMPERATURE,
        search_fn=search_fn,
        retrieve_fn=retrieve_fn,
        steering_vector=steering_vector,
        steering_scale=steering_scale,
    )
    
    return await agent.run(
        question=question,
        task_id=task_id,
        run_id=run_id,
    )


async def run_question_experiment(
    qid: str,
    question_data: Dict,
    hotpotqa_example: Dict,
    k8s_endpoint: str,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Run all experiments for a single question:
    - N baseline runs
    - N steered runs at each scale
    """
    question = hotpotqa_example["question"]
    context = hotpotqa_example["context"]
    ground_truth = hotpotqa_example["answer"]
    steering_vectors = question_data["steering_vectors"]
    
    results = {
        "question_id": qid,
        "question": question,
        "ground_truth": ground_truth,
        "cv_original": question_data["cv_steps"],
        "baseline_runs": [],
        "steered_runs": {str(scale): [] for scale in STEERING_SCALES},
    }
    
    # Select steering vectors to use (cycle through available ones)
    n_vectors = len(steering_vectors)
    
    print(f"\n  Running {N_BASELINE_RUNS} baseline runs...")
    
    # Baseline runs (no steering)
    for i in tqdm(range(N_BASELINE_RUNS), desc="  Baseline", leave=False):
        run_id = f"{qid}_baseline_{i+1:04d}"
        try:
            run = await run_single_agent(
                question=question,
                context=context,
                task_id=qid,
                run_id=run_id,
                k8s_endpoint=k8s_endpoint,
                steering_vector=None,
            )
            results["baseline_runs"].append(asdict(run))
        except Exception as e:
            print(f"    Error in baseline run {i+1}: {e}")
            results["baseline_runs"].append({
                "run_id": run_id,
                "error": str(e),
                "success": False,
            })
    
    # Steered runs at each scale
    for scale in STEERING_SCALES:
        print(f"  Running {N_STEERED_RUNS_PER_SCALE} steered runs at scale {scale}...")
        
        for i in tqdm(range(N_STEERED_RUNS_PER_SCALE), desc=f"  Scale {scale}", leave=False):
            run_id = f"{qid}_steered_{scale}_{i+1:04d}"
            # Cycle through steering vectors
            vector_idx = i % n_vectors
            steering_vec = steering_vectors[vector_idx]
            
            try:
                run = await run_single_agent(
                    question=question,
                    context=context,
                    task_id=qid,
                    run_id=run_id,
                    k8s_endpoint=k8s_endpoint,
                    steering_vector=steering_vec,
                    steering_scale=scale,
                )
                results["steered_runs"][str(scale)].append(asdict(run))
            except Exception as e:
                print(f"    Error in steered run {i+1} (scale={scale}): {e}")
                results["steered_runs"][str(scale)].append({
                    "run_id": run_id,
                    "error": str(e),
                    "success": False,
                })
    
    # Save results for this question
    output_file = output_dir / f"{qid}_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


async def main():
    parser = argparse.ArgumentParser(description="Activation steering experiment")
    parser.add_argument("--endpoint", default="http://localhost:8000", help="K8s endpoint URL")
    parser.add_argument("--question", help="Run single question by ID (for testing)")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without running")
    args = parser.parse_args()
    
    # Create output directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load steering data
    print("Loading steering data...")
    steering_data = load_steering_data()
    question_ids = list(steering_data.keys())
    
    if args.question:
        if args.question not in steering_data:
            print(f"Error: Question {args.question} not in steering data")
            print(f"Available: {question_ids}")
            return
        question_ids = [args.question]
    
    print(f"\nExperiment plan:")
    print(f"  Questions: {len(question_ids)}")
    print(f"  Baseline runs per question: {N_BASELINE_RUNS}")
    print(f"  Steering scales: {STEERING_SCALES}")
    print(f"  Steered runs per scale: {N_STEERED_RUNS_PER_SCALE}")
    print(f"  Total runs: {len(question_ids) * (N_BASELINE_RUNS + len(STEERING_SCALES) * N_STEERED_RUNS_PER_SCALE)}")
    print(f"  Output directory: {RESULTS_DIR}")
    
    if args.dry_run:
        print("\n[DRY RUN] Would run experiment on questions:")
        for qid in question_ids:
            print(f"  - {qid}: CV={steering_data[qid]['cv_steps']:.3f}")
        return
    
    # Load HotpotQA questions
    print("\nLoading HotpotQA questions...")
    hotpotqa_examples = {}
    for qid in question_ids:
        hotpotqa_examples[qid] = get_hotpotqa_question(qid)
    
    # Run experiments
    all_results = {}
    for i, qid in enumerate(question_ids, 1):
        print(f"\n{'='*60}")
        print(f"Question {i}/{len(question_ids)}: {qid}")
        print(f"CV: {steering_data[qid]['cv_steps']:.3f}")
        print(f"Question: {steering_data[qid]['question']}...")
        print(f"{'='*60}")
        
        results = await run_question_experiment(
            qid=qid,
            question_data=steering_data[qid],
            hotpotqa_example=hotpotqa_examples[qid],
            k8s_endpoint=args.endpoint,
            output_dir=RESULTS_DIR,
        )
        all_results[qid] = results
    
    # Save combined results
    combined_file = RESULTS_DIR / "all_results.json"
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print("Experiment complete!")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"Combined results: {combined_file}")


if __name__ == "__main__":
    asyncio.run(main())
