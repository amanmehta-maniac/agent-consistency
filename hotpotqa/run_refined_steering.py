"""
Refined steering experiment with smaller scales and two approaches:
1. Push away from overall centroid (original approach, smaller scales)
2. Pull toward correct-answer centroid (new approach)

For question 5ab3b0bf5542992ade7c6e39 only.
"""

import os
import sys
import json
import asyncio
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from agent import ReActAgent


class RefinedSteeringAgent(ReActAgent):
    """Agent that supports both push-away and pull-toward steering."""
    
    def __init__(
        self,
        steering_vector: np.ndarray = None,
        steering_scale: float = 1.0,
        steering_mode: str = "push",  # "push" or "pull"
        intervention_step: int = 4,
        intervention_layer: int = 72,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.steering_vector = steering_vector
        self.steering_scale = steering_scale
        self.steering_mode = steering_mode
        self.intervention_step = intervention_step
        self.intervention_layer = intervention_layer
        self._current_step = 0
        
    def reset(self):
        """Reset for new run."""
        self._current_step = 0
        
    async def _call_llm(self, messages, system_prompt):
        """Override to apply steering at intervention step."""
        self._current_step += 1
        
        should_steer = (
            self.steering_vector is not None and
            self._current_step == self.intervention_step and
            self.provider == "llama_k8s"
        )
        
        if should_steer:
            return await self._call_llm_steered(messages, system_prompt)
        
        return await super()._call_llm(messages, system_prompt)
    
    async def _call_llm_steered(self, messages, system_prompt):
        """Call LLM with steering intervention."""
        import httpx
        
        # Format messages for OpenAI-compatible API
        formatted_messages = [{"role": "system", "content": system_prompt}]
        formatted_messages.extend(messages)
        
        # Prepare steering payload
        steering_list = self.steering_vector.tolist() if isinstance(self.steering_vector, np.ndarray) else self.steering_vector
        
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": self.temperature,
            "max_tokens": 1024,
            "steering_vector": steering_list,
            "steering_scale": self.steering_scale,
            "steering_layer": self.intervention_layer,
            "steering_mode": self.steering_mode,  # "push" or "pull"
        }
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{self.k8s_endpoint}/v1/chat/completions_steered",
                json=payload,
            )
            response.raise_for_status()
            result = response.json()
        
        # Steered endpoint returns {content: ...} not {choices: [{message: {content: ...}}]}
        if "choices" in result:
            return result["choices"][0]["message"]["content"]
        return result["content"]


def compute_correct_centroid(question_id: str, pilot_dir: str, ground_truth: str) -> np.ndarray:
    """Compute centroid from runs that got the correct answer."""
    question_dir = Path(pilot_dir) / question_id
    
    correct_states = []
    
    for run_dir in sorted(question_dir.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
            continue
            
        # Check if this run got correct answer
        metadata_file = run_dir / "metadata.json"
        trajectory_file = run_dir / "trajectory.json"
        
        answer = None
        if metadata_file.exists():
            with open(metadata_file) as f:
                meta = json.load(f)
            answer = meta.get("agent_answer", "")
        if not answer and trajectory_file.exists():
            with open(trajectory_file) as f:
                traj = json.load(f)
            answer = traj.get("final_answer", "")
        
        # Check if correct (case-insensitive substring match)
        is_correct = ground_truth.lower() in str(answer).lower() if answer else False
        
        # Load hidden state at step 4, layer 72
        hidden_state_file = run_dir / "hidden_states_step_4.npy"
        if hidden_state_file.exists():
            hidden_states = np.load(hidden_state_file)
            # Extract layer 72 (last token, layer 72)
            if hidden_states.ndim == 3:
                state = hidden_states[-1, 72, :]  # last token, layer 72
            else:
                state = hidden_states[72, :]
            
            print(f"  {run_dir.name}: answer='{answer}', correct={is_correct}")
            
            if is_correct:
                correct_states.append(state)
    
    if not correct_states:
        print(f"  WARNING: No correct runs found for {question_id}")
        return None
    
    print(f"  Found {len(correct_states)} correct runs")
    centroid = np.mean(correct_states, axis=0)
    return centroid


async def run_single(
    agent: RefinedSteeringAgent,
    question: str,
    ground_truth: str,
    run_id: int,
    question_id: str,
) -> dict:
    """Run a single agent execution."""
    agent.reset()
    
    try:
        result = await agent.run(question, task_id=question_id, run_id=f"run_{run_id:04d}")
        answer = result.final_answer or ""
        correct = ground_truth.lower() in answer.lower() if answer else False
        
        return {
            "run_id": run_id,
            "success": result.success,
            "final_answer": answer,
            "correct": correct,
            "n_steps": len(result.steps),
            "error": result.error,
        }
    except Exception as e:
        return {
            "run_id": run_id,
            "success": False,
            "final_answer": None,
            "correct": False,
            "n_steps": 0,
            "error": str(e),
        }


async def run_condition(
    question: str,
    question_id: str,
    ground_truth: str,
    steering_vector: np.ndarray,
    steering_scale: float,
    steering_mode: str,
    n_runs: int,
    endpoint: str,
    model: str,
) -> dict:
    """Run multiple runs for a single condition."""
    results = []
    
    for i in range(n_runs):
        agent = RefinedSteeringAgent(
            model=model,
            provider="llama_k8s",
            k8s_endpoint=endpoint,
            steering_vector=steering_vector,
            steering_scale=steering_scale,
            steering_mode=steering_mode,
            intervention_step=4,
            intervention_layer=72,
            max_steps=15,
            temperature=0.7,
        )
        
        result = await run_single(agent, question, ground_truth, i + 1, question_id)
        results.append(result)
        
        status = "correct" if result["correct"] else ("wrong" if result["success"] else "failed")
        print(f"      Run {i+1}/{n_runs}: {result['n_steps']} steps, {status}", flush=True)
    
    # Compute metrics
    successful_runs = [r for r in results if r["success"]]
    step_counts = [r["n_steps"] for r in successful_runs]
    correct_count = sum(1 for r in results if r["correct"])
    
    if step_counts:
        mean_steps = np.mean(step_counts)
        std_steps = np.std(step_counts)
        cv = std_steps / mean_steps if mean_steps > 0 else 0
    else:
        mean_steps = std_steps = cv = 0
    
    return {
        "mode": steering_mode,
        "scale": steering_scale,
        "n_runs": n_runs,
        "n_successful": len(successful_runs),
        "mean_steps": float(mean_steps),
        "std_steps": float(std_steps),
        "cv": float(cv),
        "accuracy": correct_count / n_runs if n_runs > 0 else 0,
        "correct_count": correct_count,
        "runs": results,
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://localhost:8000")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-70B-Instruct")
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    # Question details
    question_id = "5ab3b0bf5542992ade7c6e39"
    question = "What year did Guns N Roses perform a promo for a movie starring Arnold Schwarzenegger as a former New York Police detective?"
    ground_truth = "1999"
    
    pilot_dir = Path(__file__).parent / "pilot_hidden_states_70b"
    vectors_dir = Path(__file__).parent / "steering_vectors" / question_id
    output_dir = Path(__file__).parent / "steering_results_refined"
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("REFINED STEERING EXPERIMENT")
    print("=" * 60)
    print(f"Question: {question}")
    print(f"Ground truth: {ground_truth}")
    print(f"Endpoint: {args.endpoint}")
    print()
    
    # Load overall centroid (for push-away)
    print("Loading overall centroid...")
    overall_centroid = np.load(vectors_dir / "centroid.npy")
    print(f"  Shape: {overall_centroid.shape}")
    print(f"  Norm: {np.linalg.norm(overall_centroid):.2f}")
    
    # Compute correct-answer centroid (for pull-toward)
    print("\nComputing correct-answer centroid...")
    correct_centroid = compute_correct_centroid(question_id, pilot_dir, ground_truth)
    
    # If no correct runs, use overall centroid for pull-toward as well
    if correct_centroid is None:
        print("  No correct runs found in pilot. Using overall centroid for pull-toward.")
        print("  (This tests 'pull toward mean behavior' which may still be interesting)")
        correct_centroid = overall_centroid
    
    print(f"  Shape: {correct_centroid.shape}")
    print(f"  Norm: {np.linalg.norm(correct_centroid):.2f}")
    
    # Scales to test
    push_scales = [0.01, 0.05, 0.1, 0.2]
    pull_scales = [0.05, 0.1, 0.2]
    
    if args.dry_run:
        print("\n[DRY RUN] Would run:")
        print(f"  Baseline: {args.n_runs} runs")
        print(f"  Push scales {push_scales}: {len(push_scales) * args.n_runs} runs")
        print(f"  Pull scales {pull_scales}: {len(pull_scales) * args.n_runs} runs")
        print(f"  Total: {args.n_runs + len(push_scales) * args.n_runs + len(pull_scales) * args.n_runs} runs")
        return
    
    all_results = {
        "question_id": question_id,
        "question": question,
        "ground_truth": ground_truth,
        "timestamp": datetime.now().isoformat(),
        "conditions": {},
    }
    
    # Run baseline
    print("\n" + "=" * 60)
    print("BASELINE (no steering)")
    print("=" * 60)
    baseline = await run_condition(
        question=question,
        question_id=question_id,
        ground_truth=ground_truth,
        steering_vector=None,
        steering_scale=0,
        steering_mode="none",
        n_runs=args.n_runs,
        endpoint=args.endpoint,
        model=args.model,
    )
    all_results["conditions"]["baseline"] = baseline
    print(f"  CV: {baseline['cv']:.3f}, Accuracy: {baseline['accuracy']:.1%}")
    with open(output_dir / f"{question_id}_refined.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Run push-away conditions
    print("\n" + "=" * 60)
    print("PUSH AWAY FROM OVERALL CENTROID")
    print("=" * 60)
    
    for scale in push_scales:
        print(f"\n  Scale {scale}:")
        # Steering vector = push away = (will be computed at runtime as state - centroid)
        # We pass the centroid, and mode="push" tells server to compute state - centroid
        result = await run_condition(
            question=question,
            question_id=question_id,
            ground_truth=ground_truth,
            steering_vector=overall_centroid,
            steering_scale=scale,
            steering_mode="push",
            n_runs=args.n_runs,
            endpoint=args.endpoint,
            model=args.model,
        )
        all_results["conditions"][f"push_{scale}"] = result
        print(f"    CV: {result['cv']:.3f}, Accuracy: {result['accuracy']:.1%}")
        with open(output_dir / f"{question_id}_refined.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)
    
    # Run pull-toward conditions
    print("\n" + "=" * 60)
    print("PULL TOWARD CORRECT-ANSWER CENTROID")
    print("=" * 60)
    
    for scale in pull_scales:
        print(f"\n  Scale {scale}:")
        # Pass correct centroid, mode="pull" tells server to compute centroid - state
        result = await run_condition(
            question=question,
            question_id=question_id,
            ground_truth=ground_truth,
            steering_vector=correct_centroid,
            steering_scale=scale,
            steering_mode="pull",
            n_runs=args.n_runs,
            endpoint=args.endpoint,
            model=args.model,
        )
        all_results["conditions"][f"pull_{scale}"] = result
        print(f"    CV: {result['cv']:.3f}, Accuracy: {result['accuracy']:.1%}")
        with open(output_dir / f"{question_id}_refined.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)
    
    # Save results
    output_file = output_dir / f"{question_id}_refined.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved results to: {output_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Condition':<20} {'CV':>8} {'Accuracy':>10} {'N_Success':>10}")
    print("-" * 50)
    
    for name, cond in all_results["conditions"].items():
        print(f"{name:<20} {cond['cv']:>8.3f} {cond['accuracy']:>9.1%} {cond['n_successful']:>10}")


if __name__ == "__main__":
    asyncio.run(main())
