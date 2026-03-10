"""
Replication steering experiment across 5 questions with 3 conditions each.
Tests hypotheses:
  - Push 0.05 increases CV relative to baseline
  - Pull 0.1 decreases CV or maintains it, potentially improves accuracy
"""

import os
import sys
import json
import asyncio
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agent import ReActAgent


class RefinedSteeringAgent(ReActAgent):
    def __init__(
        self,
        steering_vector: np.ndarray = None,
        steering_scale: float = 1.0,
        steering_mode: str = "push",
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
        self._current_step = 0

    async def _call_llm(self, messages, system_prompt):
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
        import httpx
        formatted_messages = [{"role": "system", "content": system_prompt}]
        formatted_messages.extend(messages)
        steering_list = self.steering_vector.tolist() if isinstance(self.steering_vector, np.ndarray) else self.steering_vector
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": self.temperature,
            "max_tokens": 1024,
            "steering_vector": steering_list,
            "steering_scale": self.steering_scale,
            "steering_layer": self.intervention_layer,
            "steering_mode": self.steering_mode,
        }
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{self.k8s_endpoint}/v1/chat/completions_steered",
                json=payload,
            )
            response.raise_for_status()
            result = response.json()
        if "choices" in result:
            return result["choices"][0]["message"]["content"]
        return result["content"]


async def run_single(agent, question, ground_truth, run_id, question_id):
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


async def run_condition(question, question_id, ground_truth, steering_vector,
                        steering_scale, steering_mode, n_runs, endpoint, model):
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

    successful_runs = [r for r in results if r["success"]]
    step_counts = [r["n_steps"] for r in successful_runs]
    correct_count = sum(1 for r in results if r["correct"])
    all_answers = [r["final_answer"] for r in results if r["final_answer"]]
    unique_answers = len(set(a.strip().lower() for a in all_answers if a))

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
        "n_parseable": len(all_answers),
        "unique_answers": unique_answers,
        "mean_steps": float(mean_steps),
        "std_steps": float(std_steps),
        "cv": float(cv),
        "accuracy": correct_count / n_runs if n_runs > 0 else 0,
        "correct_count": correct_count,
        "runs": results,
    }


QUESTIONS = [
    {
        "id": "5ab3e45655429976abd1bcd4",
        "question": "The Vermont Catamounts men's soccer team currently competes in a conference that was formerly known as what from 1988 to 1996?",
        "ground_truth": "the North Atlantic Conference",
        "category": "inconsistent",
        "pilot_cv": 0.377,
        "pilot_acc": 0.5,
    },
    {
        "id": "5a85ea095542994775f606a8",
        "question": "What science fantasy young adult series, told in first person, has a set of companion books narrating the stories of enslaved worlds and alien species?",
        "ground_truth": "Animorphs",
        "category": "inconsistent",
        "pilot_cv": 0.324,
        "pilot_acc": 0.0,
    },
    {
        "id": "5a877e5d5542993e715abf7d",
        "question": "What screenwriter with credits for \"Evolution\" co-wrote a film starring Nicolas Cage and T\u00e9a Leoni?",
        "ground_truth": "David Weissman",
        "category": "consistent-correct",
        "pilot_cv": 0.175,
        "pilot_acc": 0.8,
    },
    {
        "id": "5a7bbb64554299042af8f7cc",
        "question": "Who is older, Annie Morton or Terry Richardson?",
        "ground_truth": "Terry Richardson",
        "category": "consistent-correct",
        "pilot_cv": 0.182,
        "pilot_acc": 0.7,
    },
    {
        "id": "5a87ab905542996e4f3088c1",
        "question": "The arena where the Lewiston Maineiacs played their home games can seat how many people?",
        "ground_truth": "3,677 seated",
        "category": "consistent-wrong",
        "pilot_cv": 0.136,
        "pilot_acc": 0.0,
    },
]


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://localhost:8000")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-70B-Instruct")
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from previous run, skipping valid conditions")
    args = parser.parse_args()

    vectors_base = Path(__file__).parent / "steering_vectors"
    output_dir = Path(__file__).parent / "steering_results_replication"
    output_dir.mkdir(exist_ok=True)
    results_file = output_dir / "replication_results.json"

    print("=" * 70)
    print("REPLICATION STEERING EXPERIMENT")
    print(f"5 questions x 3 conditions x {args.n_runs} runs = {5 * 3 * args.n_runs} total")
    print("=" * 70)
    print(f"Endpoint: {args.endpoint}")
    print(f"Conditions: baseline, push_0.05, pull_0.1")
    if args.resume:
        print("RESUME MODE: skipping conditions with valid data")
    print()

    if args.dry_run:
        for q in QUESTIONS:
            print(f"  [{q['category']}] {q['id']}: {q['question'][:60]}...")
        print(f"\n  Total runs: {5 * 3 * args.n_runs}")
        return

    # Load existing results if resuming
    if args.resume and results_file.exists():
        with open(results_file) as f:
            all_results = json.load(f)
        print(f"Loaded existing results with {len(all_results.get('questions', {}))} questions")
    else:
        all_results = {
            "timestamp": datetime.now().isoformat(),
            "n_runs_per_condition": args.n_runs,
            "conditions": ["baseline", "push_0.05", "pull_0.1"],
            "questions": {},
        }

    for qi, q in enumerate(QUESTIONS, 1):
        qid = q["id"]
        question = q["question"]
        gt = q["ground_truth"]

        print(f"\n{'=' * 70}")
        print(f"QUESTION {qi}/5: [{q['category']}] (pilot CV={q['pilot_cv']:.3f}, acc={q['pilot_acc']:.0%})")
        print(f"Q: {question}")
        print(f"GT: {gt}")
        print(f"{'=' * 70}")

        centroid = np.load(vectors_base / qid / "centroid.npy")
        print(f"  Centroid norm: {np.linalg.norm(centroid):.2f}")

        # Load existing question data if resuming, or start fresh
        existing_q = all_results.get("questions", {}).get(qid, {})
        existing_conds = existing_q.get("conditions", {})

        q_results = {
            "question_id": qid,
            "question": question,
            "ground_truth": gt,
            "category": q["category"],
            "pilot_cv": q["pilot_cv"],
            "pilot_acc": q["pilot_acc"],
            "conditions": dict(existing_conds),  # preserve existing valid conditions
        }

        def _has_valid_data(cond_name):
            """Check if existing condition data is valid (majority of runs succeeded)."""
            cond = existing_conds.get(cond_name, {})
            n_runs = cond.get("n_runs", 10)
            return (args.resume
                    and cond.get("n_successful", 0) >= n_runs // 2
                    and cond.get("n_parseable", 0) >= n_runs // 2)

        # --- BASELINE ---
        if _has_valid_data("baseline"):
            b = existing_conds["baseline"]
            print(f"\n  --- BASELINE --- SKIP (resume): CV={b['cv']:.3f}, Acc={b['accuracy']:.1%}")
        else:
            print(f"\n  --- BASELINE ---")
            baseline = await run_condition(
                question=question, question_id=qid, ground_truth=gt,
                steering_vector=None, steering_scale=0, steering_mode="none",
                n_runs=args.n_runs, endpoint=args.endpoint, model=args.model,
            )
            q_results["conditions"]["baseline"] = baseline
            print(f"    CV: {baseline['cv']:.3f}, Acc: {baseline['accuracy']:.1%}, "
                  f"Unique: {baseline['unique_answers']}, Parseable: {baseline['n_parseable']}/{args.n_runs}")

        all_results["questions"][qid] = q_results
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        # --- PUSH 0.05 ---
        if _has_valid_data("push_0.05"):
            p = existing_conds["push_0.05"]
            print(f"\n  --- PUSH 0.05 --- SKIP (resume): CV={p['cv']:.3f}, Acc={p['accuracy']:.1%}")
        else:
            print(f"\n  --- PUSH 0.05 ---")
            push = await run_condition(
                question=question, question_id=qid, ground_truth=gt,
                steering_vector=centroid, steering_scale=0.05, steering_mode="push",
                n_runs=args.n_runs, endpoint=args.endpoint, model=args.model,
            )
            q_results["conditions"]["push_0.05"] = push
            print(f"    CV: {push['cv']:.3f}, Acc: {push['accuracy']:.1%}, "
                  f"Unique: {push['unique_answers']}, Parseable: {push['n_parseable']}/{args.n_runs}")

        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        # --- PULL 0.1 ---
        if _has_valid_data("pull_0.1"):
            p = existing_conds["pull_0.1"]
            print(f"\n  --- PULL 0.1 --- SKIP (resume): CV={p['cv']:.3f}, Acc={p['accuracy']:.1%}")
        else:
            print(f"\n  --- PULL 0.1 ---")
            pull = await run_condition(
                question=question, question_id=qid, ground_truth=gt,
                steering_vector=centroid, steering_scale=0.1, steering_mode="pull",
                n_runs=args.n_runs, endpoint=args.endpoint, model=args.model,
            )
            q_results["conditions"]["pull_0.1"] = pull
            print(f"    CV: {pull['cv']:.3f}, Acc: {pull['accuracy']:.1%}, "
                  f"Unique: {pull['unique_answers']}, Parseable: {pull['n_parseable']}/{args.n_runs}")

        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    print(f"\n\n{'=' * 70}")
    print("FINAL SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Question':<14} {'Category':<18} {'Condition':<12} {'CV':>6} {'Acc':>6} {'Uniq':>5} {'Parse':>6}")
    print("-" * 72)

    for qid, qr in all_results["questions"].items():
        for cname, cond in qr["conditions"].items():
            print(f"{qid[:12]:<14} {qr['category']:<18} {cname:<12} "
                  f"{cond['cv']:>6.3f} {cond['accuracy']:>5.0%} "
                  f"{cond['unique_answers']:>5} {cond['n_parseable']:>3}/{cond['n_runs']}")
        print()

    print(f"\nResults saved to: {output_dir / 'replication_results.json'}")


if __name__ == "__main__":
    asyncio.run(main())
