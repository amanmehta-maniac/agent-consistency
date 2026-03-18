#!/usr/bin/env python3
"""
Task 3: Prompting Intervention — Forced Commitment at Step 3.

At step 3, injects a "commitment prompt" into the observation that instructs
the model to commit to its current reasoning path. Compares with control
(no intervention) on 100 questions × 10 runs each.

Conditions:
  - control: standard ReAct (no intervention)
  - commitment: at step 3, append commitment instruction to observation
  - filler: at step 3, append semantically neutral filler text (matched token count)

Usage:
    python run_intervention_experiment.py --questions /data/qwen_100q_full.json \
        --output /results/intervention_experiment --condition commitment
    python run_intervention_experiment.py --questions /data/qwen_100q_full.json \
        --output /results/intervention_experiment --condition control
"""

import asyncio
import argparse
import json
import logging
import os
import sys
import time
import traceback
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

import requests

from agent import ReActAgent, AgentStep, AgentRun
from tools import create_search_fn, create_retrieve_fn

N_RUNS_PER_QUESTION = 10
MAX_STEPS = 30
TEMPERATURE = 0.5
K8S_ENDPOINT = os.environ.get("K8S_ENDPOINT", "http://localhost:8000")
HEALTH_TIMEOUT = 1800
HEALTH_POLL_INTERVAL = 10

INTERVENTION_STEP = 3

COMMITMENT_PROMPT = (
    "\n\n[IMPORTANT: Based on the evidence you have gathered so far, commit to a "
    "specific reasoning strategy for solving this question. State your committed "
    "strategy clearly in your next Thought, then follow through with it. Do not "
    "change strategies or start over — build on what you have learned.]"
)

FILLER_PROMPT = (
    "\n\n[NOTE: Please continue with the task as you normally would. "
    "Take the time you need to work through the problem. Consider the "
    "information you have gathered and proceed with your next step in "
    "whatever way seems most appropriate to you. There is no particular "
    "urgency — work at your own pace and follow your reasoning.]"
)


def setup_logging(output_dir: Path):
    log_path = output_dir / "experiment_progress.log"
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    logger = logging.getLogger("experiment")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


def wait_for_server(logger):
    logger.info(f"Waiting for server at {K8S_ENDPOINT} (timeout {HEALTH_TIMEOUT}s)...")
    start = time.time()
    while time.time() - start < HEALTH_TIMEOUT:
        try:
            resp = requests.get(f"{K8S_ENDPOINT}/health", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                logger.info(f"Server healthy: model={data.get('model_name', 'unknown')}")
                return True
        except Exception:
            pass
        time.sleep(HEALTH_POLL_INTERVAL)
    logger.error("Server did not become healthy in time")
    return False


def extract_hidden_states_array(hidden_states_dict):
    if not hidden_states_dict or "layers" not in hidden_states_dict:
        return None
    layers = hidden_states_dict["layers"]
    num_layers = len(layers)
    arrays = []
    for i in range(num_layers):
        layer_key = f"layer_{i}"
        if layer_key in layers and layers[layer_key]:
            arrays.append(np.array(layers[layer_key], dtype=np.float32))
    if arrays:
        return np.stack(arrays)
    return None


def format_context_for_tools(context_dict):
    if isinstance(context_dict, dict) and "title" in context_dict:
        return context_dict
    titles = list(context_dict.keys())
    sentences = [text.split(". ") for text in context_dict.values()]
    return {"title": titles, "sentences": sentences}


def get_completed_runs(output_dir: Path, question_id: str) -> int:
    q_dir = output_dir / question_id
    if not q_dir.exists():
        return 0
    count = 0
    for run_dir in q_dir.glob("run_*"):
        if (run_dir / "metadata.json").exists():
            count += 1
    return count


def save_run_results(output_dir, question_id, run_id, result, question, expected_answer, condition):
    run_dir = output_dir / question_id / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    final_answer = result.final_answer or ""
    correct = (
        expected_answer.lower() in final_answer.lower()
        or final_answer.lower() in expected_answer.lower()
    ) if final_answer else False

    metadata = {
        "question_id": question_id,
        "run_id": run_id,
        "condition": condition,
        "question": question,
        "expected_answer": expected_answer,
        "agent_answer": final_answer,
        "correct": correct,
        "total_steps": len(result.steps),
        "actions": [s.action for s in result.steps],
        "success": result.success,
        "error": result.error,
        "timestamp": datetime.now().isoformat(),
    }
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    trajectory = {
        "question_id": question_id,
        "run_id": run_id,
        "condition": condition,
        "question": question,
        "final_answer": final_answer,
        "steps": [
            {
                "step_number": s.step_number,
                "thought": s.thought,
                "action": s.action,
                "action_input": s.action_input,
                "observation": s.observation,
                "timestamp": s.timestamp,
            }
            for s in result.steps
        ],
    }
    with open(run_dir / "trajectory.json", "w") as f:
        json.dump(trajectory, f, indent=2)

    for step in result.steps:
        if hasattr(step, "hidden_states") and step.hidden_states:
            hs_array = extract_hidden_states_array(step.hidden_states)
            if hs_array is not None:
                np.save(run_dir / f"hidden_states_step_{step.step_number}.npy", hs_array)

    return correct


def save_progress(output_dir, progress):
    with open(output_dir / "progress.json", "w") as f:
        json.dump(progress, f, indent=2)


class InterventionReActAgent(ReActAgent):
    def __init__(self, condition="control", intervention_step=INTERVENTION_STEP, **kwargs):
        super().__init__(**kwargs)
        self.condition = condition
        self.intervention_step = intervention_step
        self._step_counter = 0

    async def run(self, question, task_id, run_id=None):
        self._step_counter = 0
        if run_id is None:
            run_id = f"{task_id}_{datetime.now().isoformat()}"

        start_time = datetime.now().isoformat()
        steps = []
        conversation_history = []
        final_answer = None
        success = False
        error = None

        try:
            if self.provider == "llama_k8s":
                conversation_history.append({
                    "role": "user",
                    "content": "Question: What is the capital of the country where the Eiffel Tower is located?"
                })
                conversation_history.append({
                    "role": "assistant",
                    "content": 'Thought: I need to find where the Eiffel Tower is located.\nAction: Search\nAction Input: {"query": "Eiffel Tower"}'
                })
                conversation_history.append({
                    "role": "user",
                    "content": "Observation: Found 3 relevant document(s): Eiffel Tower, Paris architecture, Gustave Eiffel"
                })
                conversation_history.append({
                    "role": "assistant",
                    "content": 'Thought: Search returned document titles. I need to retrieve "Eiffel Tower" to read its content.\nAction: Retrieve\nAction Input: {"title": "Eiffel Tower"}'
                })
                conversation_history.append({
                    "role": "user",
                    "content": "Observation: The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after Gustave Eiffel."
                })
                conversation_history.append({
                    "role": "assistant",
                    "content": 'Thought: The document says the Eiffel Tower is in Paris, France. The capital of France is Paris.\nAction: Finish\nAction Input: {"answer": "Paris"}'
                })
                conversation_history.append({
                    "role": "user",
                    "content": "Observation: Task completed. Final answer: Paris\n\nGood. Now answer the next question using the same format."
                })

            conversation_history.append({
                "role": "user",
                "content": f"Question: {question}"
            })

            for step_num in range(1, self.max_steps + 1):
                self._step_counter = step_num

                try:
                    response = await self._call_llm(
                        messages=conversation_history,
                        system_prompt=self._get_system_prompt(),
                    )
                except Exception as e:
                    error = f"LLM call failed at step {step_num}: {str(e)}"
                    break

                thought, action, action_input = self._parse_response(response)

                if action not in self.tools:
                    error = f"Invalid action '{action}' at step {step_num}\nRaw LLM response:\n{response[:500]}"
                    break

                try:
                    if action == "Finish":
                        if isinstance(action_input, dict):
                            final_answer = action_input.get("answer", str(action_input))
                        else:
                            final_answer = str(action_input)
                        observation = await self.tools[action](final_answer)
                        success = True
                    else:
                        if isinstance(action_input, dict):
                            param_value = (
                                action_input.get("query") or
                                action_input.get("title") or
                                action_input.get("value") or
                                list(action_input.values())[0] if action_input else ""
                            )
                        else:
                            param_value = str(action_input)
                        observation = await self.tools[action](param_value)
                except Exception as e:
                    observation = f"Error executing action: {str(e)}"
                    error = observation
                    break

                if step_num == self.intervention_step and action != "Finish":
                    if self.condition == "commitment":
                        observation = observation + COMMITMENT_PROMPT
                    elif self.condition == "filler":
                        observation = observation + FILLER_PROMPT

                hidden_states = None
                if self.provider in ("llama_spcs", "llama_k8s") and self._last_hidden_states:
                    hidden_states = self._last_hidden_states
                    self._last_hidden_states = None

                step = AgentStep(
                    step_number=step_num,
                    timestamp=datetime.now().isoformat(),
                    thought=thought,
                    action=action,
                    action_input=action_input,
                    observation=observation,
                    raw_response=response,
                    hidden_states=hidden_states,
                )
                steps.append(step)

                conversation_history.append({
                    "role": "assistant",
                    "content": response
                })
                conversation_history.append({
                    "role": "user",
                    "content": f"Observation: {observation}"
                })

                if action == "Finish":
                    break

            if not success and not error:
                error = f"Reached maximum steps ({self.max_steps}) without finishing"

        except Exception as e:
            error = f"Unexpected error: {str(e)}"

        end_time = datetime.now().isoformat()

        return AgentRun(
            run_id=run_id,
            task_id=task_id,
            model=self.model,
            provider=self.provider,
            question=question,
            steps=steps,
            final_answer=final_answer,
            success=success,
            error=error,
            start_time=start_time,
            end_time=end_time,
        )


async def run_experiment(questions_path: str, output_dir_str: str, condition: str, resume: bool):
    output_dir = Path(output_dir_str) / condition
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)

    with open(questions_path) as f:
        questions = json.load(f)

    n_questions = len(questions)
    total_target = n_questions * N_RUNS_PER_QUESTION
    logger.info(f"Intervention experiment ({condition}): {n_questions} questions x {N_RUNS_PER_QUESTION} runs = {total_target} total")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Condition: {condition}")
    logger.info(f"Intervention step: {INTERVENTION_STEP}")
    logger.info(f"Resume: {resume}")

    if not wait_for_server(logger):
        logger.error("Aborting: server not available")
        sys.exit(1)

    agent = InterventionReActAgent(
        condition=condition,
        model="llama-70b",
        provider="llama_k8s",
        k8s_endpoint=K8S_ENDPOINT,
        max_steps=MAX_STEPS,
        temperature=TEMPERATURE,
    )

    total_runs_done = 0
    total_errors = 0
    total_correct = 0
    question_summaries = []

    for q_idx, q in enumerate(questions):
        qid = q["id"]
        question = q["question"]
        answer = q["answer"]
        difficulty = q.get("difficulty", "unknown")
        context = q["context"]

        completed = get_completed_runs(output_dir, qid) if resume else 0
        if completed >= N_RUNS_PER_QUESTION:
            logger.info(f"Question {q_idx+1}/{n_questions} [{qid}] SKIP (already {completed} runs)")
            total_runs_done += completed
            continue

        start_run = completed + 1
        logger.info(f"Question {q_idx+1}/{n_questions} [{qid}] difficulty={difficulty} condition={condition} starting from run {start_run}")
        logger.info(f"  Q: {question[:80]}...")
        logger.info(f"  Expected: {answer}")

        formatted_context = format_context_for_tools(context)
        search_fn = create_search_fn(formatted_context)
        retrieve_fn = create_retrieve_fn(formatted_context)
        agent._custom_search = search_fn
        agent._custom_retrieve = retrieve_fn

        q_correct = 0
        q_errors = 0

        for run_idx in range(start_run, N_RUNS_PER_QUESTION + 1):
            run_id = f"run_{run_idx:04d}"
            try:
                result = await agent.run(
                    question=question,
                    task_id=qid,
                    run_id=run_id,
                )

                correct = save_run_results(output_dir, qid, run_id, result, question, answer, condition)
                n_steps = len(result.steps)
                hs_count = sum(1 for s in result.steps if s.hidden_states)
                status = "OK" if correct else "WRONG"
                ans_preview = (result.final_answer or "None")[:40]
                logger.info(f"  Run {run_idx}/{N_RUNS_PER_QUESTION}: {status} ({n_steps} steps, {hs_count} HS) -> {ans_preview}")

                if correct:
                    q_correct += 1
                total_runs_done += 1

            except Exception as e:
                q_errors += 1
                total_errors += 1
                total_runs_done += 1
                logger.error(f"  Run {run_idx}/{N_RUNS_PER_QUESTION}: ERROR - {e}")
                logger.error(f"  {traceback.format_exc()}")

                err_dir = output_dir / qid / run_id
                err_dir.mkdir(parents=True, exist_ok=True)
                with open(err_dir / "error.json", "w") as f:
                    json.dump({"error": str(e), "traceback": traceback.format_exc(), "timestamp": datetime.now().isoformat()}, f, indent=2)

                if not wait_for_server(logger):
                    logger.error("Server appears down after error, waiting 60s...")
                    await asyncio.sleep(60)
                    if not wait_for_server(logger):
                        logger.error("Server still down, continuing anyway...")

        actual_runs = N_RUNS_PER_QUESTION - start_run + 1
        logger.info(f"Question {q_idx+1}/{n_questions} complete: {q_correct}/{actual_runs} correct, {q_errors} errors")

        question_summaries.append({
            "question_id": qid,
            "question": question[:80],
            "difficulty": difficulty,
            "expected": answer,
            "condition": condition,
            "runs_completed": actual_runs,
            "correct": q_correct,
            "errors": q_errors,
        })

        total_correct += q_correct

        progress = {
            "condition": condition,
            "intervention_step": INTERVENTION_STEP,
            "questions_completed": q_idx + 1,
            "total_questions": n_questions,
            "total_runs_done": total_runs_done,
            "total_target": total_target,
            "total_correct": total_correct,
            "total_errors": total_errors,
            "last_updated": datetime.now().isoformat(),
            "question_summaries": question_summaries,
        }
        save_progress(output_dir, progress)

    logger.info("=" * 70)
    logger.info(f"INTERVENTION EXPERIMENT ({condition.upper()}) COMPLETE")
    logger.info(f"Total runs: {total_runs_done}/{total_target}")
    logger.info(f"Total correct: {total_correct}")
    logger.info(f"Total errors: {total_errors}")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Intervention experiment: forced commitment")
    parser.add_argument("--questions", required=True, help="Path to questions JSON file")
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument("--condition", required=True, choices=["control", "commitment", "filler"],
                        help="Experimental condition: control, commitment, or filler")
    parser.add_argument("--resume", action="store_true", help="Resume from existing results")
    args = parser.parse_args()
    asyncio.run(run_experiment(args.questions, args.output, args.condition, args.resume))


if __name__ == "__main__":
    main()
