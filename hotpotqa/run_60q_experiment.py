#!/usr/bin/env python3
"""
60-question experiment: 60 questions x 10 runs = 600 total runs.
Runs inside a k8s pod alongside the Llama 70B server sidecar.

Usage:
    python run_60q_experiment.py --questions /data/new_60_questions.json --output /results/experiment_60q
    python run_60q_experiment.py --questions /data/new_60_questions.json --output /results/experiment_60q --resume
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

from agent import ReActAgent
from tools import create_search_fn, create_retrieve_fn

N_RUNS_PER_QUESTION = 10
MAX_STEPS = 30
TEMPERATURE = 0.5
K8S_ENDPOINT = "http://localhost:8000"
HEALTH_TIMEOUT = 1200
HEALTH_POLL_INTERVAL = 10


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
                logger.info("Server is healthy")
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


def save_run_results(output_dir, question_id, run_id, result, question, expected_answer):
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


async def run_experiment(questions_path: str, output_dir_str: str, resume: bool):
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)

    with open(questions_path) as f:
        questions = json.load(f)

    n_questions = len(questions)
    total_target = n_questions * N_RUNS_PER_QUESTION
    logger.info(f"Experiment: {n_questions} questions x {N_RUNS_PER_QUESTION} runs = {total_target} total")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Resume: {resume}")

    if not wait_for_server(logger):
        logger.error("Aborting: server not available")
        sys.exit(1)

    agent = ReActAgent(
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
        logger.info(f"Question {q_idx+1}/{n_questions} [{qid}] difficulty={difficulty} starting from run {start_run}")
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

                correct = save_run_results(output_dir, qid, run_id, result, question, answer)
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
            "runs_completed": actual_runs,
            "correct": q_correct,
            "errors": q_errors,
        })

        total_correct += q_correct

        progress = {
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
    logger.info("EXPERIMENT COMPLETE")
    logger.info(f"Total runs: {total_runs_done}/{total_target}")
    logger.info(f"Total correct: {total_correct}")
    logger.info(f"Total errors: {total_errors}")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="60-question hidden state experiment")
    parser.add_argument("--questions", required=True, help="Path to questions JSON file")
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument("--resume", action="store_true", help="Resume from existing results")
    args = parser.parse_args()
    asyncio.run(run_experiment(args.questions, args.output, args.resume))


if __name__ == "__main__":
    main()
