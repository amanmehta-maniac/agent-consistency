"""
Compute steering vectors for activation steering experiment.
For each inconsistent question, compute:
1. Centroid (mean hidden state at step 4, layer 72 across 10 runs)
2. Steering vectors for each run (run_state - centroid)
"""
import numpy as np
import json
import os
from pathlib import Path

# Configuration
HIDDEN_STATES_DIR = Path("/Users/amehta/research/agent-consistency/hotpotqa/pilot_hidden_states_70b")
OUTPUT_DIR = Path("/Users/amehta/research/agent-consistency/hotpotqa/steering_vectors")
METRICS_FILE = Path("/Users/amehta/research/agent-consistency/hotpotqa/analysis_results/question_metrics.json")

STEP = 4  # Intervention step
LAYER = 72  # Target layer (0-80)
N_RUNS = 10
TOP_K = 5  # Number of most inconsistent questions

def load_question_metrics():
    """Load and sort questions by CV (highest first)."""
    with open(METRICS_FILE) as f:
        data = json.load(f)
    return sorted(data, key=lambda x: x.get('cv_steps', 0), reverse=True)

def get_top_inconsistent_questions(k=5):
    """Get top k most inconsistent questions that have hidden states."""
    metrics = load_question_metrics()
    available = []
    
    for q in metrics:
        qid = q['question_id']
        if (HIDDEN_STATES_DIR / qid).exists():
            available.append(q)
            if len(available) >= k:
                break
    
    return available

def load_hidden_states(qid: str, run: int, step: int = STEP) -> np.ndarray:
    """Load hidden states for a specific question/run/step."""
    path = HIDDEN_STATES_DIR / qid / f"run_{run:04d}" / f"hidden_states_step_{step}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    return np.load(path)

def compute_steering_vectors(qid: str, layer: int = LAYER) -> dict:
    """
    Compute centroid and steering vectors for a question.
    
    Returns:
        dict with 'centroid', 'steering_vectors', 'run_states'
    """
    # Load all run states at target layer
    run_states = []
    for run in range(1, N_RUNS + 1):
        try:
            hs = load_hidden_states(qid, run, STEP)
            # Extract target layer (shape: 8192)
            layer_state = hs[layer]
            run_states.append(layer_state)
        except FileNotFoundError as e:
            print(f"  Warning: {e}")
            continue
    
    run_states = np.array(run_states)  # (n_runs, 8192)
    
    # Compute centroid (mean across runs)
    centroid = np.mean(run_states, axis=0)  # (8192,)
    
    # Compute steering vectors (push away from centroid)
    steering_vectors = run_states - centroid  # (n_runs, 8192)
    
    # Normalize each steering vector
    norms = np.linalg.norm(steering_vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)  # Avoid division by zero
    steering_vectors_normalized = steering_vectors / norms
    
    return {
        'centroid': centroid,
        'steering_vectors': steering_vectors,
        'steering_vectors_normalized': steering_vectors_normalized,
        'run_states': run_states,
        'norms': norms.flatten()
    }

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get top inconsistent questions
    questions = get_top_inconsistent_questions(TOP_K)
    print(f"Processing {len(questions)} most inconsistent questions:")
    
    results = {}
    for i, q in enumerate(questions, 1):
        qid = q['question_id']
        print(f"\n{i}. {qid}")
        print(f"   Question: {q['question'][:60]}...")
        print(f"   CV: {q['cv_steps']:.3f}, Unique answers: {q['unique_answers']}")
        
        # Compute steering vectors
        data = compute_steering_vectors(qid, LAYER)
        
        # Save individual files
        qdir = OUTPUT_DIR / qid
        qdir.mkdir(exist_ok=True)
        
        np.save(qdir / "centroid.npy", data['centroid'])
        np.save(qdir / "steering_vectors.npy", data['steering_vectors'])
        np.save(qdir / "steering_vectors_normalized.npy", data['steering_vectors_normalized'])
        np.save(qdir / "run_states.npy", data['run_states'])
        
        # Store metadata
        results[qid] = {
            'question': q['question'],
            'cv_steps': q['cv_steps'],
            'unique_answers': q['unique_answers'],
            'accuracy': q['accuracy'],
            'mean_norm': float(np.mean(data['norms'])),
            'std_norm': float(np.std(data['norms'])),
            'centroid_norm': float(np.linalg.norm(data['centroid'])),
            'layer': LAYER,
            'step': STEP,
            'n_runs': len(data['run_states'])
        }
        
        print(f"   Centroid norm: {results[qid]['centroid_norm']:.2f}")
        print(f"   Mean steering norm: {results[qid]['mean_norm']:.2f} +/- {results[qid]['std_norm']:.2f}")
    
    # Save summary
    with open(OUTPUT_DIR / "steering_summary.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\nSaved steering vectors to {OUTPUT_DIR}")
    print(f"Questions processed: {list(results.keys())}")

if __name__ == "__main__":
    main()
