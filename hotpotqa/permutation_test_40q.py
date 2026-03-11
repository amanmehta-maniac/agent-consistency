"""
Permutation test and exact p-values for the 40-question hidden state data.

For key step×layer cells, shuffles CV labels 10,000 times to build null
distribution of Pearson r, then reports exact permutation p-values alongside
parametric p-values.

Run from hotpotqa/ directory:
    ../.venv/bin/python3 permutation_test_40q.py
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PILOT_DIR = Path("pilot_hidden_states_70b")
EASIER_DIR = Path("results_easier")
QUESTIONS_FILE = Path("pilot_questions.json")
EASIER_QUESTIONS_FILE = Path("easier_questions_selection.json")
OUTPUT_DIR = Path("analysis_results/combined_40q")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LAYERS = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80]
STEPS = list(range(1, 6))
N_PERMUTATIONS = 10000
RNG_SEED = 42


def extract_answer(text):
    if not text:
        return None
    text = str(text).lower().strip()
    if "yes" in text and "no" not in text:
        return "yes"
    elif "no" in text:
        return "no"
    elif text.startswith("yes"):
        return "yes"
    return None


def compute_pairwise_cosine_similarity(vectors):
    if len(vectors) < 2:
        return np.nan
    arr = np.array(vectors)
    sim_matrix = cosine_similarity(arr)
    n = len(vectors)
    upper_tri = sim_matrix[np.triu_indices(n, k=1)]
    return np.mean(upper_tri)


def load_pilot_questions():
    with open(QUESTIONS_FILE) as f:
        return {q["id"]: q for q in json.load(f)}


def load_easier_questions():
    with open(EASIER_QUESTIONS_FILE) as f:
        return {q["id"]: q for q in json.load(f)}


def load_pilot_data(pilot_questions):
    print("Loading 20 hard questions (npy format)...")
    results = {}
    for qdir in sorted(PILOT_DIR.iterdir()):
        if not qdir.is_dir():
            continue
        qid = qdir.name
        if qid not in pilot_questions:
            continue
        runs_data = []
        for run_dir in sorted(qdir.glob("run_*")):
            meta_file = run_dir / "metadata.json"
            traj_file = run_dir / "trajectory.json"
            if not meta_file.exists() or not traj_file.exists():
                continue
            with open(meta_file) as f:
                meta = json.load(f)
            with open(traj_file) as f:
                traj = json.load(f)
            hidden_states = {}
            for hs_file in run_dir.glob("hidden_states_step_*.npy"):
                step_num = int(hs_file.stem.split("_")[-1])
                hidden_states[step_num] = np.load(hs_file)
            runs_data.append({
                "run_id": run_dir.name,
                "final_answer": traj.get("final_answer") or meta.get("agent_answer"),
                "step_count": len(traj.get("steps", [])),
                "hidden_states": hidden_states,
                "correct": meta.get("correct"),
            })
        if runs_data:
            results[qid] = {"question": pilot_questions[qid], "runs": runs_data, "source": "hard"}
            print(f"  {qid}: {len(runs_data)} runs")
    return results


def load_easier_data(easier_questions):
    print("\nLoading 20 easier questions (json format)...")
    results = {}
    for fp in sorted(EASIER_DIR.glob("*.json")):
        qid = fp.stem
        if qid not in easier_questions:
            continue
        print(f"  {qid}...", end=" ", flush=True)
        with open(fp) as f:
            data = json.load(f)
        expected = easier_questions[qid].get("answer", "").lower().strip()
        runs_data = []
        for run in data.get("runs", []):
            hidden_states = {}
            for step in run.get("steps", []):
                step_num = step.get("step_number", 0)
                layers_dict = step.get("hidden_states", {}).get("layers", {})
                if layers_dict:
                    layer_vecs = []
                    for i in range(81):
                        lk = f"layer_{i}"
                        layer_vecs.append(layers_dict[lk] if lk in layers_dict and layers_dict[lk] else [0]*8192)
                    hidden_states[step_num] = np.array(layer_vecs, dtype=np.float32)
            ans = extract_answer(run.get("final_answer"))
            correct = (ans == expected) if ans else False
            runs_data.append({
                "run_id": run.get("run_id"),
                "final_answer": run.get("final_answer"),
                "step_count": len(run.get("steps", [])),
                "hidden_states": hidden_states,
                "correct": correct,
            })
        if runs_data:
            results[qid] = {"question": easier_questions[qid], "runs": runs_data, "source": "easier"}
            print(f"{len(runs_data)} runs")
    return results


def compute_question_metrics(qid, entry):
    runs = entry["runs"]
    step_counts = [r["step_count"] for r in runs]
    cv = np.std(step_counts) / np.mean(step_counts) if np.mean(step_counts) > 0 else 0

    similarity_by_step_layer = {}
    for step in STEPS:
        similarity_by_step_layer[step] = {}
        for layer in LAYERS:
            vectors = []
            for run in runs:
                hs = run["hidden_states"].get(step)
                if hs is not None and len(hs) > layer:
                    vectors.append(hs[layer])
            if len(vectors) >= 2:
                similarity_by_step_layer[step][layer] = compute_pairwise_cosine_similarity(vectors)

    return {
        "qid": qid,
        "source": entry["source"],
        "cv": cv,
        "similarity_by_step_layer": similarity_by_step_layer,
    }


def permutation_test(sims, cvs, n_perms, rng):
    observed_r, parametric_p = stats.pearsonr(sims, cvs)

    cvs_arr = np.array(cvs)
    sims_arr = np.array(sims)
    n = len(cvs_arr)

    sims_centered = sims_arr - sims_arr.mean()
    sims_std = np.sqrt(np.sum(sims_centered**2))

    null_rs = np.empty(n_perms)
    for i in range(n_perms):
        perm_cvs = rng.permutation(cvs_arr)
        perm_centered = perm_cvs - perm_cvs.mean()
        perm_std = np.sqrt(np.sum(perm_centered**2))
        if sims_std == 0 or perm_std == 0:
            null_rs[i] = 0.0
        else:
            null_rs[i] = np.dot(sims_centered, perm_centered) / (sims_std * perm_std)

    perm_p = (np.sum(np.abs(null_rs) >= np.abs(observed_r)) + 1) / (n_perms + 1)

    return observed_r, parametric_p, perm_p, null_rs


def main():
    print("=" * 70)
    print("PERMUTATION TEST ON 40-QUESTION HIDDEN STATE DATA")
    print(f"N permutations: {N_PERMUTATIONS:,}")
    print("=" * 70)

    pilot_questions = load_pilot_questions()
    easier_questions = load_easier_questions()
    pilot_data = load_pilot_data(pilot_questions)
    easier_data = load_easier_data(easier_questions)
    all_data = {**pilot_data, **easier_data}
    print(f"\nTotal questions loaded: {len(all_data)}")

    print("\nComputing metrics...")
    metrics = []
    for qid, entry in all_data.items():
        m = compute_question_metrics(qid, entry)
        metrics.append(m)
    print(f"  Computed metrics for {len(metrics)} questions")

    rng = np.random.default_rng(RNG_SEED)

    key_cells = []
    for step in STEPS:
        for layer in LAYERS:
            key_cells.append((step, layer))

    print(f"\nRunning permutation tests on {len(key_cells)} step×layer cells...")
    print(f"{'Step':>4} {'Layer':>5} {'n':>4} {'Obs r':>8} {'Param p':>10} {'Perm p':>10} {'Sig':>5}")
    print("-" * 55)

    results = {}
    significant_cells = []

    for step, layer in key_cells:
        cvs, sims = [], []
        for m in metrics:
            if step in m["similarity_by_step_layer"] and layer in m["similarity_by_step_layer"][step]:
                sim = m["similarity_by_step_layer"][step][layer]
                if not np.isnan(sim):
                    cvs.append(m["cv"])
                    sims.append(sim)

        if len(cvs) < 5:
            continue

        observed_r, parametric_p, perm_p, null_rs = permutation_test(sims, cvs, N_PERMUTATIONS, rng)

        sig = "*" if perm_p < 0.05 else ""
        print(f"{step:>4} {layer:>5} {len(cvs):>4} {observed_r:>8.4f} {parametric_p:>10.6f} {perm_p:>10.6f} {sig:>5}")

        cell_key = f"step{step}_layer{layer}"
        results[cell_key] = {
            "step": step,
            "layer": layer,
            "n": len(cvs),
            "observed_r": float(observed_r),
            "parametric_p": float(parametric_p),
            "permutation_p": float(perm_p),
            "significant_parametric": parametric_p < 0.05,
            "significant_permutation": perm_p < 0.05,
        }

        if perm_p < 0.05:
            significant_cells.append((step, layer, observed_r, perm_p))

    print(f"\n{'=' * 55}")
    print(f"SIGNIFICANT CELLS (permutation p < 0.05): {len(significant_cells)}")
    print(f"{'=' * 55}")
    for step, layer, r, p in sorted(significant_cells, key=lambda x: x[3]):
        print(f"  Step {step}, Layer {layer}: r={r:.4f}, perm_p={p:.6f}")

    focus_cells = [
        ("step3_layer64", "Step 3 × Layer 64"),
        ("step3_layer72", "Step 3 × Layer 72"),
        ("step3_layer80", "Step 3 × Layer 80"),
        ("step4_layer32", "Step 4 × Layer 32"),
        ("step4_layer48", "Step 4 × Layer 48"),
        ("step4_layer64", "Step 4 × Layer 64"),
        ("step4_layer72", "Step 4 × Layer 72"),
        ("step4_layer80", "Step 4 × Layer 80"),
    ]

    print(f"\n{'=' * 55}")
    print("KEY CELLS COMPARISON")
    print(f"{'=' * 55}")
    print(f"{'Cell':<22} {'r':>7} {'Param p':>10} {'Perm p':>10}")
    print("-" * 55)
    for key, label in focus_cells:
        if key in results:
            r = results[key]
            print(f"{label:<22} {r['observed_r']:>7.4f} {r['parametric_p']:>10.6f} {r['permutation_p']:>10.6f}")

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    for idx, (key, label) in enumerate(focus_cells):
        if key not in results:
            continue
        ax = axes[idx]
        r_info = results[key]
        step, layer = r_info["step"], r_info["layer"]

        cvs, sims = [], []
        for m in metrics:
            if step in m["similarity_by_step_layer"] and layer in m["similarity_by_step_layer"][step]:
                sim = m["similarity_by_step_layer"][step][layer]
                if not np.isnan(sim):
                    cvs.append(m["cv"])
                    sims.append(sim)

        _, _, _, null_rs = permutation_test(sims, cvs, N_PERMUTATIONS, rng)

        ax.hist(null_rs, bins=50, color="#cccccc", edgecolor="gray", alpha=0.8)
        ax.axvline(x=r_info["observed_r"], color="red", linewidth=2, label=f"Observed r={r_info['observed_r']:.3f}")
        ax.axvline(x=-r_info["observed_r"], color="red", linewidth=2, linestyle="--", alpha=0.5)
        ax.set_title(f"{label}\nperm p={r_info['permutation_p']:.4f}", fontsize=10)
        ax.set_xlabel("r (null)", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")

    plt.suptitle("Permutation Test Null Distributions (10,000 permutations)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "permutation_test_nulls.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: permutation_test_nulls.png")

    output = {
        "n_permutations": N_PERMUTATIONS,
        "seed": RNG_SEED,
        "n_questions": len(metrics),
        "cells": results,
        "n_significant_parametric": sum(1 for r in results.values() if r["significant_parametric"]),
        "n_significant_permutation": sum(1 for r in results.values() if r["significant_permutation"]),
    }

    out_path = OUTPUT_DIR / "permutation_test_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda o: bool(o) if isinstance(o, np.bool_) else o)
    print(f"  Saved: permutation_test_results.json")

    print(f"\n{'=' * 55}")
    print(f"SUMMARY: {output['n_significant_permutation']}/{len(results)} cells significant by permutation test")
    print(f"         {output['n_significant_parametric']}/{len(results)} cells significant by parametric test")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
