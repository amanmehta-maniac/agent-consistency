"""
Cross-model analysis: Compare Qwen 2.5 72B vs Llama 3.1 70B on the same 20 questions.

Step 1: Extract Llama 70B results for the 20-question subset (from existing data).
Step 2: After Qwen experiment completes, load Qwen results.
Step 3: Compute step 4 layer-wise correlation for both models.
Step 4: Compare profiles.

Run from hotpotqa/ directory:
    ../.venv/bin/python3 analyze_cross_model.py --llama-only
    ../.venv/bin/python3 analyze_cross_model.py  # after Qwen results are available
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import Counter
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = Path("analysis_results/combined_100q")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PILOT_DIR = Path("pilot_hidden_states_70b")
EASIER_DIR = Path("results_easier")
NEW60_DIR = Path("experiment_60q_results")
QWEN_DIR = Path("qwen_cross_model_results")

LAYERS = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80]
STEPS = list(range(1, 6))


def compute_pairwise_cosine_similarity(vectors):
    if len(vectors) < 2:
        return np.nan
    arr = np.array(vectors)
    sim_matrix = cosine_similarity(arr)
    n = len(vectors)
    upper_tri = sim_matrix[np.triu_indices(n, k=1)]
    return np.mean(upper_tri)


def extract_answer(text):
    if not text:
        return None
    text = str(text).lower().strip()
    if "yes" in text and "no" not in text:
        return "yes"
    elif "no" in text:
        return "no"
    return text[:100]


def load_llama_subset(question_ids):
    with open("cross_model_20q.json") as f:
        questions = {q["id"]: q for q in json.load(f)}

    results = {}

    for qid in question_ids:
        for data_dir in [PILOT_DIR, NEW60_DIR]:
            qdir = data_dir / qid
            if qdir.exists():
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
                    results[qid] = {
                        "question": questions[qid],
                        "runs": runs_data,
                        "difficulty": questions[qid].get("difficulty", "unknown"),
                    }
                break

        if qid not in results:
            easier_file = EASIER_DIR / f"{qid}.json"
            if easier_file.exists():
                with open(easier_file) as f:
                    data = json.load(f)
                expected = questions[qid].get("answer", "").lower().strip()
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
                    ans_raw = run.get("final_answer")
                    ans = extract_answer(ans_raw)
                    correct = (ans == expected) if ans else False
                    runs_data.append({
                        "run_id": run.get("run_id"),
                        "final_answer": ans_raw,
                        "step_count": len(run.get("steps", [])),
                        "hidden_states": hidden_states,
                        "correct": correct,
                    })
                if runs_data:
                    results[qid] = {
                        "question": questions[qid],
                        "runs": runs_data,
                        "difficulty": questions[qid].get("difficulty", "easy"),
                    }

    return results


def load_qwen_results(question_ids):
    if not QWEN_DIR.exists():
        return None

    with open("cross_model_20q.json") as f:
        questions = {q["id"]: q for q in json.load(f)}

    results = {}
    for qid in question_ids:
        qdir = QWEN_DIR / qid
        if not qdir.exists():
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
            results[qid] = {
                "question": questions[qid],
                "runs": runs_data,
                "difficulty": questions[qid].get("difficulty", "unknown"),
            }

    return results if results else None


def compute_model_metrics(data):
    metrics = []
    for qid, entry in data.items():
        runs = entry["runs"]
        step_counts = [r["step_count"] for r in runs]
        cv = np.std(step_counts) / np.mean(step_counts) if np.mean(step_counts) > 0 else 0
        correct_rate = sum(1 for r in runs if r.get("correct")) / len(runs)

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

        metrics.append({
            "qid": qid,
            "difficulty": entry["difficulty"],
            "cv": cv,
            "correct_rate": correct_rate,
            "n_runs": len(runs),
            "mean_steps": float(np.mean(step_counts)),
            "similarity_by_step_layer": similarity_by_step_layer,
        })
    return metrics


def compute_correlations(metrics, model_name):
    print(f"\n  --- {model_name}: Step 4 layer-wise correlations ---")
    print(f"  {'Layer':>5} {'n':>4} {'r':>8} {'p':>10} {'Sig':>5}")
    print("  " + "-" * 40)

    results = {}
    for layer in LAYERS:
        step = 4
        cvs, sims = [], []
        for m in metrics:
            if step in m["similarity_by_step_layer"] and layer in m["similarity_by_step_layer"][step]:
                sim = m["similarity_by_step_layer"][step][layer]
                if not np.isnan(sim):
                    cvs.append(m["cv"])
                    sims.append(sim)
        if len(cvs) >= 5:
            r, p = stats.pearsonr(sims, cvs)
            sig = "*" if p < 0.05 else ""
            print(f"  {layer:>5} {len(cvs):>4} {r:>8.4f} {p:>10.6f} {sig:>5}")
            results[layer] = {"r": float(r), "p": float(p), "n": len(cvs)}
        else:
            print(f"  {layer:>5} {len(cvs):>4}      n/a        n/a")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-only", action="store_true", help="Only extract Llama subset (before Qwen results)")
    args = parser.parse_args()

    print("=" * 70)
    print("CROSS-MODEL VALIDATION ANALYSIS")
    print("=" * 70)

    with open("cross_model_20q.json") as f:
        questions = json.load(f)
    question_ids = [q["id"] for q in questions]
    print(f"Questions: {len(question_ids)} (10 hard, 10 easy)")

    print("\nLoading Llama 3.1 70B subset...")
    llama_data = load_llama_subset(question_ids)
    print(f"  Loaded: {len(llama_data)} questions")
    llama_metrics = compute_model_metrics(llama_data)

    llama_corrs = compute_correlations(llama_metrics, "Llama 3.1 70B")

    llama_summary = {
        "model": "Llama 3.1 70B",
        "n_questions": len(llama_data),
        "mean_accuracy": float(np.mean([m["correct_rate"] for m in llama_metrics])),
        "mean_cv": float(np.mean([m["cv"] for m in llama_metrics])),
        "step4_correlations": {str(k): v for k, v in llama_corrs.items()},
    }

    qwen_data = None
    qwen_corrs = None
    qwen_summary = None

    if not args.llama_only:
        print("\nLoading Qwen 2.5 72B results...")
        qwen_data = load_qwen_results(question_ids)
        if qwen_data:
            print(f"  Loaded: {len(qwen_data)} questions")
            qwen_metrics = compute_model_metrics(qwen_data)
            qwen_corrs = compute_correlations(qwen_metrics, "Qwen 2.5 72B")

            qwen_summary = {
                "model": "Qwen 2.5 72B",
                "n_questions": len(qwen_data),
                "mean_accuracy": float(np.mean([m["correct_rate"] for m in qwen_metrics])),
                "mean_cv": float(np.mean([m["cv"] for m in qwen_metrics])),
                "step4_correlations": {str(k): v for k, v in qwen_corrs.items()},
            }
        else:
            print("  Qwen results not found yet. Run with --llama-only or wait for experiment.")

    fig_cols = 2 if qwen_corrs else 1
    fig, axes = plt.subplots(1, fig_cols, figsize=(7 * fig_cols, 5), squeeze=False)

    for col, (corrs, model_name, color) in enumerate([
        (llama_corrs, "Llama 3.1 70B", "#d62728"),
        (qwen_corrs, "Qwen 2.5 72B", "#1f77b4"),
    ]):
        if corrs is None:
            continue
        ax = axes[0][col]
        layers_plot = sorted(corrs.keys())
        rs = [corrs[l]["r"] for l in layers_plot]
        ps = [corrs[l]["p"] for l in layers_plot]
        colors = [color if p < 0.05 else "#cccccc" for p in ps]
        ax.bar(range(len(layers_plot)), rs, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(layers_plot)))
        ax.set_xticklabels([str(l) for l in layers_plot], fontsize=9)
        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("Pearson r (Similarity vs CV)", fontsize=12)
        ax.set_title(f"{model_name} (n=20)", fontsize=13)
        ax.axhline(y=0, color="black", linewidth=0.5)
        for i, (r, p) in enumerate(zip(rs, ps)):
            sig = "*" if p < 0.05 else ""
            y_off = 0.02 if r >= 0 else -0.04
            ax.text(i, r + y_off, f"{r:.2f}{sig}", ha="center", fontsize=8)
        ax.set_ylim(-0.8, 0.4)

    title = "Cross-Model Validation: Step 4 Correlation Profile"
    if qwen_corrs:
        title += "\nLlama 3.1 70B vs Qwen 2.5 72B (same 20 questions)"
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cross_model_qwen72b.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: cross_model_qwen72b.png")

    result = {
        "question_ids": question_ids,
        "llama": llama_summary,
        "qwen": qwen_summary,
    }
    with open(OUTPUT_DIR / "cross_model_qwen72b.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: cross_model_qwen72b.json")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Llama 3.1 70B: {llama_summary['n_questions']} questions, "
          f"accuracy={llama_summary['mean_accuracy']:.2f}, CV={llama_summary['mean_cv']:.3f}")
    if qwen_summary:
        print(f"  Qwen 2.5 72B:  {qwen_summary['n_questions']} questions, "
              f"accuracy={qwen_summary['mean_accuracy']:.2f}, CV={qwen_summary['mean_cv']:.3f}")


if __name__ == "__main__":
    main()
