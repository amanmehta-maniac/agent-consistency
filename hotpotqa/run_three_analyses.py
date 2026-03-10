"""
Three additional analyses on existing 40-question hidden state data.

Analysis 1: Hard vs Easy layer-wise correlation (side-by-side)
Analysis 2: Step x Layer heatmap (2D correlation map)
Analysis 3: Nearest-centroid correctness classifier (per-run prediction)

Run from hotpotqa/ directory:
    ../.venv/bin/python3 run_three_analyses.py
"""

import json
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── Directories ──────────────────────────────────────────
PILOT_DIR = Path("pilot_hidden_states_70b")
EASIER_DIR = Path("results_easier")
QUESTIONS_FILE = Path("pilot_questions.json")
EASIER_QUESTIONS_FILE = Path("easier_questions_selection.json")
OUTPUT_DIR = Path("analysis_results/combined_40q")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LAYERS = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80]
STEPS = list(range(1, 6))


# ── Helper functions ─────────────────────────────────────

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


# ── Data loading (same pattern as combined_40q_analysis.py) ──

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
                hidden_states[step_num] = np.load(hs_file)  # (81, 8192)
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
            # Determine correctness from answer matching
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
    """Compute CV, category, and similarity_by_step_layer."""
    runs = entry["runs"]
    question = entry["question"]
    expected = question.get("answer", "").lower().strip()

    step_counts = [r["step_count"] for r in runs]
    cv = np.std(step_counts) / np.mean(step_counts) if np.mean(step_counts) > 0 else 0

    correct_count = sum(1 for r in runs if r.get("correct"))
    correct_rate = correct_count / len(runs) if runs else 0

    if cv < 0.15 and correct_rate >= 0.8:
        category = "consistent-correct"
    elif cv < 0.15 and correct_rate < 0.2:
        category = "consistent-wrong"
    else:
        category = "inconsistent"

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
        "correct_rate": correct_rate,
        "category": category,
        "similarity_by_step_layer": similarity_by_step_layer,
    }


# ══════════════════════════════════════════════════════════
#  ANALYSIS 1: Hard vs Easy Layer-wise Correlation
# ══════════════════════════════════════════════════════════

def analysis1_layerwise_hard_vs_easy(metrics):
    print("\n" + "=" * 70)
    print("ANALYSIS 1: LAYER-WISE CORRELATION — HARD vs EASY")
    print("=" * 70)

    hard = [m for m in metrics if m["source"] == "hard"]
    easy = [m for m in metrics if m["source"] == "easier"]
    print(f"  Hard: {len(hard)} questions, Easy: {len(easy)} questions")

    step = 3  # canonical step for layer comparison

    results = {"hard": {}, "easy": {}}
    for label, subset in [("hard", hard), ("easy", easy)]:
        for layer in LAYERS:
            cvs, sims = [], []
            for m in subset:
                if step in m["similarity_by_step_layer"] and layer in m["similarity_by_step_layer"][step]:
                    sim = m["similarity_by_step_layer"][step][layer]
                    if not np.isnan(sim):
                        cvs.append(m["cv"])
                        sims.append(sim)
            if len(cvs) >= 5:
                r, p = stats.pearsonr(sims, cvs)
                results[label][layer] = {"r": r, "p": p, "n": len(cvs)}

    # Print table
    print(f"\n  {'Layer':>5}  {'Hard r':>8} {'Hard p':>8} {'Hard n':>6}  {'Easy r':>8} {'Easy p':>8} {'Easy n':>6}")
    print("  " + "-" * 62)
    for layer in LAYERS:
        h = results["hard"].get(layer, {})
        e = results["easy"].get(layer, {})
        hr = f"{h['r']:.3f}" if h else "  n/a"
        hp = f"{h['p']:.3f}" if h else "  n/a"
        hn = f"{h.get('n', 0):>4}" if h else " n/a"
        er = f"{e['r']:.3f}" if e else "  n/a"
        ep = f"{e['p']:.3f}" if e else "  n/a"
        en = f"{e.get('n', 0):>4}" if e else " n/a"
        h_sig = "*" if h and h["p"] < 0.05 else " "
        e_sig = "*" if e and e["p"] < 0.05 else " "
        print(f"  {layer:>5}  {hr}{h_sig} {hp} {hn}   {er}{e_sig} {ep} {en}")

    # Plot side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, label, data, color, title in [
        (ax1, "hard", results["hard"], "#d62728", "Hard Questions (n=20)"),
        (ax2, "easy", results["easy"], "#1f77b4", "Easy Questions (n=20)"),
    ]:
        layers_present = sorted(data.keys())
        rs = [data[l]["r"] for l in layers_present]
        ps = [data[l]["p"] for l in layers_present]
        colors = [color if p < 0.05 else "#cccccc" for p in ps]

        bars = ax.bar(range(len(layers_present)), rs, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(layers_present)))
        ax.set_xticklabels([str(l) for l in layers_present], fontsize=9)
        ax.set_xlabel("Layer", fontsize=12)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_title(title, fontsize=13)

        for i, (r, p) in enumerate(zip(rs, ps)):
            sig = "*" if p < 0.05 else ""
            y_off = 0.02 if r >= 0 else -0.04
            ax.text(i, r + y_off, f"{r:.2f}{sig}", ha="center", fontsize=8)

    ax1.set_ylabel("Pearson r (Similarity vs CV)", fontsize=12)
    fig.suptitle("Layer-wise Correlation at Step 3: Hard vs Easy Questions", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "layerwise_hard_vs_easy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: layerwise_hard_vs_easy.png")

    return results


# ══════════════════════════════════════════════════════════
#  ANALYSIS 2: Step x Layer Heatmap
# ══════════════════════════════════════════════════════════

def analysis2_step_layer_heatmap(metrics):
    print("\n" + "=" * 70)
    print("ANALYSIS 2: STEP x LAYER CORRELATION HEATMAP")
    print("=" * 70)

    n_layers = len(LAYERS)
    n_steps = len(STEPS)
    r_matrix = np.full((n_layers, n_steps), np.nan)
    p_matrix = np.full((n_layers, n_steps), np.nan)
    n_matrix = np.full((n_layers, n_steps), 0, dtype=int)

    for si, step in enumerate(STEPS):
        for li, layer in enumerate(LAYERS):
            cvs, sims = [], []
            for m in metrics:
                if step in m["similarity_by_step_layer"] and layer in m["similarity_by_step_layer"][step]:
                    sim = m["similarity_by_step_layer"][step][layer]
                    if not np.isnan(sim):
                        cvs.append(m["cv"])
                        sims.append(sim)
            if len(cvs) >= 5:
                r, p = stats.pearsonr(sims, cvs)
                r_matrix[li, si] = r
                p_matrix[li, si] = p
                n_matrix[li, si] = len(cvs)

    # Print the matrix
    print(f"\n  Pearson r matrix (layers x steps):")
    print(f"  {'':>8}", end="")
    for s in STEPS:
        print(f"  Step {s:>2}", end="")
    print()
    for li, layer in enumerate(LAYERS):
        print(f"  L{layer:>3}   ", end="")
        for si in range(n_steps):
            r = r_matrix[li, si]
            p = p_matrix[li, si]
            if np.isnan(r):
                print(f"    n/a ", end="")
            else:
                sig = "*" if p < 0.05 else " "
                print(f"  {r:>5.3f}{sig}", end="")
        print()

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 9))

    # Diverging colormap centered at 0
    vmax = max(0.5, np.nanmax(np.abs(r_matrix)))
    im = ax.imshow(r_matrix, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax,
                   interpolation="nearest")

    ax.set_xticks(range(n_steps))
    ax.set_xticklabels([f"Step {s}" for s in STEPS], fontsize=11)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"Layer {l}" for l in LAYERS], fontsize=11)
    ax.set_xlabel("ReAct Step", fontsize=13)
    ax.set_ylabel("Transformer Layer", fontsize=13)
    ax.set_title("Correlation Between Activation Similarity and Behavioral CV\n(n=40 questions, Pearson r)", fontsize=14)

    # Annotate cells
    for li in range(n_layers):
        for si in range(n_steps):
            r = r_matrix[li, si]
            p = p_matrix[li, si]
            if np.isnan(r):
                ax.text(si, li, "n/a", ha="center", va="center", fontsize=8, color="gray")
            else:
                sig = "*" if p < 0.05 else ""
                color = "white" if abs(r) > 0.3 else "black"
                ax.text(si, li, f"{r:.2f}{sig}", ha="center", va="center",
                        fontsize=9, fontweight="bold" if sig else "normal", color=color)

    # Add significance border
    for li in range(n_layers):
        for si in range(n_steps):
            p = p_matrix[li, si]
            if not np.isnan(p) and p < 0.05:
                rect = plt.Rectangle((si - 0.5, li - 0.5), 1, 1,
                                     fill=False, edgecolor="gold", linewidth=2.5)
                ax.add_patch(rect)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Pearson r", fontsize=12)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "step_layer_heatm.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: step_layer_heatm.png")

    return r_matrix, p_matrix, n_matrix


# ══════════════════════════════════════════════════════════
#  ANALYSIS 3: Nearest-Centroid Correctness Classifier
# ══════════════════════════════════════════════════════════

def analysis3_correctness_classifier(all_data):
    print("\n" + "=" * 70)
    print("ANALYSIS 3: NEAREST-CENTROID CORRECTNESS CLASSIFIER")
    print("=" * 70)

    step = 3
    target_layers = [72, 80]
    results = {}

    for layer in target_layers:
        print(f"\n  --- Layer {layer}, Step {step} ---")

        all_scores = []   # higher = more likely correct (distance to incorrect - distance to correct)
        all_labels = []   # 1 = correct, 0 = incorrect
        questions_used = 0

        for qid, entry in all_data.items():
            runs = entry["runs"]

            # Collect per-run vectors and correctness labels
            run_vecs = []
            run_correct = []
            for run in runs:
                hs = run["hidden_states"].get(step)
                if hs is not None and len(hs) > layer:
                    run_vecs.append(hs[layer])
                    run_correct.append(bool(run.get("correct")))

            if len(run_vecs) < 3:
                continue

            run_vecs = np.array(run_vecs)  # (n_runs, 8192)
            run_correct = np.array(run_correct)

            n_correct = run_correct.sum()
            n_incorrect = (~run_correct).sum()

            # Need at least 1 of each for centroids
            if n_correct < 1 or n_incorrect < 1:
                continue

            questions_used += 1

            # Leave-one-out evaluation
            for i in range(len(run_vecs)):
                test_vec = run_vecs[i].reshape(1, -1)
                test_label = run_correct[i]

                # Build centroids excluding test run
                mask = np.ones(len(run_vecs), dtype=bool)
                mask[i] = False
                remaining_vecs = run_vecs[mask]
                remaining_labels = run_correct[mask]

                correct_vecs = remaining_vecs[remaining_labels]
                incorrect_vecs = remaining_vecs[~remaining_labels]

                # Need at least 1 of each after LOO exclusion
                if len(correct_vecs) == 0 or len(incorrect_vecs) == 0:
                    continue

                correct_centroid = correct_vecs.mean(axis=0).reshape(1, -1)
                incorrect_centroid = incorrect_vecs.mean(axis=0).reshape(1, -1)

                # Cosine similarity (higher = closer)
                sim_correct = cosine_similarity(test_vec, correct_centroid)[0, 0]
                sim_incorrect = cosine_similarity(test_vec, incorrect_centroid)[0, 0]

                # Score: positive if closer to correct centroid
                score = sim_correct - sim_incorrect
                all_scores.append(score)
                all_labels.append(1 if test_label else 0)

        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)

        print(f"  Questions with mixed correct/incorrect: {questions_used}")
        print(f"  Total runs evaluated: {len(all_labels)}")
        print(f"  Correct runs: {all_labels.sum()}, Incorrect: {len(all_labels) - all_labels.sum()}")

        if len(set(all_labels)) < 2:
            print("  ERROR: Need both classes for AUC computation")
            results[layer] = {"auc": None, "n_questions": questions_used, "n_runs": len(all_labels)}
            continue

        auc_val = roc_auc_score(all_labels, all_scores)
        print(f"  AUC: {auc_val:.4f}")

        # Also report accuracy at threshold=0 (closer to correct → predict correct)
        preds = (all_scores > 0).astype(int)
        accuracy = np.mean(preds == all_labels)
        print(f"  Accuracy (threshold=0): {accuracy:.4f}")

        # Mean score for correct vs incorrect runs
        mean_score_correct = all_scores[all_labels == 1].mean()
        mean_score_incorrect = all_scores[all_labels == 0].mean()
        print(f"  Mean score (correct runs):   {mean_score_correct:.6f}")
        print(f"  Mean score (incorrect runs): {mean_score_incorrect:.6f}")

        results[layer] = {
            "auc": float(auc_val),
            "accuracy_at_zero": float(accuracy),
            "n_questions": questions_used,
            "n_runs": int(len(all_labels)),
            "n_correct": int(all_labels.sum()),
            "n_incorrect": int(len(all_labels) - all_labels.sum()),
            "mean_score_correct": float(mean_score_correct),
            "mean_score_incorrect": float(mean_score_incorrect),
        }

    # Save results
    out_path = OUTPUT_DIR / "correctness_classifier_results.json"
    with open(out_path, "w") as f:
        json.dump({"step": step, "layers": {str(k): v for k, v in results.items()}}, f, indent=2)
    print(f"\n  Saved: correctness_classifier_results.json")

    return results


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("THREE ADDITIONAL ANALYSES ON 40-QUESTION HIDDEN STATE DATA")
    print("=" * 70)

    # Load data
    pilot_questions = load_pilot_questions()
    easier_questions = load_easier_questions()
    pilot_data = load_pilot_data(pilot_questions)
    easier_data = load_easier_data(easier_questions)
    all_data = {**pilot_data, **easier_data}
    print(f"\nTotal questions loaded: {len(all_data)}")

    # Compute metrics
    print("\nComputing metrics...")
    metrics = []
    for qid, entry in all_data.items():
        m = compute_question_metrics(qid, entry)
        metrics.append(m)
    print(f"  Computed metrics for {len(metrics)} questions")

    # Run analyses
    a1_results = analysis1_layerwise_hard_vs_easy(metrics)
    a2_r, a2_p, a2_n = analysis2_step_layer_heatmap(metrics)
    a3_results = analysis3_correctness_classifier(all_data)

    print("\n" + "=" * 70)
    print("ALL ANALYSES COMPLETE")
    print("=" * 70)
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
