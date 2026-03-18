#!/usr/bin/env python3
"""
B1: t-SNE/PCA visualization of commitment categories.

Produces a 2D projection of step-4 hidden states (mean across runs per question)
colored by commitment category, for both Llama and Qwen.

Run from hotpotqa/ directory:
    ../.venv/bin/python3 commitment_tsne.py
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Data directories ──
PILOT_DIR = Path("pilot_hidden_states_70b")
EASIER_DIR = Path("results_easier")
NEW60_DIR = Path("experiment_60q_results")
QWEN_DIR = Path("qwen_cross_model_100q")

PILOT_Q_FILE = Path("pilot_questions.json")
EASIER_Q_FILE = Path("easier_questions_selection.json")
NEW60_Q_FILE = Path("new_60_questions.json")

OUTPUT_DIR = Path("../paper3_prep/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STEP = 4
LLAMA_LAYER = 40   # index into 81-layer array
QWEN_LAYER = 40    # same index for comparable visualization

# ── Category colors ──
COLORS = {
    "committed-correct": "#2ecc71",
    "committed-wrong":   "#e74c3c",
    "uncommitted-wrong": "#f39c12",
    "mixed":             "#95a5a6",
}
LABELS = {
    "committed-correct": "Committed correct",
    "committed-wrong":   "Committed wrong",
    "uncommitted-wrong": "Uncommitted wrong",
    "mixed":             "Mixed",
}


def load_llama_data():
    """Load all 100 Llama questions, returning per-question data."""
    results = {}

    # 20 hard (pilot, npy format)
    with open(PILOT_Q_FILE) as f:
        pilot_qs = {q["id"]: q for q in json.load(f)}
    for qdir in sorted(PILOT_DIR.iterdir()):
        if not qdir.is_dir():
            continue
        qid = qdir.name
        if qid not in pilot_qs:
            continue
        runs = []
        for run_dir in sorted(qdir.glob("run_*")):
            meta_file = run_dir / "metadata.json"
            hs_file = run_dir / f"hidden_states_step_{STEP}.npy"
            if not meta_file.exists() or not hs_file.exists():
                continue
            with open(meta_file) as f:
                meta = json.load(f)
            hs = np.load(hs_file)
            runs.append({"correct": meta.get("correct"), "step_count": len(meta.get("actions", [])),
                          "hidden_state": hs[LLAMA_LAYER]})
        if runs:
            results[qid] = runs

    # 20 easy (json format with inline hidden states)
    with open(EASIER_Q_FILE) as f:
        easier_qs = {q["id"]: q for q in json.load(f)}
    for fp in sorted(EASIER_DIR.glob("*.json")):
        qid = fp.stem
        if qid not in easier_qs:
            continue
        with open(fp) as f:
            data = json.load(f)
        expected = easier_qs[qid].get("answer", "").lower().strip()
        runs = []
        for run in data.get("runs", []):
            steps = run.get("steps", [])
            # Find step 4 hidden states
            hs_vec = None
            for step in steps:
                if step.get("step_number") == STEP:
                    layers_dict = step.get("hidden_states", {}).get("layers", {})
                    if layers_dict:
                        lk = f"layer_{LLAMA_LAYER}"
                        if lk in layers_dict and layers_dict[lk]:
                            hs_vec = np.array(layers_dict[lk], dtype=np.float32)
                    break
            if hs_vec is None:
                continue
            ans = str(run.get("final_answer", "")).lower().strip()
            correct = expected in ans or ans in expected
            runs.append({"correct": correct, "step_count": len(steps), "hidden_state": hs_vec})
        if runs:
            results[qid] = runs

    # 60 new (npy format)
    with open(NEW60_Q_FILE) as f:
        new60_qs = {q["id"]: q for q in json.load(f)}
    for qdir in sorted(NEW60_DIR.iterdir()):
        if not qdir.is_dir():
            continue
        qid = qdir.name
        if qid not in new60_qs:
            continue
        runs = []
        for run_dir in sorted(qdir.glob("run_*")):
            meta_file = run_dir / "metadata.json"
            hs_file = run_dir / f"hidden_states_step_{STEP}.npy"
            if not meta_file.exists() or not hs_file.exists():
                continue
            with open(meta_file) as f:
                meta = json.load(f)
            hs = np.load(hs_file)
            runs.append({"correct": meta.get("correct"), "step_count": len(meta.get("actions", [])),
                          "hidden_state": hs[LLAMA_LAYER]})
        if runs:
            results[qid] = runs

    return results


def load_qwen_data():
    """Load Qwen data for all questions with step-4 hidden states."""
    results = {}
    for qdir in sorted(QWEN_DIR.iterdir()):
        if not qdir.is_dir():
            continue
        qid = qdir.name
        runs = []
        for run_dir in sorted(qdir.glob("run_*")):
            meta_file = run_dir / "metadata.json"
            hs_file = run_dir / f"hidden_states_step_{STEP}.npy"
            if not meta_file.exists() or not hs_file.exists():
                continue
            with open(meta_file) as f:
                meta = json.load(f)
            hs = np.load(hs_file)
            runs.append({"correct": meta.get("correct"), "step_count": len(meta.get("actions", [])),
                          "hidden_state": hs[QWEN_LAYER]})
        if runs:
            results[qid] = runs
    return results


def classify_questions(data):
    """Classify each question into commitment categories.

    Returns: list of (mean_hidden_state, category, qid) tuples.
    """
    classified = []
    for qid, runs in data.items():
        if len(runs) < 2:
            continue
        # Compute metrics
        correct_rate = sum(1 for r in runs if r["correct"]) / len(runs)
        step_counts = [r["step_count"] for r in runs]
        mean_sc = np.mean(step_counts)
        cv = np.std(step_counts) / mean_sc if mean_sc > 0 else 0

        # Mean hidden state across runs
        hs_vecs = [r["hidden_state"] for r in runs if r["hidden_state"] is not None]
        if len(hs_vecs) < 2:
            continue
        mean_hs = np.mean(hs_vecs, axis=0)

        # Classify
        if correct_rate >= 0.8:
            cat = "committed-correct"
        elif correct_rate <= 0.2 and cv <= 0.15:
            cat = "committed-wrong"
        elif correct_rate <= 0.2 and cv > 0.15:
            cat = "uncommitted-wrong"
        else:
            cat = "mixed"

        classified.append((mean_hs, cat, qid))
    return classified


def make_tsne_plot(classified, model_name, ax, perplexity=20, seed=42):
    """Run PCA→t-SNE and plot on given axes."""
    vectors = np.array([c[0] for c in classified])
    categories = [c[1] for c in classified]

    # PCA to 50 dims first
    n_components = min(50, vectors.shape[0] - 1, vectors.shape[1])
    pca = PCA(n_components=n_components, random_state=seed)
    pca_result = pca.fit_transform(vectors)
    var_explained = sum(pca.explained_variance_ratio_[:n_components]) * 100
    print(f"  {model_name}: PCA to {n_components}D captures {var_explained:.1f}% variance")

    # t-SNE to 2D
    perp = min(perplexity, len(vectors) - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=seed,
                max_iter=2000, learning_rate="auto", init="pca")
    embedding = tsne.fit_transform(pca_result)

    # Plot each category
    for cat in ["committed-correct", "committed-wrong", "uncommitted-wrong", "mixed"]:
        mask = [i for i, c in enumerate(categories) if c == cat]
        if not mask:
            continue
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=COLORS[cat], label=f"{LABELS[cat]} ({len(mask)})",
                   s=50, alpha=0.8, edgecolors="white", linewidths=0.5, zorder=3)

    ax.set_title(model_name, fontsize=12, fontweight="bold")
    ax.set_xlabel("t-SNE 1", fontsize=9)
    ax.set_ylabel("t-SNE 2", fontsize=9)
    ax.tick_params(labelsize=8)

    # Count categories
    counts = {}
    for cat in categories:
        counts[cat] = counts.get(cat, 0) + 1
    print(f"  {model_name} categories: {counts}")

    return embedding, categories


def main():
    print("=" * 60)
    print("B1: t-SNE visualization of commitment categories")
    print("=" * 60)

    # Load data
    print("\nLoading Llama data...")
    llama_data = load_llama_data()
    print(f"  Loaded {len(llama_data)} questions")

    print("\nLoading Qwen data...")
    qwen_data = load_qwen_data()
    print(f"  Loaded {len(qwen_data)} questions")

    # Classify
    print("\nClassifying questions...")
    llama_classified = classify_questions(llama_data)
    qwen_classified = classify_questions(qwen_data)
    print(f"  Llama: {len(llama_classified)} questions classified")
    print(f"  Qwen:  {len(qwen_classified)} questions classified")

    # Create two-panel figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    print("\nRunning t-SNE...")
    make_tsne_plot(llama_classified, "Llama 3.1 70B", ax1, perplexity=20)
    make_tsne_plot(qwen_classified, "Qwen 2.5 72B", ax2, perplexity=20)

    # Shared legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.02), frameon=True)

    fig.suptitle("t-SNE of Step-4 Hidden States by Commitment Category",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)

    # Save
    out_pdf = OUTPUT_DIR / "commitment_tsne.pdf"
    out_png = OUTPUT_DIR / "commitment_tsne.png"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    print(f"\nSaved: {out_pdf}")
    print(f"Saved: {out_png}")
    plt.close()

    # Also try different perplexities to find best separation
    print("\n--- Perplexity sweep (Llama only) ---")
    for perp in [10, 15, 30]:
        fig_test, ax_test = plt.subplots(1, 1, figsize=(6, 5))
        make_tsne_plot(llama_classified, f"Llama (perp={perp})", ax_test, perplexity=perp)
        ax_test.legend(fontsize=8)
        fig_test.savefig(OUTPUT_DIR / f"commitment_tsne_llama_perp{perp}.png",
                         bbox_inches="tight", dpi=150)
        plt.close()
        print(f"  Saved perplexity={perp} variant")

    print("\nDone.")


if __name__ == "__main__":
    main()
