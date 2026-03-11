"""
Paper 3: Definitive 100-question analysis.

Combines pilot_hidden_states_70b/ (20 hard), results_easier/ (20 easy),
and experiment_60q_results/ (30 hard + 30 easy) into a unified dataset.

Analyses:
  1. Step × Layer correlation heatmap (n=100)
  2. Permutation test at key step-4 cells
  3. Hard vs Easy split at step 4
  4. Consistency categories
  5. Step progression at layer 40
  6. Nearest-centroid classifier at step 4

Run from hotpotqa/ directory:
    ../.venv/bin/python3 paper3_100q_analysis.py
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PILOT_DIR = Path("pilot_hidden_states_70b")
EASIER_DIR = Path("results_easier")
NEW60_DIR = Path("experiment_60q_results")
PILOT_Q_FILE = Path("pilot_questions.json")
EASIER_Q_FILE = Path("easier_questions_selection.json")
NEW60_Q_FILE = Path("new_60_questions.json")
OUTPUT_DIR = Path("analysis_results/combined_100q")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
    elif text.startswith("yes"):
        return "yes"
    return None


def load_pilot_data():
    print("Loading 20 hard questions (pilot, npy format)...")
    with open(PILOT_Q_FILE) as f:
        questions = {q["id"]: q for q in json.load(f)}
    results = {}
    for qdir in sorted(PILOT_DIR.iterdir()):
        if not qdir.is_dir():
            continue
        qid = qdir.name
        if qid not in questions:
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
            results[qid] = {"question": questions[qid], "runs": runs_data, "difficulty": "hard"}
            print(f"  {qid}: {len(runs_data)} runs")
    return results


def load_easier_data():
    print("\nLoading 20 easy questions (easier, json format)...")
    with open(EASIER_Q_FILE) as f:
        questions = {q["id"]: q for q in json.load(f)}
    results = {}
    for fp in sorted(EASIER_DIR.glob("*.json")):
        qid = fp.stem
        if qid not in questions:
            continue
        with open(fp) as f:
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
            results[qid] = {"question": questions[qid], "runs": runs_data, "difficulty": "easy"}
            print(f"  {qid}: {len(runs_data)} runs")
    return results


def load_new60_data():
    print("\nLoading 60 new questions (experiment_60q, npy format)...")
    with open(NEW60_Q_FILE) as f:
        questions = {q["id"]: q for q in json.load(f)}
    results = {}
    for qdir in sorted(NEW60_DIR.iterdir()):
        if not qdir.is_dir():
            continue
        qid = qdir.name
        if qid not in questions:
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
            diff = questions[qid].get("difficulty", "unknown")
            results[qid] = {"question": questions[qid], "runs": runs_data, "difficulty": diff}
            print(f"  {qid}: {len(runs_data)} runs ({diff})")
    return results


def compute_question_metrics(qid, entry):
    runs = entry["runs"]
    question = entry["question"]
    expected = question.get("answer", "").lower().strip()

    step_counts = [r["step_count"] for r in runs]
    cv = np.std(step_counts) / np.mean(step_counts) if np.mean(step_counts) > 0 else 0

    correct_count = sum(1 for r in runs if r.get("correct"))
    correct_rate = correct_count / len(runs) if runs else 0

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
        "difficulty": entry["difficulty"],
        "cv": cv,
        "correct_rate": correct_rate,
        "correct_count": correct_count,
        "n_runs": len(runs),
        "mean_steps": float(np.mean(step_counts)),
        "similarity_by_step_layer": similarity_by_step_layer,
    }


def permutation_test(sims, cvs, n_perms, rng):
    observed_r, parametric_p = stats.pearsonr(sims, cvs)
    cvs_arr = np.array(cvs)
    sims_arr = np.array(sims)
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


# ══════════════════════════════════════════════════════════
#  ANALYSIS 1: Step × Layer Heatmap (n=100)
# ══════════════════════════════════════════════════════════

def analysis1_heatmap(metrics):
    print("\n" + "=" * 70)
    print("ANALYSIS 1: STEP × LAYER CORRELATION HEATMAP (n=100)")
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

    print(f"\n  {'':>8}", end="")
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

    fig, ax = plt.subplots(figsize=(10, 9))
    vmax = max(0.5, np.nanmax(np.abs(r_matrix)))
    im = ax.imshow(r_matrix, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_xticks(range(n_steps))
    ax.set_xticklabels([f"Step {s}" for s in STEPS], fontsize=11)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"Layer {l}" for l in LAYERS], fontsize=11)
    ax.set_xlabel("ReAct Step", fontsize=13)
    ax.set_ylabel("Transformer Layer", fontsize=13)
    ax.set_title("Correlation: Activation Similarity vs Behavioral CV\n(n=100 questions, Pearson r)", fontsize=14)
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
    for li in range(n_layers):
        for si in range(n_steps):
            p = p_matrix[li, si]
            if not np.isnan(p) and p < 0.05:
                rect = plt.Rectangle((si - 0.5, li - 0.5), 1, 1, fill=False, edgecolor="gold", linewidth=2.5)
                ax.add_patch(rect)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Pearson r", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "step_layer_heatmap_100q.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: step_layer_heatmap_100q.png")

    n_sig = np.sum((~np.isnan(p_matrix)) & (p_matrix < 0.05))
    n_total = np.sum(~np.isnan(p_matrix))
    print(f"  Significant cells: {n_sig}/{n_total}")

    return r_matrix, p_matrix, n_matrix


# ══════════════════════════════════════════════════════════
#  ANALYSIS 2: Permutation Test at Key Step-4 Cells
# ══════════════════════════════════════════════════════════

def analysis2_permutation(metrics):
    print("\n" + "=" * 70)
    print("ANALYSIS 2: PERMUTATION TEST AT STEP 4 (10,000 perms)")
    print("=" * 70)

    rng = np.random.default_rng(42)
    target_layers = [32, 40, 48, 56, 64, 72, 80]
    step = 4
    N_PERMS = 10000

    results = {}
    print(f"\n  {'Layer':>5} {'n':>4} {'Obs r':>8} {'Param p':>10} {'Perm p':>10} {'Sig':>5}")
    print("  " + "-" * 50)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes_flat = axes.flatten()

    for idx, layer in enumerate(target_layers):
        cvs, sims = [], []
        for m in metrics:
            if step in m["similarity_by_step_layer"] and layer in m["similarity_by_step_layer"][step]:
                sim = m["similarity_by_step_layer"][step][layer]
                if not np.isnan(sim):
                    cvs.append(m["cv"])
                    sims.append(sim)
        if len(cvs) < 5:
            continue

        observed_r, parametric_p, perm_p, null_rs = permutation_test(sims, cvs, N_PERMS, rng)
        sig = "*" if perm_p < 0.05 else ""
        print(f"  {layer:>5} {len(cvs):>4} {observed_r:>8.4f} {parametric_p:>10.6f} {perm_p:>10.6f} {sig:>5}")

        results[layer] = {
            "n": len(cvs),
            "observed_r": float(observed_r),
            "parametric_p": float(parametric_p),
            "permutation_p": float(perm_p),
            "significant": bool(perm_p < 0.05),
        }

        ax = axes_flat[idx]
        ax.hist(null_rs, bins=50, color="#cccccc", edgecolor="gray", alpha=0.8)
        ax.axvline(x=observed_r, color="red", linewidth=2, label=f"r={observed_r:.3f}")
        ax.axvline(x=-observed_r, color="red", linewidth=2, linestyle="--", alpha=0.5)
        ax.set_title(f"Step 4 × Layer {layer}\nperm p={perm_p:.4f}", fontsize=10)
        ax.set_xlabel("r (null)", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")

    if len(target_layers) < 8:
        axes_flat[7].axis("off")

    plt.suptitle("Permutation Test Null Distributions — Step 4 Key Layers (n=100)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "permutation_test_step4_100q.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: permutation_test_step4_100q.png")

    n_sig = sum(1 for v in results.values() if v["significant"])
    print(f"  Significant: {n_sig}/{len(results)} layers at step 4")

    with open(OUTPUT_DIR / "permutation_test_100q.json", "w") as f:
        json.dump({"step": step, "n_permutations": N_PERMS, "layers": {str(k): v for k, v in results.items()}}, f, indent=2)

    return results


# ══════════════════════════════════════════════════════════
#  ANALYSIS 3: Hard vs Easy at Step 4
# ══════════════════════════════════════════════════════════

def analysis3_hard_vs_easy(metrics):
    print("\n" + "=" * 70)
    print("ANALYSIS 3: HARD vs EASY AT STEP 4 (n≈50 each)")
    print("=" * 70)

    hard = [m for m in metrics if m["difficulty"] == "hard"]
    easy = [m for m in metrics if m["difficulty"] == "easy"]
    print(f"  Hard: {len(hard)}, Easy: {len(easy)}")

    step = 4
    results = {"hard": {}, "easy": {}}

    print(f"\n  {'Layer':>5}  {'Hard r':>8} {'Hard p':>8} {'n_h':>4}  {'Easy r':>8} {'Easy p':>8} {'n_e':>4}")
    print("  " + "-" * 62)

    for layer in LAYERS:
        for label, subset in [("hard", hard), ("easy", easy)]:
            cvs, sims = [], []
            for m in subset:
                if step in m["similarity_by_step_layer"] and layer in m["similarity_by_step_layer"][step]:
                    sim = m["similarity_by_step_layer"][step][layer]
                    if not np.isnan(sim):
                        cvs.append(m["cv"])
                        sims.append(sim)
            if len(cvs) >= 5:
                r, p = stats.pearsonr(sims, cvs)
                results[label][layer] = {"r": float(r), "p": float(p), "n": len(cvs)}

        h = results["hard"].get(layer, {})
        e = results["easy"].get(layer, {})
        hr = f"{h['r']:.3f}" if h else "  n/a"
        hp = f"{h['p']:.3f}" if h else "  n/a"
        hn = f"{h.get('n', 0):>4}" if h else " n/a"
        er = f"{e['r']:.3f}" if e else "  n/a"
        ep = f"{e['p']:.3f}" if e else "  n/a"
        en = f"{e.get('n', 0):>4}" if e else " n/a"
        h_sig = "*" if h and h.get("p", 1) < 0.05 else " "
        e_sig = "*" if e and e.get("p", 1) < 0.05 else " "
        print(f"  {layer:>5}  {hr}{h_sig} {hp} {hn}   {er}{e_sig} {ep} {en}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, label, data, color, title in [
        (ax1, "hard", results["hard"], "#d62728", f"Hard Questions (n={len(hard)})"),
        (ax2, "easy", results["easy"], "#1f77b4", f"Easy Questions (n={len(easy)})"),
    ]:
        layers_present = sorted(data.keys())
        rs = [data[l]["r"] for l in layers_present]
        ps = [data[l]["p"] for l in layers_present]
        colors = [color if p < 0.05 else "#cccccc" for p in ps]
        ax.bar(range(len(layers_present)), rs, color=colors, edgecolor="black", linewidth=0.5)
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
    fig.suptitle("Step 4 Layer-wise Correlation: Hard vs Easy (n=100)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "hard_vs_easy_step4_100q.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: hard_vs_easy_step4_100q.png")

    with open(OUTPUT_DIR / "hard_vs_easy_step4_100q.json", "w") as f:
        json.dump({"step": step, "hard": {str(k): v for k, v in results["hard"].items()},
                    "easy": {str(k): v for k, v in results["easy"].items()}}, f, indent=2)

    return results


# ══════════════════════════════════════════════════════════
#  ANALYSIS 4: Consistency Categories
# ══════════════════════════════════════════════════════════

def analysis4_consistency_categories(metrics):
    print("\n" + "=" * 70)
    print("ANALYSIS 4: CONSISTENCY CATEGORIES (n=100)")
    print("=" * 70)

    categories = []
    for m in metrics:
        cr = m["correct_rate"]
        n = m["n_runs"]
        answers = []
        if cr >= 0.8:
            cat = "consistent-correct"
        elif cr <= 0.2:
            cat = "consistent-wrong"
        elif 0.2 < cr < 0.8:
            cat = "inconsistent"
        else:
            cat = "mixed"
        categories.append(cat)
        m["category"] = cat

    counts = Counter(categories)
    print(f"\n  consistent-correct: {counts.get('consistent-correct', 0)}")
    print(f"  consistent-wrong:   {counts.get('consistent-wrong', 0)}")
    print(f"  inconsistent:       {counts.get('inconsistent', 0)}")
    print(f"  mixed:              {counts.get('mixed', 0)}")
    print(f"  Total:              {sum(counts.values())}")

    by_difficulty = {}
    for m in metrics:
        d = m["difficulty"]
        c = m["category"]
        if d not in by_difficulty:
            by_difficulty[d] = Counter()
        by_difficulty[d][c] += 1

    print(f"\n  By difficulty:")
    for d in ["easy", "hard"]:
        if d in by_difficulty:
            print(f"    {d}: {dict(by_difficulty[d])}")

    result = {
        "total": dict(counts),
        "by_difficulty": {d: dict(c) for d, c in by_difficulty.items()},
        "questions": [
            {"qid": m["qid"], "difficulty": m["difficulty"], "category": m["category"],
             "correct_rate": m["correct_rate"], "cv": m["cv"]}
            for m in metrics
        ],
    }
    with open(OUTPUT_DIR / "consistency_categories_100q.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: consistency_categories_100q.json")

    return counts


# ══════════════════════════════════════════════════════════
#  ANALYSIS 5: Step Progression at Layer 40
# ══════════════════════════════════════════════════════════

def analysis5_step_progression(metrics):
    print("\n" + "=" * 70)
    print("ANALYSIS 5: STEP PROGRESSION AT LAYER 40 (n=100)")
    print("=" * 70)

    layer = 40
    rs_by_step = []
    ps_by_step = []
    ns_by_step = []

    print(f"\n  {'Step':>4} {'n':>4} {'r':>8} {'p':>10}")
    print("  " + "-" * 30)

    for step in STEPS:
        cvs, sims = [], []
        for m in metrics:
            if step in m["similarity_by_step_layer"] and layer in m["similarity_by_step_layer"][step]:
                sim = m["similarity_by_step_layer"][step][layer]
                if not np.isnan(sim):
                    cvs.append(m["cv"])
                    sims.append(sim)
        if len(cvs) >= 5:
            r, p = stats.pearsonr(sims, cvs)
            rs_by_step.append(r)
            ps_by_step.append(p)
            ns_by_step.append(len(cvs))
            sig = "*" if p < 0.05 else ""
            print(f"  {step:>4} {len(cvs):>4} {r:>8.4f} {p:>10.6f} {sig}")
        else:
            rs_by_step.append(np.nan)
            ps_by_step.append(np.nan)
            ns_by_step.append(0)

    fig, ax = plt.subplots(figsize=(8, 5))
    valid_steps = [s for s, r in zip(STEPS, rs_by_step) if not np.isnan(r)]
    valid_rs = [r for r in rs_by_step if not np.isnan(r)]
    valid_ps = [p for p in ps_by_step if not np.isnan(p)]
    colors = ["#d62728" if p < 0.05 else "#1f77b4" for p in valid_ps]
    bars = ax.bar(valid_steps, valid_rs, color=colors, edgecolor="black", linewidth=0.5, width=0.6)
    for s, r, p in zip(valid_steps, valid_rs, valid_ps):
        sig = "*" if p < 0.05 else ""
        y_off = 0.01 if r >= 0 else -0.03
        ax.text(s, r + y_off, f"{r:.3f}{sig}", ha="center", fontsize=10)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("ReAct Step", fontsize=13)
    ax.set_ylabel("Pearson r (Similarity vs CV)", fontsize=13)
    ax.set_title(f"Step Progression at Layer 40 (n=100)\nRed = p < 0.05", fontsize=14)
    ax.set_xticks(STEPS)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "step_progression_layer40_100q.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: step_progression_layer40_100q.png")

    result = {
        "layer": layer,
        "steps": {str(s): {"r": float(r), "p": float(p), "n": n}
                  for s, r, p, n in zip(STEPS, rs_by_step, ps_by_step, ns_by_step) if not np.isnan(r)},
    }
    with open(OUTPUT_DIR / "step_progression_100q.json", "w") as f:
        json.dump(result, f, indent=2)

    return rs_by_step, ps_by_step


# ══════════════════════════════════════════════════════════
#  ANALYSIS 6: Nearest-Centroid Classifier at Step 4
# ══════════════════════════════════════════════════════════

def analysis6_classifier(all_data, metrics):
    print("\n" + "=" * 70)
    print("ANALYSIS 6: NEAREST-CENTROID CLASSIFIER (Step 4)")
    print("=" * 70)

    step = 4
    target_layers = [40, 48, 56, 64, 72, 80]
    results = {}

    for layer in target_layers:
        print(f"\n  --- Layer {layer}, Step {step} ---")

        q_features = []
        q_labels = []
        q_ids = []

        for m in metrics:
            cat = m.get("category", "")
            if cat not in ("consistent-correct", "consistent-wrong", "inconsistent"):
                continue

            label = 1 if cat == "consistent-correct" else 0

            qid = m["qid"]
            if qid not in all_data:
                continue
            runs = all_data[qid]["runs"]

            vectors = []
            for run in runs:
                hs = run["hidden_states"].get(step)
                if hs is not None and len(hs) > layer:
                    vectors.append(hs[layer])

            if len(vectors) < 2:
                continue

            mean_sim = compute_pairwise_cosine_similarity(vectors)
            if np.isnan(mean_sim):
                continue

            q_features.append(mean_sim)
            q_labels.append(label)
            q_ids.append(qid)

        q_features = np.array(q_features).reshape(-1, 1)
        q_labels = np.array(q_labels)

        print(f"  Questions: {len(q_labels)} (consistent={q_labels.sum()}, other={len(q_labels)-q_labels.sum()})")

        if len(set(q_labels)) < 2 or len(q_labels) < 10:
            print("  Skipped: insufficient class diversity")
            results[layer] = {"auc": None, "n": len(q_labels)}
            continue

        n_splits = min(5, min(Counter(q_labels).values()))
        if n_splits < 2:
            print("  Skipped: too few samples for CV")
            results[layer] = {"auc": None, "n": len(q_labels)}
            continue

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_aucs = []
        all_scores = []
        all_true = []

        for train_idx, test_idx in skf.split(q_features, q_labels):
            X_train, X_test = q_features[train_idx], q_features[test_idx]
            y_train, y_test = q_labels[train_idx], q_labels[test_idx]

            if len(set(y_test)) < 2:
                continue

            centroid_pos = X_train[y_train == 1].mean()
            centroid_neg = X_train[y_train == 0].mean()

            scores = np.abs(X_test.ravel() - centroid_pos) - np.abs(X_test.ravel() - centroid_neg)
            scores = -scores

            fold_auc = roc_auc_score(y_test, scores)
            fold_aucs.append(fold_auc)
            all_scores.extend(scores)
            all_true.extend(y_test)

        if fold_aucs:
            mean_auc = np.mean(fold_aucs)
            std_auc = np.std(fold_aucs)
            overall_auc = roc_auc_score(all_true, all_scores)
            print(f"  AUC: {mean_auc:.4f} ± {std_auc:.4f} ({n_splits}-fold CV)")
            print(f"  Overall AUC: {overall_auc:.4f}")
            results[layer] = {
                "mean_auc": float(mean_auc),
                "std_auc": float(std_auc),
                "overall_auc": float(overall_auc),
                "n_folds": n_splits,
                "fold_aucs": [float(a) for a in fold_aucs],
                "n": len(q_labels),
                "n_consistent": int(q_labels.sum()),
                "n_other": int(len(q_labels) - q_labels.sum()),
            }
        else:
            print("  No valid folds")
            results[layer] = {"auc": None, "n": len(q_labels)}

    with open(OUTPUT_DIR / "classifier_step4_100q.json", "w") as f:
        json.dump({"step": step, "layers": {str(k): v for k, v in results.items()}}, f, indent=2)
    print(f"\n  Saved: classifier_step4_100q.json")

    return results


# ══════════════════════════════════════════════════════════
#  ANALYSIS 7: Leave-One-Out Sensitivity (Easy, Step 4, Layer 40)
# ══════════════════════════════════════════════════════════

def analysis7_loo_sensitivity(metrics):
    print("\n" + "=" * 70)
    print("ANALYSIS 7: LEAVE-ONE-OUT SENSITIVITY (Easy Questions, Step 4, Layer 40)")
    print("=" * 70)

    step, layer = 4, 40
    easy = [m for m in metrics if m["difficulty"] == "easy"]

    paired = []
    for m in easy:
        if step in m["similarity_by_step_layer"] and layer in m["similarity_by_step_layer"][step]:
            sim = m["similarity_by_step_layer"][step][layer]
            if not np.isnan(sim):
                paired.append({"qid": m["qid"], "cv": m["cv"], "sim": sim,
                               "correct_rate": m["correct_rate"], "category": m.get("category", "")})

    n = len(paired)
    sims = np.array([p["sim"] for p in paired])
    cvs = np.array([p["cv"] for p in paired])
    base_r, base_p = stats.pearsonr(sims, cvs)
    print(f"\n  Base correlation (n={n}): r={base_r:.4f}, p={base_p:.6f}")

    loo_rs = []
    loo_ps = []
    loo_qids = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        r_i, p_i = stats.pearsonr(sims[mask], cvs[mask])
        loo_rs.append(r_i)
        loo_ps.append(p_i)
        loo_qids.append(paired[i]["qid"])

    loo_rs = np.array(loo_rs)
    loo_ps = np.array(loo_ps)

    print(f"  LOO r range: [{loo_rs.min():.4f}, {loo_rs.max():.4f}]")
    print(f"  LOO p range: [{loo_ps.min():.6f}, {loo_ps.max():.6f}]")
    all_sig = np.all(loo_ps < 0.05)
    print(f"  All LOO p < 0.05: {all_sig}")
    n_insig = np.sum(loo_ps >= 0.05)
    print(f"  LOO iterations losing significance: {n_insig}/{n}")

    delta_r = loo_rs - base_r
    most_influential_idx = np.argmax(np.abs(delta_r))
    print(f"\n  Most influential question: {paired[most_influential_idx]['qid']}")
    print(f"    Removing it: r={loo_rs[most_influential_idx]:.4f} (delta={delta_r[most_influential_idx]:+.4f})")
    print(f"    Its CV={paired[most_influential_idx]['cv']:.4f}, sim={paired[most_influential_idx]['sim']:.4f}, "
          f"correct_rate={paired[most_influential_idx]['correct_rate']:.1f}")

    top5_idx = np.argsort(np.abs(delta_r))[-5:][::-1]
    print(f"\n  Top 5 most influential (by |delta r|):")
    print(f"  {'QID':>28} {'CV':>6} {'Sim':>6} {'CR':>5} {'Cat':>20} {'LOO r':>8} {'delta':>8}")
    print("  " + "-" * 90)
    for idx in top5_idx:
        p = paired[idx]
        print(f"  {p['qid']:>28} {p['cv']:>6.3f} {p['sim']:>6.4f} {p['correct_rate']:>5.1f} "
              f"{p['category']:>20} {loo_rs[idx]:>8.4f} {delta_r[idx]:>+8.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(loo_rs, bins=25, color="#1f77b4", edgecolor="black", alpha=0.8)
    ax.axvline(x=base_r, color="red", linewidth=2, label=f"Base r={base_r:.3f}")
    ax.axvline(x=0, color="black", linewidth=0.5, linestyle="--")
    sig_threshold_r = loo_rs[np.argmin(np.abs(loo_ps - 0.05))]
    ax.set_xlabel("Pearson r (with one question removed)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"LOO Sensitivity: Easy Questions at Step 4, Layer 40\n"
                 f"n={n}, base r={base_r:.3f}, range [{loo_rs.min():.3f}, {loo_rs.max():.3f}]", fontsize=12)
    ax.legend(fontsize=10)

    ax = axes[1]
    sorted_idx = np.argsort(delta_r)
    colors = ["#d62728" if loo_ps[i] >= 0.05 else "#1f77b4" for i in sorted_idx]
    ax.barh(range(n), delta_r[sorted_idx], color=colors, edgecolor="none", height=1.0)
    ax.set_xlabel("Change in r when removed", fontsize=11)
    ax.set_ylabel("Question (sorted)", fontsize=11)
    ax.set_title("Per-question influence on r\n(red = removal breaks significance)", fontsize=12)
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "loo_sensitivity_easy_step4_l40.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: loo_sensitivity_easy_step4_l40.png")

    result = {
        "step": step, "layer": layer, "n_easy": n,
        "base_r": float(base_r), "base_p": float(base_p),
        "loo_r_min": float(loo_rs.min()), "loo_r_max": float(loo_rs.max()),
        "loo_r_mean": float(loo_rs.mean()), "loo_r_std": float(loo_rs.std()),
        "all_loo_significant": bool(all_sig), "n_loo_insignificant": int(n_insig),
        "robust": bool(all_sig),
        "most_influential": {
            "qid": paired[most_influential_idx]["qid"],
            "cv": float(paired[most_influential_idx]["cv"]),
            "sim": float(paired[most_influential_idx]["sim"]),
            "correct_rate": float(paired[most_influential_idx]["correct_rate"]),
            "loo_r": float(loo_rs[most_influential_idx]),
            "delta_r": float(delta_r[most_influential_idx]),
        },
        "top5_influential": [
            {"qid": paired[i]["qid"], "cv": float(paired[i]["cv"]),
             "sim": float(paired[i]["sim"]), "correct_rate": float(paired[i]["correct_rate"]),
             "category": paired[i]["category"],
             "loo_r": float(loo_rs[i]), "delta_r": float(delta_r[i])}
            for i in top5_idx
        ],
        "per_question": [
            {"qid": paired[i]["qid"], "loo_r": float(loo_rs[i]), "loo_p": float(loo_ps[i]),
             "delta_r": float(delta_r[i])}
            for i in range(n)
        ],
    }
    with open(OUTPUT_DIR / "loo_sensitivity_easy_step4_l40.json", "w") as f:
        json.dump(result, f, indent=2, default=lambda o: bool(o) if isinstance(o, np.bool_) else o)
    print(f"  Saved: loo_sensitivity_easy_step4_l40.json")

    return result


# ══════════════════════════════════════════════════════════
#  ANALYSIS 8: Inconsistent Questions Focused Analysis
# ══════════════════════════════════════════════════════════

def analysis8_inconsistent_focus(all_data, metrics):
    print("\n" + "=" * 70)
    print("ANALYSIS 8: INCONSISTENT QUESTIONS FOCUSED ANALYSIS (n=12)")
    print("=" * 70)

    step = 4
    cats = {"consistent-correct": [], "consistent-wrong": [], "inconsistent": []}
    for m in metrics:
        cat = m.get("category", "")
        if cat in cats:
            cats[cat].append(m)

    for cat, qs in cats.items():
        print(f"  {cat}: {len(qs)} questions")

    target_layers = [32, 40, 48, 56, 64, 72, 80]

    print(f"\n  --- Mean activation similarity at step {step} by category ---")
    print(f"  {'Layer':>5}  {'Cons-Correct':>14} {'Cons-Wrong':>14} {'Inconsistent':>14}  "
          f"{'CC vs CW':>10} {'CC vs Inc':>10} {'CW vs Inc':>10}")
    print("  " + "-" * 95)

    cat_sims = {cat: {l: [] for l in target_layers} for cat in cats}
    for cat, qs in cats.items():
        for m in qs:
            qid = m["qid"]
            if qid not in all_data:
                continue
            runs = all_data[qid]["runs"]
            for layer in target_layers:
                vectors = []
                for run in runs:
                    hs = run["hidden_states"].get(step)
                    if hs is not None and len(hs) > layer:
                        vectors.append(hs[layer])
                if len(vectors) >= 2:
                    cat_sims[cat][layer].append(compute_pairwise_cosine_similarity(vectors))

    layer_stats = {}
    for layer in target_layers:
        cc_vals = np.array(cat_sims["consistent-correct"][layer])
        cw_vals = np.array(cat_sims["consistent-wrong"][layer])
        inc_vals = np.array(cat_sims["inconsistent"][layer])

        cc_mean = np.mean(cc_vals) if len(cc_vals) > 0 else np.nan
        cw_mean = np.mean(cw_vals) if len(cw_vals) > 0 else np.nan
        inc_mean = np.mean(inc_vals) if len(inc_vals) > 0 else np.nan
        cc_std = np.std(cc_vals) if len(cc_vals) > 0 else np.nan
        cw_std = np.std(cw_vals) if len(cw_vals) > 0 else np.nan
        inc_std = np.std(inc_vals) if len(inc_vals) > 0 else np.nan

        tests = {}
        if len(cc_vals) >= 2 and len(cw_vals) >= 2:
            t, p = stats.mannwhitneyu(cc_vals, cw_vals, alternative="two-sided")
            tests["cc_vs_cw"] = {"U": float(t), "p": float(p)}
        if len(cc_vals) >= 2 and len(inc_vals) >= 2:
            t, p = stats.mannwhitneyu(cc_vals, inc_vals, alternative="two-sided")
            tests["cc_vs_inc"] = {"U": float(t), "p": float(p)}
        if len(cw_vals) >= 2 and len(inc_vals) >= 2:
            t, p = stats.mannwhitneyu(cw_vals, inc_vals, alternative="two-sided")
            tests["cw_vs_inc"] = {"U": float(t), "p": float(p)}

        cc_str = f"{cc_mean:.4f}±{cc_std:.4f}" if not np.isnan(cc_mean) else "n/a"
        cw_str = f"{cw_mean:.4f}±{cw_std:.4f}" if not np.isnan(cw_mean) else "n/a"
        inc_str = f"{inc_mean:.4f}±{inc_std:.4f}" if not np.isnan(inc_mean) else "n/a"
        cc_cw_p = f"{tests['cc_vs_cw']['p']:.4f}" if "cc_vs_cw" in tests else "n/a"
        cc_inc_p = f"{tests['cc_vs_inc']['p']:.4f}" if "cc_vs_inc" in tests else "n/a"
        cw_inc_p = f"{tests['cw_vs_inc']['p']:.4f}" if "cw_vs_inc" in tests else "n/a"

        print(f"  {layer:>5}  {cc_str:>14} {cw_str:>14} {inc_str:>14}  "
              f"{cc_cw_p:>10} {cc_inc_p:>10} {cw_inc_p:>10}")

        layer_stats[layer] = {
            "consistent_correct": {"mean": float(cc_mean), "std": float(cc_std), "n": len(cc_vals)},
            "consistent_wrong": {"mean": float(cw_mean), "std": float(cw_std), "n": len(cw_vals)},
            "inconsistent": {"mean": float(inc_mean), "std": float(inc_std), "n": len(inc_vals)},
            "tests": tests,
        }

    print(f"\n  --- CV distribution by category ---")
    for cat_name, qs in cats.items():
        cvs = [m["cv"] for m in qs]
        if cvs:
            print(f"  {cat_name:>20}: mean CV={np.mean(cvs):.4f}, std={np.std(cvs):.4f}, "
                  f"range=[{min(cvs):.4f}, {max(cvs):.4f}]")

    print(f"\n  --- Inconsistent question details ---")
    print(f"  {'QID':>28} {'Diff':>5} {'CR':>5} {'CV':>7} {'Sim@L40':>8} {'Steps':>6}")
    print("  " + "-" * 65)
    inc_details = []
    for m in cats["inconsistent"]:
        sim_l40 = m["similarity_by_step_layer"].get(step, {}).get(40, np.nan)
        print(f"  {m['qid']:>28} {m['difficulty']:>5} {m['correct_rate']:>5.2f} {m['cv']:>7.4f} "
              f"{sim_l40:>8.4f} {m['mean_steps']:>6.1f}")
        inc_details.append({
            "qid": m["qid"], "difficulty": m["difficulty"],
            "correct_rate": float(m["correct_rate"]), "cv": float(m["cv"]),
            "sim_step4_layer40": float(sim_l40) if not np.isnan(sim_l40) else None,
            "mean_steps": float(m["mean_steps"]),
        })

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    layer_plot = 40
    for cat_name, color, marker in [
        ("consistent-correct", "#2ca02c", "o"),
        ("consistent-wrong", "#d62728", "s"),
        ("inconsistent", "#ff7f0e", "D"),
    ]:
        qs = cats[cat_name]
        cvs = [m["cv"] for m in qs]
        sims_plot = [m["similarity_by_step_layer"].get(step, {}).get(layer_plot, np.nan) for m in qs]
        valid = [(c, s) for c, s in zip(cvs, sims_plot) if not np.isnan(s)]
        if valid:
            ax.scatter([v[1] for v in valid], [v[0] for v in valid],
                       c=color, marker=marker, s=50, alpha=0.7, label=f"{cat_name} (n={len(valid)})",
                       edgecolors="black", linewidth=0.5)
    ax.set_xlabel("Activation Similarity (Step 4, Layer 40)", fontsize=11)
    ax.set_ylabel("Behavioral CV", fontsize=11)
    ax.set_title("Similarity vs CV by Consistency Category", fontsize=12)
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    positions = []
    data_for_box = []
    labels_for_box = []
    cat_colors = {"consistent-correct": "#2ca02c", "consistent-wrong": "#d62728", "inconsistent": "#ff7f0e"}
    for i, (cat_name, short) in enumerate([("consistent-correct", "CC"), ("consistent-wrong", "CW"), ("inconsistent", "Inc")]):
        vals = cat_sims[cat_name].get(40, [])
        if vals:
            data_for_box.append(vals)
            labels_for_box.append(f"{short}\n(n={len(vals)})")
            positions.append(i)
    bp = ax.boxplot(data_for_box, positions=positions, patch_artist=True, widths=0.6)
    cat_keys = ["consistent-correct", "consistent-wrong", "inconsistent"]
    for patch, cat_name in zip(bp["boxes"], cat_keys[:len(data_for_box)]):
        patch.set_facecolor(cat_colors[cat_name])
        patch.set_alpha(0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels_for_box, fontsize=10)
    ax.set_ylabel("Activation Similarity (Step 4, Layer 40)", fontsize=11)
    ax.set_title("Similarity Distribution by Category", fontsize=12)

    ax = axes[2]
    for cat_name, color, marker in [
        ("consistent-correct", "#2ca02c", "o"),
        ("consistent-wrong", "#d62728", "s"),
        ("inconsistent", "#ff7f0e", "D"),
    ]:
        qs = cats[cat_name]
        crs = [m["correct_rate"] for m in qs]
        cvs = [m["cv"] for m in qs]
        ax.scatter(crs, cvs, c=color, marker=marker, s=50, alpha=0.7,
                   label=f"{cat_name} (n={len(qs)})", edgecolors="black", linewidth=0.5)
    ax.set_xlabel("Correct Rate (across 10 runs)", fontsize=11)
    ax.set_ylabel("Behavioral CV (step count variability)", fontsize=11)
    ax.set_title("Correct Rate vs CV by Category", fontsize=12)
    ax.legend(fontsize=8)

    plt.suptitle("Inconsistent Questions: Representational Signatures (Step 4)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "inconsistent_focus_step4.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: inconsistent_focus_step4.png")

    result = {
        "step": step,
        "n_by_category": {k: len(v) for k, v in cats.items()},
        "layer_stats": {str(k): v for k, v in layer_stats.items()},
        "inconsistent_details": inc_details,
        "cv_by_category": {
            cat_name: {"mean": float(np.mean([m["cv"] for m in qs])),
                       "std": float(np.std([m["cv"] for m in qs])),
                       "n": len(qs)}
            for cat_name, qs in cats.items() if qs
        },
    }
    with open(OUTPUT_DIR / "inconsistent_focus_step4.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: inconsistent_focus_step4.json")

    return result


# ══════════════════════════════════════════════════════════
#  ANALYSIS 9: Partial Correlation Controlling for Difficulty
# ══════════════════════════════════════════════════════════

def analysis9_partial_correlation(metrics):
    print("\n" + "=" * 70)
    print("ANALYSIS 9: PARTIAL CORRELATION (controlling for accuracy)")
    print("=" * 70)

    step, layer = 4, 40
    target_layers = [32, 40, 48, 56, 64, 72, 80]

    triples = []
    for m in metrics:
        if step in m["similarity_by_step_layer"] and layer in m["similarity_by_step_layer"][step]:
            sim = m["similarity_by_step_layer"][step][layer]
            if not np.isnan(sim):
                triples.append({"qid": m["qid"], "cv": m["cv"], "sim": sim,
                                "accuracy": m["correct_rate"], "difficulty": m["difficulty"]})

    n = len(triples)
    sims = np.array([t["sim"] for t in triples])
    cvs = np.array([t["cv"] for t in triples])
    accs = np.array([t["accuracy"] for t in triples])

    raw_r, raw_p = stats.pearsonr(sims, cvs)
    print(f"\n  Raw correlation (n={n}): r={raw_r:.4f}, p={raw_p:.6f}")

    acc_cv_r, acc_cv_p = stats.pearsonr(accs, cvs)
    acc_sim_r, acc_sim_p = stats.pearsonr(accs, sims)
    print(f"  Accuracy vs CV:         r={acc_cv_r:.4f}, p={acc_cv_p:.6f}")
    print(f"  Accuracy vs Similarity: r={acc_sim_r:.4f}, p={acc_sim_p:.6f}")

    def partial_corr(x, y, z):
        _, _, res_x = np.linalg.lstsq(np.column_stack([z, np.ones(len(z))]), x, rcond=None)[:3]
        _, _, res_y = np.linalg.lstsq(np.column_stack([z, np.ones(len(z))]), y, rcond=None)[:3]
        residuals_x = x - np.column_stack([z, np.ones(len(z))]) @ np.linalg.lstsq(np.column_stack([z, np.ones(len(z))]), x, rcond=None)[0]
        residuals_y = y - np.column_stack([z, np.ones(len(z))]) @ np.linalg.lstsq(np.column_stack([z, np.ones(len(z))]), y, rcond=None)[0]
        r, p = stats.pearsonr(residuals_x, residuals_y)
        return r, p, residuals_x, residuals_y

    partial_r, partial_p, res_sim, res_cv = partial_corr(sims, cvs, accs.reshape(-1, 1))
    print(f"\n  Partial corr (sim ~ CV | accuracy): r={partial_r:.4f}, p={partial_p:.6f}")
    print(f"  Survives confound control: {'YES' if partial_p < 0.05 else 'NO'}")

    diff_binary = np.array([1 if t["difficulty"] == "hard" else 0 for t in triples])
    controls = np.column_stack([accs, diff_binary])
    partial_r2, partial_p2, res_sim2, res_cv2 = partial_corr(sims, cvs, controls)
    print(f"\n  Partial corr (sim ~ CV | accuracy + difficulty_label): r={partial_r2:.4f}, p={partial_p2:.6f}")
    print(f"  Survives both controls: {'YES' if partial_p2 < 0.05 else 'NO'}")

    print(f"\n  --- Layer-by-layer partial correlations at step {step} ---")
    print(f"  {'Layer':>5} {'Raw r':>8} {'Raw p':>10} {'Part r':>8} {'Part p':>10} {'Survives':>8}")
    print("  " + "-" * 60)

    layer_results = {}
    for l in target_layers:
        l_triples = []
        for m in metrics:
            if step in m["similarity_by_step_layer"] and l in m["similarity_by_step_layer"][step]:
                sim_l = m["similarity_by_step_layer"][step][l]
                if not np.isnan(sim_l):
                    l_triples.append({"sim": sim_l, "cv": m["cv"], "accuracy": m["correct_rate"],
                                      "difficulty": m["difficulty"]})
        if len(l_triples) < 10:
            continue
        l_sims = np.array([t["sim"] for t in l_triples])
        l_cvs = np.array([t["cv"] for t in l_triples])
        l_accs = np.array([t["accuracy"] for t in l_triples])
        l_diff = np.array([1 if t["difficulty"] == "hard" else 0 for t in l_triples])

        l_raw_r, l_raw_p = stats.pearsonr(l_sims, l_cvs)
        l_controls = np.column_stack([l_accs, l_diff])
        l_pr, l_pp, _, _ = partial_corr(l_sims, l_cvs, l_controls)
        survives = "YES" if l_pp < 0.05 else "no"
        print(f"  {l:>5} {l_raw_r:>8.4f} {l_raw_p:>10.6f} {l_pr:>8.4f} {l_pp:>10.6f} {survives:>8}")

        layer_results[l] = {
            "n": len(l_triples),
            "raw_r": float(l_raw_r), "raw_p": float(l_raw_p),
            "partial_r": float(l_pr), "partial_p": float(l_pp),
            "survives": bool(l_pp < 0.05),
        }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    hard_mask = np.array([t["difficulty"] == "hard" for t in triples])
    ax.scatter(sims[hard_mask], cvs[hard_mask], c="#d62728", alpha=0.6, s=40, label="Hard", edgecolors="black", linewidth=0.3)
    ax.scatter(sims[~hard_mask], cvs[~hard_mask], c="#1f77b4", alpha=0.6, s=40, label="Easy", edgecolors="black", linewidth=0.3)
    ax.set_xlabel("Activation Similarity (Step 4, Layer 40)", fontsize=11)
    ax.set_ylabel("Behavioral CV", fontsize=11)
    ax.set_title(f"Raw: r={raw_r:.3f}, p={raw_p:.4f}", fontsize=12)
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.scatter(res_sim2[hard_mask], res_cv2[hard_mask], c="#d62728", alpha=0.6, s=40, label="Hard", edgecolors="black", linewidth=0.3)
    ax.scatter(res_sim2[~hard_mask], res_cv2[~hard_mask], c="#1f77b4", alpha=0.6, s=40, label="Easy", edgecolors="black", linewidth=0.3)
    ax.set_xlabel("Similarity (residualized)", fontsize=11)
    ax.set_ylabel("CV (residualized)", fontsize=11)
    ax.set_title(f"Partial (| accuracy + difficulty): r={partial_r2:.3f}, p={partial_p2:.4f}", fontsize=12)
    ax.legend(fontsize=9)

    ax = axes[2]
    layers_plot = sorted(layer_results.keys())
    raw_rs = [layer_results[l]["raw_r"] for l in layers_plot]
    part_rs = [layer_results[l]["partial_r"] for l in layers_plot]
    x_pos = np.arange(len(layers_plot))
    ax.bar(x_pos - 0.15, raw_rs, 0.3, color="#1f77b4", label="Raw r", edgecolor="black", linewidth=0.5)
    ax.bar(x_pos + 0.15, part_rs, 0.3, color="#ff7f0e", label="Partial r", edgecolor="black", linewidth=0.5)
    for i, l in enumerate(layers_plot):
        sig_raw = "*" if layer_results[l]["raw_p"] < 0.05 else ""
        sig_part = "*" if layer_results[l]["partial_p"] < 0.05 else ""
        ax.text(i - 0.15, raw_rs[i] - 0.02, f"{raw_rs[i]:.2f}{sig_raw}", ha="center", fontsize=7, va="top")
        ax.text(i + 0.15, part_rs[i] - 0.02, f"{part_rs[i]:.2f}{sig_part}", ha="center", fontsize=7, va="top")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(l) for l in layers_plot], fontsize=9)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Pearson r", fontsize=11)
    ax.set_title("Raw vs Partial Correlation at Step 4", fontsize=12)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.legend(fontsize=9)

    plt.suptitle("Partial Correlation: Activation Similarity vs CV, Controlling for Accuracy & Difficulty", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "partial_correlation_step4.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: partial_correlation_step4.png")

    result = {
        "step": step, "n": n,
        "raw_r_layer40": float(raw_r), "raw_p_layer40": float(raw_p),
        "partial_r_accuracy_only": float(partial_r), "partial_p_accuracy_only": float(partial_p),
        "partial_r_accuracy_and_difficulty": float(partial_r2), "partial_p_accuracy_and_difficulty": float(partial_p2),
        "confounders": {
            "accuracy_vs_cv": {"r": float(acc_cv_r), "p": float(acc_cv_p)},
            "accuracy_vs_similarity": {"r": float(acc_sim_r), "p": float(acc_sim_p)},
        },
        "layers": {str(k): v for k, v in layer_results.items()},
    }
    with open(OUTPUT_DIR / "partial_correlation_step4.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: partial_correlation_step4.json")

    return result


# ══════════════════════════════════════════════════════════
#  ANALYSIS 10: Baseline Predictors of Consistency
# ══════════════════════════════════════════════════════════

def analysis10_baseline_comparison(all_data, metrics):
    print("\n" + "=" * 70)
    print("ANALYSIS 10: BASELINE COMPARISON — Simpler Predictors of CV")
    print("=" * 70)

    step = 4
    layer = 40

    rows = []
    for m in metrics:
        qid = m["qid"]
        if qid not in all_data:
            continue
        entry = all_data[qid]
        question_text = entry["question"].get("question", "")
        runs = entry["runs"]

        q_word_count = len(question_text.split())
        q_char_count = len(question_text)
        n_context_docs = len(entry["question"].get("context", {}))

        run_features = []
        for run in runs:
            traj_steps = []
            if "hidden_states" in run:
                pass
            meta_actions = []
            thought_lengths = []
            obs_lengths = []
            search_count_first3 = 0

            for hs_key in sorted(run["hidden_states"].keys()):
                pass

        mean_thought_len_step3 = []
        mean_response_len_step3 = []
        search_in_first3 = []
        total_thought_len = []

        for run in runs:
            run_thoughts = []
            run_search_first3 = 0
            run_step3_thought = None

            qid_in_data = qid
            if entry["difficulty"] in ("easy", "hard"):
                pass

        has_trajectories = False
        for run in runs:
            if "hidden_states" in run:
                pass

        pilot_or_new60 = qid in {qdir.name for qdir in
                                  list(Path("pilot_hidden_states_70b").iterdir()) +
                                  list(Path("experiment_60q_results").iterdir())
                                  if qdir.is_dir()}
        easier_format = not pilot_or_new60

        run_thought_lens_at_step3 = []
        run_search_first3_counts = []
        run_total_thought_lens = []

        if easier_format:
            fp = Path("results_easier") / f"{qid}.json"
            if fp.exists():
                with open(fp) as f:
                    easier_data = json.load(f)
                for run_data in easier_data.get("runs", []):
                    steps = run_data.get("steps", [])
                    search_first3 = sum(1 for s in steps[:3] if s.get("action") == "Search")
                    run_search_first3_counts.append(search_first3)
                    total_t = sum(len(s.get("thought", "")) for s in steps)
                    run_total_thought_lens.append(total_t)
                    if len(steps) >= 3:
                        run_thought_lens_at_step3.append(len(steps[2].get("thought", "")))
        else:
            base_dir = Path("pilot_hidden_states_70b") / qid
            if not base_dir.exists():
                base_dir = Path("experiment_60q_results") / qid
            if base_dir.exists():
                for run_dir in sorted(base_dir.glob("run_*")):
                    traj_file = run_dir / "trajectory.json"
                    if not traj_file.exists():
                        continue
                    with open(traj_file) as f:
                        traj = json.load(f)
                    steps = traj.get("steps", [])
                    search_first3 = sum(1 for s in steps[:3] if s.get("action") == "Search")
                    run_search_first3_counts.append(search_first3)
                    total_t = sum(len(s.get("thought", "")) for s in steps)
                    run_total_thought_lens.append(total_t)
                    if len(steps) >= 3:
                        run_thought_lens_at_step3.append(len(steps[2].get("thought", "")))

        sim_at_l40 = m["similarity_by_step_layer"].get(step, {}).get(layer, np.nan)

        rows.append({
            "qid": qid,
            "cv": m["cv"],
            "correct_rate": m["correct_rate"],
            "difficulty": m["difficulty"],
            "sim_step4_layer40": sim_at_l40,
            "question_word_count": q_word_count,
            "question_char_count": q_char_count,
            "n_context_docs": n_context_docs,
            "mean_search_first3": float(np.mean(run_search_first3_counts)) if run_search_first3_counts else np.nan,
            "mean_thought_len_step3": float(np.mean(run_thought_lens_at_step3)) if run_thought_lens_at_step3 else np.nan,
            "mean_total_thought_len": float(np.mean(run_total_thought_lens)) if run_total_thought_lens else np.nan,
            "mean_steps": m["mean_steps"],
        })

    print(f"\n  Collected features for {len(rows)} questions")

    predictors = [
        ("Hidden state similarity (S4 L40)", "sim_step4_layer40"),
        ("Question word count", "question_word_count"),
        ("Question char count", "question_char_count"),
        ("N context documents", "n_context_docs"),
        ("Mean Search actions in first 3 steps", "mean_search_first3"),
        ("Mean thought length at step 3 (chars)", "mean_thought_len_step3"),
        ("Mean total thought length (chars)", "mean_total_thought_len"),
        ("Mean step count", "mean_steps"),
        ("Accuracy (correct rate)", "correct_rate"),
    ]

    print(f"\n  {'Predictor':>45} {'n':>4} {'r':>8} {'p':>10} {'Sig':>5}")
    print("  " + "-" * 78)

    predictor_results = {}
    for label, key in predictors:
        vals = [(r[key], r["cv"]) for r in rows if not np.isnan(r.get(key, np.nan))]
        if len(vals) < 5:
            print(f"  {label:>45} {len(vals):>4}    n/a       n/a")
            continue
        xs = np.array([v[0] for v in vals])
        ys = np.array([v[1] for v in vals])
        r, p = stats.pearsonr(xs, ys)
        sig = "*" if p < 0.05 else ""
        print(f"  {label:>45} {len(vals):>4} {r:>8.4f} {p:>10.6f} {sig:>5}")
        predictor_results[key] = {"label": label, "r": float(r), "p": float(p),
                                   "n": len(vals), "significant": bool(p < 0.05)}

    print(f"\n  --- Multiple regression: CV ~ similarity + question_length + accuracy ---")
    valid_rows = [r for r in rows if not np.isnan(r["sim_step4_layer40"])]
    if len(valid_rows) >= 10:
        Y = np.array([r["cv"] for r in valid_rows])
        X = np.column_stack([
            [r["sim_step4_layer40"] for r in valid_rows],
            [r["question_word_count"] for r in valid_rows],
            [r["correct_rate"] for r in valid_rows],
            np.ones(len(valid_rows)),
        ])
        betas, residuals, rank, sv = np.linalg.lstsq(X, Y, rcond=None)
        Y_hat = X @ betas
        ss_res = np.sum((Y - Y_hat) ** 2)
        ss_tot = np.sum((Y - Y.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        n_obs = len(valid_rows)
        k = 3
        adj_r_squared = 1 - (1 - r_squared) * (n_obs - 1) / (n_obs - k - 1) if n_obs > k + 1 else r_squared

        se = np.sqrt(ss_res / (n_obs - k - 1)) if n_obs > k + 1 else 0
        XtX_inv = np.linalg.inv(X.T @ X)
        se_betas = se * np.sqrt(np.diag(XtX_inv))
        t_stats = betas / se_betas if np.all(se_betas > 0) else np.zeros_like(betas)
        p_vals = [2 * (1 - stats.t.cdf(abs(t), df=n_obs - k - 1)) for t in t_stats]

        print(f"  R² = {r_squared:.4f}, Adj R² = {adj_r_squared:.4f}, n = {n_obs}")
        var_names = ["similarity", "q_word_count", "accuracy", "intercept"]
        print(f"  {'Variable':>15} {'Beta':>10} {'SE':>10} {'t':>8} {'p':>10}")
        print("  " + "-" * 55)
        for name, b, s, t, p in zip(var_names, betas, se_betas, t_stats, p_vals):
            sig = "*" if p < 0.05 else ""
            print(f"  {name:>15} {b:>10.4f} {s:>10.4f} {t:>8.3f} {p:>10.6f} {sig}")

        regression_results = {
            "r_squared": float(r_squared), "adj_r_squared": float(adj_r_squared), "n": n_obs,
            "coefficients": {name: {"beta": float(b), "se": float(s), "t": float(t), "p": float(p)}
                             for name, b, s, t, p in zip(var_names, betas, se_betas, t_stats, p_vals)},
        }
    else:
        regression_results = None

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    plot_predictors = [
        ("sim_step4_layer40", "Hidden State Similarity\n(Step 4, Layer 40)"),
        ("question_word_count", "Question Word Count"),
        ("correct_rate", "Accuracy (Correct Rate)"),
        ("mean_search_first3", "Mean Search Actions\n(First 3 Steps)"),
        ("mean_total_thought_len", "Mean Total Thought\nLength (chars)"),
        ("mean_steps", "Mean Step Count"),
    ]
    for idx, (key, xlabel) in enumerate(plot_predictors):
        ax = axes[idx // 3][idx % 3]
        vals = [(r[key], r["cv"], r["difficulty"]) for r in rows if not np.isnan(r.get(key, np.nan))]
        if not vals:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue
        xs_h = [v[0] for v in vals if v[2] == "hard"]
        ys_h = [v[1] for v in vals if v[2] == "hard"]
        xs_e = [v[0] for v in vals if v[2] == "easy"]
        ys_e = [v[1] for v in vals if v[2] == "easy"]
        ax.scatter(xs_h, ys_h, c="#d62728", alpha=0.5, s=30, label="Hard", edgecolors="black", linewidth=0.3)
        ax.scatter(xs_e, ys_e, c="#1f77b4", alpha=0.5, s=30, label="Easy", edgecolors="black", linewidth=0.3)
        all_xs = np.array([v[0] for v in vals])
        all_ys = np.array([v[1] for v in vals])
        r_val, p_val = stats.pearsonr(all_xs, all_ys)
        sig = "*" if p_val < 0.05 else ""
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Behavioral CV", fontsize=10)
        ax.set_title(f"r={r_val:.3f}, p={p_val:.4f}{sig}", fontsize=11,
                      color="red" if p_val < 0.05 else "black")
        if idx == 0:
            ax.legend(fontsize=7, loc="upper right")

    plt.suptitle("Baseline Comparison: What Predicts Behavioral Consistency?", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "baseline_comparison_step4.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: baseline_comparison_step4.png")

    result = {
        "predictors": predictor_results,
        "regression": regression_results,
    }
    with open(OUTPUT_DIR / "baseline_comparison_step4.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: baseline_comparison_step4.json")

    return result


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("PAPER 3: DEFINITIVE 100-QUESTION ANALYSIS")
    print("=" * 70)

    pilot_data = load_pilot_data()
    easier_data = load_easier_data()
    new60_data = load_new60_data()

    all_data = {**pilot_data, **easier_data, **new60_data}
    print(f"\nTotal questions loaded: {len(all_data)}")

    n_hard = sum(1 for v in all_data.values() if v["difficulty"] == "hard")
    n_easy = sum(1 for v in all_data.values() if v["difficulty"] == "easy")
    total_runs = sum(len(v["runs"]) for v in all_data.values())
    print(f"  Hard: {n_hard}, Easy: {n_easy}")
    print(f"  Total runs: {total_runs}")

    print("\nComputing metrics...")
    metrics = []
    for qid, entry in all_data.items():
        m = compute_question_metrics(qid, entry)
        metrics.append(m)
    print(f"  Computed metrics for {len(metrics)} questions")

    r_mat, p_mat, n_mat = analysis1_heatmap(metrics)
    perm_results = analysis2_permutation(metrics)
    split_results = analysis3_hard_vs_easy(metrics)
    cat_counts = analysis4_consistency_categories(metrics)
    step_rs, step_ps = analysis5_step_progression(metrics)
    clf_results = analysis6_classifier(all_data, metrics)
    loo_results = analysis7_loo_sensitivity(metrics)
    inc_results = analysis8_inconsistent_focus(all_data, metrics)
    partial_results = analysis9_partial_correlation(metrics)
    baseline_results = analysis10_baseline_comparison(all_data, metrics)

    print("\n" + "=" * 70)
    print("ALL ANALYSES COMPLETE")
    print("=" * 70)
    print(f"Outputs saved to: {OUTPUT_DIR}")

    summary = {
        "n_questions": len(metrics),
        "n_hard": n_hard,
        "n_easy": n_easy,
        "total_runs": total_runs,
        "heatmap_significant_cells": int(np.sum((~np.isnan(p_mat)) & (p_mat < 0.05))),
        "permutation_significant_layers": sum(1 for v in perm_results.values() if v["significant"]),
        "consistency_categories": dict(cat_counts),
    }
    with open(OUTPUT_DIR / "summary_100q.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
