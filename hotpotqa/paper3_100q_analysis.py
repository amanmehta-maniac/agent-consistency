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
