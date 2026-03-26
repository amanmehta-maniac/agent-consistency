"""
paper3_v3_analyses.py — New analyses for Paper 3 v3 submission.

Task 1: Hard-question signal search (variance, prototype, trajectory, temporal stability)
Task 2: Early-run monitor reframing (k-subset AUROC, PR, early-exit with 5-fold CV)
Task 3: Commitment direction geometry (v_commit, projection, Procrustes cross-model)

Run from hotpotqa/ directory:
    ../.venv/bin/python3 paper3_v3_analyses.py

Outputs JSON results to analysis_results/v3/
"""

import json
import sys
import gc
import numpy as np
from pathlib import Path
from collections import Counter
from itertools import combinations
from scipy import stats
from scipy.spatial import procrustes
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

BASE = Path(__file__).parent

PILOT_DIR = BASE / "pilot_hidden_states_70b"
NEW60_DIR = BASE / "experiment_60q_results"
EASIER_DIR = BASE / "results_easier"
QWEN_DIR = BASE / "qwen_cross_model_100q"

PILOT_Q_FILE = BASE / "pilot_questions.json"
EASIER_Q_FILE = BASE / "easier_questions_selection.json"
NEW60_Q_FILE = BASE / "new_60_questions.json"
QWEN_Q_FILE = BASE / "qwen_100q_full.json"

OUTPUT_DIR = BASE / "analysis_results" / "v3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LLAMA_LAYERS = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80]
STEPS = list(range(1, 6))
SEED = 42

# ═══════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS (reused from reproduce_all_v2.py)
# ═══════════════════════════════════════════════════════════════════════

def pairwise_cosine_sim(vectors):
    """Mean pairwise cosine similarity (upper triangle)."""
    if len(vectors) < 2:
        return np.nan
    arr = np.array(vectors)
    sim_matrix = cosine_similarity(arr)
    n = len(vectors)
    upper_tri = sim_matrix[np.triu_indices(n, k=1)]
    return float(np.mean(upper_tri))


def pairwise_cosine_sim_variance(vectors):
    """Variance of pairwise cosine similarities (upper triangle)."""
    if len(vectors) < 2:
        return np.nan
    arr = np.array(vectors)
    sim_matrix = cosine_similarity(arr)
    n = len(vectors)
    upper_tri = sim_matrix[np.triu_indices(n, k=1)]
    return float(np.var(upper_tri))


def compute_cv(step_counts):
    """Coefficient of variation of step counts."""
    if len(step_counts) < 2 or np.mean(step_counts) == 0:
        return 0.0
    return float(np.std(step_counts) / np.mean(step_counts))


def permutation_test(x, y, n_perms=10000, seed=SEED):
    """Permutation test for Pearson r."""
    rng = np.random.default_rng(seed)
    observed_r, parametric_p = stats.pearsonr(x, y)
    x_arr = np.array(x)
    y_arr = np.array(y)
    count = 0
    for _ in range(n_perms):
        perm_y = rng.permutation(y_arr)
        r_perm, _ = stats.pearsonr(x_arr, perm_y)
        if abs(r_perm) >= abs(observed_r):
            count += 1
    perm_p = (count + 1) / (n_perms + 1)
    return float(observed_r), float(parametric_p), float(perm_p)


def bootstrap_ci(x, y, n_boot=10000, seed=SEED):
    """Bootstrap 95% CI for Pearson r."""
    rng = np.random.default_rng(seed)
    x, y = np.array(x), np.array(y)
    n = len(x)
    if n < 5:
        return np.nan, np.nan
    boot_rs = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if np.std(x[idx]) > 0 and np.std(y[idx]) > 0:
            boot_rs.append(float(stats.pearsonr(x[idx], y[idx])[0]))
    if len(boot_rs) == 0:
        return np.nan, np.nan
    boot_rs = np.array(boot_rs)
    return float(np.percentile(boot_rs, 2.5)), float(np.percentile(boot_rs, 97.5))


def cohens_d_from_split(vals, labels, group_a, group_b):
    """Cohen's d between two groups."""
    a = np.array([v for v, l in zip(vals, labels) if l == group_a])
    b = np.array([v for v, l in zip(vals, labels) if l == group_b])
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    if pooled_std == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def extract_yes_no(text):
    if not text:
        return None
    text = str(text).lower().strip()
    if "yes" in text and "no" not in text:
        return "yes"
    if "no" in text:
        return "no"
    return None


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING — retains raw hidden states for Tasks 1, 2, 3
# ═══════════════════════════════════════════════════════════════════════

def load_npy_question(qdir, layers_to_load=None):
    """Load a single npy-format question directory. Returns list of run dicts
    with hidden_states[step] = array of shape (n_layers, hidden_dim), retaining
    only the requested layers."""
    runs = []
    for run_dir in sorted(qdir.glob("run_*")):
        meta_file = run_dir / "metadata.json"
        traj_file = run_dir / "trajectory.json"
        if not meta_file.exists():
            continue
        with open(meta_file) as f:
            meta = json.load(f)
        traj = {}
        if traj_file.exists():
            with open(traj_file) as f:
                traj = json.load(f)

        hidden_states = {}
        for hs_file in run_dir.glob("hidden_states_step_*.npy"):
            step_num = int(hs_file.stem.split("_")[-1])
            hs = np.load(hs_file)  # shape (81, 8192) or (41, 5120)
            hidden_states[step_num] = hs

        runs.append({
            "step_count": meta.get("num_steps") or len(traj.get("steps", [])),
            "hidden_states": hidden_states,
            "correct": meta.get("correct"),
        })
    return runs


def load_easier_question(fp, expected_answer, layer_set=None):
    """Load a single easier JSON-format question. Returns list of run dicts."""
    if layer_set is None:
        layer_set = set(LLAMA_LAYERS)
    with open(fp) as f:
        data = json.load(f)
    runs = []
    for run in data.get("runs", []):
        hidden_states = {}
        for step in run.get("steps", []):
            step_num = step.get("step_number", 0)
            layers_dict = step.get("hidden_states", {}).get("layers", {})
            if layers_dict:
                full = np.zeros((81, 8192), dtype=np.float32)
                for i in layer_set:
                    lk = f"layer_{i}"
                    if lk in layers_dict and layers_dict[lk]:
                        full[i] = layers_dict[lk]
                hidden_states[step_num] = full
        ans = extract_yes_no(run.get("final_answer"))
        correct = (ans == expected_answer) if ans else False
        runs.append({
            "step_count": len(run.get("steps", [])),
            "hidden_states": hidden_states,
            "correct": correct,
        })
    del data
    return runs


def load_all_llama_data():
    """Load all 100 Llama questions with raw hidden states retained.
    Returns dict: qid -> {runs, difficulty, correct_rate, cv, ...}"""
    print("\n" + "=" * 70)
    print("LOADING LLAMA 100q DATA (retaining hidden states)")
    print("=" * 70)

    with open(PILOT_Q_FILE) as f:
        pilot_qs = {q["id"]: q for q in json.load(f)}
    with open(EASIER_Q_FILE) as f:
        easier_qs = {q["id"]: q for q in json.load(f)}
    with open(NEW60_Q_FILE) as f:
        new60_qs = {q["id"]: q for q in json.load(f)}

    all_data = {}

    # Pilot: 20 hard questions (npy)
    print("  Loading pilot (20 hard, npy)...")
    for qdir in sorted(PILOT_DIR.iterdir()):
        if not qdir.is_dir():
            continue
        qid = qdir.name
        if qid not in pilot_qs:
            continue
        runs = load_npy_question(qdir)
        if runs:
            all_data[qid] = {"runs": runs, "difficulty": "hard"}

    # New 60 questions (npy)
    print("  Loading new60 (60 questions, npy)...")
    for qdir in sorted(NEW60_DIR.iterdir()):
        if not qdir.is_dir():
            continue
        qid = qdir.name
        if qid not in new60_qs:
            continue
        runs = load_npy_question(qdir)
        if runs:
            diff = new60_qs[qid].get("difficulty", "unknown")
            all_data[qid] = {"runs": runs, "difficulty": diff}

    # Easier 20 questions (JSON)
    print("  Loading easier (20 easy, JSON)...")
    layer_set = set(LLAMA_LAYERS)
    for fp in sorted(EASIER_DIR.glob("*.json")):
        qid = fp.stem
        if qid not in easier_qs or qid in all_data:
            continue
        expected = easier_qs[qid].get("answer", "").lower().strip()
        runs = load_easier_question(fp, expected, layer_set)
        if runs:
            all_data[qid] = {"runs": runs, "difficulty": "easy"}

    # Compute per-question summary metrics
    for qid, entry in all_data.items():
        runs = entry["runs"]
        step_counts = [r["step_count"] for r in runs]
        entry["cv"] = compute_cv(step_counts)
        entry["correct_rate"] = sum(1 for r in runs if r.get("correct")) / len(runs)
        entry["n_runs"] = len(runs)
        entry["mean_steps"] = float(np.mean(step_counts))

    hard = sum(1 for e in all_data.values() if e["difficulty"] == "hard")
    easy = sum(1 for e in all_data.values() if e["difficulty"] == "easy")
    print(f"  Loaded {len(all_data)} questions ({hard} hard, {easy} easy)")
    return all_data


def load_qwen_data():
    """Load Qwen 100q with raw hidden states retained."""
    print("\n" + "=" * 70)
    print("LOADING QWEN 100q DATA (retaining hidden states)")
    print("=" * 70)

    with open(QWEN_Q_FILE) as f:
        qwen_qs = {q["id"]: q for q in json.load(f)}

    # Build difficulty map from Llama question files
    diff_map = {}
    with open(PILOT_Q_FILE) as f:
        for q in json.load(f):
            diff_map[q["id"]] = "hard"
    with open(EASIER_Q_FILE) as f:
        for q in json.load(f):
            diff_map[q["id"]] = "easy"
    with open(NEW60_Q_FILE) as f:
        for q in json.load(f):
            diff_map[q["id"]] = q.get("difficulty", "unknown")

    all_data = {}
    for qdir in sorted(QWEN_DIR.iterdir()):
        if not qdir.is_dir():
            continue
        qid = qdir.name
        if qid not in qwen_qs:
            continue
        runs = load_npy_question(qdir)
        if runs:
            d = qwen_qs[qid].get("difficulty", "none")
            if d == "none" and qid in diff_map:
                d = diff_map[qid]
            all_data[qid] = {"runs": runs, "difficulty": d}

    for qid, entry in all_data.items():
        runs = entry["runs"]
        step_counts = [r["step_count"] for r in runs]
        entry["cv"] = compute_cv(step_counts)
        entry["correct_rate"] = sum(1 for r in runs if r.get("correct")) / len(runs)
        entry["n_runs"] = len(runs)

    print(f"  Loaded {len(all_data)} questions")
    return all_data


def get_hidden_states_at(data, qid, step, layer):
    """Extract hidden states for all runs of a question at given step/layer.
    Returns list of 1D vectors (one per run)."""
    entry = data[qid]
    vectors = []
    for run in entry["runs"]:
        hs = run["hidden_states"].get(step)
        if hs is not None and len(hs) > layer:
            vectors.append(hs[layer].astype(np.float32))
    return vectors


def classify_questions(data):
    """Classify questions into commitment categories."""
    for qid, entry in data.items():
        cr = entry["correct_rate"]
        cv = entry["cv"]
        if cr >= 0.8:
            entry["category"] = "committed-correct"
        elif cr <= 0.2:
            if cv <= 0.15:
                entry["category"] = "committed-wrong"
            else:
                entry["category"] = "uncommitted-wrong"
        else:
            entry["category"] = "mixed"


# ═══════════════════════════════════════════════════════════════════════
# TASK 1: HARD-QUESTION SIGNAL SEARCH
# ═══════════════════════════════════════════════════════════════════════

def task1_hard_question_signal_search(data):
    """Search for a hidden-state signal predicting consistency on hard questions.

    Four approaches at step 4:
    (a) Variance of activation similarity
    (b) Prototype similarity to hard-question centroid
    (c) Trajectory slope/curvature across steps 1-4
    (d) Temporal stability (step-3 → step-4 within-run similarity)
    """
    print("\n" + "=" * 70)
    print("TASK 1: HARD-QUESTION SIGNAL SEARCH")
    print("=" * 70)

    hard_qids = [qid for qid, e in data.items() if e["difficulty"] == "hard"]
    print(f"  n = {len(hard_qids)} hard questions")

    results = {"n_hard": len(hard_qids)}
    SEARCH_LAYERS = [32, 40, 48, 56, 64, 72, 80]

    # ── 1a: Variance of pairwise cosine similarity ──
    print("\n  --- 1a: Variance of activation similarity ---")
    best_1a = {"r": 0, "layer": None}
    results_1a = {}

    for layer in SEARCH_LAYERS:
        variances = []
        cvs = []
        for qid in hard_qids:
            vectors = get_hidden_states_at(data, qid, step=4, layer=layer)
            if len(vectors) >= 2:
                var = pairwise_cosine_sim_variance(vectors)
                variances.append(var)
                cvs.append(data[qid]["cv"])

        if len(variances) >= 10:
            r, p = stats.pearsonr(variances, cvs)
            results_1a[layer] = {"r": round(r, 4), "p": round(p, 6), "n": len(variances)}
            print(f"    Layer {layer}: r={r:.4f}, p={p:.4f}, n={len(variances)}")
            if abs(r) > abs(best_1a["r"]):
                best_1a = {"r": r, "p": p, "layer": layer, "n": len(variances)}

    results["1a_variance"] = results_1a
    results["1a_best"] = {k: round(v, 6) if isinstance(v, float) else v
                          for k, v in best_1a.items()}

    # ── 1b: Prototype similarity (distance to hard-question centroid) ──
    print("\n  --- 1b: Prototype similarity to hard-question centroid ---")
    best_1b = {"r": 0, "layer": None}
    results_1b = {}

    for layer in SEARCH_LAYERS:
        # Compute mean hidden state per question (mean across runs at step 4)
        mean_hs = {}
        for qid in hard_qids:
            vectors = get_hidden_states_at(data, qid, step=4, layer=layer)
            if len(vectors) >= 2:
                mean_hs[qid] = np.mean(vectors, axis=0)

        if len(mean_hs) < 10:
            continue

        # Centroid = mean of all hard-question mean hidden states
        centroid = np.mean(list(mean_hs.values()), axis=0)

        proto_sims = []
        cvs = []
        for qid in mean_hs:
            sim = float(cosine_similarity(
                mean_hs[qid].reshape(1, -1), centroid.reshape(1, -1)
            )[0, 0])
            proto_sims.append(sim)
            cvs.append(data[qid]["cv"])

        r, p = stats.pearsonr(proto_sims, cvs)
        results_1b[layer] = {"r": round(r, 4), "p": round(p, 6), "n": len(proto_sims)}
        print(f"    Layer {layer}: r={r:.4f}, p={p:.4f}, n={len(proto_sims)}")
        if abs(r) > abs(best_1b["r"]):
            best_1b = {"r": r, "p": p, "layer": layer, "n": len(proto_sims)}

    results["1b_prototype"] = results_1b
    results["1b_best"] = {k: round(v, 6) if isinstance(v, float) else v
                          for k, v in best_1b.items()}

    # ── 1c: Trajectory slope and curvature (steps 1-4 at peak layer 40) ──
    print("\n  --- 1c: Trajectory slope/curvature (steps 1-4, layer 40) ---")
    TRAJ_LAYER = 40
    slopes = []
    curvatures = []
    cvs_traj = []

    for qid in hard_qids:
        step_sims = []
        for step in [1, 2, 3, 4]:
            vectors = get_hidden_states_at(data, qid, step=step, layer=TRAJ_LAYER)
            if len(vectors) >= 2:
                step_sims.append(pairwise_cosine_sim(vectors))
            else:
                step_sims.append(np.nan)

        if any(np.isnan(s) for s in step_sims):
            continue

        # Linear fit: slope
        x = np.array([1, 2, 3, 4], dtype=float)
        y = np.array(step_sims)
        slope, intercept = np.polyfit(x, y, 1)
        slopes.append(slope)

        # Quadratic fit: curvature = 2*a (second derivative)
        coeffs = np.polyfit(x, y, 2)
        curvature = 2 * coeffs[0]
        curvatures.append(curvature)

        cvs_traj.append(data[qid]["cv"])

    results_1c = {}
    if len(slopes) >= 10:
        r_slope, p_slope = stats.pearsonr(slopes, cvs_traj)
        r_curv, p_curv = stats.pearsonr(curvatures, cvs_traj)
        results_1c = {
            "slope": {"r": round(r_slope, 4), "p": round(p_slope, 6)},
            "curvature": {"r": round(r_curv, 4), "p": round(p_curv, 6)},
            "n": len(slopes),
        }
        print(f"    Slope: r={r_slope:.4f}, p={p_slope:.4f}")
        print(f"    Curvature: r={r_curv:.4f}, p={p_curv:.4f}")
        print(f"    n = {len(slopes)}")

    results["1c_trajectory"] = results_1c

    # ── 1d: Temporal stability (step-3 → step-4 within-run cosine) ──
    print("\n  --- 1d: Temporal stability (step-3→step-4, layer 40) ---")
    best_1d = {"r": 0, "layer": None}
    results_1d = {}

    for layer in SEARCH_LAYERS:
        stabilities = []
        cvs_stab = []
        for qid in hard_qids:
            run_sims = []
            for run in data[qid]["runs"]:
                hs3 = run["hidden_states"].get(3)
                hs4 = run["hidden_states"].get(4)
                if hs3 is not None and hs4 is not None and len(hs3) > layer and len(hs4) > layer:
                    sim = float(cosine_similarity(
                        hs3[layer].reshape(1, -1).astype(np.float32),
                        hs4[layer].reshape(1, -1).astype(np.float32)
                    )[0, 0])
                    run_sims.append(sim)
            if len(run_sims) >= 2:
                stabilities.append(np.mean(run_sims))
                cvs_stab.append(data[qid]["cv"])

        if len(stabilities) >= 10:
            r, p = stats.pearsonr(stabilities, cvs_stab)
            results_1d[layer] = {"r": round(r, 4), "p": round(p, 6), "n": len(stabilities)}
            print(f"    Layer {layer}: r={r:.4f}, p={p:.4f}, n={len(stabilities)}")
            if abs(r) > abs(best_1d["r"]):
                best_1d = {"r": r, "p": p, "layer": layer, "n": len(stabilities)}

    results["1d_temporal"] = results_1d
    results["1d_best"] = {k: round(v, 6) if isinstance(v, float) else v
                          for k, v in best_1d.items()}

    # ── Summary: find any signal with |r| > 0.20, run permutation test ──
    print("\n  --- Summary ---")
    significant_signals = []

    for label, best in [("1a_variance", best_1a), ("1b_prototype", best_1b),
                        ("1d_temporal", best_1d)]:
        if best.get("layer") is not None and abs(best["r"]) > 0.20:
            print(f"  {label}: r={best['r']:.4f} at layer {best['layer']} — running permutation test...")

            # Recompute the signal for permutation test
            if label == "1a_variance":
                signal_vals, signal_cvs = [], []
                for qid in hard_qids:
                    vectors = get_hidden_states_at(data, qid, step=4, layer=best["layer"])
                    if len(vectors) >= 2:
                        signal_vals.append(pairwise_cosine_sim_variance(vectors))
                        signal_cvs.append(data[qid]["cv"])
            elif label == "1b_prototype":
                mean_hs = {}
                for qid in hard_qids:
                    vectors = get_hidden_states_at(data, qid, step=4, layer=best["layer"])
                    if len(vectors) >= 2:
                        mean_hs[qid] = np.mean(vectors, axis=0)
                centroid = np.mean(list(mean_hs.values()), axis=0)
                signal_vals, signal_cvs = [], []
                for qid in mean_hs:
                    sim = float(cosine_similarity(
                        mean_hs[qid].reshape(1, -1), centroid.reshape(1, -1)
                    )[0, 0])
                    signal_vals.append(sim)
                    signal_cvs.append(data[qid]["cv"])
            elif label == "1d_temporal":
                signal_vals, signal_cvs = [], []
                for qid in hard_qids:
                    run_sims = []
                    for run in data[qid]["runs"]:
                        hs3 = run["hidden_states"].get(3)
                        hs4 = run["hidden_states"].get(4)
                        if (hs3 is not None and hs4 is not None and
                                len(hs3) > best["layer"] and len(hs4) > best["layer"]):
                            sim = float(cosine_similarity(
                                hs3[best["layer"]].reshape(1, -1).astype(np.float32),
                                hs4[best["layer"]].reshape(1, -1).astype(np.float32)
                            )[0, 0])
                            run_sims.append(sim)
                    if len(run_sims) >= 2:
                        signal_vals.append(np.mean(run_sims))
                        signal_cvs.append(data[qid]["cv"])

            obs_r, param_p, perm_p = permutation_test(signal_vals, signal_cvs)
            ci_lo, ci_hi = bootstrap_ci(signal_vals, signal_cvs)

            # Median-split Cohen's d
            med = np.median(signal_cvs)
            labels_split = ["high" if c >= med else "low" for c in signal_cvs]
            d = cohens_d_from_split(signal_vals, labels_split, "high", "low")

            perm_result = {
                "r": round(obs_r, 4), "param_p": round(param_p, 6),
                "perm_p": round(perm_p, 6), "ci": [round(ci_lo, 3), round(ci_hi, 3)],
                "cohens_d": round(d, 3), "layer": best["layer"], "n": len(signal_vals),
            }
            results[f"{label}_permutation"] = perm_result
            print(f"    obs_r={obs_r:.4f}, perm_p={perm_p:.4f}, d={d:.3f}, CI=[{ci_lo:.3f},{ci_hi:.3f}]")

            if perm_p < 0.05:
                significant_signals.append({
                    "metric": label, "r": obs_r, "perm_p": perm_p,
                    "layer": best["layer"], "cohens_d": d,
                })

    # Check trajectory signals
    if results_1c:
        for metric_name in ["slope", "curvature"]:
            r_val = results_1c[metric_name]["r"]
            if abs(r_val) > 0.20:
                print(f"  1c_{metric_name}: r={r_val:.4f} — running permutation test...")
                if metric_name == "slope":
                    obs_r, param_p, perm_p = permutation_test(slopes, cvs_traj)
                    ci_lo, ci_hi = bootstrap_ci(slopes, cvs_traj)
                    med = np.median(cvs_traj)
                    labels_split = ["high" if c >= med else "low" for c in cvs_traj]
                    d = cohens_d_from_split(slopes, labels_split, "high", "low")
                else:
                    obs_r, param_p, perm_p = permutation_test(curvatures, cvs_traj)
                    ci_lo, ci_hi = bootstrap_ci(curvatures, cvs_traj)
                    med = np.median(cvs_traj)
                    labels_split = ["high" if c >= med else "low" for c in cvs_traj]
                    d = cohens_d_from_split(curvatures, labels_split, "high", "low")

                perm_result = {
                    "r": round(obs_r, 4), "param_p": round(param_p, 6),
                    "perm_p": round(perm_p, 6), "ci": [round(ci_lo, 3), round(ci_hi, 3)],
                    "cohens_d": round(d, 3), "n": len(cvs_traj),
                }
                results[f"1c_{metric_name}_permutation"] = perm_result
                print(f"    obs_r={obs_r:.4f}, perm_p={perm_p:.4f}, d={d:.3f}")

                if perm_p < 0.05:
                    significant_signals.append({
                        "metric": f"1c_{metric_name}", "r": obs_r,
                        "perm_p": perm_p, "cohens_d": d,
                    })

    results["significant_signals"] = significant_signals
    results["any_significant"] = len(significant_signals) > 0
    print(f"\n  Significant signals found: {len(significant_signals)}")
    for sig in significant_signals:
        print(f"    {sig['metric']}: r={sig['r']:.4f}, perm_p={sig['perm_p']:.4f}")

    return results


# ═══════════════════════════════════════════════════════════════════════
# TASK 2: EARLY-RUN MONITOR REFRAMING
# ═══════════════════════════════════════════════════════════════════════

def task2_early_run_monitor(data):
    """Show AUROC works with k=3 runs instead of k=10.

    (a) k-subset AUROC for k ∈ {2,3,4,5} with 500 bootstrap iterations
    (b) Precision-recall at k=3 with precision ≥ 0.80 operating point
    (c) Early-exit simulation with 5-fold CV threshold selection
    """
    print("\n" + "=" * 70)
    print("TASK 2: EARLY-RUN MONITOR REFRAMING")
    print("=" * 70)

    results = {}

    # Prepare data: activation similarity at step 4, layer 40 for all 100 questions
    STEP, LAYER = 4, 40
    all_qids = sorted(data.keys())

    # Use layer profile features (layers 32-80) for LOOCV, matching paper's Feature D
    PROFILE_LAYERS = [32, 40, 48, 56, 64, 72, 80]

    # Precompute full 10-run similarities and layer profiles
    full_sims = {}
    full_profiles = {}
    cvs = {}
    for qid in all_qids:
        vectors_by_layer = {}
        for layer in PROFILE_LAYERS:
            vectors_by_layer[layer] = get_hidden_states_at(data, qid, step=STEP, layer=layer)

        # Peak similarity (layer 40)
        if len(vectors_by_layer[LAYER]) >= 2:
            full_sims[qid] = pairwise_cosine_sim(vectors_by_layer[LAYER])
            # Layer profile
            profile = []
            for layer in PROFILE_LAYERS:
                if len(vectors_by_layer[layer]) >= 2:
                    profile.append(pairwise_cosine_sim(vectors_by_layer[layer]))
                else:
                    profile.append(np.nan)
            full_profiles[qid] = profile
            cvs[qid] = data[qid]["cv"]

    valid_qids = sorted(full_sims.keys())
    print(f"  {len(valid_qids)} questions with valid similarity data")

    # Labels: quintile split (top/bottom 20% of CV)
    cv_arr = np.array([cvs[qid] for qid in valid_qids])
    low_thresh = np.percentile(cv_arr, 20)
    high_thresh = np.percentile(cv_arr, 80)
    mask = (cv_arr <= low_thresh) | (cv_arr >= high_thresh)
    quintile_labels = np.array([1 if cvs[qid] >= high_thresh else 0 for qid in valid_qids])
    quintile_idx = np.where(mask)[0]
    quintile_qids = [valid_qids[i] for i in quintile_idx]
    quintile_y = quintile_labels[quintile_idx]

    print(f"  Quintile subset: {len(quintile_qids)} questions "
          f"({sum(quintile_y)} high-CV, {len(quintile_y)-sum(quintile_y)} low-CV)")

    # ── 2a: k-subset AUROC ──
    print("\n  --- 2a: k-subset AUROC ---")
    rng = np.random.default_rng(SEED)
    N_BOOT = 500

    k_auroc_results = {}
    for k in [2, 3, 4, 5, 10]:
        boot_aucs = []
        for b in range(N_BOOT):
            # For each question, subsample k runs and recompute layer profile
            profiles_k = []
            for qid in quintile_qids:
                n_runs = data[qid]["n_runs"]
                if n_runs < k:
                    # Use all runs if fewer than k
                    run_idx = list(range(n_runs))
                else:
                    run_idx = sorted(rng.choice(n_runs, k, replace=False))

                # Compute layer profile from k-subset
                profile = []
                for layer in PROFILE_LAYERS:
                    vectors = []
                    for ri in run_idx:
                        run = data[qid]["runs"][ri]
                        hs = run["hidden_states"].get(STEP)
                        if hs is not None and len(hs) > layer:
                            vectors.append(hs[layer].astype(np.float32))
                    if len(vectors) >= 2:
                        profile.append(pairwise_cosine_sim(vectors))
                    else:
                        profile.append(np.nan)
                profiles_k.append(profile)

            X = np.nan_to_num(np.array(profiles_k))
            y = quintile_y

            # LOOCV
            n = len(X)
            probs = np.zeros(n)
            for i in range(n):
                X_train = np.delete(X, i, axis=0)
                y_train = np.delete(y, i)
                X_test = X[i:i+1]

                if len(set(y_train)) < 2:
                    probs[i] = 0.5
                    continue

                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)

                clf = LogisticRegression(max_iter=1000, random_state=SEED)
                clf.fit(X_train_s, y_train)
                probs[i] = clf.predict_proba(X_test_s)[0, 1]

            try:
                auc = roc_auc_score(y, probs)
                boot_aucs.append(auc)
            except ValueError:
                pass

        if boot_aucs:
            mean_auc = np.mean(boot_aucs)
            std_auc = np.std(boot_aucs)
            k_auroc_results[k] = {
                "mean": round(mean_auc, 3),
                "std": round(std_auc, 3),
                "ci_lo": round(np.percentile(boot_aucs, 2.5), 3),
                "ci_hi": round(np.percentile(boot_aucs, 97.5), 3),
            }
            print(f"    k={k}: AUROC={mean_auc:.3f} ± {std_auc:.3f} "
                  f"[{k_auroc_results[k]['ci_lo']}, {k_auroc_results[k]['ci_hi']}]")

    results["2a_k_auroc"] = k_auroc_results

    # ── 2b: Precision-recall at k=3 ──
    print("\n  --- 2b: Precision-recall at k=3 ---")
    # Use the k=3 LOOCV predictions from a single bootstrap
    # (deterministic: use first 3 runs for each question)
    k = 3
    profiles_k3 = []
    for qid in quintile_qids:
        n_runs = min(data[qid]["n_runs"], k)
        run_idx = list(range(n_runs))
        profile = []
        for layer in PROFILE_LAYERS:
            vectors = []
            for ri in run_idx:
                run = data[qid]["runs"][ri]
                hs = run["hidden_states"].get(STEP)
                if hs is not None and len(hs) > layer:
                    vectors.append(hs[layer].astype(np.float32))
            if len(vectors) >= 2:
                profile.append(pairwise_cosine_sim(vectors))
            else:
                profile.append(np.nan)
        profiles_k3.append(profile)

    X_k3 = np.nan_to_num(np.array(profiles_k3))
    n = len(X_k3)
    probs_k3 = np.zeros(n)
    for i in range(n):
        X_train = np.delete(X_k3, i, axis=0)
        y_train = np.delete(quintile_y, i)
        X_test = X_k3[i:i+1]
        if len(set(y_train)) < 2:
            probs_k3[i] = 0.5
            continue
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        clf = LogisticRegression(max_iter=1000, random_state=SEED)
        clf.fit(X_train_s, y_train)
        probs_k3[i] = clf.predict_proba(X_test_s)[0, 1]

    precision, recall, thresholds = precision_recall_curve(quintile_y, probs_k3)

    # Find operating point with precision >= 0.80
    valid_mask = precision >= 0.80
    if np.any(valid_mask):
        # Among valid points, pick the one with highest recall
        valid_idx = np.where(valid_mask)[0]
        best_idx = valid_idx[np.argmax(recall[valid_idx])]
        op_precision = float(precision[best_idx])
        op_recall = float(recall[best_idx])
        op_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 1.0
    else:
        op_precision = float(np.max(precision))
        op_recall = 0.0
        op_threshold = 1.0

    auc_k3 = roc_auc_score(quintile_y, probs_k3)
    results["2b_pr_k3"] = {
        "auroc": round(auc_k3, 3),
        "precision_at_op": round(op_precision, 3),
        "recall_at_op": round(op_recall, 3),
        "threshold": round(op_threshold, 3),
        "precision_curve": [round(float(p), 4) for p in precision[:20]],
        "recall_curve": [round(float(r), 4) for r in recall[:20]],
    }
    print(f"    k=3 AUROC: {auc_k3:.3f}")
    print(f"    Operating point (precision≥0.80): precision={op_precision:.3f}, "
          f"recall={op_recall:.3f}, threshold={op_threshold:.3f}")

    # ── 2c: Early-exit simulation with 5-fold CV threshold ──
    print("\n  --- 2c: Early-exit simulation (5-fold CV threshold) ---")

    # Use ALL questions (not just quintile) for this practical simulation
    # For each question: run 3 trials → compute sim → if above threshold, predict
    # "consistent" and stop; else run 7 more and majority vote
    # Compare vs fixed majority vote over 10 runs

    # First, establish ground truth: is question consistent? (median split on CV)
    all_cv = np.array([cvs[qid] for qid in valid_qids])
    median_cv = np.median(all_cv)
    # consistent = low CV (below median), label=0
    # inconsistent = high CV (above median), label=1
    all_labels = np.array([1 if cvs[qid] >= median_cv else 0 for qid in valid_qids])

    # For each question, compute k=3 similarity (deterministic: first 3 runs)
    all_sim_k3 = []
    for qid in valid_qids:
        n_runs = min(data[qid]["n_runs"], 3)
        vectors = []
        for ri in range(n_runs):
            run = data[qid]["runs"][ri]
            hs = run["hidden_states"].get(STEP)
            if hs is not None and len(hs) > LAYER:
                vectors.append(hs[LAYER].astype(np.float32))
        if len(vectors) >= 2:
            all_sim_k3.append(pairwise_cosine_sim(vectors))
        else:
            all_sim_k3.append(np.nan)
    all_sim_k3 = np.array(all_sim_k3)

    # 5-fold CV: select threshold on train, evaluate on test
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(valid_qids)):
        train_sims = all_sim_k3[train_idx]
        train_labels = all_labels[train_idx]
        test_sims = all_sim_k3[test_idx]
        test_labels = all_labels[test_idx]

        # Sweep thresholds on training fold: find best accuracy
        best_thresh = None
        best_acc = 0
        for t in np.linspace(np.nanmin(train_sims), np.nanmax(train_sims), 200):
            # Predict: sim > t → consistent (label=0), else → inconsistent (label=1)
            preds = np.array([0 if s > t else 1 for s in train_sims])
            valid = ~np.isnan(train_sims)
            acc = np.mean(preds[valid] == train_labels[valid])
            if acc > best_acc:
                best_acc = acc
                best_thresh = t

        # Evaluate on test fold
        test_preds = np.array([0 if s > best_thresh else 1 for s in test_sims])
        test_valid = ~np.isnan(test_sims)
        test_acc = np.mean(test_preds[test_valid] == test_labels[test_valid])

        # Early-exit: count how many stop at k=3 (those above threshold)
        n_early_exit = np.sum(test_sims[test_valid] > best_thresh)
        n_total = np.sum(test_valid)
        compute_savings = float(n_early_exit) / n_total * 0.7 if n_total > 0 else 0
        # 0.7 because early-exit saves 7 of 10 runs for those questions

        fold_results.append({
            "fold": fold_idx,
            "threshold": round(float(best_thresh), 4),
            "train_acc": round(float(best_acc), 4),
            "test_acc": round(float(test_acc), 4),
            "n_early_exit": int(n_early_exit),
            "n_total": int(n_total),
            "compute_savings_pct": round(compute_savings * 100, 1),
        })
        print(f"    Fold {fold_idx}: thresh={best_thresh:.4f}, "
              f"test_acc={test_acc:.4f}, early_exit={n_early_exit}/{n_total}, "
              f"savings={compute_savings*100:.1f}%")

    # Baseline: fixed majority vote accuracy (predict modal class)
    majority_baseline = float(np.mean(all_labels == stats.mode(all_labels, keepdims=False).mode))

    mean_test_acc = np.mean([f["test_acc"] for f in fold_results])
    mean_savings = np.mean([f["compute_savings_pct"] for f in fold_results])

    results["2c_early_exit"] = {
        "folds": fold_results,
        "mean_test_acc": round(mean_test_acc, 4),
        "mean_compute_savings_pct": round(mean_savings, 1),
        "majority_baseline_acc": round(majority_baseline, 4),
        "accuracy_diff": round(mean_test_acc - majority_baseline, 4),
    }
    print(f"\n    Mean test accuracy: {mean_test_acc:.4f}")
    print(f"    Majority baseline: {majority_baseline:.4f}")
    print(f"    Improvement: {mean_test_acc - majority_baseline:+.4f}")
    print(f"    Mean compute savings: {mean_savings:.1f}%")

    return results


# ═══════════════════════════════════════════════════════════════════════
# TASK 3: COMMITMENT DIRECTION GEOMETRY
# ═══════════════════════════════════════════════════════════════════════

def task3_commitment_geometry(llama_data, qwen_data):
    """Characterize commitment as a linear direction.

    (a) Compute v_commit = mean(committed-correct HS) - mean(uncommitted-wrong HS)
    (b) Cosine similarity to PC1s and easy-hard difference vector
    (c) Project all 100 questions onto v_commit and correlate with CV
    (d) Repeat for Qwen at layer 64 with Procrustes alignment
    """
    print("\n" + "=" * 70)
    print("TASK 3: COMMITMENT DIRECTION GEOMETRY")
    print("=" * 70)

    results = {}

    # Classify questions
    classify_questions(llama_data)
    classify_questions(qwen_data)

    # ── 3a: Compute v_commit for Llama at layer 40 ──
    print("\n  --- 3a: Commitment vector (Llama, layer 40) ---")
    LLAMA_COMMIT_LAYER = 40
    STEP = 4

    # Collect mean hidden states per question by category
    category_hs = {"committed-correct": [], "committed-wrong": [],
                   "uncommitted-wrong": [], "mixed": []}
    qid_mean_hs = {}  # all questions

    for qid, entry in llama_data.items():
        vectors = get_hidden_states_at(llama_data, qid, step=STEP, layer=LLAMA_COMMIT_LAYER)
        if len(vectors) >= 2:
            mean_vec = np.mean(vectors, axis=0)
            qid_mean_hs[qid] = mean_vec
            cat = entry.get("category", "mixed")
            if cat in category_hs:
                category_hs[cat].append(mean_vec)

    n_cc = len(category_hs["committed-correct"])
    n_uw = len(category_hs["uncommitted-wrong"])
    print(f"    committed-correct: {n_cc}")
    print(f"    committed-wrong: {len(category_hs['committed-wrong'])}")
    print(f"    uncommitted-wrong: {n_uw}")
    print(f"    mixed: {len(category_hs['mixed'])}")

    if n_cc < 2 or n_uw < 2:
        print("    INSUFFICIENT DATA for commitment vector — need ≥2 per category")
        results["error"] = "insufficient data for v_commit"
        return results

    centroid_cc = np.mean(category_hs["committed-correct"], axis=0)
    centroid_uw = np.mean(category_hs["uncommitted-wrong"], axis=0)
    v_commit_llama = centroid_cc - centroid_uw
    v_commit_llama_norm = v_commit_llama / np.linalg.norm(v_commit_llama)

    results["3a_llama"] = {
        "layer": LLAMA_COMMIT_LAYER,
        "n_committed_correct": n_cc,
        "n_uncommitted_wrong": n_uw,
        "v_commit_norm": round(float(np.linalg.norm(v_commit_llama)), 2),
    }
    print(f"    v_commit norm: {np.linalg.norm(v_commit_llama):.2f}")

    # ── 3b: Geometric relationships ──
    print("\n  --- 3b: Geometric relationships ---")

    # Collect all mean hidden states
    all_hs_matrix = []
    all_qids_ordered = []
    for qid in sorted(qid_mean_hs.keys()):
        all_hs_matrix.append(qid_mean_hs[qid])
        all_qids_ordered.append(qid)
    all_hs_matrix = np.array(all_hs_matrix)

    # PC1 of all questions
    pca_all = PCA(n_components=50, random_state=SEED)
    pca_all.fit(all_hs_matrix)
    pc1_all = pca_all.components_[0]
    cos_pc1_all = float(cosine_similarity(
        v_commit_llama_norm.reshape(1, -1), pc1_all.reshape(1, -1)
    )[0, 0])

    # PC1 of hard questions only
    hard_hs = np.array([qid_mean_hs[qid] for qid in all_qids_ordered
                        if llama_data[qid]["difficulty"] == "hard" and qid in qid_mean_hs])
    if len(hard_hs) >= 10:
        pca_hard = PCA(n_components=min(50, len(hard_hs) - 1), random_state=SEED)
        pca_hard.fit(hard_hs)
        pc1_hard = pca_hard.components_[0]
        cos_pc1_hard = float(cosine_similarity(
            v_commit_llama_norm.reshape(1, -1), pc1_hard.reshape(1, -1)
        )[0, 0])
    else:
        cos_pc1_hard = np.nan

    # Easy-hard centroid difference
    easy_hs = [qid_mean_hs[qid] for qid in all_qids_ordered
               if llama_data[qid]["difficulty"] == "easy" and qid in qid_mean_hs]
    hard_hs_list = [qid_mean_hs[qid] for qid in all_qids_ordered
                    if llama_data[qid]["difficulty"] == "hard" and qid in qid_mean_hs]
    if easy_hs and hard_hs_list:
        easy_centroid = np.mean(easy_hs, axis=0)
        hard_centroid = np.mean(hard_hs_list, axis=0)
        diff_vec = easy_centroid - hard_centroid
        diff_vec_norm = diff_vec / np.linalg.norm(diff_vec)
        cos_diff = float(cosine_similarity(
            v_commit_llama_norm.reshape(1, -1), diff_vec_norm.reshape(1, -1)
        )[0, 0])
    else:
        cos_diff = np.nan

    results["3b_geometry"] = {
        "cos_v_commit_pc1_all": round(cos_pc1_all, 4),
        "cos_v_commit_pc1_hard": round(cos_pc1_hard, 4) if not np.isnan(cos_pc1_hard) else None,
        "cos_v_commit_easy_hard_diff": round(cos_diff, 4) if not np.isnan(cos_diff) else None,
        "pca_variance_explained_pc1": round(float(pca_all.explained_variance_ratio_[0]), 4),
    }
    print(f"    cos(v_commit, PC1_all) = {cos_pc1_all:.4f}")
    print(f"    cos(v_commit, PC1_hard) = {cos_pc1_hard:.4f}")
    print(f"    cos(v_commit, easy-hard diff) = {cos_diff:.4f}")
    print(f"    PC1 variance explained = {pca_all.explained_variance_ratio_[0]:.4f}")

    # ── 3c: Project all 100 questions onto v_commit, correlate with CV ──
    print("\n  --- 3c: Projection onto v_commit vs CV ---")
    projections = []
    cvs = []
    difficulties = []
    proj_qids = []

    for qid in all_qids_ordered:
        proj = float(np.dot(qid_mean_hs[qid], v_commit_llama_norm))
        projections.append(proj)
        cvs.append(llama_data[qid]["cv"])
        difficulties.append(llama_data[qid]["difficulty"])
        proj_qids.append(qid)

    projections = np.array(projections)
    cvs = np.array(cvs)

    r_proj, p_proj = stats.pearsonr(projections, cvs)
    ci_lo, ci_hi = bootstrap_ci(projections, cvs)

    # Also compute for hard-only and easy-only
    hard_mask = np.array([d == "hard" for d in difficulties])
    easy_mask = np.array([d == "easy" for d in difficulties])

    r_hard, p_hard = stats.pearsonr(projections[hard_mask], cvs[hard_mask]) if sum(hard_mask) >= 5 else (np.nan, np.nan)
    r_easy, p_easy = stats.pearsonr(projections[easy_mask], cvs[easy_mask]) if sum(easy_mask) >= 5 else (np.nan, np.nan)

    # Compare to raw activation similarity signal
    raw_sims = []
    raw_cvs = []
    for qid in all_qids_ordered:
        vectors = get_hidden_states_at(llama_data, qid, step=STEP, layer=LLAMA_COMMIT_LAYER)
        if len(vectors) >= 2:
            raw_sims.append(pairwise_cosine_sim(vectors))
            raw_cvs.append(llama_data[qid]["cv"])
    r_raw, _ = stats.pearsonr(raw_sims, raw_cvs)

    results["3c_projection"] = {
        "r_all": round(r_proj, 4), "p_all": round(p_proj, 6),
        "ci": [round(ci_lo, 3), round(ci_hi, 3)],
        "r_hard": round(float(r_hard), 4) if not np.isnan(r_hard) else None,
        "p_hard": round(float(p_hard), 6) if not np.isnan(p_hard) else None,
        "r_easy": round(float(r_easy), 4) if not np.isnan(r_easy) else None,
        "p_easy": round(float(p_easy), 6) if not np.isnan(p_easy) else None,
        "r_raw_activation_sim": round(r_raw, 4),
        "n": len(projections),
    }
    # Store data for figure generation
    results["3c_figure_data"] = {
        "projections": [round(float(p), 6) for p in projections],
        "cvs": [round(float(c), 6) for c in cvs],
        "difficulties": difficulties,
        "qids": proj_qids,
    }
    print(f"    All: r={r_proj:.4f}, p={p_proj:.6f}, CI=[{ci_lo:.3f},{ci_hi:.3f}]")
    print(f"    Hard only: r={r_hard:.4f}, p={p_hard:.6f}" if not np.isnan(r_hard) else "    Hard only: insufficient data")
    print(f"    Easy only: r={r_easy:.4f}, p={p_easy:.6f}" if not np.isnan(r_easy) else "    Easy only: insufficient data")
    print(f"    Raw activation sim r: {r_raw:.4f}")

    # ── 3d: Qwen commitment vector at layer 64 + Procrustes alignment ──
    print("\n  --- 3d: Qwen commitment vector + Procrustes alignment ---")
    QWEN_COMMIT_LAYER = 64

    # Qwen: classify and compute v_commit
    qwen_category_hs = {"committed-correct": [], "uncommitted-wrong": []}
    qwen_qid_mean_hs = {}

    for qid, entry in qwen_data.items():
        vectors = get_hidden_states_at(qwen_data, qid, step=STEP, layer=QWEN_COMMIT_LAYER)
        if len(vectors) >= 2:
            mean_vec = np.mean(vectors, axis=0)
            qwen_qid_mean_hs[qid] = mean_vec
            cat = entry.get("category", "mixed")
            if cat in qwen_category_hs:
                qwen_category_hs[cat].append(mean_vec)

    n_cc_q = len(qwen_category_hs["committed-correct"])
    n_uw_q = len(qwen_category_hs["uncommitted-wrong"])
    print(f"    Qwen committed-correct: {n_cc_q}")
    print(f"    Qwen uncommitted-wrong: {n_uw_q}")

    if n_cc_q < 2 or n_uw_q < 2:
        print("    INSUFFICIENT DATA for Qwen commitment vector")
        results["3d_qwen"] = {"error": "insufficient data"}
    else:
        centroid_cc_q = np.mean(qwen_category_hs["committed-correct"], axis=0)
        centroid_uw_q = np.mean(qwen_category_hs["uncommitted-wrong"], axis=0)
        v_commit_qwen = centroid_cc_q - centroid_uw_q
        v_commit_qwen_norm = v_commit_qwen / np.linalg.norm(v_commit_qwen)

        # Qwen projection correlation
        qwen_projections = []
        qwen_cvs = []
        for qid in sorted(qwen_qid_mean_hs.keys()):
            proj = float(np.dot(qwen_qid_mean_hs[qid], v_commit_qwen_norm))
            qwen_projections.append(proj)
            qwen_cvs.append(qwen_data[qid]["cv"])

        r_qwen, p_qwen = stats.pearsonr(qwen_projections, qwen_cvs)
        print(f"    Qwen projection r={r_qwen:.4f}, p={p_qwen:.6f}")

        # Procrustes alignment for cross-model comparison
        # Find shared questions between Llama and Qwen
        shared_qids = sorted(set(qid_mean_hs.keys()) & set(qwen_qid_mean_hs.keys()))
        print(f"    Shared questions: {len(shared_qids)}")

        if len(shared_qids) >= 20:
            # PCA to 50D for both models using shared questions
            llama_shared = np.array([qid_mean_hs[qid] for qid in shared_qids])
            qwen_shared = np.array([qwen_qid_mean_hs[qid] for qid in shared_qids])

            n_components = min(50, len(shared_qids) - 1)
            pca_llama = PCA(n_components=n_components, random_state=SEED)
            pca_qwen = PCA(n_components=n_components, random_state=SEED)

            llama_pca = pca_llama.fit_transform(llama_shared)
            qwen_pca = pca_qwen.fit_transform(qwen_shared)

            # Procrustes alignment: align Qwen PCA space to Llama PCA space
            # scipy.spatial.procrustes returns (mtx1_standardized, mtx2_aligned, disparity)
            llama_aligned, qwen_aligned, disparity = procrustes(llama_pca, qwen_pca)

            # Project commitment vectors into PCA space
            v_commit_llama_pca = pca_llama.transform(v_commit_llama.reshape(1, -1))[0]
            v_commit_qwen_pca = pca_qwen.transform(v_commit_qwen.reshape(1, -1))[0]

            # Normalize PCA projections
            v_ll_pca_norm = v_commit_llama_pca / np.linalg.norm(v_commit_llama_pca)
            v_qw_pca_norm = v_commit_qwen_pca / np.linalg.norm(v_commit_qwen_pca)

            # Apply same Procrustes transform to Qwen commitment vector
            # Procrustes standardizes: center, scale, rotate. We need the rotation.
            # The procrustes function transforms mtx2 to best match mtx1.
            # To apply the same transform to v_commit_qwen_pca, we recompute manually.
            # Standardize both matrices (center + scale)
            llama_centered = llama_pca - llama_pca.mean(axis=0)
            qwen_centered = qwen_pca - qwen_pca.mean(axis=0)
            llama_scale = np.sqrt(np.sum(llama_centered**2))
            qwen_scale = np.sqrt(np.sum(qwen_centered**2))
            llama_std = llama_centered / llama_scale
            qwen_std = qwen_centered / qwen_scale

            # Find optimal rotation via SVD
            M = qwen_std.T @ llama_std
            U, S, Vt = np.linalg.svd(M)
            R = U @ Vt  # rotation matrix

            # Apply rotation to Qwen commitment vector (after centering + scaling)
            v_qw_centered = (v_commit_qwen_pca - qwen_pca.mean(axis=0)) / qwen_scale
            v_qw_rotated = v_qw_centered @ R

            # Llama commitment vector (after centering + scaling)
            v_ll_centered = (v_commit_llama_pca - llama_pca.mean(axis=0)) / llama_scale

            # Cosine similarity after Procrustes alignment
            cos_procrustes = float(cosine_similarity(
                v_ll_centered.reshape(1, -1), v_qw_rotated.reshape(1, -1)
            )[0, 0])

            # Also raw cosine (no alignment) for comparison
            cos_raw = float(cosine_similarity(
                v_ll_pca_norm.reshape(1, -1), v_qw_pca_norm.reshape(1, -1)
            )[0, 0])

            results["3d_qwen"] = {
                "layer": QWEN_COMMIT_LAYER,
                "n_committed_correct": n_cc_q,
                "n_uncommitted_wrong": n_uw_q,
                "v_commit_norm": round(float(np.linalg.norm(v_commit_qwen)), 2),
                "projection_r": round(r_qwen, 4),
                "projection_p": round(p_qwen, 6),
                "n_shared": len(shared_qids),
                "pca_components": n_components,
                "procrustes_disparity": round(float(disparity), 4),
                "cos_procrustes": round(cos_procrustes, 4),
                "cos_raw_pca": round(cos_raw, 4),
            }
            print(f"    Procrustes disparity: {disparity:.4f}")
            print(f"    cos(v_commit_llama, v_commit_qwen) after Procrustes: {cos_procrustes:.4f}")
            print(f"    cos(v_commit_llama, v_commit_qwen) raw PCA: {cos_raw:.4f}")
        else:
            results["3d_qwen"] = {
                "layer": QWEN_COMMIT_LAYER,
                "n_committed_correct": n_cc_q,
                "n_uncommitted_wrong": n_uw_q,
                "projection_r": round(r_qwen, 4),
                "error": "insufficient shared questions for Procrustes",
            }

    return results


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("PAPER 3 v3 ANALYSES")
    print("=" * 70)

    # Load data
    llama_data = load_all_llama_data()
    qwen_data = load_qwen_data()

    all_results = {}

    # Task 1
    all_results["task1"] = task1_hard_question_signal_search(llama_data)

    # Task 2
    all_results["task2"] = task2_early_run_monitor(llama_data)

    # Task 3
    all_results["task3"] = task3_commitment_geometry(llama_data, qwen_data)

    # Save results
    output_file = OUTPUT_DIR / "paper3_v3_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n{'=' * 70}")
    print(f"Results saved to {output_file}")
    print(f"{'=' * 70}")

    # Print summary
    print("\n=== SUMMARY ===")
    t1 = all_results["task1"]
    print(f"\nTask 1 (Hard-question signal):")
    print(f"  Any significant signal? {t1['any_significant']}")
    if t1["significant_signals"]:
        for sig in t1["significant_signals"]:
            print(f"    {sig['metric']}: r={sig['r']:.4f}, perm_p={sig['perm_p']:.4f}")

    t2 = all_results["task2"]
    print(f"\nTask 2 (Early-run monitor):")
    k3_auroc = t2.get("2a_k_auroc", {}).get(3, {}).get("mean", "?")
    print(f"  k=3 AUROC: {k3_auroc}")
    print(f"  k=3 PR operating point: precision={t2['2b_pr_k3']['precision_at_op']}, "
          f"recall={t2['2b_pr_k3']['recall_at_op']}")
    print(f"  Early-exit mean accuracy: {t2['2c_early_exit']['mean_test_acc']}")

    t3 = all_results["task3"]
    print(f"\nTask 3 (Commitment geometry):")
    if "3c_projection" in t3:
        print(f"  Projection r (all): {t3['3c_projection']['r_all']}")
        print(f"  Projection r (hard): {t3['3c_projection'].get('r_hard', '?')}")
    if "3d_qwen" in t3 and "cos_procrustes" in t3["3d_qwen"]:
        print(f"  Cross-model cos (Procrustes): {t3['3d_qwen']['cos_procrustes']}")


if __name__ == "__main__":
    main()
