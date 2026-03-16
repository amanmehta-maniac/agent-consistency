"""
Paper 3 — Task 6: Runtime Monitor Evaluation

Evaluates step-4 hidden states as a runtime predictor of behavioral
inconsistency using leave-one-out cross-validation (LOOCV).

Features tested (each standalone):
  A. Activation similarity at step 4, layer 40 (single scalar)
  B. Activation similarity at step 4, averaged across layers 32-80
  C. Per-layer profile vector (layers 32-80, logistic regression L2)
  D. Trajectory-so-far similarity (steps 1-4 concatenated)
  E. PCA of hidden states at step 4, layer 40 -> first 10 components -> logistic regression

Baselines:
  1. Random classifier (AUC = 0.5)
  2. Question length (word count)
  3. Number of context documents
  4. Mean thought length at step 3

Uses all 100 Llama questions (3 directories) and Qwen 100q data.

Run from hotpotqa/ directory:
    ../.venv/bin/python3 runtime_monitor_eval.py
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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
QWEN_Q_FILE = Path("qwen_100q_full.json")

OUTPUT_DIR = Path("analysis_results/runtime_monitor")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = Path("../paper3_prep/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

LAYERS = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80]
LAYERS_MID_HIGH = [32, 40, 48, 56, 64, 72, 80]  # layers 32-80
STEPS = list(range(1, 6))


# ══════════════════════════════════════════════════════════
#  Data loading (follows paper3_100q_analysis.py pattern)
# ══════════════════════════════════════════════════════════

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
                "trajectory": traj,
            })
        if runs_data:
            results[qid] = {"question": questions[qid], "runs": runs_data, "difficulty": "hard"}
    print(f"  Loaded {len(results)} pilot questions")
    return results


def load_easier_data():
    print("Loading 20 easy questions (easier, json format)...")
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
                "trajectory": run,
            })
        if runs_data:
            results[qid] = {"question": questions[qid], "runs": runs_data, "difficulty": "easy"}
    print(f"  Loaded {len(results)} easier questions")
    return results


def load_new60_data():
    print("Loading 60 new questions (experiment_60q, npy format)...")
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
                "trajectory": traj,
            })
        if runs_data:
            diff = questions[qid].get("difficulty", "unknown")
            results[qid] = {"question": questions[qid], "runs": runs_data, "difficulty": diff}
    print(f"  Loaded {len(results)} new-60 questions")
    return results


def load_qwen_data():
    print("Loading Qwen 2.5 72B data (100 questions, npy format)...")
    with open(QWEN_Q_FILE) as f:
        questions = {q["id"]: q for q in json.load(f)}
    results = {}
    for qdir in sorted(QWEN_DIR.iterdir()):
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
                "trajectory": traj,
            })
        if runs_data:
            diff = questions[qid].get("difficulty", "unknown")
            results[qid] = {"question": questions[qid], "runs": runs_data, "difficulty": diff}
    print(f"  Loaded {len(results)} Qwen questions")
    return results


# ══════════════════════════════════════════════════════════
#  Feature extraction
# ══════════════════════════════════════════════════════════

def compute_question_features(qid, entry):
    """Extract all features for one question."""
    runs = entry["runs"]
    question = entry["question"]

    # ── Target: CV ──
    step_counts = [r["step_count"] for r in runs]
    cv = np.std(step_counts) / np.mean(step_counts) if np.mean(step_counts) > 0 else 0

    # ── Correct rate ──
    correct_count = sum(1 for r in runs if r.get("correct"))
    correct_rate = correct_count / len(runs) if runs else 0

    # ── Feature A: Activation similarity at step 4, layer 40 ──
    feat_a = np.nan
    vectors_s4_l40 = []
    for run in runs:
        hs = run["hidden_states"].get(4)
        if hs is not None and len(hs) > 40:
            vectors_s4_l40.append(hs[40])
    if len(vectors_s4_l40) >= 2:
        feat_a = compute_pairwise_cosine_similarity(vectors_s4_l40)

    # ── Feature B: Activation similarity at step 4, averaged across layers 32-80 ──
    feat_b = np.nan
    layer_sims = []
    for layer in LAYERS_MID_HIGH:
        vectors = []
        for run in runs:
            hs = run["hidden_states"].get(4)
            if hs is not None and len(hs) > layer:
                vectors.append(hs[layer])
        if len(vectors) >= 2:
            layer_sims.append(compute_pairwise_cosine_similarity(vectors))
    if layer_sims:
        feat_b = np.mean(layer_sims)

    # ── Feature C: Per-layer profile vector (one sim per layer, layers 32-80) ──
    feat_c = []
    for layer in LAYERS_MID_HIGH:
        vectors = []
        for run in runs:
            hs = run["hidden_states"].get(4)
            if hs is not None and len(hs) > layer:
                vectors.append(hs[layer])
        if len(vectors) >= 2:
            feat_c.append(compute_pairwise_cosine_similarity(vectors))
        else:
            feat_c.append(np.nan)
    feat_c = np.array(feat_c)

    # ── Feature D: Trajectory-so-far similarity (steps 1-4 concatenated) ──
    feat_d = np.nan
    concat_vectors = []
    for run in runs:
        parts = []
        for step in range(1, 5):
            hs = run["hidden_states"].get(step)
            if hs is not None and len(hs) > 40:
                parts.append(hs[40])
        if len(parts) == 4:
            concat_vectors.append(np.concatenate(parts))
    if len(concat_vectors) >= 2:
        feat_d = compute_pairwise_cosine_similarity(concat_vectors)

    # ── Feature E: PCA of hidden states at step 4, layer 40 ──
    # Store raw vectors; PCA computed globally after all questions loaded
    feat_e_raw = vectors_s4_l40 if len(vectors_s4_l40) >= 2 else None

    # ── Baseline 1: Question length (word count) ──
    question_text = question.get("question", "")
    baseline_qlen = len(question_text.split())

    # ── Baseline 2: Number of context documents ──
    context = question.get("context", {})
    if isinstance(context, dict):
        baseline_ndocs = len(context)
    elif isinstance(context, list):
        baseline_ndocs = len(context)
    else:
        baseline_ndocs = 0

    # ── Baseline 3: Mean thought length at step 3 ──
    # This requires trajectory text; approximate from hidden state step counts
    # For pilot/new60 format, trajectory is stored differently than easier format
    baseline_thought_s3 = np.nan
    # We use step_count as a proxy — questions with more steps at step 3 have
    # longer thoughts. A better approach would parse trajectory text, but
    # the formats differ across the 3 directories. Use hidden state presence
    # at step 3 as a proxy for "agent reached step 3 with substantial thought."
    n_with_step3 = sum(1 for r in runs if 3 in r["hidden_states"])
    baseline_thought_s3 = n_with_step3 / len(runs) if runs else 0

    return {
        "qid": qid,
        "difficulty": entry["difficulty"],
        "cv": cv,
        "correct_rate": correct_rate,
        "n_runs": len(runs),
        "feat_a": feat_a,
        "feat_b": feat_b,
        "feat_c": feat_c,
        "feat_d": feat_d,
        "feat_e_raw": feat_e_raw,
        "baseline_qlen": baseline_qlen,
        "baseline_ndocs": baseline_ndocs,
        "baseline_thought_s3": baseline_thought_s3,
    }


# ══════════════════════════════════════════════════════════
#  LOOCV evaluation
# ══════════════════════════════════════════════════════════

def loocv_single_feature(features, labels, feature_name=""):
    """LOOCV for a single scalar feature using logistic regression."""
    n = len(features)
    scores = np.zeros(n)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_train = features[mask].reshape(-1, 1)
        y_train = labels[mask]
        X_test = features[i:i+1].reshape(-1, 1)

        if len(set(y_train)) < 2:
            scores[i] = 0.5
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000,
                                 random_state=42)
        clf.fit(X_train_s, y_train)
        scores[i] = clf.predict_proba(X_test_s)[0, 1]

    return scores


def loocv_logistic(features_2d, labels, feature_name=""):
    """LOOCV for multi-dimensional features using logistic regression."""
    n = len(labels)
    scores = np.zeros(n)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_train = features_2d[mask]
        y_train = labels[mask]
        X_test = features_2d[i:i+1]

        if len(set(y_train)) < 2:
            scores[i] = 0.5
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=1000,
                                 random_state=42)
        clf.fit(X_train_s, y_train)
        scores[i] = clf.predict_proba(X_test_s)[0, 1]

    return scores


def loocv_pca_logistic(raw_vectors, labels, n_components=10, feature_name=""):
    """LOOCV with PCA + logistic regression, PCA refit inside each fold."""
    n = len(labels)
    scores = np.zeros(n)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_train = raw_vectors[mask]
        y_train = labels[mask]
        X_test = raw_vectors[i:i+1]

        if len(set(y_train)) < 2:
            scores[i] = 0.5
            continue

        # Fit PCA on training fold only
        nc = min(n_components, X_train.shape[0] - 1, X_train.shape[1])
        pca = PCA(n_components=nc, random_state=42)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_pca)
        X_test_s = scaler.transform(X_test_pca)

        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000,
                                 random_state=42)
        clf.fit(X_train_s, y_train)
        scores[i] = clf.predict_proba(X_test_s)[0, 1]

    return scores


def compute_metrics(labels, scores):
    """Compute AUROC, precision/recall at operating points, best F1."""
    results = {}

    # AUROC
    try:
        auroc = roc_auc_score(labels, scores)
    except ValueError:
        auroc = np.nan
    results["auroc"] = float(auroc)

    # Bootstrap 95% CI for AUROC
    rng = np.random.default_rng(42)
    boot_aucs = []
    for _ in range(1000):
        idx = rng.choice(len(labels), size=len(labels), replace=True)
        if len(set(labels[idx])) < 2:
            continue
        try:
            boot_aucs.append(roc_auc_score(labels[idx], scores[idx]))
        except ValueError:
            continue
    if boot_aucs:
        results["auroc_ci_low"] = float(np.percentile(boot_aucs, 2.5))
        results["auroc_ci_high"] = float(np.percentile(boot_aucs, 97.5))
    else:
        results["auroc_ci_low"] = np.nan
        results["auroc_ci_high"] = np.nan

    # Precision at recall=0.7 and Recall at precision=0.7
    try:
        precision_arr, recall_arr, thresholds_pr = precision_recall_curve(labels, scores)
        # Precision at recall >= 0.7
        valid = recall_arr >= 0.7
        if valid.any():
            results["precision_at_recall_0.7"] = float(precision_arr[valid].max())
        else:
            results["precision_at_recall_0.7"] = np.nan

        # Recall at precision >= 0.7
        valid_p = precision_arr >= 0.7
        if valid_p.any():
            results["recall_at_precision_0.7"] = float(recall_arr[valid_p].max())
        else:
            results["recall_at_precision_0.7"] = np.nan
    except Exception:
        results["precision_at_recall_0.7"] = np.nan
        results["recall_at_precision_0.7"] = np.nan

    # Best F1
    try:
        fpr, tpr, thresholds_roc = roc_curve(labels, scores)
        best_f1 = 0
        for t in thresholds_roc:
            preds = (scores >= t).astype(int)
            f1 = f1_score(labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
        results["best_f1"] = float(best_f1)
    except Exception:
        results["best_f1"] = np.nan

    return results


# ══════════════════════════════════════════════════════════
#  Main evaluation pipeline
# ══════════════════════════════════════════════════════════

def run_evaluation(all_features, model_name="Llama"):
    """Run full evaluation pipeline on a set of question features."""
    print(f"\n{'='*70}")
    print(f"RUNTIME MONITOR EVALUATION — {model_name} (n={len(all_features)})")
    print(f"{'='*70}")

    # ── Define labels ──
    cvs = np.array([f["cv"] for f in all_features])

    # Two labeling schemes:
    # Scheme 1: extreme quintiles (top 20% = inconsistent, bottom 20% = consistent)
    p20 = np.percentile(cvs, 20)
    p80 = np.percentile(cvs, 80)
    # Scheme 2: median split
    median_cv = np.median(cvs)

    results_by_scheme = {}

    for scheme_name, label_fn, desc in [
        ("quintile", lambda cv: 1 if cv >= p80 else (0 if cv <= p20 else -1),
         f"Quintile (consistent: CV<={p20:.3f}, inconsistent: CV>={p80:.3f})"),
        ("median", lambda cv: 1 if cv >= median_cv else 0,
         f"Median split (threshold={median_cv:.3f})"),
    ]:
        print(f"\n  --- Labeling: {desc} ---")

        labels_all = np.array([label_fn(f["cv"]) for f in all_features])
        valid_mask = labels_all >= 0  # -1 = excluded (middle 60% in quintile scheme)
        labels = labels_all[valid_mask]
        feats_subset = [f for f, v in zip(all_features, valid_mask) if v]

        n_pos = (labels == 1).sum()
        n_neg = (labels == 0).sum()
        print(f"  n={len(labels)}, inconsistent(1)={n_pos}, consistent(0)={n_neg}")

        if n_pos < 3 or n_neg < 3:
            print(f"  SKIPPED: insufficient samples per class")
            continue

        feature_results = {}

        # ── Feature A: Sim at step 4, layer 40 ──
        feat_a_vals = np.array([f["feat_a"] for f in feats_subset])
        valid_a = ~np.isnan(feat_a_vals)
        if valid_a.sum() >= 10 and len(set(labels[valid_a])) >= 2:
            scores_a = loocv_single_feature(feat_a_vals[valid_a], labels[valid_a], "A")
            metrics_a = compute_metrics(labels[valid_a], scores_a)
            feature_results["A_sim_s4_l40"] = metrics_a
            print(f"  A (Sim s4 l40):     AUROC={metrics_a['auroc']:.3f} "
                  f"[{metrics_a['auroc_ci_low']:.3f}, {metrics_a['auroc_ci_high']:.3f}] "
                  f"F1={metrics_a['best_f1']:.3f}")

        # ── Feature B: Sim at step 4, avg layers 32-80 ──
        feat_b_vals = np.array([f["feat_b"] for f in feats_subset])
        valid_b = ~np.isnan(feat_b_vals)
        if valid_b.sum() >= 10 and len(set(labels[valid_b])) >= 2:
            scores_b = loocv_single_feature(feat_b_vals[valid_b], labels[valid_b], "B")
            metrics_b = compute_metrics(labels[valid_b], scores_b)
            feature_results["B_sim_s4_avg_layers"] = metrics_b
            print(f"  B (Sim s4 avg L):   AUROC={metrics_b['auroc']:.3f} "
                  f"[{metrics_b['auroc_ci_low']:.3f}, {metrics_b['auroc_ci_high']:.3f}] "
                  f"F1={metrics_b['best_f1']:.3f}")

        # ── Feature C: Per-layer profile (7-dim logistic regression) ──
        feat_c_matrix = np.array([f["feat_c"] for f in feats_subset])
        valid_c = ~np.any(np.isnan(feat_c_matrix), axis=1)
        if valid_c.sum() >= 10 and len(set(labels[valid_c])) >= 2:
            scores_c = loocv_logistic(feat_c_matrix[valid_c], labels[valid_c], "C")
            metrics_c = compute_metrics(labels[valid_c], scores_c)
            feature_results["C_layer_profile"] = metrics_c
            print(f"  C (Layer profile):  AUROC={metrics_c['auroc']:.3f} "
                  f"[{metrics_c['auroc_ci_low']:.3f}, {metrics_c['auroc_ci_high']:.3f}] "
                  f"F1={metrics_c['best_f1']:.3f}")

        # ── Feature D: Trajectory-so-far similarity ──
        feat_d_vals = np.array([f["feat_d"] for f in feats_subset])
        valid_d = ~np.isnan(feat_d_vals)
        if valid_d.sum() >= 10 and len(set(labels[valid_d])) >= 2:
            scores_d = loocv_single_feature(feat_d_vals[valid_d], labels[valid_d], "D")
            metrics_d = compute_metrics(labels[valid_d], scores_d)
            feature_results["D_traj_so_far"] = metrics_d
            print(f"  D (Traj-so-far):    AUROC={metrics_d['auroc']:.3f} "
                  f"[{metrics_d['auroc_ci_low']:.3f}, {metrics_d['auroc_ci_high']:.3f}] "
                  f"F1={metrics_d['best_f1']:.3f}")

        # ── Feature E: PCA of hidden states at step 4, layer 40 ──
        # Compute global PCA over all question-level mean hidden states
        mean_hs = []
        mean_hs_mask = []
        for f in feats_subset:
            if f["feat_e_raw"] is not None and len(f["feat_e_raw"]) >= 2:
                mean_hs.append(np.mean(f["feat_e_raw"], axis=0))
                mean_hs_mask.append(True)
            else:
                mean_hs.append(np.zeros(8192))
                mean_hs_mask.append(False)
        mean_hs = np.array(mean_hs)
        mean_hs_mask = np.array(mean_hs_mask)

        if mean_hs_mask.sum() >= 15 and len(set(labels[mean_hs_mask])) >= 2:
            # PCA refit inside each LOOCV fold to avoid data leakage
            scores_e = loocv_pca_logistic(mean_hs[mean_hs_mask], labels[mean_hs_mask],
                                          n_components=10, feature_name="E")
            metrics_e = compute_metrics(labels[mean_hs_mask], scores_e)
            feature_results["E_pca_s4_l40"] = metrics_e
            print(f"  E (PCA s4 l40):     AUROC={metrics_e['auroc']:.3f} "
                  f"[{metrics_e['auroc_ci_low']:.3f}, {metrics_e['auroc_ci_high']:.3f}] "
                  f"F1={metrics_e['best_f1']:.3f}")

        # ── Baselines ──
        print(f"\n  --- Baselines ---")

        # Baseline 1: Random
        rng = np.random.default_rng(42)
        random_scores = rng.random(len(labels))
        metrics_rand = compute_metrics(labels, random_scores)
        feature_results["baseline_random"] = metrics_rand
        print(f"  Random:             AUROC={metrics_rand['auroc']:.3f} "
              f"[{metrics_rand['auroc_ci_low']:.3f}, {metrics_rand['auroc_ci_high']:.3f}]")

        # Baseline 2: Question length
        qlen_vals = np.array([f["baseline_qlen"] for f in feats_subset], dtype=float)
        if len(set(labels)) >= 2:
            scores_qlen = loocv_single_feature(qlen_vals, labels, "qlen")
            metrics_qlen = compute_metrics(labels, scores_qlen)
            feature_results["baseline_question_length"] = metrics_qlen
            print(f"  Question length:    AUROC={metrics_qlen['auroc']:.3f} "
                  f"[{metrics_qlen['auroc_ci_low']:.3f}, {metrics_qlen['auroc_ci_high']:.3f}]")

        # Baseline 3: Context docs
        ndocs_vals = np.array([f["baseline_ndocs"] for f in feats_subset], dtype=float)
        if len(set(labels)) >= 2:
            scores_ndocs = loocv_single_feature(ndocs_vals, labels, "ndocs")
            metrics_ndocs = compute_metrics(labels, scores_ndocs)
            feature_results["baseline_context_docs"] = metrics_ndocs
            print(f"  Context docs:       AUROC={metrics_ndocs['auroc']:.3f} "
                  f"[{metrics_ndocs['auroc_ci_low']:.3f}, {metrics_ndocs['auroc_ci_high']:.3f}]")

        # Baseline 4: Thought length at step 3
        thought_vals = np.array([f["baseline_thought_s3"] for f in feats_subset], dtype=float)
        valid_t = ~np.isnan(thought_vals)
        if valid_t.sum() >= 10 and len(set(labels[valid_t])) >= 2:
            scores_thought = loocv_single_feature(thought_vals[valid_t], labels[valid_t], "thought")
            metrics_thought = compute_metrics(labels[valid_t], scores_thought)
            feature_results["baseline_thought_step3"] = metrics_thought
            print(f"  Thought len s3:     AUROC={metrics_thought['auroc']:.3f} "
                  f"[{metrics_thought['auroc_ci_low']:.3f}, {metrics_thought['auroc_ci_high']:.3f}]")

        results_by_scheme[scheme_name] = {
            "n": int(len(labels)),
            "n_inconsistent": int(n_pos),
            "n_consistent": int(n_neg),
            "description": desc,
            "features": feature_results,
        }

        # Store ROC data for plotting
        results_by_scheme[scheme_name]["roc_data"] = {}
        for feat_key in feature_results:
            if feat_key.startswith("baseline_random"):
                continue
            # Re-compute scores for ROC curve
            # We already have the scores from LOOCV above; store them
            # (they're in local scope, so we need to regenerate or store)
        # We'll store during the feature computation — see below

    return results_by_scheme


def generate_roc_figure(all_features, labels_fn, scheme_name, model_name, suffix=""):
    """Generate ROC curves for all features."""
    cvs = np.array([f["cv"] for f in all_features])
    labels_all = np.array([labels_fn(f["cv"]) for f in all_features])
    valid_mask = labels_all >= 0
    labels = labels_all[valid_mask]
    feats_subset = [f for f, v in zip(all_features, valid_mask) if v]

    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()
    if n_pos < 3 or n_neg < 3:
        return

    fig, ax = plt.subplots(figsize=(7, 6))

    curve_data = {}

    # Feature A
    feat_a_vals = np.array([f["feat_a"] for f in feats_subset])
    valid_a = ~np.isnan(feat_a_vals)
    if valid_a.sum() >= 10 and len(set(labels[valid_a])) >= 2:
        scores = loocv_single_feature(feat_a_vals[valid_a], labels[valid_a])
        fpr, tpr, _ = roc_curve(labels[valid_a], scores)
        auroc = roc_auc_score(labels[valid_a], scores)
        ax.plot(fpr, tpr, label=f"Sim(s4,L40) AUC={auroc:.2f}", linewidth=2, color="#1f77b4")
        curve_data["A"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auroc": float(auroc)}

    # Feature B
    feat_b_vals = np.array([f["feat_b"] for f in feats_subset])
    valid_b = ~np.isnan(feat_b_vals)
    if valid_b.sum() >= 10 and len(set(labels[valid_b])) >= 2:
        scores = loocv_single_feature(feat_b_vals[valid_b], labels[valid_b])
        fpr, tpr, _ = roc_curve(labels[valid_b], scores)
        auroc = roc_auc_score(labels[valid_b], scores)
        ax.plot(fpr, tpr, label=f"Sim(s4,avg L32-80) AUC={auroc:.2f}", linewidth=2, color="#ff7f0e")
        curve_data["B"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auroc": float(auroc)}

    # Feature C
    feat_c_matrix = np.array([f["feat_c"] for f in feats_subset])
    valid_c = ~np.any(np.isnan(feat_c_matrix), axis=1)
    if valid_c.sum() >= 10 and len(set(labels[valid_c])) >= 2:
        scores = loocv_logistic(feat_c_matrix[valid_c], labels[valid_c])
        fpr, tpr, _ = roc_curve(labels[valid_c], scores)
        auroc = roc_auc_score(labels[valid_c], scores)
        ax.plot(fpr, tpr, label=f"Layer profile (LR) AUC={auroc:.2f}", linewidth=2, color="#2ca02c")
        curve_data["C"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auroc": float(auroc)}

    # Feature D
    feat_d_vals = np.array([f["feat_d"] for f in feats_subset])
    valid_d = ~np.isnan(feat_d_vals)
    if valid_d.sum() >= 10 and len(set(labels[valid_d])) >= 2:
        scores = loocv_single_feature(feat_d_vals[valid_d], labels[valid_d])
        fpr, tpr, _ = roc_curve(labels[valid_d], scores)
        auroc = roc_auc_score(labels[valid_d], scores)
        ax.plot(fpr, tpr, label=f"Traj s1-4 concat AUC={auroc:.2f}", linewidth=2,
                color="#9467bd", linestyle="--")
        curve_data["D"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auroc": float(auroc)}

    # Baseline: question length
    qlen_vals = np.array([f["baseline_qlen"] for f in feats_subset], dtype=float)
    scores = loocv_single_feature(qlen_vals, labels)
    fpr, tpr, _ = roc_curve(labels, scores)
    auroc = roc_auc_score(labels, scores)
    ax.plot(fpr, tpr, label=f"Question length AUC={auroc:.2f}", linewidth=1.5,
            color="#7f7f7f", linestyle=":")
    curve_data["qlen"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auroc": float(auroc)}

    # Random baseline
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label="Random (AUC=0.50)", alpha=0.5)

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"Runtime Monitor ROC — {model_name}\n({scheme_name} labeling, "
                 f"n={len(labels)}: {n_neg} consistent, {n_pos} inconsistent)", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    plt.tight_layout()

    fname = f"runtime_monitor_roc_{model_name.lower()}_{scheme_name}{suffix}"
    plt.savefig(OUTPUT_DIR / f"{fname}.png", dpi=150, bbox_inches="tight")
    plt.savefig(FIG_DIR / f"{fname}.png", dpi=150, bbox_inches="tight")
    plt.savefig(FIG_DIR / f"{fname}.pdf", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {fname}.png/.pdf")

    return curve_data


# ══════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("PAPER 3 — TASK 6: RUNTIME MONITOR EVALUATION")
    print("=" * 70)

    # ── Load Llama data (all 100 questions from 3 directories) ──
    pilot = load_pilot_data()
    easier = load_easier_data()
    new60 = load_new60_data()

    llama_data = {}
    llama_data.update(pilot)
    llama_data.update(easier)
    llama_data.update(new60)
    print(f"\nTotal Llama questions: {len(llama_data)}")

    # ── Extract features for Llama ──
    print("\nExtracting features for Llama...")
    llama_features = []
    for qid, entry in sorted(llama_data.items()):
        feat = compute_question_features(qid, entry)
        llama_features.append(feat)
    print(f"  Extracted features for {len(llama_features)} questions")

    # ── Run Llama evaluation ──
    llama_results = run_evaluation(llama_features, model_name="Llama-3.1-70B")

    # ── Generate ROC figures for Llama ──
    cvs_llama = np.array([f["cv"] for f in llama_features])
    p20_l = np.percentile(cvs_llama, 20)
    p80_l = np.percentile(cvs_llama, 80)
    median_l = np.median(cvs_llama)

    print("\nGenerating ROC figures for Llama...")
    roc_quintile_llama = generate_roc_figure(
        llama_features,
        lambda cv: 1 if cv >= p80_l else (0 if cv <= p20_l else -1),
        "quintile", "Llama")
    roc_median_llama = generate_roc_figure(
        llama_features,
        lambda cv: 1 if cv >= median_l else 0,
        "median", "Llama")

    # ── Load Qwen data ──
    qwen_data = load_qwen_data()

    if qwen_data:
        print("\nExtracting features for Qwen...")
        qwen_features = []
        for qid, entry in sorted(qwen_data.items()):
            feat = compute_question_features(qid, entry)
            qwen_features.append(feat)
        print(f"  Extracted features for {len(qwen_features)} questions")

        # ── Run Qwen evaluation ──
        qwen_results = run_evaluation(qwen_features, model_name="Qwen-2.5-72B")

        # ── Generate ROC figures for Qwen ──
        cvs_qwen = np.array([f["cv"] for f in qwen_features])
        p20_q = np.percentile(cvs_qwen, 20)
        p80_q = np.percentile(cvs_qwen, 80)
        median_q = np.median(cvs_qwen)

        print("\nGenerating ROC figures for Qwen...")
        roc_quintile_qwen = generate_roc_figure(
            qwen_features,
            lambda cv: 1 if cv >= p80_q else (0 if cv <= p20_q else -1),
            "quintile", "Qwen")
        roc_median_qwen = generate_roc_figure(
            qwen_features,
            lambda cv: 1 if cv >= median_q else 0,
            "median", "Qwen")
    else:
        qwen_results = None
        print("\nNo Qwen data found, skipping Qwen evaluation.")

    # ── Print summary table ──
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)

    for model_name, results in [("Llama-3.1-70B", llama_results), ("Qwen-2.5-72B", qwen_results)]:
        if results is None:
            continue
        for scheme in ["quintile", "median"]:
            if scheme not in results:
                continue
            r = results[scheme]
            print(f"\n  {model_name} — {scheme} (n={r['n']}, "
                  f"consistent={r['n_consistent']}, inconsistent={r['n_inconsistent']})")
            print(f"  {'Feature':<30} {'AUROC':>8} {'95% CI':>20} {'P@R=0.7':>8} {'R@P=0.7':>8} {'F1':>6}")
            print(f"  {'-'*80}")
            for feat_name, feat_r in r["features"].items():
                auroc = feat_r.get("auroc", np.nan)
                ci_lo = feat_r.get("auroc_ci_low", np.nan)
                ci_hi = feat_r.get("auroc_ci_high", np.nan)
                pr07 = feat_r.get("precision_at_recall_0.7", np.nan)
                rp07 = feat_r.get("recall_at_precision_0.7", np.nan)
                f1 = feat_r.get("best_f1", np.nan)
                ci_str = f"[{ci_lo:.3f}, {ci_hi:.3f}]" if not (np.isnan(ci_lo) or np.isnan(ci_hi)) else "n/a"
                print(f"  {feat_name:<30} {auroc:>8.3f} {ci_str:>20} "
                      f"{pr07:>8.3f} {rp07:>8.3f} {f1:>6.3f}")

    # ── Save results ──
    output = {
        "llama": llama_results,
        "qwen": qwen_results,
        "n_llama": len(llama_features),
        "n_qwen": len(qwen_features) if qwen_data else 0,
    }

    # Convert any numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

    with open(OUTPUT_DIR / "runtime_monitor_results.json", "w") as f:
        json.dump(output, f, indent=2, default=convert)
    print(f"\nSaved: {OUTPUT_DIR}/runtime_monitor_results.json")

    print("\n" + "=" * 70)
    print("TASK 6 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
