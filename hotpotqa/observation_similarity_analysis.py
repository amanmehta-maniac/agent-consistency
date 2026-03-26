"""
observation_similarity_analysis.py — Task 1: Observation Similarity Confound

Measures observation similarity across runs and shows that activation similarity
predicts consistency ABOVE AND BEYOND observation overlap.

Computes:
1. Jaccard similarity of retrieved document sets (steps 1-3)
2. TF-IDF cosine similarity of concatenated observation text (steps 1-3)
3. Search query exact-match overlap (steps 1-3)
4. Partial correlations: activation_sim → CV, controlling for obs_sim + accuracy + difficulty
5. Multiple regression: CV ~ activation_sim + obs_sim + accuracy
6. VIF between activation_sim and obs_sim
7. "Divergent-observation, convergent-representation" question identification

Run from hotpotqa/ directory:
    ../.venv/bin/python3 observation_similarity_analysis.py
"""

import json
import re
import numpy as np
from pathlib import Path
from itertools import combinations
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

BASE = Path(__file__).parent

PILOT_DIR = BASE / "pilot_hidden_states_70b"
NEW60_DIR = BASE / "experiment_60q_results"
EASIER_DIR = BASE / "results_easier"

PILOT_Q_FILE = BASE / "pilot_questions.json"
EASIER_Q_FILE = BASE / "easier_questions_selection.json"
NEW60_Q_FILE = BASE / "new_60_questions.json"

OUTPUT_DIR = BASE / "analysis_results" / "v3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LLAMA_LAYERS = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80]
SEED = 42
ANALYSIS_STEP = 4  # hidden states analyzed at step 4
OBS_STEPS = [1, 2, 3]  # observations gathered before step 4

# ═══════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def compute_cv(step_counts):
    if len(step_counts) < 2 or np.mean(step_counts) == 0:
        return 0.0
    return float(np.std(step_counts) / np.mean(step_counts))


def pairwise_cosine_sim(vectors):
    """Mean pairwise cosine similarity (upper triangle)."""
    if len(vectors) < 2:
        return np.nan
    arr = np.array(vectors)
    sim_matrix = cosine_similarity(arr)
    n = len(vectors)
    upper_tri = sim_matrix[np.triu_indices(n, k=1)]
    return float(np.mean(upper_tri))


def extract_yes_no(text):
    if not text:
        return None
    text = str(text).lower().strip()
    if "yes" in text and "no" not in text:
        return "yes"
    if "no" in text:
        return "no"
    return None


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


def parse_search_titles(observation):
    """Extract document titles from a Search observation.
    Format: 'Found N relevant document(s): Title1, Title2, ...'
    """
    if not observation:
        return set()
    m = re.match(r"Found \d+ relevant document\(s\):\s*(.*)", observation)
    if m:
        titles = [t.strip() for t in m.group(1).split(",") if t.strip()]
        return set(titles)
    return set()


def extract_search_query(action_input):
    """Extract query string from action_input, handling both formats."""
    if not action_input:
        return ""
    # Format 1: {"query": "..."}
    if "query" in action_input:
        return str(action_input["query"]).strip()
    # Format 2: {"value": "{\"query\": \"...\"}"}  (experiment_60q)
    if "value" in action_input:
        val = str(action_input["value"])
        # Try to parse as JSON
        try:
            parsed = json.loads(val)
            if isinstance(parsed, dict) and "query" in parsed:
                return str(parsed["query"]).strip()
        except (json.JSONDecodeError, TypeError):
            pass
        # Fallback: extract with regex
        m = re.search(r'"query":\s*"([^"]*)"', val)
        if m:
            return m.group(1).strip()
        return val.strip()
    # Format 3: {"title": "..."} (Retrieve) or {"answer": "..."} (Finish)
    if "title" in action_input:
        return str(action_input["title"]).strip()
    return ""


def jaccard(set_a, set_b):
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0  # both empty = identical
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING — trajectories + hidden states + observations
# ═══════════════════════════════════════════════════════════════════════

def load_npy_question_with_obs(qdir):
    """Load a question directory: hidden states + trajectory observations."""
    runs = []
    for run_dir in sorted(qdir.glob("run_*")):
        meta_file = run_dir / "metadata.json"
        traj_file = run_dir / "trajectory.json"
        if not meta_file.exists():
            continue
        with open(meta_file) as f:
            meta = json.load(f)
        
        # Load trajectory for observations
        steps_data = []
        if traj_file.exists():
            with open(traj_file) as f:
                traj = json.load(f)
            steps_data = traj.get("steps", [])
        
        # Load hidden states
        hidden_states = {}
        for hs_file in run_dir.glob("hidden_states_step_*.npy"):
            step_num = int(hs_file.stem.split("_")[-1])
            hs = np.load(hs_file)
            hidden_states[step_num] = hs
        
        # Extract observations and queries for steps 1-3
        obs_texts = {}  # step -> observation text
        search_queries = {}  # step -> query string
        doc_titles = {}  # step -> set of titles
        for step in steps_data:
            sn = step.get("step_number", 0)
            if sn in OBS_STEPS:
                obs_texts[sn] = step.get("observation", "")
                action = step.get("action", "")
                ai = step.get("action_input", {})
                if action == "Search":
                    search_queries[sn] = extract_search_query(ai)
                    doc_titles[sn] = parse_search_titles(step.get("observation", ""))
                elif action == "Retrieve":
                    doc_titles[sn] = {extract_search_query(ai)} if ai else set()
        
        runs.append({
            "step_count": meta.get("num_steps") or len(steps_data),
            "hidden_states": hidden_states,
            "correct": meta.get("correct"),
            "obs_texts": obs_texts,
            "search_queries": search_queries,
            "doc_titles": doc_titles,
        })
    return runs


def load_easier_question_with_obs(fp, expected_answer):
    """Load an easier JSON question with observations."""
    with open(fp) as f:
        data = json.load(f)
    runs = []
    for run in data.get("runs", []):
        hidden_states = {}
        obs_texts = {}
        search_queries = {}
        doc_titles = {}
        
        for step in run.get("steps", []):
            sn = step.get("step_number", 0)
            # Load hidden states at analysis step
            layers_dict = step.get("hidden_states", {}).get("layers", {})
            if layers_dict:
                full = np.zeros((81, 8192), dtype=np.float32)
                for i in LLAMA_LAYERS:
                    lk = f"layer_{i}"
                    if lk in layers_dict and layers_dict[lk]:
                        full[i] = layers_dict[lk]
                hidden_states[sn] = full
            
            # Extract observations for steps 1-3
            if sn in OBS_STEPS:
                obs_texts[sn] = step.get("observation", "")
                action = step.get("action", "")
                ai = step.get("action_input", {})
                if action == "Search":
                    search_queries[sn] = extract_search_query(ai)
                    doc_titles[sn] = parse_search_titles(step.get("observation", ""))
                elif action == "Retrieve":
                    doc_titles[sn] = {extract_search_query(ai)} if ai else set()
        
        ans = extract_yes_no(run.get("final_answer"))
        correct = (ans == expected_answer) if ans else False
        runs.append({
            "step_count": len(run.get("steps", [])),
            "hidden_states": hidden_states,
            "correct": correct,
            "obs_texts": obs_texts,
            "search_queries": search_queries,
            "doc_titles": doc_titles,
        })
    del data
    return runs


def load_all_data():
    """Load all 100 Llama questions with hidden states AND observations."""
    print("\n" + "=" * 70)
    print("LOADING LLAMA 100q DATA (hidden states + observations)")
    print("=" * 70)
    
    with open(PILOT_Q_FILE) as f:
        pilot_qs = {q["id"]: q for q in json.load(f)}
    with open(EASIER_Q_FILE) as f:
        easier_qs = {q["id"]: q for q in json.load(f)}
    with open(NEW60_Q_FILE) as f:
        new60_qs = {q["id"]: q for q in json.load(f)}
    
    all_data = {}
    
    # Pilot: 20 hard questions
    print("  Loading pilot (20 hard, npy)...")
    for qdir in sorted(PILOT_DIR.iterdir()):
        if not qdir.is_dir():
            continue
        qid = qdir.name
        if qid not in pilot_qs:
            continue
        runs = load_npy_question_with_obs(qdir)
        if runs:
            all_data[qid] = {"runs": runs, "difficulty": "hard"}
    
    # New 60 questions
    print("  Loading new60 (60 questions, npy)...")
    for qdir in sorted(NEW60_DIR.iterdir()):
        if not qdir.is_dir():
            continue
        qid = qdir.name
        if qid not in new60_qs:
            continue
        runs = load_npy_question_with_obs(qdir)
        if runs:
            diff = new60_qs[qid].get("difficulty", "unknown")
            all_data[qid] = {"runs": runs, "difficulty": diff}
    
    # Easier 20 questions
    print("  Loading easier (20 easy, JSON)...")
    for fp in sorted(EASIER_DIR.glob("*.json")):
        qid = fp.stem
        if qid not in easier_qs or qid in all_data:
            continue
        expected = easier_qs[qid].get("answer", "").lower().strip()
        runs = load_easier_question_with_obs(fp, expected)
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


# ═══════════════════════════════════════════════════════════════════════
# OBSERVATION SIMILARITY COMPUTATION
# ═══════════════════════════════════════════════════════════════════════

def compute_obs_similarities(data):
    """Compute three observation similarity measures per question."""
    results = {}
    
    for qid, entry in data.items():
        runs = entry["runs"]
        if len(runs) < 2:
            continue
        
        # --- 1. Jaccard similarity of retrieved document sets ---
        # Collect all doc titles across steps 1-3 per run
        run_doc_sets = []
        for r in runs:
            all_titles = set()
            for s in OBS_STEPS:
                all_titles.update(r["doc_titles"].get(s, set()))
            run_doc_sets.append(all_titles)
        
        jaccard_sims = []
        for i, j in combinations(range(len(runs)), 2):
            jaccard_sims.append(jaccard(run_doc_sets[i], run_doc_sets[j]))
        mean_jaccard = float(np.mean(jaccard_sims)) if jaccard_sims else np.nan
        
        # --- 2. TF-IDF cosine similarity of concatenated observation text ---
        run_obs_texts = []
        for r in runs:
            concat = " ".join(r["obs_texts"].get(s, "") for s in OBS_STEPS)
            run_obs_texts.append(concat)
        
        # Filter out empty texts
        non_empty = [t for t in run_obs_texts if t.strip()]
        if len(non_empty) >= 2:
            try:
                vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
                tfidf_matrix = vectorizer.fit_transform(run_obs_texts)
                sim_matrix = cosine_similarity(tfidf_matrix)
                n = len(runs)
                upper_tri = sim_matrix[np.triu_indices(n, k=1)]
                mean_tfidf_sim = float(np.mean(upper_tri))
            except ValueError:
                mean_tfidf_sim = np.nan
        else:
            mean_tfidf_sim = np.nan
        
        # --- 3. Search query overlap ---
        run_query_sets = []
        for r in runs:
            queries = set()
            for s in OBS_STEPS:
                q = r["search_queries"].get(s, "")
                if q:
                    queries.add(q.lower().strip())
            run_query_sets.append(queries)
        
        query_sims = []
        for i, j in combinations(range(len(runs)), 2):
            query_sims.append(jaccard(run_query_sets[i], run_query_sets[j]))
        mean_query_sim = float(np.mean(query_sims)) if query_sims else np.nan
        
        results[qid] = {
            "jaccard_doc": mean_jaccard,
            "tfidf_obs": mean_tfidf_sim,
            "query_overlap": mean_query_sim,
        }
    
    return results


def compute_activation_similarity(data, step=4, layer=40):
    """Compute mean pairwise cosine similarity of hidden states at step/layer."""
    act_sims = {}
    for qid, entry in data.items():
        vectors = []
        for r in entry["runs"]:
            hs = r["hidden_states"].get(step)
            if hs is not None:
                vectors.append(hs[layer])
        if len(vectors) >= 2:
            act_sims[qid] = pairwise_cosine_sim(vectors)
    return act_sims


# ═══════════════════════════════════════════════════════════════════════
# STATISTICAL ANALYSES
# ═══════════════════════════════════════════════════════════════════════

def partial_correlation(x, y, covariates):
    """Partial correlation between x and y, controlling for covariates.
    Uses residualization: regress x and y on covariates, correlate residuals."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    C = np.column_stack(covariates)
    
    # Add intercept
    C_with_intercept = np.column_stack([np.ones(len(x)), C])
    
    # Residualize x
    beta_x = np.linalg.lstsq(C_with_intercept, x, rcond=None)[0]
    resid_x = x - C_with_intercept @ beta_x
    
    # Residualize y
    beta_y = np.linalg.lstsq(C_with_intercept, y, rcond=None)[0]
    resid_y = y - C_with_intercept @ beta_y
    
    r, p = stats.pearsonr(resid_x, resid_y)
    return float(r), float(p)


def compute_vif(X, idx):
    """Compute variance inflation factor for variable at index idx in X."""
    X = np.array(X)
    y_col = X[:, idx]
    other_cols = np.delete(X, idx, axis=1)
    other_with_intercept = np.column_stack([np.ones(len(y_col)), other_cols])
    beta = np.linalg.lstsq(other_with_intercept, y_col, rcond=None)[0]
    predicted = other_with_intercept @ beta
    ss_res = np.sum((y_col - predicted) ** 2)
    ss_tot = np.sum((y_col - np.mean(y_col)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return 1 / (1 - r_squared) if r_squared < 1 else float("inf")


def multiple_regression_with_stats(y, X_dict):
    """OLS regression with standardized betas, t-values, p-values, R².
    X_dict: {name: array} for each predictor."""
    names = list(X_dict.keys())
    X_raw = np.column_stack([X_dict[n] for n in names])
    y = np.array(y, dtype=float)
    
    # Standardize
    X_mean = X_raw.mean(axis=0)
    X_std = X_raw.std(axis=0)
    X_std[X_std == 0] = 1
    X_z = (X_raw - X_mean) / X_std
    y_mean = y.mean()
    y_std = y.std()
    if y_std == 0:
        y_std = 1
    y_z = (y - y_mean) / y_std
    
    # Add intercept for raw regression
    n = len(y)
    X_with_intercept = np.column_stack([np.ones(n), X_raw])
    
    # OLS
    beta_hat = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
    y_pred = X_with_intercept @ beta_hat
    residuals = y - y_pred
    
    # R² and adjusted R²
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    p = len(names)
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
    
    # Standard errors, t-values, p-values
    mse = ss_res / (n - p - 1) if n > p + 1 else 0
    try:
        cov_matrix = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
        se = np.sqrt(np.diag(cov_matrix))
    except np.linalg.LinAlgError:
        se = np.full(p + 1, np.nan)
    
    t_values = beta_hat / se
    p_values = [float(2 * stats.t.sf(abs(t), df=n - p - 1)) for t in t_values]
    
    # Standardized betas (from standardized regression)
    X_z_with_intercept = np.column_stack([np.ones(n), X_z])
    beta_z = np.linalg.lstsq(X_z_with_intercept, y_z, rcond=None)[0]
    
    results = {
        "r_squared": float(r_squared),
        "adj_r_squared": float(adj_r_squared),
        "predictors": {}
    }
    for i, name in enumerate(names):
        results["predictors"][name] = {
            "raw_beta": float(beta_hat[i + 1]),
            "std_beta": float(beta_z[i + 1]),
            "t_value": float(t_values[i + 1]),
            "p_value": float(p_values[i + 1]),
            "se": float(se[i + 1]),
        }
    
    return results


def compute_delta_r_squared(y, X_full_dict, variable_to_test):
    """Compute ΔR² for a variable: R²(full) - R²(reduced without that variable)."""
    y = np.array(y, dtype=float)
    
    # Full model
    X_full = np.column_stack([X_full_dict[n] for n in X_full_dict])
    X_full_int = np.column_stack([np.ones(len(y)), X_full])
    beta_full = np.linalg.lstsq(X_full_int, y, rcond=None)[0]
    ss_res_full = np.sum((y - X_full_int @ beta_full) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2_full = 1 - ss_res_full / ss_tot if ss_tot > 0 else 0
    
    # Reduced model (without the variable)
    reduced_names = [n for n in X_full_dict if n != variable_to_test]
    if not reduced_names:
        return r2_full
    X_red = np.column_stack([X_full_dict[n] for n in reduced_names])
    X_red_int = np.column_stack([np.ones(len(y)), X_red])
    beta_red = np.linalg.lstsq(X_red_int, y, rcond=None)[0]
    ss_res_red = np.sum((y - X_red_int @ beta_red) ** 2)
    r2_red = 1 - ss_res_red / ss_tot if ss_tot > 0 else 0
    
    return float(r2_full - r2_red)


# ═══════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def main():
    # Load data
    data = load_all_data()
    
    # Compute observation similarities
    print("\n" + "=" * 70)
    print("COMPUTING OBSERVATION SIMILARITIES")
    print("=" * 70)
    obs_sims = compute_obs_similarities(data)
    
    # Compute activation similarity at step 4, layer 40
    print("\nComputing activation similarities (step 4, layer 40)...")
    act_sims = compute_activation_similarity(data, step=4, layer=40)
    
    # Align all data: only questions with both obs and act similarity
    common_qids = sorted(set(obs_sims.keys()) & set(act_sims.keys()))
    print(f"\n  Questions with both obs and act similarity: {len(common_qids)}")
    
    # Build aligned arrays
    cvs = np.array([data[q]["cv"] for q in common_qids])
    act_sim_arr = np.array([act_sims[q] for q in common_qids])
    jaccard_arr = np.array([obs_sims[q]["jaccard_doc"] for q in common_qids])
    tfidf_arr = np.array([obs_sims[q]["tfidf_obs"] for q in common_qids])
    query_arr = np.array([obs_sims[q]["query_overlap"] for q in common_qids])
    accuracy_arr = np.array([data[q]["correct_rate"] for q in common_qids])
    difficulty_arr = np.array([1.0 if data[q]["difficulty"] == "hard" else 0.0 for q in common_qids])
    
    # Handle NaN in tfidf (some questions may have empty obs)
    tfidf_valid = ~np.isnan(tfidf_arr)
    
    # Use TF-IDF as primary obs similarity (most comprehensive)
    # Fall back to Jaccard for questions without TF-IDF
    obs_sim_primary = np.where(tfidf_valid, tfidf_arr, jaccard_arr)
    
    print(f"\n  TF-IDF valid: {tfidf_valid.sum()}, NaN: {(~tfidf_valid).sum()}")
    
    # ═══════════════════════════════════════════════════════════════════
    # ANALYSIS 1: Raw correlations of observation similarity with CV
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Observation similarity correlates with CV")
    print("=" * 70)
    
    results = {"n_questions": len(common_qids)}
    
    for name, arr in [("jaccard_doc", jaccard_arr), ("tfidf_obs", tfidf_arr), 
                       ("query_overlap", query_arr), ("activation_sim", act_sim_arr)]:
        valid = ~np.isnan(arr)
        if valid.sum() < 5:
            print(f"  {name}: insufficient valid data ({valid.sum()})")
            continue
        r, p = stats.pearsonr(arr[valid], cvs[valid])
        ci_lo, ci_hi = bootstrap_ci(arr[valid], cvs[valid])
        print(f"  {name} vs CV: r = {r:.3f}, p = {p:.4f}, 95% CI [{ci_lo:.3f}, {ci_hi:.3f}] (n={valid.sum()})")
        results[f"{name}_vs_cv"] = {
            "r": round(r, 4), "p": round(p, 6),
            "ci_lo": round(ci_lo, 4), "ci_hi": round(ci_hi, 4),
            "n": int(valid.sum()),
        }
    
    # Also: correlation between activation sim and observation sim
    for obs_name, obs_arr in [("jaccard_doc", jaccard_arr), ("tfidf_obs", tfidf_arr),
                               ("query_overlap", query_arr)]:
        valid = ~np.isnan(obs_arr)
        if valid.sum() < 5:
            continue
        r, p = stats.pearsonr(act_sim_arr[valid], obs_arr[valid])
        print(f"  activation_sim vs {obs_name}: r = {r:.3f}, p = {p:.4f}")
        results[f"activation_sim_vs_{obs_name}"] = {"r": round(r, 4), "p": round(p, 6)}
    
    # ═══════════════════════════════════════════════════════════════════
    # ANALYSIS 2: Partial correlations
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Partial correlations (activation_sim → CV)")
    print("=" * 70)
    
    # Use TF-IDF as primary observation similarity measure
    # For questions with NaN TF-IDF, use Jaccard
    valid_mask = ~np.isnan(obs_sim_primary)
    
    # a) Original partial r (controlling for accuracy + difficulty only)
    pr_orig, pp_orig = partial_correlation(
        act_sim_arr[valid_mask], cvs[valid_mask],
        [accuracy_arr[valid_mask], difficulty_arr[valid_mask]]
    )
    print(f"\n  Partial r (act_sim → CV | accuracy, difficulty): r = {pr_orig:.3f}, p = {pp_orig:.6f}")
    results["partial_r_original"] = {"r": round(pr_orig, 4), "p": round(pp_orig, 6)}
    
    # b) NEW: Partial r controlling for obs_sim + accuracy + difficulty
    pr_new, pp_new = partial_correlation(
        act_sim_arr[valid_mask], cvs[valid_mask],
        [obs_sim_primary[valid_mask], accuracy_arr[valid_mask], difficulty_arr[valid_mask]]
    )
    print(f"  Partial r (act_sim → CV | obs_sim, accuracy, difficulty): r = {pr_new:.3f}, p = {pp_new:.6f}")
    results["partial_r_controlling_obs"] = {"r": round(pr_new, 4), "p": round(pp_new, 6)}
    
    # c) Partial r of obs_sim → CV, controlling for act_sim + accuracy + difficulty
    pr_obs, pp_obs = partial_correlation(
        obs_sim_primary[valid_mask], cvs[valid_mask],
        [act_sim_arr[valid_mask], accuracy_arr[valid_mask], difficulty_arr[valid_mask]]
    )
    print(f"  Partial r (obs_sim → CV | act_sim, accuracy, difficulty): r = {pr_obs:.3f}, p = {pp_obs:.6f}")
    results["partial_r_obs_controlling_act"] = {"r": round(pr_obs, 4), "p": round(pp_obs, 6)}
    
    # Bootstrap CI for the key partial r
    print("\n  Computing bootstrap CI for partial r (act_sim → CV | obs_sim, accuracy, difficulty)...")
    rng = np.random.default_rng(SEED)
    n_valid = valid_mask.sum()
    boot_partial_rs = []
    for _ in range(5000):
        idx = rng.choice(n_valid, n_valid, replace=True)
        try:
            pr_b, _ = partial_correlation(
                act_sim_arr[valid_mask][idx], cvs[valid_mask][idx],
                [obs_sim_primary[valid_mask][idx], accuracy_arr[valid_mask][idx], difficulty_arr[valid_mask][idx]]
            )
            if not np.isnan(pr_b):
                boot_partial_rs.append(pr_b)
        except Exception:
            pass
    if boot_partial_rs:
        ci_lo = float(np.percentile(boot_partial_rs, 2.5))
        ci_hi = float(np.percentile(boot_partial_rs, 97.5))
        print(f"  Bootstrap 95% CI for partial r: [{ci_lo:.3f}, {ci_hi:.3f}]")
        results["partial_r_controlling_obs"]["ci_lo"] = round(ci_lo, 4)
        results["partial_r_controlling_obs"]["ci_hi"] = round(ci_hi, 4)
    
    # ═══════════════════════════════════════════════════════════════════
    # ANALYSIS 3: VIF between activation_sim and obs_sim
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Variance Inflation Factor")
    print("=" * 70)
    
    X_vif = np.column_stack([
        act_sim_arr[valid_mask],
        obs_sim_primary[valid_mask],
        accuracy_arr[valid_mask],
    ])
    vif_act = compute_vif(X_vif, 0)  # VIF for activation_sim
    vif_obs = compute_vif(X_vif, 1)  # VIF for obs_sim
    vif_acc = compute_vif(X_vif, 2)  # VIF for accuracy
    print(f"  VIF(activation_sim) = {vif_act:.2f}")
    print(f"  VIF(obs_sim)        = {vif_obs:.2f}")
    print(f"  VIF(accuracy)       = {vif_acc:.2f}")
    if vif_act > 5:
        print("  ⚠ WARNING: VIF > 5 for activation_sim — collinearity concern!")
    else:
        print(f"  ✓ VIF < 5 for all predictors — collinearity is not a concern")
    results["vif"] = {
        "activation_sim": round(vif_act, 3),
        "obs_sim": round(vif_obs, 3),
        "accuracy": round(vif_acc, 3),
    }
    
    # ═══════════════════════════════════════════════════════════════════
    # ANALYSIS 4: Multiple regression
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("ANALYSIS 4: Multiple regression (CV ~ act_sim + obs_sim + accuracy)")
    print("=" * 70)
    
    X_dict = {
        "activation_sim": act_sim_arr[valid_mask],
        "obs_sim": obs_sim_primary[valid_mask],
        "accuracy": accuracy_arr[valid_mask],
    }
    reg_results = multiple_regression_with_stats(cvs[valid_mask], X_dict)
    
    print(f"\n  R² = {reg_results['r_squared']:.3f}, Adj R² = {reg_results['adj_r_squared']:.3f}")
    for name, vals in reg_results["predictors"].items():
        sig = "*" if vals["p_value"] < 0.05 else ""
        print(f"  {name:20s}: β = {vals['std_beta']:+.3f}, t = {vals['t_value']:+.2f}, p = {vals['p_value']:.4f} {sig}")
    results["regression"] = reg_results
    
    # ΔR² for activation_sim
    delta_r2_act = compute_delta_r_squared(cvs[valid_mask], X_dict, "activation_sim")
    delta_r2_obs = compute_delta_r_squared(cvs[valid_mask], X_dict, "obs_sim")
    print(f"\n  ΔR²(activation_sim) = {delta_r2_act:.4f}")
    print(f"  ΔR²(obs_sim)        = {delta_r2_obs:.4f}")
    results["delta_r_squared"] = {
        "activation_sim": round(delta_r2_act, 5),
        "obs_sim": round(delta_r2_obs, 5),
    }
    
    # Also: regression with obs_sim broken out by type
    print("\n  Regression with all three obs measures:")
    X_dict_full = {
        "activation_sim": act_sim_arr[valid_mask],
        "jaccard_doc": jaccard_arr[valid_mask],
        "tfidf_obs": tfidf_arr[valid_mask] if tfidf_valid[valid_mask].all() else obs_sim_primary[valid_mask],
        "query_overlap": query_arr[valid_mask],
        "accuracy": accuracy_arr[valid_mask],
    }
    # Only run if no NaN
    has_nan = any(np.isnan(X_dict_full[k]).any() for k in X_dict_full)
    if not has_nan:
        reg_full = multiple_regression_with_stats(cvs[valid_mask], X_dict_full)
        print(f"  R² = {reg_full['r_squared']:.3f}")
        for name, vals in reg_full["predictors"].items():
            sig = "*" if vals["p_value"] < 0.05 else ""
            print(f"    {name:20s}: β = {vals['std_beta']:+.3f}, t = {vals['t_value']:+.2f}, p = {vals['p_value']:.4f} {sig}")
        results["regression_full"] = reg_full
    
    # ═══════════════════════════════════════════════════════════════════
    # ANALYSIS 5: Divergent-observation, convergent-representation
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("ANALYSIS 5: Divergent-observation, convergent-representation questions")
    print("=" * 70)
    
    # Quartile split on obs similarity and activation similarity
    obs_q25 = np.percentile(obs_sim_primary[valid_mask], 25)
    obs_q75 = np.percentile(obs_sim_primary[valid_mask], 75)
    act_q25 = np.percentile(act_sim_arr[valid_mask], 25)
    act_q75 = np.percentile(act_sim_arr[valid_mask], 75)
    
    valid_qids = [q for q, v in zip(common_qids, valid_mask) if v]
    
    # Divergent-obs, convergent-rep: low obs sim (bottom 25%) AND high act sim (top 25%)
    div_obs_conv_rep = []
    # Convergent-obs, divergent-rep: high obs sim (top 25%) AND low act sim (bottom 25%)
    conv_obs_div_rep = []
    
    for i, qid in enumerate(valid_qids):
        obs_val = obs_sim_primary[valid_mask][i]
        act_val = act_sim_arr[valid_mask][i]
        cv_val = cvs[valid_mask][i]
        
        if obs_val <= obs_q25 and act_val >= act_q75:
            div_obs_conv_rep.append({"qid": qid, "obs_sim": obs_val, "act_sim": act_val, "cv": cv_val})
        elif obs_val >= obs_q75 and act_val <= act_q25:
            conv_obs_div_rep.append({"qid": qid, "obs_sim": obs_val, "act_sim": act_val, "cv": cv_val})
    
    print(f"\n  Divergent-obs, convergent-rep (low obs sim, high act sim): {len(div_obs_conv_rep)} questions")
    if div_obs_conv_rep:
        mean_cv = np.mean([q["cv"] for q in div_obs_conv_rep])
        print(f"    Mean CV: {mean_cv:.3f}")
        for q in div_obs_conv_rep:
            print(f"    {q['qid']}: obs_sim={q['obs_sim']:.3f}, act_sim={q['act_sim']:.3f}, CV={q['cv']:.3f}")
    
    print(f"\n  Convergent-obs, divergent-rep (high obs sim, low act sim): {len(conv_obs_div_rep)} questions")
    if conv_obs_div_rep:
        mean_cv = np.mean([q["cv"] for q in conv_obs_div_rep])
        print(f"    Mean CV: {mean_cv:.3f}")
        for q in conv_obs_div_rep:
            print(f"    {q['qid']}: obs_sim={q['obs_sim']:.3f}, act_sim={q['act_sim']:.3f}, CV={q['cv']:.3f}")
    
    # Overall population mean CV for comparison
    overall_mean_cv = float(np.mean(cvs[valid_mask]))
    print(f"\n  Overall mean CV: {overall_mean_cv:.3f}")
    
    results["divergent_obs_convergent_rep"] = {
        "n": len(div_obs_conv_rep),
        "mean_cv": round(float(np.mean([q["cv"] for q in div_obs_conv_rep])), 4) if div_obs_conv_rep else None,
        "questions": div_obs_conv_rep,
    }
    results["convergent_obs_divergent_rep"] = {
        "n": len(conv_obs_div_rep),
        "mean_cv": round(float(np.mean([q["cv"] for q in conv_obs_div_rep])), 4) if conv_obs_div_rep else None,
        "questions": conv_obs_div_rep,
    }
    results["overall_mean_cv"] = round(overall_mean_cv, 4)
    
    # ═══════════════════════════════════════════════════════════════════
    # ANALYSIS 6: Summary statistics for observation similarity measures
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    for name, arr in [("jaccard_doc", jaccard_arr), ("tfidf_obs", tfidf_arr),
                       ("query_overlap", query_arr), ("activation_sim", act_sim_arr)]:
        valid = ~np.isnan(arr)
        if valid.sum() > 0:
            print(f"  {name:20s}: mean={np.mean(arr[valid]):.3f}, std={np.std(arr[valid]):.3f}, "
                  f"min={np.min(arr[valid]):.3f}, max={np.max(arr[valid]):.3f}, n={valid.sum()}")
    results["summary_stats"] = {
        "jaccard_doc": {"mean": round(float(np.nanmean(jaccard_arr)), 4), "std": round(float(np.nanstd(jaccard_arr)), 4)},
        "tfidf_obs": {"mean": round(float(np.nanmean(tfidf_arr)), 4), "std": round(float(np.nanstd(tfidf_arr)), 4)},
        "query_overlap": {"mean": round(float(np.nanmean(query_arr)), 4), "std": round(float(np.nanstd(query_arr)), 4)},
        "activation_sim": {"mean": round(float(np.nanmean(act_sim_arr)), 4), "std": round(float(np.nanstd(act_sim_arr)), 4)},
    }
    
    # ═══════════════════════════════════════════════════════════════════
    # SUCCESS CRITERION CHECK
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SUCCESS CRITERION CHECK")
    print("=" * 70)
    key_partial_r = results.get("partial_r_controlling_obs", {}).get("r", 0)
    if key_partial_r < -0.30:
        print(f"  ✓ PASS: Partial r = {key_partial_r:.3f} (< -0.30)")
        print(f"    Activation similarity predicts CV ABOVE AND BEYOND observation similarity.")
    elif key_partial_r < -0.20:
        print(f"  ⚠ MARGINAL: Partial r = {key_partial_r:.3f} (between -0.30 and -0.20)")
        print(f"    Activation similarity still contributes but observation similarity absorbs some signal.")
    else:
        print(f"  ✗ FAIL: Partial r = {key_partial_r:.3f} (> -0.20)")
        print(f"    The observation similarity confound may explain the commitment signal.")
    
    # Save results
    output_file = OUTPUT_DIR / "observation_similarity_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {output_file}")
    
    return results


if __name__ == "__main__":
    main()
