"""
mediation_analysis.py — Task 3: Bootstrap Mediation Analysis

Tests whether activation similarity mediates the effect of commitment
prompting on behavioral consistency.

IV:  condition (0=filler, 1=commitment)
M:   activation similarity at step 4, layer 40
DV:  behavioral CV

Method: Preacher & Hayes bootstrap mediation (5000 iterations)
- Path a: IV → M
- Path b: M → DV (controlling for IV)
- Indirect effect: a * b
- Direct effect: c' (IV → DV controlling for M)
- Total effect: c (IV → DV)

Run from hotpotqa/ directory:
    ../.venv/bin/python3 mediation_analysis.py
"""

import json
import numpy as np
from pathlib import Path
from itertools import combinations
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path(__file__).parent
OUTPUT_DIR = BASE / "analysis_results" / "v3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_BOOTSTRAP = 5000


def pairwise_cosine_sim(vectors):
    """Mean pairwise cosine similarity (upper triangle)."""
    if len(vectors) < 2:
        return np.nan
    arr = np.array(vectors)
    sim_matrix = cosine_similarity(arr)
    n = len(vectors)
    upper_tri = sim_matrix[np.triu_indices(n, k=1)]
    return float(np.mean(upper_tri))


def compute_cv(step_counts):
    if len(step_counts) < 2 or np.mean(step_counts) == 0:
        return 0.0
    return float(np.std(step_counts) / np.mean(step_counts))


def load_intervention_data():
    """Load commitment and filler condition data with hidden states.
    Returns matched questions with CV and activation similarity for each condition."""
    
    commit_dir = BASE / "commitment"
    filler_dir = BASE / "filler"
    
    conditions = {}
    for cond_name, cond_dir in [("commitment", commit_dir), ("filler", filler_dir)]:
        cond_data = {}
        for qdir in sorted(cond_dir.iterdir()):
            if not qdir.is_dir():
                continue
            qid = qdir.name
            runs = []
            for run_dir in sorted(qdir.glob("run_*")):
                meta_file = run_dir / "metadata.json"
                if not meta_file.exists():
                    continue
                with open(meta_file) as f:
                    meta = json.load(f)
                
                # Load hidden states at step 4
                hs_file = run_dir / "hidden_states_step_4.npy"
                hs = None
                if hs_file.exists():
                    hs = np.load(hs_file)  # shape (81, 8192)
                
                runs.append({
                    "step_count": meta.get("total_steps", 0),
                    "correct": meta.get("correct", False),
                    "hidden_state_step4": hs,
                })
            
            if runs:
                step_counts = [r["step_count"] for r in runs]
                # Compute activation similarity at step 4, layer 40
                vectors = [r["hidden_state_step4"][40] for r in runs 
                          if r["hidden_state_step4"] is not None]
                act_sim = pairwise_cosine_sim(vectors) if len(vectors) >= 2 else np.nan
                
                cond_data[qid] = {
                    "cv": compute_cv(step_counts),
                    "act_sim": act_sim,
                    "accuracy": sum(1 for r in runs if r["correct"]) / len(runs),
                    "n_runs": len(runs),
                    "n_with_hs": len(vectors),
                }
        conditions[cond_name] = cond_data
        print(f"  {cond_name}: {len(cond_data)} questions")
    
    # Match questions: both conditions, both have activation similarity
    common = set(conditions["commitment"].keys()) & set(conditions["filler"].keys())
    matched = [qid for qid in sorted(common)
               if not np.isnan(conditions["commitment"][qid]["act_sim"])
               and not np.isnan(conditions["filler"][qid]["act_sim"])]
    
    print(f"  Matched questions with act_sim in both conditions: {len(matched)}")
    return conditions, matched


def ols_regression(y, X):
    """Simple OLS. X should include intercept column. Returns betas."""
    return np.linalg.lstsq(X, y, rcond=None)[0]


def bootstrap_mediation(iv, mediator, dv, n_boot=N_BOOTSTRAP, seed=SEED):
    """Bootstrap mediation analysis (Preacher & Hayes method).
    
    IV → M (path a)
    M → DV controlling for IV (path b)  
    IV → DV controlling for M (path c', direct effect)
    IV → DV (path c, total effect)
    Indirect effect = a * b
    """
    rng = np.random.default_rng(seed)
    n = len(iv)
    
    iv = np.array(iv, dtype=float)
    mediator = np.array(mediator, dtype=float)
    dv = np.array(dv, dtype=float)
    
    def compute_paths(iv_s, med_s, dv_s):
        """Compute mediation paths for a sample."""
        ones = np.ones(len(iv_s))
        
        # Path a: IV → M
        X_a = np.column_stack([ones, iv_s])
        beta_a = ols_regression(med_s, X_a)
        a = beta_a[1]
        
        # Paths b and c': M → DV, IV → DV, controlling for each other
        X_bc = np.column_stack([ones, iv_s, med_s])
        beta_bc = ols_regression(dv_s, X_bc)
        c_prime = beta_bc[1]  # direct effect
        b = beta_bc[2]         # M → DV controlling for IV
        
        # Path c: IV → DV (total effect)
        X_c = np.column_stack([ones, iv_s])
        beta_c = ols_regression(dv_s, X_c)
        c = beta_c[1]
        
        # Indirect effect
        indirect = a * b
        
        return {
            "a": a, "b": b, "c": c, "c_prime": c_prime,
            "indirect": indirect,
        }
    
    # Observed paths
    observed = compute_paths(iv, mediator, dv)
    
    # Bootstrap
    boot_indirect = []
    boot_a = []
    boot_b = []
    boot_c_prime = []
    
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        try:
            paths = compute_paths(iv[idx], mediator[idx], dv[idx])
            boot_indirect.append(paths["indirect"])
            boot_a.append(paths["a"])
            boot_b.append(paths["b"])
            boot_c_prime.append(paths["c_prime"])
        except (np.linalg.LinAlgError, ValueError):
            pass
    
    boot_indirect = np.array(boot_indirect)
    boot_a = np.array(boot_a)
    boot_b = np.array(boot_b)
    boot_c_prime = np.array(boot_c_prime)
    
    # Percentile CIs
    def percentile_ci(arr, alpha=0.05):
        return float(np.percentile(arr, 100 * alpha / 2)), float(np.percentile(arr, 100 * (1 - alpha / 2)))
    
    # Bias-corrected accelerated (BCa) CI for indirect effect
    # Simplified: use percentile method
    indirect_ci = percentile_ci(boot_indirect)
    a_ci = percentile_ci(boot_a)
    b_ci = percentile_ci(boot_b)
    c_prime_ci = percentile_ci(boot_c_prime)
    
    # Proportion mediated
    if observed["c"] != 0:
        prop_mediated = observed["indirect"] / observed["c"]
    else:
        prop_mediated = np.nan
    
    # p-value for indirect effect (proportion of bootstrap samples crossing 0)
    if len(boot_indirect) > 0:
        # Two-tailed: proportion of samples with sign opposite to observed
        if observed["indirect"] > 0:
            p_indirect = 2 * np.mean(boot_indirect <= 0)
        else:
            p_indirect = 2 * np.mean(boot_indirect >= 0)
        p_indirect = min(p_indirect, 1.0)
    else:
        p_indirect = np.nan
    
    return {
        "n": n,
        "n_boot": len(boot_indirect),
        "paths": {
            "a": {"estimate": float(observed["a"]), "ci": a_ci, "desc": "IV → Mediator"},
            "b": {"estimate": float(observed["b"]), "ci": b_ci, "desc": "Mediator → DV (controlling IV)"},
            "c": {"estimate": float(observed["c"]), "desc": "Total effect (IV → DV)"},
            "c_prime": {"estimate": float(observed["c_prime"]), "ci": c_prime_ci, "desc": "Direct effect (IV → DV, controlling M)"},
            "indirect": {
                "estimate": float(observed["indirect"]),
                "ci": indirect_ci,
                "p_value": float(p_indirect),
                "desc": "Indirect effect (a * b)",
            },
        },
        "proportion_mediated": float(prop_mediated) if not np.isnan(prop_mediated) else None,
    }


def main():
    print("=" * 70)
    print("MEDIATION ANALYSIS: Commitment → Activation Similarity → CV")
    print("=" * 70)
    
    # Load data
    print("\nLoading intervention data...")
    conditions, matched_qids = load_intervention_data()
    
    if len(matched_qids) < 20:
        print(f"ERROR: Only {len(matched_qids)} matched questions. Need at least 20.")
        return
    
    # Build arrays: each question contributes TWO observations (filler + commitment)
    # IV: 0=filler, 1=commitment
    iv = []
    mediator = []  # activation similarity
    dv = []        # behavioral CV
    
    for qid in matched_qids:
        # Filler
        iv.append(0)
        mediator.append(conditions["filler"][qid]["act_sim"])
        dv.append(conditions["filler"][qid]["cv"])
        
        # Commitment
        iv.append(1)
        mediator.append(conditions["commitment"][qid]["act_sim"])
        dv.append(conditions["commitment"][qid]["cv"])
    
    iv = np.array(iv)
    mediator = np.array(mediator)
    dv = np.array(dv)
    
    print(f"\nTotal observations: {len(iv)} ({len(matched_qids)} questions × 2 conditions)")
    print(f"IV (condition): {(iv==0).sum()} filler, {(iv==1).sum()} commitment")
    print(f"Mediator (act_sim): mean={mediator.mean():.4f}, std={mediator.std():.4f}")
    print(f"DV (CV): mean={dv.mean():.4f}, std={dv.std():.4f}")
    
    # Descriptive stats by condition
    print(f"\nFiller:     act_sim={mediator[iv==0].mean():.4f}, CV={dv[iv==0].mean():.4f}")
    print(f"Commitment: act_sim={mediator[iv==1].mean():.4f}, CV={dv[iv==1].mean():.4f}")
    
    # Run mediation analysis
    print(f"\nRunning bootstrap mediation ({N_BOOTSTRAP} iterations)...")
    results = bootstrap_mediation(iv, mediator, dv)
    
    # Print results
    print("\n" + "=" * 70)
    print("MEDIATION RESULTS")
    print("=" * 70)
    
    for path_name, path_data in results["paths"].items():
        ci_str = ""
        if "ci" in path_data:
            ci_str = f", 95% CI [{path_data['ci'][0]:.4f}, {path_data['ci'][1]:.4f}]"
        p_str = ""
        if "p_value" in path_data:
            p_str = f", p = {path_data['p_value']:.4f}"
        print(f"  {path_name:10s}: {path_data['estimate']:+.4f}{ci_str}{p_str}")
        print(f"             {path_data['desc']}")
    
    if results["proportion_mediated"] is not None:
        print(f"\n  Proportion mediated: {results['proportion_mediated']:.2%}")
    
    # Interpret
    indirect_ci = results["paths"]["indirect"]["ci"]
    indirect_est = results["paths"]["indirect"]["estimate"]
    
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    if indirect_ci[0] * indirect_ci[1] > 0:
        # CI doesn't include 0 → significant
        print(f"  The indirect effect is SIGNIFICANT.")
        print(f"  95% CI [{indirect_ci[0]:.4f}, {indirect_ci[1]:.4f}] does not include zero.")
        print(f"  Activation similarity partially mediates the effect of commitment")
        print(f"  prompting on behavioral consistency.")
        if results["proportion_mediated"]:
            print(f"  Proportion mediated: {results['proportion_mediated']:.0%}")
    else:
        # CI includes 0 → not significant
        print(f"  The indirect effect did NOT reach significance.")
        print(f"  95% CI [{indirect_ci[0]:.4f}, {indirect_ci[1]:.4f}] includes zero.")
        print(f"  ")
        print(f"  PRE-WRITTEN NULL PARAGRAPH:")
        print(f"  'The indirect effect did not reach significance")
        print(f"  (95% CI [{indirect_ci[0]:.4f}, {indirect_ci[1]:.4f}]),")
        print(f"  consistent with limited statistical power at n={len(matched_qids)}")
        print(f"  for detecting mediation of a moderate effect. We interpret the")
        print(f"  co-occurrence of representational and behavioral changes as")
        print(f"  consistent with, but not proof of, a mediating pathway.'")
    
    # Also run with Sobel test approximation for comparison
    print("\n" + "=" * 70)
    print("SOBEL TEST (for comparison)")
    print("=" * 70)
    
    # Path a: regress M on IV
    ones = np.ones(len(iv))
    X_a = np.column_stack([ones, iv])
    beta_a = np.linalg.lstsq(X_a, mediator, rcond=None)[0]
    resid_a = mediator - X_a @ beta_a
    se_a = np.sqrt(np.sum(resid_a**2) / (len(iv) - 2) / np.sum((iv - iv.mean())**2))
    
    # Paths b, c': regress DV on IV + M
    X_bc = np.column_stack([ones, iv, mediator])
    beta_bc = np.linalg.lstsq(X_bc, dv, rcond=None)[0]
    resid_bc = dv - X_bc @ beta_bc
    XtX_inv = np.linalg.inv(X_bc.T @ X_bc)
    mse = np.sum(resid_bc**2) / (len(iv) - 3)
    se_bc = np.sqrt(mse * np.diag(XtX_inv))
    se_b = se_bc[2]
    
    a = beta_a[1]
    b = beta_bc[2]
    
    # Sobel test statistic
    sobel_se = np.sqrt(a**2 * se_b**2 + b**2 * se_a**2)
    sobel_z = (a * b) / sobel_se
    sobel_p = 2 * (1 - stats.norm.cdf(abs(sobel_z)))
    
    print(f"  a = {a:.4f} (SE = {se_a:.4f})")
    print(f"  b = {b:.4f} (SE = {se_b:.4f})")
    print(f"  a*b = {a*b:.4f}")
    print(f"  Sobel z = {sobel_z:.3f}, p = {sobel_p:.4f}")
    
    results["sobel"] = {
        "a": float(a), "se_a": float(se_a),
        "b": float(b), "se_b": float(se_b),
        "indirect": float(a * b),
        "z": float(sobel_z), "p": float(sobel_p),
    }
    
    # Save
    output_file = OUTPUT_DIR / "mediation_analysis_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {output_file}")


if __name__ == "__main__":
    main()
