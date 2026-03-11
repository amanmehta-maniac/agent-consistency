"""
Paper 3: Three new analyses for robustness.

1. Answer agreement metric + correlation with activation similarity
2. Practical monitoring demo (calibration-based flagging, precision/recall)
3. Partial correlation with answer agreement as consistency measure

Reuses data loading from paper3_100q_analysis.py.
Run from hotpotqa/ directory:
    ../.venv/bin/python3 paper3_new_analyses.py
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper3_100q_analysis import (
    load_pilot_data, load_easier_data, load_new60_data,
    compute_pairwise_cosine_similarity, extract_answer,
    LAYERS, STEPS, OUTPUT_DIR,
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_answer_agreement(runs, expected_answer):
    answers = []
    for run in runs:
        raw = run.get("final_answer")
        if raw is None:
            continue
        normalized = str(raw).lower().strip()[:100]
        if normalized:
            answers.append(normalized)
    if not answers:
        return np.nan
    counts = Counter(answers)
    most_common_count = counts.most_common(1)[0][1]
    return most_common_count / len(answers)


def get_run_vectors(runs, step, layer):
    vectors = []
    for run in runs:
        hs = run["hidden_states"].get(step)
        if hs is not None and len(hs) > layer:
            vectors.append(hs[layer])
    return vectors


def partial_corr(x, y, z):
    Z = np.column_stack([z, np.ones(len(z))])
    res_x = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    res_y = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    r, p = stats.pearsonr(res_x, res_y)
    return r, p, res_x, res_y


def analysis_answer_agreement(all_data):
    print("\n" + "=" * 70)
    print("ANALYSIS A: ANSWER AGREEMENT vs ACTIVATION SIMILARITY")
    print("=" * 70)

    step, layer = 4, 40
    records = []

    for qid, entry in all_data.items():
        runs = entry["runs"]
        expected = entry["question"].get("answer", "").lower().strip()
        agreement = compute_answer_agreement(runs, expected)
        if np.isnan(agreement):
            continue

        vectors = get_run_vectors(runs, step, layer)
        if len(vectors) < 2:
            continue
        sim = compute_pairwise_cosine_similarity(vectors)
        if np.isnan(sim):
            continue

        step_counts = [r["step_count"] for r in runs]
        cv = np.std(step_counts) / np.mean(step_counts) if np.mean(step_counts) > 0 else 0
        correct_rate = sum(1 for r in runs if r.get("correct")) / len(runs)

        records.append({
            "qid": qid,
            "agreement": agreement,
            "sim": sim,
            "cv": cv,
            "correct_rate": correct_rate,
            "difficulty": entry["difficulty"],
        })

    n = len(records)
    agreements = np.array([r["agreement"] for r in records])
    sims = np.array([r["sim"] for r in records])
    cvs = np.array([r["cv"] for r in records])

    r_ag_sim, p_ag_sim = stats.pearsonr(sims, agreements)
    r_ag_cv, p_ag_cv = stats.pearsonr(agreements, cvs)

    print(f"\n  n = {n}")
    print(f"  Answer agreement range: [{agreements.min():.2f}, {agreements.max():.2f}], mean={agreements.mean():.3f}")
    print(f"  Similarity vs Agreement:  r={r_ag_sim:.4f}, p={p_ag_sim:.6f}")
    print(f"  Agreement vs CV:          r={r_ag_cv:.4f}, p={p_ag_cv:.6f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    diffs = np.array([r["difficulty"] for r in records])
    hard_mask = diffs == "hard"
    ax.scatter(sims[hard_mask], agreements[hard_mask], c="#d62728", alpha=0.6, s=40,
               label="Hard", edgecolors="black", linewidth=0.3)
    ax.scatter(sims[~hard_mask], agreements[~hard_mask], c="#1f77b4", alpha=0.6, s=40,
               label="Easy", edgecolors="black", linewidth=0.3)
    z = np.polyfit(sims, agreements, 1)
    x_line = np.linspace(sims.min(), sims.max(), 100)
    ax.plot(x_line, np.poly1d(z)(x_line), "k--", alpha=0.5)
    ax.set_xlabel("Activation Similarity (Step 4, Layer 40)", fontsize=11)
    ax.set_ylabel("Answer Agreement", fontsize=11)
    ax.set_title(f"Similarity vs Answer Agreement\nr={r_ag_sim:.3f}, p={p_ag_sim:.4f}", fontsize=12)
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.scatter(agreements[hard_mask], cvs[hard_mask], c="#d62728", alpha=0.6, s=40,
               label="Hard", edgecolors="black", linewidth=0.3)
    ax.scatter(agreements[~hard_mask], cvs[~hard_mask], c="#1f77b4", alpha=0.6, s=40,
               label="Easy", edgecolors="black", linewidth=0.3)
    ax.set_xlabel("Answer Agreement", fontsize=11)
    ax.set_ylabel("Behavioral CV", fontsize=11)
    ax.set_title(f"Answer Agreement vs CV\nr={r_ag_cv:.3f}, p={p_ag_cv:.4f}", fontsize=12)
    ax.legend(fontsize=9)

    plt.suptitle("Answer Agreement: A Second Consistency Metric", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "answer_agreement_step4.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: answer_agreement_step4.png")

    result = {
        "step": step, "layer": layer, "n": n,
        "agreement_stats": {
            "min": float(agreements.min()), "max": float(agreements.max()),
            "mean": float(agreements.mean()), "std": float(agreements.std()),
        },
        "sim_vs_agreement": {"r": float(r_ag_sim), "p": float(p_ag_sim)},
        "agreement_vs_cv": {"r": float(r_ag_cv), "p": float(p_ag_cv)},
    }
    with open(OUTPUT_DIR / "answer_agreement_step4.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: answer_agreement_step4.json")

    return records, result


def analysis_monitoring_demo(all_data):
    print("\n" + "=" * 70)
    print("ANALYSIS B: PRACTICAL MONITORING DEMO (Calibration-Based Flagging)")
    print("=" * 70)

    step, layer = 4, 40
    n_calibration = 3

    all_flagged_correct = 0
    all_flagged_total = 0
    all_inconsistent_flagged = 0
    all_inconsistent_total = 0
    all_consistent_flagged = 0
    all_consistent_total = 0

    question_results = []

    for qid, entry in all_data.items():
        runs = entry["runs"]
        if len(runs) < n_calibration + 1:
            continue

        cal_vectors = []
        for run in runs[:n_calibration]:
            hs = run["hidden_states"].get(step)
            if hs is not None and len(hs) > layer:
                cal_vectors.append(hs[layer])
        if len(cal_vectors) < n_calibration:
            continue

        centroid = np.mean(cal_vectors, axis=0)
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-10)

        cal_distances = []
        for v in cal_vectors:
            v_norm = v / (np.linalg.norm(v) + 1e-10)
            cal_distances.append(1 - np.dot(centroid_norm, v_norm))

        test_runs = runs[n_calibration:]
        test_distances = []
        test_correct = []
        for run in test_runs:
            hs = run["hidden_states"].get(step)
            if hs is None or len(hs) <= layer:
                continue
            v = hs[layer]
            v_norm = v / (np.linalg.norm(v) + 1e-10)
            d = 1 - np.dot(centroid_norm, v_norm)
            test_distances.append(d)
            test_correct.append(run.get("correct", False))

        if not test_distances:
            continue

        cal_answers = []
        for run in runs[:n_calibration]:
            raw = run.get("final_answer")
            if raw is not None:
                ans = str(raw).lower().strip()[:100]
                if ans:
                    cal_answers.append(ans)

        if cal_answers:
            cal_majority = Counter(cal_answers).most_common(1)[0][0]
        else:
            cal_majority = None

        test_answers_match_cal = []
        for run in test_runs[:len(test_distances)]:
            raw = run.get("final_answer")
            if raw is not None:
                ans = str(raw).lower().strip()[:100]
            else:
                ans = None
            if cal_majority is not None and ans is not None:
                test_answers_match_cal.append(ans == cal_majority)
            else:
                test_answers_match_cal.append(None)

        question_results.append({
            "qid": qid,
            "difficulty": entry["difficulty"],
            "cal_distances": cal_distances,
            "test_distances": test_distances,
            "test_correct": test_correct,
            "test_answers_match_cal": test_answers_match_cal,
        })

    all_test_distances = []
    for qr in question_results:
        all_test_distances.extend(qr["test_distances"])
    threshold = np.median(all_test_distances)
    print(f"\n  Questions with sufficient data: {len(question_results)}")
    print(f"  Total test runs: {len(all_test_distances)}")
    print(f"  Threshold (median distance): {threshold:.6f}")

    flagged_inconsistent = 0
    flagged_total = 0
    unflagged_inconsistent = 0
    unflagged_total = 0

    for qr in question_results:
        for dist, correct, match_cal in zip(
            qr["test_distances"], qr["test_correct"], qr["test_answers_match_cal"]
        ):
            flagged = dist > threshold
            inconsistent = (match_cal is not None and not match_cal)

            if flagged:
                flagged_total += 1
                if inconsistent:
                    flagged_inconsistent += 1
            else:
                unflagged_total += 1
                if inconsistent:
                    unflagged_inconsistent += 1

    total_inconsistent = flagged_inconsistent + unflagged_inconsistent
    precision = flagged_inconsistent / flagged_total if flagged_total > 0 else 0
    recall = flagged_inconsistent / total_inconsistent if total_inconsistent > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n  --- Answer-Inconsistency Detection (vs calibration majority) ---")
    print(f"  Total flagged runs:      {flagged_total}")
    print(f"  Flagged & inconsistent:  {flagged_inconsistent}")
    print(f"  Total inconsistent runs: {total_inconsistent}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")

    flagged_incorrect = 0
    unflagged_incorrect = 0
    flagged_total2 = 0
    unflagged_total2 = 0
    for qr in question_results:
        for dist, correct in zip(qr["test_distances"], qr["test_correct"]):
            flagged = dist > threshold
            if flagged:
                flagged_total2 += 1
                if not correct:
                    flagged_incorrect += 1
            else:
                unflagged_total2 += 1
                if not correct:
                    unflagged_incorrect += 1

    total_incorrect = flagged_incorrect + unflagged_incorrect
    prec_correct = flagged_incorrect / flagged_total2 if flagged_total2 > 0 else 0
    rec_correct = flagged_incorrect / total_incorrect if total_incorrect > 0 else 0
    f1_correct = 2 * prec_correct * rec_correct / (prec_correct + rec_correct) if (prec_correct + rec_correct) > 0 else 0

    print(f"\n  --- Incorrectness Detection ---")
    print(f"  Flagged & incorrect:     {flagged_incorrect}")
    print(f"  Total incorrect runs:    {total_incorrect}")
    print(f"  Precision: {prec_correct:.4f}")
    print(f"  Recall:    {rec_correct:.4f}")
    print(f"  F1:        {f1_correct:.4f}")

    thresholds = np.percentile(all_test_distances, np.arange(5, 100, 5))
    pr_curve_answer = []
    pr_curve_correct = []
    for thr in thresholds:
        fi, ft, ui = 0, 0, 0
        fi2, ft2, ui2 = 0, 0, 0
        for qr in question_results:
            for dist, correct, match_cal in zip(
                qr["test_distances"], qr["test_correct"], qr["test_answers_match_cal"]
            ):
                flagged = dist > thr
                inconsistent = (match_cal is not None and not match_cal)
                if flagged:
                    ft += 1
                    ft2 += 1
                    if inconsistent:
                        fi += 1
                    if not correct:
                        fi2 += 1
                else:
                    if inconsistent:
                        ui += 1
                    if not correct:
                        ui2 += 1
        total_inc = fi + ui
        total_inc2 = fi2 + ui2
        p = fi / ft if ft > 0 else 0
        r = fi / total_inc if total_inc > 0 else 0
        p2 = fi2 / ft2 if ft2 > 0 else 0
        r2 = fi2 / total_inc2 if total_inc2 > 0 else 0
        pr_curve_answer.append({"threshold": float(thr), "precision": p, "recall": r, "n_flagged": ft})
        pr_curve_correct.append({"threshold": float(thr), "precision": p2, "recall": r2, "n_flagged": ft2})

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.hist(all_test_distances, bins=40, color="#1f77b4", edgecolor="black", alpha=0.7)
    ax.axvline(x=threshold, color="red", linewidth=2, linestyle="--", label=f"Threshold={threshold:.4f}")
    ax.set_xlabel("Cosine Distance from Calibration Centroid", fontsize=11)
    ax.set_ylabel("Count (test runs)", fontsize=11)
    ax.set_title("Distance Distribution of Test Runs", fontsize=12)
    ax.legend(fontsize=9)

    ax = axes[1]
    precs_a = [p["precision"] for p in pr_curve_answer]
    recs_a = [p["recall"] for p in pr_curve_answer]
    ax.plot(recs_a, precs_a, "o-", color="#d62728", markersize=3, label="Answer inconsistency")
    precs_c = [p["precision"] for p in pr_curve_correct]
    recs_c = [p["recall"] for p in pr_curve_correct]
    ax.plot(recs_c, precs_c, "s-", color="#1f77b4", markersize=3, label="Incorrectness")
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curves", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = axes[2]
    base_rates = []
    flag_rates = []
    for qr in question_results:
        n_test = len(qr["test_distances"])
        n_flagged = sum(1 for d in qr["test_distances"] if d > threshold)
        flag_rate = n_flagged / n_test if n_test > 0 else 0
        n_inc = sum(1 for m in qr["test_answers_match_cal"] if m is not None and not m)
        base_rate = n_inc / n_test if n_test > 0 else 0
        base_rates.append(base_rate)
        flag_rates.append(flag_rate)
    ax.scatter(base_rates, flag_rates, c="#2ca02c", alpha=0.6, s=40, edgecolors="black", linewidth=0.3)
    ax.set_xlabel("Base Inconsistency Rate (per question)", fontsize=11)
    ax.set_ylabel("Flagging Rate (per question)", fontsize=11)
    ax.set_title("Per-Question: Inconsistency vs Flagging", fontsize=12)
    r_qr, p_qr = stats.pearsonr(base_rates, flag_rates)
    ax.text(0.05, 0.95, f"r={r_qr:.3f}, p={p_qr:.4f}", transform=ax.transAxes, fontsize=10, va="top")

    plt.suptitle("Practical Monitoring: Calibration-Based Run Flagging (3 calibration runs)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "monitoring_demo_step4.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: monitoring_demo_step4.png")

    result = {
        "step": step, "layer": layer,
        "n_calibration_runs": n_calibration,
        "n_questions": len(question_results),
        "n_test_runs": len(all_test_distances),
        "threshold_median_distance": float(threshold),
        "answer_inconsistency_detection": {
            "flagged_total": flagged_total,
            "flagged_inconsistent": flagged_inconsistent,
            "total_inconsistent": total_inconsistent,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        },
        "incorrectness_detection": {
            "flagged_total": flagged_total2,
            "flagged_incorrect": flagged_incorrect,
            "total_incorrect": total_incorrect,
            "precision": float(prec_correct),
            "recall": float(rec_correct),
            "f1": float(f1_correct),
        },
        "per_question_correlation": {"r": float(r_qr), "p": float(p_qr)},
    }
    with open(OUTPUT_DIR / "monitoring_demo_step4.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: monitoring_demo_step4.json")

    return result


def analysis_partial_corr_agreement(all_data, records_from_a):
    print("\n" + "=" * 70)
    print("ANALYSIS C: PARTIAL CORRELATION WITH ANSWER AGREEMENT")
    print("=" * 70)

    step, layer = 4, 40

    agreements = np.array([r["agreement"] for r in records_from_a])
    sims = np.array([r["sim"] for r in records_from_a])
    accs = np.array([r["correct_rate"] for r in records_from_a])
    diffs = np.array([r["difficulty"] for r in records_from_a])
    n = len(records_from_a)

    raw_r, raw_p = stats.pearsonr(sims, agreements)
    print(f"\n  Raw corr (sim vs agreement, n={n}): r={raw_r:.4f}, p={raw_p:.6f}")

    pr_acc, pp_acc, res_sim_acc, res_ag_acc = partial_corr(sims, agreements, accs.reshape(-1, 1))
    print(f"  Partial (| accuracy):                r={pr_acc:.4f}, p={pp_acc:.6f}")

    diff_binary = np.array([1 if d == "hard" else 0 for d in diffs])
    controls = np.column_stack([accs, diff_binary])
    pr_both, pp_both, res_sim_both, res_ag_both = partial_corr(sims, agreements, controls)
    print(f"  Partial (| accuracy + difficulty):    r={pr_both:.4f}, p={pp_both:.6f}")

    print(f"\n  --- Comparison with CV-based partial correlation ---")
    cvs = np.array([r["cv"] for r in records_from_a])
    pr_cv_acc, pp_cv_acc, _, _ = partial_corr(sims, cvs, accs.reshape(-1, 1))
    pr_cv_both, pp_cv_both, _, _ = partial_corr(sims, cvs, controls)
    print(f"  CV partial (| accuracy):             r={pr_cv_acc:.4f}, p={pp_cv_acc:.6f}")
    print(f"  CV partial (| accuracy + difficulty): r={pr_cv_both:.4f}, p={pp_cv_both:.6f}")

    target_layers = [32, 40, 48, 56, 64, 72, 80]
    layer_results = {}

    print(f"\n  --- Layer-by-layer (agreement as DV) ---")
    print(f"  {'Layer':>5} {'Raw r':>8} {'Raw p':>10} {'Part r':>8} {'Part p':>10} {'Survives':>8}")
    print("  " + "-" * 60)

    for l in target_layers:
        l_records = []
        for qid, entry in all_data.items():
            runs = entry["runs"]
            expected = entry["question"].get("answer", "").lower().strip()
            ag = compute_answer_agreement(runs, expected)
            if np.isnan(ag):
                continue
            vectors = get_run_vectors(runs, step, l)
            if len(vectors) < 2:
                continue
            sim_l = compute_pairwise_cosine_similarity(vectors)
            if np.isnan(sim_l):
                continue
            cr = sum(1 for r in runs if r.get("correct")) / len(runs)
            l_records.append({"sim": sim_l, "agreement": ag, "accuracy": cr, "difficulty": entry["difficulty"]})

        if len(l_records) < 10:
            continue

        l_sims = np.array([r["sim"] for r in l_records])
        l_ags = np.array([r["agreement"] for r in l_records])
        l_accs = np.array([r["accuracy"] for r in l_records])
        l_diffs = np.array([1 if r["difficulty"] == "hard" else 0 for r in l_records])

        l_raw_r, l_raw_p = stats.pearsonr(l_sims, l_ags)
        l_controls = np.column_stack([l_accs, l_diffs])
        l_pr, l_pp, _, _ = partial_corr(l_sims, l_ags, l_controls)
        survives = "YES" if l_pp < 0.05 else "no"
        print(f"  {l:>5} {l_raw_r:>8.4f} {l_raw_p:>10.6f} {l_pr:>8.4f} {l_pp:>10.6f} {survives:>8}")

        layer_results[l] = {
            "n": len(l_records),
            "raw_r": float(l_raw_r), "raw_p": float(l_raw_p),
            "partial_r": float(l_pr), "partial_p": float(l_pp),
            "survives": bool(l_pp < 0.05),
        }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    hard_mask = np.array([d == "hard" for d in diffs])
    ax.scatter(res_sim_both[hard_mask], res_ag_both[hard_mask], c="#d62728", alpha=0.6, s=40,
               label="Hard", edgecolors="black", linewidth=0.3)
    ax.scatter(res_sim_both[~hard_mask], res_ag_both[~hard_mask], c="#1f77b4", alpha=0.6, s=40,
               label="Easy", edgecolors="black", linewidth=0.3)
    ax.set_xlabel("Similarity (residualized)", fontsize=11)
    ax.set_ylabel("Agreement (residualized)", fontsize=11)
    ax.set_title(f"Partial (| acc + diff): r={pr_both:.3f}, p={pp_both:.4f}", fontsize=12)
    ax.legend(fontsize=9)

    ax = axes[1]
    layers_plot = sorted(layer_results.keys())
    raw_rs = [layer_results[l]["raw_r"] for l in layers_plot]
    part_rs = [layer_results[l]["partial_r"] for l in layers_plot]
    x_pos = np.arange(len(layers_plot))
    ax.bar(x_pos - 0.15, raw_rs, 0.3, color="#1f77b4", label="Raw r", edgecolor="black", linewidth=0.5)
    ax.bar(x_pos + 0.15, part_rs, 0.3, color="#ff7f0e", label="Partial r", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(l) for l in layers_plot], fontsize=9)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Pearson r (Sim vs Agreement)", fontsize=11)
    ax.set_title("Raw vs Partial by Layer (Step 4)", fontsize=12)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.legend(fontsize=9)

    ax = axes[2]
    ax.scatter(cvs, agreements, c="#2ca02c", alpha=0.6, s=40, edgecolors="black", linewidth=0.3)
    cv_ag_r, cv_ag_p = stats.pearsonr(cvs, agreements)
    z = np.polyfit(cvs, agreements, 1)
    x_line = np.linspace(cvs.min(), cvs.max(), 100)
    ax.plot(x_line, np.poly1d(z)(x_line), "k--", alpha=0.5)
    ax.set_xlabel("Behavioral CV", fontsize=11)
    ax.set_ylabel("Answer Agreement", fontsize=11)
    ax.set_title(f"CV vs Agreement: r={cv_ag_r:.3f}, p={cv_ag_p:.4f}", fontsize=12)

    plt.suptitle("Partial Correlation: Similarity vs Answer Agreement, Controlling for Confounds", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "partial_corr_agreement_step4.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: partial_corr_agreement_step4.png")

    result = {
        "step": step, "n": n,
        "raw_r": float(raw_r), "raw_p": float(raw_p),
        "partial_r_accuracy_only": float(pr_acc), "partial_p_accuracy_only": float(pp_acc),
        "partial_r_accuracy_and_difficulty": float(pr_both), "partial_p_accuracy_and_difficulty": float(pp_both),
        "cv_based_comparison": {
            "partial_r_accuracy_only": float(pr_cv_acc), "partial_p_accuracy_only": float(pp_cv_acc),
            "partial_r_accuracy_and_difficulty": float(pr_cv_both), "partial_p_accuracy_and_difficulty": float(pp_cv_both),
        },
        "cv_vs_agreement": {"r": float(cv_ag_r), "p": float(cv_ag_p)},
        "layers": {str(k): v for k, v in layer_results.items()},
    }
    with open(OUTPUT_DIR / "partial_corr_agreement_step4.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: partial_corr_agreement_step4.json")

    return result


def main():
    print("=" * 70)
    print("PAPER 3: NEW ANALYSES (A, B, C)")
    print("=" * 70)

    pilot_data = load_pilot_data()
    easier_data = load_easier_data()
    new60_data = load_new60_data()
    all_data = {**pilot_data, **easier_data, **new60_data}
    print(f"\nTotal questions loaded: {len(all_data)}")

    records, result_a = analysis_answer_agreement(all_data)
    result_b = analysis_monitoring_demo(all_data)
    result_c = analysis_partial_corr_agreement(all_data, records)

    print("\n" + "=" * 70)
    print("ALL NEW ANALYSES COMPLETE")
    print("=" * 70)
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
