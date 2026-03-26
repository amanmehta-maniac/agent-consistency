"""
paper3_v3_figures.py — Generate figures for Paper 3 v3 new analyses.

Reads results from analysis_results/v3/paper3_v3_results.json
Outputs figures to ../paper3_prep/figures/

Run from hotpotqa/ directory:
    ../.venv/bin/python3 paper3_v3_figures.py
"""

import json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

BASE = Path(__file__).parent
RESULTS_FILE = BASE / "analysis_results" / "v3" / "paper3_v3_results.json"
FIGURES_DIR = BASE / ".." / "paper3_prep" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# House style
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def load_results():
    with open(RESULTS_FILE) as f:
        return json.load(f)


def fig_task1_prototype(results):
    """Task 1: Prototype similarity scatter (if significant)."""
    t1 = results["task1"]
    if not t1.get("any_significant"):
        print("  Task 1: No significant signals, skipping figure.")
        return

    perm = t1.get("1b_prototype_permutation")
    fig_data = t1.get("1b_figure_data")
    if not perm or perm["perm_p"] >= 0.05:
        print("  Task 1: Prototype not significant, skipping.")
        return

    r = perm["r"]
    p = perm["perm_p"]
    ci = perm["ci"]
    layer = perm["layer"]

    if not fig_data:
        print("  Task 1: No figure data stored, skipping.")
        return

    proto_sims = np.array(fig_data["proto_sims"])
    cvs = np.array(fig_data["cvs"])

    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    ax.scatter(proto_sims, cvs, c="#e74c3c", alpha=0.7, s=40,
               edgecolors="white", linewidths=0.5, zorder=3)

    # Regression line
    z = np.polyfit(proto_sims, cvs, 1)
    x_line = np.linspace(proto_sims.min(), proto_sims.max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), "--", color="#34495e",
            linewidth=1.5, alpha=0.7)

    ax.text(0.03, 0.97,
            f"r = {r:.3f}\nperm. p = {p:.4f}\n95% CI [{ci[0]:.2f}, {ci[1]:.2f}]",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_xlabel(f"Prototype similarity (layer {layer})")
    ax.set_ylabel("Behavioral CV")
    ax.set_title("Hard Questions: Prototype Distance Signal")

    path = FIGURES_DIR / "task1_prototype_signal.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig_task2_auroc_vs_k(results):
    """Task 2a: AUROC vs k curve."""
    t2 = results["task2"]
    k_data = t2["2a_k_auroc"]

    ks = sorted([int(k) for k in k_data.keys()])
    means = [k_data[str(k)]["mean"] for k in ks]
    stds = [k_data[str(k)]["std"] for k in ks]
    ci_los = [k_data[str(k)]["ci_lo"] for k in ks]
    ci_his = [k_data[str(k)]["ci_hi"] for k in ks]

    fig, ax = plt.subplots(figsize=(4, 3))

    # Shaded CI
    ax.fill_between(ks, ci_los, ci_his, alpha=0.2, color="#2980b9")
    ax.plot(ks, means, "o-", color="#2980b9", linewidth=2, markersize=6, zorder=3)

    # Reference lines
    ax.axhline(y=0.80, color="#e74c3c", linestyle="--", linewidth=1, alpha=0.7, label="AUROC = 0.80")
    ax.axhline(y=0.97, color="#27ae60", linestyle=":", linewidth=1, alpha=0.7, label="k=10 baseline")

    # Annotations
    for i, k in enumerate(ks):
        ax.annotate(f"{means[i]:.2f}", (k, means[i]),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8)

    ax.set_xlabel("Number of runs (k)")
    ax.set_ylabel("AUROC (quintile labeling)")
    ax.set_xticks(ks)
    ax.set_ylim(0.5, 1.02)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_title("Consistency Prediction AUROC vs. Run Budget")

    path = FIGURES_DIR / "auroc_vs_k.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig_task2_pr_k3(results):
    """Task 2b: Precision-recall at k=3."""
    t2 = results["task2"]
    pr = t2["2b_pr_k3"]

    # We only have the first 20 points of the PR curve stored
    # Generate a simplified version
    precision = pr["precision_curve"]
    recall = pr["recall_curve"]

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(recall, precision, "o-", color="#8e44ad", linewidth=2, markersize=4)

    # Mark operating point
    op_p = pr["precision_at_op"]
    op_r = pr["recall_at_op"]
    ax.plot(op_r, op_p, "*", color="#e74c3c", markersize=14, zorder=5,
            label=f"Operating point\n(P={op_p:.2f}, R={op_r:.2f})")

    # Reference lines
    ax.axhline(y=0.80, color="#e74c3c", linestyle="--", linewidth=1, alpha=0.5)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", framealpha=0.9)
    ax.set_title(f"Precision-Recall at k=3 (AUROC={pr['auroc']:.2f})")

    path = FIGURES_DIR / "pr_curve_k3.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig_task3_projection_scatter(results):
    """Task 3c/e: Scatter of projection onto v_commit vs CV."""
    t3 = results["task3"]
    fig_data = t3["3c_figure_data"]

    projections = np.array(fig_data["projections"])
    cvs = np.array(fig_data["cvs"])
    difficulties = fig_data["difficulties"]

    easy_mask = np.array([d == "easy" for d in difficulties])
    hard_mask = np.array([d == "hard" for d in difficulties])

    fig, ax = plt.subplots(figsize=(5, 4))

    # Plot easy and hard separately
    ax.scatter(projections[easy_mask], cvs[easy_mask],
               c="#2ecc71", alpha=0.7, s=40, edgecolors="white", linewidths=0.5,
               label="Easy", zorder=3)
    ax.scatter(projections[hard_mask], cvs[hard_mask],
               c="#e74c3c", alpha=0.7, s=40, edgecolors="white", linewidths=0.5,
               label="Hard", zorder=3)

    # Regression line (all)
    z = np.polyfit(projections, cvs, 1)
    x_line = np.linspace(projections.min(), projections.max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), "--", color="#34495e", linewidth=1.5, alpha=0.7)

    # Annotation
    proj = t3["3c_projection"]
    r_all = proj["r_all"]
    p_all = proj["p_all"]
    ax.text(0.03, 0.97, f"r = {r_all:.3f}, p = {p_all:.4f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_xlabel("Projection onto commitment direction")
    ax.set_ylabel("Behavioral CV")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_title("Commitment Direction Predicts Consistency")

    path = FIGURES_DIR / "commitment_projection_scatter.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    print("Generating Paper 3 v3 figures...")
    results = load_results()

    fig_task1_prototype(results)
    fig_task2_auroc_vs_k(results)
    fig_task2_pr_k3(results)
    fig_task3_projection_scatter(results)

    print("\nDone.")


if __name__ == "__main__":
    main()
