"""
Rankings instability analysis for 4 primary models x 200 questions.
Bootstrap single-run sampling to show how model rankings change.
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

BASE = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE / 'hotpotqa'

# 4 primary models, each with two data directories (Q1-100, Q101-200)
MODELS = {
    'Llama 3.1 70B': ['results_llama', 'results_cortex_llama'],
    'Claude Sonnet 4.5': ['results_claude', 'results_cortex_claude'],
    'GPT-5': ['results_gpt5', 'results_cortex_gpt5'],
    'Gemini 3 Pro': ['results_gemini', 'results_cortex_gemini'],
}

COLORS = {
    'Llama 3.1 70B': '#e74c3c',
    'Claude Sonnet 4.5': '#3498db',
    'GPT-5': '#2ecc71',
    'Gemini 3 Pro': '#f39c12',
}

N_BOOTSTRAP = 10000
SEED = 42


def is_correct(pred, gold):
    """Check correctness with punctuation normalization (matches common.analysis)."""
    if not pred or not gold:
        return False
    p = str(pred).strip().lower().rstrip('.,!?')
    g = str(gold).strip().lower().rstrip('.,!?')
    return g in p or p in g


def load_model_data(subdirs):
    """Load data from multiple directories for a model."""
    data = {}
    for subdir in subdirs:
        path = RESULTS_DIR / subdir
        if not path.exists():
            continue
        for fname in sorted(os.listdir(path)):
            if not fname.endswith('.json'):
                continue
            with open(path / fname) as f:
                d = json.load(f)
            qid = d['task_id']
            if qid in data:
                continue  # avoid duplicates across dirs
            gold = d['answer']
            runs = []
            for r in d['runs']:
                if isinstance(r, str):
                    runs.append(False)
                else:
                    runs.append(is_correct(r.get('final_answer', ''), gold))
            data[qid] = runs
    return data


print("Loading data...")
all_data = {}
for model_name, subdirs in MODELS.items():
    all_data[model_name] = load_model_data(subdirs)
    print(f"  {model_name}: {len(all_data[model_name])} questions")

# Find common questions across all 4 models
question_ids = sorted(set.intersection(*[set(d.keys()) for d in all_data.values()]))
n_questions = len(question_ids)
print(f"Common questions: {n_questions}")

model_names = list(MODELS.keys())

# Build correctness matrix: [n_questions x n_runs] per model
correctness = {}
for m in model_names:
    arr = []
    for qid in question_ids:
        arr.append(all_data[m][qid])
    correctness[m] = np.array(arr)

# Multi-run accuracy (ground truth ranking)
multi_run_acc = {m: correctness[m].mean() * 100 for m in model_names}
print(f"\nMulti-run accuracy (mean over all runs):")
for m in sorted(model_names, key=lambda m: -multi_run_acc[m]):
    print(f"  {m}: {multi_run_acc[m]:.1f}%")

multi_run_ranking = sorted(model_names, key=lambda m: -multi_run_acc[m])
print(f"Multi-run ranking: {' > '.join(multi_run_ranking)}")

# Bootstrap: sample 1 run per question per model
rng = np.random.RandomState(SEED)
bootstrap_accs = {m: [] for m in model_names}
bootstrap_rankings = []

for _ in range(N_BOOTSTRAP):
    sample_acc = {}
    for m in model_names:
        n_runs = correctness[m].shape[1]
        chosen = rng.randint(0, n_runs, size=n_questions)
        correct_count = sum(correctness[m][q, chosen[q]] for q in range(n_questions))
        sample_acc[m] = correct_count / n_questions * 100
        bootstrap_accs[m].append(sample_acc[m])
    ranked = sorted(model_names, key=lambda m: -sample_acc[m])
    bootstrap_rankings.append(tuple(ranked))

print(f"\n{'='*60}")
print(f"BOOTSTRAP RESULTS ({N_BOOTSTRAP:,} iterations, {n_questions} questions)")
print(f"{'='*60}")

print(f"\nAccuracy statistics per model:")
for m in model_names:
    vals = np.array(bootstrap_accs[m])
    print(f"  {m}: mean={vals.mean():.1f}%, std={vals.std():.1f}%, "
          f"min={vals.min():.1f}%, max={vals.max():.1f}%, "
          f"range={vals.max()-vals.min():.1f}pp")

rank_counts = {m: Counter() for m in model_names}
for ranking in bootstrap_rankings:
    for pos, m in enumerate(ranking):
        rank_counts[m][pos + 1] += 1

n_ranks = len(model_names)
rank_labels = ['1st', '2nd', '3rd', '4th']
print(f"\nRanking distribution (% of bootstrap samples):")
header = f"  {'Model':<20}" + "".join(f" {r:>8}" for r in rank_labels[:n_ranks])
print(header)
for m in model_names:
    pcts = [rank_counts[m][r] / N_BOOTSTRAP * 100 for r in range(1, n_ranks + 1)]
    row = f"  {m:<20}" + "".join(f" {p:>7.1f}%" for p in pcts)
    print(row)

median_ranking = tuple(multi_run_ranking)
ranking_matches = sum(1 for r in bootstrap_rankings if r == median_ranking)
ranking_differs = N_BOOTSTRAP - ranking_matches
pct_differs = ranking_differs / N_BOOTSTRAP * 100
print(f"\nRanking instability:")
print(f"  Median ranking: {' > '.join(median_ranking)}")
print(f"  Matches median: {ranking_matches}/{N_BOOTSTRAP} ({100-pct_differs:.1f}%)")
print(f"  Differs from median: {ranking_differs}/{N_BOOTSTRAP} ({pct_differs:.1f}%)")

# 95% CI via bootstrap of instability
rng2 = np.random.RandomState(SEED + 1)
instabilities = []
for _ in range(1000):
    sample = rng2.choice(N_BOOTSTRAP, size=N_BOOTSTRAP, replace=True)
    sampled_rankings = [bootstrap_rankings[i] for i in sample]
    inst = sum(1 for r in sampled_rankings if r != median_ranking) / N_BOOTSTRAP * 100
    instabilities.append(inst)
ci_lo, ci_hi = np.percentile(instabilities, [2.5, 97.5])
print(f"  95% CI: [{ci_lo:.1f}%, {ci_hi:.1f}%]")

bottom_model = multi_run_ranking[-1]
bottom_becomes_top = sum(1 for r in bootstrap_rankings if r[0] == bottom_model)
pct_bottom_top = bottom_becomes_top / N_BOOTSTRAP * 100
print(f"  Bottom model ({bottom_model}) becomes top: {bottom_becomes_top}/{N_BOOTSTRAP} ({pct_bottom_top:.1f}%)")

unique_rankings = len(set(bootstrap_rankings))
ranking_counter = Counter(bootstrap_rankings)
print(f"\n  Unique rankings observed: {unique_rankings}")
for ranking, count in ranking_counter.most_common():
    print(f"    {' > '.join(ranking)}: {count} ({count/N_BOOTSTRAP*100:.1f}%)")

# Generate figure
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(model_names))
width = 0.2
rank_colors = ['#2196F3', '#FF9800', '#F44336', '#9C27B0']

for i, rank in enumerate(range(1, n_ranks + 1)):
    offset = (i - n_ranks / 2 + 0.5) * width
    vals = [rank_counts[m][rank] / N_BOOTSTRAP * 100 for m in model_names]
    bars = ax.bar(x + offset, vals, width, label=rank_labels[i],
                  color=rank_colors[i], edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, vals):
        if val > 3:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.0f}%', ha='center', va='bottom', fontsize=7)

ax.set_xlabel('Model', fontsize=11)
ax.set_ylabel('% of Bootstrap Samples', fontsize=11)
ax.set_title(f'Model Ranking Instability Under Single-Run Evaluation\n'
             f'({N_BOOTSTRAP:,} bootstrap iterations, {n_questions} questions, 4 models)',
             fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels([n.replace(" ", "\n") for n in model_names], fontsize=9)
ax.legend(title='Rank', fontsize=9, title_fontsize=9)
ax.set_ylim(0, 105)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), 'rankings_instability.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved to {out_path}")

pdf_path = os.path.join(os.path.dirname(__file__), 'rankings_instability.pdf')
plt.savefig(pdf_path, bbox_inches='tight')
print(f"PDF saved to {pdf_path}")
