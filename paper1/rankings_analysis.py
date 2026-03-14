import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'hotpotqa')
MODELS = {
    'Claude Sonnet 4.5': 'results_claude',
    'GPT-4o': 'results_gpt4o',
    'Llama 3.1 70B': 'results_llama',
}
N_BOOTSTRAP = 10000
SEED = 42

def is_correct(pred, gold):
    if not pred or not gold:
        return False
    p, g = pred.strip().lower(), gold.strip().lower()
    return g in p or p in g

def load_model_data(subdir):
    path = os.path.join(RESULTS_DIR, subdir)
    data = {}
    for fname in sorted(os.listdir(path)):
        if not fname.endswith('.json'):
            continue
        with open(os.path.join(path, fname)) as f:
            d = json.load(f)
        qid = d['task_id']
        gold = d['answer']
        runs = []
        for r in d['runs']:
            runs.append(is_correct(r.get('final_answer', ''), gold))
        data[qid] = runs
    return data

print("Loading data...")
all_data = {}
for model_name, subdir in MODELS.items():
    all_data[model_name] = load_model_data(subdir)
    print(f"  {model_name}: {len(all_data[model_name])} questions")

question_ids = sorted(set.intersection(*[set(d.keys()) for d in all_data.values()]))
n_questions = len(question_ids)
print(f"Common questions: {n_questions}")

model_names = list(MODELS.keys())
correctness = {}
for m in model_names:
    arr = []
    for qid in question_ids:
        arr.append(all_data[m][qid])
    correctness[m] = np.array(arr)

multi_run_acc = {m: correctness[m].mean() * 100 for m in model_names}
print(f"\nMulti-run accuracy (mean over all runs):")
for m in model_names:
    print(f"  {m}: {multi_run_acc[m]:.1f}%")

multi_run_ranking = sorted(model_names, key=lambda m: -multi_run_acc[m])
print(f"Multi-run ranking: {' > '.join(multi_run_ranking)}")

rng = np.random.RandomState(SEED)
bootstrap_accs = {m: [] for m in model_names}
bootstrap_rankings = []

for _ in range(N_BOOTSTRAP):
    sample_acc = {}
    for m in model_names:
        chosen = rng.randint(0, correctness[m].shape[1], size=n_questions)
        correct_count = sum(correctness[m][q, chosen[q]] for q in range(n_questions))
        sample_acc[m] = correct_count / n_questions * 100
        bootstrap_accs[m].append(sample_acc[m])
    ranked = sorted(model_names, key=lambda m: -sample_acc[m])
    bootstrap_rankings.append(tuple(ranked))

print(f"\n{'='*60}")
print(f"BOOTSTRAP RESULTS ({N_BOOTSTRAP:,} iterations)")
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

print(f"\nRanking distribution (% of bootstrap samples):")
print(f"  {'Model':<20} {'1st':>8} {'2nd':>8} {'3rd':>8}")
for m in model_names:
    pcts = [rank_counts[m][r] / N_BOOTSTRAP * 100 for r in [1, 2, 3]]
    print(f"  {m:<20} {pcts[0]:>7.1f}% {pcts[1]:>7.1f}% {pcts[2]:>7.1f}%")

median_ranking = tuple(multi_run_ranking)
ranking_matches = sum(1 for r in bootstrap_rankings if r == median_ranking)
ranking_differs = N_BOOTSTRAP - ranking_matches
pct_differs = ranking_differs / N_BOOTSTRAP * 100
print(f"\nRanking instability:")
print(f"  Median ranking: {' > '.join(median_ranking)}")
print(f"  Matches median: {ranking_matches}/{N_BOOTSTRAP} ({100-pct_differs:.1f}%)")
print(f"  Differs from median: {ranking_differs}/{N_BOOTSTRAP} ({pct_differs:.1f}%)")

bottom_model = multi_run_ranking[-1]
bottom_becomes_top = sum(1 for r in bootstrap_rankings if r[0] == bottom_model)
pct_bottom_top = bottom_becomes_top / N_BOOTSTRAP * 100
print(f"  Bottom model ({bottom_model}) becomes top: {bottom_becomes_top}/{N_BOOTSTRAP} ({pct_bottom_top:.1f}%)")

unique_rankings = len(set(bootstrap_rankings))
ranking_counter = Counter(bootstrap_rankings)
print(f"\n  Unique rankings observed: {unique_rankings}")
for ranking, count in ranking_counter.most_common():
    print(f"    {' > '.join(ranking)}: {count} ({count/N_BOOTSTRAP*100:.1f}%)")

fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(model_names))
width = 0.25
colors = ['#2196F3', '#FF9800', '#F44336']
labels = ['1st', '2nd', '3rd']

for i, rank in enumerate([1, 2, 3]):
    vals = [rank_counts[m][rank] / N_BOOTSTRAP * 100 for m in model_names]
    bars = ax.bar(x + (i - 1) * width, vals, width, label=labels[i], color=colors[i], edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, vals):
        if val > 3:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.0f}%', ha='center', va='bottom', fontsize=8)

ax.set_xlabel('Model', fontsize=11)
ax.set_ylabel('% of Bootstrap Samples', fontsize=11)
ax.set_title('Model Ranking Instability Under Single-Run Evaluation\n(10,000 bootstrap iterations, 100 questions)', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=10)
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
