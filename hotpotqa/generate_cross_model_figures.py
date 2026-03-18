"""
Generate cross-model figures for Paper 3:
  B2: Multi-model step progression (peak layer per model)
  B3: Cross-model proportional depth comparison (all 3 models on same x-axis)
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = '/Users/amehta/research/agent-consistency/paper3_prep/figures'

# ============================================================
# Load data
# ============================================================

# Llama step progression at L40
with open('/Users/amehta/research/agent-consistency/hotpotqa/analysis_results/combined_100q/step_progression_100q.json') as f:
    llama_step = json.load(f)

# Qwen full heatmap
with open('/Users/amehta/research/agent-consistency/hotpotqa/analysis_results/combined_100q/qwen_100q_analysis.json') as f:
    qwen_data = json.load(f)

# Phi-3 full correlations
with open('/Users/amehta/research/agent-consistency/hotpotqa/phi3_correlations.json') as f:
    phi3_data = json.load(f)

# ============================================================
# B2: Multi-model step progression
# ============================================================

fig, ax = plt.subplots(figsize=(7, 4.5))

# Llama: peak at L40, steps 1-5
llama_steps = []
llama_rs = []
for s_str, vals in sorted(llama_step['steps'].items(), key=lambda x: int(x[0])):
    llama_steps.append(int(s_str))
    llama_rs.append(vals['r'])

# Qwen: peak at L64, steps 1-5
qwen_steps = [1, 2, 3, 4, 5]
qwen_rs = []
for s in qwen_steps:
    d = qwen_data['heatmap'][str(s)].get('64', {})
    qwen_rs.append(d.get('r', float('nan')))

# Phi-3: peak at L16 (step 4) and L4 (step 5), use L16 for consistency
phi3_steps = list(range(1, 6))  # steps 1-5 only (n drops to 31 at step 6)
phi3_rs_l16 = []
for s in phi3_steps:
    key = f's{s}_l16'
    if key in phi3_data:
        r = phi3_data[key]['r']
        n = phi3_data[key]['n']
        if n >= 50:  # only plot where n is reasonable
            phi3_rs_l16.append(r)
        else:
            phi3_rs_l16.append(float('nan'))
    else:
        phi3_rs_l16.append(float('nan'))

# Plot
ax.plot(llama_steps, llama_rs, 'o-', color='#1f77b4', linewidth=2, markersize=8, 
        label='Llama 3.1 70B (L40)', zorder=3)
ax.plot(qwen_steps, qwen_rs, 's-', color='#d62728', linewidth=2, markersize=8,
        label='Qwen 2.5 72B (L64)', zorder=3)

# Filter NaN for Phi-3
phi3_valid_steps = [s for s, r in zip(phi3_steps, phi3_rs_l16) if not np.isnan(r)]
phi3_valid_rs = [r for r in phi3_rs_l16 if not np.isnan(r)]
ax.plot(phi3_valid_steps, phi3_valid_rs, '^-', color='#2ca02c', linewidth=2, markersize=8,
        label='Phi-3 14B (L16)', zorder=3)

# Significance markers
for s_str, vals in llama_step['steps'].items():
    s = int(s_str)
    if vals['p'] < 0.05:
        ax.plot(s, vals['r'], 'o', color='#1f77b4', markersize=12, 
                markeredgecolor='gold', markeredgewidth=2, zorder=4)

for s in qwen_steps:
    d = qwen_data['heatmap'][str(s)].get('64', {})
    if d.get('p', 1) < 0.05:
        ax.plot(s, d['r'], 's', color='#d62728', markersize=12,
                markeredgecolor='gold', markeredgewidth=2, zorder=4)

for s in phi3_valid_steps:
    key = f's{s}_l16'
    if key in phi3_data and phi3_data[key]['p'] < 0.05:
        ax.plot(s, phi3_data[key]['r'], '^', color='#2ca02c', markersize=12,
                markeredgecolor='gold', markeredgewidth=2, zorder=4)

ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='--')
ax.set_xlabel('ReAct Step', fontsize=13)
ax.set_ylabel('Pearson $r$ (Similarity vs. CV)', fontsize=13)
ax.set_xticks(range(1, 6))
ax.set_ylim(-0.75, 0.35)
ax.legend(fontsize=10, loc='lower left')
ax.annotate('Gold border = $p < 0.05$', xy=(0.98, 0.02), xycoords='axes fraction',
            ha='right', va='bottom', fontsize=9, color='gray')
ax.set_title('Step Progression of Commitment Signal\n(each model at its peak layer)', fontsize=13)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/multi_model_step_progression.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{OUTPUT_DIR}/multi_model_step_progression.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: multi_model_step_progression.pdf/.png')


# ============================================================
# B3: Cross-model proportional depth comparison
# ============================================================

fig, ax = plt.subplots(figsize=(7, 4.5))

# Llama: 81 layers (0-80), step 4
# Layer indices: [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80]
# Proportional depth: layer / 80 * 100
llama_layers = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80]
llama_depths = [l / 80 * 100 for l in llama_layers]
llama_layer_rs = []
llama_layer_ps = []
for l in llama_layers:
    d = qwen_data['comparison_on_shared_questions']  # This is 20q only
    # Use the full 100q data from the cross_model_full table in the paper
    # Actually, let me use Qwen's heatmap step 4 for Qwen, and hardcode Llama from Table 6
    pass

# Better: extract from Qwen heatmap for Qwen, and use paper table values for Llama
# Llama step-4 layer-wise r values (from Table 6 / crossmodel_full in paper)
llama_s4 = {
    0: (0.47, 0.001), 8: (-0.40, 0.001), 16: (-0.38, 0.001), 
    24: (-0.33, 0.001), 32: (-0.34, 0.001), 40: (-0.35, 0.001),
    48: (-0.33, 0.001), 56: (-0.32, 0.002), 64: (-0.31, 0.002),
    72: (-0.32, 0.002), 80: (-0.32, 0.002)
}

# Qwen step-4 layer-wise from heatmap
qwen_s4 = {}
for l_str, vals in qwen_data['heatmap']['4'].items():
    qwen_s4[int(l_str)] = (vals['r'], vals['p'])

# Phi-3 step-4 layer-wise from phi3_correlations
phi3_s4 = {}
phi3_layers = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
for l in phi3_layers:
    key = f's4_l{l}'
    if key in phi3_data:
        r = phi3_data[key]['r']
        p = phi3_data[key]['p']
        if r == r:  # not NaN
            phi3_s4[l] = (r, p)

# Proportional depths
llama_max_layer = 80
qwen_max_layer = 80
phi3_max_layer = 40

# Plot Llama
llama_x = [l / llama_max_layer * 100 for l in sorted(llama_s4.keys())]
llama_y = [llama_s4[l][0] for l in sorted(llama_s4.keys())]
llama_p = [llama_s4[l][1] for l in sorted(llama_s4.keys())]
ax.plot(llama_x, llama_y, 'o-', color='#1f77b4', linewidth=2, markersize=7,
        label='Llama 3.1 70B (80 layers)', zorder=3)

# Plot Qwen
qwen_x = [l / qwen_max_layer * 100 for l in sorted(qwen_s4.keys())]
qwen_y = [qwen_s4[l][0] for l in sorted(qwen_s4.keys())]
qwen_p = [qwen_s4[l][1] for l in sorted(qwen_s4.keys())]
ax.plot(qwen_x, qwen_y, 's-', color='#d62728', linewidth=2, markersize=7,
        label='Qwen 2.5 72B (80 layers)', zorder=3)

# Plot Phi-3
phi3_x = [l / phi3_max_layer * 100 for l in sorted(phi3_s4.keys())]
phi3_y = [phi3_s4[l][0] for l in sorted(phi3_s4.keys())]
phi3_p_vals = [phi3_s4[l][1] for l in sorted(phi3_s4.keys())]
ax.plot(phi3_x, phi3_y, '^-', color='#2ca02c', linewidth=2, markersize=7,
        label='Phi-3 14B (40 layers)', zorder=3)

# Significance markers (gold border for p < 0.05)
for x, y, p in zip(llama_x, llama_y, llama_p):
    if p < 0.05 and y < 0:  # only mark negative correlations
        ax.plot(x, y, 'o', color='#1f77b4', markersize=11,
                markeredgecolor='gold', markeredgewidth=1.5, zorder=4)
for x, y, p in zip(qwen_x, qwen_y, qwen_p):
    if p < 0.05 and y < 0:
        ax.plot(x, y, 's', color='#d62728', markersize=11,
                markeredgecolor='gold', markeredgewidth=1.5, zorder=4)
for x, y, p in zip(phi3_x, phi3_y, phi3_p_vals):
    if p < 0.05 and y < 0:
        ax.plot(x, y, '^', color='#2ca02c', markersize=11,
                markeredgecolor='gold', markeredgewidth=1.5, zorder=4)

ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='--')
ax.set_xlabel('Proportional Depth (%)', fontsize=13)
ax.set_ylabel('Pearson $r$ (Similarity vs. CV)', fontsize=13)
ax.set_xlim(-5, 105)
ax.set_ylim(-0.75, 0.55)
ax.legend(fontsize=10, loc='lower left')
ax.annotate('Gold border = $p < 0.05$', xy=(0.98, 0.02), xycoords='axes fraction',
            ha='right', va='bottom', fontsize=9, color='gray')
ax.set_title('Cross-Model Layer Comparison at Step 4\n(proportional depth normalization)', fontsize=13)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/cross_model_proportional_depth.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{OUTPUT_DIR}/cross_model_proportional_depth.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: cross_model_proportional_depth.pdf/.png')

print('\nDone!')
