"""
Generate publication-quality research paper plots for 3-Tier Roofline Analysis.
Reads directly from results/multi_batch_consolidated.csv (single source of truth).
"""

import os
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from hardware_config import (
    POWER_MODES, MODE_15W, MODE_25W, MODE_MAXN,
    DEFAULT_PEAK_COMPUTE, DEFAULT_PEAK_BANDWIDTH, DEFAULT_RIDGE_POINT,
    RIDGE_15W, RIDGE_25W, RIDGE_MAXN,
)

# Read canonical multi-batch consolidated data
csv_paths = ['results/multi_batch_consolidated.csv', 'jetson/results/multi_batch_consolidated.csv']
csv_path = None
for p in csv_paths:
    if os.path.exists(p):
        csv_path = p
        break

if not csv_path:
    raise FileNotFoundError("Missing multi_batch_consolidated.csv in results/ or jetson/results/. Run generate_multi_batch_analysis.py first.")

def save_plot(fig, filename):
    for d in ['results', 'jetson/results']:
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, filename)
        fig.savefig(path, dpi=200, bbox_inches='tight')
        print(f"Saved: {path}")

results = []
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        row['batch_size'] = int(row['batch_size'])
        row['arithmetic_intensity'] = float(row['arithmetic_intensity'])
        row['compute_ratio_pct'] = float(row['compute_ratio_pct'])
        row['total_gflops'] = float(row['total_gflops'])
        row['total_memory_mb'] = float(row['total_memory_mb'])
        results.append(row)

model_names = ['MobileNet-v3', 'LSTM', 'YOLOv8n', 'BERT-Base', 'ResNet-50']
batch_sizes = [1, 2, 4, 8, 16, 32]

markers = {
    'MobileNet-v3': 'o',
    'LSTM':         's',
    'YOLOv8n':      '^',
    'BERT-Base':    'D',
    'ResNet-50':    'v',
}

line_styles = {
    'MobileNet-v3': '-',
    'LSTM':         '--',
    'YOLOv8n':      '-.',
    'BERT-Base':    ':',
    'ResNet-50':    '-',
}

os.makedirs('results', exist_ok=True)


# ---- Fig 1: AI vs Batch Size with 3-Tier Thresholds ----
fig, ax = plt.subplots(figsize=(6.5, 4.2))

for name in model_names:
    bs_list = []
    ai_list = []
    for r in results:
        if r['model'] == name:
            bs_list.append(r['batch_size'])
            ai_list.append(r['arithmetic_intensity'])
    ax.plot(bs_list, ai_list, line_styles[name], marker=markers[name],
            markersize=6, linewidth=1.5, label=name)

ax.axhline(y=RIDGE_15W, color='green', linestyle=':', linewidth=1.2,
           label=f'15W Ridge ({RIDGE_15W:.2f})')
ax.axhline(y=RIDGE_25W, color='orange', linestyle='-.', linewidth=1.0,
           label=f'25W Ridge ({RIDGE_25W:.2f})')
ax.axhline(y=RIDGE_MAXN, color='red', linestyle='--', linewidth=1.2,
           label=f'MAXN Ridge ({RIDGE_MAXN:.2f})')

ax.set_xlabel('Batch Size', fontweight='bold')
ax.set_ylabel('Arithmetic Intensity (FLOP/Byte)', fontweight='bold')
ax.set_title('(a) Arithmetic Intensity vs Batch Size (3-Tier Roofline)', fontweight='bold')
ax.set_xscale('log', base=2)
ax.set_xticks(batch_sizes)
ax.set_xticklabels([str(b) for b in batch_sizes])
ax.set_ylim(0, 80)
ax.legend(fontsize=7, loc='upper left')
ax.grid(True, alpha=0.3, linestyle=':')
fig.tight_layout()
save_plot(fig, 'fig_ai_vs_batch.png')
plt.close(fig)


# ---- Fig 2: Compute Ratio vs Batch Size ----
fig, ax = plt.subplots(figsize=(6.5, 4.2))

for name in model_names:
    bs_list = []
    cr_list = []
    for r in results:
        if r['model'] == name:
            bs_list.append(r['batch_size'])
            cr_list.append(r['compute_ratio_pct'])
    ax.plot(bs_list, cr_list, line_styles[name], marker=markers[name],
            markersize=6, linewidth=1.5, label=name)

ax.set_xlabel('Batch Size', fontweight='bold')
ax.set_ylabel('Compute Ratio (%)', fontweight='bold')
ax.set_title('(b) Compute-Bound FLOPs Ratio vs Batch Size', fontweight='bold')
ax.set_xscale('log', base=2)
ax.set_xticks(batch_sizes)
ax.set_xticklabels([str(b) for b in batch_sizes])
ax.set_ylim(0, 105)
ax.legend(fontsize=7, loc='center right')
ax.grid(True, alpha=0.3, linestyle=':')
fig.tight_layout()
save_plot(fig, 'fig_compute_ratio.png')
plt.close(fig)


# ---- Fig 3: Multi-Tier Roofline Plot (BS=1) ----
fig, ax = plt.subplots(figsize=(6.5, 4.2))

peak_compute_maxn = POWER_MODES[MODE_MAXN]['peak_compute']
peak_bw_maxn = POWER_MODES[MODE_MAXN]['peak_bandwidth']
peak_compute_25w = POWER_MODES[MODE_25W]['peak_compute']
peak_bw_25w = POWER_MODES[MODE_25W]['peak_bandwidth']
peak_compute_15w = POWER_MODES[MODE_15W]['peak_compute']
peak_bw_15w = POWER_MODES[MODE_15W]['peak_bandwidth']

ai_range = np.logspace(-1, 2.5, 500)
c_maxn = np.minimum(peak_bw_maxn * ai_range, peak_compute_maxn)
c_25w = np.minimum(peak_bw_25w * ai_range, peak_compute_25w)
c_15w = np.minimum(peak_bw_15w * ai_range, peak_compute_15w)

ax.plot(ai_range, c_maxn, 'r-', linewidth=1.5, label=f'MAXN Ceiling ({peak_compute_maxn:.0f} GF/s)')
ax.plot(ai_range, c_25w, 'orange', linestyle='-.', linewidth=1.2, label=f'25W Ceiling ({peak_compute_25w:.0f} GF/s)')
ax.plot(ai_range, c_15w, 'g--', linewidth=1.2, label=f'15W Ceiling ({peak_compute_15w:.0f} GF/s)')

ax.axvline(x=RIDGE_MAXN, color='red', linestyle=':', linewidth=0.8, alpha=0.6)
ax.axvline(x=RIDGE_15W, color='green', linestyle=':', linewidth=0.8, alpha=0.6)

for r in results:
    if r['batch_size'] == 1:
        ai = r['arithmetic_intensity']
        attainable = min(peak_bw_maxn * ai, peak_compute_maxn)
        ax.scatter(ai, attainable * 0.5, marker=markers[r['model']],
                   s=80, edgecolors='black', facecolors='white', linewidths=1.2,
                   zorder=5, label=f"{r['model']} (AI={ai:.1f})")

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Arithmetic Intensity (FLOP/Byte)', fontweight='bold')
ax.set_ylabel('Attainable Performance (GFLOP/s)', fontweight='bold')
ax.set_title('(c) Multi-Tier Roofline Ceilings (Jetson Orin Nano Super, BS=1)', fontweight='bold')
ax.set_xlim(0.5, 200)
ax.set_ylim(50, 10000)
ax.legend(fontsize=6.5, loc='lower right')
ax.grid(True, alpha=0.3, linestyle=':', which='both')
fig.tight_layout()
save_plot(fig, 'fig_roofline_bs1.png')
plt.close(fig)


# ---- Fig 4: 3-Tier Power Mode Recommendation Grid ----
fig, ax = plt.subplots(figsize=(6.5, 3.2))
ax.axis('off')

table_data = []
for name in model_names:
    row_data = []
    for bs in batch_sizes:
        for r in results:
            if r['model'] == name and int(r['batch_size']) == bs:
                row_data.append(r['recommended_mode'])
                break
    table_data.append(row_data)

col_labels = [f'BS={b}' for b in batch_sizes]
table = ax.table(cellText=table_data, rowLabels=model_names,
                 colLabels=col_labels, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.0, 1.4)

for i, name in enumerate(model_names):
    for j, bs in enumerate(batch_sizes):
        cell = table[i + 1, j]
        mode = table_data[i][j]
        if mode == '15W':
            cell.set_facecolor('#c6f6d5')  # Light green for 15W
        elif mode == '25W':
            cell.set_facecolor('#fefcbf')  # Light yellow for 25W
        elif 'MAXN' in mode:
            cell.set_facecolor('#fed7d7')  # Light red for MAXN_SUPER

ax.set_title('(d) Pagoda 3-Tier DVFS Selection per Model and Batch Size', fontsize=10, pad=15, fontweight='bold')
fig.tight_layout()
save_plot(fig, 'fig_power_modes.png')
plt.close(fig)

print("\nAll paper plots generated successfully from canonical multi_batch_consolidated.csv.")
