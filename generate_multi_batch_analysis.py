"""
Multi-Batch Analysis (3-Tier DVFS)
----------------------------------
Runs all 5 models at batch sizes 1, 2, 4, 8, 16, 32.
Saves comprehensive results to results/multi_batch_consolidated.csv.
Generates 4 publication-quality charts using matplotlib for 15W, 25W, and MAXN_SUPER.
"""
import os
import sys
import io
import csv

if __name__ == '__main__' and sys.platform == 'win32' and hasattr(sys.stdout, 'buffer'):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

from power_mode_selector import run_selector
from hardware_config import (
    POWER_MODES, MODE_15W, MODE_25W, MODE_MAXN,
    DEFAULT_PEAK_COMPUTE, DEFAULT_PEAK_BANDWIDTH, DEFAULT_RIDGE_POINT,
    RIDGE_15W, RIDGE_25W, RIDGE_MAXN,
)

# Test models
MODELS = [
    {'name': 'MobileNet-v3',  'key': 'mobnet',  'path': 'onnx_models/mobnet.onnx',     'domain': 'Edge Vision (CNN)'},
    {'name': 'LSTM',          'key': 'lstm',    'path': 'onnx_models/lstm.onnx',       'domain': 'Sequential (RNN)'},
    {'name': 'YOLOv8n',       'key': 'yolo',    'path': 'onnx_models/yolov8n.onnx',    'domain': 'Object Detection'},
    {'name': 'BERT-Base',     'key': 'bert',    'path': 'onnx_models/bert-base.onnx',  'domain': 'NLP (Transformer)'},
    {'name': 'ResNet-50',     'key': 'resnet',  'path': 'onnx_models/resnet50.onnx',   'domain': 'Vision (CNN)'},
]

BATCH_SIZES = [1, 2, 4, 8, 16, 32]

os.makedirs('results', exist_ok=True)

# ===== Step 1: Run all models at all batch sizes =====
all_results = []
print("=" * 75)
print("  PAGODA: MULTI-BATCH ANALYTICAL EVALUATION (3-TIER ROOFLINE)")
print(f"  Ridge Points: 15W={RIDGE_15W:.2f} | 25W={RIDGE_25W:.2f} | MAXN={RIDGE_MAXN:.2f} FLOP/Byte")
print("=" * 75)

for bs in BATCH_SIZES:
    for m in MODELS:
        if not os.path.exists(m['path']):
            continue
        rec = run_selector(
            model_path=m['path'],
            model_key=m['key'],
            batch_size=bs,
            quiet=True,
        )
        rec['display_name'] = m['name']
        rec['domain'] = m['domain']
        all_results.append(rec)
        print(f"  [{m['name']:<12} BS={bs:<2}] AI={rec['total_ai']:<6.2f} "
              f"CR={rec['compute_ratio']*100:<5.1f}%  -> {rec['recommended_mode']:<11} "
              f"({rec['expected_saving']})")

# ===== Step 2: Load actual Jetson hardware results if available =====
hw_data = {}
hw_csv_candidates = [
    'jetson/results/jetson_hardware_results.csv',
    'results/jetson_hardware_results.csv',
]
hw_csv_path = next((p for p in hw_csv_candidates if os.path.exists(p)), None)
if hw_csv_path:
    model_name_map = {
        'mobnet.onnx': 'MobileNet-v3',
        'lstm.onnx': 'LSTM',
        'yolov8n.onnx': 'YOLOv8n',
        'bert-base.onnx': 'BERT-Base',
        'resnet50.onnx': 'ResNet-50',
    }
    with open(hw_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mname = model_name_map.get(row['model'], row['model'])
            bs = int(row['batch_size'])
            pmode = row['power_mode']
            hw_data[(mname, bs, pmode)] = {
                'latency_ms': float(row['avg_latency_ms']),
                'power_w': float(row['avg_power_w']),
                'energy_mj': float(row['energy_mj']),
            }

def get_hw_summary(model_name, bs, pred_mode, ai, ridge_maxn):
    if (model_name, bs, 'MAXN_SUPER') not in hw_data:
        return ('N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A')

    modes = ['15W', '25W', 'MAXN_SUPER']
    avail = {m: hw_data[(model_name, bs, m)] for m in modes if (model_name, bs, m) in hw_data}
    lowest_energy_mode = min(avail.keys(), key=lambda m: avail[m]['energy_mj']) if avail else '15W'

    hw_15 = hw_data.get((model_name, bs, '15W'))
    hw_25 = hw_data.get((model_name, bs, '25W'))
    hw_mx = hw_data.get((model_name, bs, 'MAXN_SUPER'))
    sel_hw = hw_data.get((model_name, bs, pred_mode), hw_mx)

    e_save = ((hw_mx['energy_mj'] - sel_hw['energy_mj']) / hw_mx['energy_mj'] * 100.0) if hw_mx else 0.0
    lat_delta = ((sel_hw['latency_ms'] - hw_mx['latency_ms']) / hw_mx['latency_ms'] * 100.0) if hw_mx else 0.0

    if pred_mode == 'MAXN_SUPER':
        verdict = 'PRESERVE_PERF'
    elif e_save > 10.0 and lat_delta <= 15.0:
        verdict = 'SAVED (SLA Met)'
    elif e_save > 0.0:
        verdict = f'ENERGY_SAVED (+{e_save:.1f}%)'
    else:
        verdict = 'SUBOPTIMAL'

    return (
        lowest_energy_mode,
        round(hw_15['latency_ms'], 2) if hw_15 else 'N/A',
        round(hw_15['energy_mj'], 1) if hw_15 else 'N/A',
        round(hw_25['latency_ms'], 2) if hw_25 else 'N/A',
        round(hw_25['energy_mj'], 1) if hw_25 else 'N/A',
        round(hw_mx['latency_ms'], 2) if hw_mx else 'N/A',
        round(hw_mx['energy_mj'], 1) if hw_mx else 'N/A',
        round(sel_hw['energy_mj'], 1),
        round(e_save, 1),
        verdict
    )

# ===== Step 3: Save everything to consolidated CSV (Both results/ and jetson/results/) =====
csv_paths = ['results/multi_batch_consolidated.csv', 'jetson/results/multi_batch_consolidated.csv']
for csv_path in csv_paths:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'model', 'domain', 'batch_size', 'total_gflops', 'total_memory_mb',
            'arithmetic_intensity',
            'ridge_15w', 'ridge_25w', 'ridge_maxn',
            'regime_15w', 'regime_25w', 'regime_maxn',
            'compute_ratio_pct',
            'recommended_mode', 'expected_saving',
            'optimal_hw_mode',
            'hw_lat_15w_ms', 'hw_energy_15w_mj',
            'hw_lat_25w_ms', 'hw_energy_25w_mj',
            'hw_lat_maxn_ms', 'hw_energy_maxn_mj',
            'selected_hw_energy_mj', 'hw_energy_saved_pct', 'hw_verdict',
            'num_layers', 'num_compute_bound', 'num_memory_bound'
        ])
        for r in all_results:
            (opt_mode, l15, e15, l25, e25, lmx, emx, sel_e, esave, verdict) = get_hw_summary(
                r['display_name'], r['batch_size'], r['recommended_mode'], r['total_ai'], r['ridge_maxn']
            )
            writer.writerow([
                r['display_name'], r['domain'], r['batch_size'],
                round(r['total_flops'] / 1e9, 3),
                round(r['total_memory'] / 1e6, 2),
                round(r['total_ai'], 2),
                round(r['ridge_15w'], 2),
                round(r['ridge_25w'], 2),
                round(r['ridge_maxn'], 2),
                r['regime_15w'],
                r['regime_25w'],
                r['regime_maxn'],
                round(r['compute_ratio'] * 100, 1),
                r['recommended_mode'],
                r['expected_saving'],
                opt_mode,
                l15, e15, l25, e25, lmx, emx, sel_e, esave, verdict,
                r['num_layers'], r['num_compute_bound'], r['num_memory_bound']
            ])
    print(f"CSV saved to: {csv_path}")

# ===== Step 3: Generate Publication Charts =====
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    def save_multi_chart(fig, filename):
        for d in ['results', 'jetson/results']:
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, filename)
            fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
            print(f"Chart saved: {path}")

    model_names = [m['name'] for m in MODELS]
    colors = {
        'MobileNet-v3': '#E53E3E',
        'LSTM': '#3182CE',
        'YOLOv8n': '#38A169',
        'BERT-Base': '#D69E2E',
        'ResNet-50': '#805AD5',
    }
    markers = {
        'MobileNet-v3': 'o',
        'LSTM': 's',
        'YOLOv8n': '^',
        'BERT-Base': 'D',
        'ResNet-50': 'v',
    }

    # ------ Chart 1: AI vs Batch Size with All 3 Ridge Thresholds ------
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    for model_name in model_names:
        bs_vals = []
        ai_vals = []
        for r in all_results:
            if r['display_name'] == model_name:
                bs_vals.append(r['batch_size'])
                ai_vals.append(r['total_ai'])
        ax.plot(bs_vals, ai_vals, '-', color=colors[model_name],
                marker=markers[model_name], markersize=8, linewidth=2.5,
                label=model_name, zorder=3)

    # Draw all 3 ridge point lines
    ax.axhline(y=RIDGE_15W, color='#48BB78', linestyle=':', linewidth=1.8,
               label=f'15W Ridge ({RIDGE_15W:.2f} FLOP/B)', zorder=2)
    ax.axhline(y=RIDGE_25W, color='#ECC94B', linestyle='-.', linewidth=1.5,
               label=f'25W Ridge ({RIDGE_25W:.2f} FLOP/B)', zorder=2)
    ax.axhline(y=RIDGE_MAXN, color='#FF6B6B', linestyle='--', linewidth=2.0,
               label=f'MAXN_SUPER Ridge ({RIDGE_MAXN:.2f} FLOP/B)', zorder=2)

    ax.fill_between([0.5, 40], 0, RIDGE_15W, alpha=0.08, color='#48BB78')
    ax.fill_between([0.5, 40], RIDGE_15W, RIDGE_MAXN, alpha=0.06, color='#ECC94B')
    ax.fill_between([0.5, 40], RIDGE_MAXN, 80, alpha=0.08, color='#FF6B6B')

    ax.text(34, RIDGE_15W - 4, '15W MEM-BOUND\nZONE', fontsize=7.5, color='#48BB78',
            ha='right', alpha=0.8, fontweight='bold')
    ax.text(34, 19.3, '25W TRANSITION\nZONE', fontsize=7, color='#ECC94B',
            ha='right', alpha=0.8, fontweight='bold')
    ax.text(34, RIDGE_MAXN + 4, 'MAXN COMPUTE-BOUND\nZONE', fontsize=7.5, color='#FF6B6B',
            ha='right', alpha=0.8, fontweight='bold')

    ax.set_xlabel('Batch Size', fontsize=12, color='white', fontweight='bold')
    ax.set_ylabel('Arithmetic Intensity (FLOP/Byte)', fontsize=12, color='white', fontweight='bold')
    ax.set_title('AI vs Batch Size — Jetson Orin Nano Super (3-Tier Roofline)',
                 fontsize=14, color='white', fontweight='bold', pad=15)
    ax.set_xscale('log', base=2)
    ax.set_xticks(BATCH_SIZES)
    ax.set_xticklabels([str(b) for b in BATCH_SIZES])
    ax.set_ylim(0, 80)
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('#4a5568')
    ax.spines['left'].set_color('#4a5568')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.15, color='white')
    ax.legend(loc='upper left', fontsize=8.5, facecolor='#1a1a2e', edgecolor='#4a5568',
              labelcolor='white')

    fig.tight_layout()
    save_multi_chart(fig, 'ai_vs_batch_size.png')
    plt.close(fig)

    # ------ Chart 2: Compute Ratio vs Batch Size ------
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    fig2.patch.set_facecolor('#1a1a2e')
    ax2.set_facecolor('#16213e')

    for model_name in model_names:
        bs_vals = []
        cr_vals = []
        for r in all_results:
            if r['display_name'] == model_name:
                bs_vals.append(r['batch_size'])
                cr_vals.append(r['compute_ratio'] * 100)
        ax2.plot(bs_vals, cr_vals, '-', color=colors[model_name],
                marker=markers[model_name], markersize=8, linewidth=2.5,
                label=model_name, zorder=3)

    ax2.set_xlabel('Batch Size', fontsize=12, color='white', fontweight='bold')
    ax2.set_ylabel('Compute Ratio (%)', fontsize=12, color='white', fontweight='bold')
    ax2.set_title('Compute Ratio vs Batch Size — 3-Tier DVFS Selection',
                 fontsize=14, color='white', fontweight='bold', pad=15)
    ax2.set_xscale('log', base=2)
    ax2.set_xticks(BATCH_SIZES)
    ax2.set_xticklabels([str(b) for b in BATCH_SIZES])
    ax2.set_ylim(0, 105)
    ax2.tick_params(colors='white')
    ax2.spines['bottom'].set_color('#4a5568')
    ax2.spines['left'].set_color('#4a5568')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, alpha=0.15, color='white')
    ax2.legend(loc='center right', fontsize=8.5, facecolor='#1a1a2e', edgecolor='#4a5568',
              labelcolor='white')

    fig2.tight_layout()
    save_multi_chart(fig2, 'compute_ratio_vs_batch_size.png')
    plt.close(fig2)

    # ------ Chart 3: Roofline Plot for Batch Size 1 (All 3 Ceilings) ------
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    fig3.patch.set_facecolor('#1a1a2e')
    ax3.set_facecolor('#16213e')

    peak_compute_maxn = POWER_MODES[MODE_MAXN]['peak_compute']
    peak_bw_maxn = POWER_MODES[MODE_MAXN]['peak_bandwidth']

    peak_compute_25w = POWER_MODES[MODE_25W]['peak_compute']
    peak_bw_25w = POWER_MODES[MODE_25W]['peak_bandwidth']

    peak_compute_15w = POWER_MODES[MODE_15W]['peak_compute']
    peak_bw_15w = POWER_MODES[MODE_15W]['peak_bandwidth']

    ai_range = np.logspace(-1, 2.5, 500)
    ceiling_maxn = np.minimum(peak_bw_maxn * ai_range, peak_compute_maxn)
    ceiling_25w = np.minimum(peak_bw_25w * ai_range, peak_compute_25w)
    ceiling_15w = np.minimum(peak_bw_15w * ai_range, peak_compute_15w)

    ax3.plot(ai_range, ceiling_maxn, color='#FC8181', linewidth=2.5,
             label=f'MAXN_SUPER Ceiling ({peak_compute_maxn:.0f} GF/s, {peak_bw_maxn:.0f} GB/s)', zorder=2)
    ax3.plot(ai_range, ceiling_25w, color='#ECC94B', linestyle='-.', linewidth=2.0,
             label=f'25W Ceiling ({peak_compute_25w:.0f} GF/s, {peak_bw_25w:.0f} GB/s)', zorder=2)
    ax3.plot(ai_range, ceiling_15w, color='#48BB78', linestyle='--', linewidth=2.0,
             label=f'15W Ceiling ({peak_compute_15w:.0f} GF/s, {peak_bw_15w:.0f} GB/s)', zorder=2)

    ax3.axvline(x=RIDGE_MAXN, color='#FC8181', linestyle=':', linewidth=1, alpha=0.7)
    ax3.axvline(x=RIDGE_15W, color='#48BB78', linestyle=':', linewidth=1, alpha=0.7)

    for r in all_results:
        if r['batch_size'] == 1:
            ai = r['total_ai']
            attainable = min(peak_bw_maxn * ai, peak_compute_maxn)
            ax3.scatter(ai, attainable * 0.5, color=colors[r['display_name']],
                       marker=markers[r['display_name']], s=150, zorder=5,
                       edgecolors='white', linewidths=1.5,
                       label=f"{r['display_name']} (AI={ai:.1f})")

    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlabel('Arithmetic Intensity (FLOP/Byte)', fontsize=12, color='white', fontweight='bold')
    ax3.set_ylabel('Attainable Performance (GFLOP/s)', fontsize=12, color='white', fontweight='bold')
    ax3.set_title('Multi-Tier Roofline Ceilings — Jetson Orin Nano Super (BS=1)',
                 fontsize=14, color='white', fontweight='bold', pad=15)
    ax3.set_xlim(0.5, 200)
    ax3.set_ylim(50, 10000)
    ax3.tick_params(colors='white')
    ax3.spines['bottom'].set_color('#4a5568')
    ax3.spines['left'].set_color('#4a5568')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.grid(True, alpha=0.15, color='white', which='both')
    ax3.legend(loc='lower right', fontsize=8, facecolor='#1a1a2e', edgecolor='#4a5568',
              labelcolor='white')

    fig3.tight_layout()
    save_multi_chart(fig3, 'roofline_bs1.png')
    plt.close(fig3)

    # ------ Chart 4: Heatmap of Recommended Power Modes (3-Tier) ------
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    fig4.patch.set_facecolor('#1a1a2e')
    ax4.set_facecolor('#16213e')

    mode_num_map = {MODE_15W: 0, MODE_25W: 1, MODE_MAXN: 2}

    heatmap_data = np.zeros((len(model_names), len(BATCH_SIZES)))
    for r in all_results:
        mi = model_names.index(r['display_name'])
        bi = BATCH_SIZES.index(r['batch_size'])
        heatmap_data[mi, bi] = mode_num_map.get(r['recommended_mode'], 1)

    cmap = matplotlib.colors.ListedColormap(['#48BB78', '#ECC94B', '#FC8181'])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    im = ax4.imshow(heatmap_data, cmap=cmap, norm=norm, aspect='auto')

    mode_labels = {0: '15W', 1: '25W', 2: 'MAXN'}
    for i in range(len(model_names)):
        for j in range(len(BATCH_SIZES)):
            mode_text = mode_labels[int(heatmap_data[i, j])]
            ax4.text(j, i, mode_text, ha='center', va='center',
                    fontsize=10, fontweight='bold', color='black')

    ax4.set_xticks(range(len(BATCH_SIZES)))
    ax4.set_xticklabels([str(b) for b in BATCH_SIZES], color='white')
    ax4.set_yticks(range(len(model_names)))
    ax4.set_yticklabels(model_names, color='white')
    ax4.set_xlabel('Batch Size', fontsize=12, color='white', fontweight='bold')
    ax4.set_ylabel('Model', fontsize=12, color='white', fontweight='bold')
    ax4.set_title('Pagoda 3-Tier DVFS Selection Heatmap (15W / 25W / MAXN_SUPER)',
                 fontsize=14, color='white', fontweight='bold', pad=15)
    ax4.tick_params(colors='white')

    fig4.tight_layout()
    save_multi_chart(fig4, 'dvfs_heatmap.png')
    plt.close(fig4)

    print("\nAll charts generated successfully!")

except ImportError as e:
    print(f"\nmatplotlib error ({e}). Skipping charts.")

print("\n=== COMPLETE ===")
