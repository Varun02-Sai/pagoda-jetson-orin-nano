"""
Generate all figures for BTP Pagoda Progress v3.
Reads from real data files only. No invented numbers.
Saves PNGs to reports/figures/
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, sys

# Add project root to path
sys.path.insert(0, r'c:\Users\saiva\OneDrive\Desktop\pagoda')

OUT = r'c:\Users\saiva\OneDrive\Desktop\pagoda\reports\figures'
os.makedirs(OUT, exist_ok=True)

# ---------- Style ----------
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#333333',
    'text.color': '#222222',
    'axes.labelcolor': '#222222',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
})

# ---------- Load Data ----------
hw = pd.read_csv(r'c:\Users\saiva\OneDrive\Desktop\pagoda\results\jetson_hardware_results.csv')
rf = pd.read_csv(r'c:\Users\saiva\OneDrive\Desktop\pagoda\results\roofline_data.csv')
bm = pd.read_csv(r'c:\Users\saiva\OneDrive\Desktop\pagoda\results\benchmark_vs_actual_hardware.csv')
vr = pd.read_csv(r'c:\Users\saiva\OneDrive\Desktop\pagoda\results\validation_results.csv')

# Model name mapping
MODEL_NAMES = {'mobnet': 'MobileNet-v3', 'lstm': 'LSTM', 'yolo': 'YOLOv8n', 
               'bert': 'BERT-Base', 'resnet': 'ResNet-50'}
MODEL_COLORS = {'MobileNet-v3': '#e74c3c', 'LSTM': '#3498db', 'YOLOv8n': '#2ecc71',
                'BERT-Base': '#9b59b6', 'ResNet-50': '#e67e22'}

# ============================================================
# FIG 1: Roofline plot with ALL models at BS=1 
# ============================================================
def fig_roofline_bs1():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Hardware specs
    peak_compute = 2088.96  # GFLOP/s at MAXN
    peak_bw = 102.37        # GB/s at MAXN
    ridge = peak_compute / peak_bw  # 20.41
    
    # Roofline lines
    ai_range = np.logspace(-1, 3, 500)
    roofline = np.minimum(peak_compute, ai_range * peak_bw)
    ax.plot(ai_range, roofline, 'k-', linewidth=2.5, label='MAXN Roofline')
    
    # 15W roofline
    peak_15w = 1253.38
    bw_15w = 68.26
    roofline_15w = np.minimum(peak_15w, ai_range * bw_15w)
    ax.plot(ai_range, roofline_15w, '--', color='#e74c3c', linewidth=1.5, alpha=0.7, label='15W Roofline')
    
    # Ridge point line
    ax.axvline(x=ridge, color='gray', linestyle=':', linewidth=1, alpha=0.6)
    ax.text(ridge * 1.1, peak_compute * 0.5, f'Ridge = {ridge:.1f}', fontsize=10, color='gray')
    
    # Plot models at BS=1
    bs1 = rf[rf['batch_size'] == 1].copy()
    markers = ['o', 's', '^', 'D', 'v']
    
    for idx, (_, row) in enumerate(bs1.iterrows()):
        model = row['model']
        ai = row['arithmetic_intensity']
        gflops = row['total_gflops']
        attainable = min(peak_compute, ai * peak_bw)
        color = MODEL_COLORS.get(model, '#333')
        
        ax.scatter(ai, attainable, s=150, c=color, marker=markers[idx], 
                  edgecolors='black', linewidth=0.8, zorder=5)
        offset_y = 1.3 if model != 'LSTM' else 0.6
        ax.annotate(f'{model}\n(AI={ai:.1f})', (ai, attainable), 
                   textcoords="offset points", xytext=(10, 10 if idx % 2 == 0 else -20),
                   fontsize=9, ha='left')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Arithmetic Intensity (FLOP/Byte)')
    ax.set_ylabel('Attainable Performance (GFLOP/s)')
    ax.set_title('Roofline Model: DNN Models at Batch Size 1')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_xlim(1, 200)
    ax.set_ylim(50, 5000)
    ax.grid(True, alpha=0.3, which='both')
    
    fig.tight_layout()
    path = os.path.join(OUT, 'roofline_all_models_bs1.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')

# ============================================================
# FIG 2: Compute-bound vs Memory-bound layer distribution
# ============================================================
def fig_layer_distribution():
    fig, ax = plt.subplots(figsize=(10, 5))
    
    bs1 = rf[rf['batch_size'] == 1].copy()
    models = bs1['model'].tolist()
    cb = bs1['num_compute_bound'].tolist()
    mb = bs1['num_memory_bound'].tolist()
    total = bs1['num_layers'].tolist()
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, cb, width, label='Compute-Bound Layers', color='#3498db', edgecolor='white')
    bars2 = ax.bar(x + width/2, mb, width, label='Memory-Bound Layers', color='#e74c3c', edgecolor='white')
    
    # Add count labels
    for bar, val in zip(bars1, cb):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 3, 
                str(val), ha='center', va='bottom', fontsize=10)
    for bar, val in zip(bars2, mb):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 3,
                str(val), ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Number of Layers')
    ax.set_title('Layer Classification at Batch Size 1 (Ridge = 20.41 FLOP/Byte)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=0)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    fig.tight_layout()
    path = os.path.join(OUT, 'layer_distribution_bs1.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')

# ============================================================
# FIG 3: Energy comparison across power modes (grouped bar)
# ============================================================
def fig_energy_across_modes():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax_idx, bs in enumerate([1, 16]):
        ax = axes[ax_idx]
        subset = bm[bm['batch_size'] == bs].copy()
        models = subset['model'].tolist()
        e15 = subset['energy_15w_mj'].tolist()
        e25 = subset['energy_25w_mj'].tolist()
        emx = subset['energy_maxn_mj'].tolist()
        
        x = np.arange(len(models))
        w = 0.25
        
        ax.bar(x - w, e15, w, label='15W', color='#27ae60')
        ax.bar(x, e25, w, label='25W', color='#f39c12')
        ax.bar(x + w, emx, w, label='MAXN', color='#2980b9')
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Energy per Inference (mJ)')
        ax.set_title(f'Batch Size = {bs}')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=25, ha='right', fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    
    fig.suptitle('Energy Comparison Across Power Modes', fontsize=16, y=1.02)
    fig.tight_layout()
    path = os.path.join(OUT, 'energy_across_modes.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')

# ============================================================
# FIG 4: Latency comparison across modes (BS=1 and BS=16)
# ============================================================
def fig_latency_across_modes():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax_idx, bs in enumerate([1, 16]):
        ax = axes[ax_idx]
        subset = hw[hw['batch_size'] == bs].copy()
        
        models_order = ['mobnet', 'lstm', 'yolo', 'bert', 'resnet']
        model_labels = [MODEL_NAMES[k] for k in models_order]
        
        maxn_vals, w25_vals, w15_vals = [], [], []
        maxn_std, w25_std, w15_std = [], [], []
        
        for mk in models_order:
            for mode, vlist, slist in [('MAXN_SUPER', maxn_vals, maxn_std), 
                                        ('25W', w25_vals, w25_std),
                                        ('15W', w15_vals, w15_std)]:
                row = subset[(subset['model_key'] == mk) & (subset['power_mode'] == mode)]
                if len(row) > 0:
                    vlist.append(row.iloc[0]['avg_latency_ms'])
                    slist.append(row.iloc[0]['std_latency_ms'])
                else:
                    vlist.append(0)
                    slist.append(0)
        
        x = np.arange(len(model_labels))
        w = 0.25
        
        ax.bar(x - w, w15_vals, w, yerr=w15_std, capsize=3, label='15W', color='#27ae60', alpha=0.85)
        ax.bar(x, w25_vals, w, yerr=w25_std, capsize=3, label='25W', color='#f39c12', alpha=0.85)
        ax.bar(x + w, maxn_vals, w, yerr=maxn_std, capsize=3, label='MAXN', color='#2980b9', alpha=0.85)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Latency (ms)')
        ax.set_title(f'Batch Size = {bs}')
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, rotation=25, ha='right', fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    
    fig.suptitle('Latency with StdDev Error Bars Across Power Modes', fontsize=16, y=1.02)
    fig.tight_layout()
    path = os.path.join(OUT, 'latency_across_modes.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')

# ============================================================
# FIG 5: Prediction accuracy (what matched, what didn't)
# ============================================================
def fig_prediction_match():
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # From benchmark_vs_actual_hardware.csv
    verdicts = bm['physical_verdict'].value_counts()
    labels = verdicts.index.tolist()
    values = verdicts.values.tolist()
    
    colors = {'SLA_PRESERVED': '#2ecc71', 'ENERGY_TRADEOFF': '#e74c3c'}
    bar_colors = [colors.get(l, '#95a5a6') for l in labels]
    
    # Rename for readability
    display_labels = []
    for l in labels:
        if l == 'SLA_PRESERVED':
            display_labels.append('Correct: MAXN\n(SLA Preserved)')
        elif l == 'ENERGY_TRADEOFF':
            display_labels.append('Mismatch: 15W\n(Energy Tradeoff)')
        else:
            display_labels.append(l)
    
    bars = ax.bar(display_labels, values, color=bar_colors, edgecolor='white', width=0.5)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{val}/20', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_ylabel('Number of Workloads')
    ax.set_title('Prediction vs Hardware Ground Truth (20 Workloads)')
    ax.set_ylim(0, max(values) + 3)
    ax.grid(axis='y', alpha=0.3)
    
    fig.tight_layout()
    path = os.path.join(OUT, 'prediction_match.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')

# ============================================================
# FIG 6: 25W Pareto inefficiency (energy delta)
# ============================================================
def fig_pareto_25w():
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Calculate energy delta: (25W - MAXN) / MAXN * 100
    bs_list = bm['batch_size'].unique()
    
    models_order = ['MobileNet-v3', 'LSTM', 'YOLOv8n', 'BERT-Base', 'ResNet-50']
    x_labels = []
    deltas = []
    colors_list = []
    
    for model in models_order:
        for bs in sorted(bm[bm['model'] == model]['batch_size'].unique()):
            row = bm[(bm['model'] == model) & (bm['batch_size'] == bs)]
            if len(row) > 0:
                e25 = row.iloc[0]['energy_25w_mj']
                emx = row.iloc[0]['energy_maxn_mj']
                delta = (e25 - emx) / emx * 100
                x_labels.append(f'{model}\nBS={bs}')
                deltas.append(delta)
                colors_list.append('#e74c3c' if delta > 0 else '#2ecc71')
    
    x = np.arange(len(x_labels))
    bars = ax.bar(x, deltas, color=colors_list, edgecolor='white', width=0.7)
    
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_xlabel('Workload')
    ax.set_ylabel('Energy Delta: 25W vs MAXN (%)')
    ax.set_title('25W Mode Energy Overhead vs MAXN (Positive = 25W is Worse)')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    
    fig.tight_layout()
    path = os.path.join(OUT, 'pareto_25w.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')

# ============================================================
# FIG 7: Latency penalty % (SLA analysis)
# ============================================================
def fig_latency_penalty():
    fig, ax = plt.subplots(figsize=(10, 5))
    
    models_order = ['MobileNet-v3', 'LSTM', 'YOLOv8n', 'BERT-Base', 'ResNet-50']
    x_labels = []
    lat_delta_15w = []
    lat_delta_25w = []
    
    for model in models_order:
        for bs in sorted(bm[bm['model'] == model]['batch_size'].unique()):
            row = bm[(bm['model'] == model) & (bm['batch_size'] == bs)]
            if len(row) > 0:
                x_labels.append(f'{model}\nBS={bs}')
                lat_maxn = row.iloc[0]['latency_maxn_ms']
                e15 = row.iloc[0]['energy_15w_mj']
                e25 = row.iloc[0]['energy_25w_mj']
                # Get latency from hw data
                hw_15w = hw[(hw['model_key'] == {'MobileNet-v3':'mobnet','LSTM':'lstm','YOLOv8n':'yolo','BERT-Base':'bert','ResNet-50':'resnet'}[model]) & (hw['batch_size'] == bs) & (hw['power_mode'] == '15W')]
                hw_25w = hw[(hw['model_key'] == {'MobileNet-v3':'mobnet','LSTM':'lstm','YOLOv8n':'yolo','BERT-Base':'bert','ResNet-50':'resnet'}[model]) & (hw['batch_size'] == bs) & (hw['power_mode'] == '25W')]
                
                if len(hw_15w) > 0:
                    lat_delta_15w.append((hw_15w.iloc[0]['avg_latency_ms'] - lat_maxn) / lat_maxn * 100)
                else:
                    lat_delta_15w.append(0)
                if len(hw_25w) > 0:
                    lat_delta_25w.append((hw_25w.iloc[0]['avg_latency_ms'] - lat_maxn) / lat_maxn * 100)
                else:
                    lat_delta_25w.append(0)
    
    x = np.arange(len(x_labels))
    w = 0.35
    
    ax.bar(x - w/2, lat_delta_15w, w, label='15W', color='#27ae60', alpha=0.85)
    ax.bar(x + w/2, lat_delta_25w, w, label='25W', color='#f39c12', alpha=0.85)
    ax.axhline(y=15, color='red', linestyle='--', linewidth=1.5, label='15% SLA Threshold')
    
    ax.set_xlabel('Workload')
    ax.set_ylabel('Latency Penalty vs MAXN (%)')
    ax.set_title('Latency Penalty at Lower Power Modes (SLA = 15%)')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    fig.tight_layout()
    path = os.path.join(OUT, 'latency_penalty_sla.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')

# ============================================================
# FIG 8: Energy savings at 15W (scatter: energy saved vs latency cost)
# ============================================================
def fig_energy_vs_latency_tradeoff():
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for _, row in vr.iterrows():
        model = row['model']
        bs = row['batch_size']
        e_save = row['energy_saving_pct']
        lat_delta = row['lat_delta_pct']
        color = MODEL_COLORS.get(model, '#333')
        
        ax.scatter(lat_delta, e_save, s=100, c=color, edgecolors='black', linewidth=0.5, zorder=5)
        ax.annotate(f'BS{bs}', (lat_delta, e_save), textcoords="offset points", 
                   xytext=(5, 5), fontsize=8)
    
    # SLA lines
    ax.axvline(x=15, color='red', linestyle='--', linewidth=1.2, alpha=0.7, label='15% SLA limit')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8)
    
    # Quadrant labels
    ax.fill_between([0, 15], 0, 30, alpha=0.05, color='green')
    ax.text(5, 14, 'Win-Win\nZone', fontsize=11, color='green', alpha=0.5, ha='center')
    
    # Legend for models
    handles = [mpatches.Patch(color=c, label=m) for m, c in MODEL_COLORS.items()]
    ax.legend(handles=handles, loc='upper right', fontsize=10)
    
    ax.set_xlabel('Latency Penalty vs MAXN (%)')
    ax.set_ylabel('Energy Saved vs MAXN (%)')
    ax.set_title('Energy-Latency Tradeoff per Workload')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 35)
    ax.set_ylim(-2, 30)
    
    fig.tight_layout()
    path = os.path.join(OUT, 'energy_latency_tradeoff.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')

# ============================================================
# FIG 9: AI scaling across batch sizes
# ============================================================
def fig_ai_scaling():
    fig, ax = plt.subplots(figsize=(9, 5))
    
    bs_values = [1, 4, 8, 16]
    for model in ['MobileNet-v3', 'LSTM', 'YOLOv8n', 'BERT-Base', 'ResNet-50']:
        ais = []
        for bs in bs_values:
            row = rf[(rf['model'] == model) & (rf['batch_size'] == bs)]
            if len(row) > 0:
                ais.append(row.iloc[0]['arithmetic_intensity'])
            else:
                ais.append(np.nan)
        ax.plot(bs_values, ais, 'o-', label=model, color=MODEL_COLORS[model], 
                linewidth=2, markersize=8)
    
    ax.axhline(y=20.41, color='gray', linestyle=':', linewidth=1.5, label='Ridge (MAXN) = 20.41')
    ax.axhline(y=18.36, color='red', linestyle=':', linewidth=1, alpha=0.5, label='Ridge (15W) = 18.36')
    
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Arithmetic Intensity (FLOP/Byte)')
    ax.set_title('AI Scaling with Batch Size')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(bs_values)
    
    fig.tight_layout()
    path = os.path.join(OUT, 'ai_scaling_batch.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')

# ============================================================
# FIG 10: Power draw per model (all modes, BS=1)
# ============================================================
def fig_power_draw():
    fig, ax = plt.subplots(figsize=(9, 5))
    
    models_order = ['mobnet', 'lstm', 'yolo', 'bert', 'resnet']
    model_labels = [MODEL_NAMES[k] for k in models_order]
    
    for bs in [1]:
        subset = hw[hw['batch_size'] == bs]
        p15, p25, pmx = [], [], []
        for mk in models_order:
            for mode, plist in [('15W', p15), ('25W', p25), ('MAXN_SUPER', pmx)]:
                row = subset[(subset['model_key'] == mk) & (subset['power_mode'] == mode)]
                plist.append(row.iloc[0]['avg_power_w'] if len(row) > 0 else 0)
        
        x = np.arange(len(model_labels))
        w = 0.25
        ax.bar(x - w, p15, w, label='15W', color='#27ae60')
        ax.bar(x, p25, w, label='25W', color='#f39c12')
        ax.bar(x + w, pmx, w, label='MAXN', color='#2980b9')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Average Power Draw (W)')
    ax.set_title('Power Draw at Batch Size 1')
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    fig.tight_layout()
    path = os.path.join(OUT, 'power_draw_bs1.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')

# ============================================================
# FIG 11: Analytical model pipeline flow diagram
# (Generated as a clean matplotlib figure)
# ============================================================
def fig_model_pipeline():
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis('off')
    
    # Boxes
    boxes = [
        (0.5, 1.5, 'ONNX\nModel'),
        (2.5, 1.5, 'Per-Layer\nAnalysis\n(FLOPs, Bytes)'),
        (5.0, 1.5, 'Roofline\nClassification\n(AI vs Ridge)'),
        (7.5, 1.5, 'Power Mode\nSelection\n(MAXN / 15W)'),
        (10.0, 1.5, 'Compare with\nHardware\nMeasurements'),
    ]
    
    colors = ['#3498db', '#2ecc71', '#9b59b6', '#e67e22', '#e74c3c']
    
    for i, (x, y, text) in enumerate(boxes):
        rect = mpatches.FancyBboxPatch((x - 0.8, y - 0.7), 1.6, 1.4,
                                        boxstyle="round,pad=0.1",
                                        facecolor=colors[i], alpha=0.15,
                                        edgecolor=colors[i], linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrows
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + 0.8
        x2 = boxes[i+1][0] - 0.8
        ax.annotate('', xy=(x2, 1.5), xytext=(x1, 1.5),
                    arrowprops=dict(arrowstyle='->', color='#333', lw=2))
    
    # Labels below
    sublabels = [
        (0.5, 0.3, 'Input: .onnx file'),
        (2.5, 0.3, 'FLOPs per layer\nMemory bytes per layer'),
        (5.0, 0.3, 'AI = FLOPs / Bytes\nCompute vs Memory bound'),
        (7.5, 0.3, 'CR≥70% or AI≥Ridge\n→ MAXN, else 15W'),
        (10.0, 0.3, 'Latency, Power, Energy\non Jetson board'),
    ]
    for x, y, text in sublabels:
        ax.text(x, y, text, ha='center', va='center', fontsize=8, 
                color='#555', style='italic')
    
    ax.set_title('Analytical Model Pipeline', fontsize=14, fontweight='bold', pad=10)
    
    fig.tight_layout()
    path = os.path.join(OUT, 'model_pipeline.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')

# ============================================================
# FIG 12: Roofline equations summary (clean visual)
# ============================================================
def fig_equations():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')
    
    equations = [
        (r'$\mathrm{Per\ Layer\ FLOPs}$', 
         r'Conv: $2 \times K^2 \times C_{in} \times C_{out} \times H_{out} \times W_{out} \times B$'),
        (r'$\mathrm{Memory\ Bytes}$', 
         r'Input + Weights + Output (all in float32 = 4 bytes each)'),
        (r'$\mathrm{Arithmetic\ Intensity}$',
         r'$AI = \frac{\mathrm{Total\ FLOPs}}{\mathrm{Total\ Memory\ Bytes}}\ \ \mathrm{(FLOP/Byte)}$'),
        (r'$\mathrm{Ridge\ Point}$',
         r'$Ridge = \frac{\mathrm{Peak\ Compute\ (GFLOP/s)}}{\mathrm{Peak\ Bandwidth\ (GB/s)}}$'),
        (r'$\mathrm{Roofline\ Performance}$',
         r'$P_{attainable} = \min(\mathrm{Peak\ Compute},\ AI \times \mathrm{Peak\ BW})$'),
        (r'$\mathrm{Decision\ Rule}$',
         r'If $AI \geq Ridge$ or $CR \geq 0.70$: MAXN,  else: 15W'),
    ]
    
    y_start = 0.92
    for i, (label, eq) in enumerate(equations):
        y = y_start - i * 0.155
        ax.text(0.02, y, label, transform=ax.transAxes, fontsize=12, 
                fontweight='bold', va='top', color='#2c3e50')
        ax.text(0.35, y, eq, transform=ax.transAxes, fontsize=12, va='top')
    
    ax.set_title('Roofline Model: Key Equations', fontsize=14, fontweight='bold', pad=15)
    
    # Add device specs box
    specs_text = ('Jetson Orin Nano (MAXN): Peak = 2089 GFLOP/s, '
                  'BW = 102.4 GB/s, Ridge = 20.41 FLOP/Byte')
    ax.text(0.5, 0.02, specs_text, transform=ax.transAxes, fontsize=10,
            ha='center', va='bottom', style='italic', color='#666',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0', edgecolor='#ccc'))
    
    fig.tight_layout()
    path = os.path.join(OUT, 'roofline_equations.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')

# ============================================================
# Run all
# ============================================================
print('Generating figures...')
fig_roofline_bs1()
fig_layer_distribution()
fig_energy_across_modes()
fig_latency_across_modes()
fig_prediction_match()
fig_pareto_25w()
fig_latency_penalty()
fig_energy_vs_latency_tradeoff()
fig_ai_scaling()
fig_power_draw()
fig_model_pipeline()
fig_equations()
print(f'\nDone! All figures saved to: {OUT}')

