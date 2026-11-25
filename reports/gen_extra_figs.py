"""Generate per-layer AI scatter plots for ALL 5 models in one figure."""
import sys
sys.path.insert(0, r'c:\Users\saiva\OneDrive\Desktop\pagoda')

from layer_analysis import analyze_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUT = r'c:\Users\saiva\OneDrive\Desktop\pagoda\reports\figures'

plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 11,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
})

models = [
    ('onnx_models/mobnet.onnx', 'mobnet', 'MobileNet-v3'),
    ('onnx_models/lstm.onnx', 'lstm', 'LSTM'),
    ('onnx_models/yolov8n.onnx', 'yolo', 'YOLOv8n'),
    ('onnx_models/bert-base.onnx', 'bert', 'BERT-Base'),
    ('onnx_models/resnet50.onnx', 'resnet', 'ResNet-50'),
]

# --- FIG: All 5 models per-layer AI scatter in one figure ---
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for idx, (path, key, name) in enumerate(models):
    ax = axes[idx]
    result = analyze_model(path, key, batch_size=1, quiet=True)
    layers = [l for l in result['layers'] if l['flops'] > 0]
    
    ais = [l['ai'] for l in layers]
    colors = ['#3498db' if l['bound_maxn'] == 'COMPUTE' else '#e74c3c' for l in layers]
    cb = sum(1 for l in layers if l['bound_maxn'] == 'COMPUTE')
    mb = sum(1 for l in layers if l['bound_maxn'] == 'MEMORY')
    
    ax.scatter(range(len(ais)), ais, c=colors, s=10, alpha=0.7)
    ax.axhline(y=20.41, color='gray', linestyle='--', linewidth=1.2)
    ax.set_title(f'{name}  (CB={cb}, MB={mb})', fontsize=12, fontweight='bold')
    ax.set_xlabel('Layer Index', fontsize=9)
    ax.set_ylabel('AI (FLOP/Byte)', fontsize=9)
    ax.grid(True, alpha=0.2)
    
    # Add overall AI text
    overall_ai = result['total_ai']
    cr = result['compute_ratio'] * 100
    ax.text(0.97, 0.95, f'Overall AI={overall_ai:.1f}\nCR={cr:.0f}%', 
            transform=ax.transAxes, fontsize=9, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#ccc', alpha=0.9))

# Hide 6th subplot
axes[5].axis('off')

# Add legend in 6th subplot area
legend_elements = [
    mpatches.Patch(facecolor='#3498db', label='Compute-Bound Layer'),
    mpatches.Patch(facecolor='#e74c3c', label='Memory-Bound Layer'),
    plt.Line2D([0], [0], color='gray', linestyle='--', label='Ridge = 20.41 FLOP/Byte'),
]
axes[5].legend(handles=legend_elements, loc='center', fontsize=14, frameon=True)
axes[5].text(0.5, 0.15, 'CB = Compute-Bound layers\nMB = Memory-Bound layers\nCR = Compute Ratio (% FLOPs in CB)', 
             transform=axes[5].transAxes, ha='center', fontsize=11, color='#666')

fig.suptitle('Per-Layer Arithmetic Intensity: All 5 Models at Batch Size 1', fontsize=15, fontweight='bold', y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.96])
path = os.path.join(OUT, 'all_models_layer_ai.png')
fig.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {path}')

# --- FIG: Compute Ratio vs AI per model (all batch sizes) ---
import pandas as pd
rf = pd.read_csv(r'c:\Users\saiva\OneDrive\Desktop\pagoda\results\roofline_data.csv')

fig, ax = plt.subplots(figsize=(9, 6))

model_colors = {'MobileNet-v3': '#e74c3c', 'LSTM': '#3498db', 'YOLOv8n': '#2ecc71',
                'BERT-Base': '#9b59b6', 'ResNet-50': '#e67e22'}
markers = {'MobileNet-v3': 'o', 'LSTM': 's', 'YOLOv8n': '^', 'BERT-Base': 'D', 'ResNet-50': 'v'}

for model in model_colors:
    subset = rf[(rf['model'] == model) & (rf['batch_size'].isin([1,4,8,16]))]
    ax.scatter(subset['arithmetic_intensity'], subset['compute_ratio_pct'], 
               c=model_colors[model], marker=markers[model], s=120, 
               edgecolors='black', linewidth=0.5, label=model, zorder=5)
    # Label with batch size
    for _, row in subset.iterrows():
        ax.annotate(f'BS{int(row["batch_size"])}', 
                   (row['arithmetic_intensity'], row['compute_ratio_pct']),
                   textcoords="offset points", xytext=(6, 4), fontsize=8)

ax.axvline(x=20.41, color='gray', linestyle='--', linewidth=1.2, label='Ridge (MAXN)')
ax.axhline(y=70, color='orange', linestyle=':', linewidth=1.2, label='CR = 70% threshold')

# Decision zones
ax.fill_between([20.41, 55], 0, 100, alpha=0.04, color='blue')
ax.fill_between([0, 20.41], 70, 100, alpha=0.04, color='blue')
ax.text(35, 50, 'MAXN zone\n(AI \u2265 Ridge)', fontsize=10, color='blue', alpha=0.4, ha='center')
ax.text(8, 15, '15W zone\n(AI < Ridge\nand CR < 70%)', fontsize=10, color='red', alpha=0.4, ha='center')

ax.set_xlabel('Arithmetic Intensity (FLOP/Byte)', fontsize=12)
ax.set_ylabel('Compute Ratio (%)', fontsize=12)
ax.set_title('Compute Ratio vs Arithmetic Intensity (Decision Map)', fontsize=14)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 55)
ax.set_ylim(-5, 105)

fig.tight_layout()
path = os.path.join(OUT, 'compute_ratio_vs_ai.png')
fig.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {path}')

# --- FIG: Energy savings vs latency penalty (cleaner version) ---
fig, ax = plt.subplots(figsize=(9, 6))

bm = pd.read_csv(r'c:\Users\saiva\OneDrive\Desktop\pagoda\results\benchmark_vs_actual_hardware.csv')

for _, row in bm.iterrows():
    model = row['model']
    bs = row['batch_size']
    
    # Get 15W latency from hw data
    hw = pd.read_csv(r'c:\Users\saiva\OneDrive\Desktop\pagoda\results\jetson_hardware_results.csv')
    key_map = {'MobileNet-v3':'mobnet','LSTM':'lstm','YOLOv8n':'yolo','BERT-Base':'bert','ResNet-50':'resnet'}
    
    hw_15w = hw[(hw['model_key'] == key_map[model]) & (hw['batch_size'] == bs) & (hw['power_mode'] == '15W')]
    hw_maxn = hw[(hw['model_key'] == key_map[model]) & (hw['batch_size'] == bs) & (hw['power_mode'] == 'MAXN_SUPER')]
    
    if len(hw_15w) > 0 and len(hw_maxn) > 0:
        lat_delta = (hw_15w.iloc[0]['avg_latency_ms'] - hw_maxn.iloc[0]['avg_latency_ms']) / hw_maxn.iloc[0]['avg_latency_ms'] * 100
        e_save = (hw_maxn.iloc[0]['energy_mj'] - hw_15w.iloc[0]['energy_mj']) / hw_maxn.iloc[0]['energy_mj'] * 100
        
        color = model_colors[model]
        ax.scatter(lat_delta, e_save, s=120, c=color, marker=markers[model],
                  edgecolors='black', linewidth=0.5, zorder=5)
        ax.annotate(f'BS{bs}', (lat_delta, e_save), textcoords="offset points", 
                   xytext=(5, 5), fontsize=8)

ax.axvline(x=15, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='15% SLA limit')
ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8)

# Win-win zone
ax.fill_between([0, 15], 0, 25, alpha=0.08, color='green')
ax.text(7, 1.5, 'Win-Win Zone', fontsize=12, color='green', alpha=0.6, ha='center', fontweight='bold')

handles = [mpatches.Patch(color=c, label=m) for m, c in model_colors.items()]
handles.append(plt.Line2D([0], [0], color='red', linestyle='--', label='15% SLA'))
ax.legend(handles=handles, loc='upper right', fontsize=10)

ax.set_xlabel('Latency Penalty at 15W vs MAXN (%)', fontsize=12)
ax.set_ylabel('Energy Saved at 15W vs MAXN (%)', fontsize=12)
ax.set_title('Energy-Latency Tradeoff: 15W vs MAXN', fontsize=14)
ax.grid(True, alpha=0.3)

fig.tight_layout()
path = os.path.join(OUT, 'energy_vs_latency_scatter.png')
fig.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {path}')

print('\nDone!')

