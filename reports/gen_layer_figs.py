"""Generate per-layer data for ResNet-50 and create per-layer figure."""
import sys
sys.path.insert(0, r'c:\Users\saiva\OneDrive\Desktop\pagoda')

from layer_analysis import analyze_model
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = r'c:\Users\saiva\OneDrive\Desktop\pagoda\reports\figures'

# Generate per-layer for ResNet-50 BS=1
result = analyze_model('onnx_models/resnet50.onnx', 'resnet', batch_size=1, quiet=True)
layers = result['layers']
layers_with_flops = [l for l in layers if l['flops'] > 0]

print(f'Total layers with work: {len(layers_with_flops)}')
cb = sum(1 for l in layers_with_flops if l['bound_maxn'] == 'COMPUTE')
mb = sum(1 for l in layers_with_flops if l['bound_maxn'] == 'MEMORY')
print(f'Compute-bound: {cb}, Memory-bound: {mb}')

# Save CSV
csv_path = os.path.join(OUT, 'resnet50_layers_bs1.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['layer_name', 'layer_type', 'flops', 'memory_bytes', 'ai', 'bound_maxn'])
    for l in layers_with_flops:
        writer.writerow([l['name'], l['type'], l['flops'], l['memory'], l['ai'], l['bound_maxn']])
print(f'Saved: {csv_path}')

# --- FIG: Per-layer AI distribution for ResNet-50 ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Scatter of per-layer AI
ais = [l['ai'] for l in layers_with_flops]
flops_m = [l['flops'] / 1e6 for l in layers_with_flops]
types = [l['type'] for l in layers_with_flops]
colors_scatter = ['#3498db' if l['bound_maxn'] == 'COMPUTE' else '#e74c3c' for l in layers_with_flops]

ax1.scatter(range(len(ais)), ais, c=colors_scatter, s=20, alpha=0.7)
ax1.axhline(y=20.41, color='gray', linestyle='--', linewidth=1.5, label='Ridge (MAXN) = 20.41')
ax1.set_xlabel('Layer Index')
ax1.set_ylabel('Arithmetic Intensity (FLOP/Byte)')
ax1.set_title('ResNet-50: Per-Layer AI (BS=1)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Add color legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#3498db', label='Compute-Bound'),
                   Patch(facecolor='#e74c3c', label='Memory-Bound')]
ax1.legend(handles=legend_elements + [plt.Line2D([0], [0], color='gray', linestyle='--', label='Ridge')], 
           loc='upper right', fontsize=9)

# Right: Top-15 layers by FLOPs (bar chart)
sorted_layers = sorted(layers_with_flops, key=lambda x: x['flops'], reverse=True)[:15]
names = [f"{l['type']}" for l in sorted_layers]
flops_vals = [l['flops'] / 1e6 for l in sorted_layers]
bar_colors = ['#3498db' if l['bound_maxn'] == 'COMPUTE' else '#e74c3c' for l in sorted_layers]

y_pos = np.arange(len(names))
ax2.barh(y_pos, flops_vals, color=bar_colors, edgecolor='white')
ax2.set_yticks(y_pos)
ax2.set_yticklabels(names, fontsize=9)
ax2.invert_yaxis()
ax2.set_xlabel('FLOPs (millions)')
ax2.set_title('ResNet-50: Top 15 Layers by FLOPs')
ax2.grid(axis='x', alpha=0.3)

fig.tight_layout()
path = os.path.join(OUT, 'resnet50_layer_analysis.png')
fig.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {path}')

# --- Also generate for MobileNet (the interesting failure case) ---
result_mob = analyze_model('onnx_models/mobnet.onnx', 'mobnet', batch_size=1, quiet=True)
layers_mob = [l for l in result_mob['layers'] if l['flops'] > 0]

fig, ax = plt.subplots(figsize=(8, 5))
ais_mob = [l['ai'] for l in layers_mob]
colors_mob = ['#3498db' if l['bound_maxn'] == 'COMPUTE' else '#e74c3c' for l in layers_mob]
ax.scatter(range(len(ais_mob)), ais_mob, c=colors_mob, s=12, alpha=0.7)
ax.axhline(y=20.41, color='gray', linestyle='--', linewidth=1.5)
ax.set_xlabel('Layer Index')
ax.set_ylabel('Arithmetic Intensity (FLOP/Byte)')
ax.set_title('MobileNet-v3: Per-Layer AI (BS=1) — Mostly Memory-Bound')
ax.legend(handles=[Patch(facecolor='#3498db', label=f'Compute-Bound ({sum(1 for l in layers_mob if l["bound_maxn"]=="COMPUTE")})'),
                    Patch(facecolor='#e74c3c', label=f'Memory-Bound ({sum(1 for l in layers_mob if l["bound_maxn"]=="MEMORY")})')],
           loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()
path = os.path.join(OUT, 'mobilenet_layer_analysis.png')
fig.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {path}')

