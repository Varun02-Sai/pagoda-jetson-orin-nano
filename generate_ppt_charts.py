"""Generate standard deviation and other supplementary charts for the BTP presentation."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv
import os

# Read hardware CSV
hw_csv = r"c:\Users\saiva\OneDrive\Desktop\pagoda\results\jetson_hardware_results.csv"
data = []
with open(hw_csv, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append(row)

model_map = {
    'mobnet.onnx': 'MobileNet-v3',
    'lstm.onnx': 'LSTM',
    'yolov8n.onnx': 'YOLOv8n',
    'bert-base.onnx': 'BERT-Base',
    'resnet50.onnx': 'ResNet-50',
}

models_order = ['MobileNet-v3', 'LSTM', 'YOLOv8n', 'BERT-Base', 'ResNet-50']
modes_order = ['15W', '25W', 'MAXN_SUPER']
mode_colors = {'15W': '#2ecc71', '25W': '#f39c12', 'MAXN_SUPER': '#e74c3c'}
batch_sizes = [1, 4, 8, 16]

results_dir = r"c:\Users\saiva\OneDrive\Desktop\pagoda\results"

# Organize data
organized = {}
for row in data:
    mname = model_map.get(row['model'], row['model'])
    bs = int(row['batch_size'])
    mode = row['power_mode']
    organized[(mname, bs, mode)] = {
        'avg_lat': float(row['avg_latency_ms']),
        'std_lat': float(row['std_latency_ms']),
        'avg_pwr': float(row['avg_power_w']),
        'energy': float(row['energy_mj']),
    }

# ===== Chart 1: Latency with StdDev Error Bars (per model, grouped by batch) =====
fig, axes = plt.subplots(1, 5, figsize=(24, 5), sharey=False)
fig.suptitle('Inference Latency with Standard Deviation (Error Bars)\nNVIDIA Jetson Orin Nano Developer Kit (8GB)', fontsize=14, fontweight='bold')

for idx, model in enumerate(models_order):
    ax = axes[idx]
    x = np.arange(len(batch_sizes))
    width = 0.25
    
    for j, mode in enumerate(modes_order):
        means = []
        stds = []
        for bs in batch_sizes:
            key = (model, bs, mode)
            if key in organized:
                means.append(organized[key]['avg_lat'])
                stds.append(organized[key]['std_lat'])
            else:
                means.append(0)
                stds.append(0)
        
        bars = ax.bar(x + j*width, means, width, yerr=stds, 
                      label=mode, color=mode_colors[mode], 
                      capsize=3, edgecolor='black', linewidth=0.5,
                      error_kw={'elinewidth': 1.2, 'capthick': 1.2})
    
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Latency (ms)')
    ax.set_title(model, fontsize=11, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(batch_sizes)
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
path1 = os.path.join(results_dir, 'fig_latency_stddev.png')
plt.savefig(path1, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {path1}")

# ===== Chart 2: Energy with StdDev bars (per model) =====
fig, axes = plt.subplots(1, 5, figsize=(24, 5), sharey=False)
fig.suptitle('Energy per Inference (mJ) Across Power Modes\nNVIDIA Jetson Orin Nano Developer Kit (8GB)', fontsize=14, fontweight='bold')

for idx, model in enumerate(models_order):
    ax = axes[idx]
    x = np.arange(len(batch_sizes))
    width = 0.25
    
    for j, mode in enumerate(modes_order):
        energies = []
        for bs in batch_sizes:
            key = (model, bs, mode)
            if key in organized:
                energies.append(organized[key]['energy'])
            else:
                energies.append(0)
        
        ax.bar(x + j*width, energies, width, 
               label=mode, color=mode_colors[mode],
               edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Energy (mJ)')
    ax.set_title(model, fontsize=11, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(batch_sizes)
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
path2 = os.path.join(results_dir, 'fig_energy_per_model.png')
plt.savefig(path2, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {path2}")

# ===== Chart 3: 25W Pareto Inefficiency visualization =====
fig, ax = plt.subplots(figsize=(14, 6))
fig.suptitle('25W Mode: Pareto Inefficiency Analysis\n(Energy Savings % relative to MAXN_SUPER)', fontsize=13, fontweight='bold')

labels = []
savings_25w = []
savings_15w = []

for model in models_order:
    for bs in batch_sizes:
        maxn_key = (model, bs, 'MAXN_SUPER')
        w25_key = (model, bs, '25W')
        w15_key = (model, bs, '15W')
        if maxn_key in organized and w25_key in organized and w15_key in organized:
            maxn_e = organized[maxn_key]['energy']
            s25 = (maxn_e - organized[w25_key]['energy']) / maxn_e * 100
            s15 = (maxn_e - organized[w15_key]['energy']) / maxn_e * 100
            labels.append(f"{model[:6]} B{bs}")
            savings_25w.append(s25)
            savings_15w.append(s15)

x = np.arange(len(labels))
width = 0.35
bars1 = ax.bar(x - width/2, savings_15w, width, label='15W', color='#2ecc71', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, savings_25w, width, label='25W', color='#f39c12', edgecolor='black', linewidth=0.5)

ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='MAXN Baseline')
ax.set_ylabel('Energy Savings (%)\n(Positive = Better than MAXN)')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
path3 = os.path.join(results_dir, 'fig_pareto_inefficiency.png')
plt.savefig(path3, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {path3}")

# ===== Chart 4: Latency Penalty % (Delta from MAXN) =====
fig, ax = plt.subplots(figsize=(14, 6))
fig.suptitle('Latency Penalty (%) Relative to MAXN_SUPER Baseline\nNVIDIA Jetson Orin Nano Developer Kit (8GB)', fontsize=13, fontweight='bold')

lat_delta_15w = []
lat_delta_25w = []
labels2 = []

for model in models_order:
    for bs in batch_sizes:
        maxn_key = (model, bs, 'MAXN_SUPER')
        w25_key = (model, bs, '25W')
        w15_key = (model, bs, '15W')
        if maxn_key in organized and w25_key in organized and w15_key in organized:
            maxn_lat = organized[maxn_key]['avg_lat']
            d15 = (organized[w15_key]['avg_lat'] - maxn_lat) / maxn_lat * 100
            d25 = (organized[w25_key]['avg_lat'] - maxn_lat) / maxn_lat * 100
            labels2.append(f"{model[:6]} B{bs}")
            lat_delta_15w.append(d15)
            lat_delta_25w.append(d25)

x = np.arange(len(labels2))
width = 0.35
ax.bar(x - width/2, lat_delta_15w, width, label='15W', color='#2ecc71', edgecolor='black', linewidth=0.5)
ax.bar(x + width/2, lat_delta_25w, width, label='25W', color='#f39c12', edgecolor='black', linewidth=0.5)
ax.axhline(y=15, color='red', linestyle='--', linewidth=1.5, label='SLA Threshold (15%)')
ax.set_ylabel('Latency Delta (%)')
ax.set_xticks(x)
ax.set_xticklabels(labels2, rotation=45, ha='right', fontsize=8)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
path4 = os.path.join(results_dir, 'fig_latency_delta.png')
plt.savefig(path4, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {path4}")

# ===== Chart 5: Power Draw comparison =====
fig, axes = plt.subplots(1, 5, figsize=(24, 5), sharey=True)
fig.suptitle('Measured Power Draw (Watts) Across Power Modes\nNVIDIA Jetson Orin Nano Developer Kit (8GB)', fontsize=14, fontweight='bold')

for idx, model in enumerate(models_order):
    ax = axes[idx]
    x = np.arange(len(batch_sizes))
    width = 0.25
    
    for j, mode in enumerate(modes_order):
        powers = []
        for bs in batch_sizes:
            key = (model, bs, mode)
            if key in organized:
                powers.append(organized[key]['avg_pwr'])
            else:
                powers.append(0)
        
        ax.bar(x + j*width, powers, width, 
               label=mode, color=mode_colors[mode],
               edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Power (W)')
    ax.set_title(model, fontsize=11, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(batch_sizes)
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 14)

plt.tight_layout()
path5 = os.path.join(results_dir, 'fig_power_per_model.png')
plt.savefig(path5, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {path5}")

# ===== Chart 6: Prediction Accuracy summary =====
fig, ax = plt.subplots(figsize=(8, 5))
categories = ['SLA Preserved\n(MAXN Correct)', 'Energy Trade-Off\n(15W, SLA>15%)', 'Strict SLA Match\n(Pred=Optimal)']
values = [15, 5, 14]
total = 20
colors_bar = ['#3498db', '#e67e22', '#2ecc71']

bars = ax.bar(categories, values, color=colors_bar, edgecolor='black', linewidth=0.8)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
            f'{val}/20\n({val/total*100:.0f}%)', ha='center', fontweight='bold', fontsize=11)

ax.set_ylabel('Number of Workloads (out of 20)')
ax.set_title('Pagoda Analytical Model: Prediction Accuracy\nAgainst Physical Hardware Ground Truth', fontweight='bold')
ax.set_ylim(0, 22)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
path6 = os.path.join(results_dir, 'fig_prediction_accuracy.png')
plt.savefig(path6, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {path6}")

print("\nAll supplementary charts generated!")

