"""
Hardware Results Chart Generator (Supports MAXN_SUPER, 25W, 15W)
----------------------------------------------------------------
Generates clean research charts from the Jetson hardware benchmark CSV.
Dynamically detects whatever modes are present in the CSV.

Run after sweep_all.py finishes:
    python3 generate_hw_charts.py
"""

import os
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POSSIBLE_CSV_PATHS = [
    os.path.join(SCRIPT_DIR, 'results', 'jetson_hardware_results.csv'),
    os.path.join('jetson', 'results', 'jetson_hardware_results.csv'),
    os.path.join('results', 'jetson_hardware_results.csv'),
]

OUTPUT_DIRS = [
    os.path.join(SCRIPT_DIR, 'results'),
    os.path.join(os.path.dirname(SCRIPT_DIR), 'results') if os.path.basename(SCRIPT_DIR) == 'jetson' else 'results',
]

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'text.color': 'black',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.color': 'gray',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'legend.fontsize': 8,
    'figure.dpi': 150,
})

MODEL_ORDER = ['MobileNet-v3', 'LSTM', 'YOLOv8n', 'BERT-Base', 'ResNet-50']


def get_csv_file():
    for p in POSSIBLE_CSV_PATHS:
        if os.path.exists(p):
            return p
    return POSSIBLE_CSV_PATHS[0]


def save_fig(fig, filename):
    for d in set(OUTPUT_DIRS):
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, filename)
        fig.savefig(path, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {path}")


def load_data():
    """Load hardware results CSV into a nested dict."""
    data = {}
    csv_file = get_csv_file()
    if not os.path.exists(csv_file):
        print(f"  ERROR: {csv_file} not found in any expected location")
        return data, []

    modes_seen = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row['model']
            bs = int(row['batch_size'])
            mode = row['power_mode']
            if mode not in modes_seen:
                modes_seen.append(mode)
            key = (model, bs, mode)
            data[key] = {
                'latency': float(row['avg_latency_ms']),
                'power': float(row['avg_power_w']),
                'energy': float(row['energy_mj']),
                'throughput': float(row['throughput_sps']),
            }
    return data, modes_seen


def get_model_display_name(onnx_name):
    """Convert onnx filename to display name."""
    mapping = {
        'mobnet.onnx': 'MobileNet-v3',
        'lstm.onnx': 'LSTM',
        'yolov8n.onnx': 'YOLOv8n',
        'bert-base.onnx': 'BERT-Base',
        'resnet50.onnx': 'ResNet-50',
    }
    return mapping.get(onnx_name, onnx_name)


def plot_latency_comparison(data, modes):
    fig, ax = plt.subplots(figsize=(9, 4.5))

    models_found = []
    for m in MODEL_ORDER:
        for (model, bs, mode) in data:
            if get_model_display_name(model) == m and bs == 1:
                models_found.append((m, model))
                break

    if not models_found:
        return

    x_pos = range(len(models_found))
    bar_width = 0.8 / len(modes)
    hatches = ['', '//', 'xx', '..']
    grays = ['#2d3748', '#718096', '#cbd5e0']

    for i, mode in enumerate(modes):
        vals = []
        for display_name, onnx_name in models_found:
            key = (onnx_name, 1, mode)
            vals.append(data.get(key, {}).get('latency', 0))

        color = grays[i % len(grays)]
        ax.bar([p + i * bar_width for p in x_pos], vals, bar_width,
               label=mode, color='white', edgecolor='black',
               hatch=hatches[i % len(hatches)], linewidth=0.9)

    ax.set_xlabel('Model')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Inference Latency at Batch Size 1 (Across Power Modes)')
    ax.set_xticks([p + bar_width * (len(modes) - 1) / 2 for p in x_pos])
    ax.set_xticklabels([m for m, _ in models_found], rotation=15)
    ax.legend()
    ax.set_axisbelow(True)

    fig.tight_layout()
    save_fig(fig, 'fig_latency_comparison.png')
    plt.close(fig)


def plot_energy_comparison(data, modes):
    fig, ax = plt.subplots(figsize=(9, 4.5))

    models_found = []
    for m in MODEL_ORDER:
        for (model, bs, mode) in data:
            if get_model_display_name(model) == m and bs == 1:
                models_found.append((m, model))
                break

    if not models_found:
        return

    x_pos = range(len(models_found))
    bar_width = 0.8 / len(modes)
    hatches = ['', '//', 'xx', '..']

    for i, mode in enumerate(modes):
        vals = []
        for display_name, onnx_name in models_found:
            key = (onnx_name, 1, mode)
            vals.append(data.get(key, {}).get('energy', 0))

        ax.bar([p + i * bar_width for p in x_pos], vals, bar_width,
               label=mode, color='white', edgecolor='black',
               hatch=hatches[i % len(hatches)], linewidth=0.9)

    ax.set_xlabel('Model')
    ax.set_ylabel('Energy per Inference (mJ)')
    ax.set_title('Energy Consumption at Batch Size 1 (Across Power Modes)')
    ax.set_xticks([p + bar_width * (len(modes) - 1) / 2 for p in x_pos])
    ax.set_xticklabels([m for m, _ in models_found], rotation=15)
    ax.legend()
    ax.set_axisbelow(True)

    fig.tight_layout()
    save_fig(fig, 'fig_energy_comparison.png')
    plt.close(fig)


def plot_power_draw(data, modes):
    fig, ax = plt.subplots(figsize=(9, 4.5))

    models_found = []
    for m in MODEL_ORDER:
        for (model, bs, mode) in data:
            if get_model_display_name(model) == m and bs == 1:
                models_found.append((m, model))
                break

    if not models_found:
        return

    x_pos = range(len(models_found))
    bar_width = 0.8 / len(modes)
    hatches = ['', '//', 'xx', '..']

    for i, mode in enumerate(modes):
        vals = []
        for display_name, onnx_name in models_found:
            key = (onnx_name, 1, mode)
            vals.append(data.get(key, {}).get('power', 0))

        ax.bar([p + i * bar_width for p in x_pos], vals, bar_width,
               label=mode, color='white', edgecolor='black',
               hatch=hatches[i % len(hatches)], linewidth=0.9)

    ax.set_xlabel('Model')
    ax.set_ylabel('Average Power (W)')
    ax.set_title('Power Draw During Inference at Batch Size 1')
    ax.set_xticks([p + bar_width * (len(modes) - 1) / 2 for p in x_pos])
    ax.set_xticklabels([m for m, _ in models_found], rotation=15)
    ax.legend()
    ax.set_axisbelow(True)

    fig.tight_layout()
    save_fig(fig, 'fig_power_draw.png')
    plt.close(fig)


def plot_energy_savings(data, modes):
    if len(modes) < 2:
        return

    baseline_mode = modes[0]  # First mode is baseline (MAXN_SUPER or 25W)
    target_mode = modes[-1]   # Last mode is lowest power (15W)

    fig, ax = plt.subplots(figsize=(8, 4))
    batch_sizes = sorted(set(bs for (_, bs, _) in data))

    models_found = []
    for m in MODEL_ORDER:
        for (model, bs, mode) in data:
            if get_model_display_name(model) == m:
                if m not in [x[0] for x in models_found]:
                    models_found.append((m, model))
                break

    if not models_found:
        return

    x_pos = range(len(models_found))
    bar_width = 0.8 / len(batch_sizes)
    grays = ['white', 'lightgray', 'darkgray', 'dimgray']

    for j, bs in enumerate(batch_sizes):
        savings = []
        for display_name, onnx_name in models_found:
            base_key = (onnx_name, bs, baseline_mode)
            low_key = (onnx_name, bs, target_mode)
            saving = 0.0
            if base_key in data and low_key in data:
                e_base = data[base_key]['energy']
                e_low = data[low_key]['energy']
                if e_base > 0:
                    saving = (e_base - e_low) / e_base * 100.0
            savings.append(saving)

        ax.bar([p + j * bar_width for p in x_pos], savings, bar_width,
               label=f'BS={bs}', color=grays[j % len(grays)],
               edgecolor='black', linewidth=0.8)

    ax.set_xlabel('Model')
    ax.set_ylabel('Energy Saving (%)')
    ax.set_title(f'Energy Saving: {target_mode} vs {baseline_mode} (Baseline)')
    ax.set_xticks([p + bar_width * len(batch_sizes) / 2 for p in x_pos])
    ax.set_xticklabels([m for m, _ in models_found], rotation=15)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.legend()
    ax.set_axisbelow(True)

    fig.tight_layout()
    save_fig(fig, 'fig_energy_savings.png')
    plt.close(fig)


def main():
    print("Generating hardware result charts...")
    for d in OUTPUT_DIRS:
        os.makedirs(d, exist_ok=True)

    data, modes = load_data()
    if not data:
        print("No data to plot.")
        return

    print(f"  Power modes detected in CSV: {modes}")
    plot_latency_comparison(data, modes)
    plot_energy_comparison(data, modes)
    plot_power_draw(data, modes)
    plot_energy_savings(data, modes)
    print("Done.")


if __name__ == '__main__':
    main()
