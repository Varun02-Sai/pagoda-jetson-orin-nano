"""
Jetson Sweep Script (3-Tier DVFS)
---------------------------------
This is the main script you run on the Jetson board.
It automatically:
  1. Switches power modes (MAXN_SUPER -> 25W -> 15W) using nvpmodel
  2. Locks GPU clocks using jetson_clocks
  3. Runs all 5 models at all batch sizes
  4. Measures latency, power, and energy
  5. Saves everything to results/jetson_hardware_results.csv
  6. Compares with analytical predictions

How to run (needs root for power mode switching):
  sudo python3 sweep_all.py
"""

import os
import sys
import subprocess
import time
import csv

# -------------------------------------------------------------------
# What to test
# -------------------------------------------------------------------

MODELS = [
    {'name': 'MobileNet-v3', 'key': 'mobnet',  'onnx': '../onnx_models/mobnet.onnx'},
    {'name': 'LSTM',         'key': 'lstm',    'onnx': '../onnx_models/lstm.onnx'},
    {'name': 'YOLOv8n',      'key': 'yolo',    'onnx': '../onnx_models/yolov8n.onnx'},
    {'name': 'BERT-Base',    'key': 'bert',    'onnx': '../onnx_models/bert-base.onnx'},
    {'name': 'ResNet-50',    'key': 'resnet',  'onnx': '../onnx_models/resnet50.onnx'},
]

BATCH_SIZES = [1, 4, 8, 16]

# Jetson Orin Nano Super 8GB power modes:
# Mode 2: MAXN_SUPER (GPU 1020 MHz, EMC 3199 MHz) — uncapped baseline / max performance
# Mode 1: 25W        (GPU 918 MHz,  EMC 3199 MHz) — balanced power
# Mode 0: 15W        (GPU 612 MHz,  EMC 2133 MHz) — low power / energy-saving
POWER_MODES = {
    'MAXN_SUPER': 2,
    '25W':        1,
    '15W':        0,
}

WARMUP = 10
NUM_RUNS = 50

OUTPUT_CSV = 'results/jetson_hardware_results.csv'


# -------------------------------------------------------------------
# Power mode switching
# -------------------------------------------------------------------

def set_power_mode(mode_name, mode_id):
    """Switch to a power mode and lock the clocks."""
    print(f"\n{'#'*60}")
    print(f"  SWITCHING TO: {mode_name} (nvpmodel -m {mode_id})")
    print(f"{'#'*60}")

    try:
        result = subprocess.run(
            ['nvpmodel', '-m', str(mode_id)],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            print(f"  WARNING: nvpmodel error: {result.stderr}")
        else:
            print(f"  Power mode set OK")

        time.sleep(3)

        subprocess.run(
            ['jetson_clocks'],
            capture_output=True, text=True, timeout=10
        )
        print(f"  Clocks locked")

        time.sleep(2)

        result = subprocess.run(
            ['nvpmodel', '-q'],
            capture_output=True, text=True, timeout=5
        )
        print(f"  Current mode: {result.stdout.strip()}")
        return True

    except FileNotFoundError:
        print("  ERROR: nvpmodel not found. Not on a Jetson?")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def get_current_mode_info():
    """Check what power mode we're currently in."""
    try:
        result = subprocess.run(
            ['nvpmodel', '-q'],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


# -------------------------------------------------------------------
# Run one benchmark
# -------------------------------------------------------------------

def run_single_benchmark(model, batch_size, power_mode_name, output_csv):
    """Run run_inference.py for one model/batch/mode combo."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    inference_script = os.path.join(script_dir, 'run_inference.py')
    onnx_path = os.path.join(script_dir, model['onnx'])

    if not os.path.exists(onnx_path):
        print(f"  SKIP: {onnx_path} not found")
        return

    cmd = [
        sys.executable, inference_script,
        '--model', onnx_path,
        '--model_key', model['key'],
        '--batch_size', str(batch_size),
        '--warmup', str(WARMUP),
        '--num_runs', str(NUM_RUNS),
        '--power_mode', power_mode_name,
        '--output_csv', os.path.join(script_dir, output_csv),
    ]

    print(f"\n  >> {model['name']} BS={batch_size} @ {power_mode_name}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[:200]}")
    else:
        for line in result.stdout.split('\n'):
            if 'Latency' in line or 'Power' in line or 'Energy' in line:
                print(f"    {line.strip()}")


# -------------------------------------------------------------------
# Load analytical predictions
# -------------------------------------------------------------------

def load_analytical_predictions():
    """Load analytical model predictions from CSV."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'results/multi_batch_consolidated.csv')
    if not os.path.exists(csv_path):
        csv_path = os.path.join(script_dir, '../results/multi_batch_consolidated.csv')

    predictions = {}
    if not os.path.exists(csv_path):
        print(f"  WARNING: {csv_path} not found.")
        return predictions

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['model'], int(row['batch_size']))
            predictions[key] = {
                'ai': float(row['arithmetic_intensity']),
                'regime': row.get('regime_maxn', row.get('regime', 'N/A')),
                'compute_ratio': float(row['compute_ratio_pct']),
                'recommended_mode': row['recommended_mode'],
            }

    return predictions


# -------------------------------------------------------------------
# Compare: Analytical Predictions vs Hardware Reality
# -------------------------------------------------------------------

def compare_results(hw_csv_path, predictions):
    """Compare analytical predictions against hardware measurements."""
    if not os.path.exists(hw_csv_path):
        print("No hardware results to compare.")
        return

    hw_data = {}
    modes_seen = set()
    with open(hw_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mode = row['power_mode']
            modes_seen.add(mode)
            key = (row['model'], int(row['batch_size']), mode)
            hw_data[key] = {
                'latency': float(row['avg_latency_ms']),
                'power': float(row['avg_power_w']),
                'energy': float(row['energy_mj']),
            }

    # Pick baseline: MAXN_SUPER if available, else 25W
    if 'MAXN_SUPER' in modes_seen:
        baseline_mode = 'MAXN_SUPER'
    elif '25W' in modes_seen:
        baseline_mode = '25W'
    else:
        baseline_mode = sorted(modes_seen)[0] if modes_seen else 'unknown'

    print(f"\n{'='*95}")
    print(f"  PREDICTION vs ACTUAL HARDWARE RESULTS")
    print(f"  Baseline: {baseline_mode} (performance profile) | Predicted: 3-Tier DVFS Recommendation")
    print(f"{'='*95}")
    print(f"{'Model':<14} {'BS':<4} {'Pred':<12} {f'Lat({baseline_mode[:4]})':<11} {'Lat(Pred)':<11} "
          f"{f'Pwr({baseline_mode[:4]})':<10} {'Pwr(Pred)':<10} {'E_save%':<8} {'Result'}")
    print(f"{'-'*95}")

    model_name_map = {
        'mobnet.onnx': 'MobileNet-v3',
        'lstm.onnx': 'LSTM',
        'yolov8n.onnx': 'YOLOv8n',
        'bert-base.onnx': 'BERT-Base',
        'resnet50.onnx': 'ResNet-50',
    }

    for (model_file, bs, mode), hw in sorted(hw_data.items()):
        if mode == baseline_mode:
            model_name = model_name_map.get(model_file, model_file)
            pred_key = (model_name, bs)
            if pred_key not in predictions:
                continue

            pred = predictions[pred_key]
            pred_mode = pred['recommended_mode']

            pred_hw_key = (model_file, bs, pred_mode)
            if pred_hw_key not in hw_data:
                continue

            hw_baseline = hw
            hw_pred = hw_data[pred_hw_key]

            if hw_baseline['energy'] > 0 and hw_pred['energy'] > 0:
                e_save = ((hw_baseline['energy'] - hw_pred['energy']) / hw_baseline['energy']) * 100.0
            else:
                e_save = 0.0

            if hw_baseline['latency'] > 0:
                lat_penalty = ((hw_pred['latency'] - hw_baseline['latency']) / hw_baseline['latency']) * 100.0
            else:
                lat_penalty = 0.0

            if pred_mode == baseline_mode:
                valid = 'OK (MAX)'
            elif e_save > 0 and lat_penalty < 15:
                valid = 'SAVED'
            elif e_save > 0:
                valid = 'PARTIAL'
            else:
                valid = 'MISS'

            print(f"{model_name:<14} {bs:<4} {pred_mode:<12} "
                  f"{hw_baseline['latency']:<11.2f} {hw_pred['latency']:<11.2f} "
                  f"{hw_baseline['power']:<10.1f} {hw_pred['power']:<10.1f} "
                  f"{e_save:<8.1f} {valid}")

    print(f"{'='*95}")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  PAGODA: Jetson Orin Nano Super 3-Tier Benchmark Sweep")
    print(f"  5 models x {len(BATCH_SIZES)} batch sizes x {len(POWER_MODES)} power modes"
          f" = {5 * len(BATCH_SIZES) * len(POWER_MODES)} runs")
    print("=" * 60)

    if os.geteuid() != 0:
        print("\nWARNING: Not running as root. Need sudo for power mode switching.")
        print("Run: sudo python3 sweep_all.py\n")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(script_dir, 'results'), exist_ok=True)

    output_csv = OUTPUT_CSV
    abs_output = os.path.join(script_dir, output_csv)
    if os.path.exists(abs_output):
        os.remove(abs_output)

    # Sweep in order: MAXN_SUPER (baseline) -> 25W -> 15W
    for mode_name, mode_id in POWER_MODES.items():
        success = set_power_mode(mode_name, mode_id)
        if not success:
            print(f"  Skipping {mode_name} (couldn't set power mode)")
            continue

        for model in MODELS:
            for bs in BATCH_SIZES:
                run_single_benchmark(model, bs, mode_name, output_csv)

    # Restore default performance mode
    set_power_mode('MAXN_SUPER', 2)

    print("\nRunning comprehensive benchmark vs physical hardware verification...")
    try:
        verify_script = os.path.join(script_dir, 'verify_predictions.py')
        if os.path.exists(verify_script):
            subprocess.run([sys.executable, verify_script], cwd=script_dir)
    except Exception as e:
        print(f"  Verification failed: {e}")

    print("\nGenerating charts...")
    try:
        chart_script = os.path.join(script_dir, 'generate_hw_charts.py')
        if os.path.exists(chart_script):
            subprocess.run([sys.executable, chart_script], cwd=script_dir)
            print("  Charts saved to results/ folder")
    except Exception as e:
        print(f"  Chart generation failed: {e}")

    print(f"\nResults saved to: {output_csv}")
    print("Done!")


if __name__ == '__main__':
    main()
