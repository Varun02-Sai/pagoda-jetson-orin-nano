"""
Jetson Inference Runner
-----------------------
Runs one ONNX model on the Jetson GPU and measures latency and power.

How to run (on the Jetson board):
  python run_inference.py --model ../onnx_models/resnet50.onnx --model_key resnet --batch_size 1 --num_runs 50

The sweep script (sweep_all.py) calls this for every combination of
model, batch size, and power mode.
"""

import os
import sys
import time
import argparse
import subprocess
import csv
import numpy as np


# -------------------------------------------------------------------
# 1. Load the ONNX model
# -------------------------------------------------------------------
def load_model(onnx_path):
    """Load ONNX model. Tries TensorRT first, then CUDA, then CPU."""
    import onnxruntime as ort

    available = ort.get_available_providers()
    print(f"  Available providers: {available}")

    # Pick the best available GPU backend
    if 'TensorrtExecutionProvider' in available:
        providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider']
    elif 'CUDAExecutionProvider' in available:
        providers = ['CUDAExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']
        print("  WARNING: No GPU found. Running on CPU (won't match Jetson results).")

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_path, opts, providers=providers)
    print(f"  Loaded on: {session.get_providers()}")
    return session


# -------------------------------------------------------------------
# 2. Create dummy input matching the model's expected shape
# -------------------------------------------------------------------
# Same input shapes we use in the analytical model
# Note: LSTM input is actually int64 (elem_type=7) in the ONNX model
INPUT_CONFIGS = {
    'resnet':  {'shape': (3, 224, 224), 'dtype': np.float32},
    'mobnet':  {'shape': (3, 224, 224), 'dtype': np.float32},
    'yolo':    {'shape': (3, 640, 640), 'dtype': np.float32},
    'bert':    {'shape': (128,),        'dtype': np.int64},
    'lstm':    {'shape': (32,),         'dtype': np.int64},
}

# Default sizes for dynamic dims we might encounter
DYNAMIC_DIM_DEFAULTS = {
    'sequence_length': 32,
    'seq_len': 32,
    'batch_size': 1,  # overridden below
    'batch': 1,
}

def make_dummy_input(session, model_key, batch_size):
    """
    Create random input tensors for ALL inputs the model expects.
    Some models like LSTM have multiple inputs (input, hidden state, etc).
    """
    all_inputs = session.get_inputs()
    input_dict = {}

    for inp in all_inputs:
        name = inp.name
        onnx_shape = inp.shape
        # onnxruntime elem_type: 1=float32, 7=int64, 6=int32, 10=float16
        onnx_dtype = inp.type  # string like 'tensor(float)' or 'tensor(int64)'

        # Figure out concrete shape
        concrete_shape = []
        for i, dim in enumerate(onnx_shape):
            if isinstance(dim, int) and dim > 0:
                concrete_shape.append(dim)
            elif isinstance(dim, str):
                # Dynamic dim — check if it's a batch dim or something else
                dim_lower = dim.lower()
                if 'batch' in dim_lower or i == 0:
                    concrete_shape.append(batch_size)
                elif dim in DYNAMIC_DIM_DEFAULTS:
                    concrete_shape.append(DYNAMIC_DIM_DEFAULTS[dim])
                else:
                    # Fallback: use our known input config
                    cfg = INPUT_CONFIGS.get(model_key, {})
                    if 'shape' in cfg and i - 1 < len(cfg['shape']):
                        concrete_shape.append(cfg['shape'][i - 1])
                    else:
                        concrete_shape.append(1)
            else:
                # dim is 0 or None — treat as batch
                if i == 0:
                    concrete_shape.append(batch_size)
                else:
                    concrete_shape.append(1)

        # Make sure first dim matches batch_size for inputs that have batch dim
        # (some inputs like LSTM weights don't have batch as first dim)
        # Only override if the ONNX shape had a dynamic first dim
        if len(onnx_shape) > 0 and isinstance(onnx_shape[0], str):
            concrete_shape[0] = batch_size

        # For LSTM hidden states: shape is [2, batch_size, 256]
        # batch dim is at position 1, not 0
        if len(onnx_shape) > 1 and isinstance(onnx_shape[1], str):
            s1 = onnx_shape[1]
            if 'batch' in str(s1).lower():
                concrete_shape[1] = batch_size

        # Special case for LSTM: initial_h and initial_c states
        # The ONNX export sometimes hardcodes batch=1 in the graph shape
        # but at runtime LSTM expects it to match the actual batch size.
        # These inputs have shape [num_directions, batch, hidden_size]
        # so dim[1] needs to be batch_size.
        if model_key == 'lstm' and len(concrete_shape) == 3 and concrete_shape[0] <= 2:
            if concrete_shape[1] != batch_size:
                concrete_shape[1] = batch_size

        # Figure out dtype from ONNX type string
        if 'int64' in onnx_dtype:
            dtype = np.int64
        elif 'int32' in onnx_dtype:
            dtype = np.int32
        elif 'float16' in onnx_dtype or 'float' in onnx_dtype:
            dtype = np.float32
        else:
            dtype = np.float32

        # Generate random data
        if dtype in (np.int64, np.int32):
            data = np.random.randint(0, 100, size=concrete_shape).astype(dtype)
        else:
            data = np.random.randn(*concrete_shape).astype(dtype)

        input_dict[name] = data
        print(f"  Input '{name}': shape={concrete_shape}, dtype={dtype}")

    return input_dict


# -------------------------------------------------------------------
# 3. Read power from Jetson's hardware sensors
# -------------------------------------------------------------------
def read_power_watts():
    """
    Try to read the board's current power draw in Watts.
    Tries sysfs (dynamic hwmon discovery) first, then jtop, then tegrastats.
    Returns -1 if nothing works.
    """
    # Method 1: Find INA3221 power sensor dynamically
    # On Orin Nano 8GB, the hwmon number varies between boots
    try:
        import glob
        # Scan all hwmon devices for the INA3221 chip
        for hwmon_dir in sorted(glob.glob('/sys/class/hwmon/hwmon*')):
            name_file = os.path.join(hwmon_dir, 'name')
            if os.path.exists(name_file):
                with open(name_file) as f:
                    if 'ina3221' in f.read().strip().lower():
                        # Found INA3221 — read VDD_IN (channel 1)
                        # Power = Voltage × Current
                        volt_file = os.path.join(hwmon_dir, 'in1_input')
                        curr_file = os.path.join(hwmon_dir, 'curr1_input')
                        if os.path.exists(volt_file) and os.path.exists(curr_file):
                            with open(volt_file) as vf:
                                voltage_mv = float(vf.read().strip())
                            with open(curr_file) as cf:
                                current_ma = float(cf.read().strip())
                            return (voltage_mv * current_ma) / 1e6  # mV*mA → W
    except Exception:
        pass

    # Method 1b: Also try the old-style power_input files (some JetPack versions)
    try:
        import glob
        for power_file in glob.glob('/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/power1_input'):
            if os.path.exists(power_file):
                with open(power_file) as f:
                    return float(f.read().strip()) / 1e6  # microwatts → watts
    except Exception:
        pass

    # Method 2: Try jtop library (jetson-stats)
    try:
        from jtop import jtop
        with jtop() as jetson:
            if jetson.ok():
                pw = jetson.power
                for key in ['tot', 'total', 'POM_5V_IN', 'VDD_IN']:
                    if key in pw:
                        val = pw[key]
                        if isinstance(val, dict):
                            return val.get('power', val.get('avg', 0)) / 1000.0
                        return float(val) / 1000.0
    except Exception:
        pass

    # Method 3: Try parsing tegrastats output
    try:
        result = subprocess.run(
            ['tegrastats', '--interval', '100'],
            capture_output=True, text=True, timeout=2
        )
        for line in result.stdout.split('\n'):
            for token in line.split():
                if 'VDD_IN' in token:
                    parts = token.split('/')
                    if len(parts) >= 2:
                        return float(parts[1]) / 1000.0
    except Exception:
        pass

    return -1.0  # couldn't read power


# -------------------------------------------------------------------
# 4. Benchmark: warmup + timed runs
# -------------------------------------------------------------------
def benchmark(session, input_dict, warmup=10, num_runs=50):
    """
    Run the model many times and measure how long each run takes.
    Also samples power before and after each run.
    """
    # Warmup — let GPU clocks settle and caches warm up
    print(f"  Warmup ({warmup} runs)...")
    for _ in range(warmup):
        session.run(None, input_dict)

    # Actual timed runs
    print(f"  Benchmarking ({num_runs} runs)...")
    latencies = []
    powers = []

    for i in range(num_runs):
        power_before = read_power_watts()

        t0 = time.perf_counter()
        session.run(None, input_dict)
        t1 = time.perf_counter()

        power_after = read_power_watts()

        latency_ms = (t1 - t0) * 1000.0
        latencies.append(latency_ms)

        # Average power = (before + after) / 2
        if power_before > 0 and power_after > 0:
            powers.append((power_before + power_after) / 2.0)
        elif power_before > 0:
            powers.append(power_before)

    avg_latency = float(np.mean(latencies))
    std_latency = float(np.std(latencies))
    avg_power = float(np.mean(powers)) if powers else -1.0

    # Energy per inference = Power * Time
    # Since power is in Watts and latency in ms, P*T gives millijoules
    energy_mj = avg_power * avg_latency if avg_power > 0 else -1.0

    batch_size = list(input_dict.values())[0].shape[0]
    throughput = (batch_size * 1000.0) / avg_latency  # samples per second

    return {
        'avg_latency_ms': round(avg_latency, 3),
        'std_latency_ms': round(std_latency, 3),
        'avg_power_w': round(avg_power, 2),
        'energy_mj': round(energy_mj, 2),
        'throughput_sps': round(throughput, 2),
    }


# -------------------------------------------------------------------
# 5. Main
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Run ONNX inference on Jetson and measure latency + power')
    parser.add_argument('--model', required=True, help='Path to ONNX model')
    parser.add_argument('--model_key', required=True,
                        choices=['resnet', 'mobnet', 'yolo', 'bert', 'lstm'],
                        help='Which model (for input shape)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--warmup', type=int, default=10, help='Warmup runs')
    parser.add_argument('--num_runs', type=int, default=50, help='Timed runs')
    parser.add_argument('--output_csv', default=None, help='Append results to this CSV')
    parser.add_argument('--power_mode', default='unknown', help='Current power mode label (for CSV)')

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Model: {os.path.basename(args.model)}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Power Mode: {args.power_mode}")
    print(f"{'='*60}")

    session = load_model(args.model)
    input_dict = make_dummy_input(session, args.model_key, args.batch_size)
    results = benchmark(session, input_dict, args.warmup, args.num_runs)

    print(f"\n  Results:")
    print(f"    Avg Latency:  {results['avg_latency_ms']:.3f} ms (std={results['std_latency_ms']:.3f})")
    print(f"    Avg Power:    {results['avg_power_w']:.2f} W")
    print(f"    Energy/Infer: {results['energy_mj']:.2f} mJ")
    print(f"    Throughput:   {results['throughput_sps']:.2f} samples/sec")
    print(f"{'='*60}\n")

    # Save to CSV if requested
    if args.output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
        file_exists = os.path.exists(args.output_csv)
        with open(args.output_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    'model', 'model_key', 'batch_size', 'power_mode',
                    'avg_latency_ms', 'std_latency_ms', 'avg_power_w',
                    'energy_mj', 'throughput_sps'
                ])
            writer.writerow([
                os.path.basename(args.model), args.model_key,
                args.batch_size, args.power_mode,
                results['avg_latency_ms'], results['std_latency_ms'],
                results['avg_power_w'], results['energy_mj'],
                results['throughput_sps']
            ])
        print(f"  Results appended to: {args.output_csv}")


if __name__ == '__main__':
    main()
