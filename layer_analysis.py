"""
Layer Analysis (3-Tier DVFS)
----------------------------
Takes an ONNX model, goes through it layer by layer, and calculates:
  - FLOPs (how much computation each layer does)
  - Memory accesses (how much data each layer reads/writes)
  - Arithmetic Intensity (FLOPs / Memory) for each layer
  - Whether each layer is compute-bound or memory-bound under each Jetson power mode:
      * 15W mode (Ridge: 18.36 FLOP/Byte)
      * 25W mode (Ridge: 18.37 FLOP/Byte)
      * MAXN_SUPER mode (Ridge: 20.41 FLOP/Byte)

How to run:
    python layer_analysis.py --model_path onnx_models/resnet50.onnx --model_key resnet --batch_size 1
    python layer_analysis.py --model_path onnx_models/mobnet.onnx --model_key mobnet --batch_size 1
"""

import os
import sys
import io
import argparse
import json
import csv

# Fix for Windows printing issues
if __name__ == '__main__' and sys.platform == 'win32' and hasattr(sys.stdout, 'buffer'):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

import src.analytical_model.estimates as estimates
import src.analytical_model.estimates_lstm as estimates_lstm
from hardware_config import (
    DEFAULT_PEAK_COMPUTE, DEFAULT_PEAK_BANDWIDTH, DEFAULT_RIDGE_POINT,
    RIDGE_15W, RIDGE_25W, RIDGE_MAXN
)


# What input shape each model expects
# These match what we used when exporting the ONNX models
INPUT_SIZE_MAP = {
    'bert': (128,),           # 128 tokens
    'resnet': (3, 224, 224),  # 3-channel 224x224 image
    'mobnet': (3, 224, 224),  # same as resnet
    'yolo': (3, 640, 640),   # 3-channel 640x640 image
    'lstm': (32,),            # 32 timesteps
    'fcn': (256,),
    'pw_conv': (160, 7, 7),
    'bottleneck': (160, 56, 56),
}


def analyze_model(model_path, model_key, batch_size=1,
                  peak_compute_gflops=None, peak_bandwidth_gbps=None,
                  quiet=False):
    """
    Go through each layer of an ONNX model and calculate FLOPs, memory, AI.

    Args:
        model_path: path to the .onnx file
        model_key: which model it is (e.g. 'resnet', 'bert')
        batch_size: how many inputs at once
        peak_compute_gflops: GPU peak compute speed (default from hardware_config)
        peak_bandwidth_gbps: GPU memory bandwidth (default from hardware_config)
        quiet: if True, don't print anything

    Returns:
        dict with total FLOPs, total memory, overall AI, per-layer details, etc.
    """
    if peak_compute_gflops is None:
        peak_compute_gflops = DEFAULT_PEAK_COMPUTE
    if peak_bandwidth_gbps is None:
        peak_bandwidth_gbps = DEFAULT_PEAK_BANDWIDTH

    if model_key not in INPUT_SIZE_MAP:
        raise ValueError(
            f"Unknown model key '{model_key}'. "
            f"Available keys: {list(INPUT_SIZE_MAP.keys())}"
        )

    input_size = INPUT_SIZE_MAP[model_key]

    # Baseline Ridge point = peak compute / peak bandwidth (MAXN_SUPER baseline)
    ridge_point = peak_compute_gflops / peak_bandwidth_gbps

    # Run the analytical model to get per-layer info
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        if model_key == 'lstm':
            collector = estimates_lstm.LayerInfoCollectorONNX(
                model_path, input_size, batch_size=batch_size
            )
        else:
            collector = estimates.LayerInfoCollectorONNX(
                model_path, input_size, batch_size=batch_size
            )
        layer_infos = collector.run()

    # Go through each layer and classify it across all 3 ridge points
    layers = []
    total_flops = 0
    total_memory = 0
    compute_bound_flops = 0
    cb_flops_15w = 0
    cb_flops_25w = 0
    cb_flops_maxn = 0

    for layer_name, info in layer_infos.items():
        flops = info.get('FLOPs', 0)
        mem = info.get('Memory Accesses', 0)
        layer_type = info.get('layer_type', 'Unknown')
        layer_ai = info.get('AI', None)

        # Skip layers that don't do any real work (like Reshape, Flatten)
        if flops == 0 and mem == 0:
            continue

        total_flops += flops
        total_memory += mem

        # Classification relative to baseline (MAXN_SUPER)
        if layer_ai is not None and layer_ai >= ridge_point:
            bound = "COMPUTE"
            compute_bound_flops += flops
        else:
            bound = "MEMORY"

        # Classification relative to each specific mode's ridge point
        bound_15w = "COMPUTE" if (layer_ai is not None and layer_ai >= RIDGE_15W) else "MEMORY"
        bound_25w = "COMPUTE" if (layer_ai is not None and layer_ai >= RIDGE_25W) else "MEMORY"
        bound_maxn = "COMPUTE" if (layer_ai is not None and layer_ai >= RIDGE_MAXN) else "MEMORY"

        if bound_15w == "COMPUTE":
            cb_flops_15w += flops
        if bound_25w == "COMPUTE":
            cb_flops_25w += flops
        if bound_maxn == "COMPUTE":
            cb_flops_maxn += flops

        layers.append({
            'name': layer_name,
            'type': layer_type,
            'flops': flops,
            'memory': mem,
            'ai': layer_ai if layer_ai is not None else 0.0,
            'bound': bound,
            'bound_15w': bound_15w,
            'bound_25w': bound_25w,
            'bound_maxn': bound_maxn,
        })

    # Calculate overall model numbers
    total_ai = total_flops / total_memory if total_memory > 0 else 0.0
    total_regime = "Compute-Bound" if total_ai >= ridge_point else "Memory-Bound"
    compute_ratio = compute_bound_flops / total_flops if total_flops > 0 else 0.0

    result = {
        'model_path': model_path,
        'model_key': model_key,
        'batch_size': batch_size,
        'ridge_point': ridge_point,
        'ridge_15w': RIDGE_15W,
        'ridge_25w': RIDGE_25W,
        'ridge_maxn': RIDGE_MAXN,
        'peak_compute_gflops': peak_compute_gflops,
        'peak_bandwidth_gbps': peak_bandwidth_gbps,
        'total_flops': total_flops,
        'total_memory': total_memory,
        'total_ai': total_ai,
        'total_regime': total_regime,
        'regime_15w': 'Compute-Bound' if total_ai >= RIDGE_15W else 'Memory-Bound',
        'regime_25w': 'Compute-Bound' if total_ai >= RIDGE_25W else 'Memory-Bound',
        'regime_maxn': 'Compute-Bound' if total_ai >= RIDGE_MAXN else 'Memory-Bound',
        'compute_bound_flops': compute_bound_flops,
        'compute_ratio': compute_ratio,
        'compute_ratio_15w': cb_flops_15w / total_flops if total_flops > 0 else 0.0,
        'compute_ratio_25w': cb_flops_25w / total_flops if total_flops > 0 else 0.0,
        'compute_ratio_maxn': cb_flops_maxn / total_flops if total_flops > 0 else 0.0,
        'layers': layers,
    }

    return result


def print_report(result):
    """Print a readable table showing each layer's stats."""
    model_name = os.path.basename(result['model_path'])
    ridge = result['ridge_point']

    print()
    print("=" * 105)
    print(f"  LAYER-BY-LAYER ANALYSIS (3-TIER ROOFLINE)")
    print(f"  Model: {model_name} | Batch Size: {result['batch_size']}")
    print(f"  Hardware: {result['peak_compute_gflops']} GFLOP/s compute, "
          f"{result['peak_bandwidth_gbps']} GB/s bandwidth")
    print(f"  Ridge Points: 15W={RIDGE_15W:.2f} | 25W={RIDGE_25W:.2f} | MAXN_SUPER={RIDGE_MAXN:.2f} FLOP/Byte")
    print("=" * 105)
    print()

    # Table header
    header = (
        f"{'#':<4} | {'Layer Name':<38} | {'Type':<12} | "
        f"{'FLOPs (M)':<10} | {'Mem (MB)':<10} | {'AI':<8} | {'15W':<6} | {'25W':<6} | {'MAXN':<6}"
    )
    print(header)
    print("-" * 105)

    significant_layers = []
    for i, layer in enumerate(result['layers'], 1):
        pct = (layer['flops'] / result['total_flops'] * 100) if result['total_flops'] > 0 else 0
        if pct >= 0.1:
            significant_layers.append((i, layer))
            name_short = layer['name'][:36] + '..' if len(layer['name']) > 38 else layer['name']
            row = (
                f"{i:<4} | {name_short:<38} | {layer['type']:<12} | "
                f"{layer['flops']/1e6:<10.2f} | {layer['memory']/1e6:<10.2f} | "
                f"{layer['ai']:<8.2f} | {layer['bound_15w'][:3]:<6} | {layer['bound_25w'][:3]:<6} | {layer['bound_maxn'][:3]:<6}"
            )
            print(row)

    print("-" * 105)
    print()
    print(f"  TOTALS:")
    print(f"    Total FLOPs:       {result['total_flops']/1e9:.3f} GFLOPs")
    print(f"    Total Memory:      {result['total_memory']/1e6:.2f} MB")
    print(f"    Arithmetic Int:    {result['total_ai']:.2f} FLOP/Byte")
    print(f"    Regime (15W):      {result['regime_15w']}")
    print(f"    Regime (25W):      {result['regime_25w']}")
    print(f"    Regime (MAXN):     {result['regime_maxn']}")
    print(f"    Compute Ratio:     {result['compute_ratio']*100:.1f}% of FLOPs in compute-bound layers")
    print(f"    Layers Analyzed:   {len(result['layers'])} ({len(significant_layers)} shown above)")
    print()
    print("=" * 105)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze ONNX model layer by layer for FLOPs, memory, and arithmetic intensity.'
    )
    parser.add_argument('--model_path', required=True, help='Path to the ONNX model file.')
    parser.add_argument('--model_key', required=True,
                        help=f'Model key for input size. Options: {list(INPUT_SIZE_MAP.keys())}')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (default: 1).')
    parser.add_argument('--peak_compute', type=float, default=DEFAULT_PEAK_COMPUTE,
                        help=f'Peak compute in GFLOP/s (default: {DEFAULT_PEAK_COMPUTE} for MAXN_SUPER).')
    parser.add_argument('--peak_bandwidth', type=float, default=DEFAULT_PEAK_BANDWIDTH,
                        help=f'Peak memory bandwidth in GB/s (default: {DEFAULT_PEAK_BANDWIDTH} for MAXN_SUPER).')
    parser.add_argument('--output_json', default=None, help='Save results to JSON file.')
    parser.add_argument('--output_csv', default=None, help='Save layer results to CSV file.')

    args = parser.parse_args()

    result = analyze_model(
        model_path=args.model_path,
        model_key=args.model_key,
        batch_size=args.batch_size,
        peak_compute_gflops=args.peak_compute,
        peak_bandwidth_gbps=args.peak_bandwidth,
    )

    print_report(result)

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.output_json}")

    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or '.', exist_ok=True)
        with open(args.output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['layer_name', 'layer_type', 'flops', 'memory_bytes', 'ai', 'bound_15w', 'bound_25w', 'bound_maxn'])
            for layer in result['layers']:
                writer.writerow([
                    layer['name'], layer['type'], layer['flops'],
                    layer['memory'], round(layer['ai'], 4),
                    layer['bound_15w'], layer['bound_25w'], layer['bound_maxn']
                ])
        print(f"Layer CSV saved to {args.output_csv}")


if __name__ == '__main__':
    main()
