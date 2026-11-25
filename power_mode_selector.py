"""
Power Mode Selector (3-Tier DVFS)
---------------------------------
Given an ONNX model, this script figures out the best Jetson power mode
(15W, 25W, or MAXN_SUPER) based on Roofline analysis.

On the NVIDIA Jetson Orin Nano Super (8GB), we have 3 physical power modes:
  - 15W        (nvpmodel -m 0) — low power, 612 MHz GPU, 2133 MHz EMC (Ridge: 18.36 FLOP/Byte)
  - 25W        (nvpmodel -m 1) — balanced,  918 MHz GPU, 3199 MHz EMC (Ridge: 18.37 FLOP/Byte)
  - MAXN_SUPER (nvpmodel -m 2) — max perf, 1020 MHz GPU, 3199 MHz EMC (Ridge: 20.41 FLOP/Byte)

How to run:
    python power_mode_selector.py --model_path onnx_models/resnet50.onnx --model_key resnet
    python power_mode_selector.py --model_path onnx_models/lstm.onnx --model_key lstm --batch_size 1
"""

import os
import sys
import io
import argparse

# Fix for Windows printing issues
if __name__ == '__main__' and sys.platform == 'win32' and hasattr(sys.stdout, 'buffer'):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

from layer_analysis import analyze_model, INPUT_SIZE_MAP
from hardware_config import (
    POWER_MODES, MODE_15W, MODE_25W, MODE_MAXN,
    DEFAULT_PEAK_COMPUTE, DEFAULT_PEAK_BANDWIDTH, DEFAULT_RIDGE_POINT,
    RIDGE_15W, RIDGE_25W, RIDGE_MAXN,
    select_power_mode as select_mode_config
)


def select_power_mode(compute_ratio, overall_ai=0.0, ridge_point=None, batch_size=1):
    """
    Pick the optimal Jetson power mode among 15W, 25W, and MAXN_SUPER
    based on First-Principles Roofline Analysis.

    Args:
        compute_ratio: fraction of FLOPs in compute-bound layers (0 to 1)
        overall_ai: model overall arithmetic intensity (FLOPs / Bytes)
        ridge_point: reference ridge point (default: MAXN_SUPER ridge 20.41)
        batch_size: workload batch size

    Returns:
        (mode, reason, expected_saving) tuple
    """
    if ridge_point is None:
        ridge_point = RIDGE_MAXN

    mode = select_mode_config(ai=overall_ai, compute_ratio=compute_ratio, batch_size=batch_size)

    if mode == MODE_MAXN:
        reason = (
            f"Compute-dominant workload (AI={overall_ai:.2f} FLOP/Byte, CR={compute_ratio*100:.1f}%): "
            f"Computation intensity is near or above the ridge point ({ridge_point:.2f} FLOP/Byte). "
            f"Downclocking GPU execution units degrades latency significantly. "
            f"MAXN_SUPER preserves throughput and satisfies the real-time latency SLA (0% delay penalty)."
        )
        expected_saving = "0% latency penalty (throughput conserved)"

    elif mode == MODE_25W:
        reason = (
            f"Intermediate operating point (AI={overall_ai:.2f} FLOP/Byte, CR={compute_ratio*100:.1f}%, BS={batch_size})."
        )
        expected_saving = "Balanced Profile"

    else:  # MODE_15W
        reason = (
            f"Memory-dominant workload (AI={overall_ai:.2f} FLOP/Byte, CR={compute_ratio*100:.1f}% < 70%): "
            f"Arithmetic intensity is well below the ridge point ({ridge_point:.2f} FLOP/Byte). "
            f"Under classical Roofline theory, GPU execution units are memory-stalled, suggesting "
            f"low power mode (15W) can reduce dynamic power consumption."
        )
        expected_saving = "Low Power Profile (Energy Priority)"

    return (mode, reason, expected_saving)


def print_recommendation(result, mode, reason, expected_saving):
    """Print the recommendation in a readable format."""
    model_name = os.path.basename(result['model_path'])

    print()
    print("=" * 70)
    print("  POWER MODE RECOMMENDATION (3-TIER DVFS)")
    print("=" * 70)
    print()
    print(f"  Model:              {model_name}")
    print(f"  Batch Size:         {result['batch_size']}")
    print(f"  Total AI:           {result['total_ai']:.2f} FLOP/Byte ({result['total_regime']})")
    print(f"  15W Ridge:          {RIDGE_15W:.2f} FLOP/Byte (Mode 0: 612 MHz GPU, 2133 MHz EMC)")
    print(f"  25W Ridge:          {RIDGE_25W:.2f} FLOP/Byte (Mode 1: 918 MHz GPU, 3199 MHz EMC)")
    print(f"  MAXN_SUPER Ridge:   {RIDGE_MAXN:.2f} FLOP/Byte (Mode 2: 1020 MHz GPU, 3199 MHz EMC)")
    print()
    print(f"  Total Layers:       {len(result['layers'])}")

    cb_layers = [l for l in result['layers'] if l['bound'] == 'COMPUTE' and l['flops'] > 0]
    mb_layers = [l for l in result['layers'] if l['bound'] == 'MEMORY' and l['flops'] > 0]
    print(f"  Compute-Bound:      {len(cb_layers)} layers (relative to baseline ridge {RIDGE_MAXN:.2f})")
    print(f"  Memory-Bound:       {len(mb_layers)} layers (relative to baseline ridge {RIDGE_MAXN:.2f})")
    print()
    print(f"  Compute Ratio:      {result['compute_ratio']*100:.1f}%")
    print()
    print("-" * 70)
    print(f"  >> RECOMMENDED MODE:     {mode}")
    print(f"  >> EXPECTED SAVING:      {expected_saving}")
    print("-" * 70)
    print()
    print(f"  Reason: {reason}")
    print()

    if cb_layers:
        cb_layers_sorted = sorted(cb_layers, key=lambda x: x['flops'], reverse=True)
        print("  Top compute-heavy layers:")
        for i, layer in enumerate(cb_layers_sorted[:5], 1):
            pct = layer['flops'] / result['total_flops'] * 100
            print(f"    {i}. {layer['name'][:50]} "
                  f"(AI={layer['ai']:.1f}, {pct:.1f}% of total FLOPs)")
        print()

    print("=" * 70)


def run_selector(model_path, model_key, batch_size=1,
                 peak_compute=None, peak_bandwidth=None,
                 quiet=False):
    """
    Full pipeline: analyze model -> pick power mode -> return result dict.
    """
    if peak_compute is None:
        peak_compute = DEFAULT_PEAK_COMPUTE
    if peak_bandwidth is None:
        peak_bandwidth = DEFAULT_PEAK_BANDWIDTH

    result = analyze_model(
        model_path=model_path,
        model_key=model_key,
        batch_size=batch_size,
        peak_compute_gflops=peak_compute,
        peak_bandwidth_gbps=peak_bandwidth,
    )

    mode, reason, expected_saving = select_power_mode(
        compute_ratio=result['compute_ratio'],
        overall_ai=result['total_ai'],
        ridge_point=result['ridge_point'],
        batch_size=batch_size,
    )

    if not quiet:
        print_recommendation(result, mode, reason, expected_saving)

    return {
        'model_path': model_path,
        'model_key': model_key,
        'batch_size': batch_size,
        'total_ai': result['total_ai'],
        'total_regime': result['total_regime'],
        'regime_15w': 'Compute-Bound' if result['total_ai'] >= RIDGE_15W else 'Memory-Bound',
        'regime_25w': 'Compute-Bound' if result['total_ai'] >= RIDGE_25W else 'Memory-Bound',
        'regime_maxn': 'Compute-Bound' if result['total_ai'] >= RIDGE_MAXN else 'Memory-Bound',
        'compute_ratio': result['compute_ratio'],
        'recommended_mode': mode,
        'expected_saving': expected_saving,
        'reason': reason,
        'total_flops': result['total_flops'],
        'total_memory': result['total_memory'],
        'ridge_point': result['ridge_point'],
        'ridge_15w': RIDGE_15W,
        'ridge_25w': RIDGE_25W,
        'ridge_maxn': RIDGE_MAXN,
        'num_layers': len(result['layers']),
        'num_compute_bound': len([l for l in result['layers'] if l['bound'] == 'COMPUTE' and l['flops'] > 0]),
        'num_memory_bound': len([l for l in result['layers'] if l['bound'] == 'MEMORY' and l['flops'] > 0]),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Pick the best Jetson power mode for an ONNX model using 3-tier roofline analysis.'
    )
    parser.add_argument('--model_path', required=True, help='Path to the ONNX model file.')
    parser.add_argument('--model_key', required=True,
                        help=f'Model key for input size. Options: {list(INPUT_SIZE_MAP.keys())}')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (default: 1).')
    parser.add_argument('--peak_compute', type=float, default=DEFAULT_PEAK_COMPUTE,
                        help=f'Peak compute in GFLOP/s (default: {DEFAULT_PEAK_COMPUTE} for MAXN_SUPER).')
    parser.add_argument('--peak_bandwidth', type=float, default=DEFAULT_PEAK_BANDWIDTH,
                        help=f'Peak memory bandwidth in GB/s (default: {DEFAULT_PEAK_BANDWIDTH} for MAXN_SUPER).')

    args = parser.parse_args()

    run_selector(
        model_path=args.model_path,
        model_key=args.model_key,
        batch_size=args.batch_size,
        peak_compute=args.peak_compute,
        peak_bandwidth=args.peak_bandwidth,
    )


if __name__ == '__main__':
    main()
