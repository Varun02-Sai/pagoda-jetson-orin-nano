# Jetson Hardware Benchmark Suite (3-Tier DVFS)

These scripts run directly on the NVIDIA Jetson Orin Nano Super (8GB) board.
They measure real inference latency, power, and energy across all 3 physical power modes:
- **15W** (Mode 0): GPU 612 MHz, EMC 2133 MHz (Ridge: 18.36 FLOP/Byte)
- **25W** (Mode 1): GPU 918 MHz, EMC 3199 MHz (Ridge: 18.37 FLOP/Byte)
- **MAXN_SUPER** (Mode 2): GPU 1020 MHz, EMC 3199 MHz (Ridge: 20.41 FLOP/Byte)

## Files
- `get_hardware_specs.py` — queries sysfs for live clocks, bandwidth, and roofline ridge points
- `run_inference.py` — runs one model/batch/mode combination and measures latency + power
- `sweep_all.py` — full automated sweep (5 models x 4 batch sizes x 3 modes = 60 configs)
- `verify_predictions.py` — standalone verifier: compares analytical predictions vs physical hardware
- `generate_hw_charts.py` — generates publication charts from benchmark results
- `run_hardware_test.sh` — one-command setup and automated execution

## How to Run on Jetson
### Run Full 60-Configuration Sweep:
```bash
cd ~/pagoda/jetson
chmod +x run_hardware_test.sh
sudo bash run_hardware_test.sh
```

### Verify Predictions vs Actual Hardware:
```bash
cd ~/pagoda/jetson
python3 verify_predictions.py
```

## Output
All results are saved inside `jetson/results/`:
- `jetson_hardware_results.csv` — raw measurements (60 configurations)
- `fig_latency_comparison.png` — latency across modes
- `fig_energy_comparison.png` — energy per inference across modes
- `fig_power_draw.png` — dynamic power draw measured via INA3221 sensor
- `fig_energy_savings.png` — energy savings % relative to MAXN_SUPER baseline
