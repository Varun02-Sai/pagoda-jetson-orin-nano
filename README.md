# Pagoda: Energy and Time Roofline Study for DNN Workloads on NVIDIA Jetson Orin Nano (8GB)

**B.Tech Project (BTP-2) — Department of Data Science & Engineering, IIT Palakkad**  
- **Author:** Jillapelli Sai Varun (`142301040`)  
- **Mentor:** Prof. Abhinandan S P  
- **Target Hardware:** NVIDIA Jetson Orin Nano Developer Kit (8GB)  

---

## Overview

Adapting the **Pagoda** Roofline methodology (originally developed at IISc Bangalore for the 60W Jetson AGX Orin, `arXiv:2509.20189`), this project builds an offline ONNX analytical DVFS power-mode selector (`< 2 seconds` on a host PC) and validates its recommendations against non-circular physical hardware measurements across **60 workload configurations** (`5 models × 4 batch sizes × 3 power modes`) using onboard Texas Instruments **INA3221** sysfs power telemetry.

### Jetson Orin Nano Super (8GB) Hardware Envelope
- **GPU Architecture:** NVIDIA Ampere SM 8.7 (8 SMs, 1024 CUDA cores, 2 FLOPs/cycle FMA)
- **Memory Subsystem:** 8 GB 128-bit (16-Byte) LPDDR5 unified DRAM

| `nvpmodel` Mode | GPU Clock | Memory (EMC) | Peak Compute | Peak Bandwidth | Ridge Point ($I^*$) | Empirical Finding |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Mode 0 (`15W`)** | 612 MHz | 2133 MHz | 1253.38 GFLOP/s | 68.26 GB/s | **18.36 FLOP/B** | Coupled EMC clock drops BW `-33.3%` |
| **Mode 1 (`25W`)** | 918 MHz | 3199 MHz | 1880.06 GFLOP/s | 102.37 GB/s | **18.37 FLOP/B** | Pareto-inefficient across 19/20 runs |
| **Mode 2 (`MAXN_SUPER`)** | 1020 MHz | 3199 MHz | 2088.96 GFLOP/s | 102.37 GB/s | **20.41 FLOP/B** | Fastest baseline & race-to-sleep optimum |

---

## Key Empirical Findings

1. **100% Compute-Bound SLA Protection (`15/15`):** All 15 compute-dominant configurations (BERT-Base, ResNet-50, YOLOv8n, LSTM `BS >= 4`) are preserved in `MAXN_SUPER` with `0%` latency degradation—preventing up to **461.2 ms** slowdown on BERT-Base BS=16 vs. `15W` (and **717.6 ms** vs. `25W`).
2. **Strict 15% SLA Match Rate (`14/20 = 70.0%`):** Evaluated against independent physical ground truth (lowest energy mode within `+15%` latency SLA relative to `MAXN_SUPER`), the analytical model achieves 70.0% exact alignment, with `YOLOv8n BS=16` standing out as the lone physical win-win (`+25.9%` energy savings at `+0.8%` latency cost in `15W`).
3. **25W Mode is Pareto-Inefficient (`19/20` Workloads):** Running in `25W` consumes **+2.9% to +11.0% MORE energy** than `MAXN_SUPER` while slowing inference by **+17.4% to +29.1%**, rendering Orin Nano DVFS effectively **bimodal (`MAXN_SUPER` vs. `15W`)**.
4. **Coupled Memory Bus Scaling (`-33.3%` Bandwidth at `15W`):** Unlike the AGX Orin paper where memory frequency scales independently, `15W` on Orin Nano downclocks EMC (`3199 → 2133 MHz`). Memory-bound models (`MobileNet-v3`, `LSTM BS=1`) genuinely save `+5.8% to +13.7%` energy in `15W`, but suffer `+21.1% to +28.9%` latency penalties above the 15% SLA limit.

---

## Repository Structure

```text
├── hardware_config.py               # Canonical Jetson Orin Nano specs, ridge points & 2-regime DVFS policy
├── layer_analysis.py                # Per-layer & aggregate ONNX FLOP/DRAM/AI analyzer
├── power_mode_selector.py           # CLI tool recommending MAXN_SUPER vs 15W from ONNX graph
├── generate_multi_batch_analysis.py # Sweeps BS in {1,2,4,8,16,32} analytically -> multi_batch_consolidated.csv
├── generate_roofline_csv.py         # Synchronizes results/roofline_data.csv
├── generate_paper_plots.py          # Generates 3-tier Roofline & heatmap plots
├── generate_ppt_charts.py           # Generates hardware latency, stddev, power & Pareto plots
├── verify_predictions.py            # Non-circular verification against physical Jetson measurements
├── jetson/                          # Standalone on-device Jetson Orin Nano benchmark suite
│   ├── get_hardware_specs.py        # Queries sysfs GPU/EMC clocks and prints live Ridge Points
│   ├── run_inference.py             # ORT TensorRT/CUDA runner + TI INA3221 sysfs power polling
│   ├── sweep_all.py                 # Automated 60-config nvpmodel + jetson_clocks sweep (~45 min)
│   ├── verify_predictions.py        # Standalone verification script for Jetson board
│   ├── generate_hw_charts.py        # Hardware latency/energy chart generator
│   └── run_hardware_test.sh         # One-command Jetson sweep runner
├── src/
│   ├── analytical_model/            # Upstream Pagoda ONNX FLOPs/Bytes estimators (with sequence_length & CLI fixes)
│   ├── analysis/                    # ONNX export scripts (export_all_models.py) & exploration notebooks
│   ├── scripts/                     # Upstream NCU automation helpers
│   └── tools/                       # Upstream ERT & NCU post-processing utilities
├── results/                         # Verified hardware CSVs (60 configs), SLA comparison table & charts
├── jetson/results/                  # On-device hardware output directory
├── onnx_models/                     # Exported ONNX benchmarks (bert-base via Git LFS)
└── reports/                         # Final BTP reports (.pdf, .docx, .md), presentations (.pptx), scripts & figures/
```

---

## Quick Start

### 1. Install Dependencies & Export ONNX Models
Large `.onnx` weights (>100 MB) are excluded from Git tracking via `.gitignore`. Regenerate all 5 ONNX benchmarks (`MobileNet-v3`, `LSTM`, `YOLOv8n`, `BERT-Base`, `ResNet-50`) locally:
```bash
pip install -r requirements.txt
python src/analysis/export_all_models.py
```

### 2. Run Offline Analytical Profiling (< 2 Seconds)
```bash
python layer_analysis.py --model_path onnx_models/resnet50.onnx --model_key resnet --batch_size 1
python power_mode_selector.py --model_path onnx_models/mobnet.onnx --model_key mobnet --batch_size 1
python generate_multi_batch_analysis.py
```

### 3. Run Physical Hardware Sweep (On NVIDIA Jetson Orin Nano)
```bash
cd jetson
sudo bash run_hardware_test.sh
python3 verify_predictions.py
```

### 4. Re-Verify Predictions & Regenerate Reports Locally
```bash
python verify_predictions.py
python reports/generate_docx_report.py
```

---

## Attribution & License

Licensed under the **Apache License 2.0** ([LICENSE](LICENSE)).  
Adapted and extended for the NVIDIA Jetson Orin Nano (8GB) from the upstream Pagoda framework by **DREAM:Lab, IISc Bangalore**:  
> Prashanthi S. K., Kunal Kumar Sahoo, Amartya Ranjan Saikia, Pranav Gupta, Atharva Vinay Joshi, Priyanshu Pansari, and Yogesh Simmhan, *"Pagoda: An Energy and Time Roofline Study for DNN Workloads on Edge Accelerators,"* arXiv:2509.20189 ([dream-lab/pagoda](https://github.com/dream-lab/pagoda)).
