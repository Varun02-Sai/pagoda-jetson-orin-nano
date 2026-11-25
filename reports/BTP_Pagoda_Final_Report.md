# B.Tech Project (BTP-2) Final Report
## Pagoda: Energy and Time Roofline Study for DNN Workloads on NVIDIA Jetson Edge Accelerators

**Candidate:** Jillapelli Sai Varun (`142301040`)  
**Department:** Data Science & Engineering, Indian Institute of Technology Palakkad  
**Faculty Mentor:** Prof. Abhinandan S P (`abhinandansp@iitpkd.ac.in`)  
**Target Platform:** NVIDIA Jetson Orin Nano Developer Kit (8GB)  
**Date:** November 2026  

---

## Executive Summary

Deploying deep neural networks (DNNs) on battery- and thermally-constrained edge accelerators like the **NVIDIA Jetson Orin Nano Developer Kit (8GB)** requires balancing real-time inference latency against strict energy budgets. Statically evaluating every model, batch size, and Dynamic Voltage and Frequency Scaling (DVFS) mode on hardware takes **45+ minutes per sweep** and requires physical root access.

Adapting the **Pagoda** framework (IISc Bangalore, `arXiv:2509.20189`), this project builds an offline ONNX analytical Roofline power-mode selector (`layer_analysis.py`, `power_mode_selector.py`) that profiles computation graphs on a host laptop in **under 2 seconds**, paired with an automated hardware validation suite (`jetson/sweep_all.py`, `run_inference.py`, `verify_predictions.py`) using onboard Texas Instruments **INA3221** sysfs telemetry across **60 physical configurations** (`5 models × 4 batch sizes × 3 power modes`).

### Headline Outcomes
1. **100% Compute-Bound SLA Protection (`15/15`):** All 15 compute-dominant workloads are kept at `MAXN_SUPER` with `0%` latency degradation—preventing up to **461.2 ms** latency slowdown on BERT-Base BS=16 relative to 15W (and **717.6 ms** relative to 25W).
2. **70.0% Strict SLA Match Rate (`14/20`):** Under non-circular physical ground truth (lowest energy mode within `+15.0%` latency SLA of MAXN), the analytical model matches `14/20` configurations (`14/15` MAXN predictions exact; `1` empirical win-win missed conservatively on YOLOv8n BS=16).
3. **25W Mode is Pareto-Inefficient (`19/20` Workloads):** Running in `25W` consumes **+2.9% to +11.0% MORE energy** than `MAXN_SUPER` while slowing execution by **+17.4% to +29.1%**, rendering Orin Nano DVFS effectively **bimodal** (`MAXN_SUPER` vs. `15W`).
4. **Coupled Memory Bus Discovery (`-33.3%` Bandwidth at 15W):** Unlike the paper's 60W AGX Orin (independent clocks), Orin Nano's `15W` profile drops the LPDDR5 memory clock from **3199 MHz to 2133 MHz** alongside the GPU clock (`1020 → 612 MHz`). Memory-bound workloads (MobileNet-v3, LSTM BS=1) save **+5.8% to +13.7% energy** in 15W, but incur **+21.1% to +28.9% latency penalties**, exceeding the 15% SLA.

---

## 1. Introduction & Project Contributions

Edge AI deployments must prevent unnecessary energy burn without sacrificing latency Service Level Agreements (SLAs). While the original Pagoda work demonstrated clean energy-saving "win-win" regions on a high-end 60W Jetson AGX Orin with decoupled clock domains, its validity on resource-constrained consumer boards remained unverified.

### Core Contributions
- **Offline ONNX Analytical Profiler & Fixes:** Resolved an upstream hardcoded `sequence_length = 8` bug in `src/analytical_model/estimates.py` and fixed argument types/imports in `model.py`. Built `layer_analysis.py` and `power_mode_selector.py` for `< 2s` host profiling.
- **Hardware-Specific Orin Nano Adaptation:** Built `hardware_config.py` calibrated from live board telemetry (`get_hardware_specs.py`) and simplified power selection to a verified 2-regime policy (`if compute_ratio >= 0.70 or ai >= 18.36 -> MAXN_SUPER else 15W`).
- **Automated 60-Config INA3221 Benchmark Infrastructure:** Implemented deterministic `nvpmodel` switching, `jetson_clocks` locking, ONNX Runtime GPU execution (`TensorrtExecutionProvider` / `CUDAExecutionProvider`), and direct sysfs INA3221 power polling across 50 timed inferences per configuration.
- **Non-Circular Ground Truth Evaluation:** Validated predictions strictly against physical energy and latency telemetry rather than model-internal heuristics.

---

## 2. Hardware Platform & Roofline Theoretical Formulation

### 2.1 NVIDIA Jetson Orin Nano Developer Kit (8GB)
- **GPU Architecture:** NVIDIA Ampere (SM 8.7) — 8 SMs, 1024 CUDA Cores, 2 FLOPs per FMA cycle.
- **Memory Subsystem:** 8 GB 128-bit (16-Byte) Double Data Rate LPDDR5 shared unified memory.

$$\text{Peak Compute (GFLOP/s)} = 1024 \times 2 \times f_{\text{GPU (GHz)}}, \qquad \text{Peak BW (GB/s)} = 16 \times f_{\text{EMC (GHz)}} \times 2$$

$$\text{Ridge Point } (I^*) = \frac{\text{Peak Compute}}{\text{Peak Memory Bandwidth}} \quad [\text{FLOP/Byte}]$$

| `nvpmodel` ID | Power Mode | GPU Clock | Memory (EMC) | Peak Compute | Peak Bandwidth | Ridge Point ($I^*$) | Role |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Mode 0** | **15W** | 612 MHz | 2133 MHz | 1253.38 GFLOP/s | 68.26 GB/s | **18.36 FLOP/B** | Low-Power Profile (-33.3% BW) |
| **Mode 1** | **25W** | 918 MHz | 3199 MHz | 1880.06 GFLOP/s | 102.37 GB/s | **18.37 FLOP/B** | Pareto-Inefficient (Unused) |
| **Mode 2** | **MAXN_SUPER** | 1020 MHz | 3199 MHz | 2088.96 GFLOP/s | 102.37 GB/s | **20.41 FLOP/B** | Baseline Full Speed |

### 2.2 Time and Energy Roofline Linkage
- **Arithmetic Intensity:** $\text{AI} = \text{Total FLOPs} / \text{Total DRAM Bytes}$ (`float32` = 4 bytes/element across weights, inputs, outputs).
- **Time Roofline:** $T \approx \max\left(\frac{\text{FLOPs}}{P_{\max}}, \frac{\text{Bytes}}{b_s}\right)$.
- **Energy Roofline:** $E = P_{\text{total}} \times T = (P_{\text{dynamic}} + P_{\text{static}}) \times T$.
  - **Compute-Bound ($\text{AI} \ge I^*$):** Lowering GPU clock directly slows $T$; prolonged static power draw wastes energy (**Race-to-Sleep** favors `MAXN_SUPER`).
  - **Memory-Bound ($\text{AI} < I^*$):** Execution is constrained by memory transfers; classical theory predicts lower GPU clocks reduce $P_{\text{dynamic}}$ without inflating $T$.

---

## 3. Analytical Pipeline & Workload Characterization

```
[1. ONNX Model (.onnx)] -> [2. Per-Layer Profiling (FLOPs, Bytes)] -> [3. Layer AI vs Ridge]
     -> [4. Decision: CR >= 70% or AI >= 18.36 -> MAXN_SUPER else 15W] -> [5. Hardware Verification]
```

### Evaluated Workload Summary (Batch Sizes 1 to 16)

| Model | Domain | Input Shape | Active Layers (CB / MB @ BS=1) | Overall AI (BS=1 $\to$ 16) | Compute Ratio (BS=1 $\to$ 16) | Policy |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **MobileNet-v3** | Edge Vision | `(3, 224, 224)` | `12 / 127` | `4.36 -> 5.44` | `37.5% -> 56.6%` | `15W` (All BS) |
| **LSTM** | Sequential NLP | `(32,)` | `0 / 3` | `7.24 -> 49.63` | `0.0% -> 96.6%` | `15W` @ BS=1; `MAXN` @ BS$\ge$4 |
| **YOLOv8n** | Object Detection | `(3, 640, 640)` | `52 / 141` | `16.69 -> 17.07` | `93.1% -> 93.1%` | `MAXN` (CR $\ge 70\%$) |
| **BERT-Base** | Transformer NLP | `(128,)` | `60 / 285` | `25.00 -> 39.38` | `99.7% -> 99.7%` | `MAXN` (All BS) |
| **ResNet-50** | Deep Vision | `(3, 224, 224)` | `52 / 69` | `24.37 -> 34.05` | `99.4% -> 99.4%` | `MAXN` (All BS) |

- **Layer Count vs. FLOP Dominance:** BERT-Base has 285 memory-bound layers vs. 60 compute-bound layers, yet the 60 MatMul/Gemm layers contribute **99.7%** of total FLOPs (ResNet-50: 52 convolutions = **99.4%** FLOPs).
- **LSTM Behavior:** ONNX fuses the 32-timestep recurrent loops into 2 active `LSTM` ops plus 1 `MatMul` projection head (filtering 4 zero-FLOP helper ops). At BS=1, all 3 nodes sit below `20.41`; at BS $\ge 4$, weight reuse scales head AI up to ~50 FLOP/Byte, shifting LSTM cleanly into `MAXN_SUPER`.
- **MobileNet-v3:** Depthwise separable convolutions perform per-channel filtering with minimal weight reuse, locking AI below `5.5` across all batch sizes.

*(Visualizations: [`all_models_layer_ai.png`](./figures/all_models_layer_ai.png), [`layer_distribution_bs1.png`](./figures/layer_distribution_bs1.png), [`compute_ratio_vs_ai.png`](./figures/compute_ratio_vs_ai.png), [`roofline_all_models_bs1.png`](./figures/roofline_all_models_bs1.png), [`ai_scaling_batch.png`](./figures/ai_scaling_batch.png))*

---

## 4. Physical Hardware Experimental Setup

- **Automated Sweep (`jetson/sweep_all.py`):** Iterated over `MAXN_SUPER -> 25W -> 15W` across 5 models and batch sizes `{1, 4, 8, 16}` (`60` physical runs, ~45 min).
- **Clocks & Execution:** `sudo nvpmodel -m <id>` followed by `sudo jetson_clocks` to eliminate governor clock drift; ran on `onnxruntime-gpu` (`TensorrtExecutionProvider` / `CUDAExecutionProvider`).
- **Telemetry:** Each config executed **10 warmups + 50 timed inferences** (`time.perf_counter`) recording mean latency, std dev, throughput, and direct **INA3221** sysfs `VDD_IN` power ($E_{\text{mJ}} = P_{\text{avg}} \times T_{\text{avg}}$).
- **Non-Circular Ground Truth (`Opt_SLA15`):** True hardware optimum = lowest physical energy (mJ) mode whose mean latency penalty vs. `MAXN_SUPER` does not exceed `+15.0%`.

---

## 5. Physical Validation Results & Key Discoveries

### 5.1 Complete Benchmark Comparison (`benchmark_vs_actual_hardware.csv`)

| Model | BS | AI | CR% | Predicted | Opt (<=15% SLA) | 15W mJ | 25W mJ | MAXN mJ | 15W Lat Delta | 15W E-Save | 25W E-Save | Verdict |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **MobileNet-v3** | 1 | 4.36 | 37.5% | `15W` | `MAXN_SUPER` | 98.0 | 114.4 | 104.0 | +26.6% | +5.8% | -9.9% | `ENERGY_TRADEOFF` |
| **MobileNet-v3** | 4 | 5.18 | 56.6% | `15W` | `MAXN_SUPER` | 349.2 | 417.5 | 390.0 | +24.5% | +10.4% | -7.1% | `ENERGY_TRADEOFF` |
| **MobileNet-v3** | 8 | 5.35 | 56.6% | `15W` | `MAXN_SUPER` | 667.0 | 809.8 | 768.8 | +21.1% | +13.2% | -5.3% | `ENERGY_TRADEOFF` |
| **MobileNet-v3** | 16 | 5.44 | 56.6% | `15W` | `MAXN_SUPER` | 1233.9 | 1475.5 | 1410.5 | +21.8% | +12.5% | -4.6% | `ENERGY_TRADEOFF` |
| **LSTM** | 1 | 7.24 | 0.0% | `15W` | `MAXN_SUPER` | 53.4 | 66.8 | 61.8 | +28.9% | +13.7% | -8.0% | `ENERGY_TRADEOFF` |
| **LSTM** | 4 | 19.09 | 87.8% | `MAXN_SUPER` | `MAXN_SUPER` | 192.1 | 231.6 | 225.1 | +17.0% | +14.7% | -2.9% | `SLA_PRESERVED` |
| **LSTM** | 8 | 31.65 | 93.5% | `MAXN_SUPER` | `MAXN_SUPER` | 356.4 | 427.0 | 408.3 | +20.5% | +12.7% | -4.6% | `SLA_PRESERVED` |
| **LSTM** | 16 | 49.63 | 96.6% | `MAXN_SUPER` | `MAXN_SUPER` | 699.9 | 859.5 | 811.3 | +19.4% | +13.7% | -5.9% | `SLA_PRESERVED` |
| **YOLOv8n** | 1 | 16.69 | 93.1% | `MAXN_SUPER` | `MAXN_SUPER` | 976.2 | 1204.1 | 1125.9 | +20.3% | +13.3% | -6.9% | `SLA_PRESERVED` |
| **YOLOv8n** | 4 | 16.99 | 93.1% | `MAXN_SUPER` | `MAXN_SUPER` | 1010.5 | 1221.7 | 1100.9 | +27.9% | +8.2% | -11.0% | `SLA_PRESERVED` |
| **YOLOv8n** | 8 | 17.04 | 93.1% | `MAXN_SUPER` | `MAXN_SUPER` | 1009.0 | 1199.3 | 1107.8 | +26.4% | +8.9% | -8.3% | `SLA_PRESERVED` |
| **YOLOv8n** | 16 | 17.07 | 93.1% | `MAXN_SUPER` | `15W` | 973.6 | 1228.3 | 1313.2 | **+0.8%** | **+25.9%** | **+6.5%** | `SLA_PRESERVED` |
| **BERT-Base** | 1 | 25.00 | 99.7% | `MAXN_SUPER` | `MAXN_SUPER` | 1778.9 | 2211.0 | 2094.2 | +16.9% | +15.1% | -5.6% | `SLA_PRESERVED` |
| **BERT-Base** | 4 | 35.32 | 99.7% | `MAXN_SUPER` | `MAXN_SUPER` | 6948.7 | 8471.4 | 7821.5 | +23.0% | +11.2% | -8.3% | `SLA_PRESERVED` |
| **BERT-Base** | 8 | 37.92 | 99.7% | `MAXN_SUPER` | `MAXN_SUPER` | 13955.0 | 17476.7 | 16011.0 | +21.0% | +12.8% | -9.2% | `SLA_PRESERVED` |
| **BERT-Base** | 16 | 39.38 | 99.7% | `MAXN_SUPER` | `MAXN_SUPER` | 28081.0 | 35928.8 | 33653.7 | +17.0% | +16.6% | -6.8% | `SLA_PRESERVED` |
| **ResNet-50** | 1 | 24.37 | 99.4% | `MAXN_SUPER` | `MAXN_SUPER` | 815.4 | 992.4 | 962.4 | +17.3% | +15.3% | -3.1% | `SLA_PRESERVED` |
| **ResNet-50** | 4 | 31.54 | 99.4% | `MAXN_SUPER` | `MAXN_SUPER` | 3457.5 | 4296.1 | 3996.1 | +19.5% | +13.5% | -7.5% | `SLA_PRESERVED` |
| **ResNet-50** | 8 | 33.17 | 99.4% | `MAXN_SUPER` | `MAXN_SUPER` | 6832.8 | 8576.6 | 8023.4 | +17.6% | +14.8% | -6.9% | `SLA_PRESERVED` |
| **ResNet-50** | 16 | 34.05 | 99.4% | `MAXN_SUPER` | `MAXN_SUPER` | 11624.9 | 14431.9 | 13357.3 | +20.8% | +13.0% | -8.0% | `SLA_PRESERVED` |

### 5.2 Key Architectural Discoveries
1. **25W Pareto Inefficiency:** Across `19/20` workloads, 25W draws **2.9% to 11.0% MORE energy** than `MAXN_SUPER` while adding **17.4% to 29.1% latency** (GPU throttled `1020 -> 918 MHz`, EMC left at `3199 MHz`). Only YOLOv8n BS=16 shows a net energy drop (`+6.5%`), yet 15W yields `+25.9%` energy savings at `+0.8%` latency. Thus DVFS on Orin Nano is **bimodal (`MAXN_SUPER` vs. `15W`)**.
2. **Coupled Memory Bus Scaling:** Dropping to 15W shrinks EMC clock (`3199 -> 2133 MHz`, `-33.3%` bandwidth), causing memory-bound models (`MobileNet-v3`, `LSTM BS=1`) to suffer `+21.1%` to `+28.9%` latency penalties even while saving `+5.8%` to `+13.7%` energy.
3. **Standard Deviation & Execution Stability:** Compute-bound BERT-Base (BS=4/8 rel. std = 2.6%–3.4%) and ResNet-50 (BS=16 rel. std = 3.0%) stay deterministic due to cache reuse; MobileNet shows higher relative variance (10–17%) from DRAM contention, and YOLOv8n BS=16 shows memory saturation tail variance (26.2% rel. std).

---

## 6. Limitations, Future Work & Conclusion

- **Future Improvements:** (1) Incorporate an explicit coupled-clock bandwidth penalty factor $k_{\text{bw}}$ into the analytical Roofline transition check; (2) Run NVIDIA Nsight Compute (`ncu`) kernel-level profiling for empirical GFLOP/s & GB/s ceilings; (3) Formulate direct GFLOP/Joule Energy Roofline thresholds.
- **Conclusion:** Pagoda provides a reliable `< 2 second` pre-deployment filter that detects compute-bound workloads with **100% precision (15/15 SLA preserved, preventing up to 461.2 ms delay)** and achieves **70.0% strict SLA agreement (`14/20`)** on Jetson Orin Nano, while revealing hardware clock-coupling limits that distinguish budget edge accelerators from high-end AGX systems.

---

## 7. References
1. Prashanthi S. K. et al., *"Pagoda: An Energy and Time Roofline Study for DNN Workloads on Edge Accelerators,"* IISc Bangalore, `arXiv:2509.20189` (`https://github.com/dream-lab/pagoda`).
2. NVIDIA Corporation, *"Jetson Orin Nano Developer Kit Technical Specifications & nvpmodel Guide."*
3. Williams, Waterman, Patterson, *"Roofline: An Insightful Visual Performance Model for Multicore Architectures,"* CACM, 2009.
4. NVIDIA Corporation, *"NVIDIA Nsight Compute Profiling Guide"* & TI *"INA3221 Power Monitor Telemetry Documentation."*
