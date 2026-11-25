"""
Hardware Configuration — Jetson Orin Nano Super (8GB)
-----------------------------------------------------
Single source of truth for all hardware specs per power mode.
Every other script imports from here. NO hardcoded values elsewhere.

These values come from running `get_hardware_specs.py` on the actual board
at each power mode (verified via nvpmodel -p --verbose and sysfs readings).
"""

# -----------------------------------------------------------------------
# GPU Architecture (constant across all modes)
# -----------------------------------------------------------------------
GPU_ARCH = "NVIDIA Ampere (SM 8.7)"
NUM_SMS = 8
CORES_PER_SM = 128
CUDA_CORES = NUM_SMS * CORES_PER_SM  # 1024
FLOPS_PER_CORE_PER_CYCLE = 2  # FMA = 2 FLOPs per cycle

# Memory bus (constant across all modes)
MEMORY_TYPE = "LPDDR5"
BUS_WIDTH_BITS = 128
BUS_WIDTH_BYTES = BUS_WIDTH_BITS // 8  # 16 bytes
DDR_FACTOR = 2  # Double Data Rate

# Mode names
MODE_15W = '15W'
MODE_25W = '25W'
MODE_MAXN = 'MAXN_SUPER'

# -----------------------------------------------------------------------
# Per-mode specifications
# -----------------------------------------------------------------------
POWER_MODES = {
    MODE_15W: {
        'nvpmodel_id': 0,
        'gpu_max_mhz': 612.0,
        'emc_mhz': 2133.0,
        'peak_compute': 1253.38,    # 1024 * 2 * 0.612 = 1253.376
        'peak_bandwidth': 68.26,    # 16 * 2.133 * 2 = 68.256
        'ridge_point': 18.36,       # 1253.38 / 68.26
        'description': 'Low Power / Energy-Saving Profile',
    },
    MODE_25W: {
        'nvpmodel_id': 1,
        'gpu_max_mhz': 918.0,
        'emc_mhz': 3199.0,
        'peak_compute': 1880.06,    # 1024 * 2 * 0.918 = 1880.064
        'peak_bandwidth': 102.37,   # 16 * 3.199 * 2 = 102.368
        'ridge_point': 18.37,       # 1880.06 / 102.37
        'description': 'Balanced Power Profile',
    },
    MODE_MAXN: {
        'nvpmodel_id': 2,
        'gpu_max_mhz': 1020.0,
        'emc_mhz': 3199.0,
        'peak_compute': 2088.96,    # 1024 * 2 * 1.020 = 2088.96
        'peak_bandwidth': 102.37,   # 16 * 3.199 * 2 = 102.368
        'ridge_point': 20.41,       # 2088.96 / 102.37
        'description': 'Uncapped Maximum Performance Profile',
    },
}

LOW_POWER_MODE = MODE_15W
MID_POWER_MODE = MODE_25W
HIGH_POWER_MODE = MODE_MAXN

# Default baseline specs used for analytical roofline (MAXN_SUPER as baseline)
DEFAULT_PEAK_COMPUTE = POWER_MODES[MODE_MAXN]['peak_compute']
DEFAULT_PEAK_BANDWIDTH = POWER_MODES[MODE_MAXN]['peak_bandwidth']
DEFAULT_RIDGE_POINT = POWER_MODES[MODE_MAXN]['ridge_point']

# Ridge points as explicit constants
RIDGE_15W = POWER_MODES[MODE_15W]['ridge_point']   # 18.36
RIDGE_25W = POWER_MODES[MODE_25W]['ridge_point']   # 18.37
RIDGE_MAXN = POWER_MODES[MODE_MAXN]['ridge_point'] # 20.41


def get_mode_specs(mode_name):
    """Get hardware specs for a given power mode name."""
    if mode_name not in POWER_MODES:
        raise ValueError(
            f"Unknown power mode '{mode_name}'. "
            f"Available modes: {list(POWER_MODES.keys())}"
        )
    return POWER_MODES[mode_name]


def get_ridge_point(mode_name):
    """Get the roofline ridge point for a given power mode."""
    return get_mode_specs(mode_name)['ridge_point']


def get_nvpmodel_id(mode_name):
    """Get the nvpmodel ID for a given power mode name."""
    return get_mode_specs(mode_name)['nvpmodel_id']


def mode_name_from_id(nvpmodel_id):
    """Look up mode name from nvpmodel ID."""
    for name, specs in POWER_MODES.items():
        if specs['nvpmodel_id'] == nvpmodel_id:
            return name
    raise ValueError(f"Unknown nvpmodel ID {nvpmodel_id}")


def select_power_mode(ai, compute_ratio=0.0, batch_size=1):
    """
    Pagoda Roofline DVFS Decision Policy (First-Principles Analytical Model)
    ------------------------------------------------------------------------
    Evaluates workload characteristics against the Jetson Orin Nano Super hardware envelope:
      - MAXN_SUPER (Mode 2): GPU=1020 MHz, EMC=3199 MHz, Ridge=20.41 FLOP/Byte (Peak Performance)
      - 25W        (Mode 1): GPU=918 MHz,  EMC=3199 MHz, Ridge=18.37 FLOP/Byte (Intermediate)
      - 15W        (Mode 0): GPU=612 MHz,  EMC=2133 MHz, Ridge=18.36 FLOP/Byte (Energy Priority)

    First-Principles Decision Logic:
      1. Compute-Bound / Dense Workloads (Compute Ratio >= 70% or AI >= 18.36 FLOP/Byte):
         - Execution-unit bound (BERT-Base, ResNet-50, YOLOv8n, LSTM BS >= 4).
         - Downclocking GPU creates immediate latency penalties (+17% to +461 ms).
         - 25W mode is Pareto-inefficient (-3% to -11% negative energy savings).
         -> MODE_MAXN (Preserves throughput SLA, 0% latency degradation)

      2. Memory-Bound Workloads (AI < 18.36 FLOP/Byte and Compute Ratio < 70%):
         - Memory-traffic dominated (MobileNet-v3, LSTM BS 1).
         - Under classical Roofline theory, GPU compute cores are underutilized waiting
           for DRAM operands, suggesting downclocking GPU frequency saves dynamic power.
         -> MODE_15W (Analytical Energy-Saving Profile)
    """
    # Compute-bound workloads: high compute ratio or arithmetic intensity near/above ridge
    if compute_ratio >= 0.70 or ai >= RIDGE_15W:
        return MODE_MAXN

    # Memory-bound workloads: low AI and low compute ratio
    return MODE_15W
