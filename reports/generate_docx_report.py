"""
Pagoda: Comprehensive BTP-2 Final Project Report Generator
---------------------------------------------------------
Generates the authoritative Final Project Report from verified CSVs and figures:
  - results/multi_batch_consolidated.csv
  - results/benchmark_vs_actual_hardware.csv
  - results/jetson_hardware_results.csv
  - reports/figures/*.png & results/fig_pareto_inefficiency.png

Outputs:
  - reports/BTP_Pagoda_Final_Report.docx
  - reports/BTP_Pagoda_Research_Report.pdf (via Microsoft Word / docx2pdf)
"""

import os
import csv
import tempfile
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(ROOT_DIR, 'reports')
FIG_DIR = os.path.join(REPORTS_DIR, 'figures')
RES_DIR = os.path.join(ROOT_DIR, 'results')

NAVY = RGBColor(16, 44, 87)
SLATE_BLUE = RGBColor(36, 82, 138)
DARK_TEXT = RGBColor(34, 34, 34)
MUTED_GRAY = RGBColor(95, 95, 95)


def set_cell_background(cell, hex_color):
    shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    cell._tc.get_or_add_tcPr().append(shading)


def set_cell_margins(cell, top=80, bottom=80, left=100, right=100):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)


def set_table_borders(table, color="D0D7DE"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'  <w:top w:val="single" w:sz="6" w:space="0" w:color="{color}"/>'
        f'  <w:bottom w:val="single" w:sz="8" w:space="0" w:color="102C57"/>'
        f'  <w:insideH w:val="single" w:sz="4" w:space="0" w:color="{color}"/>'
        f'  <w:left w:val="none"/>'
        f'  <w:right w:val="none"/>'
        f'  <w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)


def prevent_row_split(row):
    trPr = row._tr.get_or_add_trPr()
    trPr.append(OxmlElement('w:cantSplit'))


def repeat_header_row(row):
    trPr = row._tr.get_or_add_trPr()
    trPr.append(OxmlElement('w:tblHeader'))


def add_page_number_footer(section):
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run_left = p.add_run("BTP-2 Final Report  |  IIT Palakkad  |  Page ")
    run_left.font.size = Pt(9)
    run_left.font.color.rgb = MUTED_GRAY

    fldSimple = OxmlElement('w:fldSimple')
    fldSimple.set(qn('w:instr'), 'PAGE')
    p._p.append(fldSimple)


def add_header_banner(section):
    header = section.header
    p = header.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = p.add_run("Pagoda: Energy and Time Roofline Study on NVIDIA Jetson Orin Nano (8GB)")
    run.font.size = Pt(8.5)
    run.font.italic = True
    run.font.color.rgb = MUTED_GRAY


def generate_clean_pipeline_diagram():
    """Create a crisp, unclipped pipeline diagram specifically for the report."""
    out_path = os.path.join(FIG_DIR, 'model_pipeline_clean.png')
    os.makedirs(FIG_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11.5, 3.4))
    ax.set_xlim(-1.1, 10.7)
    ax.set_ylim(-0.25, 3.15)
    ax.axis('off')

    boxes = [
        (0.0, 1.55, '1. ONNX\nGraph Input', '5 Models (.onnx)\nBS in {1, 4, 8, 16}', '#2B6CB0'),
        (2.4, 1.55, '2. Per-Layer\nProfiling', 'Compute FLOPs &\nDRAM Bytes (float32)', '#2F855A'),
        (4.8, 1.55, '3. Roofline\nClassification', 'AI = FLOPs / Bytes\nLayer AI vs Ridge', '#6B46C1'),
        (7.2, 1.55, '4. DVFS Policy\nSelection', 'CR >= 70% or AI >= 18.36\n-> MAXN_SUPER else 15W', '#C05621'),
        (9.6, 1.55, '5. Hardware\nValidation', 'Jetson Orin Nano + INA3221\nNon-Circular 15% SLA Check', '#C53030'),
    ]

    for x, y, title, sub, col in boxes:
        rect = mpatches.FancyBboxPatch(
            (x - 0.92, y - 0.62), 1.84, 1.24,
            boxstyle="round,pad=0.08",
            facecolor=col, alpha=0.12,
            edgecolor=col, linewidth=1.8
        )
        ax.add_patch(rect)
        ax.text(x, y, title, ha='center', va='center', fontsize=10, fontweight='bold', color='#1A202C')
        ax.text(x, 0.35, sub, ha='center', va='center', fontsize=8.2, color='#4A5568', style='italic')

    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + 1.00
        x2 = boxes[i + 1][0] - 1.00
        ax.annotate('', xy=(x2, 1.55), xytext=(x1, 1.55),
                    arrowprops=dict(arrowstyle='->,head_width=0.25,head_length=0.25', color='#2D3748', lw=1.8))

    ax.text(4.8, 2.75, 'Pagoda Analytical Modeling & Hardware Validation Pipeline',
            ha='center', va='center', fontsize=13, fontweight='bold', color='#102C57')

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out_path


def add_heading_styled(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for r in h.runs:
        r.font.name = 'Calibri'
        r.font.bold = True
        r.font.color.rgb = NAVY if level == 1 else SLATE_BLUE
        r.font.size = Pt(14 if level == 1 else 11.5)
    h.paragraph_format.space_before = Pt(12 if level == 1 else 8)
    h.paragraph_format.space_after = Pt(4)
    h.paragraph_format.keep_with_next = True
    return h


def add_body(doc, text, bold_prefix=None, space_after=6):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15
    if bold_prefix:
        r0 = p.add_run(bold_prefix)
        r0.bold = True
        r0.font.name = 'Calibri'
        r0.font.size = Pt(10.5)
        r0.font.color.rgb = DARK_TEXT
    r1 = p.add_run(text)
    r1.font.name = 'Calibri'
    r1.font.size = Pt(10.5)
    r1.font.color.rgb = DARK_TEXT
    return p


def add_bullet(doc, bold_prefix, text):
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.12
    if bold_prefix:
        r0 = p.add_run(bold_prefix)
        r0.bold = True
        r0.font.name = 'Calibri'
        r0.font.size = Pt(10.2)
        r0.font.color.rgb = DARK_TEXT
    r1 = p.add_run(text)
    r1.font.name = 'Calibri'
    r1.font.size = Pt(10.2)
    r1.font.color.rgb = DARK_TEXT
    return p


def add_equation_box(doc, lines):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.rows[0].cells[0]
    set_cell_background(cell, "F4F7FB")
    set_cell_margins(cell, top=90, bottom=90, left=160, right=160)
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'  <w:left w:val="single" w:sz="18" w:space="0" w:color="102C57"/>'
        f'  <w:top w:val="none"/><w:right w:val="none"/><w:bottom w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.2
    for idx, line in enumerate(lines):
        if idx > 0:
            p.add_run('\n')
        r = p.add_run(line)
        r.font.name = 'Consolas'
        r.font.size = Pt(9.5)
        r.font.color.rgb = NAVY
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(3)


def resolve_fig(primary_path, fallback_name=None):
    if os.path.exists(primary_path):
        return primary_path
    if fallback_name:
        fb = os.path.join(RES_DIR, fallback_name)
        if os.path.exists(fb):
            return fb
    return primary_path


def add_figure(doc, path, caption_text, width_in=6.2, fallback_name=None):
    resolved = resolve_fig(path, fallback_name)
    if not os.path.exists(resolved):
        print(f"WARNING: Missing figure {resolved}")
        return
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(6)
    p_img.paragraph_format.space_after = Pt(3)
    p_img.paragraph_format.keep_with_next = True
    run = p_img.add_run()
    run.add_picture(resolved, width=Inches(width_in))

    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_after = Pt(8)
    rc = p_cap.add_run(caption_text)
    rc.font.name = 'Calibri'
    rc.font.size = Pt(9.2)
    rc.font.italic = True
    rc.font.color.rgb = MUTED_GRAY


def build_styled_table(doc, headers, rows_data, col_widths=None, font_size=8.5, center_cols=None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table)
    center_cols = center_cols or set()

    hdr_row = table.rows[0]
    repeat_header_row(hdr_row)
    prevent_row_split(hdr_row)
    for i, htext in enumerate(headers):
        cell = hdr_row.cells[i]
        set_cell_background(cell, "102C57")
        set_cell_margins(cell, top=75, bottom=75, left=70, right=70)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        if col_widths and i < len(col_widths):
            cell.width = Inches(col_widths[i])
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if i in center_cols else WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(htext)
        r.font.name = 'Calibri'
        r.font.size = Pt(font_size)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)

    for r_idx, row_vals in enumerate(rows_data):
        row = table.add_row()
        prevent_row_split(row)
        bg = "F8FAFC" if r_idx % 2 == 1 else "FFFFFF"
        for i, val in enumerate(row_vals):
            cell = row.cells[i]
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=55, bottom=55, left=70, right=70)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            if col_widths and i < len(col_widths):
                cell.width = Inches(col_widths[i])
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if i in center_cols else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(str(val))
            r.font.name = 'Calibri'
            r.font.size = Pt(font_size)
            r.font.color.rgb = DARK_TEXT
            if i == 0:
                r.font.bold = True

    sp = doc.add_paragraph()
    sp.paragraph_format.space_after = Pt(4)
    return table


def load_csv(path):
    with open(path, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def main():
    pipeline_img = generate_clean_pipeline_diagram()

    mb_rows = load_csv(os.path.join(RES_DIR, 'multi_batch_consolidated.csv'))
    bm_rows = load_csv(os.path.join(RES_DIR, 'benchmark_vs_actual_hardware.csv'))
    hw_rows = load_csv(os.path.join(RES_DIR, 'jetson_hardware_results.csv'))

    # Index raw hardware rows by (model_name, batch_size, mode)
    model_map = {
        'mobnet.onnx': 'MobileNet-v3',
        'lstm.onnx': 'LSTM',
        'yolov8n.onnx': 'YOLOv8n',
        'bert-base.onnx': 'BERT-Base',
        'resnet50.onnx': 'ResNet-50',
    }
    hw_idx = {}
    for r in hw_rows:
        m = model_map.get(r['model'], r['model'])
        bs = int(r['batch_size'])
        mode = r['power_mode']
        hw_idx[(m, bs, mode)] = {
            'lat': float(r['avg_latency_ms']),
            'std': float(r['std_latency_ms']),
            'pwr': float(r['avg_power_w']),
            'eng': float(r['energy_mj']),
            'thr': float(r['throughput_sps']),
        }

    doc = docx.Document()
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(10.5)

    for s in doc.sections:
        s.top_margin = Inches(0.85)
        s.bottom_margin = Inches(0.85)
        s.left_margin = Inches(0.80)
        s.right_margin = Inches(0.80)
        s.different_first_page_header_footer = True
        add_header_banner(s)
        add_page_number_footer(s)

    # =========================================================================
    # TITLE BLOCK (Clean First Page Header)
    # =========================================================================
    p_inst = doc.add_paragraph()
    p_inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_inst.paragraph_format.space_before = Pt(4)
    p_inst.paragraph_format.space_after = Pt(2)
    r_inst = p_inst.add_run("INDIAN INSTITUTE OF TECHNOLOGY PALAKKAD\n")
    r_inst.font.size = Pt(13)
    r_inst.font.bold = True
    r_inst.font.color.rgb = SLATE_BLUE
    r_dept = p_inst.add_run("Department of Data Science & Engineering  |  B.Tech Project (BTP-2) Final Report")
    r_dept.font.size = Pt(10.5)
    r_dept.font.color.rgb = MUTED_GRAY

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(10)
    p_title.paragraph_format.space_after = Pt(6)
    rt = p_title.add_run("Pagoda: Energy and Time Roofline Study for DNN Workloads\non NVIDIA Jetson Edge Accelerators")
    rt.font.size = Pt(19)
    rt.font.bold = True
    rt.font.color.rgb = NAVY

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(12)
    rs = p_sub.add_run("Offline ONNX Analytical Profiling and Physical Hardware Verification on Jetson Orin Nano Developer Kit (8GB)")
    rs.font.size = Pt(11)
    rs.font.italic = True
    rs.font.color.rgb = SLATE_BLUE

    build_styled_table(
        doc,
        ["Candidate", "Roll Number", "Faculty Mentor", "Target Hardware", "Evaluation Date"],
        [["Jillapelli Sai Varun", "142301040", "Prof. Abhinandan S P", "NVIDIA Jetson Orin Nano (8GB)", "November 2026"]],
        col_widths=[1.5, 1.0, 1.5, 1.8, 1.1],
        font_size=9.0,
        center_cols={1, 4},
    )

    # =========================================================================
    # ABSTRACT / EXECUTIVE SUMMARY
    # =========================================================================
    add_heading_styled(doc, "Executive Summary", level=1)
    add_body(
        doc,
        "Deploying deep neural networks (DNNs) on battery- and thermally-constrained edge accelerators such as the "
        "NVIDIA Jetson Orin Nano Developer Kit (8GB) requires balancing real-time inference latency against strict energy budgets. "
        "Testing every candidate model, batch size, and Dynamic Voltage and Frequency Scaling (DVFS) mode directly on hardware "
        "is slow (exceeding 45 minutes per 60-configuration sweep) and requires root physical board access. "
        "Adapting the Pagoda framework (originally developed at IISc Bangalore for the 60W Jetson AGX Orin, arXiv:2509.20189), "
        "this B.Tech project implements and physically verifies an analytical Roofline power-mode selector that inspects ONNX "
        "computational graphs offline on a host laptop in under 2 seconds."
    )
    add_body(
        doc,
        "Across five diverse workloads (MobileNet-v3, LSTM, YOLOv8n, BERT-Base, and ResNet-50) swept over batch sizes {1, 4, 8, 16} "
        "and all three hardware power profiles (15W, 25W, MAXN_SUPER), physical measurements captured with onboard Texas Instruments "
        "INA3221 current/voltage sensors reveal four critical findings: "
        "(1) The analytical model achieves 100% compute-bound detection (15/15 configurations correctly preserved at MAXN_SUPER with 0% delay), "
        "preventing up to 461.2 ms of latency degradation on BERT-Base BS=16 relative to 15W (and 717.6 ms relative to 25W); "
        "(2) Under non-circular physical ground truth with a strict 15% latency SLA, the model achieves a 70.0% (14/20) optimal-mode match rate; "
        "(3) The intermediate 25W mode is Pareto-inefficient across 19/20 workloads, consuming 2.9% to 11.0% MORE energy than MAXN_SUPER while "
        "increasing latency by 17.4% to 29.1%—rendering Orin Nano DVFS effectively bimodal (MAXN_SUPER vs. 15W); and "
        "(4) Unlike Jetson AGX Orin, 15W mode on Orin Nano couples the LPDDR5 memory controller clock with the GPU clock (3199 MHz -> 2133 MHz, "
        "-33.3% bandwidth), causing memory-bound workloads to violate the 15% latency SLA (+21.1% to +28.9% delay) despite achieving +5.8% to +13.7% energy savings."
    )

    # =========================================================================
    # 1. INTRODUCTION & MOTIVATION
    # =========================================================================
    add_heading_styled(doc, "1. Introduction & Project Contributions", level=1)
    add_body(
        doc,
        "Edge AI platforms in drones, mobile robots, and smart cameras operate within tight thermal and battery envelopes. "
        "While NVIDIA Jetson devices expose multiple nvpmodel power profiles, choosing the right mode is non-trivial. "
        "Compute-bound DNNs (dense matrix multiplications in Transformers and standard ResNet convolutions) saturate GPU streaming multiprocessors "
        "and exhibit classic 'race-to-sleep' behavior: throttling clocks increases execution time more than power drops, wasting total energy. "
        "Conversely, memory-bound networks (depthwise separable convolutions in MobileNet-v3, small-batch recurrent cells in LSTM) stall on DRAM transfers; "
        "under classical Roofline theory, downclocking GPU cores should reduce dynamic power with minimal slowdown."
    )
    add_body(
        doc,
        "The foundational Pagoda research paper evaluated this principle on a high-end 60W Jetson AGX Orin featuring independent GPU and memory clock domains. "
        "A core open question addressed in this project is whether those conclusions hold on a cost- and power-constrained edge SoC—the NVIDIA Jetson Orin Nano Developer Kit (8GB)."
    )
    add_body(doc, "This project delivers four concrete engineering and research contributions:", space_after=3)
    add_bullet(doc, "Offline ONNX Analytical Profiler & Estimator Fixes: ",
               "Audited and corrected upstream Pagoda estimators (removing a hardcoded sequence_length=8 bug in src/analytical_model/estimates.py and fixing integer batch parsing in model.py) "
               "and built layer_analysis.py and power_mode_selector.py to compute per-layer FLOPs, DRAM bytes, Arithmetic Intensity (AI), and Compute Ratio in <2 seconds.")
    add_bullet(doc, "Adaptation to Orin Nano Super Hardware Envelope: ",
               "Recalculated Ampere SM 8.7 Roofline parameters in hardware_config.py from live sysfs clock readings across 15W, 25W, and MAXN_SUPER modes, "
               "and refined selection to a clean 2-regime policy (MAXN_SUPER vs. 15W) after proving 25W Pareto inefficiency.")
    add_bullet(doc, "Automated 60-Configuration Hardware Testbench: ",
               "Built jetson/sweep_all.py, run_inference.py, get_hardware_specs.py, and verify_predictions.py using nvpmodel, jetson_clocks, "
               "ONNX Runtime GPU (TensorRT/CUDA), and automated TI INA3221 sysfs power telemetry across 60 configs (50 timed runs after 10 warmups each).")
    add_bullet(doc, "Non-Circular Hardware Discovery: ",
               "Exposed why classical independent-scaling Roofline assumptions fail on coupled-clock edge SoCs (-33.3% EMC bandwidth in 15W mode) "
               "and documented the bimodal DVFS landscape and single empirical win-win operating point (YOLOv8n BS=16).")

    # =========================================================================
    # 2. HARDWARE PLATFORM & ROOFLINE THEORETICAL FRAMEWORK
    # =========================================================================
    add_heading_styled(doc, "2. Hardware Platform & Roofline Formulation", level=1)
    add_heading_styled(doc, "2.1 NVIDIA Jetson Orin Nano Developer Kit (8GB) Specifications", level=2)
    add_body(
        doc,
        "Verified directly on hardware via jetson/get_hardware_specs.py (querying nvpmodel -q, devfreq GPU max_freq, and EMC clock rates), "
        "the Jetson Orin Nano integrates an NVIDIA Ampere GPU (8 Streaming Multiprocessors, 128 CUDA cores/SM = 1024 CUDA cores, 2 FLOPs per FMA cycle) "
        "with a 128-bit (16-byte) Double Data Rate (DDR) LPDDR5 unified memory subsystem:"
    )

    add_equation_box(doc, [
        "Peak FP32 Compute (GFLOP/s) = 1024 CUDA cores x 2 FLOPs/cycle x GPU_Clock_GHz",
        "Peak Memory Bandwidth (GB/s) = 16 Bytes x EMC_Clock_GHz x 2 (DDR)",
        "Roofline Ridge Point (I*)    = Peak Compute (GFLOP/s) / Peak Bandwidth (GB/s)  [FLOP/Byte]"
    ])

    build_styled_table(
        doc,
        ["nvpmodel ID", "Power Mode", "GPU Clock", "Memory (EMC)", "Peak Compute", "Peak Bandwidth", "Ridge Point (I*)", "Empirical Role"],
        [
            ["Mode 0", "15W", "612 MHz", "2133 MHz", "1253.38 GFLOP/s", "68.26 GB/s", "18.36 FLOP/Byte", "Low-Power / Coupled BW (-33.3%)"],
            ["Mode 1", "25W", "918 MHz", "3199 MHz", "1880.06 GFLOP/s", "102.37 GB/s", "18.37 FLOP/Byte", "Pareto-Inefficient (Never Selected)"],
            ["Mode 2", "MAXN_SUPER", "1020 MHz", "3199 MHz", "2088.96 GFLOP/s", "102.37 GB/s", "20.41 FLOP/Byte", "Full Performance / SLA Baseline"],
        ],
        col_widths=[0.75, 0.90, 0.72, 0.85, 1.05, 0.95, 0.90, 1.78],
        font_size=8.2,
        center_cols={0, 2, 3, 6},
    )

    add_body(
        doc,
        "Notice a crucial hardware property visible in the table above: 15W and 25W exhibit nearly identical theoretical Ridge Points "
        "(18.36 vs. 18.37 FLOP/Byte) because transitioning from 25W/MAXN to 15W scales both GPU clock (-33.3% vs 25W, -40.0% vs MAXN) "
        "and EMC memory clock (-33.3%, 3199 -> 2133 MHz) simultaneously."
    )

    add_heading_styled(doc, "2.2 Time and Energy Roofline Formulation", level=2)
    add_body(
        doc,
        "For any ONNX operator node l (Conv, Gemm, MatMul, LSTM, BatchNormalization, Pool, Activation, Elementwise) assuming 4-byte float32 tensors, "
        "the analytical estimator extracts layer computation FLOPs_l and DRAM memory traffic Bytes_l = Weight_Bytes + Input_Bytes + Output_Bytes. "
        "For example, 2D convolution evaluates FLOPs = 2 * Batch * Cout * Hout * Wout * (Cin/groups * Kh * Kw) + Output_Elements, while MatMul/Gemm "
        "scales MACs linearly with batch size while parameter reads remain amortized across the batch."
    )
    add_equation_box(doc, [
        "Arithmetic Intensity (AI)     = Total FLOPs / Total DRAM Bytes   [FLOP/Byte]",
        "Time Roofline Latency (T)    ~ max( FLOPs / Peak_Compute , DRAM_Bytes / Peak_Bandwidth )",
        "Energy Roofline Energy (E)   = P_total * T = (P_dynamic(f, V) + P_static) * T",
        "Compute Ratio (CR)           = Sum(FLOPs of layers with AI >= Ridge) / Total_FLOPs"
    ])
    add_body(
        doc,
        "In Time Roofline, compute-bound layers (AI >= Ridge) lie on the flat compute ceiling where lower GPU clocks directly stretch T; "
        "combined with leakage/base platform power P_static, longer execution wastes energy (the race-to-sleep effect). Memory-bound layers "
        "(AI < Ridge) lie on the bandwidth-limited slope; if bandwidth remained constant, GPU downclocking would save P_dynamic without hurting T."
    )

    # =========================================================================
    # 3. ANALYTICAL PIPELINE & WORKLOAD ANALYSIS
    # =========================================================================
    add_heading_styled(doc, "3. Analytical Pipeline & Workload Characterization", level=1)
    add_figure(doc, pipeline_img,
               "Figure 1: Pagoda 5-stage offline ONNX analytical profiling and hardware validation pipeline.",
               width_in=6.6)

    add_heading_styled(doc, "3.1 Two-Regime Power Mode Decision Policy", level=2)
    add_body(
        doc,
        "While an initial three-tier heuristic (15W / 25W / MAXN_SUPER) was explored during early prototyping, physical measurements "
        "confirmed that 25W is strictly dominated by MAXN_SUPER in both latency and energy. Accordingly, hardware_config.py implements "
        "a streamlined two-regime selector operating on per-layer Compute Ratio (CR) and aggregate Arithmetic Intensity (AI):"
    )
    add_equation_box(doc, [
        "if (Compute_Ratio >= 0.70) or (Overall_AI >= 18.36 FLOP/Byte):  select MAXN_SUPER",
        "else:                                                          select 15W"
    ])
    add_body(
        doc,
        "Using a 70% FLOP-weighted Compute Ratio criterion rather than a naive 50% layer-count threshold prevents misclassifying networks "
        "that contain hundreds of lightweight elementwise layers (e.g., LayerNorm, Add, ReLU) whose runtime is nonetheless dominated by a small set of heavy tensor cores/MAC ops."
    )

    add_heading_styled(doc, "3.2 Evaluated DNN Portfolio & Per-Layer Insights", level=2)
    add_body(
        doc,
        "Five representative architectures spanning vision, object detection, sequential RNNs, and Transformers were exported to ONNX (opset 14) "
        "and profiled across batch sizes 1, 4, 8, and 16:"
    )

    # Build Portfolio Summary at BS=1 and BS=16 from multi_batch_consolidated.csv
    mb_map = {(r['model'], int(r['batch_size'])): r for r in mb_rows}
    portfolio_rows = []
    meta_specs = [
        ("MobileNet-v3", "Edge Vision (CNN)", "(3, 224, 224)", "Depthwise Separable Conv"),
        ("LSTM", "Sequential / NLP (RNN)", "(32,)", "2-Layer Fused LSTM + Linear Head"),
        ("YOLOv8n", "Object Detection", "(3, 640, 640)", "Conv2d, C2f, PANet Detection Head"),
        ("BERT-Base", "Transformer NLP", "(128,)", "12x Multi-Head Self-Attention + FFN"),
        ("ResNet-50", "Deep Vision (CNN)", "(3, 224, 224)", "Standard 3x3 & 1x1 Residual Conv"),
    ]
    for mname, dom, shape_s, ops in meta_specs:
        r1 = mb_map[(mname, 1)]
        r16 = mb_map[(mname, 16)]
        portfolio_rows.append([
            mname, dom, shape_s, ops,
            f"{r1['num_compute_bound']} / {r1['num_memory_bound']}",
            f"{float(r1['arithmetic_intensity']):.2f} -> {float(r16['arithmetic_intensity']):.2f}",
            f"{float(r1['compute_ratio_pct']):.1f}% -> {float(r16['compute_ratio_pct']):.1f}%",
        ])

    build_styled_table(
        doc,
        ["Model", "Domain", "Input Shape", "Primary Operators", "Layers (CB/MB @ BS=1)", "AI (BS=1 -> 16)", "CR% (BS=1 -> 16)"],
        portfolio_rows,
        col_widths=[0.95, 1.05, 0.85, 1.35, 0.85, 0.95, 0.90],
        font_size=8.0,
        center_cols={2, 4, 5, 6},
    )

    add_figure(doc, os.path.join(FIG_DIR, 'all_models_layer_ai.png'),
               "Figure 2: Per-layer Arithmetic Intensity scatter for all 5 models at BS=1 relative to the MAXN_SUPER Ridge Point (20.41 FLOP/Byte).",
               width_in=6.5)

    add_body(doc, "Key structural observations from Figure 2 and Figure 3:")
    add_bullet(doc, "Layer Count vs. FLOP Dominance: ",
               "BERT-Base contains 285 memory-bound layers (LayerNorm, Add, Reshape, Softmax) vs. only 60 compute-bound MatMul/Gemm layers, yet those 60 layers "
               "account for 99.7% of all floating-point operations. Similarly, ResNet-50's 52 convolution layers contribute 99.4% of total FLOPs. "
               "YOLOv8n exhibits a hybrid structure: overall AI sits slightly below the ridge (16.69-17.07 FLOP/Byte), but 52 dense convolutions contribute 93.1% of FLOPs.")
    add_bullet(doc, "Why LSTM Displays Only 3 Points at BS=1: ",
               "ONNX exports our 2-layer stacked LSTM across 32 timesteps as two fused recurrent LSTM operators followed by a Linear MatMul projection head "
               "(while 4 zero-FLOP helper nodes such as Transpose/Squeeze are filtered from active compute plots). At BS=1, all three layers are memory-bound "
               "(AI = 3.86, 3.86, 14.21; overall AI = 7.24, CR = 0%). As batch size grows to 4, 8, and 16, weight reuse in the final MatMul head lifts layer AI to 49.6 FLOP/Byte, shifting LSTM to Compute-Bound (CR = 87.8% -> 96.6%).")
    add_bullet(doc, "Why MobileNet-v3 Remains Permanently Memory-Bound: ",
               "Depthwise separable convolutions apply an independent spatial kernel per input channel with virtually zero inter-channel weight reuse. "
               "Consequently, overall AI increases only slightly from 4.36 (BS=1) to 5.44 (BS=16), well below both 18.36 and 20.41 FLOP/Byte.")

    add_figure(doc, os.path.join(FIG_DIR, 'layer_distribution_bs1.png'),
               "Figure 3: Active compute-bound vs. memory-bound layer counts at Batch Size 1 (Ridge = 20.41 FLOP/Byte).",
               width_in=5.8)

    add_figure(doc, os.path.join(FIG_DIR, 'compute_ratio_vs_ai.png'),
               "Figure 4: Two-regime DVFS decision map (Compute Ratio vs. Arithmetic Intensity) across all batch sizes.",
               width_in=5.8, fallback_name='fig_compute_ratio.png')

    add_figure(doc, os.path.join(FIG_DIR, 'roofline_all_models_bs1.png'),
               "Figure 5: Analytical Roofline ceilings comparing MAXN_SUPER (2089 GFLOP/s, 102.4 GB/s) and 15W (1253 GFLOP/s, 68.3 GB/s) at BS=1.",
               width_in=5.8, fallback_name='fig_roofline_bs1.png')

    add_figure(doc, os.path.join(FIG_DIR, 'ai_scaling_batch.png'),
               "Figure 6: Arithmetic Intensity scaling across batch sizes 1 to 16, illustrating LSTM's crossover and MobileNet-v3's flat scaling.",
               width_in=5.6, fallback_name='fig_ai_vs_batch.png')

    # =========================================================================
    # 4. EXPERIMENTAL METHODOLOGY
    # =========================================================================
    add_heading_styled(doc, "4. Physical Hardware Experimental Methodology", level=1)
    add_body(
        doc,
        "To eliminate circular validation, analytical recommendations generated entirely offline were benchmarked against independent physical measurements "
        "on the NVIDIA Jetson Orin Nano Developer Kit (8GB) via jetson/sweep_all.py (~45 minutes full runtime):"
    )
    add_bullet(doc, "Configuration Space (60 Runs): ",
               "5 models (mobnet.onnx, lstm.onnx, yolov8n.onnx, bert-base.onnx, resnet50.onnx) x 4 batch sizes (1, 4, 8, 16) x 3 modes (MAXN_SUPER, 25W, 15W).")
    add_bullet(doc, "Deterministic Clock Locking: ",
               "Each power mode was applied via sudo nvpmodel -m <id>, settled for 3 seconds, locked with sudo jetson_clocks to suppress dynamic governor hunting, and verified with nvpmodel -q.")
    add_bullet(doc, "Execution & Telemetry: ",
               "Models ran via onnxruntime-gpu (TensorrtExecutionProvider with CUDAExecutionProvider fallback). Each configuration performed 10 warmup iterations followed by "
               "50 timed inferences (time.perf_counter) to record mean latency (ms) and standard deviation (ms). Total module power (VDD_IN) was polled dynamically from the onboard "
               "TI INA3221 I2C sensor via Linux sysfs (voltage_mv * current_ma / 1e6) before and after each inference, yielding Energy (mJ) = P_avg (W) x T_avg (ms).")
    add_bullet(doc, "Non-Circular Ground Truth Definition (15% SLA): ",
               "Following the Pagoda methodology, for each workload the true optimal hardware mode (Opt_SLA15) is the mode minimizing physical Energy (mJ) among all modes whose mean "
               "latency remains within +15.0% of MAXN_SUPER baseline latency.")

    # =========================================================================
    # 5. EMPIRICAL HARDWARE RESULTS & DISCOVERIES
    # =========================================================================
    add_heading_styled(doc, "5. Empirical Hardware Results & Architectural Discoveries", level=1)

    add_heading_styled(doc, "5.1 Latency Stability, Instantaneous Power, and Race-to-Sleep Energy", level=2)
    add_figure(doc, os.path.join(FIG_DIR, 'latency_across_modes.png'),
               "Figure 7: Measured hardware latency (ms) across 15W, 25W, and MAXN_SUPER with +/-1 standard deviation error bars over 50 timed inferences (BS=1 and BS=16).",
               width_in=6.5, fallback_name='fig_latency_stddev.png')

    add_body(
        doc,
        "Standard deviation error bars in Figure 7 reveal architectural predictability: compute-bound networks (BERT-Base BS=4/8 rel. std = 2.6%-3.4%; ResNet-50 BS=16 rel. std = 3.0%) "
        "exhibit tight relative variance because dense GEMM/Conv tiles remain resident in on-chip caches. Memory-bound MobileNet-v3 shows higher relative spread "
        "(10.8% at BS=16, up to 17.1% at BS=4) due to DRAM bus arbitration sensitivity, while YOLOv8n BS=16 shows tail jitter (26.2% rel. std) as 16 high-resolution 640x640 feature maps saturate memory pressure. "
        "Across all 20 workloads, MAXN_SUPER is strictly the fastest state; throttling BERT-Base BS=16 down to 15W degrades latency by +461.2 ms (2719.85 ms -> 3181.07 ms), and 25W degrades it by +717.6 ms (3437.41 ms)."
    )

    add_figure(doc, os.path.join(FIG_DIR, 'energy_across_modes.png'),
               "Figure 8: Measured energy consumption per inference (mJ) at BS=1 and BS=16 across all 3 power modes.",
               width_in=6.5, fallback_name='fig_energy_per_model.png')

    add_figure(doc, os.path.join(FIG_DIR, 'power_draw_bs1.png'),
               "Figure 9: Instantaneous VDD_IN board power draw (Watts) measured via onboard TI INA3221 sensors at BS=1.",
               width_in=5.5, fallback_name='fig_power_per_model.png')

    add_heading_styled(doc, "5.2 Complete 20-Workload Prediction vs. Physical Hardware Ground Truth", level=2)

    val_table_rows = []
    for r in bm_rows:
        m = r['model']
        bs = int(r['batch_size'])
        lat_mx = hw_idx[(m, bs, 'MAXN_SUPER')]['lat']
        lat_15 = hw_idx[(m, bs, '15W')]['lat']
        lat_25 = hw_idx[(m, bs, '25W')]['lat']
        e_mx = float(r['energy_maxn_mj'])
        e_15 = float(r['energy_15w_mj'])
        e_25 = float(r['energy_25w_mj'])

        d15 = (lat_15 - lat_mx) / lat_mx * 100.0
        es15 = (e_mx - e_15) / e_mx * 100.0
        es25 = (e_mx - e_25) / e_mx * 100.0
        match_flag = "YES" if r['predicted_mode'] == r['optimal_hw_sla15'] else "NO"

        val_table_rows.append([
            m, str(bs), f"{float(r['arithmetic_intensity']):.2f}", f"{float(r['compute_ratio_pct']):.1f}%",
            r['predicted_mode'], r['optimal_hw_sla15'], match_flag,
            f"{e_mx:.1f} / {e_25:.1f} / {e_15:.1f}",
            f"+{d15:.1f}%", f"+{es15:.1f}%", f"{es25:+.1f}%",
            r['physical_verdict']
        ])

    build_styled_table(
        doc,
        ["Model", "BS", "AI", "CR%", "Pred", "Opt(SLA)", "Match", "E (MAXN/25W/15W mJ)", "15W Lat+", "15W E-Sav", "25W E-Sav", "Verdict"],
        val_table_rows,
        col_widths=[0.78, 0.25, 0.38, 0.42, 0.68, 0.68, 0.38, 1.15, 0.50, 0.52, 0.52, 0.84],
        font_size=7.0,
        center_cols={1, 2, 3, 6, 8, 9, 10},
    )

    add_heading_styled(doc, "5.3 Discovery 1: Pareto Inefficiency of 25W Mode (Bimodal DVFS)", level=2)
    add_body(
        doc,
        "Across 19 out of 20 workloads, Jetson Orin Nano's intermediate 25W mode consumes +2.9% to +11.0% MORE total energy than uncapped MAXN_SUPER, "
        "while simultaneously slowing inference down by +17.4% to +29.1%. Because 25W lowers the GPU ceiling (1020 -> 918 MHz) while leaving the memory bus "
        "and board voltage rails elevated (3199 MHz EMC), power drops only ~14-15% while execution time increases 18-29%, causing E = P x T to increase. "
        "The sole positive exception is YOLOv8n BS=16 (+6.5% energy saved at +6.8% latency delta in 25W), because concurrent execution of 16 640x640 images saturates "
        "internal buffers so latency degrades mildly relative to the 12.4% power drop; even there, 15W is strictly superior (+25.9% energy saved, +0.8% latency). "
        "Consequently, Orin Nano DVFS is effectively bimodal: deploy MAXN_SUPER for SLA/throughput or 15W for low-power operation, and avoid 25W."
    )

    add_figure(doc, os.path.join(RES_DIR, 'fig_pareto_inefficiency.png'),
               "Figure 10: Pareto inefficiency analysis of 25W vs. 15W energy savings (%) relative to MAXN_SUPER across all 20 workloads.",
               width_in=6.5)

    add_heading_styled(doc, "5.4 Discovery 2: Coupled Memory Bus Scaling (-33.3% Bandwidth at 15W)", level=2)
    add_body(
        doc,
        "In the original IISc Pagoda study on Jetson AGX Orin, lowering GPU frequency for memory-bound models preserved high memory bandwidth, achieving energy savings "
        "with minimal slowdown ('Win-Win Zone'). On the Jetson Orin Nano, switching to 15W mode forcibly drops EMC frequency from 3199 MHz to 2133 MHz—cutting peak memory bandwidth "
        "from 102.37 GB/s down to 68.26 GB/s (-33.3%). This violates the classical independent-scaling Roofline assumption: dropping to 15W depresses the memory slope as well as the compute ceiling. "
        "As a result, MobileNet-v3 (AI = 4.36-5.44) and LSTM BS=1 (AI = 7.24) did achieve real energy savings in 15W (+5.8% to +13.7%), proving the analytical classifier identified their low AI accurately; "
        "however, the 33% bandwidth cut inflicted +21.1% to +28.9% latency penalties, pushing them beyond the strict 15% SLA threshold."
    )

    add_figure(doc, os.path.join(FIG_DIR, 'energy_vs_latency_scatter.png'),
               "Figure 11: Energy savings vs. latency penalty at 15W relative to MAXN_SUPER. Only YOLOv8n BS=16 (top-left) lands inside the <=15% SLA Win-Win Zone.",
               width_in=5.6)

    add_figure(doc, os.path.join(FIG_DIR, 'latency_penalty_sla.png'),
               "Figure 12: Latency penalty relative to MAXN_SUPER across all 20 workloads against the 15% SLA boundary.",
               width_in=6.4, fallback_name='fig_latency_delta.png')

    add_heading_styled(doc, "5.5 Validation Summary & Practical Value", level=2)
    add_figure(doc, os.path.join(FIG_DIR, 'prediction_match.png'),
               "Figure 13: Analytical prediction summary: 15/20 MAXN compute-bound protections vs. 5/20 15W energy-latency tradeoffs (overall strict SLA match = 14/20 or 70.0%).",
               width_in=4.8, fallback_name='fig_prediction_accuracy.png')

    add_bullet(doc, "100% Compute-Bound Protection (15/15 Correct MAXN Policies): ",
               "All 15 compute-dominant configurations (BERT-Base BS=1..16, ResNet-50 BS=1..16, YOLOv8n BS=1..16, LSTM BS=4..16) were preserved at MAXN_SUPER with 0% latency penalty, "
               "preventing up to 461.2 ms slowdown on BERT-Base BS=16 (at 15W) and 717.6 ms (at 25W).")
    add_bullet(doc, "Strict 15% SLA Ground-Truth Match Rate (14/20 = 70.0%): ",
               "Matches Opt_SLA15 exactly on 14 of 15 MAXN predictions; the 1 conservative mismatch (YOLOv8n BS=16) avoided latency risk by choosing MAXN_SUPER where empirical 15W proved a unique 25.9% energy / +0.8% latency win-win. "
               "The remaining 5 mismatches (MobileNet-v3 BS=1..16 and LSTM BS=1) saved 5.8%-13.7% energy in 15W but violated the 15% SLA due to coupled memory scaling (+21.1% to +28.9% delay).")
    add_bullet(doc, "Practical Profiling Speedup: ",
               "Offline ONNX execution completes on a standard laptop in under 2 seconds vs. 45+ minutes of privileged physical board sweeps.")

    # =========================================================================
    # 6. LIMITATIONS & FUTURE WORK
    # =========================================================================
    add_heading_styled(doc, "6. Limitations & Future Work", level=1)
    add_bullet(doc, "Coupled-Clock Bandwidth Penalty Factor: ",
               "Extend Pagoda's analytical transition condition so that if lowering GPU clock reduces EMC bandwidth by delta_BW (e.g., 33.3% on Orin Nano), "
               "the expected latency increase T_mem(15W) / T_mem(MAXN) is checked directly against the user's SLA threshold before recommending 15W.")
    add_bullet(doc, "NVIDIA Nsight Compute (NCU) Empirical Roofline Calibration: ",
               "Profile CUDA kernels using NCU hardware performance counters to calibrate attainable empirical compute/bandwidth ceilings and achieved tensor core utilization beyond datasheet theoretical peaks.")
    add_bullet(doc, "Explicit Energy Roofline (GFLOP/Joule) & Multi-Board Portability: ",
               "Incorporate empirical INA3221 idle/dynamic power slopes into a closed-form GFLOP/Joule roofline ceiling and extend validation across Jetson Orin NX and Jetson AGX Orin.")

    # =========================================================================
    # 7. CONCLUSION
    # =========================================================================
    add_heading_styled(doc, "7. Conclusion", level=1)
    add_body(
        doc,
        "This B.Tech project verified the Pagoda Roofline methodology end-to-end on physical NVIDIA Jetson Orin Nano Developer Kit (8GB) hardware across 60 workload-mode configurations. "
        "Offline ONNX analysis reliably isolates compute-bound architectures in under 2 seconds (15/15 SLA-protected, preventing up to 461.2 ms latency degradation on BERT-Base) and identifies low-intensity memory-bound networks accurately. "
        "Crucially, physical INA3221 validation uncovered hardware constraints absent from high-end AGX studies: the intermediate 25W mode is Pareto-inefficient across 19/20 workloads, "
        "and coupled GPU/EMC clock scaling in 15W (-33.3% bandwidth) transforms memory-bound DVFS from a 'free' win-win into an explicit energy-latency tradeoff (+5.8% to +13.7% energy saved at +21.1% to +28.9% latency cost). "
        "These insights provide practical, grounded guidelines for hardware-aware DNN deployment on resource-constrained edge accelerators."
    )

    # =========================================================================
    # 8. REFERENCES
    # =========================================================================
    add_heading_styled(doc, "8. References", level=1)
    refs = [
        "[1] Prashanthi S. K., K. K. Sahoo, A. R. Saikia, P. Gupta, A. V. Joshi, P. Pansari, and Y. Simmhan, "
        "\"Pagoda: An Energy and Time Roofline Study for DNN Workloads on Edge Accelerators,\" IISc Bangalore, arXiv:2509.20189 (Code: https://github.com/dream-lab/pagoda).",
        "[2] NVIDIA Corporation, \"Jetson Orin Nano Developer Kit Technical Specifications & nvpmodel Power Management Guide,\" NVIDIA Embedded Documentation.",
        "[3] S. Williams, A. Waterman, and D. Patterson, \"Roofline: An Insightful Visual Performance Model for Multicore Architectures,\" Communications of the ACM, vol. 52, no. 4, pp. 65-76, 2009.",
        "[4] NVIDIA Corporation, \"NVIDIA Nsight Compute Profiling Guide & Roofline Analysis,\" NVIDIA Developer Documentation.",
        "[5] Texas Instruments, \"INA3221 Triple-Channel High-Side Measurement, Shunt and Bus Voltage Monitor with I2C-Compatible Interface\" (Linux hwmon sysfs telemetry)."
    ]
    for ref in refs:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.first_line_indent = Inches(-0.25)
        r = p.add_run(ref)
        r.font.name = 'Calibri'
        r.font.size = Pt(9.5)
        r.font.color.rgb = DARK_TEXT

    out_docx = os.path.join(REPORTS_DIR, 'BTP_Pagoda_Final_Report.docx')
    out_pdf = os.path.join(REPORTS_DIR, 'BTP_Pagoda_Research_Report.pdf')
    doc.save(out_docx)
    print(f"SUCCESS: Generated {out_docx}")

    try:
        from docx2pdf import convert
        convert(out_docx, out_pdf)
        print(f"SUCCESS: Converted PDF -> {out_pdf}")
    except Exception as e:
        print(f"PDF conversion warning: {e}")


if __name__ == '__main__':
    main()
