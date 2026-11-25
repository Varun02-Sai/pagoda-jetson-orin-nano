"""
Generate Roofline Data CSV (Synchronized with Canonical 3-Tier Multi-Batch Results)
----------------------------------------------------------------------------------
Maintains backwards compatibility by saving roofline metrics directly from
results/multi_batch_consolidated.csv to results/roofline_data.csv.
"""

import os
import shutil
import subprocess
import sys

def main():
    print("=" * 70)
    print("  Synchronizing Roofline Data CSV")
    print("=" * 70)

    src_csv = 'results/multi_batch_consolidated.csv'
    dst_csv = 'results/roofline_data.csv'

    if not os.path.exists(src_csv):
        print(f"Generating canonical {src_csv}...")
        subprocess.run([sys.executable, 'generate_multi_batch_analysis.py'], check=True)

    shutil.copyfile(src_csv, dst_csv)
    print(f"  [OK] Successfully synchronized {dst_csv} with canonical multi-batch analysis.")
    print("=" * 70)

if __name__ == '__main__':
    main()
