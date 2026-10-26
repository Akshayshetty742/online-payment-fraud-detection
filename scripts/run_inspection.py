"""
Runner script to inspect the downloaded dataset and generate inspection summaries.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.inspect_data import inspect_dataset

def main():
    raw_data_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    reports_dir = os.path.join(PROJECT_ROOT, "reports")
    
    # Locate CSV file in raw data dir
    csv_files = [f for f in os.listdir(raw_data_dir) if f.endswith(".csv")]
    if not csv_files:
        print(f"No CSV file found in {raw_data_dir}. Please run download script first.", file=sys.stderr)
        sys.exit(1)
        
    dataset_file = os.path.join(raw_data_dir, csv_files[0])
    md_output = os.path.join(reports_dir, "dataset_inspection_summary.md")
    json_output = os.path.join(reports_dir, "dataset_inspection_summary.json")
    
    print(f"Starting inspection for: {dataset_file}")
    summary = inspect_dataset(dataset_file, output_md_path=md_output, output_json_path=json_output)
    
    print("\n" + "="*60)
    print("DATASET INSPECTION COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Filename:       {summary['filename']}")
    print(f"Dimensions:     {summary['dimensions']['rows']:,} rows × {summary['dimensions']['columns']} columns")
    print(f"Duplicate rows: {summary['duplicate_rows']['count']:,} ({summary['duplicate_rows']['percentage']}%)")
    print(f"Primary Target: {summary['target_column']}")
    
    for t_col, t_data in summary['target_analysis'].items():
        print(f"\nDistribution for '{t_col}':")
        for val, stats in t_data['distribution'].items():
            print(f"  Class {val}: {stats['count']:,} ({stats['percentage']}%)")
        if t_data['imbalance_ratio']:
            print(f"  Imbalance Ratio: {t_data['imbalance_ratio']}")
            
    print("="*60)
    print(f"Reports saved to:\n  - {md_output}\n  - {json_output}")

if __name__ == "__main__":
    main()
