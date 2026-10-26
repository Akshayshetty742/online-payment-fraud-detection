"""
Dataset Inspection Module.
Performs thorough exploratory inspection of the dataset including dimensions,
column names, data types, missing values, duplicates, and target distribution.
Generates both markdown and JSON inspection reports.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

def inspect_dataset(file_path: str, output_md_path: str = None, output_json_path: str = None) -> dict:
    """
    Inspects a dataset CSV file and generates summary statistics.
    
    Args:
        file_path: Path to the CSV dataset file.
        output_md_path: Path to save the markdown report (optional).
        output_json_path: Path to save the structured JSON report (optional).
        
    Returns:
        dict: Summary inspection findings.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")
        
    file_stat = os.stat(file_path)
    file_size_bytes = file_stat.st_size
    file_size_mb = file_size_bytes / (1024 * 1024)
    filename = os.path.basename(file_path)
    
    print(f"Reading dataset: {file_path} ({file_size_mb:.2f} MB)...")
    df = pd.read_csv(file_path)
    
    n_rows, n_cols = df.shape
    columns = list(df.columns)
    
    # Missing values
    missing_counts = df.isnull().sum()
    missing_pcts = (missing_counts / n_rows) * 100
    
    # Data types and unique counts
    col_info = []
    for col in columns:
        dtype = str(df[col].dtype)
        n_missing = int(missing_counts[col])
        pct_missing = float(missing_pcts[col])
        n_unique = int(df[col].nunique())
        # First 3 non-null sample values
        samples = df[col].dropna().head(3).tolist()
        sample_str = ", ".join([str(s) for s in samples])
        if len(sample_str) > 40:
            sample_str = sample_str[:37] + "..."
            
        col_info.append({
            "column": col,
            "dtype": dtype,
            "non_null_count": n_rows - n_missing,
            "missing_count": n_missing,
            "missing_pct": round(pct_missing, 4),
            "unique_count": n_unique,
            "sample_values": sample_str
        })
        
    # Duplicate rows check
    print("Checking for duplicate rows...")
    # For large datasets, checking duplicates across all columns can be done directly
    n_duplicates = int(df.duplicated().sum())
    duplicate_pct = (n_duplicates / n_rows) * 100 if n_rows > 0 else 0.0
    
    # Target column identification
    potential_targets = [c for c in columns if c.lower() in [
        'isfraud', 'is_fraud', 'fraud', 'class', 'target', 'label', 'isflaggedfraud', 'is_flagged_fraud'
    ]]
    
    target_info = {}
    primary_target = None
    for pt in potential_targets:
        counts = df[pt].value_counts().to_dict()
        total_valid = df[pt].count()
        dist = {str(k): {"count": int(v), "percentage": round((v / total_valid) * 100, 4)} for k, v in counts.items()}
        target_info[pt] = {
            "unique_values": list(counts.keys()),
            "distribution": dist,
            "imbalance_ratio": None
        }
        if len(counts) == 2:
            vals = list(counts.values())
            imbalance = round(max(vals) / min(vals), 2) if min(vals) > 0 else "inf"
            target_info[pt]["imbalance_ratio"] = f"{imbalance}:1"
            if primary_target is None and 'flagged' not in pt.lower():
                primary_target = pt
                
    if primary_target is None and potential_targets:
        primary_target = potential_targets[0]

    # Numerical feature summary
    numeric_df = df.select_dtypes(include=[np.number])
    numeric_summary = {}
    if not numeric_df.empty:
        desc = numeric_df.describe().to_dict()
        for col, stats in desc.items():
            numeric_summary[col] = {
                "mean": round(float(stats.get("mean", 0)), 4),
                "std": round(float(stats.get("std", 0)), 4),
                "min": round(float(stats.get("min", 0)), 4),
                "25%": round(float(stats.get("25%", 0)), 4),
                "50% (median)": round(float(stats.get("50%", 0)), 4),
                "75%": round(float(stats.get("75%", 0)), 4),
                "max": round(float(stats.get("max", 0)), 4),
            }
            
    summary_data = {
        "inspection_timestamp": datetime.now().isoformat(),
        "filename": filename,
        "file_path": os.path.abspath(file_path),
        "file_size_bytes": file_size_bytes,
        "file_size_mb": round(file_size_mb, 2),
        "dimensions": {
            "rows": n_rows,
            "columns": n_cols
        },
        "duplicate_rows": {
            "count": n_duplicates,
            "percentage": round(duplicate_pct, 4)
        },
        "target_column": primary_target,
        "target_analysis": target_info,
        "columns_info": col_info,
        "numerical_features_summary": numeric_summary
    }
    
    # Save structured JSON
    if output_json_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
        print(f"JSON inspection summary saved to: {output_json_path}")
        
    # Generate Markdown Summary
    md_content = generate_markdown_report(summary_data)
    if output_md_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_md_path)), exist_ok=True)
        with open(output_md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"Markdown inspection report saved to: {output_md_path}")
        
    return summary_data

def generate_markdown_report(data: dict) -> str:
    """Generates a clean GitHub-flavored markdown report from inspection data."""
    dims = data["dimensions"]
    dups = data["duplicate_rows"]
    
    md = []
    md.append(f"# Dataset Inspection Summary: `{data['filename']}`")
    md.append("")
    md.append(f"- **Inspection Date**: {data['inspection_timestamp']}")
    md.append(f"- **File Location**: `{data['file_path']}`")
    md.append(f"- **File Size**: {data['file_size_mb']} MB ({data['file_size_bytes']:,} bytes)")
    md.append(f"- **Dimensions**: **{dims['rows']:,} rows** × **{dims['columns']} columns**")
    md.append(f"- **Duplicate Rows**: {dups['count']:,} ({dups['percentage']}%)")
    md.append(f"- **Identified Primary Target**: `{data['target_column']}`")
    md.append("")
    
    # Notice / Clarification Flag
    md.append("## ⚠️ Crucial Assignment Notice: Problem Formulation Discrepancy")
    md.append("> **Note for Professor / Viva Clarification**:")
    md.append("> The dataset target column (`isFraud`) is binary (0 = legitimate transaction, 1 = fraudulent transaction), making payment fraud detection intrinsically a **binary classification problem**.")
    md.append("> However, the professor's assignment guidelines mention regression metrics (**RMSE**, **R²**, and **MAE**).")
    md.append("> - In this project, we explicitly record and compute the professor's specified regression metrics (interpreting predictions continuously / as probabilities) while also providing appropriate classification metrics (Precision, Recall, F1, ROC-AUC, PR-AUC).")
    md.append("> - This discrepancy has been explicitly flagged for clarification and viva presentation.")
    md.append("")
    
    # Target Class Distribution
    md.append("## Target Column Distribution")
    md.append("")
    for target_col, info in data["target_analysis"].items():
        md.append(f"### Target Column: `{target_col}`")
        if info["imbalance_ratio"]:
            md.append(f"- **Class Imbalance Ratio**: `{info['imbalance_ratio']}`")
        md.append("")
        md.append("| Class / Value | Count | Percentage |")
        md.append("| :--- | :--- | :--- |")
        for cls_val, stats in info["distribution"].items():
            md.append(f"| `{cls_val}` | {stats['count']:,} | {stats['percentage']}% |")
        md.append("")
        
    # Column details table
    md.append("## Column Schema, Data Types & Missing Values")
    md.append("")
    md.append("| Column Name | Data Type | Non-Null Count | Missing Count | Missing % | Unique Count | Sample Values |")
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for col in data["columns_info"]:
        md.append(f"| `{col['column']}` | `{col['dtype']}` | {col['non_null_count']:,} | {col['missing_count']} | {col['missing_pct']}% | {col['unique_count']:,} | `{col['sample_values']}` |")
    md.append("")
    
    # Numerical features summary table
    if data["numerical_features_summary"]:
        md.append("## Numerical Features Statistical Summary")
        md.append("")
        md.append("| Column | Mean | Std Dev | Min | 25% | Median (50%) | 75% | Max |")
        md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for col, s in data["numerical_features_summary"].items():
            md.append(f"| `{col}` | {s['mean']:,} | {s['std']:,} | {s['min']:,} | {s['25%']:,} | {s['50% (median)']:,} | {s['75%']:,} | {s['max']:,} |")
        md.append("")
        
    return "\n".join(md)
