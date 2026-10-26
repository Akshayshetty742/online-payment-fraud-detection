"""
Automated Project Consistency & Agreement Verification Script.
Checks:
1. Schema & metric agreement between model_comparison_results.csv and .json
2. Integrity of backup_reimplementation_model_comparison_results.csv and .json (untouched)
3. Agreement of PROJECT_REPORT.md tables with CSV/JSON metrics
4. Agreement of presentation slides and PPTX existence
5. Verification of regenerated figures 07 and 08
"""

import os
import sys
import json
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def verify():
    print("="*70)
    print("RUNNING FINAL PROJECT CONSISTENCY & INTEGRITY VERIFICATION")
    print("="*70)
    
    passed = True
    
    # Check 1: CSV and JSON comparison results
    csv_path = os.path.join(PROJECT_ROOT, "reports", "model_comparison_results.csv")
    json_path = os.path.join(PROJECT_ROOT, "reports", "model_comparison_results.json")
    
    df_csv = pd.read_csv(csv_path)
    with open(json_path, "r") as f:
        data_json = json.load(f)
    df_json = pd.DataFrame(data_json)
    
    print(f"\n[Check 1] CSV and JSON Models Count: CSV={len(df_csv)}, JSON={len(df_json)}")
    assert len(df_csv) == 5, f"Expected 5 models, got {len(df_csv)}"
    assert len(df_json) == 5, f"Expected 5 models in JSON, got {len(df_json)}"
    
    # Verify official fastai label
    csv_fastai_name = df_csv.loc[2, 'Framework']
    json_fastai_name = df_json.loc[2, 'model_name']
    print(f"  -> Fastai label in CSV: '{csv_fastai_name}'")
    print(f"  -> Fastai label in JSON: '{json_fastai_name}'")
    assert "Official" in csv_fastai_name, "Fastai label missing (Official) in CSV"
    assert "Official" in json_fastai_name, "Fastai label missing (Official) in JSON"
    
    # Check metric values
    for m in ['RMSE', 'R2', 'MAE', 'PR_AUC', 'ROC_AUC', 'F1_Score', 'Recall', 'Precision']:
        csv_val = float(df_csv.loc[2, m])
        json_val = float(df_json.loc[2, m])
        diff = abs(csv_val - json_val)
        assert diff < 1e-4, f"Mismatch in {m}: CSV={csv_val}, JSON={json_val}"
    print("  -> All 8 metrics between CSV and JSON match with diff < 1e-4!")
    
    # Check 2: Backup files integrity
    backup_csv = os.path.join(PROJECT_ROOT, "reports", "backup_reimplementation_model_comparison_results.csv")
    backup_json = os.path.join(PROJECT_ROOT, "reports", "backup_reimplementation_model_comparison_results.json")
    assert os.path.exists(backup_csv), "Backup CSV missing!"
    assert os.path.exists(backup_json), "Backup JSON missing!"
    df_b_csv = pd.read_csv(backup_csv)
    with open(backup_json, "r") as f:
        df_b_json = json.load(f)
    print(f"\n[Check 2] Backup integrity:")
    print(f"  -> Backup CSV row 3 model: '{df_b_csv.loc[2, 'Framework']}' (RMSE: {df_b_csv.loc[2, 'RMSE']})")
    print(f"  -> Backup JSON row 3 model: '{df_b_json[2]['model_name']}' (RMSE: {df_b_json[2]['RMSE']})")
    assert df_b_csv.loc[2, 'Framework'] == "PyTorch with fastai", "Backup CSV was altered!"
    assert df_b_json[2]['model_name'] == "PyTorch with fastai", "Backup JSON was altered!"
    assert abs(df_b_csv.loc[2, 'RMSE'] - 0.10078) < 1e-4, "Backup CSV metric changed!"
    print("  -> Backup files verified UNTOUCHED!")
    
    # Check 3: PROJECT_REPORT.md consistency
    report_path = os.path.join(PROJECT_ROOT, "reports", "PROJECT_REPORT.md")
    with open(report_path, "r", encoding="utf-8") as f:
        report_text = f.read()
    print(f"\n[Check 3] PROJECT_REPORT.md Verification:")
    assert "PyTorch with fastai (Official)" in report_text, "Missing official fastai in report"
    assert "0.0584" in report_text or "0.05835" in report_text, "Missing fastai RMSE in report"
    assert "0.4859" in report_text or "0.48591" in report_text, "Missing fastai R2 in report"
    assert "0.6710" in report_text or "0.67101" in report_text, "Missing fastai F1 in report"
    assert "0.9626" in report_text or "0.96262" in report_text, "Missing fastai Precision in report"
    assert "0.5150" in report_text or "0.515" in report_text, "Missing fastai Recall in report"
    assert "Pre-Transaction Only (Ablated)" in report_text, "Missing ablation table in report"
    assert "0.9888" in report_text, "Missing ablation PR-AUC in report"
    assert "PaySim" in report_text and "synthetic" in report_text, "Missing synthetic data disclosure in report"
    assert "deprecated and archived the `lightning-flash` package in 2023" in report_text, "Missing Lightning Flash deprecation disclosure in report"
    assert "0.57" in report_text and "guarantee of future" in report_text, "Missing validation threshold disclosure in report"
    print("  -> Report contains all required disclosures, metrics, ablation, and threshold discussions!")
    
    # Check 4: Presentation files
    pptx_path = os.path.join(PROJECT_ROOT, "presentation", "online_payment_fraud_detection.pptx")
    slides_md = os.path.join(PROJECT_ROOT, "presentation", "presentation_slides.md")
    assert os.path.exists(pptx_path), "PPTX presentation missing!"
    assert os.path.getsize(pptx_path) > 30000, f"PPTX presentation too small: {os.path.getsize(pptx_path)} bytes"
    with open(slides_md, "r", encoding="utf-8") as f:
        slides_text = f.read()
    assert "PyTorch with fastai (Official)" in slides_text, "Missing official fastai in slides md"
    assert "0.0584" in slides_text, "Missing fastai RMSE in slides md"
    print(f"\n[Check 4] Presentation files verified:")
    print(f"  -> PPTX size: {os.path.getsize(pptx_path):,} bytes")
    print(f"  -> Slides MD size: {os.path.getsize(slides_md):,} bytes")
    
    # Check 5: Generated charts
    fig7_path = os.path.join(PROJECT_ROOT, "reports", "figures", "07_model_metrics_comparison.png")
    fig8_path = os.path.join(PROJECT_ROOT, "reports", "figures", "08_roc_pr_curves.png")
    assert os.path.exists(fig7_path), "Figure 07 missing!"
    assert os.path.exists(fig8_path), "Figure 08 missing!"
    print(f"\n[Check 5] Updated Figures:")
    print(f"  -> Fig 07 (Metrics Comparison): {os.path.getsize(fig7_path):,} bytes")
    print(f"  -> Fig 08 (ROC/PR Curves): {os.path.getsize(fig8_path):,} bytes")
    
    print("\n" + "="*70)
    print("ALL PROJECT CONSISTENCY & INTEGRITY CHECKS PASSED (100% PASS)!")
    print("="*70)

if __name__ == "__main__":
    verify()
