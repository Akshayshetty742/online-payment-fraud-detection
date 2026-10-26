"""
Audit Validation Checks Script.
Executes non-destructive validation checks for the official fastai audit:
1. Verify vocab mapping and positive-class probability extraction.
2. Verify row index determinism and exact sample matching.
3. Verify schema compatibility between backup files and fastai official results.
4. Train fastai with seed 42 to get validation set probabilities, select thresholds on validation set only,
   and report generalization on test set while preserving threshold=0.50 baseline.
"""

import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.features.build_features import preprocess_data, prepare_train_test_splits
from src.models.fastai_model import train_fastai_tabular
from src.models.evaluate import evaluate_predictions

def main():
    print("="*70)
    print("AUDIT CHECK 1: EXACT ROW INDICES & SPLIT REPRODUCIBILITY")
    print("="*70)
    
    data_path = os.path.join(PROJECT_ROOT, "data", "raw", "new_data.csv")
    df_raw = pd.read_csv(data_path)
    
    # Track raw indices
    df_raw_indexed = df_raw.copy()
    df_raw_indexed['raw_row_idx'] = np.arange(len(df_raw))
    
    # Preprocessing pipeline
    sample_size = min(len(df_raw), 150000)
    df_processed = preprocess_data(
        df_raw_indexed,
        sample_size=sample_size,
        filter_relevant_types_only=True,
        random_state=42
    )
    
    raw_indices_in_processed = df_processed['raw_row_idx'].values
    print(f"Total sampled rows: {len(raw_indices_in_processed)}")
    print(f"First 10 raw row indices in processed sample: {raw_indices_in_processed[:10].tolist()}")
    
    # Hash of raw indices
    raw_indices_hash = hashlib.sha256(raw_indices_in_processed.tobytes()).hexdigest()
    print(f"Raw indices in processed sample SHA-256: {raw_indices_hash}")
    
    # Split
    # Drop raw_row_idx before splitting to mirror production
    raw_idx_series = df_processed['raw_row_idx']
    df_feat_only = df_processed.drop(columns=['raw_row_idx'])
    
    splits = prepare_train_test_splits(
        df_feat_only,
        target_col='isFraud',
        test_size=0.20,
        val_size=0.10,
        scale_features=True,
        random_state=42
    )
    
    X_train, y_train = splits['X_train'], splits['y_train']
    X_val, y_val = splits['X_val'], splits['y_val']
    X_test, y_test = splits['X_test'], splits['y_test']
    
    train_bytes = X_train.values.tobytes()
    val_bytes = X_val.values.tobytes()
    test_bytes = X_test.values.tobytes()
    
    print(f"Train set: {X_train.shape}, fraud: {sum(y_train)}, SHA-256: {hashlib.sha256(train_bytes).hexdigest()[:16]}")
    print(f"Val set:   {X_val.shape}, fraud: {sum(y_val)}, SHA-256: {hashlib.sha256(val_bytes).hexdigest()[:16]}")
    print(f"Test set:  {X_test.shape}, fraud: {sum(y_test)}, SHA-256: {hashlib.sha256(test_bytes).hexdigest()[:16]}")
    
    print("\n" + "="*70)
    print("AUDIT CHECK 2: SCHEMA & BACKUP VERIFICATION")
    print("="*70)
    
    backup_json_path = os.path.join(PROJECT_ROOT, "reports", "backup_reimplementation_model_comparison_results.json")
    fastai_json_path = os.path.join(PROJECT_ROOT, "reports", "fastai_official_benchmark_results.json")
    
    with open(backup_json_path, 'r') as f:
        backup_data = json.load(f)
    with open(fastai_json_path, 'r') as f:
        fastai_data = json.load(f)
        
    print(f"Backup models count: {len(backup_data)}")
    backup_keys = list(backup_data[0].keys())
    fastai_keys = list(fastai_data.keys())
    print(f"Backup keys ({len(backup_keys)}): {backup_keys}")
    print(f"Fastai keys ({len(fastai_keys)}): {fastai_keys}")
    key_match = (backup_keys == fastai_keys)
    print(f"Schema matches exactly: {key_match}")
    
    # Check types
    type_matches = True
    for k in backup_keys:
        bt = type(backup_data[0][k])
        ft = type(fastai_data[k])
        if bt != ft:
            print(f"Type mismatch for {k}: backup is {bt}, fastai is {ft}")
            type_matches = False
    print(f"All field data types match: {type_matches}")

    print("\n" + "="*70)
    print("AUDIT CHECK 3: FASTAI TRAINING, VOCAB VERIFICATION & THRESHOLD AUDIT")
    print("="*70)
    
    # Train fastai model
    res_fastai = train_fastai_tabular(
        X_train, y_train,
        X_val, y_val,
        X_test,
        epochs=10,
        batch_size=512,
        lr=1e-2,
        random_state=42
    )
    
    learn = res_fastai['model']
    vocab = list(learn.dls.vocab)
    print(f"\nfastai dls.vocab: {vocab}")
    print(f"vocab[0] -> {vocab[0]} (Type: {type(vocab[0])})")
    print(f"vocab[1] -> {vocab[1]} (Type: {type(vocab[1])})")
    assert vocab == ['0', '1'], f"Unexpected vocab: {vocab}"
    print("CONFIRMED: preds[:, 1] is mathematically and programmatically class '1' (isFraud=1).")
    
    # Get validation probabilities
    val_dl = learn.dls.test_dl(X_val)
    val_preds, _ = learn.get_preds(dl=val_dl)
    val_probs = val_preds[:, 1].numpy().astype(np.float32)
    
    test_probs = res_fastai['y_prob']
    
    print("\n" + "-"*50)
    print("THRESHOLD OPTIMIZATION (VALIDATION SET ONLY)")
    print("-" * 50)
    
    thresholds = np.linspace(0.01, 0.99, 99)
    best_f1_val = -1.0
    best_thresh_f1 = 0.50
    best_recall_thresh = None
    
    for t in thresholds:
        val_eval = evaluate_predictions(y_val, val_probs, "fastai_val", threshold=float(t))
        if val_eval['F1_Score'] > best_f1_val:
            best_f1_val = val_eval['F1_Score']
            best_thresh_f1 = float(t)
        if val_eval['Recall'] >= 0.90 and best_recall_thresh is None:
            best_recall_thresh = float(t)
            
    print(f"Validation F1-Optimal Threshold: {best_thresh_f1:.2f} (Val F1: {best_f1_val:.4f})")
    print(f"Validation High-Recall (>=90%) Threshold: {best_recall_thresh}")
    
    # Evaluate on Validation set with: 0.50, best_thresh_f1, best_recall_thresh
    val_050 = evaluate_predictions(y_val, val_probs, "fastai_val_0.50", threshold=0.50)
    val_opt_f1 = evaluate_predictions(y_val, val_probs, f"fastai_val_opt_f1_{best_thresh_f1:.2f}", threshold=best_thresh_f1)
    
    print(f"\nValidation Set Results (tau=0.50):")
    print(f"  Precision: {val_050['Precision']}, Recall: {val_050['Recall']}, F1: {val_050['F1_Score']}, TP: {val_050['TP']}, FP: {val_050['FP']}")
    print(f"Validation Set Results (tau={best_thresh_f1:.2f}):")
    print(f"  Precision: {val_opt_f1['Precision']}, Recall: {val_opt_f1['Recall']}, F1: {val_opt_f1['F1_Score']}, TP: {val_opt_f1['TP']}, FP: {val_opt_f1['FP']}")
    
    if best_recall_thresh is not None:
        val_rec90 = evaluate_predictions(y_val, val_probs, f"fastai_val_rec90_{best_recall_thresh:.2f}", threshold=best_recall_thresh)
        print(f"Validation Set Results (tau={best_recall_thresh:.2f}):")
        print(f"  Precision: {val_rec90['Precision']}, Recall: {val_rec90['Recall']}, F1: {val_rec90['F1_Score']}, TP: {val_rec90['TP']}, FP: {val_rec90['FP']}")
    
    print("\n" + "-"*50)
    print("GENERALIZATION ON UNSEEN TEST SET (NO TEST TUNING)")
    print("-" * 50)
    
    test_050 = evaluate_predictions(y_test, test_probs, "fastai_test_0.50", threshold=0.50)
    test_opt_f1 = evaluate_predictions(y_test, test_probs, f"fastai_test_opt_f1_{best_thresh_f1:.2f}", threshold=best_thresh_f1)
    
    print(f"Test Set Baseline (tau=0.50 - Mandated Standard):")
    print(f"  RMSE: {test_050['RMSE']}, R2: {test_050['R2']}, MAE: {test_050['MAE']}")
    print(f"  Precision: {test_050['Precision']}, Recall: {test_050['Recall']}, F1: {test_050['F1_Score']}")
    print(f"  TP: {test_050['TP']}, FP: {test_050['FP']}, TN: {test_050['TN']}, FN: {test_050['FN']}")
    print(f"  ROC_AUC: {test_050['ROC_AUC']}, PR_AUC: {test_050['PR_AUC']}")
    
    print(f"\nTest Set with Validation-Selected F1 Threshold (tau={best_thresh_f1:.2f}):")
    print(f"  Precision: {test_opt_f1['Precision']}, Recall: {test_opt_f1['Recall']}, F1: {test_opt_f1['F1_Score']}")
    print(f"  TP: {test_opt_f1['TP']}, FP: {test_opt_f1['FP']}, TN: {test_opt_f1['TN']}, FN: {test_opt_f1['FN']}")
    
    if best_recall_thresh is not None:
        test_rec90 = evaluate_predictions(y_test, test_probs, f"fastai_test_rec90_{best_recall_thresh:.2f}", threshold=best_recall_thresh)
        print(f"\nTest Set with Validation-Selected High Recall Threshold (tau={best_recall_thresh:.2f}):")
        print(f"  Precision: {test_rec90['Precision']}, Recall: {test_rec90['Recall']}, F1: {test_rec90['F1_Score']}")
        print(f"  TP: {test_rec90['TP']}, FP: {test_rec90['FP']}, TN: {test_rec90['TN']}, FN: {test_rec90['FN']}")

    # Save validation threshold audit dictionary for report generation
    audit_data = {
        "validation_optimal_f1_threshold": best_thresh_f1,
        "validation_high_recall_threshold": best_recall_thresh,
        "val_results_050": val_050,
        "val_results_opt_f1": val_opt_f1,
        "test_results_050": test_050,
        "test_results_opt_f1": test_opt_f1,
        "test_results_rec90": test_rec90 if best_recall_thresh is not None else None
    }
    
    out_audit_json = os.path.join(PROJECT_ROOT, "reports", "fastai_val_threshold_audit_data.json")
    with open(out_audit_json, "w") as f:
        json.dump(audit_data, f, indent=2)
    print(f"\nValidation threshold audit data saved to: {out_audit_json}")

if __name__ == "__main__":
    main()
