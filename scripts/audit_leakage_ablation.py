"""
Leakage & Feature Ablation Audit Script for Online Payment Fraud Detection.
Evaluates XGBoost across multiple feature sets to isolate the source of high predictive performance
and determine pre-transaction vs post-transaction hindsight leakage.
Saves results strictly to audit output files without modifying existing experiment results.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix
)

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.features.build_features import preprocess_data, prepare_train_test_splits

def evaluate_preds(y_true, y_prob, name, train_time, feature_cols):
    y_pred = (y_prob >= 0.5).astype(int)
    rmse = np.sqrt(mean_squared_error(y_true, y_prob))
    r2 = r2_score(y_true, y_prob)
    mae = mean_absolute_error(y_true, y_prob)
    
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(rec, prec)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return {
        "experiment_name": name,
        "n_features": len(feature_cols),
        "features_used": feature_cols,
        "RMSE": round(float(rmse), 5),
        "R2": round(float(r2), 5),
        "MAE": round(float(mae), 5),
        "PR_AUC": round(float(pr_auc), 5),
        "ROC_AUC": round(float(roc_auc), 5),
        "F1_Score": round(float(f1), 5),
        "Precision": round(float(precision), 5),
        "Recall": round(float(recall), 5),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "train_time_s": round(float(train_time), 2)
    }

def run_leakage_ablation():
    print("="*70)
    print("STARTING FOCUSED LEAKAGE & ABLATION AUDIT EXPERIMENT")
    print("="*70)
    
    data_path = os.path.join(PROJECT_ROOT, "data", "raw", "new_data.csv")
    print(f"Loading raw dataset from {data_path}...")
    df_raw = pd.read_csv(data_path)
    
    # Standard 150k sample matching run_experiments.py
    sample_size = min(len(df_raw), 150000)
    df_processed = preprocess_data(
        df_raw,
        sample_size=sample_size,
        filter_relevant_types_only=True,
        random_state=42
    )
    
    splits = prepare_train_test_splits(
        df_processed,
        target_col='isFraud',
        test_size=0.20,
        val_size=0.10,
        scale_features=False, # Tree models don't need scaling, keeps feature interpretability exact
        random_state=42
    )
    
    X_train_df, y_train = splits['X_train'], splits['y_train']
    X_val_df, y_val = splits['X_val'], splits['y_val']
    X_test_df, y_test = splits['X_test'], splits['y_test']
    
    all_features = list(X_train_df.columns)
    print(f"\nAll 18 Preprocessed Features:\n{all_features}")
    
    # Calculate class weight
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / max(1, pos_count)
    
    # Define ablation scenarios
    ablation_configs = [
        {
            "name": "Exp 1: Full Baseline (Current 18 features)",
            "features": all_features,
            "rationale": "Direct reproduction of baseline XGBoost model"
        },
        {
            "name": "Exp 2: Exclude isFlaggedFraud (17 features)",
            "features": [f for f in all_features if f != 'isFlaggedFraud'],
            "rationale": "Verifies whether the static business rule column drives performance"
        },
        {
            "name": "Exp 3: Exclude errorBalanceOrig (17 features)",
            "features": [f for f in all_features if f != 'errorBalanceOrig'],
            "rationale": "Tests model reliance specifically on the origin balance discrepancy"
        },
        {
            "name": "Exp 4: Exclude All Engineered Discrepancies (15 features)",
            "features": [f for f in all_features if f not in ['errorBalanceOrig', 'errorBalanceDest', 'orig_emptied']],
            "rationale": "Tests whether tree splits can reconstruct discrepancies using raw balances alone"
        },
        {
            "name": "Exp 5: Strict Pre-Transaction Features (Pre-Auth Only, 10 features)",
            "features": [
                'step', 'amount', 'oldbalanceOrg', 'oldbalanceDest',
                'hour_of_day', 'day_of_week', 'is_merchant_dest',
                'amount_to_oldbalance_ratio', 'type_CASH_OUT', 'type_TRANSFER'
            ],
            "rationale": "Simulates realistic real-time gateway scoring before transaction execution"
        },
        {
            "name": "Exp 6: Pre-Transaction + Simulated Ledger State (12 features)",
            "features": [
                'step', 'amount', 'oldbalanceOrg', 'oldbalanceDest',
                'hour_of_day', 'day_of_week', 'is_merchant_dest',
                'amount_to_oldbalance_ratio', 'type_CASH_OUT', 'type_TRANSFER',
                'orig_emptied' # Account would be emptied if amount >= oldbalanceOrg
            ],
            "rationale": "Includes hypothetical empty-account intent without post-execution discrepancies"
        }
    ]
    
    audit_results = []
    feature_importances_dict = {}
    
    for cfg in ablation_configs:
        exp_name = cfg["name"]
        feats = cfg["features"]
        print("\n" + "-"*60)
        print(f"RUNNING: {exp_name}")
        print(f"Feature count: {len(feats)}")
        print(f"Rationale: {cfg['rationale']}")
        
        X_tr = X_train_df[feats].values
        X_va = X_val_df[feats].values
        X_te = X_test_df[feats].values
        
        t0 = time.time()
        model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            tree_method='hist',
            eval_metric='logloss',
            early_stopping_rounds=25,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_tr, y_train, eval_set=[(X_va, y_val)], verbose=False)
        train_time = time.time() - t0
        
        y_prob = model.predict_proba(X_te)[:, 1]
        metrics = evaluate_preds(y_test, y_prob, exp_name, train_time, feats)
        metrics["rationale"] = cfg["rationale"]
        audit_results.append(metrics)
        
        print(f"Results -> RMSE: {metrics['RMSE']}, R2: {metrics['R2']}, PR-AUC: {metrics['PR_AUC']}, F1: {metrics['F1_Score']}, Recall: {metrics['Recall']}, Precision: {metrics['Precision']}")
        print(f"Confusion Matrix: TP={metrics['TP']}, FP={metrics['FP']}, FN={metrics['FN']}, TN={metrics['TN']}")
        
        # Save feature importances
        booster = model.get_booster()
        gain_dict = booster.get_score(importance_type='gain')
        # Map f0, f1 to actual feature names
        named_gain = {}
        for idx, f_name in enumerate(feats):
            key = f"f{idx}"
            named_gain[f_name] = round(gain_dict.get(key, 0.0), 2)
        # Sort descending
        sorted_gain = sorted(named_gain.items(), key=lambda x: x[1], reverse=True)
        feature_importances_dict[exp_name] = sorted_gain
        print(f"Top 5 Features by Gain: {sorted_gain[:5]}")
        
    # Compile results into DataFrame
    df_audit = pd.DataFrame(audit_results)
    display_cols = ['experiment_name', 'n_features', 'RMSE', 'R2', 'PR_AUC', 'F1_Score', 'Precision', 'Recall', 'FP', 'FN']
    print("\n" + "="*70)
    print("ABLATION AUDIT SUMMARY TABLE:")
    print("="*70)
    print(df_audit[display_cols].to_string(index=False))
    
    # Save outputs to reports
    reports_dir = os.path.join(PROJECT_ROOT, "reports")
    csv_path = os.path.join(reports_dir, "audit_leakage_ablation_results.csv")
    json_path = os.path.join(reports_dir, "audit_leakage_ablation_results.json")
    
    df_audit.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump({
            "results": audit_results,
            "feature_importances": feature_importances_dict
        }, f, indent=2)
        
    print(f"\nAudit results successfully saved to:\n  - {csv_path}\n  - {json_path}")
    return df_audit

if __name__ == "__main__":
    run_leakage_ablation()
