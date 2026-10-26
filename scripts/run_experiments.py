"""
Master Experimentation and Framework Benchmarking Pipeline.
Runs End-to-End:
1. Exploratory Data Analysis & Visualization Generation
2. Feature Engineering & Preprocessing
3. Stratified Train / Validation / Test Splitting
4. Training & Benchmarking all 5 Frameworks:
   - XGBoost
   - Keras
   - PyTorch with fastai
   - PyTorch with TabNet
   - PyTorch with Lightning Flash
5. Comparative Evaluation (RMSE, R2, MAE, PR-AUC, ROC-AUC, F1, Simplicity, Transparency, Efficacy)
6. Comparative Plots & Reports Output
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.build_features import preprocess_data, prepare_train_test_splits
from src.visualization.eda_plots import generate_all_eda_plots
from src.models.evaluate import evaluate_predictions, compare_frameworks_summary
from src.models.xgboost_model import train_xgboost
from src.models.keras_model import train_keras
from src.models.fastai_model import train_fastai_tabular
from src.models.tabnet_model import train_tabnet
from src.models.lightning_model import train_lightning_tabular

def plot_evaluation_charts(results: list, prob_dict: dict, y_test: np.ndarray, output_dir: str):
    """Generates comparison charts for metrics, ROC curves, and PR curves."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Metric Comparison Bar Chart
    df_res = pd.DataFrame(results)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Professor's Regression Metrics
    df_reg = df_res[['model_name', 'RMSE', 'MAE']].melt(id_vars='model_name', var_name='Metric', value_name='Score')
    sns.barplot(data=df_reg, x='model_name', y='Score', hue='Metric', ax=axes[0], palette='Blues_d')
    axes[0].set_title("Professor's Regression Metrics (Lower is Better)", fontweight='bold', pad=10)
    axes[0].set_ylabel('Score')
    axes[0].set_xlabel('')
    axes[0].tick_params(axis='x', rotation=25)
    
    # Classification Metrics (PR-AUC, ROC-AUC)
    df_auc = df_res[['model_name', 'PR_AUC', 'ROC_AUC']].melt(id_vars='model_name', var_name='Metric', value_name='Score')
    sns.barplot(data=df_auc, x='model_name', y='Score', hue='Metric', ax=axes[1], palette='Greens_d')
    axes[1].set_title('Ranking Quality: PR-AUC & ROC-AUC (Higher is Better)', fontweight='bold', pad=10)
    axes[1].set_ylabel('Score')
    axes[1].set_xlabel('')
    axes[1].tick_params(axis='x', rotation=25)
    
    # F1-Score & Recall
    df_f1 = df_res[['model_name', 'F1_Score', 'Recall']].melt(id_vars='model_name', var_name='Metric', value_name='Score')
    sns.barplot(data=df_f1, x='model_name', y='Score', hue='Metric', ax=axes[2], palette='Reds_d')
    axes[2].set_title('Detection Efficacy: F1-Score & Recall', fontweight='bold', pad=10)
    axes[2].set_ylabel('Score')
    axes[2].set_xlabel('')
    axes[2].tick_params(axis='x', rotation=25)
    
    plt.tight_layout()
    bar_path = os.path.join(output_dir, '07_model_metrics_comparison.png')
    plt.savefig(bar_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Precision-Recall & ROC Curves
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for (m_name, y_prob), col in zip(prob_dict.items(), colors):
        # ROC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax_roc.plot(fpr, tpr, label=f"{m_name}", color=col, linewidth=2)
        # PR
        pre, rec, _ = precision_recall_curve(y_test, y_prob)
        ax_pr.plot(rec, pre, label=f"{m_name}", color=col, linewidth=2)
        
    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random Chance')
    ax_roc.set_title('Receiver Operating Characteristic (ROC) Curves', fontweight='bold', pad=10)
    ax_roc.set_xlabel('False Positive Rate (1 - Specificity)')
    ax_roc.set_ylabel('True Positive Rate (Recall)')
    ax_roc.legend(loc='lower right')
    ax_roc.grid(True, linestyle='--', alpha=0.5)
    
    baseline_pr = np.sum(y_test) / len(y_test)
    ax_pr.axhline(y=baseline_pr, color='k', linestyle='--', alpha=0.6, label=f'Baseline ({baseline_pr:.4f})')
    ax_pr.set_title('Precision-Recall Curves (Essential for Class Imbalance)', fontweight='bold', pad=10)
    ax_pr.set_xlabel('Recall (Fraud Detection Rate)')
    ax_pr.set_ylabel('Precision (Positive Predictive Value)')
    ax_pr.legend(loc='upper right')
    ax_pr.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    curves_path = os.path.join(output_dir, '08_roc_pr_curves.png')
    plt.savefig(curves_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Evaluation charts saved to:\n  - {bar_path}\n  - {curves_path}")

def run_pipeline():
    print("="*70)
    print("STARTING END-TO-END FRAUD DETECTION BENCHMARKING PIPELINE")
    print("="*70)
    
    raw_data_path = os.path.join(PROJECT_ROOT, "data", "raw", "new_data.csv")
    reports_fig_dir = os.path.join(PROJECT_ROOT, "reports", "figures")
    reports_dir = os.path.join(PROJECT_ROOT, "reports")
    
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw dataset not found at: {raw_data_path}")
        
    file_size_mb = os.path.getsize(raw_data_path) / (1024 * 1024)
    print(f"Loading raw dataset: {raw_data_path} ({file_size_mb:.2f} MB)...")
    df_raw = pd.read_csv(raw_data_path)
    print(f"Loaded records: {len(df_raw):,} rows, {df_raw.shape[1]} columns.")
    
    # Step 1: Generate full EDA charts
    print("\n--- STEP 1: GENERATING EXPLORATORY DATA ANALYSIS (EDA) CHARTS ---")
    generate_all_eda_plots(df_raw, reports_fig_dir)
    
    # Step 2: Feature Engineering & Preprocessing
    print("\n--- STEP 2: PREPROCESSING & DOMAIN FEATURE ENGINEERING ---")
    # For training 5 distinct deep learning and tree models, we use a stratified sample of 120,000 records
    # (or full dataset if smaller) to enable thorough convergence across all 5 frameworks within reasonable academic runtime.
    sample_size = min(len(df_raw), 150000)
    df_processed = preprocess_data(
        df_raw,
        sample_size=sample_size,
        filter_relevant_types_only=True, # Fraud strictly exists in TRANSFER and CASH_OUT
        random_state=42
    )
    
    # Step 3: Train / Val / Test Splits
    print("\n--- STEP 3: CREATING STRATIFIED TRAIN / VAL / TEST SPLITS ---")
    splits = prepare_train_test_splits(
        df_processed,
        target_col='isFraud',
        test_size=0.20,
        val_size=0.10,
        scale_features=True,
        random_state=42
    )
    
    X_train, y_train = splits['X_train'], splits['y_train']
    X_val, y_val = splits['X_val'], splits['y_val']
    X_test, y_test = splits['X_test'], splits['y_test']
    
    results = []
    prob_dict = {}
    
    # Framework 1: XGBoost
    print("\n" + "="*50)
    print("FRAMEWORK 1: XGBoost (Extreme Gradient Boosting)")
    print("="*50)
    res_xgb = train_xgboost(X_train, y_train, X_val, y_val, X_test)
    eval_xgb = evaluate_predictions(y_test, res_xgb['y_prob'], "XGBoost", training_time=res_xgb['training_time'], inference_time=res_xgb['inference_time'])
    results.append(eval_xgb)
    prob_dict["XGBoost"] = res_xgb['y_prob']
    print(f"XGBoost -> RMSE: {eval_xgb['RMSE']}, R2: {eval_xgb['R2']}, PR-AUC: {eval_xgb['PR_AUC']}, F1: {eval_xgb['F1_Score']}")
    
    # Framework 2: Keras Deep Learning
    print("\n" + "="*50)
    print("FRAMEWORK 2: Keras (Deep Tabular MLP)")
    print("="*50)
    res_keras = train_keras(X_train, y_train, X_val, y_val, X_test, epochs=12, batch_size=512)
    eval_keras = evaluate_predictions(y_test, res_keras['y_prob'], "Keras", training_time=res_keras['training_time'], inference_time=res_keras['inference_time'])
    results.append(eval_keras)
    prob_dict["Keras"] = res_keras['y_prob']
    print(f"Keras -> RMSE: {eval_keras['RMSE']}, R2: {eval_keras['R2']}, PR-AUC: {eval_keras['PR_AUC']}, F1: {eval_keras['F1_Score']}")
    
    # Framework 3: PyTorch with fastai Tabular Learner
    print("\n" + "="*50)
    print("FRAMEWORK 3: PyTorch with fastai (Tabular Learner)")
    print("="*50)
    res_fastai = train_fastai_tabular(X_train, y_train, X_val, y_val, X_test, epochs=10, batch_size=512)
    eval_fastai = evaluate_predictions(y_test, res_fastai['y_prob'], "PyTorch with fastai", training_time=res_fastai['training_time'], inference_time=res_fastai['inference_time'])
    results.append(eval_fastai)
    prob_dict["PyTorch with fastai"] = res_fastai['y_prob']
    print(f"PyTorch with fastai -> RMSE: {eval_fastai['RMSE']}, R2: {eval_fastai['R2']}, PR-AUC: {eval_fastai['PR_AUC']}, F1: {eval_fastai['F1_Score']}")
    
    # Framework 4: PyTorch with TabNet
    print("\n" + "="*50)
    print("FRAMEWORK 4: PyTorch with TabNet (Attentive Transformer)")
    print("="*50)
    res_tabnet = train_tabnet(X_train, y_train, X_val, y_val, X_test, max_epochs=12, batch_size=1024)
    eval_tabnet = evaluate_predictions(y_test, res_tabnet['y_prob'], "PyTorch with TabNet", training_time=res_tabnet['training_time'], inference_time=res_tabnet['inference_time'])
    results.append(eval_tabnet)
    prob_dict["PyTorch with TabNet"] = res_tabnet['y_prob']
    print(f"PyTorch with TabNet -> RMSE: {eval_tabnet['RMSE']}, R2: {eval_tabnet['R2']}, PR-AUC: {eval_tabnet['PR_AUC']}, F1: {eval_tabnet['F1_Score']}")
    
    # Framework 5: PyTorch with Lightning Flash
    print("\n" + "="*50)
    print("FRAMEWORK 5: PyTorch with Lightning Flash")
    print("="*50)
    res_lightning = train_lightning_tabular(X_train, y_train, X_val, y_val, X_test, max_epochs=10, batch_size=512)
    eval_lightning = evaluate_predictions(y_test, res_lightning['y_prob'], "PyTorch with Lightning Flash", training_time=res_lightning['training_time'], inference_time=res_lightning['inference_time'])
    results.append(eval_lightning)
    prob_dict["PyTorch with Lightning Flash"] = res_lightning['y_prob']
    print(f"PyTorch with Lightning Flash -> RMSE: {eval_lightning['RMSE']}, R2: {eval_lightning['R2']}, PR-AUC: {eval_lightning['PR_AUC']}, F1: {eval_lightning['F1_Score']}")
    
    # Step 4: Comparative Reporting & Summary
    print("\n--- STEP 4: COMPILING FRAMEWORK COMPARATIVE SUMMARY ---")
    df_compare = compare_frameworks_summary(results)
    print("\n" + df_compare[['Framework', 'RMSE', 'R2', 'MAE', 'PR_AUC', 'F1_Score', 'Train_Time_s']].to_string(index=False))
    
    # Save CSV and JSON
    csv_out = os.path.join(reports_dir, "model_comparison_results.csv")
    json_out = os.path.join(reports_dir, "model_comparison_results.json")
    df_compare.to_csv(csv_out, index=False)
    with open(json_out, "w") as f:
        json.dump(results, f, indent=2)
        
    # Generate Evaluation Charts
    print("\n--- STEP 5: GENERATING COMPARATIVE EVALUATION CHARTS ---")
    plot_evaluation_charts(results, prob_dict, y_test, reports_fig_dir)
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS & BENCHMARKING COMPLETED SUCCESSFULLY!")
    print(f"Results saved to:\n  - {csv_out}\n  - {json_out}\n  - {reports_fig_dir}")
    print("="*70)
    return df_compare

if __name__ == "__main__":
    run_pipeline()
