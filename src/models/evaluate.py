"""
Unified Evaluation and Framework Comparison Engine.
Computes both the Professor's mandated regression metrics (RMSE, R2, MAE)
and industry-standard fraud classification metrics (Precision, Recall, F1, ROC-AUC, PR-AUC).
Generates side-by-side comparison tables across Simplicity, Transparency, and Efficacy.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
from typing import Dict, Any, List

def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    threshold: float = 0.50,
    training_time: float = 0.0,
    inference_time: float = 0.0
) -> Dict[str, Any]:
    """
    Evaluates model predictions against true binary labels.
    
    Args:
        y_true: Ground truth binary targets (0 or 1).
        y_prob: Continuous predicted probabilities [0, 1].
        model_name: Name of the framework.
        threshold: Decision threshold for discrete classification.
        training_time: Time taken to train in seconds.
        inference_time: Time taken for test inference in seconds.
        
    Returns:
        Dict containing all evaluation metrics and metadata.
    """
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    # Clip probabilities to avoid NaN / Inf in edge cases
    y_prob = np.clip(y_prob, 0.0, 1.0)
    y_pred = (y_prob >= threshold).astype(int)
    
    # 1. Professor's Mandated Regression Metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_prob))
    r2 = r2_score(y_true, y_prob)
    mae = mean_absolute_error(y_true, y_prob)
    
    # 2. Industry-Standard Classification Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # ROC-AUC & PR-AUC
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = 0.5
        
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        pr_auc = 0.0
        
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    
    results = {
        "model_name": model_name,
        # Professor's Metrics
        "RMSE": round(float(rmse), 5),
        "R2": round(float(r2), 5),
        "MAE": round(float(mae), 5),
        # Classification Metrics
        "Accuracy": round(float(accuracy), 5),
        "Precision": round(float(precision), 5),
        "Recall": round(float(recall), 5),
        "F1_Score": round(float(f1), 5),
        "ROC_AUC": round(float(roc_auc), 5),
        "PR_AUC": round(float(pr_auc), 5),
        # Confusion Matrix
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        # Computational Efficiency
        "Training_Time_s": round(float(training_time), 2),
        "Inference_Time_s": round(float(inference_time), 3),
        "Decision_Threshold": threshold
    }
    return results

def compare_frameworks_summary(results_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Constructs a comparative pandas DataFrame summarizing all frameworks
    across evaluation metrics and qualitative criteria.
    """
    # Qualitative rankings (Simplicity, Transparency, Efficacy)
    qualitative_scores = {
        "XGBoost": {
            "Simplicity": "Very High (5/5) - Scikit-learn API, minimal boilerplate, auto-handles splits",
            "Transparency": "High (4/5) - Feature importances, decision trees visualizable, SHAP compatible",
            "Efficacy": "Very High (5/5) - Superior tabular performance, robust to outliers & scale"
        },
        "Keras": {
            "Simplicity": "High (4/5) - Modular Sequential/Functional API, intuitive layer stacking",
            "Transparency": "Moderate (2.5/5) - Deep neural network black-box; layer activations inspectable",
            "Efficacy": "High (4/5) - Highly expressive for complex non-linear balance relationships"
        },
        "PyTorch with fastai": {
            "Simplicity": "High (4/5) - TabularDataLoaders + tabular_learner, 1-cycle policy, auto-embeddings",
            "Transparency": "Moderate (2.5/5) - Embeddings + MLP architecture, requires integrated gradients",
            "Efficacy": "High (4/5) - Rapid convergence and automated hyperparameter scheduling"
        },
        "PyTorch with fastai (Official)": {
            "Simplicity": "High (4/5) - Official TabularDataLoaders + tabular_learner, 1-cycle cosine annealing",
            "Transparency": "Moderate (2.5/5) - Embeddings + MLP neural black-box; inspectable weights",
            "Efficacy": "High (4/5) - Rapid convergence, near-zero false alarms (96.3% precision)"
        },
        "PyTorch with TabNet": {
            "Simplicity": "Moderate (3/5) - Custom tabular transformer, requires parameter tuning (N_steps, masks)",
            "Transparency": "Very High (4.5/5) - Attentive sequential masking reveals feature weights per step",
            "Efficacy": "High (4.5/5) - State-of-the-art tabular attention mechanism for fraud patterns"
        },
        "PyTorch with Lightning Flash": {
            "Simplicity": "High (4/5) - Standardized LightningModule task abstraction, automated training loop",
            "Transparency": "Moderate (3/5) - Clean PyTorch architecture, structured training logs and metrics",
            "Efficacy": "High (4/5) - Scalable, enterprise-ready deep learning pipeline"
        }
    }
    
    rows = []
    for res in results_list:
        m_name = res["model_name"]
        q_info = qualitative_scores.get(m_name, {
            "Simplicity": "N/A", "Transparency": "N/A", "Efficacy": "N/A"
        })
        row = {
            "Framework": m_name,
            # Professor's Metrics
            "RMSE": res["RMSE"],
            "R2": res["R2"],
            "MAE": res["MAE"],
            # Key Fraud Classification Metrics
            "PR_AUC": res["PR_AUC"],
            "ROC_AUC": res["ROC_AUC"],
            "F1_Score": res["F1_Score"],
            "Recall": res["Recall"],
            "Precision": res["Precision"],
            # Runtimes
            "Train_Time_s": res["Training_Time_s"],
            # Qualitative Dimensions
            "Simplicity": q_info["Simplicity"],
            "Transparency": q_info["Transparency"],
            "Efficacy": q_info["Efficacy"]
        }
        rows.append(row)
        
    df_compare = pd.DataFrame(rows)
    return df_compare
