"""
XGBoost Framework Implementation for Online Payment Fraud Detection.
Uses Extreme Gradient Boosted decision trees with class-imbalance weighting.
"""

import time
import xgboost as xgb
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    params: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Trains an XGBoost classifier with early stopping and class imbalance weighting.
    
    Returns:
        Dict with model, test probabilities, training time, and feature importances.
    """
    start_time = time.time()
    
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    scale_pos_weight = (n_neg / max(n_pos, 1))
    
    default_params = {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': scale_pos_weight,
        'eval_metric': 'logloss',
        'random_state': 42,
        'tree_method': 'hist',
        'early_stopping_rounds': 20
    }
    if params:
        default_params.update(params)
        
    print(f"[XGBoost] Initializing model with scale_pos_weight={scale_pos_weight:.2f}...")
    model = xgb.XGBClassifier(**default_params)
    
    # Train
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    training_time = time.time() - start_time
    print(f"[XGBoost] Training completed in {training_time:.2f} seconds.")
    
    # Inference on Test Set
    inf_start = time.time()
    y_prob = model.predict_proba(X_test)[:, 1]
    inference_time = time.time() - inf_start
    
    # Feature importances
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)
    
    return {
        "model": model,
        "y_prob": y_prob,
        "training_time": training_time,
        "inference_time": inference_time,
        "feature_importances": feat_imp.to_dict()
    }
