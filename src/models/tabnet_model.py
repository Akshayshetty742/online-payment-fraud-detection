"""
PyTorch TabNet Framework Implementation for Online Payment Fraud Detection.
Uses attentive transformer sequential decision steps for tabular interpretability.
"""

import time
import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from typing import Dict, Any

def train_tabnet(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    max_epochs: int = 20,
    batch_size: int = 1024,
    virtual_batch_size: int = 128
) -> Dict[str, Any]:
    """
    Trains PyTorch TabNetClassifier with attentive feature selection and early stopping.
    """
    start_time = time.time()
    
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    # Class weights for TabNet: 1 (class weight dictionary or pos_weight)
    class_weights = {
        0: 1.0,
        1: float(n_neg / max(n_pos, 1))
    }
    
    print(f"[TabNet] Initializing TabNetClassifier (Decision steps: 3)...")
    clf = TabNetClassifier(
        n_d=16,
        n_a=16,
        n_steps=3,
        gamma=1.3,
        lambda_sparse=1e-4,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size": 10, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type='sparsemax',
        device_name='auto',
        verbose=0
    )
    
    # Train
    clf.fit(
        X_train=X_train.values,
        y_train=y_train,
        eval_set=[(X_val.values, y_val)],
        eval_name=['valid'],
        eval_metric=['auc'],
        max_epochs=max_epochs,
        patience=5,
        batch_size=batch_size,
        virtual_batch_size=virtual_batch_size,
        num_workers=0,
        drop_last=False,
        weights=class_weights
    )
    training_time = time.time() - start_time
    print(f"[TabNet] Training completed in {training_time:.2f} seconds.")
    
    # Inference on Test Set
    inf_start = time.time()
    y_prob = clf.predict_proba(X_test.values)[:, 1]
    inference_time = time.time() - inf_start
    
    # Feature importances extracted from sequential decision masks
    importances = clf.feature_importances_
    feat_imp = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)
    
    return {
        "model": clf,
        "y_prob": y_prob,
        "training_time": training_time,
        "inference_time": inference_time,
        "feature_importances": feat_imp.to_dict()
    }
