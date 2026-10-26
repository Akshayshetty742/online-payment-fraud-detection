"""
Official fastai Framework Implementation for Tabular Fraud Detection.
Uses fastai.tabular.all (TabularDataLoaders, tabular_learner, fit_one_cycle).
"""

import time
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any

from fastai.tabular.all import (
    TabularDataLoaders, tabular_learner, accuracy,
    RocAucBinary, set_seed, CrossEntropyLossFlat
)

def train_fastai_tabular(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    epochs: int = 10,
    batch_size: int = 512,
    lr: float = 1e-2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Trains a fastai tabular learner using official fastai.tabular.all API.

    Parameters:
    - X_train, X_val, X_test: Scaled preprocessed DataFrames
    - y_train, y_val: Binary ground-truth arrays
    - epochs: Number of 1-Cycle training epochs
    - batch_size: DataLoader mini-batch size
    - lr: Peak learning rate for 1-Cycle cosine annealing
    - random_state: Reproducibility seed

    Returns:
    - Dict with 'model' (fastai Learner), 'y_prob' (1D numpy array of test probabilities),
      'training_time' (seconds), 'inference_time' (seconds)
    """
    set_seed(random_state, reproducible=True)
    start_time = time.time()
    
    feature_names = list(X_train.columns)
    
    # Prepare unified DataFrame with train and val partitions
    # Categorical string labels '0' and '1' ensure fastai treats this as binary classification with Softmax
    df_train = X_train.copy()
    df_train['isFraud'] = y_train.astype(int).astype(str)
    
    df_val = X_val.copy()
    df_val['isFraud'] = y_val.astype(int).astype(str)
    
    df_all = pd.concat([df_train, df_val], ignore_index=True)
    n_train = len(df_train)
    n_total = len(df_all)
    splits = (list(range(n_train)), list(range(n_train, n_total)))
    
    print(f"[fastai] Initializing TabularDataLoaders ({n_train:,} train, {len(df_val):,} val)...")
    dls = TabularDataLoaders.from_df(
        df_all,
        y_names='isFraud',
        cont_names=feature_names,
        cat_names=[],
        procs=[], # Preprocessing already fitted on training set
        splits=splits,
        bs=batch_size
    )
    
    # Calculate pos_weight for binary classification
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    class_weights = torch.tensor([1.0, min(float(n_neg / max(1, n_pos)), 50.0)], dtype=torch.float32)
    
    print(f"[fastai] Building tabular_learner (layers=[128, 64, 32])...")
    learn = tabular_learner(
        dls,
        layers=[128, 64, 32],
        loss_func=CrossEntropyLossFlat(weight=class_weights),
        metrics=[accuracy, RocAucBinary()]
    )

    print(f"[fastai] Training with fit_one_cycle for {epochs} epochs (lr={lr})...")
    learn.fit_one_cycle(epochs, lr)
    training_time = time.time() - start_time
    print(f"[fastai] Training completed in {training_time:.2f} seconds.")

    # Vectorized batch inference on Test set
    inf_start = time.time()
    test_dl = dls.test_dl(X_test)
    preds, _ = learn.get_preds(dl=test_dl)
    inference_time = time.time() - inf_start

    # preds has shape (len(X_test), 2): [P(y=0), P(y=1)]
    y_prob = preds[:, 1].numpy().astype(np.float32)
    print(f"[fastai] Inference on {len(X_test):,} test samples completed in {inference_time:.3f} seconds.")

    return {
        "model": learn,
        "y_prob": y_prob,
        "training_time": training_time,
        "inference_time": inference_time
    }
