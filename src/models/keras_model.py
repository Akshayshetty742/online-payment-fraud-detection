"""
Keras Deep Learning Framework Implementation for Online Payment Fraud Detection.
Uses multi-layer perceptron (MLP) with BatchNormalization, Dropout, and balanced cross-entropy.
"""

import time
import os

# Set backend before importing Keras
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import pandas as pd
import keras
from keras import layers, models, callbacks
from typing import Dict, Any

def build_keras_mlp(input_dim: int) -> keras.Model:
    """Builds a regularized Deep Tabular MLP architecture."""
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    return model

def train_keras(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    epochs: int = 15,
    batch_size: int = 512
) -> Dict[str, Any]:
    """
    Trains Keras MLP with class weights and early stopping.
    """
    start_time = time.time()
    
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    total = n_pos + n_neg
    # Balanced class weights
    class_weight = {
        0: (total / (2.0 * max(n_neg, 1))),
        1: (total / (2.0 * max(n_pos, 1)))
    }
    
    input_dim = X_train.shape[1]
    print(f"[Keras] Building Deep Tabular MLP (Input dim: {input_dim})...")
    model = build_keras_mlp(input_dim)
    
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=4,
        restore_best_weights=True,
        verbose=0
    )
    
    # Train
    model.fit(
        X_train.values, y_train,
        validation_data=(X_val.values, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=[early_stop],
        verbose=0
    )
    training_time = time.time() - start_time
    print(f"[Keras] Training completed in {training_time:.2f} seconds.")
    
    # Inference on Test Set
    inf_start = time.time()
    y_prob = model.predict(X_test.values, batch_size=batch_size, verbose=0).ravel()
    inference_time = time.time() - inf_start
    
    return {
        "model": model,
        "y_prob": y_prob,
        "training_time": training_time,
        "inference_time": inference_time
    }
