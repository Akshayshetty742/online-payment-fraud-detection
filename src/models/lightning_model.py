"""
PyTorch with Lightning Flash Framework Implementation.
Implements the task-oriented tabular classification abstraction using PyTorch Lightning.
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from typing import Dict, Any

class LightningTabularClassifier(pl.LightningModule):
    """
    LightningModule encapsulating Flash-style tabular classification task.
    """
    def __init__(self, input_dim: int, lr: float = 1e-3, pos_weight: float = 1.0):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.pos_weight = torch.tensor([pos_weight], dtype=torch.float32)
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.20),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.10),
            
            nn.Linear(32, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(-1)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        pw = self.pos_weight.to(self.device)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)
        self.log('train_loss', loss, prog_bar=False)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
        self.log('val_loss', loss, prog_bar=False)
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

def train_lightning_tabular(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    max_epochs: int = 12,
    batch_size: int = 512
) -> Dict[str, Any]:
    """
    Trains the PyTorch Lightning Tabular Classification Task.
    """
    start_time = time.time()
    
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    pos_weight = float(n_neg / max(n_pos, 1))
    
    input_dim = X_train.shape[1]
    task = LightningTabularClassifier(input_dim=input_dim, lr=1e-3, pos_weight=pos_weight)
    
    train_ds = TensorDataset(torch.tensor(X_train.values, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val.values, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    print(f"[PyTorch Lightning] Training LightningTabularClassifier for {max_epochs} epochs...")
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False
    )
    
    trainer.fit(task, train_dataloaders=train_loader, val_dataloaders=val_loader)
    training_time = time.time() - start_time
    print(f"[PyTorch Lightning] Training completed in {training_time:.2f} seconds.")
    
    # Inference on Test Set
    inf_start = time.time()
    task.eval()
    device = task.device
    test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = task(test_tensor)
        y_prob = torch.sigmoid(logits).cpu().numpy()
    inference_time = time.time() - inf_start
    
    return {
        "model": task,
        "y_prob": y_prob,
        "training_time": training_time,
        "inference_time": inference_time
    }
