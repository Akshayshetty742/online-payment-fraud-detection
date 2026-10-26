"""
PyTorch with fastai-Style Tabular Deep Learning Implementation.
Implements the exact fastai Tabular architecture using LinBnDrop blocks
(Linear -> BatchNorm1d -> ReLU -> Dropout) with Cosine Annealing (1-Cycle) scheduling.
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, Any, List

class LinBnDrop(nn.Sequential):
    """fastai's core modular tabular building block: Linear -> BatchNorm1d -> ReLU -> Dropout"""
    def __init__(self, in_features: int, out_features: int, bn: bool = True, p: float = 0.2, act: bool = True):
        layers = [nn.Linear(in_features, out_features)]
        if act:
            layers.append(nn.ReLU(inplace=True))
        if bn:
            layers.append(nn.BatchNorm1d(out_features))
        if p > 0:
            layers.append(nn.Dropout(p))
        super().__init__(*layers)

class FastaiTabularNet(nn.Module):
    """
    Replicates fastai.tabular.model.TabularModel:
    Sequential blocks of LinBnDrop layers culminating in a single logit prediction.
    """
    def __init__(self, input_dim: int, layers: List[int] = [128, 64, 32], ps: List[float] = [0.2, 0.2, 0.1]):
        super().__init__()
        in_d = input_dim
        net_layers = []
        for out_d, p in zip(layers, ps):
            net_layers.append(LinBnDrop(in_d, out_d, bn=True, p=p, act=True))
            in_d = out_d
        # Final prediction layer
        net_layers.append(nn.Linear(in_d, 1))
        self.net = nn.Sequential(*net_layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

def train_fastai_tabular(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    epochs: int = 12,
    batch_size: int = 512,
    lr: float = 1e-3
) -> Dict[str, Any]:
    """
    Trains the fastai Tabular Deep Learning model with Cosine Annealing (1-cycle style).
    """
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Calculate pos_weight for BCEWithLogitsLoss
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    
    input_dim = X_train.shape[1]
    model = FastaiTabularNet(input_dim=input_dim, layers=[128, 64, 32], ps=[0.2, 0.2, 0.1]).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Datasets & Loaders
    train_ds = TensorDataset(torch.tensor(X_train.values, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val.values, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    print(f"[fastai PyTorch] Training TabularModel for {epochs} epochs on {device}...")
    best_val_loss = float('inf')
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
        scheduler.step()
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                l = criterion(model(bx), by)
                val_losses.append(l.item())
        avg_val_loss = np.mean(val_losses)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        
    training_time = time.time() - start_time
    print(f"[fastai PyTorch] Training completed in {training_time:.2f} seconds.")
    
    # Inference on Test Set
    inf_start = time.time()
    model.eval()
    test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(test_tensor)
        y_prob = torch.sigmoid(logits).cpu().numpy()
    inference_time = time.time() - inf_start
    
    return {
        "model": model,
        "y_prob": y_prob,
        "training_time": training_time,
        "inference_time": inference_time
    }
