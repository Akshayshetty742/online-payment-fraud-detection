"""
Benchmark script for Official fastai 2.8.12 Integration.
Evaluates the official fastai tabular model on the exact standardized 150,000-sample partition.
Outputs exact metrics and saves to reports/fastai_official_benchmark_results.json.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.features.build_features import preprocess_data, prepare_train_test_splits
from src.models.fastai_model import train_fastai_tabular
from src.models.evaluate import evaluate_predictions

def run_fastai_benchmark():
    print("="*70)
    print("STARTING BENCHMARK: OFFICIAL FASTAI 2.8.12 INTEGRATION")
    print("="*70)
    
    data_path = os.path.join(PROJECT_ROOT, "data", "raw", "new_data.csv")
    print(f"Loading raw dataset from {data_path}...")
    df_raw = pd.read_csv(data_path)
    
    # Identical 150k stratified sample
    sample_size = min(len(df_raw), 150000)
    df_processed = preprocess_data(
        df_raw,
        sample_size=sample_size,
        filter_relevant_types_only=True,
        random_state=42
    )
    
    # Identical stratified splits
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
    
    print("\n" + "="*50)
    print("TRAINING OFFICIAL FASTAI TABULAR LEARNER")
    print("="*50)
    res_fastai = train_fastai_tabular(
        X_train, y_train,
        X_val, y_val,
        X_test,
        epochs=10,
        batch_size=512,
        lr=1e-2,
        random_state=42
    )
    
    eval_fastai = evaluate_predictions(
        y_test,
        res_fastai['y_prob'],
        model_name="PyTorch with fastai (Official)",
        training_time=res_fastai['training_time'],
        inference_time=res_fastai['inference_time']
    )
    
    print("\n" + "="*70)
    print("OFFICIAL FASTAI 2.8.12 EVALUATION METRICS:")
    print("="*70)
    for k, v in eval_fastai.items():
        print(f"  {k:20s}: {v}")
        
    # Save results
    out_json = os.path.join(PROJECT_ROOT, "reports", "fastai_official_benchmark_results.json")
    with open(out_json, "w") as f:
        json.dump(eval_fastai, f, indent=2)
        
    print(f"\nResults saved to: {out_json}")
    return eval_fastai

if __name__ == "__main__":
    run_fastai_benchmark()
