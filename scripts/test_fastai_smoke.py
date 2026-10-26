"""
Smoke Test for Official fastai Compatibility in Isolated Virtual Environment.
Tests:
1. Import of official fastai and fastai.tabular.all
2. Creation of TabularDataLoaders on a tiny subset (500 rows) of the dataset
3. Instantiation of tabular_learner
4. Training for 1 epoch with fit_one_cycle
5. Model inference / prediction
"""

import os
import sys
import time

def run_smoke_test():
    print("="*60)
    print("STARTING FASTAI ISOLATED SMOKE TEST")
    print("="*60)
    
    # Step 1: Verify Imports
    print("\n[Step 1] Attempting to import fastai...")
    try:
        import fastai
        print(f"  -> SUCCESS: fastai version {fastai.__version__} imported.")
    except Exception as e:
        print(f"  -> FAILED: Could not import fastai: {e}")
        sys.exit(1)
        
    print("\n[Step 2] Attempting to import fastai.tabular.all...")
    try:
        from fastai.tabular.all import (
            TabularDataLoaders, tabular_learner, accuracy,
            Categorify, FillMissing, Normalize, RocAucBinary
        )
        print("  -> SUCCESS: fastai.tabular.all imported successfully.")
    except Exception as e:
        print(f"  -> FAILED: Could not import fastai.tabular.all: {e}")
        sys.exit(1)
        
    # Step 3: Load tiny subset from raw data
    print("\n[Step 3] Loading tiny dataset subset (500 rows)...")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, "data", "raw", "new_data.csv")
    
    import pandas as pd
    import numpy as np
    
    # Load 500 rows with both classes
    df_raw = pd.read_csv(data_path, nrows=50000)
    fraud_rows = df_raw[df_raw['isFraud'] == 1].head(50)
    non_fraud_rows = df_raw[df_raw['isFraud'] == 0].head(450)
    df_sample = pd.concat([fraud_rows, non_fraud_rows]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Add minimal domain features
    df_sample['errorBalanceOrig'] = df_sample['newbalanceOrig'] + df_sample['amount'] - df_sample['oldbalanceOrg']
    df_sample['orig_emptied'] = ((df_sample['newbalanceOrig'] == 0) & (df_sample['oldbalanceOrg'] > 0)).astype(int)
    df_sample['is_merchant_dest'] = df_sample['nameDest'].astype(str).str.startswith('M').astype(int)
    
    print(f"  -> Sample created: {len(df_sample)} rows, {df_sample['isFraud'].sum()} fraud cases.")
    
    # Step 4: TabularDataLoaders
    print("\n[Step 4] Building TabularDataLoaders...")
    cat_names = ['type', 'orig_emptied', 'is_merchant_dest']
    cont_names = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'errorBalanceOrig']
    
    dls = TabularDataLoaders.from_df(
        df_sample,
        y_names='isFraud',
        cat_names=cat_names,
        cont_names=cont_names,
        procs=[Categorify, FillMissing, Normalize],
        valid_pct=0.2,
        bs=64,
        seed=42
    )
    print("  -> DataLoaders created successfully.")
    
    # Step 5: tabular_learner
    print("\n[Step 5] Initializing tabular_learner...")
    learn = tabular_learner(
        dls,
        layers=[64, 32],
        metrics=accuracy
    )
    print("  -> Learner initialized successfully.")
    
    # Step 6: 1-Cycle training test
    print("\n[Step 6] Running fit_one_cycle(1, 1e-2)...")
    t0 = time.time()
    learn.fit_one_cycle(1, 1e-2)
    elapsed = time.time() - t0
    print(f"  -> 1-Cycle training completed in {elapsed:.2f} seconds.")
    
    # Step 7: Prediction test
    print("\n[Step 7] Testing model inference on test DataFrame...")
    test_df = df_sample.drop(columns=['isFraud']).head(5)
    test_dl = dls.test_dl(test_df)
    preds, _ = learn.get_preds(dl=test_dl)
    print(f"  -> Batch test_dl prediction succeeded! Shape: {preds.shape}")
    print(f"  -> First 3 predicted probability vectors:\n{preds[:3].numpy()}")
    
    print("\n" + "="*60)
    print("FASTAI SMOKE TEST PASSED COMPLETELY WITH 0 ERRORS!")
    print("="*60)

if __name__ == "__main__":
    run_smoke_test()
