"""
Feature engineering and preprocessing pipeline for online payment fraud detection.
Transforms raw transaction records into machine learning ready features.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Tuple, Dict, Any

def create_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates financial fraud domain features from transaction balances and steps.
    
    Domain rationale:
    1. errorBalanceOrig: Checks if sender's post-transaction balance matches expected balance.
    2. errorBalanceDest: Checks if recipient's balance increased by exact transaction amount.
    3. hour_of_day: Periodicity of fraudulent vs legitimate transactions.
    4. is_merchant_dest: Whether the recipient is a merchant ('M') or customer ('C').
    5. orig_emptied: Flag indicating if the originating account was completely emptied.
    """
    df_feat = df.copy()
    
    # 1. Temporal cyclical features
    df_feat['hour_of_day'] = df_feat['step'] % 24
    df_feat['day_of_week'] = (df_feat['step'] // 24) % 7
    
    # 2. Balance discrepancy features (fraud signatures)
    df_feat['errorBalanceOrig'] = df_feat['newbalanceOrig'] + df_feat['amount'] - df_feat['oldbalanceOrg']
    df_feat['errorBalanceDest'] = df_feat['oldbalanceDest'] + df_feat['amount'] - df_feat['newbalanceDest']
    
    # 3. Behavioral flags
    df_feat['orig_emptied'] = ((df_feat['newbalanceOrig'] == 0) & (df_feat['oldbalanceOrg'] > 0)).astype(int)
    df_feat['dest_zero_start'] = (df_feat['oldbalanceDest'] == 0).astype(int)
    df_feat['dest_zero_end'] = (df_feat['newbalanceDest'] == 0).astype(int)
    
    # 4. Merchant recipient flag
    if 'nameDest' in df_feat.columns:
        df_feat['is_merchant_dest'] = df_feat['nameDest'].astype(str).str.startswith('M').astype(int)
        
    # 5. Relative transaction ratio
    df_feat['amount_to_oldbalance_ratio'] = np.where(
        df_feat['oldbalanceOrg'] > 0,
        df_feat['amount'] / (df_feat['oldbalanceOrg'] + 1.0),
        0.0
    )
    
    return df_feat

def preprocess_data(
    df: pd.DataFrame,
    sample_size: int = None,
    filter_relevant_types_only: bool = False,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Preprocesses the raw dataset:
    - Optionally samples a representative subset preserving class distribution.
    - Applies domain feature engineering.
    - Encodes categorical transaction types.
    - Drops raw non-generalizable ID strings (nameOrig, nameDest).
    """
    print(f"Initial dataset shape: {df.shape}")
    
    # In PaySim fraud ONLY occurs in TRANSFER and CASH_OUT.
    # Optionally filter to relevant types or keep all types for complete pipeline.
    if filter_relevant_types_only:
        df = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])].copy()
        print(f"Filtered to TRANSFER & CASH_OUT: {df.shape}")
        
    if sample_size and sample_size < len(df):
        print(f"Performing stratified sampling to {sample_size:,} records...")
        # Stratified sampling to preserve fraud ratio
        pos_df = df[df['isFraud'] == 1]
        neg_df = df[df['isFraud'] == 0]
        
        pos_ratio = len(pos_df) / len(df)
        n_pos = max(int(sample_size * pos_ratio), min(len(pos_df), 1000))
        n_neg = sample_size - n_pos
        
        sampled_pos = pos_df.sample(n=min(n_pos, len(pos_df)), random_state=random_state)
        sampled_neg = neg_df.sample(n=min(n_neg, len(neg_df)), random_state=random_state)
        df = pd.concat([sampled_pos, sampled_neg]).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        print(f"Sampled shape: {df.shape} (Fraud cases: {df['isFraud'].sum():,})")
        
    # Apply domain feature engineering
    df = create_domain_features(df)
    
    # One-hot encode transaction type
    if 'type' in df.columns:
        type_dummies = pd.get_dummies(df['type'], prefix='type', dtype=int)
        df = pd.concat([df, type_dummies], axis=1)
        df.drop(columns=['type'], inplace=True)
        
    # Drop identifier strings
    cols_to_drop = [c for c in ['nameOrig', 'nameDest'] if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)
    
    return df

def prepare_train_test_splits(
    df: pd.DataFrame,
    target_col: str = 'isFraud',
    test_size: float = 0.20,
    val_size: float = 0.10,
    scale_features: bool = True,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Splits preprocessed data into Train, Validation, and Test sets with stratification.
    Optionally fits StandardScaler on Train and transforms Val and Test.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col].values
    
    # First split: train+val vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # Second split: train vs val
    val_ratio_adjusted = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio_adjusted, stratify=y_train_val, random_state=random_state
    )
    
    feature_names = list(X.columns)
    scaler = None
    
    if scale_features:
        scaler = StandardScaler()
        # Scale numeric features
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names)
        X_val = pd.DataFrame(scaler.transform(X_val), columns=feature_names)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=feature_names)
    else:
        X_train = X_train.reset_index(drop=True)
        X_val = X_val.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        
    print(f"Split completed:")
    print(f"  Train set:      {X_train.shape[0]:,} samples (Fraud: {sum(y_train):,})")
    print(f"  Validation set: {X_val.shape[0]:,} samples (Fraud: {sum(y_val):,})")
    print(f"  Test set:       {X_test.shape[0]:,} samples (Fraud: {sum(y_test):,})")
    
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": feature_names,
        "scaler": scaler
    }
