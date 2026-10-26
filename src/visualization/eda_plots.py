"""
Exploratory Data Analysis (EDA) Visualization Module.
Generates publication-quality charts for fraud distribution, transaction types,
amount distributions, balance anomalies, and correlation matrices.
"""

import os
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set clean aesthetic style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_palette(["#2b5c8f", "#d9534f", "#5cb85c", "#f0ad4e", "#5bc0de"])
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

def plot_class_distribution(df: pd.DataFrame, output_dir: str) -> str:
    """Plots class distribution illustrating the extreme imbalance."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    counts = df['isFraud'].value_counts()
    labels = ['Legitimate (0)', 'Fraudulent (1)']
    
    # Bar chart (Log scale)
    bars = ax1.bar(labels, counts.values, color=['#2b5c8f', '#d9534f'], width=0.5, edgecolor='black', alpha=0.85)
    ax1.set_yscale('log')
    ax1.set_ylabel('Transaction Count (Log Scale)', fontsize=12)
    ax1.set_title('Transaction Volume by Class (Log Scale)', fontsize=13, fontweight='bold', pad=12)
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval * 1.15, f"{int(yval):,}", ha='center', va='bottom', fontsize=11, fontweight='bold')
        
    # Donut chart
    wedges, texts, autotexts = ax2.pie(
        counts.values,
        labels=labels,
        autopct='%1.3f%%',
        colors=['#2b5c8f', '#d9534f'],
        startangle=140,
        pctdistance=0.75,
        explode=[0, 0.15],
        wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2)
    )
    plt.setp(autotexts, size=11, weight="bold", color="white")
    ax2.set_title('Class Imbalance Proportion', fontsize=13, fontweight='bold', pad=12)
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, '01_class_distribution.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    return out_path

def plot_transaction_types_and_fraud(df: pd.DataFrame, output_dir: str) -> str:
    """Plots transaction type breakdown and highlights where fraud actually occurs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 1. Total count by type
    type_counts = df['type'].value_counts()
    sns.barplot(x=type_counts.index, y=type_counts.values, ax=ax1, palette='Blues_r', edgecolor='black')
    ax1.set_title('Total Transaction Count by Type', fontsize=13, fontweight='bold', pad=12)
    ax1.set_ylabel('Total Transactions', fontsize=11)
    ax1.set_xlabel('Transaction Type', fontsize=11)
    for i, v in enumerate(type_counts.values):
        ax1.text(i, v + (max(type_counts.values)*0.01), f"{v:,}", ha='center', fontsize=9, rotation=15)
        
    # 2. Fraud count by type
    fraud_by_type = df.groupby('type')['isFraud'].sum()
    colors = ['#d9534f' if v > 0 else '#95a5a6' for v in fraud_by_type.values]
    bars = ax2.bar(fraud_by_type.index, fraud_by_type.values, color=colors, edgecolor='black', alpha=0.9)
    ax2.set_title('Fraud Incident Count by Transaction Type (Key Finding)', fontsize=13, fontweight='bold', pad=12)
    ax2.set_ylabel('Fraud Cases', fontsize=11)
    ax2.set_xlabel('Transaction Type', fontsize=11)
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 10, f"{int(yval):,}", ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    plt.tight_layout()
    out_path = os.path.join(output_dir, '02_transaction_types_and_fraud.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    return out_path

def plot_amount_distribution(df: pd.DataFrame, output_dir: str) -> str:
    """Plots amount distribution comparing legitimate vs fraudulent transactions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Boxplot with log-transformed amount
    df_plot = df.copy()
    df_plot['log_amount'] = np.log10(df_plot['amount'] + 1.0)
    df_plot['Fraud_Label'] = df_plot['isFraud'].map({0: 'Legitimate', 1: 'Fraud'})
    
    sns.boxplot(x='Fraud_Label', y='log_amount', data=df_plot, ax=ax1, palette=['#2b5c8f', '#d9534f'], width=0.4)
    ax1.set_title('Transaction Amount ($) Distribution [Log10 Scale]', fontsize=13, fontweight='bold', pad=12)
    ax1.set_ylabel('Log10(Amount + 1)', fontsize=11)
    ax1.set_xlabel('Transaction Category', fontsize=11)
    
    # KDE distribution for relevant types (TRANSFER & CASH_OUT)
    rel_df = df_plot[df_plot['type'].isin(['TRANSFER', 'CASH_OUT'])]
    sns.kdeplot(data=rel_df[rel_df['isFraud'] == 0]['log_amount'], ax=ax2, label='Legitimate', color='#2b5c8f', fill=True, alpha=0.3)
    sns.kdeplot(data=rel_df[rel_df['isFraud'] == 1]['log_amount'], ax=ax2, label='Fraudulent', color='#d9534f', fill=True, alpha=0.3)
    ax2.set_title('KDE Density of Amount: Legitimate vs Fraud', fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel('Log10(Amount + 1)', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.legend(fontsize=11)
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, '03_amount_distributions.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    return out_path

def plot_balance_anomalies(df: pd.DataFrame, output_dir: str) -> str:
    """Plots sender balance changes vs transaction amounts illustrating account clearing."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    fraud_df = df[df['isFraud'] == 1]
    # Sample legitimate for readable scatter
    legit_df = df[df['isFraud'] == 0].sample(n=min(len(fraud_df), 3000), random_state=42)
    
    # 1. Legitimate transactions
    ax1.scatter(legit_df['oldbalanceOrg'], legit_df['newbalanceOrig'], alpha=0.25, color='#2b5c8f', s=15, label='Legitimate')
    ax1.set_title('Legitimate: Old Balance vs New Balance (Origin)', fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel('Old Balance Origin ($)', fontsize=11)
    ax1.set_ylabel('New Balance Origin ($)', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # 2. Fraudulent transactions (notice clustering on newbalanceOrig = 0)
    ax2.scatter(fraud_df['oldbalanceOrg'], fraud_df['newbalanceOrig'], alpha=0.35, color='#d9534f', s=18, label='Fraud')
    ax2.set_title('Fraudulent: Old Balance vs New Balance (Origin)', fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel('Old Balance Origin ($)', fontsize=11)
    ax2.set_ylabel('New Balance Origin ($)', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.6)
    # Add annotation pointing to y=0 line
    ax2.annotate('Accounts Completely Emptied (New Balance = 0)',
                 xy=(np.median(fraud_df['oldbalanceOrg']), 0),
                 xytext=(np.median(fraud_df['oldbalanceOrg'])*0.5, np.max(fraud_df['oldbalanceOrg'])*0.25),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=10, fontweight='bold', color='#d9534f')
                 
    plt.tight_layout()
    out_path = os.path.join(output_dir, '04_balance_anomalies.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    return out_path

def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str) -> str:
    """Plots correlation matrix of numerical and domain-engineered features."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Compute domain features for correlation
    numeric_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFraud']
    avail_cols = [c for c in numeric_cols if c in df.columns]
    corr = df[avail_cols].corr()
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.8, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .75}, annot=True, fmt='.2f', ax=ax)
    ax.set_title('Feature Correlation Matrix (Pearson)', fontsize=14, fontweight='bold', pad=14)
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, '05_correlation_matrix.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    return out_path

def plot_temporal_fraud_rate(df: pd.DataFrame, output_dir: str) -> str:
    """Plots transaction volume and fraud frequency across 24-hour cycle."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    
    df_temp = df.copy()
    df_temp['hour'] = df_temp['step'] % 24
    
    hourly_total = df_temp.groupby('hour').size()
    hourly_fraud = df_temp.groupby('hour')['isFraud'].sum()
    hourly_rate = (hourly_fraud / hourly_total) * 100
    
    # 1. Total volume
    ax1.plot(hourly_total.index, hourly_total.values, marker='o', color='#2b5c8f', linewidth=2)
    ax1.set_title('Hourly Total Transaction Volume (24-Hour Cycle)', fontsize=12, fontweight='bold', pad=10)
    ax1.set_ylabel('Total Transactions', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # 2. Fraud rate percentage
    ax2.plot(hourly_rate.index, hourly_rate.values, marker='s', color='#d9534f', linewidth=2)
    ax2.set_title('Hourly Fraud Rate (%) - Peak Anomalies During Off-Peak Hours', fontsize=12, fontweight='bold', pad=10)
    ax2.set_ylabel('Fraud Rate (%)', fontsize=11)
    ax2.set_xlabel('Hour of Day (0 - 23)', fontsize=11)
    ax2.set_xticks(range(0, 24))
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, '06_temporal_fraud_rate.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    return out_path

def generate_all_eda_plots(df: pd.DataFrame, output_dir: str) -> list:
    """Generates all 6 EDA figures and returns their filepaths."""
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    print("Generating EDA Plot 1: Class Distribution...")
    paths.append(plot_class_distribution(df, output_dir))
    
    print("Generating EDA Plot 2: Transaction Types & Fraud Incidents...")
    paths.append(plot_transaction_types_and_fraud(df, output_dir))
    
    print("Generating EDA Plot 3: Amount Distributions...")
    paths.append(plot_amount_distribution(df, output_dir))
    
    print("Generating EDA Plot 4: Balance Anomalies...")
    paths.append(plot_balance_anomalies(df, output_dir))
    
    print("Generating EDA Plot 5: Correlation Matrix...")
    paths.append(plot_correlation_heatmap(df, output_dir))
    
    print("Generating EDA Plot 6: Temporal Fraud Rate...")
    paths.append(plot_temporal_fraud_rate(df, output_dir))
    
    print(f"All {len(paths)} EDA figures generated successfully in: {output_dir}")
    return paths
