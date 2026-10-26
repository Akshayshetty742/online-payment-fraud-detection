# Dataset Inspection Summary: `new_data.csv`

- **Inspection Date**: 2026-10-26T11:26:23.669663
- **File Location**: `C:\Users\shett\OneDrive\Desktop\mlbd\data\raw\new_data.csv`
- **File Size**: 230.0 MB (241,172,414 bytes)
- **Dimensions**: **3,114,353 rows** × **11 columns**
- **Duplicate Rows**: 0 (0.0%)
- **Identified Primary Target**: `isFraud`

## ⚠️ Crucial Assignment Notice: Problem Formulation Discrepancy
> **Note for Professor / Viva Clarification**:
> The dataset target column (`isFraud`) is binary (0 = legitimate transaction, 1 = fraudulent transaction), making payment fraud detection intrinsically a **binary classification problem**.
> However, the professor's assignment guidelines mention regression metrics (**RMSE**, **R²**, and **MAE**).
> - In this project, we explicitly record and compute the professor's specified regression metrics (interpreting predictions continuously / as probabilities) while also providing appropriate classification metrics (Precision, Recall, F1, ROC-AUC, PR-AUC).
> - This discrepancy has been explicitly flagged for clarification and viva presentation.

## Target Column Distribution

### Target Column: `isFraud`
- **Class Imbalance Ratio**: `1170.25:1`

| Class / Value | Count | Percentage |
| :--- | :--- | :--- |
| `0` | 3,111,694 | 99.9146% |
| `1` | 2,659 | 0.0854% |

### Target Column: `isFlaggedFraud`
- **Class Imbalance Ratio**: `3114352.0:1`

| Class / Value | Count | Percentage |
| :--- | :--- | :--- |
| `0` | 3,114,352 | 100.0% |
| `1` | 1 | 0.0% |

## Column Schema, Data Types & Missing Values

| Column Name | Data Type | Non-Null Count | Missing Count | Missing % | Unique Count | Sample Values |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `step` | `int64` | 3,114,353 | 0 | 0.0% | 236 | `1, 1, 1` |
| `type` | `str` | 3,114,353 | 0 | 0.0% | 5 | `PAYMENT, PAYMENT, TRANSFER` |
| `amount` | `float64` | 3,114,353 | 0 | 0.0% | 2,819,687 | `9839.64, 1864.28, 181.0` |
| `nameOrig` | `str` | 3,114,353 | 0 | 0.0% | 3,112,055 | `C1231006815, C1666544295, C1305486145` |
| `oldbalanceOrg` | `float64` | 3,114,353 | 0 | 0.0% | 1,007,150 | `170136.0, 21249.0, 181.0` |
| `newbalanceOrig` | `float64` | 3,114,353 | 0 | 0.0% | 1,340,601 | `160296.36, 19384.72, 0.0` |
| `nameDest` | `str` | 3,114,353 | 0 | 0.0% | 1,322,698 | `M1979787155, M2044282225, C553264065` |
| `oldbalanceDest` | `float64` | 3,114,353 | 0 | 0.0% | 1,789,913 | `0.0, 0.0, 0.0` |
| `newbalanceDest` | `float64` | 3,114,353 | 0 | 0.0% | 1,635,807 | `0.0, 0.0, 0.0` |
| `isFraud` | `int64` | 3,114,353 | 0 | 0.0% | 2 | `0, 0, 1` |
| `isFlaggedFraud` | `int64` | 3,114,353 | 0 | 0.0% | 2 | `0, 0, 0` |

## Numerical Features Statistical Summary

| Column | Mean | Std Dev | Min | 25% | Median (50%) | 75% | Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `step` | 128.2565 | 77.6647 | 1.0 | 39.0 | 154.0 | 188.0 | 236.0 |
| `amount` | 158,546.1669 | 268,208.8516 | 0.0 | 12,744.41 | 76,911.12 | 212,253.85 | 10,000,000.0 |
| `oldbalanceOrg` | 843,161.9683 | 2,909,341.5687 | 0.0 | 0.0 | 14,599.0 | 113,374.0 | 38,939,424.03 |
| `newbalanceOrig` | 864,721.5613 | 2,945,796.7399 | 0.0 | 0.0 | 0.0 | 153,381.04 | 38,946,233.02 |
| `oldbalanceDest` | 996,592.4 | 2,303,332.0892 | 0.0 | 0.0 | 139,077.31 | 946,675.74 | 42,655,769.2 |
| `newbalanceDest` | 1,104,047.3373 | 2,385,704.0806 | 0.0 | 0.0 | 225,911.5 | 1,130,956.83 | 42,655,769.2 |
| `isFraud` | 0.0009 | 0.0292 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| `isFlaggedFraud` | 0.0 | 0.0006 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
