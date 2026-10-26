# Dataset Inspection Summary: `new_data.csv`

- **Inspection Date**: 2026-10-26T11:49:58.585946
- **File Location**: `C:\Users\shett\OneDrive\Desktop\mlbd\data\raw\new_data.csv`
- **File Size**: 470.67 MB (493,534,783 bytes)
- **Dimensions**: **6,362,620 rows** × **11 columns**
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
- **Class Imbalance Ratio**: `773.7:1`

| Class / Value | Count | Percentage |
| :--- | :--- | :--- |
| `0` | 6,354,407 | 99.8709% |
| `1` | 8,213 | 0.1291% |

### Target Column: `isFlaggedFraud`
- **Class Imbalance Ratio**: `397662.75:1`

| Class / Value | Count | Percentage |
| :--- | :--- | :--- |
| `0` | 6,362,604 | 99.9997% |
| `1` | 16 | 0.0003% |

## Column Schema, Data Types & Missing Values

| Column Name | Data Type | Non-Null Count | Missing Count | Missing % | Unique Count | Sample Values |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `step` | `int64` | 6,362,620 | 0 | 0.0% | 743 | `1, 1, 1` |
| `type` | `str` | 6,362,620 | 0 | 0.0% | 5 | `PAYMENT, PAYMENT, TRANSFER` |
| `amount` | `float64` | 6,362,620 | 0 | 0.0% | 5,316,900 | `9839.64, 1864.28, 181.0` |
| `nameOrig` | `str` | 6,362,620 | 0 | 0.0% | 6,353,307 | `C1231006815, C1666544295, C1305486145` |
| `oldbalanceOrg` | `float64` | 6,362,620 | 0 | 0.0% | 1,845,844 | `170136.0, 21249.0, 181.0` |
| `newbalanceOrig` | `float64` | 6,362,620 | 0 | 0.0% | 2,682,586 | `160296.36, 19384.72, 0.0` |
| `nameDest` | `str` | 6,362,620 | 0 | 0.0% | 2,722,362 | `M1979787155, M2044282225, C553264065` |
| `oldbalanceDest` | `float64` | 6,362,620 | 0 | 0.0% | 3,614,697 | `0.0, 0.0, 0.0` |
| `newbalanceDest` | `float64` | 6,362,620 | 0 | 0.0% | 3,555,499 | `0.0, 0.0, 0.0` |
| `isFraud` | `int64` | 6,362,620 | 0 | 0.0% | 2 | `0, 0, 1` |
| `isFlaggedFraud` | `int64` | 6,362,620 | 0 | 0.0% | 2 | `0, 0, 0` |

## Numerical Features Statistical Summary

| Column | Mean | Std Dev | Min | 25% | Median (50%) | 75% | Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `step` | 243.3972 | 142.332 | 1.0 | 156.0 | 239.0 | 335.0 | 743.0 |
| `amount` | 179,861.9035 | 603,858.2315 | 0.0 | 13,389.57 | 74,871.94 | 208,721.4775 | 92,445,516.64 |
| `oldbalanceOrg` | 833,883.1041 | 2,888,242.673 | 0.0 | 0.0 | 14,208.0 | 107,315.175 | 59,585,040.37 |
| `newbalanceOrig` | 855,113.6686 | 2,924,048.503 | 0.0 | 0.0 | 0.0 | 144,258.41 | 49,585,040.37 |
| `oldbalanceDest` | 1,100,701.6665 | 3,399,180.113 | 0.0 | 0.0 | 132,705.665 | 943,036.7075 | 356,015,889.35 |
| `newbalanceDest` | 1,224,996.3982 | 3,674,128.9421 | 0.0 | 0.0 | 214,661.44 | 1,111,909.25 | 356,179,278.92 |
| `isFraud` | 0.0013 | 0.0359 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| `isFlaggedFraud` | 0.0 | 0.0016 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
