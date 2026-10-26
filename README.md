# Online Payment Fraud Detection using Machine Learning

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Project Phase](https://img.shields.io/badge/Phase-1%3A%20Setup%20%26%20Inspection-success.svg)](#)
[![Subject](https://img.shields.io/badge/Course-ML%20with%20Business%20Data-orange.svg)](#)

A structured machine learning college project designed to detect fraudulent transactions in online financial payments using five state-of-the-art modeling frameworks.

---

## 📌 Project Overview

In digital banking and modern e-commerce, real-time automated fraud detection is critical for safeguarding customer assets and minimizing institutional losses. This project analyzes large-scale transactional data to distinguish legitimate financial behavior from illicit, fraudulent activities.

### Professor's Assignment Requirements
1. **Prescribed Dataset**:
   - Source: [Google Drive Dataset Link](https://drive.google.com/file/d/127JqP3WGjBVihR-ZcUR86T3wwy3_g63v/view)
   - File ID: `127JqP3WGjBVihR-ZcUR86T3wwy3_g63v`
   - File: `new_data.csv` (~471 MB)
2. **Exploratory Data Analysis (EDA)**:
   - In-depth statistical analysis, distribution of transaction types, class imbalance study, and feature correlation.
3. **Five Modeling Frameworks**:
   - **XGBoost** (Extreme Gradient Boosting on decision trees)
   - **Keras** (Neural network architectures)
   - **PyTorch with fastai** (High-level deep learning abstractions)
   - **PyTorch with TabNet** (Attentive transformer architecture for tabular data)
   - **PyTorch with Lightning Flash** (Rapid deep learning prototyping)
4. **Framework Comparative Evaluation**:
   - Evaluated across three primary qualitative dimensions: **Simplicity**, **Transparency**, and **Efficacy**.
5. **Evaluation Metrics**:
   - Metrics specified by the professor: **RMSE** (Root Mean Squared Error), **R²** (Coefficient of Determination), and **MAE** (Mean Absolute Error).
   - Complementary classification metrics: **Precision**, **Recall**, **F1-Score**, **ROC-AUC**, and **PR-AUC**.
6. **Academic Deliverables**:
   - Structured project report (Introduction, Methodology, EDA, Results, and Conclusion).
   - Presentation slides and viva voce preparation material.
   - Clean, version-controlled GitHub repository.

---

## ⚠️ Important Academic Clarification: Problem Formulation & Metrics

> **Critical Notice for Evaluation & Viva Voce**:
> 
> In traditional financial machine learning, fraud detection is formulated as a **Binary Classification** problem (`isFraud = 0` for legitimate, `isFraud = 1` for fraud) due to extreme class imbalance.
> 
> However, the professor's assignment specifications explicitly mandate reporting regression metrics: **RMSE**, **R²**, and **MAE**.
> 
> **How this project adheres to the requirements:**
> - We strictly maintain the professor's requirement: Models output continuous risk scores / posterior probabilities $\hat{y} \in [0, 1]$, and we compute the requested regression metrics (**RMSE**, **R²**, and **MAE**) against true binary labels $y \in \{0, 1\}$.
> - Alongside regression metrics, we also report standard fraud classification metrics (**Precision**, **Recall**, **F1-Score**, and **AUPRC**) to provide a complete, academically rigorous, and industry-standard comparison.
> - This dual-evaluation approach ensures strict compliance with grading rubrics while demonstrating deep conceptual understanding during viva questioning.

---

## 📁 Project Directory Structure

```text
mlbd/
├── data/
│   ├── raw/                           # Original, untouched dataset (new_data.csv)
│   │   └── README.md
│   └── processed/                     # Cleaned, feature-engineered datasets
│       └── README.md
├── notebooks/                         # Exploratory data analysis & prototyping
├── reports/                           # Generated reports and inspection logs
│   ├── figures/                       # Saved charts and plots
│   ├── dataset_inspection_summary.md  # Detailed Phase 1 data inspection report
│   └── dataset_inspection_summary.json# Machine-readable inspection metadata
├── scripts/                           # Standalone execution scripts
│   ├── download_dataset.py            # Automated dataset downloader
│   └── run_inspection.py              # Dataset inspection runner
├── src/                               # Modular Python source code
│   ├── __init__.py
│   ├── data/                          # Data acquisition and inspection modules
│   │   ├── __init__.py
│   │   ├── download.py
│   │   └── inspect_data.py
│   └── utils/                         # Shared utility functions
│       └── __init__.py
├── .gitignore                         # Comprehensive ignore rules
├── requirements.txt                   # Project dependencies
└── README.md                          # Project documentation
```

---

## 🚀 Setup and Installation

### 1. Environment Preparation
Ensure Python 3.10 or higher is installed. Create and activate a virtual environment:

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
Install the required packages for initial setup and exploratory data analysis:

```bash
pip install -r requirements.txt
```

### 3. Download the Prescribed Dataset
If the dataset is not yet downloaded, run:

```bash
python scripts/download_dataset.py
```

### 4. Run Dataset Inspection
To perform full schema validation, null checks, duplicate checks, and class distribution analysis:

```bash
python scripts/run_inspection.py
```

Inspection reports will be generated automatically in `reports/dataset_inspection_summary.md`.

---

## 📊 Phase 1 Deliverables Summary

- [x] Workspace verification: Confirmed isolated environment separate from external projects.
- [x] Clean modular project structure implemented.
- [x] Automated dataset download script with Google Drive session handling.
- [x] Raw dataset acquisition (`new_data.csv`).
- [x] Comprehensive inspection module for dimensions, missingness, duplicates, and class balance.
- [x] Inspection reports saved in `reports/`.
- [x] Regression vs. classification rubric discrepancy documented for professor review and viva defense.
- [x] Initial `requirements.txt` configured without unnecessary bloat.

---

## 🔮 Roadmap / Next Phases

- **Phase 2**: Exploratory Data Analysis (EDA), distribution visualization, and feature engineering.
- **Phase 3**: Implementation of the 5 frameworks (XGBoost, Keras, fastai, TabNet, Lightning Flash).
- **Phase 4**: Comparative framework analysis (Simplicity, Transparency, Efficacy) and metric computation (RMSE, R², MAE + Classification metrics).
- **Phase 5**: Final report generation, slide presentation, viva guide, and GitHub repository finalization.
