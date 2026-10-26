# Comprehensive Viva Voce Preparation Guide
## Subject: Machine Learning with Business Data (MLBD)
### Project: Online Payment Fraud Detection using Machine Learning

---

## 🎓 SECTION 1: Project Overview & Academic Problem Formulation

### Q1: Can you summarize this project in two minutes?
**Answer**:
"Our project focuses on detecting fraudulent transactions in digital mobile financial services using the prescribed PaySim dataset (~6.36 million transactions, ~471 MB). Fraud accounts for less than 0.1% of transactions, presenting an extreme class imbalance challenge. We engineered critical domain features—specifically balance discrepancies where originating accounts are completely emptied—and evaluated five diverse modeling frameworks: **XGBoost**, **Keras Neural Networks**, **PyTorch with fastai**, **PyTorch with TabNet**, and **PyTorch with Lightning Flash**. We compared these frameworks across **Simplicity**, **Transparency**, and **Efficacy**, recording both the professor's specified regression metrics (**RMSE**, **R²**, **MAE**) and industry-standard classification metrics (**PR-AUC**, **ROC-AUC**, and **F1-Score**)."

---

### Q2: Why did the professor specify regression metrics (RMSE, R², MAE) for a fraud detection problem, and how did you resolve this discrepancy?
**Answer**:
"In academic and industrial practice, fraud detection is inherently a **binary classification problem** because the target variable $y \in \{0, 1\}$ represents discrete states (0 = legitimate, 1 = fraudulent).
However, machine learning classifiers generate continuous posterior probability estimates $\hat{p} \in [0, 1]$ representing fraud risk score.
To strictly adhere to the professor's assignment requirements without altering them:
1. We treated the predicted risk probabilities $\hat{p}$ as continuous outputs and evaluated them directly against binary labels to calculate **RMSE**, **R²**, and **MAE**.
2. Simultaneously, we evaluated standard classification metrics: **Precision**, **Recall**, **F1-Score**, **ROC-AUC**, and **PR-AUC (AUPRC)**.
This dual-evaluation strategy demonstrates thoroughness, 100% rubric compliance, and conceptual clarity during defense."

---

### Q3: What is the significance of RMSE, R², and MAE when applied to probability predictions?
**Answer**:
- **RMSE (Root Mean Squared Error)**: Penalizes confident misclassifications heavily. If the model assigns $\hat{p} = 0.90$ to a legitimate transaction, $(1 - 0.9)^2 = 0.81$ heavily inflates RMSE.
- **MAE (Mean Absolute Error)**: Measures average absolute calibration error $|\hat{p} - y|$.
- **R² (Coefficient of Determination)**: Measures the proportion of variance in the binary indicator explained by the model's continuous risk probabilities. In severely imbalanced datasets, standard baseline variance is small ($\bar{y} \approx 0.001$), so high $R^2$ indicates high discriminative power.

---

## 📊 SECTION 2: Exploratory Data Analysis & Domain Feature Engineering

### Q4: What is the PaySim dataset and what are its key columns?
**Answer**:
"PaySim is a financial transaction simulator developed by researchers based on aggregated mobile money data from an African financial service provider.
Key columns include:
- `step`: Time unit in hours (1 step = 1 hour, spanning 30 days simulation).
- `type`: Transaction modality (`PAYMENT`, `TRANSFER`, `CASH_OUT`, `DEBIT`, `CASH_IN`).
- `amount`: Transacted currency value.
- `nameOrig` & `nameDest`: Account IDs of sender and recipient.
- `oldbalanceOrg` & `newbalanceOrig`: Sender balances before and after transaction.
- `oldbalanceDest` & `newbalanceDest`: Recipient balances before and after transaction.
- `isFraud`: Binary ground truth target.
- `isFlaggedFraud`: Simulation rule flagging transactions over 200,000 currency units."

---

### Q5: What was the single most important domain finding in your EDA?
**Answer**:
"Fraud occurs **exclusively** in two transaction types: **`TRANSFER`** and **`CASH_OUT`**.
In `PAYMENT`, `CASH_IN`, and `DEBIT`, the fraud count is strictly zero.
Perpetrators operate in a two-stage modus operandi:
1. First, initiate an unauthorized **`TRANSFER`** from the victim's account to an accomplice account.
2. Second, immediately execute a **`CASH_OUT`** to withdraw liquid physical cash before security flags can freeze the account.
Filtering our training to these relevant types drastically increases computational signal-to-noise ratio."

---

### Q6: What domain features did you engineer and why?
**Answer**:
1. **`errorBalanceOrig = newbalanceOrig + amount - oldbalanceOrg`**:
   In legitimate transactions, the post-balance should equal pre-balance minus amount (error = 0). Fraudsters often drain accounts completely regardless of transaction limits.
2. **`orig_emptied = (newbalanceOrig == 0) & (oldbalanceOrg > 0)`**:
   Binary indicator capturing the signature behavior where an account's entire life savings or balance is zeroed out in a single transaction.
3. **`errorBalanceDest = oldbalanceDest + amount - newbalanceDest`**:
   Checks whether the recipient's balance legitimately expanded by the transacted amount.
4. **`hour_of_day = step % 24`**:
   Captures the 24-hour diurnal cycle; fraudulent activities frequently occur in off-peak early morning hours when automated human surveillance is lowest.

---

### Q7: Why is Accuracy a misleading metric in fraud detection?
**Answer**:
"Because of extreme class imbalance (99.91% legitimate vs 0.085% fraud). A naive 'dummy' classifier that labels **every single transaction as legitimate** achieves a stunning **99.91% accuracy** while catching exactly **zero fraud cases (Recall = 0.0%)**, leading to catastrophic business losses. Therefore, **PR-AUC (Area Under Precision-Recall Curve)**, **Recall**, and **F1-Score** are the appropriate metrics."

---

### Q8: What is the difference between ROC-AUC and PR-AUC, and which is better here?
**Answer**:
- **ROC-AUC**: Plots True Positive Rate (Recall) vs False Positive Rate ($FP / (FP + TN)$). Because the number of true negatives ($TN$) is in the millions, a large surge in False Positives ($FP$) results in an imperceptible change in FPR. Thus, ROC-AUC looks deceptively optimistic (e.g. 0.999).
- **PR-AUC (Average Precision)**: Plots Precision ($TP / (TP + FP)$) against Recall ($TP / (TP + FN)$). It does NOT include $TN$ and is solely focused on the positive minority class. Any increase in false alarms immediately penalizes Precision. Thus, **PR-AUC is the gold standard for fraud evaluation**."

---

## 🤖 SECTION 3: The Five Modeling Frameworks Compared

### Q9: Explain Framework 1: XGBoost.
**Answer**:
- **Core Technology**: Extreme Gradient Boosted decision trees.
- **Why it fits tabular fraud data**: Decision trees perform non-linear partitioning, natively handle missing values, and are invariant to monotonic feature scaling.
- **Handling Imbalance**: We use the `scale_pos_weight = N_negative / N_positive` hyperparameter, which scales gradients for the minority class during tree building.
- **Evaluation**: Highest PR-AUC, extremely fast training (~1-3 seconds), and complete transparency via split importance and SHAP values.

---

### Q10: Explain Framework 2: Keras Deep Learning.
**Answer**:
- **Core Technology**: Deep Multilayer Perceptron (MLP) built on Keras with PyTorch backend.
- **Architecture**: Sequential dense layers (`128 -> 64 -> 32 -> 1`) regularized with **BatchNormalization** and **Dropout** (0.3, 0.2, 0.1).
- **Handling Imbalance**: Weighted Binary Cross-Entropy loss using balanced class weights computed from inverse class frequencies.
- **Evaluation**: High expressiveness for non-linear interactions, but requires feature scaling (StandardScaler) and has a higher computational footprint than decision trees.

---

### Q11: Explain Framework 3: PyTorch with fastai.
**Answer**:
- **Core Technology**: Fastai-style Tabular Architecture built with `LinBnDrop` modular blocks and Cosine Annealing learning rate scheduling (One-Cycle policy).
- **Key Advantage**: Fastai revolutionized deep learning on tabular data by demonstrating that embedding categorical variables alongside continuous normalization and warm-up/cosine annealing learning rate schedules enables neural nets to compete with gradient boosting.
- **Evaluation**: Rapid convergence in fewer epochs, lower validation loss volatility, but moderate black-box transparency.

---

### Q12: Explain Framework 4: PyTorch with TabNet.
**Answer**:
- **Core Technology**: Tabular attention transformer architecture developed by Google Cloud AI (Arik & Pfister).
- **Mechanism**: Operates in sequential decision steps ($N_{steps} = 3$). At each step, a **Sparsemax Attention Mask** selects which features the network should focus on, passing only attended features to the decision layer.
- **Handling Imbalance**: Ghost Batch Normalization (`virtual_batch_size = 128`) and class-weighted BCE.
- **Evaluation**: Outstanding transparency—it explicitly outputs attention masks showing *which feature triggered the fraud alert* at each step.

---

### Q13: Explain Framework 5: PyTorch with Lightning Flash.
**Answer**:
- **Core Technology**: Modular task-oriented abstraction using PyTorch Lightning.
- **Mechanism**: Separates research code (the model definition, forward pass, and loss calculation in `LightningModule`) from engineering boilerplate (training loop, device placement, precision, validation tracking handled by `pl.Trainer`).
- **Evaluation**: Highest engineering scalability and code organization, clean decoupling of model and training logic.

---

## ⚖️ SECTION 4: Qualitative Comparison (Simplicity, Transparency, Efficacy)

### Q14: How do the 5 frameworks rank on SIMPLICITY?
**Answer**:
1. **XGBoost (5/5)**: Standard scikit-learn API (`fit`, `predict_proba`), zero neural network tensor/batch setup, minimal lines of code.
2. **Keras (4.5/5)**: Declarative layer stacking, intuitive `model.compile()` and `model.fit()`.
3. **PyTorch with Lightning (4/5)**: Eliminates manual loop engineering; boilerplate is isolated into clean hooks.
4. **PyTorch with fastai (4/5)**: High-level tabular abstraction, automated scheduler.
5. **PyTorch TabNet (3/5)**: More complex hyperparameter space ($N_d, N_a, \gamma, \lambda_{sparse}$, virtual batch size).

---

### Q15: How do the 5 frameworks rank on TRANSPARENCY?
**Answer**:
1. **PyTorch TabNet (5/5)**: Attentive sequential masks show exactly which features contributed to decision at step 1, 2, and 3.
2. **XGBoost (4.5/5)**: Exact feature importance gain, decision paths, and SHAP tree explainer compatibility.
3. **PyTorch with Lightning (3/5)**: Structured logging of gradient norms, metrics, and weights.
4. **Keras (2.5/5)**: Black-box hidden representations; requires Integrated Gradients or Layer Activation visualization.
5. **PyTorch with fastai (2.5/5)**: Black-box neural representations; embedding weights can be inspected.

---

### Q16: How do the 5 frameworks rank on EFFICACY?
**Answer**:
1. **XGBoost (5/5)**: Superior tabular PR-AUC and F1-score with sub-second inference and robust resistance to unnormalized scale.
2. **PyTorch TabNet (4.5/5)**: Matches gradient boosting closely on tabular pattern recognition.
3. **PyTorch Lightning (4/5)**: High recall and stable convergence with class weighting.
4. **Keras (4/5)**: Strong probabilistic risk scores, lowest MAE.
5. **PyTorch with fastai (4/5)**: Fast convergence and strong ranking capability.

---

## 💼 SECTION 5: Practical Business Impact & Deployment

### Q17: In a real banking payment gateway, which framework would you deploy and why?
**Answer**:
"For real-time operational transaction screening (processing 5,000+ transactions/sec with < 15ms SLA):
- **Primary Recommendation: XGBoost**. It offers sub-millisecond C++ inference latency, requires no GPU infrastructure, does not require real-time feature normalization, and can be serialized to ONNX or Treelite for instant deployment.
- **Secondary / Auditing Recommendation: TabNet**. For high-value enterprise investigations, TabNet's attention masks provide compliance auditors and regulators with explainable justification for why a transaction was halted."
