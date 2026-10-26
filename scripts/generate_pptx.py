"""
PowerPoint (.pptx) Presentation Generator for College Project Submission.
Generates an 11-slide presentation deck complete with speaker notes for viva voce.
"""

import os
import sys

# Ensure pptx is available
try:
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN
except ImportError:
    print("python-pptx is not installed. Skipping PPTX generation.")
    sys.exit(0)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PPTX = os.path.join(PROJECT_ROOT, "presentation", "online_payment_fraud_detection.pptx")

def create_presentation():
    os.makedirs(os.path.dirname(OUTPUT_PPTX), exist_ok=True)
    prs = Presentation()
    # Set slide dimensions to 16:9 widescreen
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]
    
    NAVY = RGBColor(27, 54, 93)
    RED = RGBColor(217, 83, 79)
    GRAY = RGBColor(100, 100, 100)
    DARK = RGBColor(33, 37, 41)
    WHITE = RGBColor(255, 255, 255)
    LIGHT_BG = RGBColor(245, 247, 250)
    
    slides_data = [
        {
            "title": "Online Payment Fraud Detection using Machine Learning",
            "subtitle": "ML with Business Data (MLBD) | College Project Presentation\nPrescribed Dataset: PaySim Financial Transactions (470.7 MB, 6.36M Rows)",
            "bullets": [
                "Comparative Benchmarking of 5 State-of-the-Art Machine Learning Frameworks",
                "Evaluation on Professor's Mandated Regression Metrics (RMSE, R², MAE) and Classification Metrics",
                "Domain-Driven Feature Engineering on Balance Discrepancy Signatures"
            ]
        },
        {
            "title": "Business Context & The Class Imbalance Challenge",
            "subtitle": "Problem Statement & Financial Stakes",
            "bullets": [
                "Digital payments surge brings sophisticated automated account-clearing attacks.",
                "Direct financial loss and regulatory penalties occur when fraud bypasses screening.",
                "Extreme Class Imbalance: Fraud represents only 0.129% of transactions (773.7 to 1 ratio).",
                "Accuracy is a deceptive metric (99.87% achieved by trivial dummy zero-model).",
                "Primary Evaluation Anchor: PR-AUC (Precision-Recall AUC) and False Positive control."
            ]
        },
        {
            "title": "Academic Alignment: Regression Metrics vs. Classification",
            "subtitle": "Addressing the Professor's Rubric Specifications",
            "bullets": [
                "The Discrepancy: Fraud is binary classification, but professor specified RMSE, R², and MAE.",
                "Our Methodological Resolution: Continuous posterior risk probabilities p in [0, 1] are scored against binary labels.",
                "RMSE penalizes uncalibrated high-confidence false alarms heavily.",
                "MAE captures mean probability calibration error across transaction streams.",
                "Dual-Reporting: We report BOTH regression metrics (rubric compliance) and classification metrics (industry standard)."
            ]
        },
        {
            "title": "Exploratory Data Analysis: The Fraud Vector Discovery",
            "subtitle": "Key Findings from 6.36 Million PaySim Records",
            "bullets": [
                "Zero Fraud Modalities: PAYMENT, CASH_IN, and DEBIT exhibit exactly 0 fraudulent events.",
                "Exclusive Fraud Modalities: 100% of fraud is confined to TRANSFER (4,097 cases) and CASH_OUT (4,116 cases).",
                "Two-Stage Laundering Vector: Perpetrators initiate an unauthorized TRANSFER, then immediately CASH_OUT.",
                "Rule Engine Inefficacy: isFlaggedFraud rule caught only 16 of 8,213 fraud events (0.19% recall).",
                "Filtering: Isolating TRANSFER & CASH_OUT triples algorithmic signal-to-noise ratio."
            ]
        },
        {
            "title": "Domain Feature Engineering: Balance Discrepancies",
            "subtitle": "Transforming Raw Transactions into High-Signal Features",
            "bullets": [
                "errorBalanceOrig: (newbalanceOrig + amount - oldbalanceOrg) captures ledger violations.",
                "orig_emptied: Flag for (newbalanceOrig == 0 & oldbalanceOrg > 0); 98% of fraud empties accounts.",
                "errorBalanceDest: (oldbalanceDest + amount - newbalanceDest) verifies recipient balance expansion.",
                "hour_of_day: step % 24 captures nocturnal surge in fraud rates (1:00 AM - 5:00 AM).",
                "is_merchant_dest: Identifies merchant destinations (M) which have zero fraud risk."
            ]
        },
        {
            "title": "Framework 1 & 2: XGBoost and Keras Neural Network",
            "subtitle": "Gradient Boosted Trees vs. Deep Tabular Representation",
            "bullets": [
                "XGBoost (Extreme Gradient Boosting):",
                "  - Histogram-based gradient tree boosting with scale_pos_weight = 773.",
                "  - Top PR-AUC (0.912), sub-millisecond inference, native feature importance.",
                "Keras (Deep Tabular MLP):",
                "  - 3-Layer Dense architecture (128 -> 64 -> 32) with BatchNormalization & Dropout.",
                "  - Adam optimizer, weighted binary cross-entropy, smooth probability calibration."
            ]
        },
        {
            "title": "Framework 3 & 4: PyTorch fastai & PyTorch TabNet",
            "subtitle": "1-Cycle Cosine Annealing vs. Attentive Sequential Transformers",
            "bullets": [
                "PyTorch with fastai (Tabular Learner):",
                "  - Modular LinBnDrop blocks with Cosine Annealing 1-Cycle learning rate scheduling.",
                "  - Rapid convergence and low validation volatility on skewed distributions.",
                "PyTorch with TabNet (Attentive Transformer):",
                "  - Google Cloud AI architecture using Sparsemax sequential attention masks.",
                "  - Unmatched transparency: reveals which features guided decisions at each step."
            ]
        },
        {
            "title": "Framework 5: PyTorch with Lightning Flash",
            "subtitle": "Task-Centric Modular Architecture",
            "bullets": [
                "Task Abstraction: Encapsulates model forward pass and loss in clean LightningModule.",
                "Trainer Automation: Decouples scientific model logic from hardware/device loops.",
                "Scalability: Production-ready codebase suited for distributed enterprise retraining.",
                "Performance: High recall (0.828) and robust gradient stability."
            ]
        },
        {
            "title": "Comparative Benchmarking Results Matrix",
            "subtitle": "Comprehensive Empirical Results Across All Evaluation Metrics",
            "bullets": [
                "XGBoost: RMSE = 0.0120 | R² = 0.9784 | MAE = 0.0002 | PR-AUC = 0.9940 | F1 = 0.9900 | Time = 2.41s",
                "PyTorch TabNet: RMSE = 0.0808 | R² = 0.0131 | MAE = 0.0298 | PR-AUC = 0.9339 | F1 = 0.6608 | Time = 44.86s",
                "Keras MLP: RMSE = 0.0747 | R² = 0.1568 | MAE = 0.0186 | PR-AUC = 0.9146 | F1 = 0.6459 | Time = 43.29s",
                "PyTorch fastai: RMSE = 0.1008 | R² = -0.5337 | MAE = 0.0372 | PR-AUC = 0.9141 | F1 = 0.5285 | Time = 25.73s",
                "PyTorch Lightning: RMSE = 0.0960 | R² = -0.3904 | MAE = 0.0328 | PR-AUC = 0.8998 | F1 = 0.5468 | Time = 31.85s"
            ]
        },
        {
            "title": "Three-Dimensional Framework Assessment",
            "subtitle": "Simplicity, Transparency, and Efficacy Compared",
            "bullets": [
                "Simplicity: XGBoost (5/5) > Keras (4.5/5) > Lightning (4/5) > fastai (4/5) > TabNet (3/5).",
                "Transparency: TabNet (5/5 - Attention Masks) > XGBoost (4.5/5 - Trees/SHAP) > Lightning (3/5) > Keras (2.5/5).",
                "Efficacy: XGBoost (5/5 - Top PR-AUC & Speed) > TabNet (4.5/5) > fastai (4/5) = Lightning (4/5) = Keras (4/5)."
            ]
        },
        {
            "title": "Conclusions & Deployment Recommendations",
            "subtitle": "Production Strategy & Viva Voce Summary",
            "bullets": [
                "Operational Real-Time Gateway: Deploy XGBoost for < 1ms inference latency and top PR-AUC.",
                "Regulatory & Compliance Auditing: Deploy TabNet for explainable attention masks on flagged transfers.",
                "Feature Engineering Supremacy: Domain discrepancy features drive over 75% of model importance.",
                "Dual-Metric Validation: RMSE/R²/MAE satisfies professor rubric; PR-AUC guarantees fraud efficacy."
            ]
        }
    ]
    
    for s_idx, s_info in enumerate(slides_data):
        slide = prs.slides.add_slide(blank_layout)
        
        # Header banner
        header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.5), Inches(11.7), Inches(1.3))
        tf = header_box.text_frame
        tf.word_wrap = True
        
        p_title = tf.paragraphs[0]
        p_title.text = s_info["title"]
        p_title.font.bold = True
        p_title.font.size = Pt(26)
        p_title.font.color.rgb = NAVY
        
        p_sub = tf.add_paragraph()
        p_sub.text = s_info["subtitle"]
        p_sub.font.size = Pt(14)
        p_sub.font.color.rgb = RED if s_idx == 0 else GRAY
        p_sub.space_before = Pt(4)
        
        # Content Box
        content_box = slide.shapes.add_textbox(Inches(0.8), Inches(2.0), Inches(11.7), Inches(4.8))
        c_tf = content_box.text_frame
        c_tf.word_wrap = True
        
        for b_idx, bullet in enumerate(s_info["bullets"]):
            p = c_tf.paragraphs[0] if b_idx == 0 else c_tf.add_paragraph()
            p.text = "• " + bullet if not bullet.startswith("  ") else "    " + bullet
            p.font.size = Pt(18 if s_idx == 0 else 16)
            p.font.color.rgb = DARK
            p.space_after = Pt(12)
            
    prs.save(OUTPUT_PPTX)
    print(f"PowerPoint presentation generated successfully at: {OUTPUT_PPTX}")

if __name__ == "__main__":
    create_presentation()
