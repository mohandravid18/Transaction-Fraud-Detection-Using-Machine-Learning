Markdown

# Transaction Fraud Detection Using Machine Learning

## Project Overview
This project implements a **credit card transaction fraud detection system** using machine learning on a highly imbalanced dataset. It detects fraudulent transactions (class imbalance: ~0.17% fraud) with a focus on maximizing fraud capture while minimizing false positives — critical for real-world deployment in banking and payment systems.

**Key Highlights:**
- Applied **Random Forest** and **XGBoost** classifiers.
- Handled extreme class imbalance using **SMOTE** (Synthetic Minority Over-sampling Technique).
- Evaluated models using **AUPRC** (Area Under Precision-Recall Curve), Precision-Recall curves, and Confusion Matrix — metrics suitable for imbalanced data.
- Performed comprehensive **EDA** on transaction amounts, time patterns, and class distribution.
- Generated visualizations and saved processed data for reproducibility.

## Dataset
- **Source**: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (Kaggle)
- **Size**: 284,807 transactions
- **Features**: 
  - `Time`: Seconds elapsed between each transaction and the first transaction
  - `Amount`: Transaction amount
  - `V1` to `V28`: Principal components obtained via PCA (anonymized features)
  - `Class`: Target label (0 = Normal, 1 = Fraud)
- **Imbalance**: Only 492 frauds (~0.172%)

## Project Structure

.
├── fraud_detection_plots/
│   ├── 01_eda_transaction_patterns.png      # EDA visualizations
│   └── 02_pr_curve_and_confusion.png        # Model evaluation plots
├── fraud_detection_data/
│   ├── creditcard_raw.csv                   # Original dataset
│   └── creditcard_processed_train.csv       # Scaled & split training data
├── fraud_detection.py                       # Main script
└── README.md                                # This file
text

## Requirements
Install the required packages using:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost

##How to Run

    Download the dataset:
        Go to: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
        Download creditcard.csv
        Place it in the same directory as the script
    Run the script:

Bash

python fraud_detection.py

    Output:
        Creates folders: fraud_detection_plots and fraud_detection_data
        Saves EDA and evaluation plots
        Saves raw and processed datasets
        Prints model performance (AUPRC scores) and business insights

##Key Findings

    Fraudulent transactions tend to have smaller amounts and occur at different times compared to normal ones.
    XGBoost typically outperforms Random Forest on AUPRC (often > 0.85).
    SMOTE effectively improves recall for the fraud class without severely hurting precision.
    Tree-based models with oversampling achieve excellent fraud detection while keeping false positives low.

##Evaluation Metrics (Focus)

    AUPRC: Primary metric due to high class imbalance (better than ROC-AUC)
    Precision-Recall Curve: Shows trade-off between catching fraud (recall) and avoiding false alerts (precision)
    Confusion Matrix: Highlights false positives (costly) and false negatives (risky)

##Business Applications

    Real-time fraud alerting in payment systems
    Risk scoring for transactions
    Reducing manual review workload
    Deployable with custom threshold tuning based on business tolerance for false positives

##Future Improvements

    Threshold optimization for desired precision/recall
    Feature importance analysis using SHAP values
    Anomaly detection alternatives (Isolation Forest, Autoencoders)
    Real-time inference pipeline
