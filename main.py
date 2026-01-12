# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_recall_curve, auc, confusion_matrix, 
                             classification_report, average_precision_score)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import warnings
import os

warnings.filterwarnings('ignore')

# ===================================================================
# Create folders for plots and data
# ===================================================================
plots_dir = 'fraud_detection_plots'
data_dir = 'fraud_detection_data'

os.makedirs(plots_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

print(f"Plots will be saved in: ./{plots_dir}/")
print(f"Processed data will be saved in: ./{data_dir}/\n")

# ===================================================================
# 1. Load Dataset (Download from Kaggle if needed)
# ===================================================================
# Dataset URL: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# File: creditcard.csv (284,807 transactions)
df = pd.read_csv('creditcard.csv')  # Place in your working directory or adjust path

print(f"Dataset loaded: {df.shape[0]} transactions, {df.shape[1]} features")
fraud_rate = df['Class'].mean() * 100
print(f"Fraud rate: {fraud_rate:.3f}% ({df['Class'].sum()} frauds)\n")

# Save raw dataset
raw_csv = f"{data_dir}/creditcard_raw.csv"
df.to_csv(raw_csv, index=False)
print(f"Raw dataset saved: {raw_csv}")

# ===================================================================
# 2. Exploratory Data Analysis (EDA)
# ===================================================================
plt.figure(figsize=(14, 10))

# Class distribution
plt.subplot(2, 2, 1)
sns.countplot(x='Class', data=df)
plt.title('Class Distribution (0: Normal, 1: Fraud)')
plt.yscale('log')

# Transaction Amount distribution
plt.subplot(2, 2, 2)
sns.histplot(df[df['Class']==0]['Amount'], bins=50, label='Normal', alpha=0.6)
sns.histplot(df[df['Class']==1]['Amount'], bins=50, label='Fraud', color='red', alpha=0.6)
plt.title('Transaction Amount Distribution by Class')
plt.legend()
plt.xlim(0, 2000)  # Most transactions are small

# Time pattern
plt.subplot(2, 2, 3)
sns.histplot(df[df['Class']==0]['Time'] / 3600, bins=48, label='Normal', alpha=0.6)
sns.histplot(df[df['Class']==1]['Time'] / 3600, bins=48, label='Fraud', color='red', alpha=0.6)
plt.title('Transactions Over Time (Hours from Start)')
plt.xlabel('Time (hours)')
plt.legend()

# Amount vs Time (scatter sample for visibility)
plt.subplot(2, 2, 4)
sample_normal = df[df['Class']==0].sample(1000)
sample_fraud = df[df['Class']==1]
plt.scatter(sample_normal['Time']/3600, sample_normal['Amount'], alpha=0.5, label='Normal')
plt.scatter(sample_fraud['Time']/3600, sample_fraud['Amount'], color='red', label='Fraud')
plt.title('Amount vs Time (Sampled)')
plt.xlabel('Time (hours)')
plt.ylabel('Amount')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '01_eda_transaction_patterns.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 01_eda_transaction_patterns.png")

# ===================================================================
# 3. Preprocessing & Train-Test Split
# ===================================================================
X = df.drop('Class', axis=1)
y = df['Class']

# Scale Amount and Time (others are PCA-transformed)
scaler = StandardScaler()
X['Amount_scaled'] = scaler.fit_transform(X[['Amount']])
X['Time_scaled'] = scaler.fit_transform(X[['Time']])
X = X.drop(['Amount', 'Time'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Save processed train set
processed_train = X_train.copy()
processed_train['Class'] = y_train
processed_csv = f"{data_dir}/creditcard_processed_train.csv"
processed_train.to_csv(processed_csv, index=False)
print(f"Processed training data saved: {processed_csv}\n")

# ===================================================================
# 4. Models with SMOTE for Imbalance
# ===================================================================
models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    'XGBoost': xgb.XGBClassifier(scale_pos_weight=(len(y_train)-sum(y_train))/sum(y_train), random_state=42)
}

# Pipeline with SMOTE + Model
results = {}

for name, model in models.items():
    print(f"Training {name} with SMOTE...")
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('model', model)
    ])
    pipeline.fit(X_train, y_train)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)
    
    auprc = average_precision_score(y_test, y_prob)
    results[name] = {'auprc': auprc, 'y_prob': y_prob, 'y_pred': y_pred}
    
    print(f"{name} AUPRC: {auprc:.4f}")

# ===================================================================
# 5. Evaluation: Precision-Recall Curves & Confusion Matrices
# ===================================================================
plt.figure(figsize=(14, 6))

# PR Curves
plt.subplot(1, 2, 1)
for name in models:
    precision, recall, _ = precision_recall_curve(y_test, results[name]['y_prob'])
    plt.plot(recall, precision, label=f"{name} (AUPRC={results[name]['auprc']:.4f})")
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()

# Confusion Matrices (using best model - XGBoost usually wins)
best_model = max(results, key=lambda x: results[x]['auprc'])
cm = confusion_matrix(y_test, results[best_model]['y_pred'])
plt.subplot(1, 2, 2)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - {best_model}\n(FP: {cm[0,1]}, FN: {cm[1,0]})')
plt.ylabel('True')
plt.xlabel('Predicted')

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '02_pr_curve_and_confusion.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 02_pr_curve_and_confusion.png")

# ===================================================================
# Final Insights
# ===================================================================
print("\n" + "="*60)
print("FRAUD DETECTION INSIGHTS")
print("="*60)
best = max(results, key=lambda x: results[x]['auprc'])
print(f"• Highly imbalanced dataset: Fraud rate {fraud_rate:.3f}%")
print(f"• Best model: {best} with AUPRC {results[best]['auprc']:.4f}")
print(f"• SMOTE + tree-based models handle imbalance well")
print(f"• Focus on high precision at acceptable recall to minimize false alerts")
print(f"• Fraudulent transactions often smaller amounts & different time patterns")
print("• Deploy with threshold tuning for desired FP/FN trade-off")
print("="*60)

print(f"\nProject Complete!")
print(f"All plots saved in: './{plots_dir}/'")
print(f"Raw & processed datasets saved in: './{data_dir}/'")
print("\nDownload the dataset from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")