"""
Stage 1: Baseline fraud detector on the Kaggle Credit Card Fraud Detection dataset.

HOW TO RUN THIS ON KAGGLE:
1. Go to kaggle.com/datasets -> search "Credit Card Fraud Detection" (by mlg-ulb) -> Add Data to a new notebook
2. Create a new Kaggle Notebook, add that dataset
3. Paste this whole script into a cell and run
4. Report back: the printed metrics block at the end

WHAT THIS DOES:
- Loads the real, anonymized transaction data (284,807 transactions, 492 fraud = 0.172%)
- Trains two baseline models: Logistic Regression and Random Forest
- Evaluates BOTH on a held-out test set (never seen during training)
- Reports precision, recall, F1, and a false-positive cost estimate
- This is intentionally simple — stage 2 will improve on whichever baseline underperforms
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

# ---- 1. Load data ----
# Reproducible on any machine: reads DATA_PATH from an environment variable
# if set, otherwise falls back to this default. To run on a different
# machine, set the env var instead of editing this file:
#   PowerShell:  $env:DATA_PATH="C:\path\to\creditcard.csv"
#   Mac/Linux:   export DATA_PATH=/path/to/creditcard.csv
DEFAULT_DATA_PATH = r"C:\Users\thous\OneDrive\Desktop\razorpay-fraud-detector\creditcard.csv"
DATA_PATH = os.environ.get("DATA_PATH", DEFAULT_DATA_PATH)

REQUIRED_COLUMNS = {"Time", "Amount", "Class"} | {f"V{i}" for i in range(1, 29)}

try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"ERROR: could not find the dataset at '{DATA_PATH}'.")
    print("Download creditcard.csv from Kaggle (mlg-ulb/creditcardfraud) and "
          "either update DEFAULT_DATA_PATH above, or set the DATA_PATH "
          "environment variable to its location.")
    sys.exit(1)
except pd.errors.EmptyDataError:
    print(f"ERROR: the file at '{DATA_PATH}' is empty or not a valid CSV.")
    sys.exit(1)

missing_cols = REQUIRED_COLUMNS - set(df.columns)
if missing_cols:
    print(f"ERROR: dataset is missing expected columns: {sorted(missing_cols)}")
    print("This script expects the standard Kaggle creditcard.csv schema "
          "(Time, V1-V28, Amount, Class). Check you downloaded the right file.")
    sys.exit(1)

n_missing_values = df.isnull().sum().sum()
if n_missing_values > 0:
    print(f"WARNING: found {n_missing_values} missing values, dropping those rows.")
    df = df.dropna()

if len(df) == 0:
    print("ERROR: no usable rows remain after cleaning. Check the input file.")
    sys.exit(1)

print(f"Total transactions: {len(df):,}")
print(f"Fraud transactions: {df['Class'].sum():,} ({df['Class'].mean()*100:.3f}%)")
print(f"Columns: {list(df.columns)}")

# ---- 2. Train/test split (stratified so both sets keep the same fraud ratio) ----
X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"\nTrain set: {len(X_train):,} ({y_train.sum()} fraud)")
print(f"Test set:  {len(X_test):,} ({y_test.sum()} fraud)  <-- never trained on")

# ---- 3. Scale Amount and Time (the only non-PCA'd columns) ----
scaler = StandardScaler()
X_train = X_train.copy()
X_test = X_test.copy()
X_train[["Amount", "Time"]] = scaler.fit_transform(X_train[["Amount", "Time"]])
X_test[["Amount", "Time"]] = scaler.transform(X_test[["Amount", "Time"]])

# ---- 4. Baseline model 1: Logistic Regression (class_weight balanced, since 0.17% positive) ----
print("\n" + "="*60)
print("MODEL 1: Logistic Regression (class_weight='balanced')")
print("="*60)
lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
lr_probs = lr.predict_proba(X_test)[:, 1]

print(classification_report(y_test, lr_preds, target_names=["Legit", "Fraud"], digits=4))
print(f"ROC-AUC: {roc_auc_score(y_test, lr_probs):.4f}")

# ---- 5. Baseline model 2: Random Forest ----
print("\n" + "="*60)
print("MODEL 2: Random Forest (class_weight='balanced')")
print("="*60)
rf = RandomForestClassifier(
    n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_probs = rf.predict_proba(X_test)[:, 1]

print(classification_report(y_test, rf_preds, target_names=["Legit", "Fraud"], digits=4))
print(f"ROC-AUC: {roc_auc_score(y_test, rf_probs):.4f}")

# ---- 6. Confusion matrices + false-positive cost framing ----
def report_cost(name, y_true, y_pred, cost_per_fp=2.0, cost_per_fn=50.0):
    """
    cost_per_fp: estimated cost of wrongly flagging a legit transaction
                 (customer friction, support ticket, lost trust) — illustrative unit
    cost_per_fn: estimated cost of missing real fraud (the actual loss + chargeback)
                 — illustrative unit, set higher since missed fraud is usually costlier
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_cost = fp * cost_per_fp + fn * cost_per_fn
    print(f"\n{name} confusion matrix:")
    print(f"  True Negatives:  {tn:,}")
    print(f"  False Positives: {fp:,}  (legit txns wrongly flagged)")
    print(f"  False Negatives: {fn:,}  (fraud txns missed)")
    print(f"  True Positives:  {tp:,}")
    print(f"  Illustrative cost (FP={cost_per_fp}, FN={cost_per_fn} units): {total_cost:,.1f}")
    return tn, fp, fn, tp

report_cost("Logistic Regression", y_test, lr_preds)
report_cost("Random Forest", y_test, rf_preds)

print("\nDone. Report back these numbers.")
