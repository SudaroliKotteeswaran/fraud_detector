"""
Stage 2b: XGBoost hyperparameter tuning.

The default-params XGBoost in stage 2 lost to Random Forest on cost, despite
having a better PR-AUC. This script checks whether that's fixable with real
tuning, rather than assuming default XGBoost is XGBoost's ceiling.

Two things tested:
1. A small grid search over max_depth / learning_rate / subsample / min_child_weight
2. Whether dropping scale_pos_weight (and letting threshold tuning alone handle
   the imbalance) gives better-calibrated probabilities

This uses RandomizedSearchCV with PR-AUC as the scoring metric (not accuracy,
which is meaningless on a 0.17% imbalance) and 3-fold CV to avoid overfitting
the search to one train/test split.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_curve, precision_score, recall_score, f1_score,
    confusion_matrix, auc, make_scorer, average_precision_score
)
from xgboost import XGBClassifier

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

X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train = X_train.copy()
X_test = X_test.copy()
X_train[["Amount", "Time"]] = scaler.fit_transform(X_train[["Amount", "Time"]])
X_test[["Amount", "Time"]] = scaler.transform(X_test[["Amount", "Time"]])

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# ---- Randomized search, scored on PR-AUC (average_precision) ----
param_dist = {
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "n_estimators": [200, 300, 500],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5, 10],
    "scale_pos_weight": [1, scale_pos_weight / 4, scale_pos_weight / 2, scale_pos_weight],
}

xgb_base = XGBClassifier(eval_metric="aucpr", random_state=42)

search = RandomizedSearchCV(
    xgb_base, param_distributions=param_dist,
    n_iter=25, scoring="average_precision",
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    random_state=42, n_jobs=-1, verbose=1
)

print("Running randomized search (25 candidates x 3-fold CV = 75 fits)...")
print("This will take a few minutes on CPU.\n")
search.fit(X_train, y_train)

print(f"\nBest CV PR-AUC: {search.best_score_:.4f}")
print(f"Best params: {search.best_params_}")

best_xgb = search.best_estimator_
tuned_probs = best_xgb.predict_proba(X_test)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_test, tuned_probs)
pr_auc = auc(recall, precision)
print(f"\nTuned XGBoost test-set PR-AUC: {pr_auc:.4f}")
print("(compare to stage 2: RF=0.8623, default XGBoost=0.8763)")

# ---- Cost-optimal threshold, same cost assumption as stage 2 ----
COST_FP = 2.0
COST_FN = 50.0

def find_best_threshold(y_true, probs, cost_fp, cost_fn):
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    best_cost = np.inf
    best_threshold = 0.5
    best_stats = None
    for t in thresholds:
        preds = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        cost = fp * cost_fp + fn * cost_fn
        if cost < best_cost:
            best_cost = cost
            best_threshold = t
            best_stats = (tn, fp, fn, tp)
    return best_threshold, best_cost, best_stats

t, cost, (tn, fp, fn, tp) = find_best_threshold(y_test, tuned_probs, COST_FP, COST_FN)
preds = (tuned_probs >= t).astype(int)
print(f"\n{'='*60}")
print(f"Tuned XGBoost — cost-optimal threshold: {t:.4f}")
print(f"{'='*60}")
print(f"  Precision: {precision_score(y_test, preds):.4f}")
print(f"  Recall:    {recall_score(y_test, preds):.4f}")
print(f"  F1:        {f1_score(y_test, preds):.4f}")
print(f"  Confusion: TN={tn:,} FP={fp} FN={fn} TP={tp}")
print(f"  Total cost at this threshold: {cost:,.1f}")
print(f"  (compare to stage 2: RF=608.0, default XGBoost=668.0)")

print("\nDone. Report back the full output.")
