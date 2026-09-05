"""
Stage 3: Fraud-spike detector.

This is the layer that makes this an "AI Risk Manager" system rather than just
a transaction classifier: instead of only scoring individual transactions, it
watches the RATE of flagged fraud over time and raises an alert when that rate
spikes well above its recent baseline — the pattern a real risk team would
actually want surfaced (e.g. "fraud rate just jumped 4x in the last hour"),
not a flood of individual transaction alerts.

Pipeline:
1. Load data, train Random Forest at the cost-optimal threshold from stage 2
   (0.095) — this is the flagged-fraud signal the spike layer consumes.
2. Bucket transactions into fixed time windows using the dataset's `Time`
   column (seconds elapsed since the first transaction in the set).
3. For each window, compute: total transactions, model-flagged fraud count,
   flagged-fraud rate.
4. Compute a rolling baseline (mean + std of flagged-fraud rate over the
   PRIOR N windows, not including the current one — this avoids the baseline
   being contaminated by the spike it's supposed to detect).
5. Flag a window as a "spike" when its rate exceeds baseline_mean + k*std
   (a standard z-score-style anomaly rule).
6. Plot flagged-fraud rate over time with spikes highlighted, and print a
   summary table of every window flagged as a spike.

This uses the MODEL's flagged fraud, not ground-truth Class labels, because
in a real deployment you would not have ground truth at alert time — this
mirrors how the system would actually run in production.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

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

# ---- 1. Train Random Forest on the same split as before, at the chosen threshold ----
X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_s = X_train.copy()
X_test_s = X_test.copy()
X_train_s[["Amount", "Time"]] = scaler.fit_transform(X_train_s[["Amount", "Time"]])
X_test_s[["Amount", "Time"]] = scaler.transform(X_test_s[["Amount", "Time"]])

rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
rf.fit(X_train_s, y_train)

CHOSEN_THRESHOLD = 0.095  # from stage 2's cost-optimal search
test_probs = rf.predict_proba(X_test_s)[:, 1]
test_flagged = (test_probs >= CHOSEN_THRESHOLD).astype(int)

# ---- 2. Build a dataframe of test-set transactions with their real Time and flag ----
window_df = pd.DataFrame({
    "Time": X_test["Time"].values,          # original (unscaled) seconds-elapsed
    "flagged": test_flagged,
    "true_fraud": y_test.values,             # kept only for a side-by-side sanity check
})

# ---- 3. Bucket into fixed windows (1 hour = 3600 seconds) ----
WINDOW_SECONDS = 3600
window_df["window"] = (window_df["Time"] // WINDOW_SECONDS).astype(int)

agg = window_df.groupby("window").agg(
    total_txns=("flagged", "size"),
    flagged_count=("flagged", "sum"),
    true_fraud_count=("true_fraud", "sum"),
).reset_index()
agg["flagged_rate"] = agg["flagged_count"] / agg["total_txns"]

print(f"Total windows: {len(agg)} (each = {WINDOW_SECONDS}s = {WINDOW_SECONDS/3600:.1f}hr)")
print(agg.head(10))

# ---- 4. Rolling baseline: mean+std of the PRIOR N windows (not including current) ----
BASELINE_WINDOWS = 6   # look back 6 windows (6 hours) to establish "normal"
Z_THRESHOLD = 2.0       # flag if current rate > baseline_mean + 2*std

agg["baseline_mean"] = agg["flagged_rate"].shift(1).rolling(BASELINE_WINDOWS, min_periods=3).mean()
agg["baseline_std"] = agg["flagged_rate"].shift(1).rolling(BASELINE_WINDOWS, min_periods=3).std()
agg["spike_threshold"] = agg["baseline_mean"] + Z_THRESHOLD * agg["baseline_std"]
agg["is_spike"] = agg["flagged_rate"] > agg["spike_threshold"]

# Windows without enough history to build a baseline can't be evaluated — mark explicitly
agg.loc[agg["baseline_mean"].isna(), "is_spike"] = False

# ---- 5. Report flagged spike windows ----
spikes = agg[agg["is_spike"]]
print(f"\n{'='*60}")
print(f"SPIKE WINDOWS DETECTED: {len(spikes)} / {len(agg)}")
print(f"{'='*60}")
if len(spikes) > 0:
    print(spikes[["window", "total_txns", "flagged_count", "flagged_rate",
                   "baseline_mean", "spike_threshold", "true_fraud_count"]].to_string(index=False))
else:
    print("No spikes detected at current threshold. Try lowering Z_THRESHOLD or BASELINE_WINDOWS.")

# ---- 6. Plot ----
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(agg["window"], agg["flagged_rate"], label="Flagged-fraud rate", color="steelblue")
ax.plot(agg["window"], agg["spike_threshold"], label="Spike threshold (baseline + 2σ)",
        color="orange", linestyle="--")
ax.scatter(spikes["window"], spikes["flagged_rate"], color="red", zorder=5,
           label="Flagged spike", s=60)
ax.set_xlabel(f"Window ({WINDOW_SECONDS/3600:.0f}-hour buckets)")
ax.set_ylabel("Flagged-fraud rate")
ax.set_title("Fraud-Spike Detection: Flagged-Fraud Rate Over Time")
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig("fraud_spike_timeline.png", dpi=150, bbox_inches="tight")
print("\nSaved fraud_spike_timeline.png")

print("\nDone. Report back: the full output and fraud_spike_timeline.png.")
