FRAUD DETECTION


Razorpay AI Buildathon 2026 · Track 02

An AI-powered fraud monitoring system that doesn't just ask:

“Is this transaction suspicious?”

It also asks:

“Is something unusual happening right now?”

The system combines transaction-level fraud detection with real-time fraud-spike detection so that a risk team can focus on important attack patterns instead of manually reviewing hundreds of individual alerts.

🎯 The Problem

Traditional fraud systems often generate alerts one transaction at a time.

Imagine a merchant suddenly receives 40 suspicious transactions within an hour.

A traditional system might produce:

⚠️ Transaction suspicious
⚠️ Transaction suspicious
⚠️ Transaction suspicious
⚠️ Transaction suspicious
...

The risk team has to figure out whether these alerts are isolated incidents or part of a larger attack.

We want to answer a more useful question:

“Has the merchant's fraud rate suddenly increased compared with its normal behavior?”

That's the idea behind Fraud-Spike Detector.

💡 Our Solution

The system has two layers.

Layer 1 — Transaction Fraud Detection

A machine-learning model evaluates individual transactions and predicts whether they should be flagged as potentially fraudulent.

We compare:

Logistic Regression
Random Forest
XGBoost
Tuned XGBoost

Instead of relying on accuracy, we measure:

Precision
Recall
PR-AUC
False positives
False negatives
Estimated cost

This matters because the dataset is extremely imbalanced:

284,807 transactions → only 492 confirmed fraud cases.

An accuracy score alone would therefore be misleading.

Layer 2 — Fraud-Spike Detection

The fraud predictions are then aggregated into 1-hour windows.

For every hour, we calculate:

Total transactions
        ↓
Transactions flagged as fraud
        ↓
Fraud flag rate
        ↓
Compare with recent baseline
        ↓
Detect unusual spike

The detector compares the current hour with the previous 6 hours.

If the current fraud rate is significantly higher than the recent baseline, the system raises an alert.

🧠 How It Works
                    TRANSACTION DATA
                           │
                           ▼
                 ┌───────────────────┐
                 │  Data Preparation │
                 │ Time + Amount     │
                 │ V1 ... V28        │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │   Random Forest   │
                 │ Fraud Classifier  │
                 └─────────┬─────────┘
                           │
                           ▼
                Fraud / Not Fraud Flag
                           │
                           ▼
                 ┌───────────────────┐
                 │  1-Hour Buckets   │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ Recent 6-Hour     │
                 │ Fraud Baseline    │
                 └─────────┬─────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Current rate >      │
                │ baseline + 2×std ?  │
                └──────────┬──────────┘
                           │
                    ┌──────┴──────┐
                    ▼             ▼
                  NORMAL        SPIKE 🚨
📊 Dataset

We use the Credit Card Fraud Detection dataset from Kaggle.

It contains:

284,807 transactions
492 confirmed fraud cases
Fraud rate: approximately 0.173%
Data covering two days
V1–V28: anonymized PCA features
Time: transaction timestamp
Amount: transaction amount

The dataset is intentionally not included in this repository because of its size and Kaggle terms.

To reproduce the experiment:

Download creditcard.csv from Kaggle.
Create a data/ directory.
Place the dataset inside it.
Set the DATA_PATH environment variable.
🤖 Model Selection

We tested multiple machine-learning approaches.

Model	PR-AUC	Threshold	Precision	Recall	False Positives	False Negatives	Total Cost
Logistic Regression	—	0.5	0.061	0.918	1,386	8	3,172
Random Forest	0.862	0.095	0.750	0.888	29	11	608
XGBoost	0.876	0.5	0.874	0.847	—	—	668
Tuned XGBoost	0.880	0.065	0.674	0.888	42	11	634
Why Random Forest?

XGBoost achieved the highest PR-AUC.

However, PR-AUC isn't the only thing that matters in a fraud system.

At the selected operating point, Random Forest produced:

88.8% recall
75.0% precision
Only 29 false positives
11 false negatives
Lowest estimated total cost: 608

So we selected Random Forest for deployment because it offered the best cost/operational trade-off, rather than simply choosing the model with the highest headline metric.

💰 Cost-Aware Evaluation

False positives and false negatives don't have the same impact.

For this experiment, we use an explicit cost assumption:

False Positive = 2 cost units
False Negative = 50 cost units

The idea is simple:

False positive

A legitimate customer gets flagged.

Possible consequences:

Extra customer friction
Manual review
Support workload
False negative

A fraudulent transaction gets through.

Possible consequences:

Direct financial loss
Chargeback
Investigation cost

Important: the 2:50 ratio is an experimental assumption, not a claim about Razorpay's real fraud costs.

We state this explicitly so that the results are reproducible and honest.

🚨 Fraud-Spike Detection

After the Random Forest makes its predictions, we don't immediately create an alert for every transaction.

Instead, we look at the bigger picture.

Transactions are grouped into one-hour windows.

For each window:

Fraud rate =
Flagged transactions / Total transactions

We then calculate a baseline using the previous six windows.

Baseline = mean(previous 6 windows)
           +
           2 × standard deviation(previous 6 windows)

If the current fraud rate crosses this threshold:

🚨 FRAUD SPIKE DETECTED
Important design decision

The current window is not included in its own baseline.

Otherwise, the spike could influence the threshold that is supposed to detect it.

📈 Results

The detector identified:

5 spike windows out of 48

The strongest spike occurred at window 26.

During that window:

6 transactions were flagged
All 6 were confirmed fraud
Precision within that window: 100%
Fraud rate was approximately 6× the trailing baseline

This demonstrates the type of signal a risk team can act on:

Instead of receiving many disconnected transaction alerts, the system highlights a period where fraudulent activity is unusually concentrated.

🔍 Why We Use Model Predictions

A common mistake when building fraud demos is to use the actual fraud labels to detect spikes.

We don't do that.

At production time, the system doesn't know whether a transaction is actually fraudulent.

It only has the model's prediction.

Therefore:

Transaction
     ↓
ML prediction
     ↓
Flag / No flag
     ↓
Aggregate predictions
     ↓
Detect spike

The spike detector operates on model predictions, not ground-truth labels.

This makes the experiment closer to how a real monitoring system would operate.

🏗️ Architecture
                  creditcard.csv
                       │
                       ▼
              ┌──────────────────┐
              │ Data Preparation │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │  Random Forest   │
              │ threshold = .095 │
              └────────┬─────────┘
                       │
                       ▼
              Transaction Flags
                       │
                       ▼
              ┌──────────────────┐
              │ 1-Hour Grouping  │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ 6-Window Rolling │
              │    Baseline      │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ Rate > Mean+2σ ? │
              └────────┬─────────┘
                       │
                 ┌─────┴─────┐
                 ▼           ▼
               Normal       Alert 🚨
📁 Project Structure
razorpay-fraud-detector/
│
├── data/
│   └── creditcard.csv
│
├── stage1_baseline.py
├── stage2_threshold_xgboost.py
├── stage2b_xgboost_tuning.py
├── stage3_spike_detector.py
│
├── pr_curve.png
├── fraud_spike_timeline.png
│
├── requirements.txt
└── README.md
▶️ Running the Project

Install dependencies:

pip install -r requirements.txt

Set the dataset location.

Windows PowerShell
$env:DATA_PATH="C:\path\to\creditcard.csv"
macOS / Linux
export DATA_PATH=/path/to/creditcard.csv

Then run the experiments:

python stage1_baseline.py
python stage2_threshold_xgboost.py
python stage2b_xgboost_tuning.py
python stage3_spike_detector.py

The scripts produce the model comparisons, threshold analysis, and fraud-spike results.

🛡️ Robustness

The project also checks for common data problems before processing.

Missing dataset

Instead of throwing an unexplained Python error, the system tells the user that the dataset is missing and explains what needs to be fixed.

Incorrect schema

The expected columns are checked before processing.

Missing values

Rows containing missing values are handled with a warning instead of silently causing downstream failures.

The goal is simple:

Fail clearly, rather than fail mysteriously.

⚠️ Honest Limitations

This is a buildathon prototype, not a production fraud engine.

There are several important limitations.

1. Cost ratio is an assumption

The 2:50 false-positive/false-negative ratio is an experimental assumption.

Real merchant data could produce a different ratio and potentially change which model is optimal.

2. Short baseline

The spike detector uses only six previous hours.

The dataset covers only two days, so this is practical for the experiment but not ideal for production.

A production system would learn normal behavior from weeks or months of historical data.

3. Dataset doesn't guarantee coordinated attacks

The dataset wasn't specifically created to represent coordinated fraud attacks.

Therefore, the spike detector is evaluated on the patterns present in this dataset rather than on a dedicated attack simulation.

4. Limited explainability

The V1–V28 features are anonymized PCA components.

Therefore, the system can tell us:

“This transaction looks suspicious.”

But it cannot reliably explain:

“This transaction is suspicious because the customer's device changed.”

A production implementation should use interpretable merchant/payment features where appropriate.

🧪 What We Learned

The most important takeaway wasn't simply which ML model achieved the highest score.

It was this:

Fraud detection should optimize for operational decisions, not just model performance.

XGBoost had the best PR-AUC.

Random Forest had the better cost profile at the chosen operating point.

And the spike detector adds another layer of value by turning individual predictions into a merchant-level risk signal.

That changes the workflow from:

"Here are dozens of suspicious transactions."

to:

🚨 "Fraud activity is unusually high right now.
    Risk team should investigate this window."
🚀 Why This Fits Razorpay

A large payment platform doesn't only need to identify suspicious transactions.

It also needs to help risk teams answer:

Is fraud increasing?
When did it start?
How unusual is the activity?
Which time window deserves investigation?
Are alerts becoming noisy?

This project focuses on that second layer of intelligence.

Transaction-level detection tells us:

“This payment looks suspicious.”

Spike detection tells us:

“Something unusual may be happening to the merchant right now.”

That distinction is the core idea behind Fraud-Spike Detector.
