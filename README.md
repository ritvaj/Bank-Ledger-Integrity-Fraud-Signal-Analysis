<p align="center">
  <img src="assets/project_banner.png" width="700" alt="Fraud-signal banner — Transaction Behavior Analytics">
</p>
<h1 align="center">Transaction Behavior Analytics & Fraud Signal Modeling</h1>

A dynamic fraud-detection scoring system built on 200,000 PaySim transactions, using Python (pandas, NumPy) to identify behavioral risk signals, anomaly pathways, that can help us single out mules and potential frauds. 

The project evaluates how interpretable, rule-driven scoring can support fast, cost-efficient fraud detection operations before introducing machine-learning models.

---

## Why This Project?

Fintech payment systems face fast-evolving fraud patterns where attackers show common inconsistent ledger behaviors like too many transactions in a single time window (velocity spikes), or emptying balance immediately after receiving an amount (pass-through flows) rather than simple rule violations. Detecting these anomalies early, without relying entirely on costly or opaque ML systems, is a core operational challenge.

This project examines how far behavioral analytics, engineered features, and interpretable scoring can push fraud detection on their own before introducing machine learning! 
It also raises a practical question every fraud-ops team faces: *how much actual frauds can be successfully flagged with transparent, rule-based scoring before the added cost, latency, and governance burden of ML becomes justified?*

Alongside the business framing, the project serves as an end-to-end application of Python (NumPy, pandas), feature engineering, anomaly exploration, and A/B testing of Fraud score thresholds on a large-scale transaction dataset.

---

 ## End-to-End Analysis Workflow

```
┌──────────────────────────────────────────┐
│        Raw Transactions Dataset          │
└──────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│        Cleaning & Preparation            │
└──────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│           Feature Engineering            │
│   - mismatch signals                     │
│   - velocity indicators                  │
│   - balance anomalies                    │
│   - mule-behavior scoring                │
└──────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│          Exploratory Analysis            │
└──────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│ Fraud Signal Score (Weighted Rule-Based) │
└──────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│          Threshold Evaluation            │
└──────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│  Insights & High-Risk Account Detection  │
└──────────────────────────────────────────┘
```

---

## 🎯 Objectives

- Understand transaction behavior at scale and map strong patterns linked to fraud signals.  
- Engineer interpretable features (mismatch signals, velocity rules, mule behaviors, balance anomalies) and check how each feature can accurately flag transactions.  
- Build a **Fraud Signal Score** - a weighted, modular and easily interpretable calculated score using multiple features that are good indicators of fraud transactions, interpretable risk index.  
- Evaluate multiple scoring thresholds against real fraud cases using precision, recall, lift, and false-positive behavior to understand operational trade-offs.
- Identify high-risk accounts and transaction patterns for further investigation.

---

## 🛠 Tools & Skills Applied

- **Visual Studio Code:** End-to-end script development (ETL → features → EDA → model evaluation)
- **Python (pandas, NumPy):** Data cleaning, feature engineering, anomaly rule construction, and threshold evaluation
- **Feature Engineering:** Velocity rules, mismatch patterns, balance signals, mule behavior indicators
- **Data Modeling:** Transaction behavior segmentation, risk-signal construction  
- **Matplotlib & Seaborn:** Behavioral visualizations, density plots, threshold curves  
- **Fraud Analytics:** Threshold testing, precision–recall evaluation, high-risk account identification

---

## Repository Structure

```
transaction-behavior-analytics/
│
├── python/
│   ├── etl/
│   │   └── data_cleaning.py
│   │   ├── feature_engg_behavioral.py 
│   │   ├── feature_engg_mismatch.py        
        
│   ├── analysis/
│   │   ├── _threshold_evaluation.py
│   │   ├── plots.py
│   └── README.md
│
├── outputs/
│   ├── figures/
│   │   ├── 01_log_amount_distribution.png
│   │   ├── 02_fraud_vs_nonfraud_amount.png
│   │   ├── 03_origin_mismatch_by_type.png
│   │   ├── 04_Mismatch_Rate_vs_Destination_Activity_Level.png
│   │   ├── 05_Either_vs_Both_Mismatch.png
│   │   ├── 06_Mule_Score_Fraud_vs_Non_Fraud.png
│   │   ├── 07_FraudScore_fraud_vs_nonfraud.png
│   │   └── precision_recall_curve.png
│   │
│   ├── tables/
│   │   ├── summary_log_amount.csv
│   │   ├── summary_fraud_vs_nonfraud.csv
│   │   ├── summary_origin_mismatch.csv
│   │   ├── summary_destination_mismatch.csv
│   │   ├── summary_either_both_mismatch.csv
│   │   ├── summary_mulescore_distribution.csv
│   │   ├── threshold_evaluation_results.csv
│   │   └── top10_high_risk_accounts.csv  
│
└── README.md
```

---

## Visual Gallery (Exploratory Analysis)

Exploratory analysis of transaction behavior using engineered features:

1. **Log-Amount Distribution**  
2. **Fraud vs Non-Fraud Transaction Amount**  
3. **Origin Mismatch Rate by Type**  
4. **Destination Activity vs Mismatch Rate**  
5. **Either vs Both Mismatch**  
6. **Mule Score Density (Fraud vs Non-Fraud)**  
7. **Fraud Signal Score Distribution**  

Threshold model evaluation:
1. **Precision-Recall Curve** 

_All images are stored in `outputs/figures/`._

---

## Feature Engineering Overview

Fraud is often behavioral, not statistical.  
This project focuses on engineering **interpretable, rule-based signals** across four categories and testing whether they are good individual indicators of actual fraud. The important features are shown as below:

### **1. Ledger Mismatch Signals**
- `origin_mismatch` — flag for when the sender's old and new balance dont match with the amount sent (1/0)
- `dest_mismatch` — flag when the receiver's old and new balance dont match up with the amount received (1/0)  
- `either_mismatch` — flag when either origin_mismatch or dest_mismatch is True (1/0)
- `both_mismatch` — flag when both origin and dest mismatch (1/0)

### **2. Amount & Balance Anomalies**
- `log_amount` — A column with logarithmic values to amount column. This will help us to properly plot the extremely skewed amount values because log helps compress huge values and spreads out small ones, enabling us to see patterns (float)
- `is_high_amount` — Flag when amount is higher than 99th percentile and zscore threshold (1/0)
- `balance_ratio` — Ratio that tells us how large the amount is relative to origin's balance (post-tx balance / pre-tx balance) (float)  
- `insufficient_funds` — flag when balance is less than the amount in a transaction (1/0) 
- `origin_drain_by_type` — When origin ends up with near zero balance after a transaction. A drain metric for origin by the type of transaction.

### **3. Velocity Features**
- `orig_tx_count_step` — number of recent outgoing transactions from origin (int)
- `dest_tx_count_step` — number of recent incoming transactions to destination (int)
- `dest_tx_count_last3` — number of recent incoming transactions to destination in the last 3 consecutive steps (1 hour window) (int)

### **4. Mule Behavior Indicators**
- `is_pass_through` — Flag a mule account behavior where receiver ends with 0 balance after receiving money i.e. destination immediately forwards funds (1/0)
- `is_many_senders` — Flag when destination receives from many distinct senders (1/0)  
- `is_dest_high_velocity` — Flag when there is a high number of incoming transactions to destination (1/0) 
- `is_high_amount` — Flag when amount is higher than 99th percentile and zscore threshold (1/0)
- `is_new_dest` — Flag first time receiver

---

## Model Performance Visuals

The following visuals summarize how the engineered behavioral features translate into risk separation, scoring quality, and threshold performance. 
To go through all the visuals go to:

[outputs/figures](outputs/figures)


### Mismatch Rate vs Destination Activity

<p align="center">
  <img src="outputs/figures/04_Mismatch_Rate_vs_Destination_Activity_Level.png" width="650">
</p>


### Mule Score Density — Fraud vs Non-Fraud

<p align="center">
  <img src="outputs/figures/06_Mule_Score_Fraud_vs_Non_Fraud.png" width="650">
</p>


### Fraud Signal Score Distribution — Fraud vs Non-Fraud

<p align="center">
  <img src="outputs/figures/07_FraudScore_fraud_vs_nonfraud.png" width="650">
</p>

## Key Insights

### 1. Destination-side activity exposes abnormal behavior  
Mismatch rates jump when receivers get multiple inbound payments in short windows which is a classic mule consolidation pattern.

### 2. Behavioral signals clearly separate fraud from normal activity  
Mismatch rules, velocity spikes, and mule-like aggregation consistently push fraud transactions to higher risk scores even without ML.

### 3. Mule behaviour shows clear separation from normal activity
Fraud cases spread into higher mule-scores, while normal users stay tightly clustered near zero.
This shift captures behaviours like repeated inflows, pass-through patterns, and short-term bursts which means these are the signals that appear far more often in mule-like fraud than in genuine accounts. 

### 4. Fraud shows a mild upward shift, not extreme outliers
Fraud cases don’t stand out as extreme outliers. Instead, they show a subtle shift toward higher scores, meaning risky behaviour emerges through accumulated weak signals rather than a single obvious anomaly.

---

## Fraud Signal Score: 

Once each feature was tested on its individual performance, the following were selected to be cumulatively used to build a weighted, rule-based scoring system combining key anomaly features into a single interpretable risk measure:

fraud_signal_score =

2.5 * mule_score_w_high (weighted mule score calculated using the mule behavior features shown above)

2.5 * both_mismatch

2.0 * is_dest_velocity (`dest_tx_count_last3` > 2)

1.0 * is_pass_through

1.0 * is_many_senders

1.0 * dest_burst (`dest_tx_count_step` >= 3)

1.0 * is_new_dest


Why this approach?

- Fully explainable and modular  
- No black-box ML  
- Easy to tune  
- Fast enough for real-time systems  

The output is a **single behavioral risk score** per transaction. Based on the dataset we can decide on thresholds above which a transaction will be flagged as Fraudulent.

---

## Threshold Evaluation
To operationalize the Fraud Signal Score, we test fraud signal score thresholds 3, 4, 5, 6 (range 0-9) to see how well each cutoff separates fraud from normal traffic.
Each threshold is compared on precision, recall, false-positive rate, lift, and True Positive/False Positive/False Negative/True Negative.

Tested thresholds: **3, 4, 5, 6**

For each threshold, the following were calculated:

- Precision  
- Recall  
- False Positive Rate  
- Lift  
- Fraud rate among flagged transactions  
- TP / FP / FN / TN counts  

The **Precision–Recall Curve** shows that even with only baseline 0.13% fraud (precision will be low), the rule-based score ranks risky accounts earlier and provides lift over random checks. This helps prioritize reviews more efficiently before adding machine learning.

<p align="center">
  <img src="outputs/figures/precision_recall_curve.png" width="650">
</p>

The table below shows the evaluation results:

[threshold_evaluation_results.csv](outputs/tables/threshold_evaluation_results.csv)


### Top 10 high-risk accounts (based on cumulative score):
Using the final Fraud Signal Score, we identify the top 10 destination accounts with the strongest anomaly patterns (high maximum score, high total score, and dense inbound activity).

These accounts exhibit behaviors typical of money-mule aggregation or pass-through flows.
Table provided in:

[top10_high_risk_accounts.csv](outputs/tables/top10_high_risk_accounts.csv)

---

### ⚠️ Dataset Context (to set expectations)

PaySim is synthetic and extremely imbalanced. As mentioned in the dataset overview, it injects artificial fraud patterns into simulated mobile-money logs, rather than capturing real attacker behaviour.
It also cancels fraud transactions by design, which breaks real balance dynamics and making features like old/new balances unusable for genuine fraud detection. In a nutshell:
- Fraud = **0.13%**  
- Many fraud patterns are injected into the dataset and hence are semi-random  
- Merchant/receiver roles aren't fully realistic  

Even so, engineered rules still surface meaningful behavior patterns, showing how far **interpretable, low-cost** scoring can go before ML is needed.

---

## Future Work

### **1. Add a Lightweight ML Baseline**  
Compare this dynamic fraud detection model using a **Fraud Signal Score** with a simple ML model (Logistic Regression / Random Forest) to measure how much additional precision and lift ML provides beyond rule-based scoring.

### **2. Introduce Temporal & Rolling-Window Features**  
Add time-aware indicators like time since last transaction, rolling velocity counts, and burst windows that will help to better capture real-world evolving fraud behavior.

### **3. Scale to the Full 2M-Row PaySim Dataset**  
Move beyond the 200k sample and process the full dataset using efficient engines like **DuckDB** (local SQL over Parquet) or **PySpark** (distributed).  
Converting data to Parquet + performing feature engineering in DuckDB is the most practical next step before scaling further.

---

## 📚 References

- **PaySim Transaction Simulation Dataset** - Kaggle  
  https://www.kaggle.com/datasets/ealaxi/paysim1  

- **pandas Documentation** - Data manipulation and feature engineering  
  https://pandas.pydata.org/docs/

- **Matplotlib Documentation** - Plotting and visualization  
  https://matplotlib.org/stable/contents.html



## 👤 About Me

I’m **Ritvaj Madotra**, a data analyst passionate about using **Python, SQL, and business analytics** to design interpretable, impact-driven solutions.  
📌 Connect: [LinkedIn](https://www.linkedin.com/in/ritvajmadotra) | [GitHub](https://github.com/ritvaj)
