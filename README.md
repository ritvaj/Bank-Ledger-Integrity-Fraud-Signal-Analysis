<p align="center">
  <img src="assets/project_banner.png" width="700" alt="Fraud-signal banner — Transaction Behavior Analytics">
</p>
<h1 align="center">Transaction Behavior Analytics & Fraud Signal Modeling</h1>

A Python-based fraud analytics project analyzing **200,000 simulated financial transactions** to engineer behavioral risk features, surface anomaly patterns, and evaluate threshold-based fraud detection strategies.  
Instead of relying solely on ML models, this project explores a **cost-efficient, interpretable rule-based fraud scoring system** inspired by real fintech risk operations.

---

## 💡 Why This Project?

Most fraud-detection research focuses on machine learning models.  
But real-world fintech teams rely heavily on **behavioral rules, dynamic thresholds, and interpretable signals** — because they are fast, cheap, transparent, and easy to update when fraudsters adapt.

This project frames the problem like a business analyst at a fintech company:

> **“How far can we push fraud detection using pure analytics, feature engineering, and interpretable scoring — before using ML?”**

It also serves as an end-to-end application of Python, NumPy, pandas, feature engineering, and exploratory visual analytics on a large transaction dataset.

---

## 🎯 Objectives

- Understand transaction behavior at scale and map patterns linked to fraud signals.  
- Engineer interpretable features (mismatch signals, velocity rules, mule behaviors, balance anomalies).  
- Build a **Fraud Signal Score** — a weighted, interpretable risk index.  
- Evaluate multiple thresholds using precision, recall, lift, and false-positive behavior.  
- Identify high-risk accounts and transaction patterns for further investigation.

---

## 📁 Repository Structure

```
transaction-behavior-analytics/
│
├── python/
│   ├── etl/
│   │   └── data_cleaning.py
│   ├── analysis/
│   │   ├── feature_engineering_part1.py
│   │   ├── feature_engineering_part2.py
│   │   ├── plots_eda.py
│   │   ├── fraud_signal_model.py
│   │   └── ab_testing.py
│   └── README.md
│
├── outputs/
│   ├── figures/
│   │   ├── 01_log_amount_distribution.png
│   │   ├── 02_fraud_vs_nonfraud_amount.png
│   │   ├── 03_origin_mismatch_by_type.png
│   │   ├── 04_mismatch_rate_vs_destination_activity_level.png
│   │   ├── 05_either_vs_both_mismatch.png
│   │   ├── 06_mule_score_fraud_vs_nonfraud.png
│   │   ├── 07_fraudscore_fraud_vs_nonfraud.png
│   │   └── precision_recall_curve.png
│   │
│   ├── tables/
│   │   ├── summary_log_amount.csv
│   │   ├── summary_fraud_vs_nonfraud.csv
│   │   ├── summary_origin_mismatch.csv
│   │   ├── summary_destination_mismatch.csv
│   │   ├── summary_either_both_mismatch.csv
│   │   ├── summary_mulescore_distribution.csv
│   │   ├── threshold_metrics.csv
│   │   └── top10_high_risk_accounts.csv
│
├── assets/
│   └── (optional banner or project images)
│
└── README.md
```


---

## 📊 Visual Gallery (EDA)

Exploratory analysis of transaction behavior using engineered features:

1. **Log-Amount Distribution**  
2. **Fraud vs Non-Fraud Transaction Amount**  
3. **Origin Mismatch Rate by Type**  
4. **Destination Activity vs Mismatch Rate**  
5. **Either vs Both Mismatch**  
6. **Mule Score Density (Fraud vs Non-Fraud)**  
7. **Fraud Signal Score Distribution**  

A Precision–Recall Curve is shown later under model evaluation.

_All images are stored in `outputs/figures/`._

---

## 🧠 Feature Engineering Overview

Fraud is often behavioral — not statistical.  
This project focuses on engineering **interpretable, rule-based signals** across four categories:

### **1. Ledger Mismatch Signals**
- `origin_mismatch`  
- `dest_mismatch`  
- `either_mismatch`  
- `both_mismatch`  

### **2. Amount & Balance Anomalies**
- `log_amount`  
- `is_high_amount`  
- `balance_ratio`  
- `insufficient_funds`  
- `origin_balance_drain`  

### **3. Velocity Features**
- Origin velocity counts  
- Destination velocity counts  
- `dest_tx_count_step`  
- Destination bursts  

### **4. Mule Behavior Indicators**
- `is_pass_through`  
- `is_many_senders`  
- High-velocity destinations  
- First-time receiver indicator  

---

## 🔍 Fraud Signal Score (Interpretable Risk Index)

A weighted sum of anomaly indicators:

fraud_signal_score =
2.5 * mule_score_high

2.5 * both_mismatch

2.0 * is_dest_velocity

1.0 * is_pass_through

1.0 * is_many_senders

1.0 * dest_burst

1.0 * is_new_dest



Why this approach?

- Fully explainable  
- No black-box ML  
- Easy to tune  
- Fast enough for real-time systems  

The output is a **single behavioral risk score** per transaction.

---

## 📈 Threshold Evaluation & A/B Testing

Tested thresholds: **3, 4, 5, 6**

For each threshold:

- Precision  
- Recall  
- False Positive Rate  
- Lift  
- Fraud rate among flagged transactions  
- TP / FP / FN / TN counts  

A **Precision–Recall Curve** visualizes the trade-off under class imbalance.

Tables are provided in:

outputs/tables/threshold_metrics.csv
outputs/tables/ab_test_results.csv


## 📊 Model Performance Visuals

The following visuals summarize how the engineered behavioral features translate into risk separation, scoring quality, and threshold performance.

---

### 🔄 Destination Activity vs Mismatch Rate

<p align="center">
  <img src="outputs/figures/04_mismatch_rate_vs_destination_activity_level.png" width="650">
</p>

**Insight:**  
Destination mismatch rates increase sharply as receiver activity rises.  
Even single-inbound transactions show moderate anomaly levels, but when a destination collects **2–4 inbound payments within the same hour**, mismatch rates spike dramatically.

This pattern aligns with **mule account behavior**, where funds are aggregated rapidly from multiple unrelated sources.  
Although higher-activity buckets have smaller sample sizes, the overall upward trend is clear:  
**abnormal destination-side behavior is one of the strongest early fraud signals in the dataset.**

---

### 🕵️ Mule Score Density — Fraud vs Non-Fraud

<p align="center">
  <img src="outputs/figures/06_mule_score_fraud_vs_nonfraud.png" width="650">
</p>

**Insight:**  
Fraudulent transactions consistently show **higher mule-scores**, while legitimate users cluster tightly near **zero**.  
The fraud density curve exhibits a **clear right-shift**, reflecting behaviors such as multiple inbound sends, rapid aggregation, and short-lived receiver accounts.  
High mule scores are extremely rare among normal users, making this a **highly reliable behavioral risk indicator**.

---

### 🔐 Fraud Signal Score Distribution — Fraud vs Non-Fraud

<p align="center">
  <img src="outputs/figures/07_fraudscore_fraud_vs_nonfraud.png" width="650">
</p>

**Insight:**  
Fraud transactions cluster between **1–3**, while non-fraud behavior is concentrated around **0–1**.  
Despite some overlap (expected in real systems), fraud shows a pronounced **right-shift** and heavier mid-score tail.  
This validates that the combined rule-based signals capture **meaningful anomaly structure** from behavioral patterns.

---

### 🎯 Precision–Recall Curve (Threshold Evaluation)

<p align="center">
  <img src="outputs/figures/precision_recall_curve.png" width="650">
</p>

**Insight:**  
With fraud occurring only **0.13%** of the time, baseline precision is extremely low — yet the Fraud Signal Score shows **clear ranking power**, producing a meaningful curve instead of noise.  
While absolute precision is modest (normal for synthetic imbalance), the model demonstrates **strong relative ordering**, enabling better queueing and investigation prioritization in real fraud operations.




Top 10 high-risk accounts (based on cumulative score):


---

## 🔑 Key Insights

- Mismatch + velocity features were the strongest behavioral discriminators.  
- Mule-behavior features separated fraud and non-fraud distributions cleanly.  
- First-time receivers showed disproportionately high anomaly rates.  
- Rule-based scoring delivered interpretable trade-offs suitable for real fintech risk teams.  
- Precision was limited by dataset imbalance, but behavioral clustering remained strong.

---

## 🚀 Future Enhancements

- Add a lightweight ML model to compare performance with rule-based scoring.  
- Incorporate temporal drift analysis and rolling-window velocity features.  
- Expose a real-time scoring API using FastAPI.  
- Automate threshold tuning with Bayesian optimization.  
- Scale analysis to the full 2M-row PaySim dataset.

---

## 👤 About Me

I’m **Ritvaj Madotra**, a data analyst passionate about using **Python, SQL, and business analytics** to design interpretable, impact-driven solutions.  
📌 Connect: **LinkedIn | GitHub**
