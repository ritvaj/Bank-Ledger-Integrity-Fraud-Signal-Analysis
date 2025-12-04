import pandas as pd
import numpy as np

df = pd.read_csv('Transaction_FE_step3.csv')

print(df.shape)

# Amount features
# Amount feature1: log amount
df['log_amount'] = np.log1p(df['amount'])

print('log amount (first 5)')

print(df[['amount', 'log_amount']].head())

# Amount feature2: amount_zscore
amount_mean = df['amount'].mean()
amount_std = df['amount'].std()

df['amount_zscore'] = (df['amount'] - amount_mean)/amount_std

print('\n amount_zscore: ')
print(df[['amount', 'amount_zscore']].head())

# Amount feature3: is_high_amount
z_threshold = 2.5
df['is_high_amount'] = df['amount_zscore'] > z_threshold

print('is_high_amount counts:')
print(df['is_high_amount'].value_counts())


# Balance features
# Balance feature1: balance ratio (how large the amount is relative to the origin's balance)
df['balance_ratio'] = df['amount']/(df['oldbalanceOrg'] + 1)
print('balance ratio (first 5): ')
print(df[['amount', 'oldbalanceOrg', 'balance_ratio']].head(5))

# Balance feature2: is low balance, oldbalanceOrg < amount 0.2
df['is_low_balance'] = df['balance_ratio'] > 0.9
print('is_low_balance counts: ')
print(df['is_low_balance'].value_counts())

print(df.groupby('isFraud')['is_low_balance'].mean())


# Balance feature3: insufficient funds flag
df['insufficient_funds'] = df['amount']> df['oldbalanceOrg']
print('insufficient funds counts: ')
print(df['insufficient_funds'].value_counts())

# Balance feature4: Insuffucient funds by type (cash out and transfer because these need sufficient balance)
df['insufficient_funds_CASH_OUT'] = (df['type'] == 'CASH_OUT') & (df['amount'] > df['oldbalanceOrg'])
df['insufficient_funds_TRANSFER'] = (df['type'] == 'TRANSFER') & (df['amount'] > df['oldbalanceOrg'])

print('insufficient funds: ')
print(df[['insufficient_funds_CASH_OUT', 'insufficient_funds_TRANSFER']].sum())

# Balance Feature5: origin_balance_drain
df['origin_drain'] = (df['oldbalanceOrg'] > 0) & (df['newbalanceOrig'] == 0)

print("\nOrigin balance drain counts:")
print(df['origin_drain'].value_counts())

# Balance Feature6: origin_drain_by_type
df['origin_drain_CASH_OUT'] = (df['type'] == 'CASH_OUT') & df['origin_drain']
df['origin_drain_TRANSFER'] = (df['type'] == 'TRANSFER') & df['origin_drain']

print("\nOrigin drain by type:")
print(df[['origin_drain_CASH_OUT', 'origin_drain_TRANSFER']].sum())

# Velocity Feature: Sender

# sort by sender & time
df = df.sort_values(["nameOrig", "step"]).reset_index(drop=True)

# Velocity feature1: origin_tx_count_step
df["origin_tx_count_step"] = df.groupby(["nameOrig", "step"])["step"].transform("count")


# compute top 10 senders with highest count
top_10 = (
    df.groupby("nameOrig")["origin_tx_count_step"]
      .max()
      .sort_values(ascending=False)
      .head(10)
      .reset_index()
)

print("\nTop 10 highest origin-side velocity bursts:")
print(top_10)

# sanity check print
print(df[["nameOrig","step","origin_tx_count_step"]].head(20))

# Velocity Feature: Destination
# Sort by receiver & step
df = df.sort_values(["nameDest", "step"]).reset_index(drop=True)

# Velocity feature1: dest_tx_count_step
df["dest_tx_count_step"] = df.groupby(["nameDest", "step"])["step"].transform("count")

print("\nSample dest_tx_count_step:")
print(df[["nameDest","step","dest_tx_count_step"]].head(10))

top_10 = df.groupby(['nameDest'])['dest_tx_count_step'].max().sort_values(ascending = False).head(10).reset_index()
print(top_10)

# Destination-side step velocity shows clear anomalies: several receivers processed 3–4 inbound transactions within a single hour. 
# This behavior is consistent with mule accounts aggregating funds from multiple sources before a cash-out.


# Velocity Feature2 — dest_tx_count_last3 (last 3 STEPS)

# 1) Count transactions per (nameDest, step)
tx_per_step = (
    df.groupby(["nameDest", "step"])["step"]
      .count()
      .rename("tx_count_step")
      .reset_index()
)

# 2) Create shifted copies for step-1 and step-2
t0 = tx_per_step.copy()               # current step
t1 = tx_per_step.copy()
t2 = tx_per_step.copy()

t1["step"] = t1["step"] + 1           # previous step aligns to current
t2["step"] = t2["step"] + 2           # 2-steps-back aligns to current

# 3) Merge them to accumulate last-3-step counts
merged = t0.merge(t1, on=["nameDest", "step"], how="left", suffixes=("", "_p1"))
merged = merged.merge(t2, on=["nameDest", "step"], how="left", suffixes=("", "_p2"))

# fill missing values with zeros
merged[["tx_count_step_p1", "tx_count_step_p2"]] = merged[["tx_count_step_p1", "tx_count_step_p2"]].fillna(0)

# final last-3-steps count
merged["dest_tx_count_last3"] = (
    merged["tx_count_step"] +
    merged["tx_count_step_p1"] +
    merged["tx_count_step_p2"]
)

# 4) Merge back into main df
df = df.merge(
    merged[["nameDest", "step", "dest_tx_count_last3"]],
    on=["nameDest", "step"],
    how="left"
)

df["dest_tx_count_last3"] = df["dest_tx_count_last3"].fillna(0).astype(int)

print("\nSample dest_tx_count_last3:")
print(df[["nameDest","step","dest_tx_count_last3"]].head(10))

top10_steps = (
    df.groupby("nameDest")["dest_tx_count_last3"]
      .max()
      .sort_values(ascending=False)
      .head(10)
      .reset_index()
)

print("\nTop 10 — dest_tx_count_last3 (TRUE last 3 steps):")
print(top10_steps)


# Features to find Mule behavior

# Feature1: Is Many senders (network feature)
df["dest_unique_senders"] = (
    df.groupby("nameDest")["nameOrig"]
      .transform("nunique")
)

df["is_many_senders"] = (df["dest_unique_senders"] > 3).astype(int)

# Feature2: High velocity receiving (behavioral feature)
# We ALREADY have dest_tx_count_last3 from earlier velocity code.
df["is_dest_high_velocity"] = (df["dest_tx_count_last3"] > 2).astype(int)


# Feature3: Is Pass-through behavior (behavioral feature- receives money but ends with zero balance)
df["is_pass_through"] = (
    (df["amount"] > 0) &
    (df["newbalanceDest"] == 0)
).astype(int)


# Feature4: Is High amount (amount feature)
# We ALREADY have is_high_amount as an existing column


# Weighted Mule Score
df["mule_score_w"] = (
     3* df["is_many_senders"]
    + 2* df["is_dest_high_velocity"]
    + 1* df["is_pass_through"]
    + 1* df["is_high_amount"]
).astype(int)


print(df[["nameDest","is_many_senders","is_dest_high_velocity","is_pass_through","is_high_amount","mule_score_w"]].head(5))

# top 10 accounts exhibiting mule behavior
top_10 = df.groupby('nameDest')['mule_score_w'].max().sort_values(ascending = False).head(10).reset_index()
print(top_10)

df.to_csv('Transaction_FE_final.csv', index = False)
print('saved')