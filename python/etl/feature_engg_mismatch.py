import pandas as pd
import numpy as np

df = pd.read_csv('Transaction_final.csv')

# Feature: Origin

# origin delta
df['orig_delta'] = df['oldbalanceOrg'] - df['newbalanceOrig']

# function to compute expected_delta since CASH_IN = amount added
def expected_delta(row):
    if row['type'] == 'CASH_IN':
        return -row['amount']
    else:
        return row['amount']

# apply the function
df['expected_delta'] = df.apply(expected_delta, axis = 1)

# Find absolute difference
df["delta_diff_directional"] = (df["orig_delta"] - df["expected_delta"]).abs()

# Flag mismatches beyond tolerance (1.0 allowed accounting for rounding off)
tolerance = 1.0
df['orig_delta_mismatch_dir'] = df['delta_diff_directional'] > tolerance

# outputs
print('\n first 10 rows: ')
print(df[["amount","oldbalanceOrg","newbalanceOrig","orig_delta","delta_diff_directional",
          "orig_delta_mismatch_dir"]].head(10).to_string(index=True))

print('\n orig_delta describe: ')
print(df['orig_delta'].describe())

print('\n Mismatch count: ', df['orig_delta_mismatch_dir'].sum())
print('\n Mismatch rate: ', df['orig_delta_mismatch_dir'].mean())

df['dest_delta'] = df['oldbalanceDest'] - df['newbalanceDest']

# Feature: Destination

# expected results, destination receives money
df['expected_dest_delta'] = -df['amount']

#difference
df['dest_delta_diff'] = (df['dest_delta'] - df['expected_dest_delta']).abs()

# mismatch flag with same tolerance = 1.0
tolerance = 1.0
df['dest_delta_mismatch'] = df['dest_delta_diff'] > tolerance

# diagnostics
print("\n- FIRST 10 DESTINATION DELTA ROWS - ")
print(df[[
    "amount",
    "type",
    "oldbalanceDest",
    "newbalanceDest",
    "dest_delta",
    "expected_dest_delta",
    "dest_delta_diff",
    "dest_delta_mismatch"
]].head(10).to_string(index=True))

print('\n Destination mismatch count: ', df['dest_delta_mismatch'].sum())
print('\n Destination mismatch rate: ', df['dest_delta_mismatch'].mean())

print('mismatch rate by type: ')
print(df.groupby('type')['dest_delta_mismatch'].mean().sort_values(ascending = False))

print('mismatch rate by isFraud: ')
print(df.groupby('isFraud')['dest_delta_mismatch'].mean())

# Feature: Either and both

#renaming for clarity
df['origin_mismatch'] = df['orig_delta_mismatch_dir']
df['dest_mismatch'] = df['dest_delta_mismatch']

# Logical OR condition
df['either_mismatch'] = df['origin_mismatch'] | df['dest_mismatch']

# Logical AND condition
df['both_mismatch'] = df['origin_mismatch'] & df['dest_mismatch']

# Diagnostics
print('mismatch counts: ')
print('origin mismatches', df['origin_mismatch'].sum())
print('dest mismatches', df['dest_mismatch'].sum())
print('either mismatches', df['either_mismatch'].sum())
print('both mismatch', df['both_mismatch'].sum())

print('mismatch rates by isFraud: ')
print(df.groupby('isFraud')[['origin_mismatch', 'dest_mismatch', 'either_mismatch', 'both_mismatch']].mean())

df.to_csv('Transaction_FE_step3.csv', index = False)
print('saved')