import pandas as pd

df = pd.read_csv('Transactions_200kII.csv')

# Checks
print(len(df))
print(df.dtypes)


# Cleaning I - Handling duplicates

# duplicate count
dup_count = df.duplicated().sum()
print('number of dupes', dup_count)

# sample of duplicates
if dup_count > 0:
    dup_indices = df[df.duplicated(keep = False)].index.tolist()[:10]
    print('sample rows:', dup_indices)
    print('\n first 3 pairs: ')
    print(df[df.duplicated(keep = False)].head(6).to_string(index = False))
else:
    print('No duplicates')


# drop exact duplicates
df_dedup = df.drop_duplicates(keep = 'first').reset_index(drop = True)

print("deduplicated length", len(df_dedup))

# Cleaning II - Handling missing amount values

#convert to int and remove commas
df['amount'] = pd.to_numeric(df['amount'].astype(str).str.replace(",", ""), errors = 'coerce')

print('Nulls after converting', df['amount'].isnull().sum())

# input missing amount with median
median_value = df['amount'].median()
df.loc[df['amount'].isnull(), 'amount'] = median_value

df["amount was missing"] = df['amount'].isnull()
print('rows imported:', df['amount was missing'].sum())

# Cleaning III - Handling Extreme values if any

# Basic stats
print('amount min, median, max: ', df['amount'].min(), df['amount'].median(), df['amount'].max())

# listing top 10 amounts
print("\nTop 10 amounts (index, amount):")
print(df[['amount']].sort_values('amount', ascending=False).head(10))

df_sorted = df.sort_values("amount", ascending = False)

# Removing two unbelievably large values

to_remove = df_sorted.head(2).index

print("removing: ", to_remove)

df_clean = df.drop(to_remove)

print(df_clean.shape)

# Cleaning IV - Naming and imputing null values and final checks.

cols_to_keep = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
    "amount was missing",
    "is_outlier"]

df = df[[col for col in cols_to_keep if col in df.columns]]

print(df.shape)

print(df[df['type']=='CASH_IN'][['oldbalanceOrg','newbalanceOrig']].head(10))


# impute the null values
print('Null counts', df.isnull().sum())

for col in ['oldbalanceOrg', 'newbalanceOrig']:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)


print('New null counts', df.isnull().sum())


# Final checks
print('checks:\n')
print('Rows and cols', df.shape)

print('Data Types', df.dtypes)

print('Nulls', df.isnull().sum())

print('Duplicates', df.duplicated().sum())

print('Percentile for amount values: ', df['amount'].describe(percentiles = [0.01, 0.50, 0.95, 0.999]))

print(df['type'].value_counts())

overall_fraud_rate = df['isFraud'].mean()

print('overall fraud rate', overall_fraud_rate)

print(df['isFraud'].unique())

# Removing 1 duplicate

df[df.duplicated()]

dup_indices = df[df.duplicated()].index
print(dup_indices)

df = df.drop(index = 199998).reset_index(drop = True)

print('Duplicates', df.duplicated().sum())

# Final clean file
df.to_csv("Transaction_final.csv", index=False)