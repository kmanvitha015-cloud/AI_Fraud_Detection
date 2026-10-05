import pandas as pd
import numpy as np
import os

np.random.seed(42)

# Number of transactions
n = 2000

# Create transaction IDs
transaction_id = [f"TXN{i:05d}" for i in range(1, n + 1)]

# 0 = Debit, 1 = Credit
transaction_type = np.random.randint(0, 2, n)

# Transaction amount
amount = np.random.uniform(100, 100000, n).round(2)

# Transaction hour
hour = np.random.randint(0, 24, n)

# Distance from usual location
distance = np.random.uniform(0, 1000, n).round(2)

# Number of recent transactions
recent_transactions = np.random.randint(0, 15, n)

# Fraud rules for synthetic dataset
fraud = (
    ((amount > 70000) & (distance > 500))
    | ((hour < 5) & (amount > 30000))
    | ((recent_transactions > 10) & (amount > 50000))
).astype(int)

# Create DataFrame
df = pd.DataFrame({
    "transaction_id": transaction_id,
    "transaction_type": transaction_type,
    "amount": amount,
    "hour": hour,
    "distance": distance,
    "recent_transactions": recent_transactions,
    "fraud": fraud
})

# Create data folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# Save dataset
df.to_csv("data/transactions.csv", index=False)

print("Dataset created successfully!")
print(f"Total transactions: {len(df)}")
print(f"Debit transactions: {(df['transaction_type'] == 0).sum()}")
print(f"Credit transactions: {(df['transaction_type'] == 1).sum()}")
print(f"Normal transactions: {(df['fraud'] == 0).sum()}")
print(f"Fraud transactions: {(df['fraud'] == 1).sum()}")

print("\nFirst 5 transactions:")
print(df.head())
