import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Create folders
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Generate sample data
np.random.seed(42)
n = 10000

amount = np.random.exponential(1500, n)
hour = np.random.randint(0, 24, n)
distance = np.random.exponential(10, n)
transactions = np.random.poisson(3, n)

# 0 = Debit, 1 = Credit
transaction_type = np.random.randint(0, 2, n)

# Generate sample fraud probability
fraud_probability = (
    0.02
    + 0.000015 * amount
    + 0.025 * ((hour >= 0) & (hour <= 5))
    + 0.012 * (distance > 25)
    + 0.015 * (transactions > 6)
    + 0.02 * (transaction_type == 1)
)

fraud_probability = np.clip(fraud_probability, 0, 0.8)
fraud = np.random.binomial(1, fraud_probability)

# Create dataset
df = pd.DataFrame({
    "amount": amount,
    "hour": hour,
    "distance": distance,
    "transactions": transactions,
    "transaction_type": transaction_type,
    "fraud": fraud
})

# Save dataset
df.to_csv("data/transactions.csv", index=False)

# Prepare training data
X = df.drop("fraud", axis=1)
y = df["fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Save model
joblib.dump(model, "models/fraud_model.pkl")

print("\nModel saved successfully!")
print("Location: models/fraud_model.pkl")

print("\nModel trained successfully!")
print("Debit and credit transactions are ready.")




