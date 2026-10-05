import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/fraud_model.pkl")

print("Fraud detection model loaded successfully!\n")


def test_transaction(transaction_type, amount, hour, distance, transactions):
    data = pd.DataFrame([{
        "amount": amount,
        "hour": hour,
        "distance": distance,
        "transactions": transactions,
        "transaction_type": transaction_type
    }])

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    transaction_name = "Debit" if transaction_type == 0 else "Credit"

    print("Transaction Type:", transaction_name)
    print("Amount: ₹", amount)
    print("Hour:", hour)
    print("Distance:", distance)
    print("Recent Transactions:", transactions)

    if prediction == 1:
        print("🚨 FRAUD DETECTED")
    else:
        print("✅ NORMAL TRANSACTION")

    print("Fraud Probability:", round(probability * 100, 2), "%")
    print("-" * 50)


# Test 1 - Debit
print("TEST 1")
test_transaction(
    transaction_type=0,
    amount=5000,
    hour=14,
    distance=5,
    transactions=2
)

# Test 2 - Credit
print("TEST 2")
test_transaction(
    transaction_type=1,
    amount=5000,
    hour=14,
    distance=5,
    transactions=2
)

# Test 3 - Suspicious Debit
print("TEST 3")
test_transaction(
    transaction_type=0,
    amount=80000,
    hour=2,
    distance=100,
    transactions=10
)

# Test 4 - Suspicious Credit
print("TEST 4")
test_transaction(
    transaction_type=1,
    amount=90000,
    hour=3,
    distance=150,
    transactions=12
)
