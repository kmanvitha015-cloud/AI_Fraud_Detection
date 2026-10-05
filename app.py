import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="AI Fraud Detection System",
    page_icon="🛡️",
    layout="wide"
)

# Load trained model
model = joblib.load("models/fraud_model.pkl")

# Title
st.title("🛡️ AI-Powered Real-Time Fraud Detection and Alert System")

st.write(
    "Enter transaction details to detect potentially fraudulent "
    "Debit or Credit transactions."
)

st.divider()

# Transaction input
st.subheader("💳 Transaction Details")

col1, col2 = st.columns(2)

with col1:
    transaction_type = st.selectbox(
        "Transaction Type",
        ["Debit", "Credit"]
    )

    amount = st.number_input(
        "Transaction Amount (₹)",
        min_value=1.0,
        value=5000.0,
        step=100.0
    )

    hour = st.slider(
        "Transaction Hour",
        min_value=0,
        max_value=23,
        value=14
    )

with col2:
    distance = st.number_input(
        "Distance from Usual Location (km)",
        min_value=0.0,
        value=5.0,
        step=1.0
    )

    transactions = st.number_input(
        "Recent Transactions",
        min_value=0,
        value=2,
        step=1
    )

st.divider()

# Check transaction
if st.button("🔍 Check Transaction", use_container_width=True):

    # Convert transaction type
    transaction_type_value = (
        0 if transaction_type == "Debit" else 1
    )

    # Prepare transaction data
    input_data = pd.DataFrame([{
        "amount": amount,
        "hour": hour,
        "distance": distance,
        "transactions": transactions,
        "transaction_type": transaction_type_value
    }])

    # AI prediction
    prediction = model.predict(input_data)[0]

    # Fraud probability
    probability = model.predict_proba(input_data)[0][1]

    # Current time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Result text
    if prediction == 1:
        result = "Fraud"
        alert = "FRAUD ALERT"
    else:
        result = "Normal"
        alert = "Normal"

    # Display result
    st.divider()
    st.subheader("📊 Detection Result")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Transaction Type", transaction_type)

    with col2:
        st.metric("Amount", f"₹{amount:,.2f}")

    with col3:
        st.metric(
            "Fraud Probability",
            f"{probability * 100:.2f}%"
        )

    if prediction == 1:
        st.error("🚨 FRAUD ALERT!")
        st.warning(
            "This transaction has been identified as potentially fraudulent."
        )
    else:
        st.success("✅ TRANSACTION APPEARS NORMAL")
        st.info(
            "No significant fraud indicators were detected."
        )

    # Create transaction record
    transaction_record = pd.DataFrame([{
        "Date & Time": current_time,
        "Transaction Type": transaction_type,
        "Amount": amount,
        "Hour": hour,
        "Distance": distance,
        "Recent Transactions": transactions,
        "Fraud Probability": round(probability * 100, 2),
        "Result": result
    }])

    # Save transaction history
    history_file = "history.csv"

    if os.path.exists(history_file):
        history = pd.read_csv(history_file)
        history = pd.concat(
            [history, transaction_record],
            ignore_index=True
        )
    else:
        history = transaction_record

    history.to_csv(history_file, index=False)

    st.success("✅ Transaction saved to history.")

# Transaction history section
# Transaction History
st.divider()

st.subheader("📜 Transaction History")

if os.path.exists("history.csv"):

    history = pd.read_csv("history.csv")

    if len(history) > 0:

        st.write(f"Total transactions: **{len(history)}**")

        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True
        )

        # Download history
        csv_data = history.to_csv(index=False)

        st.download_button(
            label="⬇️ Download Transaction History",
            data=csv_data,
            file_name="transaction_history.csv",
            mime="text/csv"
        )

    else:
        st.info("Transaction history is empty.")

else:
    st.info("No transactions have been checked yet.")
    
# Dashboard Analytics
st.divider()

st.subheader("📊 Dashboard Analytics")

if os.path.exists("history.csv"):

    history = pd.read_csv("history.csv")

    col1, col2, col3, col4 = st.columns(4)

    total_transactions = len(history)
    total_fraud = len(history[history["Result"] == "Fraud"])
    total_normal = len(history[history["Result"] == "Normal"])
    total_amount = history["Amount"].sum()

    with col1:
        st.metric(
            "Total Transactions",
            total_transactions
        )

    with col2:
        st.metric(
            "Fraud Transactions",
            total_fraud
        )

    with col3:
        st.metric(
            "Normal Transactions",
            total_normal
        )

    with col4:
        st.metric(
            "Total Amount",
            f"₹{total_amount:,.2f}"
        )

    st.subheader("Transaction Type Distribution")

    transaction_counts = history["Transaction Type"].value_counts()

    st.bar_chart(transaction_counts)

    st.subheader("Fraud vs Normal Transactions")

    result_counts = history["Result"].value_counts()

    st.bar_chart(result_counts)

    st.subheader("Transaction Amount Analysis")

    st.line_chart(
        history["Amount"]
    )

else:

    st.info(
        "Check some transactions to generate dashboard analytics."
    )

    st.dataframe(
        history,
        use_container_width=True
    )

    # Download history
    csv_data = history.to_csv(index=False)

    st.download_button(
        label="⬇️ Download Transaction History",
        data=csv_data,
        file_name="transaction_history.csv",
        mime="text/csv"
    )
