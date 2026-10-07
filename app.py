import streamlit as st
import pandas as pd
import sys
import os

# Add src folder to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from predict import predict_churn


# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# Title
st.title("📊 Customer Churn Prediction")
st.write("Predict whether a customer is likely to churn using Machine Learning.")


# Customer Information
st.header("Customer Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=72, value=12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

with col2:
    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )
    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )
    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )
    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )
    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )
    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )


# Contract and Billing
st.header("Contract & Billing")

col3, col4 = st.columns(2)

with col3:
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

with col4:
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=50.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=500.0
    )


# Calculate total services
service_values = [
    online_security,
    online_backup,
    device_protection,
    tech_support,
    streaming_tv,
    streaming_movies
]

total_services = sum(value == "Yes" for value in service_values)


# Prediction
if st.button("🔮 Predict Churn", type="primary"):

    customer_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "TotalServices": total_services
    }])

    prediction, probability = predict_churn(customer_data)

    st.header("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn.")
    else:
        st.success("✅ Customer is unlikely to churn.")

    st.metric(
        "Churn Probability",
        f"{probability * 100:.2f}%"
    )