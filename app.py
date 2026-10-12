import streamlit as st
import pandas as pd
import sys
import os

# Add src folder to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from predict import predict_churn

# Page configuration
st.set_page_config(
    page_title="Customer Churn Intelligence Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished SaaS feel
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.8rem;
    }
    .metric-card {
        background: #F8FAFC;
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #E2E8F0;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        font-weight: 700;
        padding: 12px;
        background: linear-gradient(90deg, #2563EB, #4F46E5);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8644/8644487.png", width=70)
    st.title("System Controls")
    st.markdown("**Model Engine:** Logistic Regression (Calibrated)")
    st.markdown("**Baseline Accuracy:** `80.31%`")
    st.markdown("**Target Class:** Telecom Subscriber Churn")
    st.divider()
    st.markdown("💡 **Pro Tip:** Customers with Month-to-month contracts and high Fiber Optic monthly charges carry highest churn probabilities.")
    st.info("Built with Scikit-Learn, Streamlit & Pandas.")

# Main Header
st.markdown("<div class='main-title'>Telecom Customer Churn Intelligence</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Predict subscriber attrition probability in real-time using production Machine Learning.</div>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🎯 Single Customer Evaluation", "📊 Model Metrics & Intelligence"])

with tab1:
    col_demo, col_services, col_contract = st.columns([1, 1, 1], gap="medium")

    with col_demo:
        st.subheader("1. Demographics")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Account Tenure (Months)", min_value=0, max_value=72, value=12)

    with col_services:
        st.subheader("2. Subscribed Services")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    with col_contract:
        st.subheader("3. Billing & Contract")
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ])
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=130.0, value=75.0, step=0.5)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=9000.0, value=float(monthly_charges * max(1, tenure)), step=10.0)

    # Compute Total Active Value-Added Services
    service_values = [online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies]
    total_services = sum(value == "Yes" for value in service_values)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Evaluate Churn Risk"):
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
        
        st.divider()
        st.subheader("Valuation Outcome & Risk Profile")
        
        res_col1, res_col2, res_col3 = st.columns([1.5, 1, 1])

        with res_col1:
            if prediction == 1:
                st.error("⚠️ **HIGH ATTRITION RISK DETECTED**\n\nThe customer profile matches high-probability churn behaviors. Immediate retention offer recommended.")
            else:
                st.success("✅ **STABLE RETENTION PROFILE**\n\nThe subscriber demonstrates healthy engagement metrics with low propensity to cancel.")

        with res_col2:
            st.metric("Churn Propensity", f"{probability * 100:.2f}%", delta=f"{(probability - 0.5) * 100:+.1f}% vs Threshold")

        with res_col3:
            risk_tier = "CRITICAL" if probability > 0.70 else ("ELEVATED" if probability > 0.40 else "SAFE")
            st.metric("Risk Classification Tier", risk_tier)

with tab2:
    st.subheader("Model Performance Benchmarking")
    st.markdown("""
    The system was trained on historical telecom records (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) comparing **Logistic Regression, Random Forest, and XGBoost**.
    """)
    
    col_bench1, col_bench2 = st.columns(2)
    with col_bench1:
        metrics_df = pd.DataFrame({
            "Classifier Architecture": ["Logistic Regression (Current)", "XGBoost Classifier", "Random Forest Classifier"],
            "Accuracy": ["80.31%", "79.39%", "78.32%"],
            "Churn Class F1": ["0.61", "0.59", "0.58"],
            "Deployment Status": ["Production Deployed", "Benchmark Baseline", "Benchmark Baseline"]
        })
        st.dataframe(metrics_df, use_container_width=True)

    with col_bench2:
        st.markdown("**Top Churn Indicators:**")
        st.markdown("""
        1. **Contract Type:** Month-to-month contracts account for $>75\%$ of all cancellations.
        2. **Tenure Duration:** Risk drops by over $60\%$ once a subscriber passes the 24-month mark.
        3. **Internet Product:** Fiber optic subscribers without Online Security or Tech Support show elevated attrition rates.
        """)