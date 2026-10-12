# 📊 Enterprise Telecom Customer Churn Intelligence System

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-1572B6?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

> **Predict subscriber attrition risk in real time, extract demographic churn drivers, and deliver actionable retention intelligence using production Machine Learning.**

---

## 📌 Executive Summary

Customer churn is one of the most critical KPI metrics in subscription and telecommunication industries. Acquiring a new customer costs **5x to 7x more** than retaining an existing subscriber. 

This repository delivers an **end-to-end Machine Learning intelligence engine** that predicts individual subscriber churn propensity and highlights underlying risk factors (contract duration, fiber optic setup, and missing value-added services). The system includes an interactive **Streamlit Web Application** for real-time customer profiling and risk tiering.

---

## 🎯 Architecture & Data Flow

```text
[Raw Telecom Subscriber Ledger]
               │
               ▼
 [Exploratory Data Analysis (EDA)]
 (Univariate & Bivariate Correlation)
               │
               ▼
[ColumnTransformer Preprocessor Pipeline]
 ├── StandardScaler (Tenure, MonthlyCharges, TotalCharges)
 └── OneHotEncoder (Contract, InternetService, PaymentMethod)
               │
               ▼
 [Model Benchmarking & Validation]
 (Logistic Regression vs Random Forest vs XGBoost)
               │
               ▼
[Production Serialization & Serving]
 (preprocessor.pkl + logistic_regression_model.pkl)
               │
               ▼
[Interactive Streamlit SaaS Dashboard]
 (Instant Risk Tiering & Probability Deltas)
```

---

## 📊 Model Performance Benchmarking

Trained and evaluated on historical telecom subscriber data (`Telco-Customer-Churn.csv`) with an 80/20 train-test split:

| Classifier Architecture | Test Accuracy | Churn Precision | Churn Recall | Churn F1-Score | Deployment Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 **Logistic Regression (Calibrated)** | **80.31%** | **65.00%** | **57.00%** | **0.61** | **Production Deployed** |
| 🥈 **XGBoost Classifier** | 79.39% | 63.50% | 55.20% | 0.59 | Benchmark Baseline |
| 🥉 **Random Forest Classifier** | 78.32% | 62.10% | 54.80% | 0.58 | Benchmark Baseline |

---

## 🔍 Key Business Insights

1. **Contract Inelasticity:** Subscribers on **Month-to-month contracts** exhibit a $>75\%$ share of all churn events. Encouraging annual contracts is the highest-leverage retention strategy.
2. **Tenure Critical Threshold:** Churn likelihood drops precipitously once a customer remains active beyond **24 months**.
3. **Product Bundling:** Customers without **Online Security** or **Tech Support** on Fiber Optic connections demonstrate elevated cancellation rates. Bundling free security trials drastically reduces churn probability.

---

## 📁 Repository Directory Structure

```text
Customer-Churn-Prediction/
│
├── requirements.txt                   # Production package dependencies
├── README.md                          # Enterprise project documentation
├── .gitignore                         # Python, environment & cache ignore patterns
├── app.py                             # Polished Streamlit web application
│
├── models/                            # Serialized Machine Learning Artifacts
│   ├── logistic_regression_model.pkl  # Calibrated model weights
│   └── preprocessor.pkl               # Scikit-learn feature encoding pipeline
│
├── notebooks/                         # Exploratory Data Analysis & Modeling
│   └── customer_churn_analysis.ipynb  # End-to-end data pipeline workbook
│
└── src/                               # Production Source Code
    └── predict.py                     # Standalone inference & probability scoring engine
```

---

## 🚀 Quickstart & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/RedamJaswanth/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Launch the Interactive Dashboard
```bash
streamlit run app.py
```
Open your browser at **`http://localhost:8501`** to interact with the live risk evaluation engine.

---

## 👨‍💻 Author & Maintainer

* **Author:** Redam Jaswanth
* **Email:** [jaanujaswanth4@gmail.com](mailto:jaanujaswanth4@gmail.com)
* **GitHub:** [@RedamJaswanth](https://github.com/RedamJaswanth)
