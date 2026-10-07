# 📊 Customer Churn Prediction

A Machine Learning project that predicts whether a telecom customer is likely to churn.

## 🎯 Objective

Build a customer churn prediction system using historical telecom customer data and Machine Learning classification algorithms.

## 🧠 Models Used

- Logistic Regression
- Random Forest
- XGBoost

## 📊 Model Performance

| Model | Accuracy |
|---|---:|
| Logistic Regression | 80.31% |
| XGBoost | 79.39% |
| Random Forest | 78.32% |

**Best Model:** Logistic Regression

## 🔍 Important Metrics

For the churn class:

- Precision: 65%
- Recall: 57%
- F1-score: 61%

## 🛠️ Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Streamlit
- Joblib

## 📁 Project Structure

```text
Customer_Churn_Prediction/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── notebooks/
│   └── customer_churn_analysis.ipynb
│
├── models/
│   ├── logistic_regression_model.pkl
│   └── preprocessor.pkl
│
├── src/
│   └── predict.py
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md

