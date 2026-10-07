import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "logistic_regression_model.pkl"
)

PREPROCESSOR_PATH = os.path.join(
    BASE_DIR,
    "models",
    "preprocessor.pkl"
)

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)


def predict_churn(customer_data):
    customer_data_processed = preprocessor.transform(customer_data)

    prediction = model.predict(customer_data_processed)[0]
    probability = model.predict_proba(customer_data_processed)[0][1]

    return prediction, probability