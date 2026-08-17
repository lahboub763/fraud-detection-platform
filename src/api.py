from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib


# ============================================================
# LOAD MODEL AND PREPROCESSOR
# ============================================================

model = joblib.load(
    "models/xgboost_fraud_model.pkl"
)

preprocessor = joblib.load(
    "models/preprocessor.pkl"
)


# ============================================================
# CREATE FASTAPI APP
# ============================================================

app = FastAPI(
    title="Fraud Detection API",
    description="API for detecting fraudulent transactions",
    version="1.0.0"
)


# ============================================================
# INPUT DATA
# ============================================================

class Transaction(BaseModel):

    amount: float
    country: str
    payment_method: str
    device: str
    hour: int


# ============================================================
# HOME ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Fraud Detection API is running"
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(transaction: Transaction):

    data = pd.DataFrame([
        {
            "amount": transaction.amount,
            "country": transaction.country,
            "payment_method": transaction.payment_method,
            "device": transaction.device,
            "hour": transaction.hour
        }
    ])

    processed_data = preprocessor.transform(
        data
    )

    probability = model.predict_proba(
        processed_data
    )[0][1]

    prediction = int(
        probability >= 0.50
    )

    if prediction == 1:
        result = "Fraud"
    else:
        result = "Not Fraud"

    return {
        "prediction": result,
        "fraud_probability": round(
            float(probability),
            4
        )
    }