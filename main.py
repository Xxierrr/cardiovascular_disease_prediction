from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib


# Create FastAPI application
app = FastAPI(
    title="Cardiovascular Disease Prediction API",
    description="Machine Learning API for cardiovascular disease prediction",
    version="1.0.0"
)


# Load trained model
model = joblib.load("models/model.joblib")


# Input data structure
class PatientData(BaseModel):
    age_years: float
    height: float
    weight: float
    ap_hi: float
    ap_lo: float
    cholesterol: int
    gluc: int
    smoke: int
    alco: int
    active: int
    gender: int


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Cardiovascular Disease Prediction API is running"
    }


# Health check
@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# Prediction endpoint
@app.post("/predict")
def predict(data: PatientData):

    # Calculate BMI
    bmi = data.weight / ((data.height / 100) ** 2)

    # Create input DataFrame
    input_data = pd.DataFrame([{
        "height": data.height,
        "weight": data.weight,
        "ap_hi": data.ap_hi,
        "ap_lo": data.ap_lo,
        "age_years": data.age_years,
        "bmi": bmi,
        "gender": data.gender,
        "cholesterol": data.cholesterol,
        "gluc": data.gluc,
        "smoke": data.smoke,
        "alco": data.alco,
        "active": data.active
    }])

    # Prediction
    prediction = model.predict(input_data)[0]

    # Probability
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        result = "Cardiovascular disease detected"
    else:
        result = "No cardiovascular disease detected"

    return {
        "prediction": int(prediction),
        "result": result,
        "probability": round(float(probability), 4),
        "bmi": round(float(bmi), 2)
    }