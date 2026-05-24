# FastAPI backend for ClaimGuard.
# It loads the model I trained in train_model.py and exposes a /predict endpoint.
# start it with:  uvicorn main:app --reload

import json
import joblib
import pandas as pd

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="ClaimGuard API")

# let the frontend (running on a different port) call this api
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# load the trained model and the feature info once when the app starts
model = joblib.load("model.pkl")
with open("feature_info.json") as f:
    feature_info = json.load(f)


# the shape of the data we expect from the form
class Claim(BaseModel):
    months_as_customer: int
    age: int
    policy_annual_premium: float
    policy_deductable: int
    umbrella_limit: int
    number_of_vehicles_involved: int
    bodily_injuries: int
    witnesses: int
    total_claim_amount: int
    incident_hour_of_the_day: int
    insured_sex: str
    insured_relationship: str
    incident_type: str
    collision_type: str
    incident_severity: str
    authorities_contacted: str
    property_damage: str
    police_report_available: str


def get_risk_level(prob):
    # just simple thresholds to make the result easier to read
    if prob >= 0.7:
        return "High"
    elif prob >= 0.4:
        return "Medium"
    else:
        return "Low"


@app.get("/")
def home():
    return {"message": "ClaimGuard API is running"}


# the frontend calls this to fill the dropdown options
@app.get("/features")
def features():
    return feature_info


@app.post("/predict")
def predict(claim: Claim):
    # the model expects a dataframe with the same columns, so build one row
    row = pd.DataFrame([claim.dict()])

    prob = model.predict_proba(row)[0][1]  # probability of fraud (class 1)
    prob = round(float(prob), 3)

    return {
        "fraud_probability": prob,
        "prediction": "Fraud" if prob >= 0.5 else "Legitimate",
        "risk_level": get_risk_level(prob),
    }
