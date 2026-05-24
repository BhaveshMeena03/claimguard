# Some basic tests for the API. Run with: pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# a normal looking claim I use for testing
sample_claim = {
    "months_as_customer": 200,
    "age": 38,
    "policy_annual_premium": 1250.0,
    "policy_deductable": 1000,
    "umbrella_limit": 0,
    "number_of_vehicles_involved": 1,
    "bodily_injuries": 0,
    "witnesses": 1,
    "total_claim_amount": 50000,
    "incident_hour_of_the_day": 12,
    "insured_sex": "MALE",
    "insured_relationship": "husband",
    "incident_type": "Single Vehicle Collision",
    "collision_type": "Rear Collision",
    "incident_severity": "Minor Damage",
    "authorities_contacted": "Police",
    "property_damage": "NO",
    "police_report_available": "YES",
}


def test_home():
    res = client.get("/")
    assert res.status_code == 200


def test_features_has_options():
    res = client.get("/features")
    assert res.status_code == 200
    data = res.json()
    assert "incident_severity" in data["categorical"]


def test_predict_returns_probability():
    res = client.post("/predict", json=sample_claim)
    assert res.status_code == 200
    data = res.json()
    # probability should be between 0 and 1
    assert 0 <= data["fraud_probability"] <= 1
    assert data["risk_level"] in ["Low", "Medium", "High"]
