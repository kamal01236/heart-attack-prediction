# tests/test_app.py
import json
from src.app import app

def test_predict_valid():
    client = app.test_client()
    payload = {"instances":[{"Gender":"M","age":55,"education":2,"currentSmoker":1,
                           "cigsPerDay":10,"BP Meds":0,"prevalentStroke":0,
                           "prevalentHyp":1,"diabetes":0,"tot cholesterol":200,
                           "Systolic BP":130,"Diastolic BP":85,"BMI":26.0,
                           "heartRate":72,"glucose":90,"glucose_missing":0}]}
    rv = client.post("/predict", data=json.dumps(payload), content_type="application/json")
    assert rv.status_code == 200
    assert "results" in rv.get_json()

def test_predict_invalid():
    client = app.test_client()
    rv = client.post("/predict", json={"bad":"data"})
    assert rv.status_code == 400


def test_thresholds_endpoint():
    client = app.test_client()
    rv = client.get("/thresholds")
    assert rv.status_code == 200
    data = rv.get_json()
    # threshold metadata should include at least one key
    assert isinstance(data, dict)
    assert any("threshold" in k for k in data.keys())

def test_health_endpoint():
    client = app.test_client()
    rv = client.get("/health")
    assert rv.status_code == 200
    data = rv.get_json()
    assert data["status"] == "ok"
    assert "model_path" in data


def make_payload(label_like):
    """Helper: return a payload shaped like data with slight feature changes."""
    base = {"Gender":"M","age":55,"education":2,"currentSmoker":1,
            "cigsPerDay":10,"BP Meds":0,"prevalentStroke":0,
            "prevalentHyp":1,"diabetes":0,"tot cholesterol":200,
            "Systolic BP":130,"Diastolic BP":85,"BMI":26.0,
            "heartRate":72,"glucose":90,"glucose_missing":0}
    if label_like == "positive":
        base["age"] = 70   # higher risk
        base["Systolic BP"] = 180
    elif label_like == "negative":
        base["age"] = 30   # low risk
        base["Systolic BP"] = 110
    return {"instances": [base]}

def test_true_positive_case():
    client = app.test_client()
    # Positive-like input + low threshold to ensure predicted 1
    rv = client.post("/predict", json={**make_payload("positive"), "threshold": 0.1})
    pred = rv.get_json()["results"][0]["predicted_label"]
    assert pred == 1  # TP (assuming ground truth positive-like)

def test_true_negative_case():
    client = app.test_client()
    # Negative-like input + high threshold so model predicts 0
    rv = client.post("/predict", json={**make_payload("negative"), "threshold": 0.9})
    pred = rv.get_json()["results"][0]["predicted_label"]
    assert pred == 0  # TN (assuming ground truth negative-like)

def test_false_positive_case():
    client = app.test_client()
    # Negative-like input + artificially low threshold, so predicted 1 incorrectly
    rv = client.post("/predict", json={**make_payload("negative"), "threshold": 0.0})
    pred = rv.get_json()["results"][0]["predicted_label"]
    assert pred == 1  # FP

def test_false_negative_case():
    client = app.test_client()
    # Positive-like input + artificially high threshold, so predicted 0 incorrectly
    rv = client.post("/predict", json={**make_payload("positive"), "threshold": 0.99})
    pred = rv.get_json()["results"][0]["predicted_label"]
    assert pred == 0  # FN

