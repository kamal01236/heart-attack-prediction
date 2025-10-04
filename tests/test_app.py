# tests/test_app.py
import json
from heart_attack_prediction.app import app

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
