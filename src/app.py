# app.py
import os, joblib, json
import pandas as pd
from flask import Flask, request, jsonify

MODEL_PATH = "models/final_model.pkl"
# fallback to RF model name if present
if not os.path.exists(MODEL_PATH) and os.path.exists("models/final_randomforest_model.pkl"):
    MODEL_PATH = "models/final_randomforest_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model not found. Run training first. Expected at: " + MODEL_PATH)

bundle = joblib.load(MODEL_PATH)
if isinstance(bundle, dict) and "model" in bundle:
    model = bundle["model"]
    meta = bundle.get("threshold_selection", {})
else:
    model = bundle
    meta = {}

DEFAULT_THRESHOLD = float(meta.get("threshold_cost_min", meta.get("threshold_optimal_f1", 0.5)))

app = Flask(__name__)

def validate_instances(instances):
    if not isinstance(instances, list) or len(instances) == 0:
        return False, "Field 'instances' must be a non-empty list of feature dicts."
    # basic check: ensure dicts
    for i, inst in enumerate(instances):
        if not isinstance(inst, dict):
            return False, f"Instance at index {i} is not a JSON object."
    return True, ""

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    if data is None or "instances" not in data:
        return jsonify({"error": "Missing 'instances' in payload"}), 400
    instances = data["instances"]
    ok, msg = validate_instances(instances)
    if not ok:
        return jsonify({"error": msg}), 400
    threshold = float(data.get("threshold", DEFAULT_THRESHOLD))
    try:
        X = pd.DataFrame(instances)
        proba = model.predict_proba(X)[:, 1]
        preds = (proba >= threshold).astype(int)
        results = []
        for inst, p, pr in zip(instances, proba, preds):
            results.append({"input": inst, "probability": float(p), "predicted_label": int(pr)})
        return jsonify({"threshold_used": threshold, "results": results})
    except Exception as e:
        return jsonify({"error": "Model prediction failed", "details": str(e)}), 500

@app.route("/thresholds", methods=["GET"])
def thresholds():
    return jsonify(meta)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_path": MODEL_PATH})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
