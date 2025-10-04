
from flask import Flask, request, jsonify, send_file
import pandas as pd
import joblib
import os
import traceback
from train_model import DATA_PATH, MODEL_PATH, PNG_PATH, DOT_PATH, TEXT_PATH, RANDOM_SEED

app = Flask(__name__)

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

def validate_input(data):
    # Basic validation: check required fields, types, and value ranges
    required = [
        "age", "Gender", "BMI", "BP Meds", "diabetes", "heartRate", "tot cholesterol",
        "prevalentStroke", "Systolic BP", "Diastolic BP", "glucose", "education",
        "currentSmoker", "cigsPerDay", "prevalentHyp"
    ]
    errors = []
    if isinstance(data, dict):
        data = [data]
    for idx, row in enumerate(data):
        for field in required:
            if field not in row:
                errors.append(f"Missing field '{field}' in record {idx}")
            elif row[field] is None:
                errors.append(f"Null value for '{field}' in record {idx}")
    return errors

@app.route("/train", methods=["POST"])
def train():
    try:
        import src.train_model as train_mod
        # Re-run the training script (reloads data, retrains, saves model and tree)
        metrics = train_mod.main_api()
        return jsonify({"status": "success", **metrics})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e), "trace": traceback.format_exc()}), 500

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    errors = validate_input(data)
    if errors:
        return jsonify({"status": "error", "errors": errors}), 400
    model = load_model()
    if model is None:
        return jsonify({"status": "error", "message": "Model not trained. Please call /train first."}), 400
    df = pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame([data])
    # Add glucose_missing column if model expects it
    if "glucose_missing" in getattr(model, "feature_names_in_", []) or "glucose_missing" in df.columns or True:
        # Always add glucose_missing to match training logic
        df["glucose_missing"] = df["glucose"].isnull().astype(int) if "glucose" in df.columns else 1
    try:
        preds = model.predict(df)
        return jsonify({"status": "success", "predictions": preds.tolist()})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e), "trace": traceback.format_exc()}), 500

@app.route("/tree", methods=["GET"])
def tree():
    if os.path.exists(PNG_PATH):
        return send_file(PNG_PATH, mimetype="image/png")
    else:
        return jsonify({"status": "error", "message": "Tree visualization not found. Please train the model first."}), 404

if __name__ == "__main__":
    app.run(debug=True, port=5000)
