# Heart Attack Prediction (Decision Tree)

## Overview
This project predicts the risk of heart attack using a Decision Tree Classifier trained on patient data.

- **Dataset:** `US_Heart_Patients.csv`
- **Model:** Scikit-learn Decision Tree with preprocessing (imputation + one-hot encoding)
- **Deliverables:** Jupyter notebook, training script, Flask API, and pickle model

## Setup (Local)
```bash
git clone <your-repo-url>
cd heart-attack-prediction
python -m venv venv
source venv/bin/activate   # (or venv\Scripts\activate on Windows)
pip install -r requirements.txt
```

## Training the Model
```bash
python src/train_model.py
```

## Running the API
```bash
python src/app.py
```

Then send a POST request to http://127.0.0.1:5000/predict with JSON input.
