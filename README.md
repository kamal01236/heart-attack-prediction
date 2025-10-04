# Heart Attack Prediction Pipeline

## Overview
This project trains a RandomForest model to predict heart attack risk and provides tools to evaluate and serve predictions.
It includes EDA, preprocessing, DecisionTree baseline, RandomForest training, threshold selection (F1, recall target, and cost minimization), and a Flask API.

## Files
- `notebook/pipeline_workflow.ipynb` - interactive notebook with EDA, DecisionTree baseline and RandomForest training + threshold tuning.
- `src/train_model.py` - training script that saves model and artifacts to `models/`.
- `src/app.py` - Flask API to serve predictions and threshold metadata.
- `tests/test_app.py` - unit tests for the API.
- `requirements.txt` - Python dependencies.
- `Dockerfile` - containerize the API.
- `docs/LLD_threshold_selection.md` - low-level design and decision rationale.

## Quickstart
1. Create environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

2. Place dataset at `data/US_Heart_Patients.csv`.

3. Train model and produce artifacts:
```bash
python src/train_model.py
```

4. Run the API:
```bash
python app.py
```

5. Run tests:
```bash
pytest -q
```

## Threshold policies
- `threshold_optimal_f1`: threshold maximizing F1 score on validation/test split.
- `threshold_for_recall_target`: lowest threshold achieving a target recall (default 0.80) and maximizing precision among those thresholds.
- `threshold_cost_min`: threshold minimizing expected operational cost (FP*cost_fp + FN*cost_fn).

By default the training script saves selection metadata to `models/threshold_selection.json` and the app uses the cost-minimizing threshold when available.

