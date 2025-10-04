# Heart Attack Prediction Pipeline

This repository contains a small, reproducible pipeline for training, evaluating and serving a RandomForest model that predicts heart-attack risk from patient features. It includes preprocessing utilities, model training with threshold selection policies, simple visualizations (PR/ROC curves and a pruned decision tree), and a Flask API for inference.

## High-level architecture

- `src/` — All source code for training and serving. Key modules:
	- `src/config.py` — central configuration (paths, random seed, thresholds, costs).
	- `src/data_preprocessing.py` — deterministic, reusable preprocessing helpers (missing indicators, winsorization).
	- `src/train_model.py` — training pipeline: preprocessing, RandomForest grid-search, evaluation, threshold selection, artifact saving.
	- `src/model_utils.py` — load/save model bundles and threshold helper functions used by both training and the API.
	- `src/app.py` — Flask app exposing prediction and metadata endpoints.

- `notebooks/` — interactive notebooks used for EDA and experimentation.
- `data/` — dataset (CSV) and any processed data (typically gitignored).
- `models/` — saved model bundles and artifacts (plots, feature importances, threshold selection files). Models are typically gitignored in production.
- `tests/` — unit tests for the API and utilities.
- `docs/` — design docs (threshold selection rationale, reports).

## Design decisions

- Model bundle format: training saves a dictionary `{"model": pipeline, "threshold_selection": {...}}` using joblib. This keeps estimator and metadata together.
- Threshold policies: the pipeline computes three useful thresholds and exposes them in `models/threshold_selection.json`:
	- `threshold_optimal_f1`: threshold that maximizes F1 on validation/test set.
	- `threshold_for_recall_target`: lowest threshold achieving a target recall (configurable, default 0.80); among those thresholds we pick the one with the highest precision.
	- `threshold_cost_min`: threshold minimizing expected operational cost = FP*cost_fp + FN*cost_fn (configurable costs in `src/config.py`).
- API default behavior: the Flask app picks the preferred threshold in `model_utils.choose_threshold` (by default `threshold_cost_min` then `threshold_optimal_f1`) and allows callers to override the threshold per-request.

## Files & outputs produced by training

- Model: `models/final_model.pkl` (joblib dict with model + threshold metadata)
- Plots: `models/pr_curve.png`, `models/roc_curve.png`
- Threshold metrics: `models/threshold_metrics.csv`, `models/threshold_selection.json`, `models/threshold_selection.txt`
- Feature importances: `models/feature_importances.csv`
- (Optional) Tree visualizations: `models/tree_pruned.png`, `models/tree.svg` / `models/tree.dot`

## Final Deliverables

This repo contains the following deliverables :

1. Jupyter Notebooks for EDA / experimentation
   - `notebooks/02_pipeline_workflow.ipynb` — interactive EDA + pipeline exploration. (Also see `src/train_model.py` for the final training run.) 

2. Python files for API
   - `src/app.py` — Flask inference API (POST `/predict`, GET `/thresholds`, GET `/health`). See tests in `tests/test_app.py`. 

3. Model pickle (Final Model)
   - Expected location after training: `models/final_model.pkl`.
   - If `models/final_model.pkl` is not present in the repo, run training to generate it:
     ```
     python -m src.train_model
     ```
     This will create `models/final_model.pkl` and additional artifacts. See `src/train_model.py` for details. :contentReference[oaicite:9]{index=9}

4. Model artifacts created by training (saved to `models/`)
   - `pr_curve.png`, `roc_curve.png`, `threshold_metrics.csv`, `threshold_selection.json`, `threshold_selection.txt`, `feature_importances.csv`, `tree_pruned.png`, optionally `tree.svg`. These are saved by `src/train_model.py`. :contentReference[oaicite:10]{index=10}

5. Small write-up describing solution & model summary
   - `MODEL_SUMMARY.md` (added to repo) — short summary including best model hyperparameters, F1/precision/recall on test, chosen threshold, and a short recommendation. (See `MODEL_SUMMARY.md` file.)

## Running locally (recommended development setup)

1. Create virtual environment and install dependencies (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Ensure the dataset is placed at `data/US_Heart_Patients.csv` (the repository already contains this file in `data/`).

3. Train the model and produce artifacts:

```powershell
# From the repository root
python -m src.train_model
# or
python src/train_model.py
```

Training will run a GridSearchCV over a RandomForest pipeline, save the best model and threshold metadata to `models/`, and write PR/ROC images and threshold CSV/JSON/TXT.

4. Run the API server (development):

```powershell
python -m src.app
# The app listens on 0.0.0.0:5000 by default
```

5. Run tests:

```powershell
pytest -q
```

## API

Endpoints (simple and lightweight):

- POST /predict — run inference on one or more instances
	- Request JSON schema:
		- `instances`: list of feature dictionaries (each dict is a single patient's features)
		- optional `threshold`: numerical threshold in [0,1] to override the default/metadata selected threshold
	- Response JSON:
		- `threshold_used`: threshold applied
		- `results`: list with one entry per input: `{ input: <original input>, probability: <float>, predicted_label: <0|1> }`

Example request (curl / PowerShell body):

```json
{
	"instances": [
		{ "Gender": "M", "age": 55, "tot cholesterol": 200, "Systolic BP": 130, "Diastolic BP": 85, "BMI": 26.0, "glucose": 90 }
	]
}
```

The app also exposes helper endpoints:

- GET /thresholds — returns the threshold selection metadata (contents of `models/threshold_selection.json`).
- GET /health — simple health check reporting `model_path` and `status`.

Notes:
- The app loads the saved model bundle on import. For test isolation you can mock `model_utils.load_model_bundle` or modify the app to lazy-load on first request.

## Threshold selection rationale (summary)

- F1-optimal chooses a threshold that balances precision and recall for overall best harmonic mean.
- Recall-target picks a threshold that guarantees a minimum recall (safety-first clinical setting), then selects the highest precision among those thresholds to reduce false positives.
- Cost-minimization lets you encode operational costs for FP and FN; it picks the threshold that minimizes expected cost FP*C_FP + FN*C_FN.

All thresholds and threshold-by-metric CSV are saved so you can inspect the trade-offs and select the policy appropriate to your application.

## Development notes & next steps

- CI: add a GitHub Actions workflow that (1) sets up Python, (2) installs requirements, (3) runs pytest, (4) optionally runs linting.
- Packaging: if you plan to publish or reuse this project as a library, add setup.cfg/pyproject.toml and typing info.
- Performance: RandomForest pipeline uses GridSearchCV with n_jobs=-1; adjust parallelism for CI or smaller machines.
- Model size: model artifacts are stored in `models/` and are ignored by default in `.gitignore`. Keep large artifacts out of the repo in production and use an artifact store.

## Where to look in the code

- `src/train_model.py` — full training flow with preprocessing, GridSearchCV, threshold computation and artifact saving.
- `src/model_utils.py` — safe load/save and helpers to select thresholds for the API and other tools.
- `src/data_preprocessing.py` — small deterministic preprocessing helpers used by training and inference.

## Contact / Contributing

If you want help extending the API (add batching, authentication, or a FastAPI migration), setting up CI or porting to a microservice platform, tell me which piece you'd like done next and I can implement it.

---
Generated: updated README to better document architecture, usage and decision rationale.

