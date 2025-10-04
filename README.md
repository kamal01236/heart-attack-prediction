

# Heart Attack Prediction (Decision Tree)

## Overview
This project predicts the risk of heart attack using a Decision Tree Classifier trained on US patient data. It is designed for reproducibility, interpretability, and ease of experimentation and deployment.

---

## Deliverables & Solution Write-up

### 1. Jupyter Notebook for Model Creation/Experimentation
- **File:** `notebooks/pipeline_workflow.ipynb`
- **Purpose:**
  - Step-by-step workflow for data loading, EDA, preprocessing, model training, hyperparameter tuning, evaluation, and model saving.
  - Includes visualizations (distributions, outliers, correlations) and markdown explanations.
  - Allows experimentation and easy modification of the pipeline.

### 2. Python Files for API
- **Files:**
  - `src/app.py`: Flask API exposing endpoints for model training (`/train`), prediction (`/predict`), and decision tree visualization (`/tree`).
  - `src/train_model.py`: Standalone script for training and saving the model, can be run independently or via the API.
- **API Endpoints:**
  - `/train` (POST): Trains the model, saves it, and returns metrics.
  - `/predict` (POST): Accepts patient data (JSON), returns prediction(s).
  - `/tree` (GET): Returns or generates a visualization of the trained decision tree (image path or SVG).

### 3. Model Pickle File (Final Model)
- **File:** `models/final_decision_tree_model.pkl`
- **Purpose:**
  - Stores the trained Decision Tree model (with preprocessing pipeline) for fast loading and inference.
  - Used by both the API and notebook for predictions.

### 4. Write-up: How This Problem is Solved

#### Problem Statement
Given a dataset of US heart patients, the goal is to build a machine learning model that predicts the risk of heart attack based on patient features. The solution must be reproducible, interpretable, and easy to use for both experimentation and deployment.

#### Solution Approach
1. **Data Loading:**
	- The dataset is loaded using pandas, with careful inspection for missing values and data types.
2. **EDA:**
	- The notebook provides visual and statistical analysis: first 10 rows, 5-point summary, outlier detection, missing value analysis, correlation heatmap, and feature distributions.
3. **Preprocessing:**
	- Missing values are imputed (median for numerics, mode for categoricals).
	- Outliers are capped using the IQR method.
	- Categorical features are encoded (one-hot or label encoding as appropriate).
	- Feature engineering and selection are performed as needed.
4. **Model Training & Tuning:**
	- Data is split (70/30, stratified, random_state=13).
	- A Decision Tree Classifier is trained within a pipeline (with preprocessing).
	- Hyperparameters are tuned using GridSearchCV (cross-validation, F1 score).
	- Model performance is evaluated using F1, confusion matrix, and classification report.
5. **Model Saving:**
	- The best model is saved as a pickle file for reuse.
6. **API & Automation:**
	- The Flask API provides endpoints for retraining, prediction, and tree visualization.
	- The `/train` endpoint automates the full pipeline, making it easy to retrain with new data.
	- The `/tree` endpoint generates a visual representation of the decision tree, viewable in VS Code.
7. **Reproducibility & Best Practices:**
	- All code uses `random_state=13`.
	- The workflow is modular, well-documented, and robust to missing/malformed data.
	- The LLD (`Low_Level_Design.md`) documents all design decisions and best practices.

#### How to Use
1. **Experiment:** Use the Jupyter notebook for EDA, model building, and analysis.
2. **Train:** Use the API (`/train`) or script (`src/train_model.py`) to train and save the model.
3. **Serve:** Start the API (`src/app.py`) to enable prediction and visualization endpoints.
4. **Predict:** Use `/predict` to get risk predictions for new patients.
5. **Visualize:** Use `/tree` or open the generated image in VS Code to interpret the model.

---

## Project Structure

```
├── data/US_Heart_Patients.csv         # Dataset
├── models/final_decision_tree_model.pkl  # Trained model
├── models/tree_visualization.png      # Decision tree image (generated)
├── notebooks/01_pipeline_workflow.ipynb  # EDA, training, evaluation
├── src/train_model.py                 # Training script
├── src/app.py                         # Flask API
├── requirements.txt                   # Dependencies
├── README.md                          # This file
├── Low_Level_Design.md                # Detailed LLD
```

---

## Setup Instructions

1. **Clone the repository:**
	```bash
	git clone <your-repo-url>
	cd heart-attack-prediction
	```
2. **Create and activate a virtual environment:**
	```bash
	python -m venv venv
	venv\Scripts\activate   # On Windows
	# or
	source venv/bin/activate # On Mac/Linux
	```
3. **Install dependencies:**
	```bash
	pip install -r requirements.txt
	```

---

## Step 1: Train the Model (via API or Script)

### Option A: Train via API Endpoint
Start the API server (see below), then:
```bash
curl -X POST http://127.0.0.1:5000/train
```
This will:
- Load and preprocess the data
- Train and tune the Decision Tree
- Save the model to `models/final_decision_tree_model.pkl`
- Return training metrics (F1, confusion matrix, best hyperparameters)

### Option B: Train via Script
```bash
python src/train_model.py
```
This will perform the same steps as above and print metrics to the console.

---

## Step 2: Run the API Server

```bash
python src/app.py
```
The server will start at `http://127.0.0.1:5000/`

---

## Step 3: Use the API Endpoints

### 1. `/train` (POST)
Retrain the model (see above). Returns training metrics.

### 2. `/predict` (POST)
Predict heart attack risk for one or more patients.

**Example request1:**
```bash

Invoke-WebRequest -Uri "http://127.0.0.1:5000/predict" `
     -Method POST `
     -ContentType "application/json" `
     -Body '{
         "age": 55,
         "Gender": "Male",
         "BMI": 26.5,
         "BP Meds": 0,
         "diabetes": 0,
         "heartRate": 89,
         "tot cholesterol": 210,
         "prevalentStroke": 0,
         "Systolic BP": 130,
         "Diastolic BP": 85,
         "glucose": 90,
         "education": 2,
         "currentSmoker": 1,
         "cigsPerDay": 10,
         "prevalentHyp": 1
     }'
```
**Response:**
```json
{"predictions": [0]}
```

### 3. `/tree` (GET)
Get a visualization of the trained decision tree.

- The API will return the path to the generated image (e.g., `models/tree_visualization.png`).
- Open this file in VS Code to view the tree structure.

---

## Step 4: Visualize the Decision Tree in VS Code

After training, the API or script will generate `models/tree_visualization.png`.

1. In VS Code, open the `models/tree_visualization.png` file to view the tree.
2. You can also use the `/tree` endpoint to regenerate or fetch the latest visualization.

---

## Step 5: Test the API with Sample Input

1. Start the API server.
2. Use `curl`, Postman, or VS Code REST client to send a POST request to `/predict` with a sample patient JSON.
3. The response will indicate the predicted risk (0 = no heart attack, 1 = risk).

---

## Additional Notes

- All code uses `random_state=13` for reproducibility.
- The Jupyter notebook provides full EDA, preprocessing, and model evaluation.
- The API is robust to missing or malformed data and provides clear error messages.
- For more details, see `Low_Level_Design.md`.

---

## References
- [Pandas Documentation](https://pandas.pydata.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Seaborn Documentation](https://seaborn.pydata.org/)
- [Flask Documentation](https://flask.palletsprojects.com/)
