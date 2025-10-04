# src/train_model.py
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix, 
    make_scorer, precision_recall_curve, average_precision_score
)

RANDOM_SEED = 13
np.random.seed(RANDOM_SEED)

# Paths
DATA_PATH = "data/US_Heart_Patients.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "final_randomforest_model.pkl")
PR_CURVE_PATH = os.path.join(MODEL_DIR, "precision_recall_curve.png")
os.makedirs(MODEL_DIR, exist_ok=True)

def main_api():
    # Load dataset
    df = pd.read_csv(DATA_PATH)
    print("Shape:", df.shape)
    print("Missing values:\n", df.isnull().sum())

    # Add missing indicator for glucose
    if "glucose" in df.columns:
        df["glucose_missing"] = df["glucose"].isnull().astype(int)

    # Winsorization
    winsor_cols = ["tot cholesterol", "Systolic BP", "Diastolic BP", "BMI", "glucose"]
    for col in winsor_cols:
        if col in df.columns:
            low = df[col].quantile(0.01)
            high = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=low, upper=high)

    # Split features/target
    target_col = "Heart-Att"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    if y.dtype == "object" or y.dtype.name == "category":
        y = LabelEncoder().fit_transform(y.astype(str))

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    print("Numeric features:", numeric_features)
    print("Categorical features:", categorical_features)

    # Preprocessing
    numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
    )

    # Pipeline with RandomForest
    clf = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=RANDOM_SEED, class_weight="balanced"))
    ])

    # Hyperparameter tuning
    param_grid = {
        "classifier__n_estimators": [200],
        "classifier__max_depth": [5, 10, None],
        "classifier__min_samples_split": [5, 10],
        "classifier__min_samples_leaf": [2, 5]
    }

    grid = GridSearchCV(
        clf,
        param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED),
        scoring=make_scorer(f1_score, average="binary", pos_label=1),
        n_jobs=-1,
        verbose=1
    )
    print("Starting GridSearchCV...")
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    print("Best params:", grid.best_params_)
    print("Best CV score (f1):", grid.best_score_)

    # --- Default 0.5 threshold evaluation ---
    y_pred = best_model.predict(X_test)
    print("F1 Score (Test @0.5):", f1_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # --- Probability outputs for threshold tuning ---
    y_proba = best_model.predict_proba(X_test)[:, 1]

    # Precision-Recall Curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    avg_prec = average_precision_score(y_test, y_proba)

    plt.figure(figsize=(8,6))
    plt.plot(recall, precision, marker='.', label=f'RandomForest (AP={avg_prec:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(PR_CURVE_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Precision-Recall curve saved to {PR_CURVE_PATH}")

    # --- Evaluate at custom thresholds ---
    for thresh in [0.5, 0.4, 0.3]:
        y_thresh = (y_proba >= thresh).astype(int)
        f1 = f1_score(y_test, y_thresh)
        cm = confusion_matrix(y_test, y_thresh)
        report = classification_report(y_test, y_thresh, digits=3)
        print(f"\n--- Threshold = {thresh} ---")
        print("F1:", f1)
        print("Confusion Matrix:\n", cm)
        print("Report:\n", report)

    # Save model
    joblib.dump(best_model, MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

    return best_model


if __name__ == "__main__":
    main_api()
