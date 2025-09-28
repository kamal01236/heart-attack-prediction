import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, classification_report, confusion_matrix

RANDOM_SEED = 13
np.random.seed(RANDOM_SEED)

# Load dataset
df = pd.read_csv("data/US_Heart_Patients.csv")

# Explicit target column
target_col = "Heart-Att"
X = df.drop(columns=[target_col])
y = df[target_col]

# Encode target if categorical
if y.dtype == "object" or y.dtype.name == "category":
    y = LabelEncoder().fit_transform(y.astype(str))

# Features
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = [c for c in X.columns if c not in numeric_features]

# Preprocessing pipelines
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

# Pipeline
clf = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(random_state=RANDOM_SEED))
])

# Hyperparameter tuning
param_grid = {
    "classifier__max_depth": [3, 5, 7, None],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4]
}
grid = GridSearchCV(clf, param_grid, cv=5, scoring="f1", n_jobs=-1)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_

# Evaluate
y_pred = best_model.predict(X_test)
print("F1 Score (Test):", f1_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(best_model, "models/final_decision_tree_model.pkl")
print("✅ Model saved to models/final_decision_tree_model.pkl")
