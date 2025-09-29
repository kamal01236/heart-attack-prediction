# src/train_model.py
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree, export_text
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

RANDOM_SEED = 13
np.random.seed(RANDOM_SEED)

# Paths
DATA_PATH = "data/US_Heart_Patients.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "final_decision_tree_model.pkl")
PNG_PATH = os.path.join(MODEL_DIR, "tree.png")
DOT_PATH = os.path.join(MODEL_DIR, "tree.dot")
TEXT_PATH = os.path.join(MODEL_DIR, "tree.txt")

os.makedirs(MODEL_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

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
print("Starting GridSearchCV...")
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
print("Best params:", grid.best_params_)
print("Best CV score (f1):", grid.best_score_)

# Evaluate
y_pred = best_model.predict(X_test)
print("F1 Score (Test):", f1_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(best_model, MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")

# ---------------------------
# Visualize the trained tree
# ---------------------------

# Extract classifier and feature names if possible
def extract_clf_and_feature_names(pipeline, input_feature_names):
    """
    Given a fitted pipeline with a ColumnTransformer called 'preprocessor' and a final
    estimator called 'classifier', return (clf, feature_names_list_or_None).
    """
    clf = None
    feature_names = None

    # get classifier
    if hasattr(pipeline, "named_steps"):
        clf = pipeline.named_steps.get("classifier", None)
        preproc = pipeline.named_steps.get("preprocessor", None)
    else:
        # pipeline is not a Pipeline object
        clf = pipeline
        preproc = None

    # Try to get feature names after preprocessing
    if preproc is not None:
        try:
            # sklearn >= 1.0: ColumnTransformer has get_feature_names_out
            feature_names = preproc.get_feature_names_out(input_feature_names)
            feature_names = [str(f) for f in feature_names]
        except Exception:
            # fallback: try manual extraction (works for our pipeline)
            try:
                # numeric names
                num_cols = numeric_features
                # categorical OneHotEncoder inside transformer pipeline
                cat_transformer = None
                transformers = preproc.transformers_
                # find the categorical transformer tuple
                for name, trans, cols in transformers:
                    if name == "cat" or trans is categorical_transformer:
                        # try to access onehot inside the pipeline
                        cat_transformer = trans
                        cat_cols = cols
                        break
                onehot = None
                if hasattr(cat_transformer, "named_steps"):
                    for step in cat_transformer.named_steps.values():
                        if step.__class__.__name__.lower().startswith("onehot"):
                            onehot = step
                            break
                else:
                    if cat_transformer.__class__.__name__.lower().startswith("onehot"):
                        onehot = cat_transformer

                if onehot is not None:
                    try:
                        cat_names = onehot.get_feature_names_out(cat_cols)
                    except Exception:
                        cat_names = onehot.get_feature_names(cat_cols)
                    feature_names = list(num_cols) + list(cat_names)
                else:
                    feature_names = list(num_cols) + list(cat_cols)
            except Exception:
                feature_names = None
    else:
        # If no preprocessor, but classifier has feature_names_in_
        try:
            feature_names = list(clf.feature_names_in_)
        except Exception:
            feature_names = None

    return clf, feature_names

clf_model, feature_names = extract_clf_and_feature_names(best_model, X.columns)

# Plot using matplotlib
try:
    print("Plotting tree to PNG:", PNG_PATH)
    plt.figure(figsize=(20, 12))
    if feature_names is not None:
        plot_tree(clf_model, feature_names=feature_names, filled=True, rounded=True, fontsize=8)
    else:
        plot_tree(clf_model, filled=True, rounded=True, fontsize=8)
    plt.savefig(PNG_PATH, bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved tree PNG to:", PNG_PATH)
except Exception as e:
    print("Failed to produce PNG with plot_tree:", e)

# Export DOT (graphviz) for better rendering if graphviz is present
try:
    print("Exporting DOT file:", DOT_PATH)
    try:
        export_graphviz(clf_model,
                        out_file=DOT_PATH,
                        feature_names=feature_names,
                        filled=True,
                        rounded=True,
                        special_characters=True)
        print("Saved DOT to:", DOT_PATH)
    except Exception as e:
        print("export_graphviz failed:", e)
except Exception as e:
    print("DOT export failed:", e)

# Save textual version
try:
    print("Exporting textual tree to:", TEXT_PATH)
    txt = export_text(clf_model, feature_names=feature_names if feature_names else None)
    with open(TEXT_PATH, "w") as f:
        f.write(txt)
    print("Saved textual tree to:", TEXT_PATH)
except Exception as e:
    print("Failed to save textual tree:", e)

print("Done.")
