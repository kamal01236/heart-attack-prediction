# src/train_model.py
"""
Final training script for Heart Attack Prediction.
- Random seed: 13 everywhere
- Preprocessing: glucose_missing, winsorization, imputation, one-hot encoding
- Model: RandomForest with GridSearchCV
- Threshold evaluation: F1-optimal, recall-target, cost-minimization
- Saves artifacts: models/, PR/ROC plots, threshold CSV/JSON/TXT, feature importances, pruned tree PNG, optional SVG (graphviz)
"""

import os
import json
import joblib
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score,
    roc_curve, auc
)
from sklearn.tree import plot_tree, export_graphviz

# Try to import graphviz for better tree rendering (optional)
try:
    import graphviz
    HAS_GRAPHVIZ = True
except Exception:
    HAS_GRAPHVIZ = False

warnings.filterwarnings("ignore")

# Reproducibility
RANDOM_SEED = 13
np.random.seed(RANDOM_SEED)

# Paths
DATA_PATH = "data/US_Heart_Patients.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "final_model.pkl")
PR_CURVE = os.path.join(MODEL_DIR, "pr_curve.png")
ROC_CURVE = os.path.join(MODEL_DIR, "roc_curve.png")
THRESH_CSV = os.path.join(MODEL_DIR, "threshold_metrics.csv")
THRESH_JSON = os.path.join(MODEL_DIR, "threshold_selection.json")
THRESH_TXT = os.path.join(MODEL_DIR, "threshold_selection.txt")
FEATURE_IMP_CSV = os.path.join(MODEL_DIR, "feature_importances.csv")
TREE_PNG = os.path.join(MODEL_DIR, "tree_pruned.png")
TREE_SVG = os.path.join(MODEL_DIR, "tree.svg")

# Policy params (tune for your clinical context)
COST_FP = 1    # cost assigned to a false positive
COST_FN = 10   # cost assigned to a false negative (usually higher)
TARGET_RECALL = 0.80  # business-driven minimum recall

def extract_feature_names(preprocessor, input_feature_names):
    """
    Best-effort extraction of feature names after ColumnTransformer.
    Returns list of names or None.
    """
    try:
        names = preprocessor.get_feature_names_out(input_feature_names)
        return [str(n) for n in names]
    except Exception:
        pass

    try:
        names = []
        for name, transformer, cols in preprocessor.transformers_:
            if transformer == "passthrough" or transformer is None:
                if hasattr(cols, "__iter__"):
                    names.extend(list(cols))
                else:
                    names.append(cols)
                continue
            # if transformer is a pipeline, try to extract OneHotEncoder
            if hasattr(transformer, "named_steps"):
                onehot = None
                for step in transformer.named_steps.values():
                    if step.__class__.__name__.lower().startswith("onehot"):
                        onehot = step
                        break
                if onehot is not None:
                    try:
                        ohe_names = onehot.get_feature_names_out(cols)
                    except Exception:
                        ohe_names = onehot.get_feature_names(cols)
                    names.extend(list(ohe_names))
                else:
                    if hasattr(cols, "__iter__"):
                        names.extend(list(cols))
                    else:
                        names.append(cols)
            else:
                if transformer.__class__.__name__.lower().startswith("onehot"):
                    try:
                        ohe_names = transformer.get_feature_names_out(cols)
                    except Exception:
                        ohe_names = transformer.get_feature_names(cols)
                    names.extend(list(ohe_names))
                else:
                    if hasattr(cols, "__iter__"):
                        names.extend(list(cols))
                    else:
                        names.append(cols)
        return names
    except Exception:
        return None

def safe_save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)

def evaluate_thresholds(y_true, y_proba, thresholds, cost_fp=COST_FP, cost_fn=COST_FN):
    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cost = int(fp) * cost_fp + int(fn) * cost_fn
        rows.append({
            "threshold": float(t),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
            "precision": float(precision), "recall": float(recall), "f1": float(f1), "cost": float(cost)
        })
    return pd.DataFrame(rows)

def main():
    # Load dataset
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}. Please place the CSV there.")
    df = pd.read_csv(DATA_PATH)
    print("Shape:", df.shape)
    print("Missing values:\n", df.isnull().sum())

    # Feature engineering: missing indicator for glucose
    if "glucose" in df.columns:
        df["glucose_missing"] = df["glucose"].isnull().astype(int)

    # Winsorize numeric columns to reduce extreme outlier impact
    winsor_cols = ["tot cholesterol", "Systolic BP", "Diastolic BP", "BMI", "glucose"]
    for col in winsor_cols:
        if col in df.columns:
            low = df[col].quantile(0.01)
            high = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=low, upper=high)

    # Split features/target
    target_col = "Heart-Att"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    if y.dtype == "object" or y.dtype.name == "category":
        y = LabelEncoder().fit_transform(y.astype(str))

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]
    print("Numeric features:", numeric_features)
    print("Categorical features:", categorical_features)

    # Preprocessing pipelines
    numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                        ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer([("num", numeric_transformer, numeric_features),
                                      ("cat", categorical_transformer, categorical_features)])

    # Train/test split (70/30 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
    )

    # Pipeline and estimator
    clf = Pipeline([("preprocessor", preprocessor),
                    ("classifier", RandomForestClassifier(random_state=RANDOM_SEED, class_weight="balanced"))])

    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [5, 10, None],
        "classifier__min_samples_split": [5, 10],
        "classifier__min_samples_leaf": [2, 5]
    }

    grid = GridSearchCV(clf, param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED),
                        scoring="f1", n_jobs=-1, verbose=1)
    print("Starting GridSearchCV...")
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    print("Best params:", grid.best_params_)
    print("Best CV score (f1):", grid.best_score_)

    # Default evaluation with 0.5
    y_pred = best_model.predict(X_test)
    print("F1 Score (Test @0.5):", f1_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Probabilities for threshold tuning
    y_proba = best_model.predict_proba(X_test)[:, 1]

    # Compute PR and ROC curves + save images
    precision, recall, pr_thresh = precision_recall_curve(y_test, y_proba)
    avg_prec = average_precision_score(y_test, y_proba)
    fpr, tpr, roc_thresh = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', label=f'RandomForest (AP={avg_prec:.3f})')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curve'); plt.legend(); plt.grid(True)
    plt.savefig(PR_CURVE, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved PR curve to {PR_CURVE}")

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, marker='.', label=f'ROC AUC={roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curve'); plt.legend(); plt.grid(True)
    plt.savefig(ROC_CURVE, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved ROC curve to {ROC_CURVE}")

    # Evaluate thresholds 0.01..0.99
    thresholds = np.linspace(0.01, 0.99, 99)
    thresh_df = evaluate_thresholds(y_test, y_proba, thresholds, cost_fp=COST_FP, cost_fn=COST_FN)
    thresh_df.to_csv(THRESH_CSV, index=False)
    safe_save_json = lambda obj, p: open(p, "w").write(json.dumps(obj, indent=2, default=str))
    print("Saved threshold metrics to", THRESH_CSV)

    # Choose thresholds according to different policies
    # 1) Optimal F1
    idx_f1 = thresh_df["f1"].idxmax()
    threshold_optimal_f1 = float(thresh_df.loc[idx_f1, "threshold"])
    best_f1 = float(thresh_df.loc[idx_f1, "f1"])

    # 2) Recall target: lowest threshold achieving recall >= TARGET_RECALL, among those pick highest precision
    candidates = thresh_df[thresh_df["recall"] >= TARGET_RECALL]
    if len(candidates) > 0:
        # among candidates pick the one with the highest precision (minimize FP while meeting recall)
        cand_idx = candidates["precision"].idxmax()
        threshold_for_recall = float(candidates.loc[cand_idx, "threshold"])
        thresh_recall_precision = float(candidates.loc[cand_idx, "precision"])
        thresh_recall_f1 = float(candidates.loc[cand_idx, "f1"])
    else:
        threshold_for_recall = None
        thresh_recall_precision = None
        thresh_recall_f1 = None

    # 3) Cost minimization: pick threshold with minimal expected cost = FP*COST_FP + FN*COST_FN
    idx_cost = thresh_df["cost"].idxmin()
    threshold_cost_min = float(thresh_df.loc[idx_cost, "threshold"])
    cost_min_val = float(thresh_df.loc[idx_cost, "cost"])

    selection = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "threshold_optimal_f1": threshold_optimal_f1,
        "best_f1": best_f1,
        "threshold_for_recall_target": threshold_for_recall,
        "recall_target": TARGET_RECALL,
        "threshold_for_recall_precision": thresh_recall_precision,
        "threshold_for_recall_f1": thresh_recall_f1,
        "threshold_cost_min": threshold_cost_min,
        "cost_min_value": cost_min_val,
        "costs": {"FP": COST_FP, "FN": COST_FN},
        "notes": [
            "threshold_optimal_f1 maximizes F1 (balanced precision & recall).",
            "threshold_for_recall_target is the lowest threshold achieving recall >= TARGET_RECALL; among those, we pick the one with highest precision.",
            "threshold_cost_min minimizes expected operational cost = FP*cost_fp + FN*cost_fn."
        ]
    }
    safe_save_json(selection, THRESH_JSON)
    print("Saved threshold selection to", THRESH_JSON)

    # Save human-readable selection text
    with open(THRESH_TXT, "w") as f:
        f.write("Threshold selection summary\\n")
        f.write("===========================\\n\\n")
        f.write(f"Timestamp (UTC): {selection['timestamp']}\\n\\n")
        f.write(f"Optimal F1 threshold: {selection['threshold_optimal_f1']:.3f}\\n")
        f.write(f" - F1 at this threshold: {selection['best_f1']:.3f}\\n")
        row_f1 = thresh_df[thresh_df['threshold'] == selection['threshold_optimal_f1']].iloc[0]
        f.write(f" - Precision: {row_f1['precision']:.3f}\\n")
        f.write(f" - Recall: {row_f1['recall']:.3f}\\n\\n")

        if threshold_for_recall is not None:
            f.write(f"Threshold achieving recall >= {TARGET_RECALL:.2f}: {threshold_for_recall:.3f}\\n")
            f.write(f" - Precision at that threshold: {thresh_recall_precision:.3f}\\n")
            f.write(f" - F1 at that threshold: {thresh_recall_f1:.3f}\\n\\n")
        else:
            f.write(f"No threshold achieves recall >= {TARGET_RECALL:.2f}. Consider improving the model or lowering the target.\\n\\n")

        f.write(f"Threshold minimizing expected cost: {threshold_cost_min:.3f}\\n")
        f.write(f" - Expected cost at this threshold: {cost_min_val:.1f} (FP*{COST_FP} + FN*{COST_FN})\\n\\n")
        f.write("Recommendation:\\n")
        f.write(" - If objective is balanced detection, use threshold_optimal_f1.\\n")
        f.write(" - If clinical priority is recall, use threshold_for_recall_target (if available).\\n")
        f.write(" - If operating under explicit costs, use threshold_cost_min.\\n")

    # Save model + metadata
    joblib.dump({"model": best_model, "threshold_selection": selection}, MODEL_PATH)
    print("Saved model & metadata to", MODEL_PATH)

    # Feature importances (best-effort mapping to post-preprocessing names)
    try:
        preproc = best_model.named_steps["preprocessor"]
        clf = best_model.named_steps["classifier"]
        feature_names = extract_feature_names(preproc, X.columns)
        importances = clf.feature_importances_
        if feature_names is not None and len(feature_names) == len(importances):
            fi_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)
            fi_df.to_csv(FEATURE_IMP_CSV, index=False)
            print("Saved feature importances to", FEATURE_IMP_CSV)
        else:
            fi_df = pd.DataFrame({"feature_index": list(range(len(importances))), "importance": importances}).sort_values("importance", ascending=False)
            fi_df.to_csv(FEATURE_IMP_CSV, index=False)
            print("Saved feature importances by index to", FEATURE_IMP_CSV)
    except Exception as e:
        print("Feature importance extraction failed:", e)

    # Tree visualization: SVG via graphviz if available; pruned PNG via matplotlib to avoid label overlap
    try:
        clf_model = best_model.named_steps["classifier"]
        fnames = None
        try:
            fnames = extract_feature_names(best_model.named_steps["preprocessor"], X.columns)
        except Exception:
            fnames = None

        # Try DOT -> SVG via graphviz
        try:
            dot_path = os.path.join(MODEL_DIR, "tree.dot")
            export_graphviz(clf_model, out_file=dot_path, feature_names=fnames, filled=True, rounded=True, special_characters=True)
            if HAS_GRAPHVIZ:
                with open(dot_path, "r") as fh:
                    dot_src = fh.read()
                gv = graphviz.Source(dot_src)
                gv.format = "svg"
                gv.render(filename=os.path.splitext(TREE_SVG)[0], cleanup=True)
                print("Saved tree SVG to", TREE_SVG)
        except Exception as e:
            print("Graphviz export failed:", e)

        # Save pruned view as PNG (max_depth=3)
        try:
            plt.figure(figsize=(20, 12))
            plot_tree(clf_model, max_depth=3, feature_names=fnames, filled=True, rounded=True, fontsize=10)
            plt.tight_layout()
            plt.savefig(TREE_PNG, dpi=200, bbox_inches="tight")
            plt.close()
            print("Saved pruned tree PNG to", TREE_PNG)
        except Exception as e:
            print("Pruned tree PNG generation failed:", e)
    except Exception as e:
        print("Tree visualization failed:", e)

    print("Done. Key outputs:")
    print(" - Model:", MODEL_PATH)
    print(" - PR curve:", PR_CURVE)
    print(" - ROC curve:", ROC_CURVE)
    print(" - Threshold metrics (CSV/JSON/TXT):", THRESH_CSV, THRESH_JSON, THRESH_TXT)
    print(" - Feature importances:", FEATURE_IMP_CSV)
    print(" - Tree images:", TREE_PNG, TREE_SVG)

if __name__ == '__main__':
    main()
