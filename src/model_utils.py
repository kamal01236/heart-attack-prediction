import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from . import config


def load_model_bundle(path=None):
    """Load a saved model bundle (may be a raw model or dict with 'model' and metadata).
    Returns (model, metadata_dict).
    """
    if path is None:
        path = config.MODEL_PATH
    if not os.path.exists(path):
        # fallback: try alternative name used historically
        alt = os.path.join(config.MODEL_DIR, "final_randomforest_model.pkl")
        if os.path.exists(alt):
            path = alt
        else:
            raise FileNotFoundError(f"Model not found at {path}")
    bundle = joblib.load(path)
    if isinstance(bundle, dict) and "model" in bundle:
        return bundle["model"], bundle.get("threshold_selection", {})
    return bundle, {}


def save_model_bundle(model, threshold_selection: dict, path=None):
    if path is None:
        path = config.MODEL_PATH
    joblib.dump({"model": model, "threshold_selection": threshold_selection}, path)


def choose_threshold(meta: dict, policy_preference: list = None):
    """Pick a threshold from metadata using a preference order.
    Default preference: cost_min, optimal_f1, 0.5
    """
    if policy_preference is None:
        policy_preference = ["threshold_cost_min", "threshold_optimal_f1", "threshold_for_recall_target"]
    for key in policy_preference:
        if key in meta and meta.get(key) is not None:
            try:
                return float(meta.get(key))
            except Exception:
                continue
    return 0.5


def evaluate_thresholds(y_true, y_proba, thresholds, cost_fp=config.COST_FP, cost_fn=config.COST_FN):
    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        precision = float(np.nan)
        recall = float(np.nan)
        try:
            precision = float(np.sum((y_true==1)&(y_pred==1))/max(1, int(np.sum(y_pred==1))))
        except Exception:
            precision = 0.0
        try:
            recall = float(np.sum((y_true==1)&(y_pred==1))/max(1, int(np.sum(y_true==1))))
        except Exception:
            recall = 0.0
        f1 = 2*(precision*recall)/(precision+recall) if (precision+recall) > 0 else 0.0
        cost = int(fp) * cost_fp + int(fn) * cost_fn
        rows.append({
            "threshold": float(t),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
            "precision": float(precision), "recall": float(recall), "f1": float(f1), "cost": float(cost)
        })
    return pd.DataFrame(rows)
