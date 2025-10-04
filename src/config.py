import os

# Reproducibility
RANDOM_SEED = 13

# Paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "US_Heart_Patients.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_FILENAME = "final_model.pkl"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

PR_CURVE = os.path.join(MODEL_DIR, "pr_curve.png")
ROC_CURVE = os.path.join(MODEL_DIR, "roc_curve.png")
THRESH_CSV = os.path.join(MODEL_DIR, "threshold_metrics.csv")
THRESH_JSON = os.path.join(MODEL_DIR, "threshold_selection.json")
THRESH_TXT = os.path.join(MODEL_DIR, "threshold_selection.txt")
FEATURE_IMP_CSV = os.path.join(MODEL_DIR, "feature_importances.csv")
TREE_PNG = os.path.join(MODEL_DIR, "tree_pruned.png")
TREE_SVG = os.path.join(MODEL_DIR, "tree.svg")

# Policy params (tune for your clinical context)
COST_FP = 1
COST_FN = 10
TARGET_RECALL = 0.80
