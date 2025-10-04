# LLD - Threshold selection and training pipeline
See earlier documentation in README.md. This file summarizes design decisions.
- Preprocessing: missing indicators + winsorize + median/mode imputation + one-hot encoding
- Models: DecisionTree baseline (tuned) and RandomForest (tuned)
- Threshold selection: F1-optimal, recall-target, cost-minimization
- Serving: Flask API with payload validation and endpoints /predict, /thresholds, /health
