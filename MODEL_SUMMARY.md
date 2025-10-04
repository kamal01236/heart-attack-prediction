# Model Summary — Heart Attack Prediction

**Model type:** RandomForest pipeline (preprocessing + RandomForest) — selected via GridSearchCV.  
**Random seed:** 13.

## Model selection rationale

We experimented with both **DecisionTreeClassifier** and **RandomForestClassifier**:

- A single **Decision Tree** was used as a baseline and for visualization purposes.  
- The **RandomForestClassifier** consistently provided better generalization, higher F1, and more stable recall/precision trade-offs.  
- Therefore, **the final predictive model is a RandomForestClassifier**, trained within a preprocessing pipeline and tuned with GridSearchCV.

For interpretability, we also visualize one pruned Decision Tree from the RandomForest (see `models/tree_pruned.png`), 
but this is only for explanation — the deployed model is the full RandomForest ensemble.


## Best hyperparameters (GridSearchCV result)
- classifier__n_estimators: 200
- classifier__max_depth: 10
- classifier__min_samples_split: 5
- classifier__min_samples_leaf: 2

*(Replace above with exact `grid.best_params_` from your training run.)*

## Test set performance (example — replace with your numbers)
- F1 (at default 0.5): 0.72
- Precision: 0.68
- Recall: 0.77
- ROC AUC: 0.86
- Average Precision (PR AUC): 0.70

## Selected thresholds
- threshold_optimal_f1: 0.42
- threshold_for_recall_target (recall>=0.80): 0.28  (if available)
- threshold_cost_min: 0.34

## Confusion Matrix (test set) at chosen threshold (example)
|         | Pred 0 | Pred 1 |
|---------|--------|--------|
| True 0  |  210   |   30   |
| True 1  |   25   |   35   |

## Recommendation
- For balanced operation, use `threshold_optimal_f1`.
- If clinical setting requires higher recall (sensitivity), use `threshold_for_recall_target` (if it exists); otherwise re-train / tune to improve recall.
- For an operational policy that penalizes false negatives strongly, use `threshold_cost_min`.

## How to reproduce
1. Place `data/US_Heart_Patients.csv` in the `data/` directory.
2. Install requirements: `pip install -r requirements.txt`
3. Run training: `python -m src.train_model`
4. The artifacts will be written to `models/` including `final_model.pkl` and threshold metadata.

