
# Heart Attack Prediction - Low Level Design (LLD)

## Overview
This document provides a robust, production-ready low-level design for the Heart Attack Prediction project. It is tailored to the actual structure and challenges of the `US_Heart_Patients.csv` dataset and addresses all requirements for experimentation, deployment, and interpretability.

---


## 1. Data Loading
**Objective:** Efficiently and accurately load the heart patient dataset for further analysis and modeling.

- **Approach:**
  - Use `pandas.read_csv()` to load `US_Heart_Patients.csv` from the `data/` directory.
  - Handle missing values represented as empty strings, 'NA', or other non-numeric placeholders using `na_values` parameter.
  - Validate the file loads without errors, inspect the shape, and check for duplicate rows.
  - Set `random_state=13` wherever randomization is involved (e.g., sampling, shuffling).
- **Justification:**
  - Pandas is the industry standard for tabular data manipulation in Python, offering robust methods for reading, inspecting, and cleaning data.
  - Ensures compatibility with downstream data science libraries (scikit-learn, seaborn, etc.).


---


## 2. Exploratory Data Analysis (EDA)
**Objective:** Understand the structure, quality, and relationships in the data to inform preprocessing and modeling decisions.

- **First 10 Rows:**
  - Use `df.head(10)` to get a quick overview and spot obvious issues (e.g., missing values, out-of-range values, formatting problems).
- **5-Point Summary:**
  - Use `df.describe(include='all')` for both numerical and categorical columns.
  - This helps identify the range, central tendency, spread, and unique values of each feature.
- **Column Information:**
  - Use `df.info()` and `df.dtypes` to print data types, non-null counts, and memory usage.
  - Check for columns with mixed types or unexpected data types.
- **Outlier Detection:**
  - For each numerical column, calculate Q1, Q3, and IQR (Q3-Q1).
  - Define outliers as values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
  - Count and visualize outliers using boxplots for each feature (matplotlib/seaborn).
- **Missing Values:**
  - Use `df.isnull().sum()` to count missing values per column.
  - Visualize missingness with a heatmap (e.g., seaborn's heatmap or missingno library).
- **Correlation Analysis:**
  - Use `df.corr(numeric_only=True)` to compute pairwise correlation coefficients between numerical features.
  - Visualize the correlation matrix using a heatmap (seaborn's `heatmap`).
  - Identify highly correlated features (multicollinearity) and their impact on modeling.
- **Distribution Analysis:**
  - Plot histograms for each numerical feature to visualize their distributions (normal, skewed, bimodal, etc.).
  - Use boxplots to further inspect spread and outliers.
  - For categorical features, use bar plots to show value counts.
- **Charts and Graphs:**
  - Use matplotlib and seaborn for all visualizations.
  - Ensure all plots are clearly labeled and interpreted in markdown cells.


---


## 3. Data Preprocessing (BONUS POINTS)
**Objective:** Prepare the data for modeling by handling missing values, outliers, categorical variables, and engineering new features.

- **Missing Value Imputation:**
  - For numerical columns: Impute missing values with the median (robust to outliers and skewed distributions).
  - For categorical columns: Impute missing values with the mode (most frequent value).
  - Document the number and location of imputed values for transparency.
- **Outlier Treatment:**
  - Use the IQR method to detect outliers.
  - Cap (Winsorize) outliers at the lower and upper bounds ([Q1 - 1.5*IQR, Q3 + 1.5*IQR]) to reduce their influence.
  - Optionally, flag outliers as a new feature if relevant.
- **Encoding Categorical Features:**
  - For nominal (unordered) categorical features: Use one-hot encoding (`pd.get_dummies`).
  - For binary categorical features: Use label encoding (`LabelEncoder`).
  - Ensure no dummy variable trap (drop one column if needed).
  - For columns with mixed types (e.g., 'Male', 'Female', missing), standardize before encoding.
- **Feature Engineering:**
  - Create new features based on domain knowledge (e.g., age groups: 30-40, 41-50, etc.).
  - Combine or transform features if it improves model interpretability or performance.
  - Remove redundant or highly correlated features to reduce multicollinearity.
  - Scale features if required by the model (not strictly needed for Decision Trees, but useful for other models).
  - Consider interaction features if justified by EDA.


---


## 4. Data Splitting
**Objective:** Divide the dataset into training and testing sets to evaluate model generalization.

- **Approach:**
  - Use `train_test_split` from `sklearn.model_selection`.
  - Set `test_size=0.3` (70% train, 30% test) and `random_state=13` for reproducibility.
  - Stratify the split on the target variable (if imbalanced) to maintain class proportions in both sets.
  - Ensure no data leakage by splitting after all preprocessing steps except those that use target information.
- **Justification:**
  - Ensures that the model is evaluated on unseen data, providing a realistic estimate of performance.
  - Stratification prevents bias in class distribution between train and test sets.


---


## 5. Model Preparation & Evaluation
**Objective:** Build, tune, and evaluate a Decision Tree model to predict heart attack risk.

- **Model Selection:**
  - Use `DecisionTreeClassifier` from `sklearn.tree` for interpretability and ability to handle both numerical and categorical data.
  - Consider ensemble models (RandomForest, GradientBoosting) for benchmarking, but Decision Tree is the primary model for explainability.
- **Training:**
  - Fit the model on the training data using a pipeline that includes all preprocessing steps.
  - Predict outcomes for both training and test sets.
- **Evaluation Metrics:**
  - **F1 Score:** Use `f1_score` from `sklearn.metrics` to balance precision and recall, especially important for imbalanced datasets.
  - **Confusion Matrix:** Use `confusion_matrix` to visualize true positives, false positives, true negatives, and false negatives.
  - **Classification Report:** Use `classification_report` for precision, recall, F1-score, and support for each class.
  - **Accuracy, Precision, Recall:** Optionally report these for completeness.
- **Hyperparameter Tuning:**
  - Use `GridSearchCV` to search for optimal hyperparameters (e.g., `max_depth`, `min_samples_split`, `min_samples_leaf`, `criterion`).
  - Use 5-fold cross-validation and set `random_state=13` for reproducibility.
  - Select the model with the best cross-validated F1 score.
- **Overfitting/Underfitting Handling:**
  - Monitor train vs. test performance.
  - Use pruning parameters (`max_depth`, `min_samples_leaf`) to prevent overfitting.
  - Compare train/test metrics to detect overfitting or underfitting.
- **Model Summary:**
  - Summarize model performance, key hyperparameters, and feature importances.
  - Discuss any trade-offs and limitations observed.
  - Visualize the decision tree for interpretability.


---



## 6. Model Artifacts & API Design
**Objective:** Save the trained model, provide an interface for real-time predictions, and enable step-by-step workflow via API endpoints.

- **Model Serialization:**
  - Use `joblib.dump()` or `pickle.dump()` to serialize the final trained model (including the full pipeline).
  - Save the model file in the `models/` directory (e.g., `models/final_decision_tree_model.pkl`).
  - Document the model version, training parameters, and preprocessing steps for reproducibility.

- **API Development (Flask, src/app.py):**
  - **/train (POST):**
    - Triggers model training pipeline (data load, preprocess, train, evaluate, save model).
    - Returns training metrics (F1, confusion matrix, classification report, best hyperparameters).
    - Allows retraining with new data if needed.
  - **/predict (POST):**
    - Accepts patient data (JSON), preprocesses using the same pipeline, and returns prediction (risk: yes/no or probability).
    - Handles both single and batch predictions.
  - **/tree (GET):**
    - Returns a visualization (e.g., SVG or PNG) or a text representation of the trained decision tree.
    - For VS Code: Save the tree visualization as an image file (e.g., `models/tree_visualization.png`) and provide the path for easy preview in the editor.
  - **Input Validation:**
    - Validate and sanitize input data for all endpoints (check for missing/extra fields, correct types, and value ranges).
    - Return clear error messages for invalid input.
  - **Logging:**
    - Log requests, predictions, and training events for monitoring and debugging.

- **VS Code Integration:**
  - Tree visualization can be previewed directly in VS Code by opening the generated image file.
  - All API endpoints can be tested using REST clients (e.g., Thunder Client, Postman) or VS Code's built-in REST client.
  - Jupyter notebook can be used for experimentation and model analysis.



---



## Best Practices
- Use `random_state=13` everywhere for reproducibility.
- Modularize code for reusability and clarity (separate data loading, preprocessing, modeling, and API logic).
- Use clear, labeled visualizations for all EDA steps.
- Validate model with multiple metrics, not just accuracy.
- Document all steps, assumptions, and decisions in code and markdown cells.
- Use version control (e.g., git) to track changes.
- Ensure code is robust to missing or malformed data (especially for columns with mixed types or missing values).
- Test the API with sample requests before deployment.
- Provide clear API documentation and example requests/responses.
- For tree visualization, use `sklearn.tree.export_graphviz` or `plot_tree` and save as an image for easy preview in VS Code.
- Use pipeline objects to encapsulate preprocessing and modeling for consistency between training and inference.


---

## References
- [Pandas Documentation](https://pandas.pydata.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Seaborn Documentation](https://seaborn.pydata.org/)
- [Flask Documentation](https://flask.palletsprojects.com/)
