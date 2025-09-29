# Heart Attack Prediction - Low Level Design (LLD)

## Overview
This document provides a detailed low-level design for the Heart Attack Prediction project. It covers the best possible approaches for each step, justifies design decisions, and answers all requirements from the problem statement.

---

## 1. Data Loading
**Objective:** Efficiently and accurately load the heart patient dataset for further analysis and modeling.

- **Approach:**
  - Use `pandas.read_csv()` to load `US_Heart_Patients.csv` from the `data/` directory.
  - Validate that the file loads without errors and inspect the shape and basic structure of the DataFrame.
  - Set `random_state=13` wherever randomization is involved (e.g., sampling, shuffling).
- **Justification:**
  - Pandas is the industry standard for tabular data manipulation in Python, offering robust methods for reading, inspecting, and cleaning data.
  - Ensures compatibility with downstream data science libraries (scikit-learn, seaborn, etc.).


---

## 2. Exploratory Data Analysis (EDA)
**Objective:** Understand the structure, quality, and relationships in the data to inform preprocessing and modeling decisions.

- **First 10 Rows:**
  - Use `df.head(10)` to get a quick overview of the data and spot obvious issues (e.g., unexpected values, formatting problems).
- **5-Point Summary:**
  - Use `df.describe()` to obtain count, mean, std, min, 25th percentile (Q1), median (Q2), 75th percentile (Q3), and max for all numerical columns.
  - This helps identify the range, central tendency, and spread of each feature.
- **Column Information:**
  - Use `df.info()` to print data types, non-null counts, and memory usage.
  - Use `df.dtypes` to explicitly list the type of each column (e.g., int, float, object).
- **Outlier Detection:**
  - For each numerical column, calculate Q1, Q3, and IQR (Q3-Q1).
  - Define outliers as values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
  - Count the number of outliers per column and report them.
  - Visualize outliers using boxplots for each feature (matplotlib/seaborn).
- **Missing Values:**
  - Use `df.isnull().sum()` to count missing values per column.
  - Visualize missingness with a heatmap (e.g., seaborn's heatmap or missingno library).
- **Correlation Analysis:**
  - Use `df.corr()` to compute pairwise correlation coefficients between numerical features.
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
  - For numerical columns: Impute missing values with the median (robust to outliers).
  - For categorical columns: Impute missing values with the mode (most frequent value).
  - Document the number and location of imputed values.
- **Outlier Treatment:**
  - Use the IQR method to detect outliers.
  - Cap (Winsorize) outliers at the lower and upper bounds ([Q1 - 1.5*IQR, Q3 + 1.5*IQR]) to reduce their influence.
  - Optionally, flag outliers as a new feature if relevant.
- **Encoding Categorical Features:**
  - For nominal (unordered) categorical features: Use one-hot encoding (`pd.get_dummies`).
  - For binary categorical features: Use label encoding (`LabelEncoder`).
  - Ensure no dummy variable trap (drop one column if needed).
- **Feature Engineering:**
  - Create new features based on domain knowledge (e.g., age groups: 30-40, 41-50, etc.).
  - Combine or transform features if it improves model interpretability or performance.
  - Remove redundant or highly correlated features to reduce multicollinearity.
  - Scale features if required by the model (not strictly needed for Decision Trees, but useful for other models).


---

## 4. Data Splitting
**Objective:** Divide the dataset into training and testing sets to evaluate model generalization.

- **Approach:**
  - Use `train_test_split` from `sklearn.model_selection`.
  - Set `test_size=0.3` (70% train, 30% test) and `random_state=13` for reproducibility.
  - Stratify the split on the target variable (if imbalanced) to maintain class proportions in both sets.
- **Justification:**
  - Ensures that the model is evaluated on unseen data, providing a realistic estimate of performance.
  - Stratification prevents bias in class distribution between train and test sets.


---

## 5. Model Preparation & Evaluation
**Objective:** Build, tune, and evaluate a Decision Tree model to predict heart attack risk.

- **Model Selection:**
  - Use `DecisionTreeClassifier` from `sklearn.tree` due to its interpretability and ability to handle both numerical and categorical data.
- **Training:**
  - Fit the model on the training data.
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
- **Model Summary:**
  - Summarize model performance, key hyperparameters, and feature importances.
  - Discuss any trade-offs and limitations observed.


---

## 6. Model Artifacts
**Objective:** Save the trained model and provide an interface for real-time predictions.

- **Model Serialization:**
  - Use `joblib.dump()` or `pickle.dump()` to serialize the final trained model.
  - Save the model file in the `models/` directory (e.g., `models/heart_attack_model.pkl`).
  - Document the model version and training parameters for reproducibility.
- **API Development:**
  - Use Flask to build a REST API in `src/app.py`.
  - Load the serialized model at startup.
  - Expose a `/predict` endpoint that accepts patient data (JSON), preprocesses it, and returns the prediction (risk of heart attack: yes/no or probability).
  - Validate and sanitize input data in the API.
  - Optionally, log requests and predictions for monitoring.


---

## Best Practices
- Use `random_state=13` everywhere for reproducibility.
- Modularize code for reusability and clarity (separate data loading, preprocessing, modeling, and API logic).
- Use clear, labeled visualizations for all EDA steps.
- Validate model with multiple metrics, not just accuracy.
- Document all steps, assumptions, and decisions in code and markdown cells.
- Use version control (e.g., git) to track changes.
- Ensure code is robust to missing or malformed data.
- Test the API with sample requests before deployment.


---

## References
- [Pandas Documentation](https://pandas.pydata.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Seaborn Documentation](https://seaborn.pydata.org/)
- [Flask Documentation](https://flask.palletsprojects.com/)
