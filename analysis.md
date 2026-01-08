# Churn Prediction Analysis Report

## 1. Project Overview
**Objective**: Build an end-to-end MLOps pipeline for predicting customer churn.
**Target**: `churn` (Binary: Yes/No).
**Dataset**: Telco Customer Churn.

## 2. Exploratory Data Analysis (EDA)
-   **Class Imbalance**: ~26.5% Churn (Imbalanced).
-   **Missing Values**: `internet_service` column had ~22% missing values initially thought to be random, but later identified as a meaningful "No Internet Service" category.
-   **Key Features**: Contract type, Tenure, and Monthly Charges are top drivers.

## 3. Experimentation Summary

### Phase A: Baseline (The "Over-Engineering" Trap)
-   **Approach**:
    -   Imputation: KNN (Computationaly expensive).
    -   Encoding: One-Hot Encoding (Exploded feature space).
    -   Balancing: SMOTE (Synthetic minority oversampling).
-   **Best Model**: XGBoost
-   **Result**: F1 Score **0.59**.
-   **Issue**: High accuracy (78%) but poor Recall (49%). The complex imputation likely introduced noise, masking the "No Internet Service" signal.

### Phase B: Gap Analysis & Simplify (The "Data-Centric" Fix)
-   **Hypothesis**: "Less is More". Tree-based models (RF, XGB, LightGBM) handle categorical data and missing values well naturally. One-Hot encoding can sometimes dilutes information for these models.
-   **New Approach**:
    -   Imputation: **None** (Fill with "Unknown" / Native handling).
    -   Encoding: **Label Encoding** (Preserves single-column information).
    -   Balancing: **Sample Weights** (Penalize wrong predictions on minority class) instead of SMOTE.
-   **Result**: Immediate jump in performance.

### Phase C: Final Benchmark (10 Models)
We tested 10 classifiers using the simplified approach.

| Model | Accuracy | F1 Score | Recall (Class 1) | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **LightGBM** | **84.40%** | **0.7428** | **High** | **CHAMPION** |
| CatBoost | 84.20% | 0.7399 | High | Runner-up |
| Random Forest | 83.88% | 0.7340 | Moderate | Strong Baseline |
| XGBoost | 82.75% | 0.7264 | Moderate | Good |
| Logistic Reg | 70.15% | 0.6245 | Low | Too Simple |
| SVC | 60.62% | 0.4709 | Low | Poor |

## 4. Key Learnings & "Why F1 Score?"

### The Learning: Complexity vs. Suitability
> **"Simple approach (Label Encoding + Native Weighting) was far superior to our complex one (KNN + One-Hot)."**

*   **Engineering Lesson**: We initially over-engineered the data (KNN, SMOTE). This corrupted the underlying signal. By respecting the nature of Gradient Boosted Trees (which handle raw-like data well), we gained **+15% F1 score** with **less code**.
*   **Data Lesson**: Missing values are often information (e.g., "No Internet"). Imputing them removes that signal.

### The Metric: Why Maximize F1?
In Churn Prediction, **Accuracy is a trap**.
*   If 80% of users stay, a model that predicts "No Churn" for everyone has **80% Accuracy** but **0% Value**.
*   **Precision**: Of those we predicted to churn, how many actually did? (Cost: Wasting retention budget).
*   **Recall**: Of those who actually churned, how many did we find? (Cost: Losing customers).
*   **F1 Score**: The harmonic mean of Precision and Recall. It forces the model to balance "Catching Churners" vs "Not crying wolf".
*   **Our Result (0.74)**: A high F1 means we are effectively identifying 70-80% of churners without spamming loyal customers. This is business-ready.

## 5. Next Steps: Pipeline Construction
With **LightGBM** as our Champion Model, we proceed to **MLOps Implementation**.
We will move from Notebooks (`experiment/`) to Modular Code (`src/`):
1.  `src/data_loader.py`: Replicate the "Load CSV + Fill NA" logic.
2.  `src/preprocessing.py`: Replicate "Label Encoding + Split".
3.  `src/train.py`: LightGBM training with `class_weight='balanced'` and MLflow logging.
4.  `src/evaluate.py`: Generate metrics and confusion matrix.
