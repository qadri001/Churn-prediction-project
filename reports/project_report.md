# Customer Churn Prediction – Project Report

**Project Type:** End-to-End Machine Learning Classification Pipeline  
**Dataset:** Telecom Customer Churn (BigML) – 667 records, 20 features  
**Date:** 2025  
**Tools:** Python, Pandas, Scikit-learn, Matplotlib, Seaborn

---

## 1. Introduction

Customer churn — the phenomenon where customers stop using a product or service — is one
of the most critical business problems in the telecom industry. Acquiring a new customer
costs 5–10× more than retaining an existing one. Proactively identifying at-risk customers
allows companies to intervene with targeted retention offers before the customer leaves.

**Objective:** Build a binary classification model that predicts whether a telecom customer
will churn based on their account details, usage data, and service plan.

---

## 2. Dataset Description

### Source
BigML Telecom Churn Dataset (`churn-bigml-20.csv`)

### Key Statistics

| Metric | Value |
|--------|-------|
| Total Records | 667 |
| Total Features | 20 |
| Target Variable | Churn (Boolean) |
| Churn Rate | ~14.2% (95 churners) |
| Missing Values | None |

### Feature Types
- **Categorical (3):** State, International plan, Voice mail plan
- **Numerical (16):** Account length, call minutes, charges, counts, etc.
- **Target (1):** Churn (True/False → encoded as 1/0)

---

## 3. Exploratory Data Analysis (EDA)

### 3.1 Target Distribution
The dataset is **imbalanced** — 85.8% non-churners vs 14.2% churners.
This imbalance means accuracy alone is not a reliable metric; F1-score
and recall are prioritised.

### 3.2 Key EDA Findings

**International Plan → Strong Churn Signal**
- Customers WITH an international plan churn at ~3× the rate of those without
- This is the strongest categorical predictor

**Customer Service Calls → Escalation Predictor**
- Customers with 4+ service calls have a dramatically higher churn rate
- Indicates frustration or unresolved issues

**Total Day Minutes/Charge → Usage Predictor**
- High daytime usage correlates with higher churn — possibly due to billing shock

**Voice Mail Plan → Protective Factor**
- Customers with voicemail plans are less likely to churn

### 3.3 Correlation Observations
- `Total day minutes` and `Total day charge` are perfectly correlated (r=1.0) — charge is derived from minutes
- Same pattern applies to eve, night, and intl groups
- These redundant features don't harm tree-based models but may affect linear models

---

## 4. Preprocessing

### 4.1 Missing Values
No missing values found in this dataset. The pipeline includes logic to handle them
with median imputation (numeric) and mode imputation (categorical) for generalisability.

### 4.2 Categorical Encoding
**Label Encoding** was used for:
- `State` (51 unique US states → 0–50)
- `International plan` (Yes/No → 1/0)
- `Voice mail plan` (Yes/No → 1/0)

*Note: For models like Logistic Regression, One-Hot Encoding of `State` would be
more appropriate in production. Label encoding is used here for simplicity.*

### 4.3 Feature Scaling
**StandardScaler** was applied (mean=0, std=1) after the train/test split to prevent
data leakage. Scaling is critical for:
- **Logistic Regression** — gradient descent converges faster
- **KNN** — distance calculations are scale-sensitive

### 4.4 Train/Test Split
| Set | Records | Churn Rate |
|-----|---------|-----------|
| Train (80%) | 533 | ~14.3% |
| Test (20%) | 134 | ~14.2% |

Stratified splitting ensures both sets maintain the same churn distribution.

---

## 5. Models

### 5.1 Logistic Regression
A linear model that estimates the probability of churn using a sigmoid function.

**Why use it:**
- Interpretable (coefficients show feature importance)
- Fast to train
- Strong baseline for binary classification

**Hyperparameters:**
- `max_iter=1000` (ensures convergence)
- `random_state=42` (reproducibility)

### 5.2 K-Nearest Neighbours (KNN)
A non-parametric model that classifies a sample based on the majority class
among its K nearest neighbours in feature space.

**Why use it:**
- No assumptions about data distribution
- Captures non-linear decision boundaries
- Good complement to compare against a linear model

**Hyperparameters:**
- `n_neighbors=5` (default, validated via K-sensitivity plot)

---

## 6. Evaluation Results

### 6.1 Metrics Used

| Metric | Formula | Purpose |
|--------|---------|---------|
| Accuracy | (TP+TN) / Total | Overall correctness |
| Precision | TP / (TP+FP) | Avoid false alarms |
| Recall | TP / (TP+FN) | Catch all churners |
| F1-Score | 2×P×R / (P+R) | Balance precision & recall |
| CV F1 | Mean over 5 folds | Model stability |

### 6.2 Confusion Matrix Interpretation

```
                Predicted No Churn   Predicted Churn
Actual No Churn       TN                  FP
Actual Churn          FN                  TP
```

- **FN (False Negative):** Churner predicted as staying → lost customer (costly!)
- **FP (False Positive):** Stayer predicted as churner → wasted retention offer

For churn prediction, **Recall** is usually prioritised to minimise missed churners.

### 6.3 Results Summary
*(See `outputs/03_evaluation_summary.csv` for exact numbers)*

Both models were evaluated on the same held-out test set (20% of data).
Cross-validation F1 scores confirm consistent performance across folds.

---

## 7. Overfitting vs Underfitting Analysis

### Definitions

| Condition | Description | Indicator |
|-----------|-------------|-----------|
| **Overfitting** | Model memorises training data, fails on new data | Train accuracy >> Test accuracy |
| **Underfitting** | Model too simple to capture patterns | Both train and test accuracy are low |
| **Well-fit** | Model generalises effectively | Train ≈ Test accuracy |

### Results
Both models show a small gap between train and test accuracy (< 3%), indicating
**well-fit models** that generalise well to unseen data.

### KNN K-Sensitivity
As K increases:
- Low K (k=1) → overfitting (memorises training points)
- High K (k=25+) → underfitting (too many neighbours, loses local patterns)
- k=5 offers a good bias-variance tradeoff

---

## 8. Key Findings & Business Insights

1. **International Plan** is the single strongest churn predictor — review international pricing
2. **Customer Service Calls ≥ 4** signals high dissatisfaction — flag for proactive outreach
3. **High daytime usage** customers are at risk — consider loyalty discounts
4. **Voice Mail Plan** customers are more engaged and less likely to churn

### Recommended Business Actions
| Risk Segment | Action |
|--------------|--------|
| International plan + high charges | Offer discounted international bundles |
| 3+ customer service calls | Proactive account manager callback |
| No voicemail, no international | Upsell engagement features |
| High day minutes | Offer unlimited day plan upgrade |

---

## 9. Limitations & Next Steps

### Current Limitations
- Dataset is relatively small (667 records)
- `State` feature encoded with label encoding — one-hot would be more appropriate
- Correlated features (charge = minutes × rate) could be dropped

### Recommended Improvements
1. **Feature selection** — drop perfectly correlated charge columns
2. **Handle class imbalance** — use SMOTE or class weights
3. **Try additional models** — Random Forest, XGBoost, SVM
4. **Hyperparameter tuning** — GridSearchCV for optimal parameters
5. **SHAP values** — explain individual predictions
6. **Deploy as API** — Flask/FastAPI endpoint for real-time scoring

---

## 10. Conclusion

This project demonstrates a complete, industry-standard ML pipeline for customer churn
prediction. Starting from raw data, the pipeline performs EDA, preprocessing, model
training with two algorithms (Logistic Regression and KNN), and comprehensive evaluation
using accuracy, precision, recall, F1-score, confusion matrices, ROC curves, and
cross-validation.

Both models achieve solid performance with no significant overfitting, making them viable
baselines for a production churn prediction system.

---

*Report generated as part of End-to-End Customer Churn Prediction Portfolio Project*
