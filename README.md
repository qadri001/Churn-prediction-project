# 📊 End-to-End Customer Churn Prediction System

> **Portfolio-grade ML project** | Classification | Scikit-learn | Python

---

## 🎯 Problem Statement

Predict whether a telecom customer will **churn (leave the service)** based on their
usage patterns, service plans, and interaction history.

---

## 📁 Project Structure

```
churn_project/
├── data/
│   └── churn-bigml-20.csv        ← Dataset (667 rows, 20 features)
├── notebooks/
│   └── churn_prediction.ipynb    ← Interactive Jupyter Notebook
├── src/
│   └── churn_pipeline.py         ← Full runnable Python pipeline
├── outputs/                      ← Auto-generated charts & CSV
│   ├── 01_eda_overview.png
│   ├── 02_model_evaluation.png
│   ├── 03_evaluation_summary.csv
│   └── 04_overfitting_analysis.png
├── reports/
│   └── project_report.md         ← Written project report
└── README.md
```

---

## 🗂️ Dataset Description

| Column | Type | Description |
|--------|------|-------------|
| State | Categorical | US state of the customer |
| Account length | Numeric | Days as a customer |
| Area code | Numeric | Phone area code |
| International plan | Categorical | Yes/No |
| Voice mail plan | Categorical | Yes/No |
| Number vmail messages | Numeric | Voicemail message count |
| Total day minutes | Numeric | Daytime call minutes |
| Total day calls | Numeric | Daytime call count |
| Total day charge | Numeric | Daytime charge ($) |
| Total eve minutes | Numeric | Evening call minutes |
| Total eve calls | Numeric | Evening call count |
| Total eve charge | Numeric | Evening charge ($) |
| Total night minutes | Numeric | Night call minutes |
| Total night calls | Numeric | Night call count |
| Total night charge | Numeric | Night charge ($) |
| Total intl minutes | Numeric | International call minutes |
| Total intl calls | Numeric | International call count |
| Total intl charge | Numeric | International charge ($) |
| Customer service calls | Numeric | Number of CS calls made |
| **Churn** | Boolean | **Target: True = churned** |

**Shape:** 667 rows × 20 columns  
**Churn Rate:** ~14.2%

---

## ⚙️ ML Pipeline Steps

### 1. 📥 Load Data
- Read CSV with pandas
- Inspect shape, column types, missing values

### 2. 🔍 Exploratory Data Analysis (EDA)
- Target distribution (pie + bar chart)
- Churn by International Plan & Voice Mail Plan
- Numerical feature distributions by churn status
- Correlation heatmap of all numeric features

### 3. 🔧 Preprocessing & Feature Engineering
| Step | Method |
|------|--------|
| Missing values | Median (numeric) / Mode (categorical) |
| Categorical encoding | LabelEncoder (State, plans) |
| Target encoding | Boolean → 0/1 integer |
| Feature scaling | StandardScaler (zero mean, unit variance) |
| Train/Test split | 80% train / 20% test, stratified |

### 4. 🤖 Model Training
| Model | Key Parameters |
|-------|---------------|
| Logistic Regression | max_iter=1000, random_state=42 |
| KNN | n_neighbors=5 |

### 5. 📈 Evaluation Metrics
- **Accuracy** — overall correct predictions
- **Precision** — of predicted churners, how many actually churned?
- **Recall** — of actual churners, how many did we catch?
- **F1-Score** — harmonic mean of precision and recall
- **Confusion Matrix** — visual breakdown of TP/FP/TN/FN
- **5-Fold Cross-Validation** — stability check

### 6. 🔬 Overfitting vs Underfitting
Compare train vs test accuracy for each model:
- **Gap < 3%** → Well-fit model ✅
- **Gap > 5%** → Potential overfitting ⚠️
- **Both low** → Underfitting ⚠️

---

## 🚀 How to Run

### Option A – Python Script
```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# Run the full pipeline
cd churn_project
python src/churn_pipeline.py
```

### Option B – Jupyter Notebook
```bash
pip install jupyter pandas numpy matplotlib seaborn scikit-learn
jupyter notebook notebooks/churn_prediction.ipynb
```

Outputs are automatically saved to the `outputs/` folder.

---

## 📊 Expected Outputs

| File | Description |
|------|-------------|
| `01_eda_overview.png` | EDA charts (distribution, correlation, box plots) |
| `02_model_evaluation.png` | Confusion matrices, ROC curves, metric comparison |
| `03_evaluation_summary.csv` | Numeric evaluation results for both models |
| `04_overfitting_analysis.png` | Train vs Test accuracy per model |

---

## 🔑 Key Findings

1. **International Plan** — customers with international plan churn at 3× higher rate
2. **Customer Service Calls** — 4+ calls is a strong churn predictor
3. **Total Day Charge** — higher charges correlate with higher churn
4. **Logistic Regression** — strong interpretable baseline
5. **KNN** — competitive but sensitive to k value

---

## 📦 Requirements

```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
jupyter>=1.0.0
```

---

## 🎓 Portfolio Value

> *"Built a complete churn prediction ML pipeline including EDA, preprocessing,
> feature engineering, classification with Logistic Regression and KNN, model
> evaluation with multiple metrics, and overfitting analysis."*

---

*Project by: [Your Name] | Dataset: BigML Telecom Churn | Tools: Python, Scikit-learn, Matplotlib*
