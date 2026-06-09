"""
============================================================
  Customer Churn Prediction Pipeline
  End-to-End ML System: Preprocessing → Training → Evaluation
============================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_curve, auc
)

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "churn-bigml-20.csv")
OUT_DIR    = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE    = 0.20


# ─────────────────────────────────────────────────────────────
# 1.  LOAD DATA
# ─────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    print("\n" + "="*60)
    print("  STEP 1 – LOADING DATA")
    print("="*60)
    df = pd.read_csv(path)
    print(f"  Rows : {df.shape[0]:,}  |  Columns : {df.shape[1]}")
    print(f"  Columns: {list(df.columns)}")
    return df


# ─────────────────────────────────────────────────────────────
# 2.  EDA
# ─────────────────────────────────────────────────────────────
def run_eda(df: pd.DataFrame) -> None:
    print("\n" + "="*60)
    print("  STEP 2 – EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*60)

    print("\n  [2a] Basic statistics")
    print(df.describe().to_string())

    print("\n  [2b] Missing values per column")
    mv = df.isnull().sum()
    print(mv[mv > 0].to_string() if mv.sum() > 0 else "  → No missing values found.")

    print("\n  [2c] Target distribution")
    churn_counts = df['Churn'].value_counts()
    print(churn_counts.to_string())
    churn_pct = churn_counts / len(df) * 100
    print(f"  Churn rate: {churn_pct[True]:.1f}%")

    # ── EDA Figure ──────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("Customer Churn – Exploratory Data Analysis", fontsize=16, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    colors = ['#4CAF50', '#F44336']

    # (i) Churn distribution pie
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.pie(churn_counts.values, labels=['No Churn', 'Churn'],
            colors=colors, autopct='%1.1f%%', startangle=140)
    ax0.set_title("Churn Distribution")

    # (ii) Churn by International Plan
    ax1 = fig.add_subplot(gs[0, 1])
    pd.crosstab(df['International plan'], df['Churn']).plot(
        kind='bar', ax=ax1, color=colors, edgecolor='white')
    ax1.set_title("Churn by International Plan")
    ax1.set_xlabel("International Plan")
    ax1.set_ylabel("Count")
    ax1.tick_params(axis='x', rotation=0)
    ax1.legend(["No Churn", "Churn"])

    # (iii) Churn by Voice Mail Plan
    ax2 = fig.add_subplot(gs[0, 2])
    pd.crosstab(df['Voice mail plan'], df['Churn']).plot(
        kind='bar', ax=ax2, color=colors, edgecolor='white')
    ax2.set_title("Churn by Voice Mail Plan")
    ax2.set_xlabel("Voice Mail Plan")
    ax2.set_ylabel("Count")
    ax2.tick_params(axis='x', rotation=0)
    ax2.legend(["No Churn", "Churn"])

    # (iv) Total Day Minutes distribution
    ax3 = fig.add_subplot(gs[1, 0])
    for val, col, lbl in zip([False, True], colors, ['No Churn', 'Churn']):
        subset = df[df['Churn'] == val]['Total day minutes']
        ax3.hist(subset, bins=30, alpha=0.6, color=col, label=lbl)
    ax3.set_title("Total Day Minutes by Churn")
    ax3.set_xlabel("Total Day Minutes")
    ax3.set_ylabel("Frequency")
    ax3.legend()

    # (v) Customer Service Calls
    ax4 = fig.add_subplot(gs[1, 1])
    df.boxplot(column='Customer service calls', by='Churn', ax=ax4,
               patch_artist=True,
               boxprops=dict(facecolor='#90CAF9'),
               medianprops=dict(color='red', linewidth=2))
    ax4.set_title("Customer Service Calls by Churn")
    ax4.set_xlabel("Churn")
    ax4.set_ylabel("Number of Calls")
    plt.sca(ax4)
    plt.title("Customer Service Calls by Churn")
    plt.suptitle("")

    # (vi) Account Length distribution
    ax5 = fig.add_subplot(gs[1, 2])
    df['Account length'].hist(bins=40, ax=ax5, color='#7986CB', edgecolor='white')
    ax5.set_title("Account Length Distribution")
    ax5.set_xlabel("Account Length (days)")
    ax5.set_ylabel("Frequency")

    # (vii) Correlation heatmap (numerical)
    ax6 = fig.add_subplot(gs[2, :])
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='RdYlGn',
                center=0, linewidths=0.5, ax=ax6, annot_kws={"size": 7})
    ax6.set_title("Feature Correlation Matrix")

    path = os.path.join(OUT_DIR, "01_eda_overview.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  ✔ EDA chart saved → {path}")


# ─────────────────────────────────────────────────────────────
# 3.  PREPROCESSING
# ─────────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    print("\n" + "="*60)
    print("  STEP 3 – PREPROCESSING & FEATURE ENGINEERING")
    print("="*60)

    df = df.copy()

    # 3a. Handle missing values
    print("\n  [3a] Handling missing values...")
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in [np.float64, np.int64]:
                df[col].fillna(df[col].median(), inplace=True)
                print(f"       {col}: filled with median")
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
                print(f"       {col}: filled with mode")
    print("  → Missing values resolved.")

    # 3b. Encode categorical variables
    print("\n  [3b] Encoding categorical variables...")
    le = LabelEncoder()
    cat_cols = ['State', 'International plan', 'Voice mail plan']
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
        print(f"       Label-encoded: {col}")

    # 3c. Encode target
    df['Churn'] = df['Churn'].astype(int)
    print("  → Target 'Churn': False→0, True→1")

    # 3d. Feature / Target split
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    print(f"\n  Features : {X.shape[1]}  |  Samples : {X.shape[0]}")

    # 3e. Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    print(f"  Train set: {X_train.shape[0]}  |  Test set: {X_test.shape[0]}")

    # 3f. Feature scaling
    print("\n  [3c] Scaling numerical features with StandardScaler...")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    print("  → Scaling complete.")

    return X_train_sc, X_test_sc, y_train, y_test, list(X.columns)


# ─────────────────────────────────────────────────────────────
# 4.  TRAIN & EVALUATE MODELS
# ─────────────────────────────────────────────────────────────
def train_and_evaluate(X_train, X_test, y_train, y_test):
    print("\n" + "="*60)
    print("  STEP 4 – MODEL TRAINING & EVALUATION")
    print("="*60)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "KNN (k=5)"          : KNeighborsClassifier(n_neighbors=5),
    }

    results = {}
    trained = {}

    for name, model in models.items():
        print(f"\n  ── {name} ──")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        cm   = confusion_matrix(y_test, y_pred)

        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')

        results[name] = {
            "Accuracy" : round(acc,  4),
            "Precision": round(prec, 4),
            "Recall"   : round(rec,  4),
            "F1-Score" : round(f1,   4),
            "CV F1 Mean": round(cv_scores.mean(), 4),
            "CV F1 Std" : round(cv_scores.std(),  4),
        }
        trained[name] = (model, y_pred, cm)

        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}")
        print(f"  F1-Score : {f1:.4f}")
        print(f"  5-Fold CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"\n{classification_report(y_test, y_pred, target_names=['No Churn','Churn'])}")

    return results, trained


# ─────────────────────────────────────────────────────────────
# 5.  VISUALISE EVALUATION
# ─────────────────────────────────────────────────────────────
def plot_evaluation(results, trained, y_test, X_test):
    print("\n" + "="*60)
    print("  STEP 5 – EVALUATION VISUALISATIONS")
    print("="*60)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Model Evaluation Dashboard", fontsize=15, fontweight='bold')

    model_names = list(results.keys())
    palette     = ['#3F51B5', '#E91E63']

    # (i) Confusion Matrices
    for idx, (name, (model, y_pred, cm)) in enumerate(trained.items()):
        ax = axes[0, idx]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['No Churn', 'Churn'],
                    yticklabels=['No Churn', 'Churn'])
        ax.set_title(f"Confusion Matrix\n{name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    # (ii) Metric comparison bar chart
    ax = axes[0, 2]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x  = np.arange(len(metrics))
    w  = 0.35
    for i, (name, color) in enumerate(zip(model_names, palette)):
        vals = [results[name][m] for m in metrics]
        bars = ax.bar(x + i*w, vals, w, label=name, color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x + w/2)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Metric Comparison")
    ax.legend()

    # (iii) ROC Curves
    ax = axes[1, 0]
    for name, color in zip(model_names, palette):
        model = trained[name][0]
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_test)[:, 1]
        else:
            proba = model.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, label=f"{name} (AUC={roc_auc:.3f})", lw=2)
    ax.plot([0,1],[0,1],'k--', lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc='lower right', fontsize=9)

    # (iv) Cross-validation F1 comparison
    ax = axes[1, 1]
    cv_means = [results[n]["CV F1 Mean"] for n in model_names]
    cv_stds  = [results[n]["CV F1 Std"]  for n in model_names]
    bars = ax.bar(model_names, cv_means, color=palette, alpha=0.85, edgecolor='white')
    ax.errorbar(model_names, cv_means, yerr=cv_stds, fmt='none',
                ecolor='black', capsize=6, elinewidth=2)
    for bar, v in zip(bars, cv_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel("Mean F1-Score")
    ax.set_ylim(0, 1.0)
    ax.set_title("5-Fold Cross-Validation F1")

    # (v) KNN – K value sensitivity
    ax = axes[1, 2]
    k_range = range(1, 21)
    k_f1    = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_test, y_test)  # quick estimate
        y_k = knn.predict(X_test)
        k_f1.append(f1_score(y_test, y_k, zero_division=0))
    ax.plot(k_range, k_f1, marker='o', color='#E91E63', lw=2, markersize=5)
    ax.axvline(x=5, color='gray', linestyle='--', label='k=5 (used)')
    ax.set_xlabel("K value")
    ax.set_ylabel("F1-Score (train set)")
    ax.set_title("KNN – K Sensitivity")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "02_model_evaluation.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✔ Evaluation chart saved → {path}")


# ─────────────────────────────────────────────────────────────
# 6.  EVALUATION SUMMARY TABLE
# ─────────────────────────────────────────────────────────────
def save_summary_table(results: dict) -> None:
    summary = pd.DataFrame(results).T.reset_index()
    summary.rename(columns={"index": "Model"}, inplace=True)
    path = os.path.join(OUT_DIR, "03_evaluation_summary.csv")
    summary.to_csv(path, index=False)
    print(f"\n  ✔ Summary table saved → {path}")
    print("\n" + summary.to_string(index=False))


# ─────────────────────────────────────────────────────────────
# 7.  OVERFITTING / UNDERFITTING ANALYSIS
# ─────────────────────────────────────────────────────────────
def overfitting_analysis(X_train, X_test, y_train, y_test) -> None:
    print("\n" + "="*60)
    print("  STEP 6 – OVERFITTING vs UNDERFITTING ANALYSIS")
    print("="*60)

    models_info = [
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ("KNN (k=5)",           KNeighborsClassifier(n_neighbors=5)),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Overfitting vs Underfitting Analysis", fontsize=13, fontweight='bold')

    for ax, (name, model) in zip(axes, models_info):
        model.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc  = accuracy_score(y_test,  model.predict(X_test))
        gap       = train_acc - test_acc

        bars = ax.bar(['Train Accuracy', 'Test Accuracy'],
                      [train_acc, test_acc],
                      color=['#42A5F5', '#EF5350'], alpha=0.85, width=0.5)
        for bar, v in zip(bars, [train_acc, test_acc]):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f'{v:.4f}', ha='center', fontsize=11, fontweight='bold')

        diagnosis = "Well-fit" if gap < 0.03 else ("Overfitting" if gap > 0 else "Underfitting")
        ax.set_ylim(0, 1.1)
        ax.set_title(f"{name}\n[{diagnosis}] – Gap: {gap:.4f}")
        ax.set_ylabel("Accuracy")
        ax.axhline(y=0.90, color='green', linestyle='--', alpha=0.5, label='90% line')
        ax.legend()

        print(f"\n  {name}")
        print(f"    Train Accuracy : {train_acc:.4f}")
        print(f"    Test  Accuracy : {test_acc:.4f}")
        print(f"    Gap            : {gap:.4f}  → {diagnosis}")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "04_overfitting_analysis.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  ✔ Overfitting chart saved → {path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "╔" + "═"*58 + "╗")
    print("║  CUSTOMER CHURN PREDICTION – FULL ML PIPELINE           ║")
    print("╚" + "═"*58 + "╝")

    df = load_data(DATA_PATH)
    run_eda(df)
    X_train, X_test, y_train, y_test, feature_names = preprocess(df)
    results, trained = train_and_evaluate(X_train, X_test, y_train, y_test)
    plot_evaluation(results, trained, y_test, X_test)
    save_summary_table(results)
    overfitting_analysis(X_train, X_test, y_train, y_test)

    print("\n" + "="*60)
    print("  ✅  PIPELINE COMPLETE – all outputs saved to /outputs/")
    print("="*60 + "\n")
