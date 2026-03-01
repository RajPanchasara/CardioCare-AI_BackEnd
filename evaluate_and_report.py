"""
CardioCare AI - Complete Model Evaluation & Report Generator
=============================================================
Generates ALL required academic evaluation artifacts:
  1. Problem Statement & Dataset Summary
  2. Class Distribution Analysis
  3. Preprocessing Documentation
  4. Cross-Validation (5-fold)
  5. Multi-Model Comparison Table
  6. Hyperparameter Tuning (GridSearchCV)
  7. Final Model Evaluation (Classification Report, Confusion Matrix)
  8. Overfitting / Underfitting Check
  9. Performance Graphs:
     - Confusion Matrix Heatmap
     - ROC Curve
     - Precision-Recall Curve
     - Accuracy Comparison Bar Chart
     - Feature Importance
     - Learning Curve

Run:  python evaluate_and_report.py
Output saved to:  Weekly_Task/outputs/
"""

import os
import sys
import warnings
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, GridSearchCV,
    learning_curve
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB

warnings.filterwarnings('ignore')

# ─── Paths ────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEEKLY_DIR = os.path.join(BASE_DIR, 'Weekly_Task')
OUTPUT_DIR = os.path.join(WEEKLY_DIR, 'outputs')
CSV_PATH = os.path.join(WEEKLY_DIR, 'cardio_train.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')

os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_NAMES = [
    'age', 'gender', 'height', 'weight',
    'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
    'smoke', 'alco', 'active'
]

# ─── Plot Style ────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.grid': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.size': 11,
    'figure.dpi': 150,
})


def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  > Saved: {name}")


# ══════════════════════════════════════════════════════
# 1. LOAD & PREPARE DATA
# ══════════════════════════════════════════════════════
print("\n" + "="*60)
print("CardioCare AI - Complete Evaluation Report")
print("="*60)

print("\n[1/9] Loading dataset...")
df = pd.read_csv(CSV_PATH, sep=';')

# Age: days -> years
if df['age'].mean() > 200:
    df['age'] = (df['age'] / 365.25).astype(int)

# Drop ID column if present
if 'id' in df.columns:
    df = df.drop('id', axis=1)

print(f"  Dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"  Target: 'cardio' (0 = No disease, 1 = Disease)")
print(f"  Features: {FEATURE_NAMES}")

# ── Problem Statement & Summary ──
summary = {
    'problem_statement': (
        'Cardiovascular disease (CVD) is the leading cause of death worldwide. '
        'This project builds a Machine Learning model to predict the presence '
        'of cardiovascular disease based on patient examination data including '
        'age, blood pressure, cholesterol, glucose levels, and lifestyle factors.'
    ),
    'dataset': {
        'name': 'Cardiovascular Disease Dataset (Kaggle)',
        'rows': int(df.shape[0]),
        'columns': int(df.shape[1]),
        'target': 'cardio (binary: 0/1)',
        'features': {
            'age': 'Patient age in years',
            'gender': 'Gender (1=Female, 2=Male)',
            'height': 'Height in cm',
            'weight': 'Weight in kg',
            'ap_hi': 'Systolic blood pressure',
            'ap_lo': 'Diastolic blood pressure',
            'cholesterol': 'Cholesterol level (1=Normal, 2=Above Normal, 3=Well Above)',
            'gluc': 'Glucose level (1=Normal, 2=Above Normal, 3=Well Above)',
            'smoke': 'Smoking status (0=No, 1=Yes)',
            'alco': 'Alcohol intake (0=No, 1=Yes)',
            'active': 'Physical activity (0=No, 1=Yes)',
        },
    },
    'summary_statistics': json.loads(df.describe().to_json()),
}

with open(os.path.join(OUTPUT_DIR, 'dataset_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print("  > Saved: dataset_summary.json")


# ══════════════════════════════════════════════════════
# 2. CLASS DISTRIBUTION
# ══════════════════════════════════════════════════════
print("\n[2/9] Class distribution analysis...")
class_counts = df['cardio'].value_counts()
print(f"  Class 0 (No CVD): {class_counts[0]:,} ({class_counts[0]/len(df)*100:.1f}%)")
print(f"  Class 1 (CVD):    {class_counts[1]:,} ({class_counts[1]/len(df)*100:.1f}%)")
balance_ratio = class_counts.min() / class_counts.max()
print(f"  Balance ratio: {balance_ratio:.3f} ({'Balanced' if balance_ratio > 0.8 else 'Imbalanced'})")

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(['No CVD (0)', 'CVD (1)'], class_counts.values,
              color=['#4CAF50', '#F44336'], edgecolor='white', width=0.5)
for bar, count in zip(bars, class_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 300,
            f'{count:,}\n({count/len(df)*100:.1f}%)', ha='center', fontweight='bold')
ax.set_title('Class Distribution - Target Variable (cardio)', fontweight='bold')
ax.set_ylabel('Count')
save(fig, 'class_distribution.png')


# ══════════════════════════════════════════════════════
# 3. PREPARE FEATURES
# ══════════════════════════════════════════════════════
print("\n[3/9] Preparing features...")
X = df[FEATURE_NAMES].values
y = df['cardio'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"  Train: {X_train.shape[0]:,} samples")
print(f"  Test:  {X_test.shape[0]:,} samples")
print(f"  Scaling: StandardScaler (zero mean, unit variance)")


# ══════════════════════════════════════════════════════
# 4. MULTI-MODEL COMPARISON
# ══════════════════════════════════════════════════════
print("\n[4/9] Training and comparing models...")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree':       DecisionTreeClassifier(max_depth=10, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Linear SVM':          CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=42)),
    'Naive Bayes':         GaussianNB(),
}

results = []
for name, mdl in models.items():
    mdl.fit(X_train_scaled, y_train)
    y_pred = mdl.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results.append({
        'Model': name,
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1-Score': round(f1, 4),
    })
    print(f"  {name:<25} Acc={acc:.4f}  F1={f1:.4f}")

results_df = pd.DataFrame(results).sort_values('F1-Score', ascending=False)
results_df.to_csv(os.path.join(OUTPUT_DIR, 'model_comparison.csv'), index=False)
print("  > Saved: model_comparison.csv")

# ── Accuracy Comparison Bar Chart ──
fig, ax = plt.subplots(figsize=(10, 5))
colors = ['#F44336' if r['Model'] == 'Gradient Boosting' else '#90CAF9'
          for _, r in results_df.iterrows()]
bars = ax.barh(results_df['Model'], results_df['Accuracy'], color=colors, edgecolor='white')
for bar, acc in zip(bars, results_df['Accuracy']):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
            f'{acc:.4f}', va='center', fontweight='bold', fontsize=10)
ax.set_xlabel('Accuracy')
ax.set_title('Model Accuracy Comparison', fontweight='bold')
ax.set_xlim(0.5, max(results_df['Accuracy']) + 0.05)
ax.invert_yaxis()
save(fig, 'accuracy_comparison.png')


# ══════════════════════════════════════════════════════
# 5. CROSS-VALIDATION (5-Fold)
# ══════════════════════════════════════════════════════
print("\n[5/9] Cross-validation (5-fold stratified)...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}
for name, mdl in models.items():
    scores = cross_val_score(mdl, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    cv_results[name] = {
        'mean': round(scores.mean(), 4),
        'std': round(scores.std(), 4),
        'folds': [round(s, 4) for s in scores],
    }
    print(f"  {name:<25} Mean={scores.mean():.4f} ± {scores.std():.4f}")

with open(os.path.join(OUTPUT_DIR, 'cross_validation.json'), 'w') as f:
    json.dump(cv_results, f, indent=2)
print("  > Saved: cross_validation.json")


# ══════════════════════════════════════════════════════
# 6. HYPERPARAMETER TUNING (GradientBoosting)
# ══════════════════════════════════════════════════════
print("\n[6/9] Hyperparameter tuning (GridSearchCV on GradientBoosting)...")

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.1, 0.2],
}

grid = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=0
)
grid.fit(X_train_scaled, y_train)

best_params = grid.best_params_
best_score = grid.best_score_
print(f"  Best params: {best_params}")
print(f"  Best CV F1:  {best_score:.4f}")

tuning_results = {
    'best_params': best_params,
    'best_cv_f1': round(best_score, 4),
    'all_results': [
        {'params': r['params'], 'mean_f1': round(r['mean_test_score'], 4)}
        for r in sorted(grid.cv_results_['params'].__class__(
            [dict(zip(grid.cv_results_['params'][0].keys(),
                       [grid.cv_results_[f'param_{k}'][i] for k in grid.cv_results_['params'][0].keys()]))
             for i in range(len(grid.cv_results_['params']))]),
            key=lambda x: 0)  # placeholder
    ] if False else []  # simplified
}
tuning_results = {
    'best_params': best_params,
    'best_cv_f1': round(best_score, 4),
}
with open(os.path.join(OUTPUT_DIR, 'hyperparameter_tuning.json'), 'w') as f:
    json.dump(tuning_results, f, indent=2)
print("  > Saved: hyperparameter_tuning.json")


# ══════════════════════════════════════════════════════
# 7. FINAL MODEL EVALUATION (Deployed Model)
# ══════════════════════════════════════════════════════
print("\n[7/9] Evaluating deployed model (model.pkl)...")

deployed_model = joblib.load(MODEL_PATH)
deployed_scaler = joblib.load(SCALER_PATH)

X_test_deployed = deployed_scaler.transform(X_test)
y_pred = deployed_model.predict(X_test_deployed)
y_proba = deployed_model.predict_proba(X_test_deployed)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")

# ── Classification Report ──
report = classification_report(y_test, y_pred, target_names=['No CVD', 'CVD'], output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(os.path.join(OUTPUT_DIR, 'classification_report.csv'))
print("  > Saved: classification_report.csv")

# ── Confusion Matrix Heatmap ──
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=['No CVD', 'CVD'],
            yticklabels=['No CVD', 'CVD'], ax=ax, cbar_kws={'shrink': 0.8},
            annot_kws={'size': 16, 'fontweight': 'bold'})
ax.set_xlabel('Predicted', fontweight='bold')
ax.set_ylabel('Actual', fontweight='bold')
ax.set_title('Confusion Matrix - Deployed Model', fontweight='bold')
save(fig, 'confusion_matrix.png')

# ── ROC Curve ──
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(7, 6))
ax.fill_between(fpr, tpr, alpha=0.15, color='#F44336')
ax.plot(fpr, tpr, color='#F44336', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4, label='Random Baseline')
ax.set_xlabel('False Positive Rate', fontweight='bold')
ax.set_ylabel('True Positive Rate', fontweight='bold')
ax.set_title('ROC Curve - Deployed Model', fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
save(fig, 'roc_curve.png')

# ── Precision-Recall Curve ──
prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)
ap = average_precision_score(y_test, y_proba)

fig, ax = plt.subplots(figsize=(7, 6))
ax.fill_between(rec_curve, prec_curve, alpha=0.15, color='#2196F3')
ax.plot(rec_curve, prec_curve, color='#2196F3', lw=2.5,
        label=f'PR Curve (AP = {ap:.4f})')
ax.set_xlabel('Recall', fontweight='bold')
ax.set_ylabel('Precision', fontweight='bold')
ax.set_title('Precision-Recall Curve - Deployed Model', fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.set_xlim(0, 1.02)
ax.set_ylim(0, 1.05)
save(fig, 'precision_recall_curve.png')

# ── Feature Importance ──
if hasattr(deployed_model, 'feature_importances_'):
    importances = deployed_model.feature_importances_
    sorted_idx = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors_fi = ['#F44336' if i >= len(FEATURE_NAMES)-3 else '#BBDEFB'
                 for i in range(len(sorted_idx))]
    ax.barh([FEATURE_NAMES[i] for i in sorted_idx], importances[sorted_idx],
            color=['#F44336' if importances[sorted_idx[j]] >= np.sort(importances)[-3]
                   else '#BBDEFB' for j in range(len(sorted_idx))],
            edgecolor='white')
    for j, idx in enumerate(sorted_idx):
        ax.text(importances[idx] + 0.002, j,
                f'{importances[idx]:.3f}', va='center', fontsize=9)
    ax.set_xlabel('Importance (Gini)', fontweight='bold')
    ax.set_title('Feature Importance - GradientBoosting', fontweight='bold')
    save(fig, 'feature_importance.png')
else:
    print("  [!] Model does not have feature_importances_")


# ══════════════════════════════════════════════════════
# 8. OVERFITTING / UNDERFITTING CHECK
# ══════════════════════════════════════════════════════
print("\n[8/9] Overfitting / underfitting check...")

y_train_pred = deployed_model.predict(deployed_scaler.transform(X_train))
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy

print(f"  Training Accuracy: {train_acc:.4f}")
print(f"  Testing Accuracy:  {test_acc:.4f}")
print(f"  Gap:               {train_acc - test_acc:.4f}")

if train_acc > 0.90 and test_acc < train_acc - 0.10:
    fit_status = "OVERFITTING"
elif train_acc < 0.70 and test_acc < 0.70:
    fit_status = "UNDERFITTING"
else:
    fit_status = "WELL FITTED"

print(f"  Verdict: {fit_status}")

fit_report = {
    'training_accuracy': round(train_acc, 4),
    'testing_accuracy': round(test_acc, 4),
    'gap': round(train_acc - test_acc, 4),
    'verdict': fit_status,
}
with open(os.path.join(OUTPUT_DIR, 'overfitting_check.json'), 'w') as f:
    json.dump(fit_report, f, indent=2)
print("  > Saved: overfitting_check.json")

# ── Learning Curve ──
print("  Generating learning curve (this may take a minute)...")
train_sizes, train_scores, val_scores = learning_curve(
    GradientBoostingClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
    ),
    X_train_scaled, y_train,
    train_sizes=np.linspace(0.1, 1.0, 8),
    cv=3, scoring='accuracy', n_jobs=-1
)

fig, ax = plt.subplots(figsize=(8, 5))
ax.fill_between(train_sizes, train_scores.mean(axis=1) - train_scores.std(axis=1),
                train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1, color='#F44336')
ax.fill_between(train_sizes, val_scores.mean(axis=1) - val_scores.std(axis=1),
                val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1, color='#2196F3')
ax.plot(train_sizes, train_scores.mean(axis=1), 'o-', color='#F44336', lw=2, label='Training')
ax.plot(train_sizes, val_scores.mean(axis=1), 'o-', color='#2196F3', lw=2, label='Validation')
ax.set_xlabel('Training Set Size', fontweight='bold')
ax.set_ylabel('Accuracy', fontweight='bold')
ax.set_title('Learning Curve - GradientBoosting', fontweight='bold')
ax.legend(fontsize=10)
ax.set_ylim(0.6, 1.0)
save(fig, 'learning_curve.png')


# ══════════════════════════════════════════════════════
# 9. FINAL SUMMARY
# ══════════════════════════════════════════════════════
print("\n[9/9] Writing final summary...")

final_summary = {
    'selected_model': 'GradientBoostingClassifier',
    'model_file': 'model.pkl',
    'scaler_file': 'scaler.pkl',
    'model_version': 'v1.0',
    'dataset_size': int(df.shape[0]),
    'feature_count': len(FEATURE_NAMES),
    'test_metrics': {
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'roc_auc': round(roc_auc, 4),
    },
    'cross_validation': cv_results.get('Gradient Boosting', {}),
    'hyperparameter_tuning': tuning_results,
    'overfitting_check': fit_report,
    'best_model_comparison': results,
}

with open(os.path.join(OUTPUT_DIR, 'final_summary.json'), 'w') as f:
    json.dump(final_summary, f, indent=2)
print("  > Saved: final_summary.json")


# ══════════════════════════════════════════════════════
print("\n" + "="*60)
print("EVALUATION COMPLETE")
print("="*60)
print(f"\nAll outputs saved to: {OUTPUT_DIR}")
print("\nGenerated files:")
for f_name in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f_name))
    print(f"  - {f_name} ({size:,} bytes)")
print()
