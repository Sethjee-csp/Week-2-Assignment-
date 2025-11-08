"""
Credit Approval Prediction with Logistic Regression
Uses synthetic dataset from sklearn.datasets.make_classification
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve, accuracy_score,
    precision_score, recall_score, f1_score
)

# Set random seed for reproducibility
np.random.seed(42)

# Create output directory
OUTPUT_DIR = Path('outputs')
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*60)
print("CREDIT APPROVAL PREDICTION - LOGISTIC REGRESSION")
print("="*60)

# ============================================
# 1. DATASET GENERATION
# ============================================
print("\n1. Generating Synthetic Dataset...")

# Generate synthetic classification data
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=8,
    n_redundant=2,
    n_classes=2,
    weights=[0.6, 0.4],  # Imbalanced classes
    flip_y=0.05,
    random_state=42
)

# Create meaningful feature names
feature_names = [
    'income', 'credit_score', 'debt_ratio', 'employment_years',
    'num_credit_lines', 'loan_amount', 'age', 'savings',
    'feature_8', 'feature_9'
]

# Create DataFrame
df = pd.DataFrame(X, columns=feature_names)
df['approved'] = y

# Add a categorical feature (employment type)
employment_types = np.random.choice(['Full-Time', 'Part-Time', 'Self-Employed'], size=1000)
df['employment_type'] = employment_types

# Introduce some missing values randomly (only in features, not target)
missing_mask = np.random.random((df.shape[0], df.shape[1]-1)) < 0.05
df_features = df.drop('approved', axis=1)
df_features = df_features.mask(missing_mask)
df = pd.concat([df_features, df[['approved']]], axis=1)

print(f"Dataset shape: {df.shape}")
print(f"Target distribution:\n{df['approved'].value_counts()}")
print(f"Missing values:\n{df.isnull().sum()}")

# ============================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================
print("\n2. Performing EDA...")

# Class distribution
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Target distribution
axes[0, 0].bar(['Rejected (0)', 'Approved (1)'], df['approved'].value_counts().sort_index())
axes[0, 0].set_title('Credit Approval Distribution')
axes[0, 0].set_ylabel('Count')

# Feature distributions
df['income'].hist(bins=30, ax=axes[0, 1], edgecolor='black')
axes[0, 1].set_title('Income Distribution')
axes[0, 1].set_xlabel('Income')

df['credit_score'].hist(bins=30, ax=axes[1, 0], edgecolor='black')
axes[1, 0].set_title('Credit Score Distribution')
axes[1, 0].set_xlabel('Credit Score')

# Boxplot for debt ratio
df.boxplot(column='debt_ratio', by='approved', ax=axes[1, 1])
axes[1, 1].set_title('Debt Ratio by Approval Status')
axes[1, 1].set_xlabel('Approved')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'eda_distributions.png', dpi=300, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'eda_distributions.png'}")
plt.close()

# Correlation heatmap
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'correlation_heatmap.png'}")
plt.close()

# ============================================
# 3. PREPROCESSING
# ============================================
print("\n3. Preprocessing Data...")

# Separate features and target
X = df.drop('approved', axis=1)
y = df['approved']

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = ['employment_type']

print(f"Numeric features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# ============================================
# 4. MODEL DEVELOPMENT
# ============================================
print("\n4. Training Logistic Regression Model...")

# Create complete pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(
        C=1.0,  # Regularization strength
        class_weight='balanced',  # Handle class imbalance
        max_iter=1000,
        random_state=42,
        solver='lbfgs'
    ))
])

# Fit the model
model_pipeline.fit(X_train, y_train)
print("Model training completed!")

# Extract model coefficients
model = model_pipeline.named_steps['classifier']
coefficients = model.coef_[0]

# Get feature names after preprocessing
feature_names_after = (
    numeric_features +
    list(model_pipeline.named_steps['preprocessor']
         .named_transformers_['cat']
         .named_steps['onehot']
         .get_feature_names_out(categorical_features))
)

# Calculate odds ratios
odds_ratios = np.exp(coefficients)

# Create coefficient dataframe
coef_df = pd.DataFrame({
    'Feature': feature_names_after,
    'Coefficient': coefficients,
    'Odds_Ratio': odds_ratios
}).sort_values('Coefficient', key=abs, ascending=False)

print("\nTop 10 Features by Coefficient Magnitude:")
print(coef_df.head(10).to_string(index=False))

# Visualize coefficients
plt.figure(figsize=(12, 8))
top_features = coef_df.head(15)
colors = ['green' if x > 0 else 'red' for x in top_features['Coefficient']]
plt.barh(range(len(top_features)), top_features['Coefficient'], color=colors)
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('Coefficient Value')
plt.title('Top 15 Feature Coefficients (Logistic Regression)')
plt.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'feature_coefficients.png', dpi=300, bbox_inches='tight')
print(f"\nSaved: {OUTPUT_DIR / 'feature_coefficients.png'}")
plt.close()

# Visualize odds ratios
plt.figure(figsize=(12, 8))
top_or = coef_df.head(15).copy()
colors = ['green' if x > 1 else 'red' for x in top_or['Odds_Ratio']]
plt.barh(range(len(top_or)), top_or['Odds_Ratio'], color=colors)
plt.yticks(range(len(top_or)), top_or['Feature'])
plt.xlabel('Odds Ratio')
plt.title('Top 15 Feature Odds Ratios')
plt.axvline(x=1, color='black', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'odds_ratios.png', dpi=300, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'odds_ratios.png'}")
plt.close()

# ============================================
# 5. MODEL EVALUATION
# ============================================
print("\n5. Evaluating Model Performance...")

# Predictions
y_pred = model_pipeline.predict(X_test)
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "="*60)
print("MODEL PERFORMANCE METRICS")
print("="*60)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print("="*60)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Rejected', 'Approved'],
            yticklabels=['Rejected', 'Approved'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
print(f"\nSaved: {OUTPUT_DIR / 'confusion_matrix.png'}")
plt.close()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'roc_curve.png', dpi=300, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'roc_curve.png'}")
plt.close()

# ============================================
# 6. THRESHOLD ANALYSIS
# ============================================
print("\n6. Threshold Analysis...")

# Analyze different thresholds
thresholds_to_test = [0.3, 0.4, 0.5, 0.6, 0.7]
threshold_results = []

for threshold in thresholds_to_test:
    y_pred_threshold = (y_pred_proba >= threshold).astype(int)
    acc = accuracy_score(y_test, y_pred_threshold)
    prec = precision_score(y_test, y_pred_threshold)
    rec = recall_score(y_test, y_pred_threshold)
    f1_t = f1_score(y_test, y_pred_threshold)

    threshold_results.append({
        'Threshold': threshold,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1_t
    })

threshold_df = pd.DataFrame(threshold_results)
print("\nThreshold Analysis Results:")
print(threshold_df.to_string(index=False))

# Plot threshold analysis
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].plot(threshold_df['Threshold'], threshold_df['Accuracy'], marker='o')
axes[0, 0].set_title('Accuracy vs Threshold')
axes[0, 0].set_xlabel('Threshold')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].grid(alpha=0.3)

axes[0, 1].plot(threshold_df['Threshold'], threshold_df['Precision'], marker='o', color='green')
axes[0, 1].set_title('Precision vs Threshold')
axes[0, 1].set_xlabel('Threshold')
axes[0, 1].set_ylabel('Precision')
axes[0, 1].grid(alpha=0.3)

axes[1, 0].plot(threshold_df['Threshold'], threshold_df['Recall'], marker='o', color='orange')
axes[1, 0].set_title('Recall vs Threshold')
axes[1, 0].set_xlabel('Threshold')
axes[1, 0].set_ylabel('Recall')
axes[1, 0].grid(alpha=0.3)

axes[1, 1].plot(threshold_df['Threshold'], threshold_df['F1'], marker='o', color='red')
axes[1, 1].set_title('F1 Score vs Threshold')
axes[1, 1].set_xlabel('Threshold')
axes[1, 1].set_ylabel('F1 Score')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'threshold_analysis.png', dpi=300, bbox_inches='tight')
print(f"\nSaved: {OUTPUT_DIR / 'threshold_analysis.png'}")
plt.close()

# Precision-Recall tradeoff
plt.figure(figsize=(8, 6))
plt.plot(threshold_df['Threshold'], threshold_df['Precision'],
         marker='o', label='Precision', color='green')
plt.plot(threshold_df['Threshold'], threshold_df['Recall'],
         marker='s', label='Recall', color='orange')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision-Recall Tradeoff')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'precision_recall_tradeoff.png', dpi=300, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'precision_recall_tradeoff.png'}")
plt.close()

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print(f"All outputs saved to: {OUTPUT_DIR.absolute()}")
print("="*60)
print("\nKey Insights:")
print(f"1. Model achieved ROC-AUC of {roc_auc:.4f}")
print(f"2. At default threshold (0.5): Precision={precision:.4f}, Recall={recall:.4f}")
print("3. For fintech: Consider higher threshold (0.6-0.7) to reduce false positives")
print("4. Review top features with highest odds ratios for business interpretation")
print("="*60)