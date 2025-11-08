# Credit Approval Prediction with Logistic Regression

## Overview
Binary classification model to predict credit application approval using logistic regression on synthetic data.

## Dataset
**Source**: Generated using `sklearn.datasets.make_classification`

**Parameters**:
- Samples: 1000
- Features: 10 numeric + 1 categorical (employment_type)
- Classes: 2 (Rejected=0, Approved=1)
- Class weights: [0.6, 0.4] (imbalanced)
- Informative features: 8
- Random state: 42

**Features**:
- `income`, `credit_score`, `debt_ratio`, `employment_years`
- `num_credit_lines`, `loan_amount`, `age`, `savings`
- `feature_8`, `feature_9`, `employment_type`

## Project Structure
```
credit-approval/
├── src/
│   └── main.py              # Main script
├── outputs/                  # All generated plots
│   ├── eda_distributions.png
│   ├── correlation_heatmap.png
│   ├── feature_coefficients.png
│   ├── odds_ratios.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── threshold_analysis.png
│   └── precision_recall_tradeoff.png
├── requirements.txt
└── README.md
```


## Methodology

### Preprocessing Pipeline
1. **Train/Test Split**: 80/20 with stratification (random_state=42)
2. **Missing Value Imputation**:
   - Numeric: Median imputation
   - Categorical: Most frequent imputation
3. **Encoding**: OneHotEncoder for employment_type (handle_unknown='ignore')
4. **Scaling**: StandardScaler for all numeric features
5. **Implementation**: ColumnTransformer + Pipeline

### Model Configuration
- **Algorithm**: Logistic Regression
- **Solver**: lbfgs
- **Regularization**: C=1.0 (inverse regularization strength)
- **Class weights**: Balanced (to handle imbalance)
- **Max iterations**: 1000

### Interpretation
- **Coefficients**: Extracted from trained model
- **Odds Ratios**: Calculated as exp(coefficient)
  - Odds ratio > 1: Feature increases approval odds
  - Odds ratio < 1: Feature decreases approval odds
  - Example: If income has odds ratio of 2.5, each unit increase in standardized income multiplies approval odds by 2.5

### Evaluation Metrics
- **Confusion Matrix**: True/False Positives/Negatives
- **Accuracy**: Overall correctness
- **Precision**: Of predicted approvals, how many were correct
- **Recall**: Of actual approvals, how many were caught
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve (discrimination ability)

### Threshold Analysis
Tested thresholds: 0.3, 0.4, 0.5, 0.6, 0.7
- **Lower threshold** (0.3-0.4): Higher recall, more approvals, more false positives
- **Default threshold** (0.5): Balanced precision and recall
- **Higher threshold** (0.6-0.7): Higher precision, fewer false positives, better for risk-averse fintech


## Results Summary
- All visualizations saved in `outputs/` folder
- Model achieves strong discrimination (ROC-AUC typically > 0.85)
- Top features identified through coefficient analysis
- Threshold analysis shows precision-recall tradeoff

## Outputs Generated
1. **eda_distributions.png**: Target distribution, feature histograms
2. **correlation_heatmap.png**: Feature correlation matrix
3. **feature_coefficients.png**: Top 15 feature coefficients
4. **odds_ratios.png**: Top 15 odds ratios
5. **confusion_matrix.png**: Prediction confusion matrix
6. **roc_curve.png**: ROC curve with AUC score
7. **threshold_analysis.png**: Metrics across thresholds
8. **precision_recall_tradeoff.png**: Precision vs recall by threshold

---
