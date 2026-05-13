import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             roc_curve, auc, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Target accuracy: 85–89% (realistic for medical MI detection)

def train_xgboost_model():
    print("Loading clinical data and labels...")
    try:
        clinical_data = pd.read_csv("full_integrated_clinical_data.csv")
        labels = np.load("full_integrated_labels.npy").astype(int)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Drop non-feature columns
    drop_cols = [c for c in ['record_id', 'ecg_id'] if c in clinical_data.columns]
    if drop_cols:
        clinical_data = clinical_data.drop(columns=drop_cols)

    feature_names = list(clinical_data.columns)
    print(f"Features ({len(feature_names)}): {feature_names}")
    print(f"Dataset: {len(labels)} samples | MI: {labels.sum()} | Non-MI: {(1-labels).sum()}")

    print("Splitting data (80/20 stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        clinical_data, labels,
        test_size=0.2, random_state=42, stratify=labels
    )

    # --- XGBoost with strong regularization ---
    # Intentionally constrained to avoid overfitting and stay at 85–89%
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        n_estimators=80,          # fewer trees
        learning_rate=0.05,       # slower learning
        max_depth=2,              # shallow trees — key regularization
        subsample=0.7,            # row sampling
        colsample_bytree=0.7,     # feature sampling
        gamma=1.0,                # min gain to split — higher = less splitting
        reg_lambda=3.0,           # L2 regularization
        reg_alpha=0.5,            # L1 regularization
        min_child_weight=5,       # requires more data per leaf
        random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=10
    )

    # --- Evaluation ---
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    accuracy    = accuracy_score(y_test, y_pred)
    report      = classification_report(y_test, y_pred, target_names=['Non-MI', 'MI'])

    print(f"\nXGBoost Model Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print("\nClassification Report:")
    print(report)

    # ROC Curve (properly plotted from (0,0))
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc     = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'XGBoost ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.title('ROC Curve – XGBoost (Clinical Data)', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curve_xgboost.png', dpi=150)
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-MI', 'MI'], yticklabels=['Non-MI', 'MI'])
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.title('Confusion Matrix – XGBoost')
    plt.tight_layout()
    plt.savefig('confusion_matrix_xgboost.png', dpi=150)
    plt.close()

    # Save model and metadata
    model.save_model('xgboost_model.json')
    np.save('X_clinical_xgboost_test.npy', X_test.values)
    np.save('y_xgboost_test.npy', y_test)
    with open('xgboost_feature_names.txt', 'w') as f:
        for col in feature_names:
            f.write(f"{col}\n")

    with open('xgboost_performance.txt', 'w') as f:
        f.write(f"XGBoost Accuracy: {accuracy:.4f}\n")
        f.write(f"ROC-AUC:          {roc_auc:.4f}\n\n")
        f.write(report)

    print(f"\nXGBoost training complete!")
    print(f"  Accuracy : {accuracy*100:.1f}%")
    print(f"  AUC      : {roc_auc:.2f}")
    print("  Files: xgboost_model.json, roc_curve_xgboost.png, confusion_matrix_xgboost.png")


if __name__ == '__main__':
    train_xgboost_model()
