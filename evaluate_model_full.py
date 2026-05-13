import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_full_model():
    print("Loading test data and trained model...")
    X_ecg_test = np.load('X_ecg_test.npy')
    X_clinical_test = np.load('X_clinical_test_scaled.npy')
    y_test = np.load('y_test.npy')

    model = tf.keras.models.load_model('mi_detection_model_full.h5')

    print("Making predictions on the test set...")
    y_pred_prob = model.predict([X_ecg_test, X_clinical_test]).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")

    # --- ROC Curve (FIXED) ---
    # sklearn's roc_curve already handles the (0,0) and (1,1) boundary points
    # DO NOT manually concatenate [0] or [1] — that was causing the broken curve
    print("Generating ROC Curve...")
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.title('Receiver Operating Characteristic (ROC) - Full Dataset', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curve_full.png', dpi=150)
    plt.close()
    print("ROC curve saved as 'roc_curve_full.png'")

    # --- Confusion Matrix ---
    print("Generating Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-MI', 'MI'], yticklabels=['Non-MI', 'MI'])
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title('Confusion Matrix - Full Dataset', fontsize=14)
    plt.tight_layout()
    plt.savefig('confusion_matrix_full.png', dpi=150)
    plt.close()
    print("Confusion matrix saved as 'confusion_matrix_full.png'")

    # --- Classification Report ---
    report = classification_report(y_test, y_pred, target_names=['Non-MI', 'MI'])
    print("\nClassification Report:\n", report)

    with open('performance_report.txt', 'w') as f:
        f.write("--- Final Model Performance Report ---\n")
        f.write(f"Model Accuracy: {accuracy:.4f}\n")
        f.write(f"ROC-AUC Score:  {roc_auc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    print("\nEvaluation complete! Files saved: roc_curve_full.png, confusion_matrix_full.png, performance_report.txt")


if __name__ == '__main__':
    required_files = [
        'X_ecg_test.npy', 'X_clinical_test_scaled.npy',
        'y_test.npy', 'mi_detection_model_full.h5'
    ]
    missing = [f for f in required_files if not os.path.exists(f)]

    if not missing:
        evaluate_full_model()
    else:
        print(f"Error: Missing required files: {missing}")
        print("Please run train_model_full.py first.")
