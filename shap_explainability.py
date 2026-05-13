
import numpy as np
import pandas as pd
import tensorflow as tf
import shap
import matplotlib.pyplot as plt
import os

def run_shap_explainability():
    print("Loading test data and trained model...")
    # Load test data saved by the training script
    try:
        X_ecg_test = np.load('X_ecg_test.npy')
        X_clinical_test = np.load('X_clinical_test_scaled.npy')
        y_test = np.load('y_test.npy')
        model = tf.keras.models.load_model('mi_detection_model_full.h5')
    except FileNotFoundError:
        print("Error: Required files not found. Please ensure you have run 'train_model_full.py' successfully.")
        return

    # Select a small number of samples for explanation (e.g., 5 samples)
    # SHAP is computationally expensive, so we use a small number for the demonstration.
    num_samples_to_explain = min(5, X_ecg_test.shape[0])
    sample_indices = np.random.choice(X_ecg_test.shape[0], num_samples_to_explain, replace=False)
    
    X_ecg_sample = X_ecg_test[sample_indices]
    X_clinical_sample = X_clinical_test[sample_indices]

    print(f"Explaining {num_samples_to_explain} samples using SHAP...")

    # Define a wrapper function for the model's prediction that SHAP can use
    def model_predict_wrapper(clinical_data):
        # We focus SHAP on the clinical features for clear visualization
        # We repeat the ECG data to match the number of clinical samples being explained
        num_clinical = clinical_data.shape[0]
        ecg_repeated = np.repeat(X_ecg_sample[0:1], num_clinical, axis=0)
        return model.predict([ecg_repeated, clinical_data])

    # Use a subset of the clinical data as background for the explainer
    background_indices = np.random.choice(X_clinical_test.shape[0], 50, replace=False)
    X_clinical_background = X_clinical_test[background_indices]

    # Initialize the SHAP explainer
    explainer = shap.KernelExplainer(model_predict_wrapper, X_clinical_background)

    # Calculate SHAP values for the clinical features
    print("Calculating SHAP values (this may take a few minutes)...")
    shap_values = explainer.shap_values(X_clinical_sample)

    # Visualize the results
    print("Generating SHAP summary plot...")
    clinical_feature_names = pd.read_csv('full_integrated_clinical_data.csv').columns.tolist()
    
    # Create the summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_clinical_sample, feature_names=clinical_feature_names, show=False)
    plt.title("SHAP Feature Importance: Clinical Data")
    plt.tight_layout()
    plt.savefig('shap_summary_clinical.png')
    plt.close()
    
    print("\n" + "="*40)
    print("SHAP Explainability complete!")
    print("File generated: shap_summary_clinical.png")
    print("="*40)

if __name__ == '__main__':
    run_shap_explainability()
