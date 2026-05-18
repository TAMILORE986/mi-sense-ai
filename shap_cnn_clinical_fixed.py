import shap
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

print("Loading model...")
full_model = tf.keras.models.load_model('mi_detection_model_full.h5')

print("Loading clinical test data...")
X_clinical_test = np.load('X_clinical_test_scaled.npy')
print(f"Test samples: {X_clinical_test.shape[0]} features: {X_clinical_test.shape[1]}")

# Create a dummy ECG input (all zeros). The shape is expected by the first input of the model.
# The model has two inputs: ecg_input and clinical_input. We'll fix ecg_input.
ecg_input_shape = full_model.input[0].shape[1:]   # e.g., (12, 1000) or (12, time_steps)
fixed_ecg = np.zeros((1,) + ecg_input_shape)      # batch size 1
print(f"Using fixed ECG of shape {fixed_ecg.shape}")

# Prediction function: takes clinical features (2D array) and uses the fixed ECG.
def predict_clinical_only(clinical_x):
    n_samples = clinical_x.shape[0]
    ecg_batch = np.repeat(fixed_ecg, n_samples, axis=0)   # same dummy ECG for all samples
    preds = full_model.predict([ecg_batch, clinical_x.astype(np.float32)], verbose=0)
    return preds.flatten()

# Background dataset (50 random samples)
background = X_clinical_test[np.random.choice(X_clinical_test.shape[0], 50, replace=False)]

print("Initialising KernelExplainer...")
explainer = shap.KernelExplainer(predict_clinical_only, background)

# Samples to explain (100 random samples)
X_to_explain = X_clinical_test[np.random.choice(X_clinical_test.shape[0], 100, replace=False)]
print("Computing SHAP values (this may take a few minutes)...")
shap_values = explainer.shap_values(X_to_explain, nsamples=100)

# Feature names
try:
    clinical_df = pd.read_csv('full_integrated_clinical_data.csv')
    drop_cols = [c for c in ['record_id', 'ecg_id'] if c in clinical_df.columns]
    if drop_cols:
        clinical_df = clinical_df.drop(columns=drop_cols)
    feature_names = list(clinical_df.columns)
except:
    feature_names = [f'Feature_{i+1}' for i in range(X_clinical_test.shape[1])]

print("Generating SHAP bar plot...")
shap.summary_plot(shap_values, X_to_explain, feature_names=feature_names, plot_type="bar", show=False)
fig = plt.gcf()
fig.set_size_inches(10, 8)
plt.tight_layout()
plt.savefig('shap_cnn_clinical.png', dpi=300, bbox_inches='tight')
plt.show()
print("Figure 4.12 saved as 'shap_cnn_clinical.png'")