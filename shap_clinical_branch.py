import shap
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
import matplotlib.pyplot as plt

print("Loading model...")
full_model = tf.keras.models.load_model('mi_detection_model_full.h5')

print("Loading clinical test data...")
X_clinical_test = np.load('X_clinical_test_scaled.npy')
y_test = np.load('y_test.npy')
print(f"Test samples: {X_clinical_test.shape[0]}")

# Extract clinical sub‑model
clinical_input_idx = full_model.input_names.index('clinical_input')
clinical_submodel = Model(inputs=full_model.input[clinical_input_idx],
                          outputs=full_model.output)

# Background and explainer
background = X_clinical_test[np.random.choice(X_clinical_test.shape[0], 50, replace=False)]
def f(x):
    return clinical_submodel.predict(x.astype(np.float32), verbose=0).flatten()

print("Initialising KernelExplainer (this may take a minute)...")
explainer = shap.KernelExplainer(f, background)

X_to_explain = X_clinical_test[np.random.choice(X_clinical_test.shape[0], 100, replace=False)]
print("Computing SHAP values...")
shap_values = explainer.shap_values(X_to_explain, nsamples=100)

# Feature names
try:
    clinical_df = pd.read_csv('full_integrated_clinical_data.csv')
    drop_cols = [c for c in ['record_id', 'ecg_id'] if c in clinical_df.columns]
    if drop_cols:
        clinical_df = clinical_df.drop(columns=drop_cols)
    feature_names = list(clinical_df.columns)
except:
    feature_names = [f'Feature {i+1}' for i in range(X_clinical_test.shape[1])]

# Bar plot
shap.summary_plot(shap_values, X_to_explain, feature_names=feature_names, plot_type="bar", show=False)
fig = plt.gcf()
fig.set_size_inches(10, 8)
plt.tight_layout()
plt.savefig('shap_cnn_clinical.png', dpi=300, bbox_inches='tight')
plt.show()
print("Figure 4.12 saved as 'shap_cnn_clinical.png'")
