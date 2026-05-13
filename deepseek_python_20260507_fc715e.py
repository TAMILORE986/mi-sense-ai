import numpy as np

# Generate random 12-lead ECG (10 seconds, 100Hz = 1000 samples)
ecg_data = np.random.randn(1000, 12)

# Scale to realistic mV range (-2 to 2)
ecg_data = ecg_data * 0.5

# Save as .npy file
np.save('12_lead_ecg.npy', ecg_data)

print("✅ Created: 12_lead_ecg.npy")
print(f"Shape: {ecg_data.shape} (1000 time points × 12 leads)")