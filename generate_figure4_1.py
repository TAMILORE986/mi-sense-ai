import os
import pandas as pd
import wfdb
import matplotlib.pyplot as plt
import numpy as np

# === PATHS ===
PTBXL_ROOT = r"C:\Users\USER\Downloads\ML_Project\ptb-xl"
# Use the correct CSV filename (ptbx1_database.csv, not ptbxl_database.csv)
CSV_PATH = os.path.join(PTBXL_ROOT, "ptbx1_database.csv")

# === LOAD DATABASE ===
df = pd.read_csv(CSV_PATH)   # fixed parentheses

# Find one MI record (diagnostic_class = 'MI')
mi_row = df[df['diagnostic_class'] == 'MI'].iloc[0]
mi_filename = mi_row['filename_hr']      # or 'filename_lr' – both work
mi_full_path = os.path.join(PTBXL_ROOT, mi_filename)

# Find one normal (non-MI) record (diagnostic_class = 'NORM')
norm_row = df[df['diagnostic_class'] == 'NORM'].iloc[0]
norm_filename = norm_row['filename_hr']
norm_full_path = os.path.join(PTBXL_ROOT, norm_filename)

print(f"Using MI record: {mi_filename}")
print(f"Using Normal record: {norm_filename}")

# === FUNCTION TO PLOT 12 LEADS ===
def plot_12lead(record_path, title, ax):
    record = wfdb.rdrecord(record_path)
    signals = record.p_signal.T
    fs = record.fs
    time = np.arange(signals.shape[1]) / fs
    for i, lead_name in enumerate(record.sig_name):
        ax[i].plot(time, signals[i], linewidth=0.8, color='black')
        ax[i].set_ylabel(lead_name, rotation=0, ha='right', va='center')
        ax[i].tick_params(axis='x', labelbottom=False)
        ax[i].set_xlim([0, 2.5])
    ax[0].set_title(title)
    for a in ax:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)

# === CREATE FIGURE ===
fig, axes = plt.subplots(12, 2, figsize=(12, 10), sharex='col')
plt.subplots_adjust(hspace=0.2, wspace=0.3)

plot_12lead(mi_full_path, "MI Patient (ST elevation in V2-V4)", axes[:, 0])
plot_12lead(norm_full_path, "Non-MI Patient (normal morphology)", axes[:, 1])

for col in range(2):
    axes[-1, col].set_xlabel("Time (seconds)")

output_file = "figure_4_1_ecg_side_by_side.png"
plt.savefig(output_file, dpi=200, bbox_inches='tight')
plt.show()
print(f"\n✓ Figure saved as '{output_file}'")
