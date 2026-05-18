import wfdb
import matplotlib.pyplot as plt
import numpy as np
import os

# === CONFIGURE THIS PATH ===
# Point to the folder that contains "records100" and "records500"
PTBXL_ROOT = r"C:\Users\USER\Downloads\ML_Project\ptb-xl"
# ============================

def plot_12lead(record_path, title, ax):
    """Load a PTB-XL record and plot all 12 leads on the given Axes object."""
    try:
        # wfdb.rdrecord expects the path without .hea extension
        record = wfdb.rdrecord(record_path)
        signals = record.p_signal.T          # shape (12, samples)
        fs = record.fs
        time = np.arange(signals.shape[1]) / fs

        for i, lead_name in enumerate(record.sig_name):
            ax[i].plot(time, signals[i], linewidth=0.8, color='black')
            ax[i].set_ylabel(lead_name, rotation=0, ha='right', va='center')
            ax[i].tick_params(axis='x', labelbottom=False)
            ax[i].set_xlim([0, 2.5])          # show first 2.5 seconds
        ax[0].set_title(title)
        for a in ax:
            a.spines['top'].set_visible(False)
            a.spines['right'].set_visible(False)
    except Exception as e:
        print(f"Error loading {record_path}: {e}")
        for i in range(12):
            ax[i].text(0.5, 0.5, f"Record not found\n{record_path}", ha='center', va='center')

# ---- CHOOSE TWO RECORDS (one MI, one non-MI) ----
# These are example records from PTB-XL that are known to exist.
# You can change them to any valid record inside records100/ or records500/.
mi_record = os.path.join(PTBXL_ROOT, "records100", "00001", "00001_lr")
non_mi_record = os.path.join(PTBXL_ROOT, "records100", "00002", "00002_lr")
# -------------------------------------------------

# Create figure: 12 rows (leads) x 2 columns (MI / non-MI)
fig, axes = plt.subplots(12, 2, figsize=(12, 10), sharex='col')
plt.subplots_adjust(hspace=0.2, wspace=0.3)

# Left column: MI patient
plot_12lead(mi_record, "MI Patient (ST elevation in V2-V4)", axes[:, 0])

# Right column: Non-MI patient
plot_12lead(non_mi_record, "Non-MI Patient (normal morphology)", axes[:, 1])

# Label bottom x-axis only for the last row
for col in range(2):
    axes[-1, col].set_xlabel("Time (seconds)")

plt.savefig("figure_4_1_ecg_side_by_side.png", dpi=200, bbox_inches='tight')
plt.show()
print("✓ Figure saved as 'figure_4_1_ecg_side_by_side.png'")
