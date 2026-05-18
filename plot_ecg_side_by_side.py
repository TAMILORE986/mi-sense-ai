import wfdb
import matplotlib.pyplot as plt
import numpy as np
import os

# === CONFIGURE THESE PATHS ===
data_path = "data/"          # folder where PTB-XL files are stored (adjust if needed)
record_mi = "records100/00001/00001_lr"      # example MI record – you must change this!
record_non_mi = "records100/00001/00002_lr"  # example Non-MI record – change this!
# =============================

def plot_12lead(record_name, title, ax):
    """Load a PTB-XL record and plot all 12 leads on a given Axes object."""
    try:
        record = wfdb.rdrecord(os.path.join(data_path, record_name))
        signals = record.p_signal.T  # shape (12, samples)
        time = np.arange(signals.shape[1]) / record.fs
        
        for i, lead in enumerate(record.sig_name):
            ax[i].plot(time, signals[i], linewidth=0.8, color='black')
            ax[i].set_ylabel(lead, rotation=0, ha='right', va='center')
            ax[i].tick_params(axis='x', labelbottom=False)
            ax[i].set_xlim([0, 2.5])  # show first 2.5 seconds
        ax[0].set_title(title)
        for a in ax:
            a.spines['top'].set_visible(False)
            a.spines['right'].set_visible(False)
    except Exception as e:
        print(f"Error loading {record_name}: {e}")
        for i in range(12):
            ax[i].text(0.5, 0.5, f"Record not found\n{record_name}", ha='center', va='center')

# Create figure with 12 rows (leads) and 2 columns (MI / Non-MI)
fig, axes = plt.subplots(12, 2, figsize=(12, 10), sharex='col')
plt.subplots_adjust(hspace=0.2, wspace=0.3)

# Plot MI record on left column
plot_12lead(record_mi, "MI Patient (ST elevation in V2-V4)", axes[:, 0])

# Plot Non-MI record on right column
plot_12lead(record_non_mi, "Non-MI Patient (normal morphology)", axes[:, 1])

# Label bottom x-axis only for the last row
for col in range(2):
    axes[-1, col].set_xlabel("Time (seconds)")

plt.savefig("figure_4_1_ecg_side_by_side.png", dpi=200, bbox_inches='tight')
plt.show()
print("Figure saved as 'figure_4_1_ecg_side_by_side.png'")
