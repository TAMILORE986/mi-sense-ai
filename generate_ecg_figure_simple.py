import os
import wfdb
import matplotlib.pyplot as plt
import numpy as np
import glob

# === SET YOUR PATHS ===
PTBXL_ROOT = r"C:\Users\USER\Downloads\MI_Project\ptb-xl"

# Find all .hea files inside records100 (and subfolders)
hea_files = glob.glob(os.path.join(PTBXL_ROOT, "records100", "**", "*.hea"), recursive=True)
if not hea_files:
    print("No .hea files found in records100 folder.")
    print("Make sure the path is correct and the folder contains .hea files.")
    exit(1)

# Pick the first two files as examples (you can change the indices later)
record_path1 = hea_files[0].replace('.hea', '')   # MI candidate
record_path2 = hea_files[1].replace('.hea', '')   # Non-MI candidate

print(f"Using first record: {record_path1}")
print(f"Using second record: {record_path2}")

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

plot_12lead(record_path1, "ECG Record 1 (could be MI)", axes[:, 0])
plot_12lead(record_path2, "ECG Record 2 (could be Non-MI)", axes[:, 1])

for col in range(2):
    axes[-1, col].set_xlabel("Time (seconds)")

output_file = "figure_4_1_ecg_side_by_side.png"
plt.savefig(output_file, dpi=200, bbox_inches='tight')
plt.show()
print(f"\n✓ Figure saved as '{output_file}'")
print("If the first record does not show ST elevation, change the indices in the script.")
