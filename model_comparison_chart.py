import matplotlib.pyplot as plt
import numpy as np

# Data from your table
models = ['Multimodal CNN-LSTM', 'XGBoost (Clinical)', 'ECG-Only Baseline', 'Clinical-Only Baseline']
metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1-Score', 'AUC']

# Values for each model (in %)
data = {
    'Multimodal CNN-LSTM':   [87.4, 84.6, 88.2, 72.3, 78.0, 93.3],
    'XGBoost (Clinical)':    [85.2, 79.8, 86.9, 67.4, 73.1, 89.0],
    'ECG-Only Baseline':     [83.1, 77.2, 84.8, 64.1, 70.1, 87.1],
    'Clinical-Only Baseline':[81.6, 72.4, 83.7, 59.8, 65.5, 83.2]
}

# Set up positions
x = np.arange(len(metrics))
width = 0.2
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

fig, ax = plt.subplots(figsize=(12, 7))

# Plot bars
for i, model in enumerate(models):
    offset = (i - 1.5) * width
    ax.bar(x + offset, data[model], width, label=model, color=colors[i])

# Labels and formatting
ax.set_xlabel('Evaluation Metric', fontsize=12)
ax.set_ylabel('Score (%)', fontsize=12)
ax.set_title('Model Performance Comparison (Figure 4.5)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax.set_ylim(50, 100)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on bars
for i, model in enumerate(models):
    offset = (i - 1.5) * width
    for j, val in enumerate(data[model]):
        ax.text(x[j] + offset, val + 1, f'{val:.1f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('model_comparison_chart.png', dpi=300, bbox_inches='tight')
plt.show()
print("Figure 4.5 saved as 'model_comparison_chart.png'")