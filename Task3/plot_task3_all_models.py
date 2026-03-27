import os
import matplotlib.pyplot as plt

base_dir = '/Users/bhsingh/Documents/Final_Paraphrase/Exclude_Negative/final_folder/analysis/outputs_all_models_20260325'
models = ['T4', 'T8', 'T16', 'T32', 'T64']

variance_explained = []
correlations = []

for m in models:
    fpath = os.path.join(base_dir, m, 'task3_report.txt')
    var = 0.0
    corr = 0.0
    if os.path.exists(fpath):
        with open(fpath, 'r') as f:
            for line in f:
                if 'PCA:' in line:
                    try:
                        # Extract "75.9%"
                        var_str = line.split(', ')[1].split('%')[0]
                        var = float(var_str)
                    except: pass
                if 'Diversity PC:' in line:
                    try:
                        # Extract "0.344"
                        corr_str = line.split('|r|=')[1].split(' ')[0]
                        corr = abs(float(corr_str))
                    except: pass
    variance_explained.append(var)
    correlations.append(corr)

fig, ax1 = plt.subplots(figsize=(8, 5))

color = 'tab:blue'
ax1.set_xlabel('Diffusion Steps (Model)')
ax1.set_ylabel('Max PC Correlation (|r|)', color=color)
bars = ax1.bar(models, correlations, color=color, alpha=0.7)
ax1.set_ylim(0, 0.6)
ax1.tick_params(axis='y', labelcolor=color)

for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.3f}", ha='center', va='bottom', color=color, fontweight='bold')

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Explained Variance (%)', color=color)
ax2.plot(models, variance_explained, color=color, marker='o', linewidth=2)
ax2.set_ylim(60, 90)
ax2.tick_params(axis='y', labelcolor=color)

for i, txt in enumerate(variance_explained):
    ax2.text(i, txt + 1.0, f"{txt:.1f}%", ha='center', color=color, fontweight='bold')

plt.title('Task 3: All Models Comparison (PCA Variance & Diversity Correlation)')
fig.tight_layout()
plt.savefig('/Users/bhsingh/Documents/Final_Paraphrase/Exclude_Negative/final_folder/Task3/task3_all_models_comparison.png', dpi=300)
print("Saved task3_all_models_comparison.png")
