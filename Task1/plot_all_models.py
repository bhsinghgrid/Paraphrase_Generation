import os
import matplotlib.pyplot as plt

base_dir = '/Users/bhsingh/Documents/Final_Paraphrase/Exclude_Negative/final_folder/analysis/outputs_all_models_20260325'
models = ['T4', 'T8', 'T16', 'T32', 'T64']

speedups_64 = []
encoder_costs = []

for m in models:
    fpath = os.path.join(base_dir, m, 'task1_kv_cache.txt')
    if os.path.exists(fpath):
        with open(fpath, 'r') as f:
            lines = f.readlines()
            # Extract 64-token speedup
            sp_64 = 1.0
            enc = 0.0
            for line in lines:
                if line.strip().startswith('64'):
                    parts = line.split()
                    try:
                        sp_64 = float(parts[3].replace('x', ''))
                        enc = float(parts[4].replace('%', ''))
                    except:
                        pass
            speedups_64.append(sp_64)
            encoder_costs.append(enc)
    else:
        speedups_64.append(0)
        encoder_costs.append(0)

fig, ax1 = plt.subplots(figsize=(8, 5))

color = 'tab:blue'
ax1.set_xlabel('Diffusion Steps (Model)')
ax1.set_ylabel('Speedup at 64 Tokens (x)', color=color)
bars = ax1.bar(models, speedups_64, color=color, alpha=0.7)
ax1.tick_params(axis='y', labelcolor=color)

# Add values on bars
for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}x", ha='center', va='bottom', color=color, fontweight='bold')

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Encoder Cost (%)', color=color)
ax2.plot(models, encoder_costs, color=color, marker='o', linewidth=2)
ax2.tick_params(axis='y', labelcolor=color)

for i, txt in enumerate(encoder_costs):
    ax2.text(i, txt + 1.0, f"{txt:.1f}%", ha='center', color=color, fontweight='bold')

plt.title('Task 1: All Models Comparison (Speedup & Encoder Cost)')
fig.tight_layout()
plt.savefig('/Users/bhsingh/Documents/Final_Paraphrase/Exclude_Negative/final_folder/Task1/task1_all_models_comparison.png', dpi=300)
print("Saved task1_all_models_comparison.png")
