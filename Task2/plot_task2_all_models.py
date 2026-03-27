import os
import matplotlib.pyplot as plt

base_dir = '/Users/bhsingh/Documents/Final_Paraphrase/Exclude_Negative/final_folder/analysis/outputs_all_models_20260325'
models = ['T4', 'T8', 'T16', 'T32', 'T64']

semantic_scores = []
tfidf_corrs = []

for m in models:
    fpath = os.path.join(base_dir, m, 'task2_report.txt')
    score = 0.0
    corr = 0.0
    if os.path.exists(fpath):
        with open(fpath, 'r') as f:
            for line in f:
                if 'Multi-sample semantic score' in line:
                    try:
                        score = float(line.split(': ')[1].strip())
                    except: pass
                if 'TF-IDF vs attention stability corr' in line:
                    try:
                        corr = float(line.split(': ')[1].strip())
                    except: pass
    semantic_scores.append(score)
    tfidf_corrs.append(corr)

fig, ax1 = plt.subplots(figsize=(8, 5))

color = 'tab:blue'
ax1.set_xlabel('Diffusion Steps (Model)')
ax1.set_ylabel('TF-IDF vs Attention Corr', color=color)
bars = ax1.bar(models, tfidf_corrs, color=color, alpha=0.7)
ax1.tick_params(axis='y', labelcolor=color)

for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.4f}", ha='center', va='bottom', color=color, fontweight='bold')

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Semantic Score', color=color)
ax2.plot(models, semantic_scores, color=color, marker='o', linewidth=2)
ax2.tick_params(axis='y', labelcolor=color)

for i, txt in enumerate(semantic_scores):
    ax2.text(i, txt + 0.005, f"{txt:.4f}", ha='center', color=color, fontweight='bold')

plt.title('Task 2: All Models Comparison (TF-IDF Corr & Semantic Score)')
fig.tight_layout()
plt.savefig('/Users/bhsingh/Documents/Final_Paraphrase/Exclude_Negative/final_folder/Task2/task2_all_models_comparison.png', dpi=300)
print("Saved task2_all_models_comparison.png")
