# Results Index

All generated analysis outputs are inside:

- `analysis/outputs_ablation/T4/`
- `analysis/outputs_ablation/T8/`
- `analysis/outputs_ablation/T16/`
- `analysis/outputs_ablation/T32/`
- `analysis/outputs_ablation/T64/`

For each `T*` folder, you get:

- Task 1: `task1_kv_cache.txt` + 3 plots
- Task 2: `task2_report.txt` + attention/drift/TF-IDF plots
- Task 3: `task3_report.txt` + concept/PCA/diversity plots
- Task 4: `task4_report.txt` + `task4_3d.png`
- Task 5: `task5_report.txt` + quality-diversity plot

If any folder is missing Task 3, run:

```bash
.venv/bin/python analysis/run_analysis.py \
  --task 3 \
  --task3_samples 120 \
  --checkpoint ablation_results/T64/best_model.pt \
  --output_dir analysis/outputs_ablation/T64
```
