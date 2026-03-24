# Sanskrit Diffusion Client - Final Package

This folder is a ready-to-run package for:

1. Running the UI (`app.py`)
2. Running inference (`inference.py`)
3. Running all 5 analysis tasks (single model or all ablation models)
4. Deploying to Hugging Face (Model + Space)

It is prepared so a non-technical user can follow step-by-step commands.

## What Is Included

- Core code: `app.py`, `inference.py`, `train.py`, `config.py`
- Model code: `model/`, `diffusion/`, `data/`
- Analysis code and reports: `analysis/`
- Ablation checkpoints: `ablation_results/T4,T8,T16,T32,T64`
- Ablation configs: `ablation_configs/`
- Tokenizers: `sanskrit_src_tokenizer.json`, `sanskrit_tgt_tokenizer.json`
- Deployment scripts: `upload_hf_model.py`, `upload_hf_space.py`, `deploy_hf.sh`

## Quick Start (Mac/Linux)

From inside this `final_folder`:

```bash
chmod +x setup_local.sh run_ui.sh run_inference_cli.sh run_all_tasks.sh
./setup_local.sh
./run_ui.sh
```

Then open: `http://127.0.0.1:7860`

## Quick Inference (CLI)

```bash
./run_inference_cli.sh "dharmo rakṣati rakṣitaḥ"
```

## Train Model (Shell Scripts)

Single run:

```bash
./train_model.sh d3pm_cross_attention False
```

Run all 4 default combinations:

```bash
./train_all_models.sh
```

## Run All Tasks on All Ablation Models

```bash
./run_all_tasks.sh
```

Outputs are saved in:

- `analysis/outputs_ablation/T4`
- `analysis/outputs_ablation/T8`
- `analysis/outputs_ablation/T16`
- `analysis/outputs_ablation/T32`
- `analysis/outputs_ablation/T64`

## Run Single Task Manually

Example (Task 2 on T4):

```bash
.venv/bin/python analysis/run_analysis.py \
  --task 2 \
  --checkpoint ablation_results/T4/best_model.pt \
  --output_dir analysis/outputs_ablation/T4 \
  --input "dharmo rakṣati rakṣitaḥ"
```

## Deploy to Hugging Face

1. Login:
```bash
huggingface-cli login
```

2. Upload model:
```bash
.venv/bin/python upload_hf_model.py \
  --repo-id <hf_username>/<model_repo_name> \
  --checkpoint ablation_results/T4/best_model.pt
```

3. Upload space:
```bash
.venv/bin/python upload_hf_space.py \
  --repo-id <hf_username>/<space_repo_name>
```

## Non-Technical Notes

- If UI says "load model first", click **Refresh Models** then **Load Selected Model**.
- Best starting checkpoint for stable output is usually `T4` or `T16`.
- If internet is slow/offline, dataset-related tasks may fall back to cached data.
