# Sanskrit Diffusion Project

This folder is the shareable project package for:
- training D3PM Sanskrit models
- running inference locally
- running Tasks 1 to 5
- preparing deployment to Hugging Face model repos and Spaces

It is designed so another user can clone this folder on a new laptop, create a virtual environment, fill in `.env`, and run the project without editing code.

## What Is Included

- `app.py`: Gradio demo UI
- `inference.py`: inference and cleanup logic
- `train.py`: model training entrypoint
- `analysis/`: code for Tasks 1 to 5
- `ablation_configs/`: diffusion-step ablation configs
- `ablation_results/`: local checkpoints if available on your machine
- `model/`, `diffusion/`, `data/`: core model and data code
- `Task1/` to `Task5/`: prepared reports and figures
- `upload_hf_model.py`, `upload_hf_space.py`: Hugging Face upload helpers
-  `sh folder/`,: inside this folder contain the all the `.sh` by which you run whole project in easy way first remove all file from this folder into the original folder

## Before You Start

1. Use Python `3.11` if possible.
2. Copy `.env.example` to `.env`.
3. Fill in the values you need, especially:
   - `HF_TOKEN`
   - `HF_DEFAULT_MODEL_REPO`
   - `HF_MODEL_CHECKPOINT`
   - `MODEL_TYPE`
   - `INCLUDE_NEG`

## Quick Local Setup

```bash
cd final_folder
./setup_local.sh
```

This will:
- create `.venv` if needed
- install dependencies
- create `.env` from `.env.example` if missing

## Run The UI

```bash
./run_ui.sh
```

By default this uses:
- `GRADIO_SERVER_PORT` from `.env`, else `7860`
- local cache folders:
  - `.hf_cache/`
  - `.mplconfig/`

## Run CLI Inference

```bash
./run_inference_cli.sh
```

Or directly:

```bash
.venv/bin/python inference.py --cli
```

## Train Models

Single run:

```bash
./train_model.sh d3pm_cross_attention False
./train_model.sh d3pm_encoder_decoder False
```

Full standard matrix:

```bash
./train_all_models.sh
```

## Run Tasks 1 To 5

For the default ablation root from `.env` or shell:

```bash
./run_all_tasks.sh
```

For encoder-decoder only:

```bash
MODEL_ROOT=ablation_results/encoder_decoder \
OUT_ROOT=analysis/outputs_all_models_20260325/encoder_decoder \
bash analysis/run_all_ablation_tasks.sh
```

For cross-attention only:

```bash
MODEL_ROOT=ablation_results \
OUT_ROOT=analysis/outputs_all_models_20260325 \
bash run_all_tasks.sh
```

## GitHub Upload

This folder is now prepared for GitHub as source code.

Ignored automatically:
- `.env`
- `.venv/`
- `.hf_cache/`
- `.mplconfig/`
- IDE files
- Python cache files
- generated outputs
- local logs

Important:
- large checkpoints are not ideal for normal GitHub commits
- use Hugging Face model repos or Git LFS for `.pt` files
- if another user does not have local checkpoints, they should download them from Hugging Face or place them under `ablation_results/`

## Hugging Face Deployment

### Model upload

```bash
.venv/bin/python upload_hf_model.py \
  --repo-id your-username/your-model-repo \
  --checkpoint ablation_results/T4/best_model.pt
```

If `HF_TOKEN` is present in `.env`, it will be used automatically.

### Space upload

```bash
.venv/bin/python upload_hf_space.py \
  --repo-id your-username/your-space-repo
```

### Easy combined deployment

```bash
./deploy_hf_easy.sh your-username your-project-name
```

This uses:
- `.env`
- `HF_TOKEN` if present
- `HF_MODEL_CHECKPOINT` for model upload

## How Another Person Should Run This On Their Laptop

1. Clone the repository.
2. Enter `final_folder/`.
3. Run:

```bash
./setup_local.sh
```

4. Copy `.env.example` to `.env` and edit values.
5. If checkpoints are not present locally:
   - download them from your Hugging Face model repo, or
   - place them under `ablation_results/`
6. Start the UI:

```bash
./run_ui.sh
```

7. Run analysis if needed:

```bash
./run_all_tasks.sh
```

## Notes

- Current inference uses cleanup and fallback logic in `inference.py`.
- Some checkpoints may produce noisy raw output; the final displayed output may come from cleanup/fallback.
- For demo use, keep the Hugging Face model repo as the source of truth for released checkpoints.

## Main Files

- [app.py](/Users/bhsingh/Documents/Final_Paraphrase/Exclude_Negative/final_folder/app.py)
- [inference.py](/Users/bhsingh/Documents/Final_Paraphrase/Exclude_Negative/final_folder/inference.py)
- [train.py](/Users/bhsingh/Documents/Final_Paraphrase/Exclude_Negative/final_folder/train.py)
- [analysis/run_analysis.py](/Users/bhsingh/Documents/Final_Paraphrase/Exclude_Negative/final_folder/analysis/run_analysis.py)
- [run_all_tasks.sh](/Users/bhsingh/Documents/Final_Paraphrase/Exclude_Negative/final_folder/run_all_tasks.sh)
- [analysis/run_all_ablation_tasks.sh](/Users/bhsingh/Documents/Final_Paraphrase/Exclude_Negative/final_folder/analysis/run_all_ablation_tasks.sh)
- [deploy_hf_easy.sh](/Users/bhsingh/Documents/Final_Paraphrase/Exclude_Negative/final_folder/deploy_hf_easy.sh)
