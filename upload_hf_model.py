"""
upload_hf_model.py
==================

Package and upload the Sanskrit D3PM checkpoint to a Hugging Face model repo.

This is for the current project architecture:
  - custom `best_model.pt` checkpoint
  - project config/tokenizers
  - local inference/runtime code

Example:
    ./.venv/bin/python upload_hf_model.py \
        --repo-id your-username/sanskrit-iast-devanagari-d3pm \
        --checkpoint results8/d3pm_cross_attention_neg_False/best_model.pt

If you already ran `huggingface-cli login`, the token is optional.
Otherwise pass `--token hf_...`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

from config import CONFIG
from export_hf_transformers import export_package


ROOT = Path(__file__).resolve().parent


def find_checkpoint(explicit: str | None) -> Path:
    if explicit:
        ckpt = Path(explicit)
        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        return ckpt

    for base in ["results9", "results8", "results7", "results6", "results"]:
        base_path = ROOT / base
        if not base_path.is_dir():
            continue
        for subdir in sorted(base_path.iterdir(), reverse=True):
            if not subdir.is_dir():
                continue
            for name in ["best_model.pt", "best_val_model.pt"]:
                ckpt = subdir / name
                if ckpt.is_file():
                    return ckpt
    raise FileNotFoundError("Could not auto-detect a checkpoint. Pass --checkpoint explicitly.")


def copy_if_exists(src: Path, dst: Path):
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def write_model_card(dst: Path, repo_id: str, ckpt_path: Path):
    model_name = repo_id.split("/")[-1]
    text = f"""---
library_name: pytorch
tags:
  - sanskrit
  - transliteration
  - devanagari
  - diffusion
  - d3pm
---

# {model_name}

This repository contains the exported checkpoint and runtime files for the Sanskrit
IAST -> Devanagari D3PM cross-attention model from this project.

## Included Files

- `best_model.pt`: primary demo checkpoint
- `best_val_model.pt`: best validation-loss checkpoint (if available)
- `quality_predictor.pt`: Task 5 quality predictor (if available)
- `project_config.json`: serialized training/inference config
- `sanskrit_src_tokenizer_v1000.json`
- `sanskrit_tgt_tokenizer_v2000.json`
- `inference.py`
- `models/`
- `diffusion/`

## Local Load Example

```bash
git clone https://huggingface.co/{repo_id}
cd {model_name}
python inference.py --model best_model.pt --cli
```

## Python Example

```python
from huggingface_hub import snapshot_download
repo_dir = snapshot_download("{repo_id}")
print("Downloaded to:", repo_dir)
```

The checkpoint exported here came from:

`{ckpt_path.as_posix()}`
"""
    dst.write_text(text)


def stage_export(repo_id: str, checkpoint: Path) -> Path:
    exp_dir = checkpoint.parent
    tmp_dir = Path(tempfile.mkdtemp(prefix="hf_export_", dir=str(ROOT / "analysis" / "outputs" if (ROOT / "analysis" / "outputs").exists() else ROOT)))
    export_package(tmp_dir, checkpoint)
    copy_if_exists(exp_dir / "best_val_model.pt", tmp_dir / "best_val_model.pt")
    copy_if_exists(exp_dir / "quality_predictor.pt", tmp_dir / "quality_predictor.pt")
    copy_if_exists(exp_dir / "summary.txt", tmp_dir / "summary.txt")
    copy_if_exists(exp_dir / "split_metadata.json", tmp_dir / "split_metadata.json")
    copy_if_exists(ROOT / "pyproject.toml", tmp_dir / "pyproject.toml")
    (tmp_dir / "project_config.json").write_text(json.dumps(CONFIG, indent=2))
    write_model_card(tmp_dir / "README.md", repo_id=repo_id, ckpt_path=checkpoint)
    return tmp_dir


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Upload this Sanskrit D3PM model to Hugging Face.")
    p.add_argument(
        "--repo-id",
        required=True,
        help="Target Hugging Face model repo, e.g. your-username/sanskrit-iast-devanagari-d3pm",
    )
    p.add_argument("--checkpoint", default=None, help="Path to best_model.pt. Auto-detected if omitted.")
    p.add_argument("--token", default=os.getenv("HF_TOKEN"), help="HF token. Uses HF_TOKEN env var if omitted.")
    p.add_argument("--private", action="store_true", help="Create the model repo as private.")
    p.add_argument("--message", default="Upload Sanskrit D3PM model export", help="Commit message for the upload.")
    return p


def main():
    args = build_parser().parse_args()
    checkpoint = find_checkpoint(args.checkpoint)
    export_dir = stage_export(args.repo_id, checkpoint)

    api = HfApi(token=args.token) if args.token else HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=str(export_dir),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=args.message,
    )
    print(f"Uploaded model to https://huggingface.co/{args.repo_id}")
    print(f"Staged export: {export_dir}")


if __name__ == "__main__":
    main()
