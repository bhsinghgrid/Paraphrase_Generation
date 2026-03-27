from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

from env_utils import load_local_env


ROOT = Path(__file__).resolve().parent
load_local_env(__file__)


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def stage_model_export(checkpoint: Path, repo_id: str) -> Path:
    tmp_dir = Path(tempfile.mkdtemp(prefix="hf_model_stage_"))

    copy_if_exists(checkpoint, tmp_dir / "best_model.pt")
    for name in [
        "config.py",
        "inference.py",
        "requirements.txt",
        "env_utils.py",
        "sanskrit_src_tokenizer.json",
        "sanskrit_tgt_tokenizer.json",
    ]:
        copy_if_exists(ROOT / name, tmp_dir / name)

    for folder in ["model", "diffusion"]:
        src = ROOT / folder
        if src.is_dir():
            shutil.copytree(src, tmp_dir / folder, dirs_exist_ok=True)

    reports_src = ROOT / "analysis" / "outputs_all_models_20260325"
    if reports_src.is_dir():
        shutil.copytree(reports_src, tmp_dir / "analysis_reports" / reports_src.name, dirs_exist_ok=True)

    readme = f"""---
license: mit
language:
- sa
- en
tags:
- sanskrit
- diffusion
- d3pm
- pytorch
---

# Sanskrit D3PM Model Package

This model repo was exported from `final_folder`.

## Included

- `best_model.pt`
- `config.py`
- `inference.py`
- `model/`
- `diffusion/`
- tokenizers
- compact analysis reports

## Local Usage

```bash
python inference.py --model best_model.pt --cli
```

## Repo

`{repo_id}`
"""
    (tmp_dir / "README.md").write_text(readme, encoding="utf-8")
    return tmp_dir


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Upload model files from final_folder to a Hugging Face model repo.")
    p.add_argument("--repo-id", required=True, help="Target HF model repo id")
    p.add_argument(
        "--checkpoint",
        default=os.getenv("HF_MODEL_CHECKPOINT", "ablation_results/T4/best_model.pt"),
        help="Path to checkpoint file",
    )
    p.add_argument("--token", default=os.getenv("HF_TOKEN"), help="HF token")
    p.add_argument("--private", action="store_true", help="Create repo as private")
    p.add_argument("--message", default="Upload model package from final_folder", help="Commit message")
    return p


def main() -> None:
    args = build_parser().parse_args()
    checkpoint = (ROOT / args.checkpoint).resolve() if not os.path.isabs(args.checkpoint) else Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    staged = stage_model_export(checkpoint, args.repo_id)
    api = HfApi(token=args.token) if args.token else HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=str(staged),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=args.message,
    )
    print(f"Uploaded model to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
