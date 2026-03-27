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


def stage_space(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in [
        "app.py",
        "config.py",
        "inference.py",
        "requirements.txt",
        "env_utils.py",
        "sanskrit_src_tokenizer.json",
        "sanskrit_tgt_tokenizer.json",
    ]:
        shutil.copy2(ROOT / name, out_dir / name)

    for folder in ["model", "diffusion", "data"]:
        src = ROOT / folder
        if src.is_dir():
            shutil.copytree(src, out_dir / folder, dirs_exist_ok=True)

    analysis_src = ROOT / "analysis" / "outputs_all_models_20260325"
    if analysis_src.is_dir():
        shutil.copytree(analysis_src, out_dir / "analysis_outputs" / analysis_src.name, dirs_exist_ok=True)

    readme = """---
title: Sanskrit Diffusion Demo
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
---

# Sanskrit Diffusion Demo

Gradio Space packaged from `final_folder`.

Set these Space variables:

- `HF_DEFAULT_MODEL_REPO`
- `HF_DEFAULT_MODEL_FILE`
- `HF_TOKEN` if private access is needed
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Upload Space files from final_folder to a Hugging Face Space repo.")
    p.add_argument("--repo-id", required=True, help="Target HF Space repo id")
    p.add_argument("--token", default=os.getenv("HF_TOKEN"), help="HF token")
    return p


def main() -> None:
    args = build_parser().parse_args()
    api = HfApi(token=args.token) if args.token else HfApi()
    api.create_repo(args.repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="hf_space_stage_") as tmpdir:
        staged = Path(tmpdir)
        stage_space(staged)
        api.upload_folder(folder_path=str(staged), repo_id=args.repo_id, repo_type="space")
    print(f"Uploaded space to https://huggingface.co/spaces/{args.repo_id}")


if __name__ == "__main__":
    main()
