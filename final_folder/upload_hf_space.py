from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parent


def stage_space(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    copies = [
        ("app.py", "app.py"),
        ("inference.py", "inference.py"),
        ("config.py", "config.py"),
        ("requirements-space.txt", "requirements.txt"),
        ("README-space.md", "README.md"),
        ("sanskrit_src_tokenizer_v1000.json", "sanskrit_src_tokenizer_v1000.json"),
        ("sanskrit_tgt_tokenizer_v2000.json", "sanskrit_tgt_tokenizer_v2000.json"),
    ]
    for src_name, dst_name in copies:
        shutil.copy2(ROOT / src_name, out_dir / dst_name)

    shutil.copytree(ROOT / "models", out_dir / "models", dirs_exist_ok=True)
    shutil.copytree(ROOT / "diffusion", out_dir / "diffusion", dirs_exist_ok=True)
    analysis_src = ROOT / "results8" / "d3pm_cross_attention_neg_False" / "analysis"
    if analysis_src.is_dir():
        shutil.copytree(analysis_src, out_dir / "analysis_outputs", dirs_exist_ok=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Upload Gradio Space for Sanskrit transliteration.")
    p.add_argument("--repo-id", required=True, help="HF Space repo id, e.g. user/space-name")
    p.add_argument("--token", default=None, help="HF token with write permission")
    return p


def main():
    args = build_parser().parse_args()
    api = HfApi(token=args.token)
    api.create_repo(args.repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="hf_space_") as tmpdir:
        staged = Path(tmpdir)
        stage_space(staged)
        api.upload_folder(
            folder_path=str(staged),
            repo_id=args.repo_id,
            repo_type="space",
        )
        print(f"Uploaded space to https://huggingface.co/spaces/{args.repo_id}")


if __name__ == "__main__":
    main()
