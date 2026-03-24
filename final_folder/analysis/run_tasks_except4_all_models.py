"""
Run Tasks 1,2,3,5 for every available checkpoint (excluding Task 4).

Usage:
  python analysis/run_tasks_except4_all_models.py
  python analysis/run_tasks_except4_all_models.py --input "dharmo rakṣati rakṣitaḥ"
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = ROOT / "analysis" / "outputs_multi"


def discover_checkpoints() -> list[Path]:
    roots = [ROOT / "results", ROOT / "results7", ROOT / "ablation_results"]
    out: list[Path] = []
    for base in roots:
        if not base.exists():
            continue
        for ckpt in sorted(base.glob("*/best_model.pt")):
            out.append(ckpt)
    return out


def slug_for_checkpoint(ckpt: Path) -> str:
    root = ckpt.parent.parent.name
    exp = ckpt.parent.name
    return f"{root}__{exp}"


def run_task(task: str, ckpt: Path, input_text: str, out_dir: Path) -> tuple[int, float]:
    cmd = [
        sys.executable,
        str(ROOT / "analysis" / "run_analysis.py"),
        "--task", task,
        "--checkpoint", str(ckpt),
        "--output_dir", str(out_dir),
    ]
    if task == "2":
        cmd.extend(["--input", input_text])

    start = datetime.now()
    env = os.environ.copy()
    env.setdefault("HF_HOME", "/tmp/hf_home")
    env.setdefault("HF_DATASETS_CACHE", "/tmp/hf_datasets")
    env.setdefault("HF_HUB_CACHE", "/tmp/hf_hub")
    env.setdefault("TRANSFORMERS_CACHE", "/tmp/hf_transformers")
    os.makedirs(env["HF_HOME"], exist_ok=True)
    os.makedirs(env["HF_DATASETS_CACHE"], exist_ok=True)
    os.makedirs(env["HF_HUB_CACHE"], exist_ok=True)
    os.makedirs(env["TRANSFORMERS_CACHE"], exist_ok=True)

    proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
    seconds = (datetime.now() - start).total_seconds()
    return proc.returncode, seconds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="dharmo rakṣati rakṣitaḥ")
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    args = parser.parse_args()

    checkpoints = discover_checkpoints()
    if not checkpoints:
        raise FileNotFoundError("No checkpoints found under results/results7/ablation_results.")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    tasks = ["1", "2", "3", "5"]
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "tasks": tasks,
        "checkpoints": [],
    }

    for ckpt in checkpoints:
        slug = slug_for_checkpoint(ckpt)
        model_out = out_root / slug
        model_out.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Checkpoint: {ckpt} ===")
        model_item = {
            "checkpoint": str(ckpt),
            "output_dir": str(model_out),
            "tasks": [],
        }

        for task in tasks:
            task_out = model_out / f"task{task}"
            task_out.mkdir(parents=True, exist_ok=True)
            print(f"-> Running task {task} ...")
            code, sec = run_task(task, ckpt, args.input, task_out)
            item = {
                "task": task,
                "exit_code": code,
                "seconds": round(sec, 2),
                "output_dir": str(task_out),
            }
            model_item["tasks"].append(item)
            status = "OK" if code == 0 else "FAILED"
            print(f"   {status} ({sec:.1f}s)")

        summary["checkpoints"].append(model_item)

    summary_path = out_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
