import copy
import json
import os
import subprocess
import sys
import shutil
import threading
import uuid
import time
from datetime import datetime
from pathlib import Path

import gradio as gr
import torch
from huggingface_hub import hf_hub_download

from env_utils import load_local_env
from config import CONFIG
from inference import (
    _resolve_device,
    load_model,
    run_inference,
    _decode_clean,
    _decode_with_cleanup,
    _iast_to_deva,
    _compute_cer,
)
from model.tokenizer import SanskritSourceTokenizer, SanskritTargetTokenizer

load_local_env(__file__)


RESULTS_DIR = "generated_results"
DEFAULT_ANALYSIS_OUT = "analysis_outputs/outputs_all_models_20260325/T4"
os.makedirs(RESULTS_DIR, exist_ok=True)
_BG_JOBS = {}
_CHECKPOINT_CACHE = None

try:
    import mlflow
except Exception:
    mlflow = None

_MLFLOW_READY = False
FLOW_STEPS = [
    "Start",
    "Load Model (checkpoint/config/device/eval)",
    "Load Tokenizers",
    "Input (IAST)",
    "Source Tokenization",
    "Encoder (run once)",
    "KV Cache prepared",
    "Initialize x_T (MASK)",
    "Diffusion loop (T→0, with Task2/Task3 hooks)",
    "Final x0",
    "Decode to Devanagari",
    "Evaluation/Tasks (Task4/Task5)",
]


def _setup_mlflow_once():
    global _MLFLOW_READY
    if _MLFLOW_READY:
        return
    if mlflow is None:
        return
    try:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:/tmp/mlruns")
        experiment = os.environ.get("MLFLOW_EXPERIMENT_NAME", "hf-space-sanskrit-d3pm")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        _MLFLOW_READY = True
    except Exception:
        _MLFLOW_READY = False


def _mlflow_event(run_name: str, params: dict | None = None, metrics: dict | None = None, tags: dict | None = None):
    _setup_mlflow_once()
    if not _MLFLOW_READY or mlflow is None:
        return
    try:
        with mlflow.start_run(run_name=run_name, nested=False):
            if tags:
                mlflow.set_tags({k: str(v) for k, v in tags.items()})
            if params:
                mlflow.log_params({k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in params.items()})
            if metrics:
                mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
    except Exception:
        pass


def _build_flow_markdown(model_loaded=False, inference_ready=False, task_states=None):
    lines = ["### Execution Flow"]
    task_states = task_states or {}
    any_task_activity = any(v != "pending" for v in task_states.values()) if task_states else False
    for i, step in enumerate(FLOW_STEPS, start=1):
        status = "⬜"
        if model_loaded and i <= 3:
            status = "✅"
        if (inference_ready or model_loaded) and i <= 11:
            status = "✅"
        if i == 12 and any_task_activity:
            status = "✅"
        lines.append(f"{status} {i}. {step}")
    if task_states:
        lines.append("")
        lines.append("### Task Status")
        for k in ["1", "2", "3", "4", "5"]:
            v = task_states.get(k, "pending")
            icon = "✅" if v == "done" else ("🔄" if v.startswith("running") else ("❌" if v == "failed" else "⬜"))
            lines.append(f"{icon} Task {k}: {v}")
    return "\n".join(lines)


def _task5_cfg(lambda_min, lambda_max, lambda_step, task5_samples):
    lo = float(lambda_min)
    hi = float(lambda_max)
    st = max(0.1, float(lambda_step))
    if hi < lo:
        lo, hi = hi, lo
    vals = []
    cur = lo
    while cur <= hi + 1e-9 and len(vals) < 30:
        vals.append(round(cur, 2))
        cur += st
    if not vals:
        vals = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    return {"scales": vals, "samples": max(5, int(task5_samples))}

HF_DEFAULT_MODEL_REPO = os.environ.get("HF_DEFAULT_MODEL_REPO", "bhsinghgrid/DevaFlow")
HF_DEFAULT_MODEL_FILE = os.environ.get("HF_DEFAULT_MODEL_FILE", "best_model.pt")
HF_CHECKPOINT_REPO = os.environ.get("HF_CHECKPOINT_REPO", "bhsinghgrid/devflow2")
HF_CHECKPOINT_FILE = os.environ.get("HF_CHECKPOINT_FILE", "best_model.pt")
HF_MODEL_REPOS = [
    repo.strip()
    for repo in os.environ.get("HF_MODEL_REPOS", "bhsinghgrid/DevaFlow,bhsinghgrid/devflow2").split(",")
    if repo.strip()
]
HF_DEFAULT_MODEL_TYPE = os.environ.get("HF_DEFAULT_MODEL_TYPE", "d3pm_cross_attention")
HF_DEFAULT_INCLUDE_NEG = os.environ.get("HF_DEFAULT_INCLUDE_NEG", "false")
HF_DEFAULT_NUM_STEPS = os.environ.get("HF_DEFAULT_NUM_STEPS")
HF_DEFAULT_MODEL_SETTINGS_FILE = os.environ.get("HF_DEFAULT_MODEL_SETTINGS_FILE", "model_settings.json")


def _download_hf_model_settings():
    try:
        cache_dir = Path(".hf_model_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        settings_path = hf_hub_download(
            repo_id=HF_DEFAULT_MODEL_REPO,
            filename=HF_DEFAULT_MODEL_SETTINGS_FILE,
            local_dir=str(cache_dir),
        )
        with open(settings_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


HF_DEFAULT_SETTINGS = _download_hf_model_settings()


def _repo_cache_dir(repo_id: str) -> Path:
    safe = repo_id.replace("/", "__")
    path = Path(".hf_model_cache") / safe
    path.mkdir(parents=True, exist_ok=True)
    return path


def _download_hf_checkpoint(repo_id: str, filename: str = "best_model.pt"):
    try:
        cache_dir = _repo_cache_dir(repo_id)
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(cache_dir),
        )
    except Exception:
        return None


def _download_hf_settings_for_repo(repo_id: str):
    try:
        cache_dir = _repo_cache_dir(repo_id)
        settings_path = hf_hub_download(
            repo_id=repo_id,
            filename=HF_DEFAULT_MODEL_SETTINGS_FILE,
            local_dir=str(cache_dir),
        )
        with open(settings_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def discover_checkpoints():
    global _CHECKPOINT_CACHE
    if _CHECKPOINT_CACHE is not None:
        return list(_CHECKPOINT_CACHE)
    found = []
    local_roots = [
        ("ablation_results", "cross_attention"),
        (os.path.join("ablation_results", "encoder_decoder"), "encoder_decoder"),
        ("results7", "other"),
        ("results", "other"),
    ]
    for root, family in local_roots:
        if not os.path.isdir(root):
            continue
        for entry in sorted(os.listdir(root)):
            ckpt = os.path.join(root, entry, "best_model.pt")
            if not os.path.exists(ckpt):
                continue
            found.append(
                {
                    "label": f"{entry}  [{root}]",
                    "path": ckpt,
                    "experiment": entry,
                    "root": root,
                    "family": family,
                }
            )
    for repo_id in HF_MODEL_REPOS:
        settings = _download_hf_settings_for_repo(repo_id)
        model_type = settings.get("model_type", "")
        family = "encoder_decoder" if model_type == "d3pm_encoder_decoder" else "cross_attention"
        num_steps = settings.get("num_steps", HF_DEFAULT_NUM_STEPS)
        step_label = f"T{num_steps}" if num_steps else "HF"
        found.append(
            {
                "label": f"{repo_id}  [{family}:{step_label}]",
                "path": None,
                "experiment": step_label,
                "root": "hf",
                "family": family,
                "repo_id": repo_id,
                "repo_file": HF_CHECKPOINT_FILE,
                "hf_settings": settings,
            }
        )
    _CHECKPOINT_CACHE = list(found)
    return list(found)


def _guess_analysis_dir(experiment: str, ckpt_path: str, family: str = "cross_attention", settings: dict | None = None) -> str:
    settings = settings or {}
    base = Path("analysis_outputs")
    packaged = base / "outputs_all_models_20260325"
    step = None
    if experiment and experiment.startswith("T") and experiment[1:].isdigit():
        step = experiment
    elif settings.get("num_steps"):
        step = f"T{int(settings['num_steps'])}"
    else:
        for part in Path(ckpt_path or "").parts:
            if part.startswith("T") and part[1:].isdigit():
                step = part
                break
    if packaged.exists() and step:
        if family == "encoder_decoder" and (packaged / "encoder_decoder" / step).is_dir():
            return str(packaged / "encoder_decoder" / step)
        if (packaged / step).is_dir():
            return str(packaged / step)
    if base.exists():
        if family == "encoder_decoder" and step and (base / "encoder_decoder" / step).is_dir():
            return str(base / "encoder_decoder" / step)
        if step and (base / step).is_dir():
            return str(base / step)
        if (base / "T4").is_dir():
            return str(base / "T4")
    return os.path.join("analysis", "outputs_ui", experiment or "default")


def checkpoint_map():
    return {item["label"]: item for item in discover_checkpoints()}


def default_checkpoint_label():
    cps = discover_checkpoints()
    if not cps:
        return None
    for item in cps:
        path = item.get("path")
        if path and path.endswith("ablation_results/T4/best_model.pt"):
            return item["label"]
    for item in cps:
        if item.get("repo_id") == HF_DEFAULT_MODEL_REPO:
            return item["label"]
    return cps[0]["label"]


def infer_model_type(experiment_name: str, root: str = "") -> str:
    if root == "ablation_results":
        return "d3pm_cross_attention"
    if experiment_name.startswith("d3pm_cross_attention"):
        return "d3pm_cross_attention"
    if experiment_name.startswith("d3pm_encoder_decoder"):
        return "d3pm_encoder_decoder"
    if experiment_name.startswith("baseline_cross_attention"):
        return "baseline_cross_attention"
    if experiment_name.startswith("baseline_encoder_decoder"):
        return "baseline_encoder_decoder"
    return CONFIG["model_type"]


def infer_include_negative(experiment_name: str, root: str = "") -> bool:
    if root == "ablation_results":
        return False
    if "_neg_True" in experiment_name:
        return True
    if "_neg_False" in experiment_name:
        return False
    return CONFIG["data"]["include_negative_examples"]


def build_runtime_cfg(ckpt_path: str, item: dict | None = None):
    item = item or {}
    if item.get("root") == "hf":
        experiment = item.get("experiment", "hf")
        root = "hf"
        hf_settings = item.get("hf_settings", {})
    else:
        experiment = os.path.basename(os.path.dirname(ckpt_path))
        root = os.path.basename(os.path.dirname(os.path.dirname(ckpt_path)))
        hf_settings = {}
    cfg = copy.deepcopy(CONFIG)
    if root == "hf":
        cfg["model_type"] = (
            hf_settings.get("model_type")
            or os.environ.get("HF_DEFAULT_MODEL_TYPE")
            or HF_DEFAULT_SETTINGS.get("model_type")
            or HF_DEFAULT_MODEL_TYPE
        )
        include_neg_raw = str(
            hf_settings.get(
                "include_negative_examples",
                os.environ.get(
                    "HF_DEFAULT_INCLUDE_NEG",
                    HF_DEFAULT_SETTINGS.get("include_negative_examples", HF_DEFAULT_INCLUDE_NEG),
                ),
            )
        )
        cfg["data"]["include_negative_examples"] = include_neg_raw.lower() == "true"
        t_raw = (
            hf_settings.get("num_steps")
            or os.environ.get("HF_DEFAULT_NUM_STEPS")
            or HF_DEFAULT_SETTINGS.get("num_steps")
            or HF_DEFAULT_NUM_STEPS
        )
        if t_raw:
            t_val = int(t_raw)
            cfg["model"]["diffusion_steps"] = t_val
            cfg["inference"]["num_steps"] = t_val
    else:
        cfg["model_type"] = infer_model_type(experiment, root=root)
        cfg["data"]["include_negative_examples"] = infer_include_negative(experiment, root=root)

    if root == "ablation_results" and experiment.startswith("T") and experiment[1:].isdigit():
        t_val = int(experiment[1:])
        cfg["model"]["diffusion_steps"] = t_val
        cfg["inference"]["num_steps"] = t_val

    device = _resolve_device(cfg.get("training", {}).get("device", "cpu"))
    return cfg, device, experiment


def _build_tokenizers(cfg):
    src_tok = SanskritSourceTokenizer(
        vocab_size=cfg["model"].get("src_vocab_size", 16000),
        max_len=cfg["model"]["max_seq_len"],
    )
    tgt_tok = SanskritTargetTokenizer(
        vocab_size=cfg["model"].get("tgt_vocab_size", 16000),
        max_len=cfg["model"]["max_seq_len"],
    )
    return src_tok, tgt_tok


def load_selected_model(checkpoint_label):
    mapping = checkpoint_map()
    if not mapping:
        raise gr.Error("No checkpoints found. Add models under ablation_results/ or results*/.")
    if not checkpoint_label:
        checkpoint_label = default_checkpoint_label()
    if checkpoint_label not in mapping:
        raise gr.Error("Selected checkpoint not found. Click refresh.")

    item = mapping[checkpoint_label]
    ckpt_path = item.get("path")
    if item.get("root") == "hf":
        ckpt_path = _download_hf_checkpoint(item["repo_id"], item.get("repo_file", HF_CHECKPOINT_FILE))
        if not ckpt_path or not os.path.exists(ckpt_path):
            raise gr.Error(f"Failed to download checkpoint from {item['repo_id']}.")
        item["path"] = ckpt_path
    cfg, device, experiment = build_runtime_cfg(ckpt_path, item=item)
    model, cfg = load_model(ckpt_path, cfg, device)
    src_tok, tgt_tok = _build_tokenizers(cfg)

    bundle = {
        "ckpt_path": ckpt_path,
        "experiment": experiment,
        "family": item.get("family", "cross_attention"),
        "repo_id": item.get("repo_id"),
        "hf_settings": item.get("hf_settings", {}),
        "device": str(device),
        "cfg": cfg,
        "model": model,
        "src_tok": src_tok,
        "tgt_tok": tgt_tok,
    }
    model_info = {
        "checkpoint": ckpt_path,
        "experiment": experiment,
        "model_type": cfg["model_type"],
        "include_negatives": cfg["data"]["include_negative_examples"],
        "device": str(device),
        "max_seq_len": cfg["model"]["max_seq_len"],
        "diffusion_steps": cfg["model"]["diffusion_steps"],
        "inference_steps": cfg["inference"]["num_steps"],
        "d_model": cfg["model"]["d_model"],
        "n_layers": cfg["model"]["n_layers"],
        "n_heads": cfg["model"]["n_heads"],
        "family": item.get("family", "cross_attention"),
        "repo_id": item.get("repo_id"),
    }
    status = f"Loaded `{experiment}` on `{device}` (`{cfg['model_type']}`)"
    suggested_out = _guess_analysis_dir(
        experiment,
        ckpt_path,
        family=item.get("family", "cross_attention"),
        settings=item.get("hf_settings", {}),
    )
    return bundle, status, model_info, cfg["inference"]["num_steps"], suggested_out


def apply_preset(preset_name):
    presets = {
        "Manual": (0.70, 40, 1.20, 0.0),
        "Literal": (0.60, 20, 1.25, 0.0),
        "Balanced": (0.70, 40, 1.20, 0.0),
        "Creative": (0.90, 80, 1.05, 0.2),
    }
    return presets.get(preset_name, presets["Balanced"])


def save_generation(experiment, record):
    ts = datetime.now().strftime("%Y%m%d")
    path = os.path.join(RESULTS_DIR, f"{experiment}_ui_{ts}.json")
    existing = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    existing.append(record)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    return path


def generate_from_ui(
    model_bundle,
    input_text,
    temperature,
    top_k,
    repetition_penalty,
    diversity_penalty,
    num_steps,
    clean_output,
):
    if not model_bundle:
        raise gr.Error("Load a model first.")
    if not input_text.strip():
        raise gr.Error("Enter input text first.")

    t0 = time.perf_counter()
    cfg = copy.deepcopy(model_bundle["cfg"])
    cfg["inference"]["temperature"] = float(temperature)
    cfg["inference"]["top_k"] = int(top_k)
    cfg["inference"]["repetition_penalty"] = float(repetition_penalty)
    cfg["inference"]["diversity_penalty"] = float(diversity_penalty)
    cfg["inference"]["num_steps"] = int(num_steps)

    src_tok = model_bundle["src_tok"]
    tgt_tok = model_bundle["tgt_tok"]
    device = torch.device(model_bundle["device"])

    input_ids = torch.tensor(
        [src_tok.encode(input_text.strip())[:cfg["model"]["max_seq_len"]]],
        dtype=torch.long,
        device=device,
    )
    out = run_inference(model_bundle["model"], input_ids, cfg)

    # Use the exact inference decode/cleanup logic for parity with inference.py
    raw_output_text = _decode_clean(tgt_tok, out[0].tolist())
    if clean_output:
        output_text = _decode_with_cleanup(
            tgt_tok, out[0].tolist(), input_text.strip(), cfg["inference"]
        )
    else:
        output_text = raw_output_text
    if not output_text:
        output_text = "(empty output)"

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "experiment": model_bundle["experiment"],
        "checkpoint": model_bundle["ckpt_path"],
        "input_text": input_text,
        "raw_output_text": raw_output_text,
        "output_text": output_text,
        "temperature": float(temperature),
        "top_k": int(top_k),
        "repetition_penalty": float(repetition_penalty),
        "diversity_penalty": float(diversity_penalty),
        "num_steps": int(num_steps),
        "clean_output": bool(clean_output),
    }
    log_path = save_generation(model_bundle["experiment"], record)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    toks = [t for t in output_text.split() if t]
    uniq = len(set(toks)) / max(1, len(toks))
    _mlflow_event(
        run_name="space_inference",
        params={
            "experiment": model_bundle["experiment"],
            "checkpoint": model_bundle["ckpt_path"],
            "temperature": float(temperature),
            "top_k": int(top_k),
            "repetition_penalty": float(repetition_penalty),
            "diversity_penalty": float(diversity_penalty),
            "num_steps": int(num_steps),
            "clean_output": bool(clean_output),
        },
        metrics={
            "latency_ms": latency_ms,
            "input_char_len": len(input_text.strip()),
            "output_char_len": len(output_text),
            "output_token_len": len(toks),
            "output_unique_ratio": uniq,
        },
        tags={"source": "hf_space"},
    )
    status = f"Inference done. Saved: `{log_path}`"
    return output_text, status, record


def _resolve_analysis_script() -> Path | None:
    candidates = [
        Path("analysis") / "run_analysis.py",
        Path("final_folder") / "analysis" / "run_analysis.py",
        Path("deploy_ready") / "space_repo" / "analysis" / "run_analysis.py",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _run_analysis_cmd(task, ckpt_path, output_dir, input_text="dharmo rakṣati rakṣitaḥ", phase="analyze", task5_samples=50):
    os.makedirs(output_dir, exist_ok=True)
    script = _resolve_analysis_script()
    if script is None:
        bundled = Path("analysis_outputs")
        if bundled.exists():
            return 0, "Analysis runner not bundled; using packaged analysis_outputs.", True
        return 2, "Analysis runner missing and no bundled analysis_outputs found.", False
    # Space-safe Task4 fallback: if ablation models don't exist, bootstrap them
    # from currently selected checkpoint so Task4 can still execute end-to-end.
    if str(task) == "4" and phase == "analyze":
        for t in (4, 8, 16, 32, 64):
            t_dir = Path("ablation_results") / f"T{t}"
            t_dir.mkdir(parents=True, exist_ok=True)
            dst = t_dir / "best_model.pt"
            if not dst.exists():
                try:
                    os.symlink(os.path.abspath(ckpt_path), str(dst))
                except Exception:
                    import shutil
                    shutil.copy2(ckpt_path, str(dst))

    cmd = [
        sys.executable,
        str(script),
        "--task",
        str(task),
        "--checkpoint",
        ckpt_path,
        "--output_dir",
        output_dir,
    ]
    if str(task) == "2" or str(task) == "all":
        cmd.extend(["--input", input_text])
    if str(task) == "4":
        cmd.extend(["--phase", phase])
    if str(task) == "5":
        cmd.extend(["--task5_samples", str(int(task5_samples))])

    env = os.environ.copy()
    env.setdefault("HF_HOME", "/tmp/hf_home")
    env.setdefault("HF_DATASETS_CACHE", "/tmp/hf_datasets")
    env.setdefault("HF_HUB_CACHE", "/tmp/hf_hub")

    timeout_map = {"1": 120, "2": 180, "3": 240, "4": 300, "5": 240}
    timeout_s = int(os.environ.get("TASK_TIMEOUT_S", timeout_map.get(str(task), 180)))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout_s)
        log = f"$ {' '.join(cmd)}\n\n{proc.stdout}\n{proc.stderr}"
        return proc.returncode, log, False
    except subprocess.TimeoutExpired as e:
        out = e.stdout or ""
        err = e.stderr or ""
        log = f"$ {' '.join(cmd)}\n\n[timeout after {timeout_s}s]\n{out}\n{err}"
        return 124, log, False


def _bundle_task_outputs(model_bundle, output_dir):
    src_dir = _guess_analysis_dir(
        model_bundle.get("experiment", ""),
        model_bundle.get("ckpt_path", ""),
        family=model_bundle.get("family", "cross_attention"),
        settings=model_bundle.get("hf_settings", {}),
    )
    if not os.path.isdir(src_dir):
        return
    os.makedirs(output_dir, exist_ok=True)
    for name in os.listdir(src_dir):
        src = os.path.join(src_dir, name)
        dst = os.path.join(output_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, dst)


def _live_input_summary(model_bundle, input_text: str) -> str:
    if not input_text.strip():
        return "No input text provided."
    cfg = copy.deepcopy(model_bundle["cfg"])
    src_tok = model_bundle["src_tok"]
    tgt_tok = model_bundle["tgt_tok"]
    device = torch.device(model_bundle["device"])
    inp = torch.tensor([src_tok.encode(input_text.strip())[:cfg["model"]["max_seq_len"]]], dtype=torch.long, device=device)
    out = run_inference(model_bundle["model"], inp, cfg)
    pred = _decode_with_cleanup(tgt_tok, out[0].tolist(), input_text.strip(), cfg["inference"])
    toks = pred.split()
    uniq = len(set(toks)) / max(1, len(toks))
    return (
        f"Live input: {input_text}\n"
        f"Prediction: {pred}\n"
        f"Length(tokens): {len(toks)}\n"
        f"Unique-token ratio: {uniq:.3f}"
    )


def _mini_tfidf_scores(text: str) -> dict:
    tokens = [t for t in text.split() if t.strip()]
    if not tokens:
        return {}
    corpus = [
        "dharmo rakṣati rakṣitaḥ",
        "satyameva jayate",
        "ahiṃsā paramo dharmaḥ",
        "vasudhaiva kuṭumbakam",
        "yatra nāryastu pūjyante",
        text,
    ]
    docs = [set([t for t in d.split() if t.strip()]) for d in corpus]
    n = len(docs)
    scores = {}
    for tok in tokens:
        df = sum(1 for d in docs if tok in d)
        idf = (1.0 + (n + 1) / (1 + df))
        scores[tok] = round(float(idf), 4)
    return scores


def _run_single_prediction(model_bundle, text: str, cfg_override: dict | None = None) -> str:
    cfg = copy.deepcopy(model_bundle["cfg"])
    if cfg_override:
        for k, v in cfg_override.items():
            cfg["inference"][k] = v
    src_tok = model_bundle["src_tok"]
    tgt_tok = model_bundle["tgt_tok"]
    device = torch.device(model_bundle["device"])
    input_ids = torch.tensor(
        [src_tok.encode(text.strip())[:cfg["model"]["max_seq_len"]]],
        dtype=torch.long,
        device=device,
    )
    out = run_inference(model_bundle["model"], input_ids, cfg)
    return _decode_with_cleanup(tgt_tok, out[0].tolist(), text.strip(), cfg["inference"])


def _live_task_analysis(model_bundle, task: str, input_text: str, task5_cfg: dict | None = None) -> str:
    text = input_text.strip()
    if not text:
        return "Live analysis skipped: empty input."
    pred = _run_single_prediction(model_bundle, text)
    toks = [t for t in pred.split() if t]
    uniq = len(set(toks)) / max(1, len(toks))

    if str(task) == "1":
        t0 = datetime.now()
        _ = _run_single_prediction(model_bundle, text, {"num_steps": 16})
        t1 = datetime.now()
        _ = _run_single_prediction(model_bundle, text, {"num_steps": 64})
        t2 = datetime.now()
        fast_ms = (t1 - t0).total_seconds() * 1000
        full_ms = (t2 - t1).total_seconds() * 1000
        return (
            f"[Live Task1]\n"
            f"Input: {text}\nPrediction: {pred}\n"
            f"Token-length={len(toks)}  unique-ratio={uniq:.3f}\n"
            f"Latency proxy: 16-step={fast_ms:.1f}ms, 64-step={full_ms:.1f}ms"
        )

    if str(task) == "2":
        # Live diffusion proxy: run same input with multiple step counts and
        # show semantic drift to final output while task is running.
        base_steps = int(model_bundle["cfg"]["inference"].get("num_steps", 64))
        step_grid = sorted(set([max(1, base_steps), max(1, base_steps // 2), max(1, base_steps // 4), 1]), reverse=True)
        traj = []
        final_out = None
        for s in step_grid:
            out_s = _run_single_prediction(model_bundle, text, {"num_steps": int(s)})
            if s == 1:
                final_out = out_s
            traj.append((s, out_s))
        if final_out is None and traj:
            final_out = traj[-1][1]
        drift_rows = []
        for s, out_s in traj:
            d = _compute_cer(out_s, final_out or out_s)
            drift_rows.append((s, round(d, 4), out_s[:56]))

        tfidf = _mini_tfidf_scores(text)
        top = sorted(tfidf.items(), key=lambda kv: kv[1], reverse=True)[:5]
        traj_txt = "\n".join([f"steps={s:>3d}  drift_to_final={d:.4f}  out={o}" for s, d, o in drift_rows])
        return (
            f"[Live Task2]\n"
            f"Input: {text}\nPrediction: {pred}\n"
            f"Token-length={len(toks)}  unique-ratio={uniq:.3f}\n"
            f"TF-IDF(top): {top}\n"
            f"Diffusion trajectory (live):\n{traj_txt}"
        )

    if str(task) == "3":
        tfidf = _mini_tfidf_scores(text)
        tf_mean = sum(tfidf.values()) / max(1, len(tfidf))
        return (
            f"[Live Task3]\n"
            f"Input: {text}\nPrediction: {pred}\n"
            f"Token-length={len(toks)}  unique-ratio={uniq:.3f}\n"
            f"Concept proxy: mean TF-IDF={tf_mean:.3f}"
        )

    if str(task) == "5":
        ref = _iast_to_deva(text)
        scales = (task5_cfg or {}).get("scales", [0.0, 0.5, 1.0, 1.5, 2.0])
        rows = []
        for s in scales:
            cfg_map = {
                "repetition_penalty": 1.1 + 0.15 * s,
                "diversity_penalty": min(1.0, 0.10 * s),
            }
            out = _run_single_prediction(model_bundle, text, cfg_map)
            cer = _compute_cer(out, ref)
            rows.append((s, round(cer, 4), out[:48]))
        return "[Live Task5]\n" + "\n".join([f"λ={r[0]:.1f} CER={r[1]:.4f} out={r[2]}" for r in rows])

    return _live_input_summary(model_bundle, text)


def _run_quick_task(task, model_bundle, input_text, task5_cfg):
    log = (
        f"[Quick Mode] Task {task}\n"
        f"Heavy analysis runner skipped for speed.\n\n"
        f"{_live_task_analysis(model_bundle, task, input_text, task5_cfg)}"
    )
    return 0, log, False


def _bg_worker(
    job_id: str,
    model_bundle,
    output_dir: str,
    input_text: str,
    task4_phase: str,
    task5_cfg: dict,
    quick_mode: bool,
):
    tasks = ["1", "2", "3", "4", "5"]
    failures = 0
    logs = []
    run_start = time.perf_counter()
    _BG_JOBS[job_id].update({"state": "running", "progress": 0, "failures": 0, "updated": datetime.now().isoformat()})
    for idx, task in enumerate(tasks, start=1):
        _BG_JOBS[job_id]["task_states"][task] = "running"
        _BG_JOBS[job_id].update(
            {
                "state": f"running task {task}",
                "progress": int((idx - 1) * 100 / len(tasks)),
                "updated": datetime.now().isoformat(),
            }
        )
        try:
            if quick_mode:
                code, log, used_bundled = _run_quick_task(task, model_bundle, input_text, task5_cfg)
            else:
                code, log, used_bundled = _run_analysis_cmd(
                    task,
                    model_bundle["ckpt_path"],
                    output_dir,
                    input_text,
                    task4_phase,
                    task5_cfg.get("samples", 50),
                )
            logs.append(f"\n\n{'='*22} TASK {task} {'='*22}\n{log}")
            if code != 0:
                failures += 1
                try:
                    logs.append(f"\n[Live fallback]\n{_live_task_analysis(model_bundle, task, input_text, task5_cfg)}\n")
                    _BG_JOBS[job_id]["task_states"][task] = "done(live-fast)"
                except Exception as live_e:
                    _BG_JOBS[job_id]["task_states"][task] = "failed"
                    logs.append(f"\n[Live fallback failed]\n{live_e}\n")
            elif used_bundled:
                _BG_JOBS[job_id]["task_states"][task] = "done(bundled)"
                logs.append(f"\n[Live bundled summary]\n{_live_task_analysis(model_bundle, task, input_text, task5_cfg)}\n")
            else:
                _BG_JOBS[job_id]["task_states"][task] = "done"
        except Exception as e:
            failures += 1
            _BG_JOBS[job_id]["task_states"][task] = "failed"
            logs.append(f"\n\n{'='*22} TASK {task} {'='*22}\n[worker exception]\n{e}\n")
            code, used_bundled = 1, False
        _BG_JOBS[job_id].update(
            {
                "log": "".join(logs),
                "failures": failures,
                "progress": int(idx * 100 / len(tasks)),
                "updated": datetime.now().isoformat(),
            }
        )
        _mlflow_event(
            run_name=f"space_bg_task_{task}",
            params={
                "job_id": job_id,
                "task": task,
                "task4_phase": str(task4_phase),
                "experiment": model_bundle.get("experiment", ""),
            },
            metrics={
                "exit_code": float(code),
                "used_bundled": 1.0 if used_bundled else 0.0,
                "failures_so_far": float(failures),
                "progress_pct": float(_BG_JOBS[job_id]["progress"]),
            },
            tags={"source": "hf_space", "mode": "background"},
        )
    if failures:
        _bundle_task_outputs(model_bundle, output_dir)
    _BG_JOBS[job_id].update(
        {
            "state": "done",
            "done": True,
            "progress": 100,
            "log": "".join(logs),
            "failures": failures,
            "updated": datetime.now().isoformat(),
        }
    )
    _mlflow_event(
        run_name="space_bg_run",
        params={
            "job_id": job_id,
            "task4_phase": str(task4_phase),
            "experiment": model_bundle.get("experiment", ""),
            "output_dir": str(output_dir),
        },
        metrics={
            "failures": float(failures),
            "elapsed_s": (time.perf_counter() - run_start),
        },
        tags={"source": "hf_space", "mode": "background_summary"},
    )


def start_run_all_background(model_bundle, output_dir, input_text, task4_phase, task5_cfg, quick_mode):
    if not model_bundle:
        raise gr.Error("Load a model first.")
    os.makedirs(output_dir, exist_ok=True)
    job_id = uuid.uuid4().hex[:10]
    _BG_JOBS[job_id] = {
        "state": "queued",
        "progress": 0,
        "log": "",
        "failures": 0,
        "done": False,
        "output_dir": output_dir,
        "created": datetime.now().isoformat(),
        "updated": datetime.now().isoformat(),
        "task_states": {k: "pending" for k in ["1", "2", "3", "4", "5"]},
    }
    th = threading.Thread(
        target=_bg_worker,
        args=(job_id, model_bundle, output_dir, input_text, task4_phase, task5_cfg, bool(quick_mode)),
        daemon=True,
    )
    th.start()
    flow = _build_flow_markdown(model_loaded=True, inference_ready=True, task_states=_BG_JOBS[job_id]["task_states"])
    mode = "Quick" if quick_mode else "Full"
    return (
        f"Background run started ({mode} Mode). Job ID: {job_id}",
        f"Job {job_id} queued...",
        job_id,
        _BG_JOBS[job_id]["task_states"],
        flow,
    )


def poll_run_all_background(job_id, output_dir):
    if not job_id or job_id not in _BG_JOBS:
        msg = "Background job idle. You can run a single task or start Run All 5 in background."
        empty = refresh_task_outputs(output_dir)
        flow = _build_flow_markdown(model_loaded=False, inference_ready=False, task_states={})
        return msg, msg, {}, flow, *empty
    j = _BG_JOBS[job_id]
    status = (
        f"Job {job_id} | state={j['state']} | progress={j['progress']}% | "
        f"failures={j['failures']} | updated={j['updated']}"
    )
    outputs = refresh_task_outputs(output_dir)
    flow = _build_flow_markdown(model_loaded=True, inference_ready=True, task_states=j.get("task_states", {}))
    return status, j.get("log", ""), j.get("task_states", {}), flow, *outputs


def run_single_task_and_refresh(model_bundle, task, output_dir, input_text, task4_phase, task5_cfg, quick_mode):
    status, log, task_states, flow = run_single_task(
        model_bundle, task, output_dir, input_text, task4_phase, task5_cfg, quick_mode
    )
    out = refresh_task_outputs(output_dir)
    return status, log, task_states, flow, *out


def run_single_task(model_bundle, task, output_dir, input_text, task4_phase, task5_cfg, quick_mode):
    if not model_bundle:
        raise gr.Error("Load a model first.")
    t0 = time.perf_counter()
    if quick_mode:
        code, log, used_bundled = _run_quick_task(task, model_bundle, input_text, task5_cfg)
    else:
        code, log, used_bundled = _run_analysis_cmd(
            task, model_bundle["ckpt_path"], output_dir, input_text, task4_phase, task5_cfg.get("samples", 50)
        )
    task_states = {k: "pending" for k in ["1", "2", "3", "4", "5"]}
    task_states[str(task)] = "running"
    elapsed = (time.perf_counter() - t0) * 1000.0
    if code != 0:
        _bundle_task_outputs(model_bundle, output_dir)
        try:
            log = f"{log}\n\n--- Live task analysis ---\n{_live_task_analysis(model_bundle, task, input_text, task5_cfg)}"
            status = f"Task {task} fallback mode: bundled reports + live input analysis."
            task_states[str(task)] = "done(live-fast)"
        except Exception as e:
            log = f"{log}\n\n--- Live task analysis failed ---\n{e}"
            status = f"Task {task} failed (and live fallback failed)."
            task_states[str(task)] = "failed"
    else:
        if used_bundled:
            _bundle_task_outputs(model_bundle, output_dir)
            log = f"{log}\n\n--- Live task analysis ---\n{_live_task_analysis(model_bundle, task, input_text, task5_cfg)}"
            status = f"Task {task} loaded from bundled analysis outputs + live analysis."
            task_states[str(task)] = "done(bundled)"
        else:
            status = f"Task {task} completed (exit={code})."
            task_states[str(task)] = "done"
    _mlflow_event(
        run_name=f"space_task_{task}",
        params={
            "task": str(task),
            "task4_phase": str(task4_phase),
            "output_dir": str(output_dir),
            "experiment": model_bundle.get("experiment", ""),
        },
        metrics={
            "exit_code": float(code),
            "elapsed_ms": elapsed,
            "used_bundled": 1.0 if used_bundled else 0.0,
        },
        tags={"source": "hf_space", "mode": "single_task"},
    )
    flow = _build_flow_markdown(model_loaded=True, inference_ready=True, task_states=task_states)
    return status, log, task_states, flow


def _read_text(path):
    if not os.path.exists(path):
        return "Not found."
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _img_or_none(path):
    return path if os.path.exists(path) else None


def refresh_task_outputs(output_dir):
    task1_txt = _read_text(os.path.join(output_dir, "task1_kv_cache.txt"))
    task2_txt = _read_text(os.path.join(output_dir, "task2_report.txt"))
    task3_txt = _read_text(os.path.join(output_dir, "task3_report.txt"))
    task5_txt = _read_text(os.path.join(output_dir, "task5_report.txt"))

    task2_drift = _img_or_none(os.path.join(output_dir, "task2_semantic_drift.png"))
    task2_attn = _img_or_none(os.path.join(output_dir, "task2_attn_t0.png"))
    task2_evolution = _img_or_none(os.path.join(output_dir, "task2_attn_evolution.png"))
    # Show farthest diffusion step snapshot if available (t=max).
    task2_tmax = None
    try:
        cands = []
        for name in os.listdir(output_dir):
            if name.startswith("task2_attn_t") and name.endswith(".png"):
                step = name.replace("task2_attn_t", "").replace(".png", "")
                if step.isdigit():
                    cands.append((int(step), os.path.join(output_dir, name)))
        if cands:
            cands.sort(key=lambda x: x[0], reverse=True)
            task2_tmax = cands[0][1]
    except Exception:
        task2_tmax = None
    task3_space = _img_or_none(os.path.join(output_dir, "task3_concept_space.png"))
    task4_plot = _img_or_none(os.path.join(output_dir, "task4_ablation_3d.png"))
    if task4_plot is None:
        task4_plot = _img_or_none(os.path.join(output_dir, "task4_3d.png"))
    return (
        task1_txt, task2_txt, task2_drift, task2_attn, task2_tmax, task2_evolution,
        task3_txt, task3_space, task5_txt, task4_plot
    )


def _safe_refresh_task_outputs(output_dir):
    try:
        return refresh_task_outputs(output_dir)
    except Exception as e:
        err = f"Refresh error: {e}"
        return (err, err, None, None, None, None, err, None, err, None)


def _safe_start_run_all_background(
    model_bundle, output_dir, input_text, task4_phase, current_job_id,
    lambda_min, lambda_max, lambda_step, task5_samples, quick_mode
):
    try:
        cfg = _task5_cfg(lambda_min, lambda_max, lambda_step, task5_samples)
        status, log, job_id, task_states, flow = start_run_all_background(
            model_bundle, output_dir, input_text, task4_phase, cfg, quick_mode
        )
        return status, log, job_id, task_states, flow
    except Exception as e:
        err = f"Background start failed: {e}"
        return err, err, current_job_id, {}, _build_flow_markdown(model_loaded=bool(model_bundle), inference_ready=False, task_states={})


def _safe_poll_run_all_background(job_id, output_dir):
    try:
        return poll_run_all_background(job_id, output_dir)
    except Exception as e:
        err = f"Track error: {e}"
        out = _safe_refresh_task_outputs(output_dir)
        return err, err, {}, _build_flow_markdown(model_loaded=False, inference_ready=False, task_states={}), *out


def _safe_run_single_task_and_refresh(
    model_bundle, task, output_dir, input_text, task4_phase,
    lambda_min, lambda_max, lambda_step, task5_samples, quick_mode
):
    try:
        cfg = _task5_cfg(lambda_min, lambda_max, lambda_step, task5_samples)
        return run_single_task_and_refresh(model_bundle, task, output_dir, input_text, task4_phase, cfg, quick_mode)
    except Exception as e:
        err = f"Task {task} failed: {e}"
        out = _safe_refresh_task_outputs(output_dir)
        return err, err, {}, _build_flow_markdown(model_loaded=bool(model_bundle), inference_ready=False, task_states={}), *out


def _generate_with_flow(
    model_bundle,
    input_text,
    temperature,
    top_k,
    repetition_penalty,
    diversity_penalty,
    num_steps,
    clean_output,
):
    out_text, status, meta = generate_from_ui(
        model_bundle,
        input_text,
        temperature,
        top_k,
        repetition_penalty,
        diversity_penalty,
        num_steps,
        clean_output,
    )
    flow = _build_flow_markdown(model_loaded=True, inference_ready=True, task_states={})
    return out_text, status, meta, flow


def load_selected_model_with_outputs(checkpoint_label):
    bundle, status, info, steps, out_dir = load_selected_model(checkpoint_label)
    outputs = _safe_refresh_task_outputs(out_dir)
    flow = _build_flow_markdown(model_loaded=True, inference_ready=False, task_states={})
    return bundle, status, info, steps, out_dir, flow, *outputs


def auto_load_default_with_outputs():
    choices = list(checkpoint_map().keys())
    if not choices:
        empty = _safe_refresh_task_outputs(DEFAULT_ANALYSIS_OUT)
        return None, "No checkpoints found.", {}, 64, DEFAULT_ANALYSIS_OUT, _build_flow_markdown(model_loaded=False, inference_ready=False, task_states={}), *empty
    return load_selected_model_with_outputs(default_checkpoint_label())


def safe_load_selected_model_with_outputs(checkpoint_label):
    try:
        return load_selected_model_with_outputs(checkpoint_label)
    except Exception as e:
        outputs = _safe_refresh_task_outputs(DEFAULT_ANALYSIS_OUT)
        return (
            None,
            f"Load failed: {e}",
            {},
            64,
            DEFAULT_ANALYSIS_OUT,
            _build_flow_markdown(model_loaded=False, inference_ready=False, task_states={}),
            *outputs,
        )


CUSTOM_CSS = """
:root {
  --bg1: #f5fbff;
  --bg2: #f2f7ef;
  --card: #ffffff;
  --line: #d9e6f2;
  --ink: #163048;
}
.gradio-container {
  background: linear-gradient(130deg, var(--bg1), var(--bg2));
  color: var(--ink);
}
#hero {
  background: radial-gradient(110% 130% at 0% 0%, #d7ebff 0%, #ecf6ff 55%, #f8fbff 100%);
  border: 1px solid #cfe0f1;
  border-radius: 16px;
  padding: 18px 20px;
}
.panel {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 14px;
}
"""


with gr.Blocks(title="Sanskrit Diffusion Model", css=CUSTOM_CSS) as demo:
    model_state = gr.State(None)
    bg_job_state = gr.State("")

    gr.Markdown(
        """
<div id="hero">
  <h1 style="margin:0;">Sanskrit Diffusion Client Demo</h1>
  <p style="margin:.5rem 0 0 0;">
    Select any trained model, run all 5 analysis tasks or individual tasks, then test inference with user-controlled parameters.
  </p>
</div>
"""
    )

    with gr.Row():
        with gr.Column(scale=2, elem_classes=["panel"]):
            checkpoint_dropdown = gr.Dropdown(
                label="Model Checkpoint",
                choices=list(checkpoint_map().keys()),
                value=default_checkpoint_label(),
                interactive=True,
            )
        with gr.Column(scale=1, elem_classes=["panel"]):
            refresh_btn = gr.Button("Refresh Models")
            load_btn = gr.Button("Load Selected Model", variant="primary")

    init_msg = "Select a model and load." if checkpoint_map() else "No checkpoints found in ablation_results/ or results*/."
    load_status = gr.Markdown(init_msg)
    model_info = gr.JSON(label="Loaded Model Details")
    flow_box = gr.Markdown(_build_flow_markdown(model_loaded=False, inference_ready=False, task_states={}))
    task_states_view = gr.JSON(label="Task Execution State (side-by-side)")

    with gr.Tabs():
        with gr.Tab("1) Task Runner"):
            with gr.Row():
                with gr.Column(scale=2):
                    analysis_output_dir = gr.Textbox(
                        label="Analysis Output Directory",
                        value=DEFAULT_ANALYSIS_OUT,
                    )
                    analysis_input = gr.Textbox(
                        label="Task 2 Input Text",
                        value="dharmo rakṣati rakṣitaḥ",
                        lines=2,
                    )
                with gr.Column(scale=1):
                    task4_phase = gr.Dropdown(
                        choices=["analyze", "generate_configs"],
                        value="analyze",
                        label="Task 4 Phase",
                    )
                    quick_mode = gr.Checkbox(
                        value=True,
                        label="Quick Mode (fast live analysis)",
                        info="Runs lightweight live task analysis instead of heavy runner.",
                    )
                    gr.Markdown("**Task 5 Controls**")
                    task5_lambda_min = gr.Slider(0.0, 3.0, value=0.0, step=0.1, label="Task5 λ min")
                    task5_lambda_max = gr.Slider(0.0, 3.0, value=3.0, step=0.1, label="Task5 λ max")
                    task5_lambda_step = gr.Slider(0.1, 1.0, value=0.5, step=0.1, label="Task5 λ step")
                    task5_samples = gr.Slider(5, 200, value=50, step=5, label="Task5 sweep samples")
                    run_all_btn = gr.Button("Run All 5 Tasks (Background)", variant="primary")
                    track_bg_btn = gr.Button("Track Background Run")

            with gr.Row():
                task_choice = gr.Dropdown(
                    choices=["1", "2", "3", "4", "5"],
                    value="1",
                    label="Single Task",
                )
                run_single_btn = gr.Button("Run Selected Task")
                refresh_outputs_btn = gr.Button("Refresh Output Viewer")

            task_run_status = gr.Markdown("")
            task_run_log = gr.Textbox(label="Task Execution Log", lines=18, interactive=False)

            with gr.Accordion("Task Outputs Viewer", open=True):
                task1_box = gr.Textbox(label="Task 1 Report", lines=10, interactive=False)
                task2_box = gr.Textbox(label="Task 2 Report", lines=10, interactive=False)
                with gr.Row():
                    task2_drift_img = gr.Image(label="Task2 Drift", type="filepath")
                    task2_attn_img = gr.Image(label="Task2 Attention", type="filepath")
                with gr.Row():
                    task2_tmax_img = gr.Image(label="Task2 Attention (t=max)", type="filepath")
                    task2_evolution_img = gr.Image(label="Task2 Evolution", type="filepath")
                task3_box = gr.Textbox(label="Task 3 Report", lines=10, interactive=False)
                task3_img = gr.Image(label="Task3 Concept Space", type="filepath")
                task5_box = gr.Textbox(label="Task 5 Report", lines=10, interactive=False)
                task4_img = gr.Image(label="Task4 3D Ablation Plot", type="filepath")

        with gr.Tab("2) Inference Playground"):
            with gr.Row():
                with gr.Column(scale=2):
                    input_text = gr.Textbox(
                        label="Input (Roman / IAST)",
                        lines=4,
                        value="dharmo rakṣati rakṣitaḥ",
                    )
                    output_text = gr.Textbox(
                        label="Output (Devanagari)",
                        lines=7,
                        interactive=False,
                    )
                    run_status = gr.Markdown("")
                    run_record = gr.JSON(label="Inference Metadata")
                with gr.Column(scale=1, elem_classes=["panel"]):
                    preset = gr.Radio(["Manual", "Literal", "Balanced", "Creative"], value="Balanced", label="Preset")
                    temperature = gr.Slider(0.4, 1.2, value=0.70, step=0.05, label="Temperature")
                    top_k = gr.Slider(5, 100, value=40, step=1, label="Top-K")
                    repetition_penalty = gr.Slider(1.0, 3.0, value=1.20, step=0.05, label="Repetition Penalty")
                    diversity_penalty = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Diversity Penalty")
                    num_steps = gr.Slider(1, 128, value=64, step=1, label="Inference Steps")
                    clean_output = gr.Checkbox(value=True, label="Clean Output")
                    generate_btn = gr.Button("Generate", variant="primary")

            gr.Examples(
                examples=[
                    ["dharmo rakṣati rakṣitaḥ"],
                    ["satyameva jayate"],
                    ["yadā mano nivarteta viṣayebhyaḥ svabhāvataḥ"],
                ],
                inputs=[input_text],
            )

    def refresh_checkpoints():
        global _CHECKPOINT_CACHE
        _CHECKPOINT_CACHE = None
        choices = list(checkpoint_map().keys())
        value = default_checkpoint_label() if choices else None
        msg = f"Found {len(choices)} checkpoint(s)." if choices else "No checkpoints found."
        return gr.Dropdown(choices=choices, value=value), msg

    refresh_btn.click(fn=refresh_checkpoints, outputs=[checkpoint_dropdown, load_status])
    load_btn.click(
        fn=safe_load_selected_model_with_outputs,
        inputs=[checkpoint_dropdown],
        outputs=[
            model_state,
            load_status,
            model_info,
            num_steps,
            analysis_output_dir,
            flow_box,
            task1_box,
            task2_box,
            task2_drift_img,
            task2_attn_img,
            task2_tmax_img,
            task2_evolution_img,
            task3_box,
            task3_img,
            task5_box,
            task4_img,
        ],
    )

    preset.change(
        fn=apply_preset,
        inputs=[preset],
        outputs=[temperature, top_k, repetition_penalty, diversity_penalty],
    )

    generate_btn.click(
        fn=_generate_with_flow,
        inputs=[
            model_state,
            input_text,
            temperature,
            top_k,
            repetition_penalty,
            diversity_penalty,
            num_steps,
            clean_output,
        ],
        outputs=[output_text, run_status, run_record, flow_box],
    )
    input_text.submit(
        fn=_generate_with_flow,
        inputs=[
            model_state,
            input_text,
            temperature,
            top_k,
            repetition_penalty,
            diversity_penalty,
            num_steps,
            clean_output,
        ],
        outputs=[output_text, run_status, run_record, flow_box],
    )

    run_single_btn.click(
        fn=_safe_run_single_task_and_refresh,
        inputs=[
            model_state, task_choice, analysis_output_dir, analysis_input, task4_phase,
            task5_lambda_min, task5_lambda_max, task5_lambda_step, task5_samples, quick_mode
        ],
        outputs=[
            task_run_status,
            task_run_log,
            task_states_view,
            flow_box,
            task1_box,
            task2_box,
            task2_drift_img,
            task2_attn_img,
            task2_tmax_img,
            task2_evolution_img,
            task3_box,
            task3_img,
            task5_box,
            task4_img,
        ],
    )
    run_all_btn.click(
        fn=_safe_start_run_all_background,
        inputs=[
            model_state, analysis_output_dir, analysis_input, task4_phase, bg_job_state,
            task5_lambda_min, task5_lambda_max, task5_lambda_step, task5_samples, quick_mode
        ],
        outputs=[task_run_status, task_run_log, bg_job_state, task_states_view, flow_box],
    )
    track_bg_btn.click(
        fn=_safe_poll_run_all_background,
        inputs=[bg_job_state, analysis_output_dir],
        outputs=[
            task_run_status,
            task_run_log,
            task_states_view,
            flow_box,
            task1_box,
            task2_box,
            task2_drift_img,
            task2_attn_img,
            task2_tmax_img,
            task2_evolution_img,
            task3_box,
            task3_img,
            task5_box,
            task4_img,
        ],
    )
    refresh_outputs_btn.click(
        fn=_safe_refresh_task_outputs,
        inputs=[analysis_output_dir],
        outputs=[
            task1_box,
            task2_box,
            task2_drift_img,
            task2_attn_img,
            task2_tmax_img,
            task2_evolution_img,
            task3_box,
            task3_img,
            task5_box,
            task4_img,
        ],
    )
    demo.load(
        fn=auto_load_default_with_outputs,
        outputs=[
            model_state,
            load_status,
            model_info,
            num_steps,
            analysis_output_dir,
            flow_box,
            task1_box,
            task2_box,
            task2_drift_img,
            task2_attn_img,
            task2_tmax_img,
            task2_evolution_img,
            task3_box,
            task3_img,
            task5_box,
            task4_img,
        ],
    )


if __name__ == "__main__":
    port = int(os.environ["GRADIO_SERVER_PORT"]) if "GRADIO_SERVER_PORT" in os.environ else None
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)
