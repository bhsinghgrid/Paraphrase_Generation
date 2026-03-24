import copy
import json
import os
import subprocess
import sys
from datetime import datetime
from pickle import TRUE

import gradio as gr
import torch

from config import CONFIG
from inference import _resolve_device, load_model, run_inference, _decode_clean, _decode_with_cleanup
from model.tokenizer import SanskritSourceTokenizer, SanskritTargetTokenizer


RESULTS_DIR = "generated_results"
DEFAULT_ANALYSIS_OUT = "analysis/outputs"
os.makedirs(RESULTS_DIR, exist_ok=True)


def discover_checkpoints():
    found = []
    for root in ("ablation_results", "results7", "results"):
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
                }
            )
    return found


def checkpoint_map():
    return {item["label"]: item for item in discover_checkpoints()}


def default_checkpoint_label():
    cps = discover_checkpoints()
    if not cps:
        return None
    for item in cps:
        if item["path"].endswith("ablation_results/T4/best_model.pt"):
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


def build_runtime_cfg(ckpt_path: str):
    experiment = os.path.basename(os.path.dirname(ckpt_path))
    root = os.path.basename(os.path.dirname(os.path.dirname(ckpt_path)))
    cfg = copy.deepcopy(CONFIG)
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

    ckpt_path = mapping[checkpoint_label]["path"]
    cfg, device, experiment = build_runtime_cfg(ckpt_path)
    model, cfg = load_model(ckpt_path, cfg, device)
    src_tok, tgt_tok = _build_tokenizers(cfg)

    bundle = {
        "ckpt_path": ckpt_path,
        "experiment": experiment,
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
    }
    status = f"Loaded `{experiment}` on `{device}` (`{cfg['model_type']}`)"
    suggested_out = os.path.join("analysis", "outputs_ui", experiment)
    return bundle, status, model_info, cfg["inference"]["num_steps"], suggested_out


def apply_preset(preset_name):
    presets = {
        "Manual": (0.70, 40, 1.20, 0.0),
        "Literal": (0.60, 20, 1.25, 0.0),
        "Balanced": (0.70, 40, 1.20, 0.0),
        "Creative": (0.90, 80, 1.05, 0.2),
    }
    return presets.get(preset_name, presets["Balanced"])


def clean_generated_text(text: str, max_consecutive: int = 2) -> str:
    text = " ".join(text.split())
    if not text:
        return text
    tokens = text.split()
    cleaned = []
    prev = None
    run = 0
    for tok in tokens:
        if tok == prev:
            run += 1
        else:
            prev = tok
            run = 1
        if run <= max_consecutive:
            cleaned.append(tok)
    out = " ".join(cleaned).replace(" ।", "।").replace(" ॥", "॥")
    return " ".join(out.split())


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
    status = f"Inference done. Saved: `{log_path}`"
    return output_text, status, record


def _run_analysis_cmd(task, ckpt_path, output_dir, input_text="dharmo rakṣati rakṣitaḥ", phase="analyze"):
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        sys.executable,
        "analysis/run_analysis.py",
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

    env = os.environ.copy()
    env.setdefault("HF_HOME", "/tmp/hf_home")
    env.setdefault("HF_DATASETS_CACHE", "/tmp/hf_datasets")
    env.setdefault("HF_HUB_CACHE", "/tmp/hf_hub")

    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    log = f"$ {' '.join(cmd)}\n\n{proc.stdout}\n{proc.stderr}"
    return proc.returncode, log


def run_single_task(model_bundle, task, output_dir, input_text, task4_phase):
    if not model_bundle:
        raise gr.Error("Load a model first.")
    code, log = _run_analysis_cmd(task, model_bundle["ckpt_path"], output_dir, input_text, task4_phase)
    status = f"Task {task} {'completed' if code == 0 else 'failed'} (exit={code})."
    return status, log


def run_all_tasks(model_bundle, output_dir, input_text, task4_phase):
    if not model_bundle:
        raise gr.Error("Load a model first.")
    logs = []
    failures = 0
    for task in ["1", "2", "3", "4", "5"]:
        code, log = _run_analysis_cmd(task, model_bundle["ckpt_path"], output_dir, input_text, task4_phase)
        logs.append(f"\n\n{'='*22} TASK {task} {'='*22}\n{log}")
        if code != 0:
            failures += 1
    status = f"Run-all finished with {failures} failed task(s)." if failures else "All 5 tasks completed."
    return status, "".join(logs)


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
    task3_space = _img_or_none(os.path.join(output_dir, "task3_concept_space.png"))
    task4_plot = _img_or_none(os.path.join(output_dir, "task4_ablation_3d.png"))
    if task4_plot is None:
        task4_plot = _img_or_none(os.path.join(output_dir, "task4_3d.png"))
    return task1_txt, task2_txt, task2_drift, task2_attn, task3_txt, task3_space, task5_txt, task4_plot


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


with gr.Blocks(title="Sanskrit Diffusion Client Demo", css=CUSTOM_CSS) as demo:
    model_state = gr.State(None)

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
                    run_all_btn = gr.Button("Run All 5 Tasks", variant="primary")

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
        choices = list(checkpoint_map().keys())
        value = default_checkpoint_label() if choices else None
        msg = f"Found {len(choices)} checkpoint(s)." if choices else "No checkpoints found."
        return gr.Dropdown(choices=choices, value=value), msg

    def auto_load_default():
        choices = list(checkpoint_map().keys())
        if not choices:
            return None, "No checkpoints found.", {}, 64, DEFAULT_ANALYSIS_OUT
        return load_selected_model(default_checkpoint_label())

    refresh_btn.click(fn=refresh_checkpoints, outputs=[checkpoint_dropdown, load_status])
    load_btn.click(
        fn=load_selected_model,
        inputs=[checkpoint_dropdown],
        outputs=[model_state, load_status, model_info, num_steps, analysis_output_dir],
    )

    preset.change(
        fn=apply_preset,
        inputs=[preset],
        outputs=[temperature, top_k, repetition_penalty, diversity_penalty],
    )

    generate_btn.click(
        fn=generate_from_ui,
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
        outputs=[output_text, run_status, run_record],
    )
    input_text.submit(
        fn=generate_from_ui,
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
        outputs=[output_text, run_status, run_record],
    )

    run_single_btn.click(
        fn=run_single_task,
        inputs=[model_state, task_choice, analysis_output_dir, analysis_input, task4_phase],
        outputs=[task_run_status, task_run_log],
    )
    run_all_btn.click(
        fn=run_all_tasks,
        inputs=[model_state, analysis_output_dir, analysis_input, task4_phase],
        outputs=[task_run_status, task_run_log],
    )
    refresh_outputs_btn.click(
        fn=refresh_task_outputs,
        inputs=[analysis_output_dir],
        outputs=[
            task1_box,
            task2_box,
            task2_drift_img,
            task2_attn_img,
            task3_box,
            task3_img,
            task5_box,
            task4_img,
        ],
    )
    demo.load(
        fn=auto_load_default,
        outputs=[model_state, load_status, model_info, num_steps, analysis_output_dir],
    )
    demo.load(
        fn=refresh_task_outputs,
        inputs=[analysis_output_dir],
        outputs=[
            task1_box,
            task2_box,
            task2_drift_img,
            task2_attn_img,
            task3_box,
            task3_img,
            task5_box,
            task4_img,
        ],
    )


if __name__ == "__main__":
    port = int(os.environ["GRADIO_SERVER_PORT"]) if "GRADIO_SERVER_PORT" in os.environ else None
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)
