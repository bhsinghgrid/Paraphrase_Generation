import copy
import json
import os
from datetime import datetime

import gradio as gr
import torch

from config import CONFIG
from inference import load_model, run_inference, _build_tokenizers, _resolve_device


RESULTS_DIR = "generated_results"
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
            found.append({
                "label": f"{entry}  [{root}]",
                "path": ckpt,
                "experiment": entry,
                "root": root,
            })
    return found


def default_checkpoint_label():
    checkpoints = discover_checkpoints()
    if not checkpoints:
        return None
    for item in checkpoints:
        if item["path"].endswith("ablation_results/T4/best_model.pt"):
            return item["label"]
    return checkpoints[0]["label"]


def checkpoint_map():
    return {item["label"]: item for item in discover_checkpoints()}


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
    device = _resolve_device(cfg)
    return cfg, device, experiment


def load_selected_model(checkpoint_label):
    mapping = checkpoint_map()
    if checkpoint_label not in mapping:
        raise gr.Error("Selected checkpoint was not found. Refresh the dropdown.")

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
        "d_model": cfg["model"]["d_model"],
        "n_layers": cfg["model"]["n_layers"],
        "n_heads": cfg["model"]["n_heads"],
    }
    status = f"Loaded `{experiment}` on `{device}`."
    return bundle, status, model_info, cfg["inference"]["num_steps"]


def apply_preset(preset_name):
    presets = {
        "Manual": (0.70, 40, 1.20, 0.0, 64),
        "Literal": (0.60, 20, 1.25, 0.0, 64),
        "Balanced": (0.70, 40, 1.20, 0.0, 64),
        "Creative": (0.85, 80, 1.20, 0.2, 64),
    }
    return presets.get(preset_name, presets["Balanced"])


def task_notes_md():
    return """
### Task Notes

**Task 1: KV Cache**
- Benchmark encoder caching vs standard generation.
- Best for engineering evaluation, not language quality evaluation.

**Task 2: Attention + Drift**
- Shows internal attention maps and output stabilization over diffusion steps.
- Useful for diagnostics and mentor discussion of model behavior.

**Task 3: Concept Vectors**
- Experimental PCA steering over decoder hidden states.
- Current outputs are exploratory, not strong semantic evidence yet.

**Task 4: Step Ablation**
- Requires retraining separate checkpoints for each diffusion step count.
- Use this UI for generation only; ablation analysis runs from `analysis/run_analysis.py`.

**Task 5: Quality Guidance**
- Advanced experimental feature in the analysis pipeline.
- Not exposed in this UI because the current evidence is still under validation.
"""


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


def clean_generated_text(text: str, max_consecutive: int = 2, max_occurrence_ratio: float = 0.15) -> str:
    """
    Lightweight cleanup for repetitive diffusion outputs.
    Keeps Sanskrit tokens but trims pathological token loops.
    """
    text = " ".join(text.split())
    if not text:
        return text

    tokens = text.split()
    cleaned = []

    # 1) Limit consecutive token repetitions.
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

    # 2) Limit global over-dominant tokens (common in collapse cases).
    if cleaned:
        max_occ = max(3, int(len(cleaned) * max_occurrence_ratio))
        counts = {}
        filtered = []
        for tok in cleaned:
            c = counts.get(tok, 0) + 1
            counts[tok] = c
            if c <= max_occ:
                filtered.append(tok)
        cleaned = filtered

    out = " ".join(cleaned)
    out = out.replace(" ।", "।").replace(" ॥", "॥")
    out = " ".join(out.split())
    return out


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
        [src_tok.encode(input_text.strip())],
        dtype=torch.long,
        device=device,
    )
    out = run_inference(model_bundle["model"], input_ids, cfg)
    clean = [x for x in out[0].tolist() if x > 4]
    raw_output_text = tgt_tok.decode(clean).strip()
    output_text = clean_generated_text(raw_output_text) if clean_output else raw_output_text
    if not output_text:
        output_text = "(empty output)"

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "experiment": model_bundle["experiment"],
        "checkpoint": model_bundle["ckpt_path"],
        "input_text": input_text,
        "raw_output_text": raw_output_text,
        "output_text": output_text,
        "clean_output": bool(clean_output),
        "temperature": float(temperature),
        "top_k": int(top_k),
        "repetition_penalty": float(repetition_penalty),
        "diversity_penalty": float(diversity_penalty),
        "num_steps": int(num_steps),
    }
    log_path = save_generation(model_bundle["experiment"], record)
    status = f"Generated with `{model_bundle['experiment']}`. Saved to `{log_path}`."
    return output_text, status, record


with gr.Blocks(title="Sanskrit D3PM Studio") as demo:
    model_state = gr.State(None)

    gr.Markdown(
        """
# Sanskrit D3PM Studio

Load any available checkpoint, generate Devanagari output from Roman/IAST Sanskrit,
and inspect the settings used for evaluation or demos.
"""
    )

    with gr.Row():
        with gr.Column(scale=2):
            checkpoint_dropdown = gr.Dropdown(
                label="Available Checkpoints",
                choices=list(checkpoint_map().keys()),
                value=default_checkpoint_label(),
                interactive=True,
            )
        with gr.Column(scale=1):
            refresh_btn = gr.Button("Refresh List")
            load_btn = gr.Button("Load Model", variant="primary")

    load_status = gr.Markdown("Select a checkpoint and load it.")
    model_info = gr.JSON(label="Loaded Model Info")

    with gr.Row():
        with gr.Column(scale=2):
            input_text = gr.Textbox(
                label="Input Text (Roman / IAST Sanskrit)",
                placeholder="dharmo rakṣati rakṣitaḥ",
                lines=4,
            )
            output_text = gr.Textbox(
                label="Generated Output (Devanagari)",
                lines=6,
                interactive=False,
            )
            generate_btn = gr.Button("Generate", variant="primary")
        with gr.Column(scale=1):
            preset = gr.Radio(
                ["Manual", "Literal", "Balanced", "Creative"],
                value="Balanced",
                label="Inference Preset",
            )
            temperature = gr.Slider(0.4, 1.2, value=0.70, step=0.05, label="Temperature")
            top_k = gr.Slider(5, 100, value=40, step=1, label="Top-K")
            repetition_penalty = gr.Slider(1.0, 3.0, value=1.20, step=0.05, label="Repetition Penalty")
            diversity_penalty = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Diversity Penalty")
            num_steps = gr.Slider(1, 128, value=64, step=1, label="Inference Steps")
            clean_output = gr.Checkbox(value=True, label="Clean Output (dedupe loops)")

    run_status = gr.Markdown("")
    run_record = gr.JSON(label="Last Generation Metadata")

    with gr.Accordion("Task Details and Evaluation Notes", open=False):
        task_notes = gr.Markdown(task_notes_md())

    gr.Examples(
        examples=[
            ["dharmo rakṣati rakṣitaḥ"],
            ["satyameva jayate"],
            ["ahaṃ brahmāsmi"],
            ["yatra nāryastu pūjyante"],
        ],
        inputs=[input_text],
        label="Quick Examples",
    )

    def refresh_checkpoints():
        choices = list(checkpoint_map().keys())
        value = choices[0] if choices else None
        return gr.Dropdown(choices=choices, value=value)

    refresh_btn.click(fn=refresh_checkpoints, outputs=[checkpoint_dropdown])
    load_btn.click(
        fn=load_selected_model,
        inputs=[checkpoint_dropdown],
        outputs=[model_state, load_status, model_info, num_steps],
    )
    preset.change(
        fn=apply_preset,
        inputs=[preset],
        outputs=[temperature, top_k, repetition_penalty, diversity_penalty, num_steps],
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


if __name__ == "__main__":
    port = int(os.environ["GRADIO_SERVER_PORT"]) if "GRADIO_SERVER_PORT" in os.environ else None
    demo.launch(server_name="127.0.0.1", server_port=port, share=False)
