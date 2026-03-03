# new_app_fixed.py
import gradio as gr
import os
import json
import torch
import sys
from datetime import datetime

# add project root to path (adjust if needed)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.tokenizer import SanskritTokenizer
from model.new_d3pm_model import SanskritModel
from diffusion.reverse_process import ReverseDiffusion
from diffusion.scheduler import OptimizedCosineScheduler

# ===========================
# CONFIG (single source of truth)
# ===========================
CONFIG = {
    "model_type": "baseline_encoder_decoder",  # will be updated by load_model
    "model": {
        "vocab_size": 16000,
        "max_seq_len": 80,
        "diffusion_steps": 12,
        "d_model": 384,
        "n_layers": 6,
        "n_heads": 8,
        "d_ff": 1536,
        "dropout": 0.1
    },
    "diffusion": {
        "mask_token_id": 0
    },
    "training": {
        "precision": "float32",
        "device": "mps"  # or "cuda" / "cpu"
    }
}

# device & dtype
device = torch.device(CONFIG["training"]["device"])
dtype = torch.float16 if CONFIG["training"]["precision"] == "float16" else torch.float32

# ===========================
# Globals for loaded model & tokenizer & reverse diffusion
# ===========================
current_model_type = None               # internal key: 'baseline_encoder_decoder' or 'd3pm_cross_attention'
model_instance = None                   # SanskritModel instance
tokenizer = SanskritTokenizer(CONFIG["model"]["vocab_size"])
reverse_diffusion = None                # ReverseDiffusion instance (only for d3pm)
stored_results = {}                     # caching outputs per model for saving

# helper: get checkpoint path by model_type string
def _checkpoint_path_for(cfg_type):
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "production_model3")
    fname = f"best_{cfg_type}.pt"
    return os.path.join(base_dir, fname)

# safe forward wrapper (keeps API consistent)
def _apply_safe_forward_patch(model):
    original_forward = model.forward
    def safe_forward(*args, **kwargs):
        out = original_forward(*args, **kwargs)
        # if model returns logits (tensor), normalize to (logits, None)
        if not isinstance(out, tuple):
            return out, None
        return out
    model.forward = safe_forward

# ===========================
# load_model: builds & loads checkpoint, sets up reverse diffusion if needed
# ===========================
def load_model(model_choice):
    """
    model_choice: UI string "Encoder-Decoder" or "Cross Attention"
    This function updates CONFIG['model_type'], constructs model, loads checkpoint, and builds reverse_diffusion if needed.
    """
    global current_model_type, model_instance, tokenizer, reverse_diffusion, CONFIG

    # map UI -> internal checkpoint type
    if model_choice == "Encoder-Decoder":
        cfg_type = "baseline_encoder_decoder"
    elif model_choice == "Cross Attention":
        cfg_type = "d3pm_cross_attention"
    else:
        raise ValueError("Unknown model choice")

    # If already loaded and same -> nothing to do
    if current_model_type == cfg_type and model_instance is not None:
        return

    # update config model_type so SanskritModel constructor sees it
    CONFIG["model_type"] = cfg_type

    # build tokenizer for the assumed vocab size (if you have saved tokenizer file, you can load instead)
    tokenizer = SanskritTokenizer(CONFIG["model"]["vocab_size"])

    # construct model (pass full CONFIG so constructor can read model_type)
    model_instance = SanskritModel(CONFIG)

    # load checkpoint (non-strict to tolerate extra keys)
    ckpt_path = _checkpoint_path_for(cfg_type)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    # If checkpoint contains wrapper dict with 'state_dict', handle that
    if isinstance(state, dict) and "state_dict" in state:
        state_dict = state["state_dict"]
    else:
        state_dict = state

    # Try to load state (non-strict to avoid diffusion-only keys conflict)
    model_instance.load_state_dict(state_dict, strict=False)

    # move to device/dtype
    model_instance.to(device, dtype=dtype)
    model_instance.eval()

    # patch forward for consistent outputs
    _apply_safe_forward_patch(model_instance)

    # if this model is diffusion-style (d3pm), build reverse_diffusion
    if cfg_type == "d3pm_cross_attention":
        scheduler = OptimizedCosineScheduler(CONFIG)
        reverse_diffusion = ReverseDiffusion(scheduler)
    else:
        reverse_diffusion = None

    # persist globals
    globals_to_update = globals()
    globals_to_update["current_model_type"] = cfg_type
    globals_to_update["model_instance"] = model_instance
    globals_to_update["tokenizer"] = tokenizer
    globals_to_update["reverse_diffusion"] = reverse_diffusion

    print(f"[INFO] Loaded model '{cfg_type}' from {ckpt_path}")

# ===========================
# generation: handles both baseline and diffusion models
# ===========================
@torch.no_grad()
def generate_text(user_input, model_choice, beam_width=3, max_len=80):
    """
    - For baseline (encoder-decoder): runs model(input_ids, input_ids) and greedy-decodes.
    - For d3pm_cross_attention: uses reverse_diffusion.generate_beam(...) which handles timesteps internally.
    """
    # Ensure model loaded
    load_model(model_choice)
    cfg_type = CONFIG["model_type"]

    # encode
    input_ids = tokenizer.encode(user_input)
    if not isinstance(input_ids, list):
        # ensure plain list
        input_ids = list(map(int, input_ids))
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    mask_token_id = CONFIG["diffusion"]["mask_token_id"]

    if cfg_type == "d3pm_cross_attention":
        # use reverse diffusion beam search (this abstraction was in your earlier code)
        if reverse_diffusion is None:
            # defensive: build if missing
            scheduler = OptimizedCosineScheduler(CONFIG)
            rd = ReverseDiffusion(scheduler)
        else:
            rd = reverse_diffusion

        # generate_beam should return tensor of shape (beam, seq_len) or list; adjust accordingly
        generated_ids = rd.generate_beam(
            model_instance,
            condition=input_tensor,
            beam_width=beam_width,
            num_steps=CONFIG["model"]["diffusion_steps"]
        )

        # If returned as (N, L) tensor, take first beam
        if isinstance(generated_ids, torch.Tensor):
            seq = generated_ids[0].tolist()
        else:
            # assume list of lists
            seq = generated_ids[0]

        # decode and strip mask tokens
        out_ids = [int(i) for i in seq if int(i) != mask_token_id]
        output_text = tokenizer.decode(out_ids).strip()
        return output_text

    else:
        # baseline encoder-decoder (autoregressive / single forward)
        out = model_instance(input_tensor, input_tensor)
        # model may return (logits, extra) due to safe_forward: handle either
        if isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out

        # logits shape -> (batch, seq, vocab)
        if logits is None:
            raise RuntimeError("Model returned None logits")

        preds = torch.argmax(logits, dim=-1)  # greedy
        pred_seq = preds[0].tolist()
        out_ids = [int(i) for i in pred_seq if int(i) != mask_token_id]
        output_text = tokenizer.decode(out_ids).strip()
        return output_text

# ===========================
# Gradio UI wiring
# ===========================
os.makedirs("results", exist_ok=True)

# def chat_ui(user_input, chat_history, model_choice, beam_width):
#     try:
#         output = generate_text(user_input, model_choice, beam_width=beam_width)
#         # append to UI history
#         chat_history = chat_history or []
#         chat_history.append((user_input, output))
#
#         # save per-model results
#         key = model_choice.replace(" ", "_")
#         stored_results.setdefault(key, [])
#         stored_results[key].append({
#             "time": datetime.utcnow().isoformat(),
#             "input": user_input,
#             "output": output
#         })
#         with open(f"results/inference_{key}.json", "w", encoding="utf-8") as f:
#             json.dump(stored_results[key], f, ensure_ascii=False, indent=2)
#
#         return "", chat_history
#     except Exception as e:
#         # surface a readable error inside the chat input (do not crash Gradio)
#         return f"Error: {e}", chat_history
def chat_ui(user_input, chat_history, model_choice, beam_width):
    try:
        output = generate_text(user_input, model_choice, beam_width=beam_width)
        # initialize history if None
        chat_history = chat_history or []

        # append new message in the correct format
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": output})

        # save per-model results
        key = model_choice.replace(" ", "_")
        stored_results.setdefault(key, [])
        stored_results[key].append({
            "time": datetime.now().astimezone().isoformat(),  # use timezone-aware datetime
            "input": user_input,
            "output": output
        })
        with open(f"results/inference_{key}.json", "w", encoding="utf-8") as f:
            json.dump(stored_results[key], f, ensure_ascii=False, indent=2)

        # return empty string to clear input box, and updated chat_history
        return "", chat_history

    except Exception as e:
        # show error in input box, keep history unchanged
        return f"Error: {e}", chat_history

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🕉 Sanskrit Neural Generator")

    model_dropdown = gr.Dropdown(["Encoder-Decoder", "Cross Attention"],
                                value="Encoder-Decoder",
                                label="Model Architecture")

    with gr.Accordion("⚙️ Advanced Controls", open=False):
        beam_slider = gr.Slider(1, 10, value=3, step=1, label="Beam Width")
        max_len_slider = gr.Slider(10, 200, value=CONFIG["model"]["max_seq_len"], step=1, label="Max Length (baseline)")

    chatbot = gr.Chatbot(height=450)
    msg = gr.Textbox(placeholder="Type Sanskrit input here...")
    clear = gr.Button("Clear Chat")

    msg.submit(chat_ui,
               inputs=[msg, chatbot, model_dropdown, beam_slider],
               outputs=[msg, chatbot])

    clear.click(lambda: [], None, chatbot)

if __name__ == "__main__":
    demo.launch()