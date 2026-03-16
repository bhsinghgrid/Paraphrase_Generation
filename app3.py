import torch
import torch.nn.functional as F
import os
import json
import sys
from types import MethodType

# =========================================================
# CONFIG
# =========================================================
CONFIG = {
    "model_type": "d3pm_cross_attention",
    "model": {
        "vocab_size": 16000,
        "src_vocab_size": 16000,
        "tgt_vocab_size": 16000,
        "max_seq_len": 80,
        "diffusion_steps": 128,
        "d_model": 1024,
        "n_layers": 8,
        "n_heads": 8,
        "d_ff": 4096,
        "dropout": 0.2,
    },
    "diffusion": {"mask_token_id": 0},
    "training": {"precision": "float32", "device": "mps"},
}

# =========================================================
# DEVICE
# =========================================================
device = torch.device(CONFIG["training"]["device"])
dtype = torch.float16 if CONFIG["training"]["precision"] == "float16" else torch.float32

# =========================================================
# IMPORT MODULES
# =========================================================
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.tokenizer import SanskritTokenizer
from model.sanskrit_model import SanskritModel
from diffusion.reverse_process import ReverseDiffusion
from diffusion.scheduler import OptimizedCosineScheduler

# =========================================================
# USER MODEL SELECTION
# =========================================================
print("Available models: d3pm_encoder_decoder, d3pm_cross_attention")
selected_model = input("Choose model [d3pm_encoder_decoder]: ").strip()
if not selected_model:
    selected_model = "d3pm_encoder_decoder"
CONFIG["model_type"] = selected_model

# =========================================================
# LOAD TOKENIZER & MODEL
# =========================================================
tokenizer = SanskritTokenizer(CONFIG["model"]["vocab_size"])
model = SanskritModel(CONFIG)

# model_path = os.path.join(
#     os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
#     "production_model3",
#     f"best_{CONFIG['model_type']}.pt"
# )
model_path = f"/Users/bhsingh/Documents/Final_Paraphrase/Exclude_Negative/results7/d3pm_cross_attention_neg_True/best_model.pt"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found: {model_path}")

model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
model.to(device, dtype=dtype)
model.eval()
model.mask_token_id = CONFIG["diffusion"]["mask_token_id"]

num_steps = CONFIG["model"]["diffusion_steps"]

# =========================================================
# SAFE FORWARD PATCH
# =========================================================
original_forward = model.forward

def safe_forward(*args, **kwargs):
    output = original_forward(*args, **kwargs)
    if not isinstance(output, tuple):
        return output, None
    return output

model.forward = safe_forward

# =========================================================
# REVERSE DIFFUSION SETUP
# =========================================================
scheduler = OptimizedCosineScheduler(CONFIG)
reverse_diffusion = ReverseDiffusion(scheduler)

# =========================================================
# PATCHED p_sample_step
# =========================================================
def p_sample_step_override(self, *args, **kwargs):

    model_arg = kwargs.get("model") or args[0]
    x_t = kwargs.get("x_t") or args[1]
    t = kwargs.get("t") or args[2]
    condition = kwargs.get("condition") or args[3]
    beam_width = int(kwargs.get("beam_width", 1))

    if x_t.dim() == 1:
        x_t = x_t.unsqueeze(0)

    if condition.dim() == 1:
        condition = condition.unsqueeze(0)

    if isinstance(t, int):
        t = torch.tensor([t], device=condition.device)

    logits, _ = model_arg(condition, x_t, t)

    temperature = getattr(self, "temperature", 1.0)
    repetition_penalty = getattr(self, "repetition_penalty", 1.0)
    diversity_penalty = getattr(self, "diversity_penalty", 0.0)
    sampling_mode = getattr(self, "sampling_mode", False)

    if diversity_penalty > 0:
        logits += diversity_penalty * logits.var(dim=-1, keepdim=True)

    B, L, V = logits.shape

    for b in range(B):
        prev_tokens = x_t[b].view(-1)
        for tok in prev_tokens.unique():
            tok_i = int(tok.item())
            if tok_i == CONFIG["diffusion"]["mask_token_id"]:
                continue
            logits[b, :, tok_i] /= repetition_penalty

    logits = logits / max(temperature, 1e-6)
    probs = F.softmax(logits, dim=-1)

    probs_flat = probs.view(-1, V)

    candidates = []

    for k in range(beam_width):

        if sampling_mode:

            topk_probs, topk_ids = torch.topk(probs_flat, min(50, V), dim=-1)
            topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-12)
            next_ids = torch.multinomial(topk_probs, 1).squeeze(-1)
            next_tokens = topk_ids[torch.arange(topk_ids.size(0)), next_ids].view(B, L)

        else:

            topk_probs, topk_ids = torch.topk(probs_flat, beam_width, dim=-1)
            next_tokens = topk_ids[:, k].view(B, L)

        chosen_probs = probs_flat[torch.arange(probs_flat.size(0)), next_tokens.view(-1)]
        score = torch.log(chosen_probs + 1e-12).sum().item()

        candidates.append((next_tokens, score))

    return candidates

reverse_diffusion.p_sample_step = MethodType(p_sample_step_override, reverse_diffusion)

# =========================================================
# GENERATE OUTPUT
# =========================================================
@torch.no_grad()
def generate_output(input_text, diversity_level="medium",
                    repetition_penalty=1.15, diversity_penalty=0.0, length_penalty=1.0):

    input_ids = tokenizer.encode(input_text)
    input_tensor = torch.tensor([input_ids], device=device)

    if diversity_level == "low":
        beam_width, temperature, sampling_mode = 2, 0.6, False
    elif diversity_level == "high":
        beam_width, temperature, sampling_mode = 4, 0.95, True
    else:
        beam_width, temperature, sampling_mode = 3, 0.75, False

    reverse_diffusion.temperature = temperature
    reverse_diffusion.repetition_penalty = repetition_penalty
    reverse_diffusion.diversity_penalty = diversity_penalty
    reverse_diffusion.sampling_mode = sampling_mode
    reverse_diffusion.length_penalty = length_penalty

    generated_ids = reverse_diffusion.generate_beam(
        model,
        condition=input_tensor,
        beam_width=beam_width,
        num_steps=num_steps
    )

    mask_id = CONFIG["diffusion"]["mask_token_id"]
    tokens = [tid for tid in generated_ids[0].tolist() if tid != mask_id]

    return tokenizer.decode(tokens).strip()

# =========================================================
# RESULTS STORAGE
# =========================================================
os.makedirs("results1", exist_ok=True)

results_file = f"results1/inference_{CONFIG['model_type']}.json"

stored_results = []

if os.path.exists(results_file):

    with open(results_file, "r", encoding="utf-8") as f:
        stored_results = json.load(f)

# =========================================================
# GRADIO UI
# =========================================================
import gradio as gr

def gradio_generate(input_text, diversity, rp, dp, lp):

    try:

        out = generate_output(
            input_text,
            diversity,
            float(rp),
            float(dp),
            float(lp)
        )

        stored_results.append({
            "model": CONFIG["model_type"],
            "input": input_text,
            "output": out,
            "diversity": diversity,
            "repetition_penalty": float(rp),
            "diversity_penalty": float(dp),
            "length_penalty": float(lp)
        })

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(stored_results, f, ensure_ascii=False, indent=4)

        return out

    except Exception as e:
        return str(e)


with gr.Blocks(title="Sanskrit Diffusion Paraphraser") as demo:

    gr.Markdown("## Sanskrit Diffusion Paraphrase Generator")

    input_text = gr.Textbox(label="Input Sanskrit (IAST)")

    diversity = gr.Dropdown(
        ["low", "medium", "high"],
        value="medium",
        label="Diversity Level"
    )

    rp = gr.Number(value=1.15, label="Repetition Penalty")

    dp = gr.Number(value=0.0, label="Diversity Penalty")

    lp = gr.Number(value=1.0, label="Length Penalty")

    btn = gr.Button("Generate")

    output = gr.Textbox(label="Generated Output")

    btn.click(
        fn=gradio_generate,
        inputs=[input_text, diversity, rp, dp, lp],
        outputs=output
    )

# =========================================================
# LAUNCH
# =========================================================
if __name__ == "__main__":

    demo.launch()