# # import torch
# # import os
# # import json
# # import sys
# # # =========================================================
# # # CONFIG
# # # =========================================================
# # CONFIG = {
# #     "model_type": "d3pm_cross_attention",  # change to any model
# #     "model": {
# #         "vocab_size": 16000,
# #         "max_seq_len": 80,
# #         "diffusion_steps": 8,
# #         "d_model": 384,
# #         "n_layers": 6,
# #         "n_heads": 8,
# #         "d_ff": 1536,
# #         "dropout": 0.1
# #     },
# #     "diffusion": {
# #         "mask_token_id": 0
# #     },
# #     "training": {
# #         "precision": "float32",
# #         "device": "mps"  # "cuda" or "cpu"
# #     }
# # }
# #
# # # =========================================================
# # # DEVICE & DTYPE
# # # =========================================================
# # device = torch.device(CONFIG["training"]["device"])
# # dtype = torch.float16 if CONFIG["training"]["precision"] == "float16" else torch.float32
# #
# # # =========================================================
# # # IMPORT MODEL & TOKENIZER (same folder)
# # # =========================================================
# # # Add project root to sys.path
# # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# #
# # from model.tokenizer import SanskritTokenizer
# # from model.new_d3pm_model import SanskritModel  # or baseline_cross_attention if used
# #
# # # =========================================================
# # # LOAD TOKENIZER & MODEL
# # # =========================================================
# # tokenizer = SanskritTokenizer(CONFIG["model"]["vocab_size"])
# # model = SanskritModel(CONFIG)
# #
# # # Model path in same folder
# # # model_path = f"production_model3/best_{CONFIG['model_type']}.pt"
# # model_path = f"/Users/bhsingh/Documents/Generation/production_model3/best_d3pm_cross_attention.pt"
# # if not os.path.exists(model_path):
# #     raise FileNotFoundError(f"Model not found: {model_path}")
# #
# # model.load_state_dict(torch.load(model_path, map_location=device))
# # model.to(device, dtype=dtype)
# # model.eval()
# #
# # mask_token_id = CONFIG["diffusion"]["mask_token_id"]
# # num_steps = CONFIG["model"]["diffusion_steps"]
# #
# # # =========================================================
# # # GENERATION FUNCTION
# # # =========================================================
# # @torch.no_grad()
# # def generate_output(model, tokenizer, input_text):
# #     input_ids = tokenizer.encode(input_text)
# #     input_tensor = torch.tensor([input_ids], device=device)
# #
# #     # Random timestep if diffusion
# #     t = torch.randint(0, num_steps, (1,), device=device)
# #
# #     # Forward pass
# #     logits, _ = model(input_tensor, input_tensor, t) if "d3pm" in CONFIG["model_type"] else model(input_tensor, input_tensor)
# #
# #     # Greedy decoding
# #     preds = torch.argmax(logits, dim=-1)
# #     output_text = tokenizer.decode([id for id in preds[0].tolist() if id != mask_token_id]).strip()
# #     return output_text
# #
# # # =========================================================
# # # INTERACTIVE PROMPT
# # # =========================================================
# # os.makedirs("results1", exist_ok=True)
# # results_file = f"results1/inference_{CONFIG['model_type']}.json"
# #
# # # Load previous results1 if exist
# # if os.path.exists(results_file):
# #     with open(results_file, "r", encoding="utf-8") as f:
# #         stored_results = json.load(f)
# # else:
# #     stored_results = []
# #
# # print(f"📝 Enter input for model '{CONFIG['model_type']}' (type 'quit' to exit):")
# #
# # while True:
# #     input_text = input("> ")
# #     if input_text.lower() in ["quit", "exit"]:
# #         break
# #
# #     output_text = generate_output(model, tokenizer, input_text)
# #     print("✅ Output:", output_text)
# #
# #     # Store input/output/model
# #     stored_results.append({
# #         "model": CONFIG["model_type"],
# #         "input": input_text,
# #         "output": output_text
# #     })
# #
# #     # Save to JSON after each input
# #     with open(results_file, "w", encoding="utf-8") as f:
# #         json.dump(stored_results, f, ensure_ascii=False, indent=4)
# #
# # print(f"\n💾 All inputs and outputs saved to {results_file}")
# import torch
# import os
# import json
# import sys
#
# # =========================================================
# # CONFIG
# # =========================================================
# CONFIG = {
#     # "model_type": "d3pm_cross_attention",
#     "model_type": "d3pm_encoder_decoder",
#     # "model_type": "baseline_encoder_decoder",
#     "model": {
#         "vocab_size": 16000,
#         "max_seq_len": 80,
#         "diffusion_steps": 12,
#         "d_model": 384,
#         "n_layers": 6,
#         "n_heads": 8,
#         "d_ff": 1536,
#         "dropout": 0.1
#     },
#     "diffusion": {
#         "mask_token_id": 0
#     },
#     "training": {
#         "precision": "float32",
#         "device": "mps"  # or cuda / cpu
#     }
# }
#
# # =========================================================
# # DEVICE
# # =========================================================
# device = torch.device(CONFIG["training"]["device"])
# dtype = torch.float16 if CONFIG["training"]["precision"] == "float16" else torch.float32
#
# # =========================================================
# # IMPORT PROJECT MODULES
# # =========================================================
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#
# from model.tokenizer import SanskritTokenizer
# from model.new_d3pm_model import SanskritModel
# from diffusion.reverse_process import ReverseDiffusion
# from diffusion.scheduler import OptimizedCosineScheduler
#
# # =========================================================
# # LOAD TOKENIZER & MODEL
# # =========================================================
# tokenizer = SanskritTokenizer(CONFIG["model"]["vocab_size"])
# model = SanskritModel(CONFIG)
#
# model_path = os.path.join(
#     os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
#     "production_model3",
#     f"best_{CONFIG['model_type']}.pt"
# )
#
# if not os.path.exists(model_path):
#     raise FileNotFoundError(f"Model not found: {model_path}")
#
# model.load_state_dict(torch.load(model_path, map_location=device))
# model.to(device, dtype=dtype)
# model.eval()
#
# mask_token_id = CONFIG["diffusion"]["mask_token_id"]
# num_steps = CONFIG["model"]["diffusion_steps"]
#
# # =========================================================
# # FIX 1: SAFE MODEL FORWARD (unpack error fix)
# # =========================================================
# original_forward = model.forward
#
# def safe_forward(*args, **kwargs):
#     output = original_forward(*args, **kwargs)
#
#     # If only logits returned, wrap into tuple
#     if not isinstance(output, tuple):
#         return output, None
#
#     return output
#
# model.forward = safe_forward
#
# # =========================================================
# # REVERSE DIFFUSION SETUP
# # =========================================================
# scheduler = OptimizedCosineScheduler(CONFIG)
# reverse_diffusion = ReverseDiffusion(scheduler)
#
# # =========================================================
# # FIX 2: SAFE TIMESTEP PATCH (expand crash fix)
# # =========================================================
# old_p_sample_step = reverse_diffusion.p_sample_step
#
# # def safe_p_sample_step(model, condition, x_t, t, beam_width):
# #     B = condition.shape[0]
# #
# #     # Convert to tensor
# #     if not isinstance(t, torch.Tensor):
# #         t = torch.tensor(t, device=condition.device)
# #
# #     t = t.to(condition.device)
# #
# #     # Ensure shape (B,)
# #     if t.numel() == 1:
# #         value = int(t.reshape(-1)[0].item())
# #         t = torch.full((B,), value, device=condition.device, dtype=torch.long)
# #     else:
# #         t = t.reshape(-1)
# #         if t.shape[0] != B:
# #             value = int(t[0].item())
# #             t = torch.full((B,), value, device=condition.device, dtype=torch.long)
# #         else:
# #             t = t.long()
# #
# #     return old_p_sample_step(model, condition, x_t, t, beam_width)
# #
# # reverse_diffusion.p_sample_step = safe_p_sample_step
#
# # =========================================================
# # GENERATION FUNCTION
# # =========================================================
# @torch.no_grad()
# def generate_output(input_text, beam_width=3):
#
#     # Encode + force batch dimension
#     input_ids = tokenizer.encode(input_text)
#     input_tensor = torch.tensor([input_ids], device=device)
#
#     # Beam reverse diffusion
#     generated_ids = reverse_diffusion.generate_beam(
#         model,
#         condition=input_tensor,
#         beam_width=beam_width,
#         num_steps=num_steps
#     )
#
#     # Decode removing mask tokens
#     output_text = tokenizer.decode(
#         [id for id in generated_ids[0].tolist() if id != mask_token_id]
#     ).strip()
#
#     return output_text
#
# # =========================================================
# # INTERACTIVE LOOP
# # =========================================================
# os.makedirs("results1", exist_ok=True)
# results_file = f"results1/inference_{CONFIG['model_type']}.json"
#
# if os.path.exists(results_file):
#     with open(results_file, "r", encoding="utf-8") as f:
#         stored_results = json.load(f)
# else:
#     stored_results = []
#
# print(f"\n📝 Model: {CONFIG['model_type']}")
# print("Type 'quit' to exit.\n")
#
# while True:
#     input_text = input("> ").strip()
#
#     if input_text.lower() in ["quit", "exit"]:
#         break
#
#     try:
#         output_text = generate_output(input_text, beam_width=3)
#         print("✅ Output:", output_text)
#
#         stored_results.append({
#             "model": CONFIG["model_type"],
#             "input": input_text,
#             "output": output_text
#         })
#
#         with open(results_file, "w", encoding="utf-8") as f:
#             json.dump(stored_results, f, ensure_ascii=False, indent=4)
#
#     except Exception as e:
#         print("❌ Inference Error:", str(e))
#
# print(f"\n💾 Saved to {results_file}")

# full_inference.py
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
    "model_type": "d3pm_encoder_decoder",  # default, overridden by user
    "model": {
        "vocab_size": 16000,
        "max_seq_len": 80,
        "diffusion_steps": 8,  # reduced for better semantic similarity
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
        "device": "mps"  # or "cuda"/"cpu"
    }
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
from model.new_d3pm_model import SanskritModel
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

model_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "production_model3",
    f"best2_{CONFIG['model_type']}.pt"
)
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found: {model_path}")

model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
model.to(device, dtype=dtype)
model.eval()
model.mask_token_id = CONFIG["diffusion"]["mask_token_id"]  # fix mask_token_id access

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
# PATCHED p_sample_step: hybrid top-k + multinomial, penalties, no-repeat bigrams
# =========================================================
def p_sample_step_override(self, *args, **kwargs):
    model_arg = kwargs.get("model") or (args[0] if len(args) > 0 else None)
    x_t = kwargs.get("x_t") or (args[1] if len(args) > 1 else None)
    t = kwargs.get("t") or (args[2] if len(args) > 2 else None)
    condition = kwargs.get("condition") or (args[3] if len(args) > 3 else None)
    beam_width = int(kwargs.get("beam_width", 1))

    if any(v is None for v in [model_arg, x_t, t, condition]):
        raise RuntimeError("Cannot resolve p_sample_step arguments for override")

    if x_t.dim() == 1: x_t = x_t.unsqueeze(0)
    if condition.dim() == 1: condition = condition.unsqueeze(0)
    if isinstance(t, int): t = torch.tensor([t], device=condition.device)
    if t.dim() == 0: t = t.unsqueeze(0)
    if t.dim() == 1 and t.shape[0] != x_t.shape[0]: t = t.expand(x_t.shape[0])

    logits, _ = model_arg(condition, x_t, t) if isinstance(model_arg.forward(condition, x_t, t), tuple) else (model_arg(condition, x_t, t), None)

    temperature = getattr(self, "temperature", 1.0)
    repetition_penalty = getattr(self, "repetition_penalty", 1.0)
    diversity_penalty = getattr(self, "diversity_penalty", 0.0)
    sampling_mode = getattr(self, "sampling_mode", False)

    # diversity penalty (variance)
    if diversity_penalty > 0.0:
        logits += diversity_penalty * logits.var(dim=-1, keepdim=True)

    # repetition penalty
    B, L, V = logits.shape
    for b in range(B):
        prev_tokens = x_t[b].view(-1)
        for tok in prev_tokens.unique():
            tok_i = int(tok.item())
            if tok_i == CONFIG["diffusion"]["mask_token_id"]:
                continue
            logits[b, :, tok_i] /= repetition_penalty

    # temperature
    logits = logits / max(temperature, 1e-6)

    # probs
    probs = F.softmax(logits, dim=-1)
    B, L, V = probs.shape
    probs_flat = probs.view(-1, V)
    candidates = []

    for k in range(beam_width):
        if sampling_mode:
            # sample from top-50 tokens
            topk_probs, topk_ids = torch.topk(probs_flat, min(50, V), dim=-1)
            topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-12)
            next_ids = torch.multinomial(topk_probs, 1).squeeze(-1)
            next_tokens = topk_ids[torch.arange(topk_ids.size(0)), next_ids].view(B, L)
        else:
            topk_probs, topk_ids = torch.topk(probs_flat, beam_width, dim=-1)
            next_tokens = topk_ids[:, k].view(B, L)

        # No-repeat bigram blocking
        for b in range(B):
            tok_seq = next_tokens[b].tolist()
            seen_bigrams = set()
            for i in range(len(tok_seq)-1):
                bigram = (tok_seq[i], tok_seq[i+1])
                if bigram in seen_bigrams:
                    tok_seq[i+1] = CONFIG["diffusion"]["mask_token_id"]
                else:
                    seen_bigrams.add(bigram)
            next_tokens[b] = torch.tensor(tok_seq, device=next_tokens.device)

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
    reverse_diffusion.repetition_penalty = max(1.0, repetition_penalty)
    reverse_diffusion.diversity_penalty = max(0.0, diversity_penalty)
    reverse_diffusion.sampling_mode = sampling_mode
    reverse_diffusion.length_penalty = max(0.0, length_penalty)

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
# INTERACTIVE LOOP
# =========================================================
os.makedirs("results1", exist_ok=True)
results_file = f"results1/inference_{CONFIG['model_type']}.json"

stored_results = []
if os.path.exists(results_file):
    with open(results_file, "r", encoding="utf-8") as f:
        stored_results = json.load(f)

print(f"\n📝 Model: {CONFIG['model_type']} (device={device})")
print("Type 'quit' to exit.\n")

while True:
    inp = input("> ").strip()
    if inp.lower() in ("quit", "exit") or not inp:
        break

    diversity_level = input("Choose diversity (low/medium/high) [medium]: ").strip().lower()
    if diversity_level not in ("low", "medium", "high"):
        diversity_level = "medium"

    rp = input("Repetition penalty (>=1.0) [1.15]: ").strip()
    dp = input("Diversity penalty (>=0.0) [0.0]: ").strip()
    lp = input("Length penalty (>=0.0) [1.0]: ").strip()
    try:
        repetition_penalty = float(rp) if rp else 1.15
    except: repetition_penalty = 1.15
    try:
        diversity_penalty = float(dp) if dp else 0.0
    except: diversity_penalty = 0.0
    try:
        length_penalty = float(lp) if lp else 1.0
    except: length_penalty = 1.0

    try:
        out = generate_output(inp, diversity_level, repetition_penalty, diversity_penalty, length_penalty)
        print("✅ Output:", out)

        stored_results.append({
            "model": CONFIG["model_type"],
            "input": inp,
            "output": out,
            "diversity": diversity_level,
            "repetition_penalty": repetition_penalty,
            "diversity_penalty": diversity_penalty,
            "length_penalty": length_penalty
        })

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(stored_results, f, ensure_ascii=False, indent=4)

    except Exception as e:
        print("❌ Inference Error:", str(e))

print(f"\n💾 All inputs and outputs saved to {results_file}")