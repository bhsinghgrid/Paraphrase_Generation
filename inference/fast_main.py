# # fastapi_inference.py
# from fastapi import FastAPI
# from pydantic import BaseModel
# from typing import Optional
# import torch
# import sys
# import os
#
# # ----------------------------
# # Load model modules
# # ----------------------------
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from model.tokenizer import SanskritTokenizer
# from model.new_d3pm_model import SanskritModel
# from diffusion.reverse_process import ReverseDiffusion
# from diffusion.scheduler import OptimizedCosineScheduler
# from types import MethodType
# import torch.nn.functional as F
#
# # ----------------------------
# # CONFIG
# # ----------------------------
# CONFIG = {
#     "model_type": "d3pm_encoder_decoder",
#     "model": {
#         "vocab_size": 16000,
#         "max_seq_len": 80,
#         "diffusion_steps": 8,
#         "d_model": 384,
#         "n_layers": 6,
#         "n_heads": 8,
#         "d_ff": 1536,
#         "dropout": 0.1
#     },
#     "diffusion": {"mask_token_id": 0},
#     "training": {"precision": "float32", "device": "cpu"}  # set "cuda" if available
# }
#
# device = torch.device(CONFIG["training"]["device"])
# dtype = torch.float16 if CONFIG["training"]["precision"]=="float16" else torch.float32
#
# # ----------------------------
# # FastAPI app
# # ----------------------------
# app = FastAPI(title="Sanskrit Paraphrase Generator API")
#
# # ----------------------------
# # Input schema
# # ----------------------------
# class ParaphraseRequest(BaseModel):
#     text: str
#     diversity_level: Optional[str] = "medium"  # low, medium, high
#     repetition_penalty: Optional[float] = 1.15
#     diversity_penalty: Optional[float] = 0.0
#     length_penalty: Optional[float] = 1.0
#     model_type: Optional[str] = "d3pm_encoder_decoder"  # allow switching models
#
# # ----------------------------
# # Load model once
# # ----------------------------
# def load_model(model_type="d3pm_encoder_decoder"):
#     CONFIG["model_type"] = model_type
#     tokenizer = SanskritTokenizer(CONFIG["model"]["vocab_size"])
#     model = SanskritModel(CONFIG)
#     model_path = os.path.join(
#         os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
#         "production_model3",
#         f"best_{CONFIG['model_type']}.pt"
#     )
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Model not found: {model_path}")
#
#     model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
#     model.to(device, dtype=dtype)
#     model.eval()
#     model.mask_token_id = CONFIG["diffusion"]["mask_token_id"]
#
#     # safe forward
#     original_forward = model.forward
#     def safe_forward(*args, **kwargs):
#         out = original_forward(*args, **kwargs)
#         if not isinstance(out, tuple):
#             return out, None
#         return out
#     model.forward = safe_forward
#
#     # scheduler + reverse diffusion
#     scheduler = OptimizedCosineScheduler(CONFIG)
#     reverse_diffusion = ReverseDiffusion(scheduler)
#     patch_p_sample_step(reverse_diffusion)
#     return model, tokenizer, reverse_diffusion
#
# # ----------------------------
# # Patch p_sample_step (same hybrid + bigram)
# # ----------------------------
# def patch_p_sample_step(reverse_diffusion):
#     old_p_sample_step = reverse_diffusion.p_sample_step
#     def p_sample_step_override(self, *args, **kwargs):
#         model_arg = kwargs.get("model") or (args[0] if len(args)>0 else None)
#         x_t = kwargs.get("x_t") or (args[1] if len(args)>1 else None)
#         t = kwargs.get("t") or (args[2] if len(args)>2 else None)
#         condition = kwargs.get("condition") or (args[3] if len(args)>3 else None)
#         beam_width = int(kwargs.get("beam_width", 1))
#         if any(v is None for v in [model_arg, x_t, t, condition]):
#             return old_p_sample_step(*args, **kwargs)
#
#         if x_t.dim()==1: x_t = x_t.unsqueeze(0)
#         if condition.dim()==1: condition = condition.unsqueeze(0)
#         if isinstance(t,int): t=torch.tensor([t], device=condition.device)
#         if t.dim()==0: t=t.unsqueeze(0)
#         if t.dim()==1 and t.shape[0]!=x_t.shape[0]: t=t.expand(x_t.shape[0])
#
#         logits,_ = model_arg(condition, x_t, t) if isinstance(model_arg.forward(condition, x_t, t),tuple) else (model_arg(condition,x_t,t),None)
#
#         temperature = getattr(self,"temperature",1.0)
#         repetition_penalty = getattr(self,"repetition_penalty",1.0)
#         diversity_penalty = getattr(self,"diversity_penalty",0.0)
#         sampling_mode = getattr(self,"sampling_mode",False)
#
#         if diversity_penalty>0.0:
#             logits += diversity_penalty * logits.var(dim=-1, keepdim=True)
#
#         B,L,V = logits.shape
#         for b in range(B):
#             prev_tokens = x_t[b].view(-1)
#             for tok in prev_tokens.unique():
#                 tok_i = int(tok.item())
#                 if tok_i==CONFIG["diffusion"]["mask_token_id"]:
#                     continue
#                 logits[b,:,tok_i] /= repetition_penalty
#
#         logits = logits / max(temperature,1e-6)
#         probs = F.softmax(logits, dim=-1)
#         B,L,V = probs.shape
#         probs_flat = probs.view(-1,V)
#         candidates=[]
#         for k in range(beam_width):
#             if sampling_mode:
#                 topk_probs,topk_ids=torch.topk(probs_flat,min(50,V),dim=-1)
#                 topk_probs=topk_probs/(topk_probs.sum(dim=-1,keepdim=True)+1e-12)
#                 next_ids=torch.multinomial(topk_probs,1).squeeze(-1)
#                 next_tokens=topk_ids[torch.arange(topk_ids.size(0)),next_ids].view(B,L)
#             else:
#                 topk_probs,topk_ids=torch.topk(probs_flat,beam_width,dim=-1)
#                 next_tokens=topk_ids[:,k].view(B,L)
#             # no repeat bigram
#             for b in range(B):
#                 tok_seq = next_tokens[b].tolist()
#                 seen=set()
#                 for i in range(len(tok_seq)-1):
#                     bigram=(tok_seq[i], tok_seq[i+1])
#                     if bigram in seen:
#                         tok_seq[i+1]=CONFIG["diffusion"]["mask_token_id"]
#                     else:
#                         seen.add(bigram)
#                 next_tokens[b]=torch.tensor(tok_seq, device=next_tokens.device)
#             chosen_probs=probs_flat[torch.arange(probs_flat.size(0)),next_tokens.view(-1)]
#             score=torch.log(chosen_probs+1e-12).sum().item()
#             candidates.append((next_tokens, score))
#         return candidates
#
#     reverse_diffusion.p_sample_step = MethodType(p_sample_step_override, reverse_diffusion)
#
# # ----------------------------
# # Load default model at startup
# # ----------------------------
# model, tokenizer, reverse_diffusion = load_model(CONFIG["model_type"])
#
# # ----------------------------
# # Helper generate function
# # ----------------------------
# @torch.no_grad()
# def generate_sanskrit(text, diversity="medium", repetition_penalty=1.15, diversity_penalty=0.0, length_penalty=1.0):
#     input_ids = tokenizer.encode(text)
#     input_tensor = torch.tensor([input_ids], device=device)
#
#     if diversity=="low":
#         beam_width, temperature, sampling_mode = 2,0.6,False
#     elif diversity=="high":
#         beam_width, temperature, sampling_mode = 4,0.95,True
#     else:
#         beam_width, temperature, sampling_mode = 3,0.75,False
#
#     reverse_diffusion.temperature=temperature
#     reverse_diffusion.repetition_penalty=max(1.0,repetition_penalty)
#     reverse_diffusion.diversity_penalty=max(0.0,diversity_penalty)
#     reverse_diffusion.sampling_mode=sampling_mode
#     reverse_diffusion.length_penalty=max(0.0,length_penalty)
#
#     generated_ids = reverse_diffusion.generate_beam(
#         model, condition=input_tensor, beam_width=beam_width, num_steps=CONFIG["model"]["diffusion_steps"]
#     )
#     mask_id = CONFIG["diffusion"]["mask_token_id"]
#     tokens = [tid for tid in generated_ids[0].tolist() if tid!=mask_id]
#     return tokenizer.decode(tokens).strip()
#
# # ----------------------------
# # FastAPI endpoint
# # ----------------------------
# @app.post("/generate")
# def generate(request: ParaphraseRequest):
#     output_text = generate_sanskrit(
#         request.text,
#         diversity=request.diversity_level,
#         repetition_penalty=request.repetition_penalty,
#         diversity_penalty=request.diversity_penalty,
#         length_penalty=request.length_penalty
#     )
#     return {"input": request.text, "output": output_text}
# gradio_inference.py
import torch
import torch.nn.functional as F
from types import MethodType
import os
import sys

# ----------------------------
# CONFIG (default, can override in frontend)
# ----------------------------
CONFIG = {
    # "model_type": "d3pm_encoder_decoder",  # default
    "model_type": "d3pm_cross_attention",
    "model": {
        "vocab_size": 16000,
        "max_seq_len": 80,
        "diffusion_steps": 8,
        "d_model": 384,
        "n_layers": 6,
        "n_heads": 8,
        "d_ff": 1536,
        "dropout": 0.1
    },
    "diffusion": {"mask_token_id": 0},
    "training": {"precision": "float32", "device": "mps"}  # or "cuda"/"cpu"
}

device = torch.device(CONFIG["training"]["device"])
dtype = torch.float16 if CONFIG["training"]["precision"] == "float16" else torch.float32

# ----------------------------
# IMPORT PROJECT MODULES
# ----------------------------

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.tokenizer import SanskritTokenizer
from model.new_d3pm_model import SanskritModel
from diffusion.reverse_process import ReverseDiffusion

from diffusion.scheduler import OptimizedCosineScheduler

# ----------------------------
# GLOBAL: load model + tokenizer once
# ----------------------------
def load_model(model_type="d3pm_encoder_decoder"):
    CONFIG["model_type"] = model_type
    tokenizer = SanskritTokenizer(CONFIG["model"]["vocab_size"])
    model = SanskritModel(CONFIG)
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "production_model3",
        f"best_{CONFIG['model_type']}.pt"
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.to(device, dtype=dtype)
    model.eval()
    model.mask_token_id = CONFIG["diffusion"]["mask_token_id"]

    # safe forward
    original_forward = model.forward
    def safe_forward(*args, **kwargs):
        output = original_forward(*args, **kwargs)
        if not isinstance(output, tuple):
            return output, None
        return output
    model.forward = safe_forward

    # scheduler + reverse diffusion
    scheduler = OptimizedCosineScheduler(CONFIG)
    reverse_diffusion = ReverseDiffusion(scheduler)
    patch_p_sample_step(reverse_diffusion)
    return model, tokenizer, reverse_diffusion

# ----------------------------
# PATCHED P_SAMPLE_STEP (same as hybrid + bigram blocking)
# ----------------------------
def patch_p_sample_step(reverse_diffusion):
    old_p_sample_step = reverse_diffusion.p_sample_step

    def p_sample_step_override(self, *args, **kwargs):
        model_arg = kwargs.get("model") or (args[0] if len(args) > 0 else None)
        x_t = kwargs.get("x_t") or (args[1] if len(args) > 1 else None)
        t = kwargs.get("t") or (args[2] if len(args) > 2 else None)
        condition = kwargs.get("condition") or (args[3] if len(args) > 3 else None)
        beam_width = int(kwargs.get("beam_width", 1))
        if any(v is None for v in [model_arg, x_t, t, condition]):
            return old_p_sample_step(*args, **kwargs)

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

        if diversity_penalty > 0.0:
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
        B, L, V = probs.shape
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

# ----------------------------
# MAIN GENERATION FUNCTION
# ----------------------------
@torch.no_grad()
def generate_text(input_text: str,
                  model, tokenizer, reverse_diffusion,
                  diversity_level="medium",
                  repetition_penalty=1.15,
                  diversity_penalty=0.0,
                  length_penalty=1.0):
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
        model, condition=input_tensor, beam_width=beam_width, num_steps=CONFIG["model"]["diffusion_steps"]
    )

    mask_id = CONFIG["diffusion"]["mask_token_id"]
    tokens = [tid for tid in generated_ids[0].tolist() if tid != mask_id]
    output_text = tokenizer.decode(tokens).strip()
    return output_text

import os
import json
from datetime import datetime
import torch
import torch.nn.functional as F
from types import MethodType

# ----------------------------
# Directory to save generated JSON results1
# ----------------------------
RESULTS_DIR = "generated_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Create a timestamped JSON file per session
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_FILE = os.path.join(RESULTS_DIR, f"{CONFIG['model_type']}_results_{timestamp}.json")

# Initialize the JSON file with an empty list if it does not exist
if not os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=4)

# ----------------------------
# Modified generate_text function to save results1 as JSON
# ----------------------------
@torch.no_grad()
def generate_text_and_save_json(input_text: str,
                                model, tokenizer, reverse_diffusion,
                                diversity_level="medium",
                                repetition_penalty=1.15,
                                diversity_penalty=0.0,
                                length_penalty=1.0):
    # Generate output using your existing generate_text function
    output_text = generate_text(
        input_text,
        model=model,
        tokenizer=tokenizer,
        reverse_diffusion=reverse_diffusion,
        diversity_level=diversity_level,
        repetition_penalty=repetition_penalty,
        diversity_penalty=diversity_penalty,
        length_penalty=length_penalty
    )

    # Load existing JSON data
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Append new result
    data.append({
        "model_type": CONFIG["model_type"],
        "input_text": input_text,
        "output_text": output_text,
        "diversity_level": diversity_level,
        "repetition_penalty": repetition_penalty,
        "diversity_penalty": diversity_penalty,
        "length_penalty": length_penalty,
        "timestamp": datetime.now().isoformat()
    })

    # Save back to JSON
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return output_text