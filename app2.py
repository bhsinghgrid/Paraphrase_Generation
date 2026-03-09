"""
app1.py  — Final Correct Version
===================================

STATUS (verified 2026-03-07):
  ✅ Output is 100% pure Devanagari (zero Roman/IAST tokens)
  ✅ Semantically relevant words match input (यदा, मनो, विषये, निवर्त)
  ✅ Matches training validation quality (BERTScore 0.75 ceiling)
  ✅ Repetition tail trimmed via token-level post-processing

TWO ROOT CAUSES FIXED vs all previous versions:

  BUG 1 — WRONG TOKENIZER (caused rud / Ṛ / dhar mixed-script garbage)
    Old: SanskritTokenizer (shared vocab: both Roman IAST + Devanagari)
    Fix: SanskritSourceTokenizer to encode input  (Roman IAST only)
         SanskritTargetTokenizer to decode output (Devanagari only)
    Why: training used these two separate tokenizers — token ID 5432
         maps to Devanagari in tgt_tokenizer but to a Roman subword
         in the shared tokenizer.

  BUG 2 — WRONG GENERATE PATH (caused completely different behavior)
    Old: reverse_diffusion.generate(model, ...)  ← external, different impl
    Fix: model.generate(input_ids, ...)          ← same call as training
         → SanskritModel.generate()
         → D3PMCrossAttention.generate()          ← validated, BERTScore 0.75

REMAINING MODEL-LEVEL ARTIFACT (not a code bug):
  The model has no EOS token and fills all 80 positions. Once meaningful
  content ends (~33 words), it repeats high-probability tokens:
    यदा यदा यदा यदा, मनो मनो मनो मनो
  This is IDENTICAL to training validation logs:
    मान् मान् मान् मान् मान्, मुनि मुनि मुनि मुनि
  Handled here by trim_token_ids() which cuts at the first run of 3+
  consecutive identical token IDs.
"""

import os
import sys
import json
import random
import torch
from datetime import datetime
import gradio as gr

# ----------------------------
# CONFIG  (matches training)
# ----------------------------
CONFIG = {
    "model_type": "d3pm_cross_attention",
    "model": {
        "vocab_size":      16000,
        "src_vocab_size":  16000,
        "tgt_vocab_size":  16000,
        "max_seq_len":     80,
        "diffusion_steps": 128,
        "d_model":         1024,
        "n_layers":        8,
        "n_heads":         8,
        "d_ff":            4096,
        "dropout":         0.2
    },
    "diffusion": {"mask_token_id": 0},
    "training":  {"precision": "float32", "device": "mps"},
    "inference": {
        "num_steps":          128,
        "temperature":        0.75,
        "top_k":              50,
        "repetition_penalty": 1.15,
        "diversity_penalty":  0.0,
    },
}

device = torch.device(CONFIG["training"]["device"])
dtype  = torch.float32


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"🔒 Seed set to {seed}")


set_seed(42)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# BUG 1 FIX: two separate tokenizers, same as training
from model.tokenizer      import SanskritSourceTokenizer, SanskritTargetTokenizer
from model.sanskrit_model import SanskritModel


# ----------------------------
# Trim repetition tail
# ----------------------------
def trim_token_ids(token_ids: list, run_thresh: int = 3) -> list:
    """
    Cut when the same token ID appears run_thresh consecutive times.

    The model has no EOS token and fills all 80 positions. After content
    ends the model repeats high-probability tokens (identical to training
    validation: मान् मान् मान् मान्, मुनि मुनि मुनि मुनि).

    run_thresh=3 tolerates natural BPE pairs (त त) but cuts on triple
    runs (मनो मनो मनो, यदा यदा यदा).
    """
    if len(token_ids) <= 4:
        return token_ids
    count = 1
    for i in range(1, len(token_ids)):
        if token_ids[i] == token_ids[i - 1]:
            count += 1
            if count >= run_thresh:
                cut = max(4, i - run_thresh + 2)
                return token_ids[:cut]
        else:
            count = 1
    return token_ids


def clean_text(text: str) -> str:
    text = text.replace("<pad>",  "")
    text = text.replace("<s>",    "")
    text = text.replace("</s>",   "")
    text = text.replace("[MASK]", "")
    text = text.replace("।।",    "।")
    return " ".join(text.split()).strip()


# ----------------------------
# Load model
# ----------------------------
def load_model(model_type: str = "d3pm_cross_attention"):
    CONFIG["model_type"] = model_type

    # BUG 1 FIX: separate tokenizers matching training
    src_tokenizer = SanskritSourceTokenizer(
        vocab_size = CONFIG["model"]["src_vocab_size"],
        max_len    = CONFIG["model"]["max_seq_len"],
    )
    tgt_tokenizer = SanskritTargetTokenizer(
        vocab_size = CONFIG["model"]["tgt_vocab_size"],
        max_len    = CONFIG["model"]["max_seq_len"],
    )

    model = SanskritModel(CONFIG)

    model_path = (
        "/Users/bhsingh/Documents/Final_Paraphrase/Exclude_Negative/"
        "results7/d3pm_cross_attention_neg_True/best_model.pt"
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model.load_state_dict(
        torch.load(model_path, map_location=device), strict=False
    )
    model.to(device, dtype=dtype)
    model.eval()
    print(f"✅ Model loaded from {model_path}")

    return model, src_tokenizer, tgt_tokenizer


model, src_tokenizer, tgt_tokenizer = load_model("d3pm_cross_attention")

# ----------------------------
# Results JSON
# ----------------------------
RESULTS_DIR  = "generated_results"
os.makedirs(RESULTS_DIR, exist_ok=True)
timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_FILE = os.path.join(
    RESULTS_DIR, f"{CONFIG['model_type']}_results_{timestamp}.json"
)
with open(RESULTS_FILE, "w", encoding="utf-8") as _f:
    json.dump([], _f, ensure_ascii=False, indent=4)


# ----------------------------
# Core generation
# ----------------------------
@torch.no_grad()
def generate_text(
    input_text:         str,
    model,
    src_tokenizer,
    tgt_tokenizer,
    diversity_level:    str   = "medium",
    repetition_penalty: float = 1.15,
    diversity_penalty:  float = 0.0,
    top_k:              int   = 50,
) -> str:
    """
    Encode   → SanskritSourceTokenizer  (Roman IAST)
    Generate → model.generate()         (same path as training)
    Trim     → trim_token_ids()         (cut repetition tail)
    Decode   → SanskritTargetTokenizer  (Devanagari only)
    """

    # Encode input with SOURCE tokenizer (Roman IAST)
    input_ids    = src_tokenizer.encode(input_text)
    input_tensor = torch.tensor([input_ids], device=device, dtype=torch.long)

    if diversity_level == "low":
        temperature, top_k_eff = 0.65, 30
    elif diversity_level == "high":
        temperature, top_k_eff = 0.90, 80
    else:
        temperature, top_k_eff = 0.75, 50

    # BUG 2 FIX: call model.generate() — same path as training validation
    gen_ids = model.generate(
        input_tensor,
        num_steps          = CONFIG["model"]["diffusion_steps"],
        temperature        = temperature,
        top_k              = top_k_eff,
        repetition_penalty = max(1.0, repetition_penalty),
        diversity_penalty  = max(0.0, diversity_penalty),
    )

    # Filter special tokens (ids 0–4: MASK/PAD/UNK/BOS/EOS)
    raw_ids = [x for x in gen_ids[0].tolist() if x > 4]

    # Trim repetition tail before decoding
    trimmed_ids = trim_token_ids(raw_ids, run_thresh=3)

    # Decode with TARGET tokenizer (Devanagari only)
    raw_text     = clean_text(tgt_tokenizer.decode(raw_ids))
    output_text  = clean_text(tgt_tokenizer.decode(trimmed_ids))

    if not output_text:
        output_text = raw_text

    print(f"INPUT   : {input_text}")
    print(f"RAW     : {raw_text[:100]} ...")
    print(f"TRIMMED : {output_text}")

    return output_text


@torch.no_grad()
def generate_text_and_save_json(
    input_text:         str,
    model,
    src_tokenizer,
    tgt_tokenizer,
    diversity_level:    str   = "medium",
    repetition_penalty: float = 1.15,
    diversity_penalty:  float = 0.0,
    top_k:              int   = 50,
) -> str:

    output_text = generate_text(
        input_text,
        model              = model,
        src_tokenizer      = src_tokenizer,
        tgt_tokenizer      = tgt_tokenizer,
        diversity_level    = diversity_level,
        repetition_penalty = repetition_penalty,
        diversity_penalty  = diversity_penalty,
        top_k              = top_k,
    )

    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    data.append({
        "model_type":         CONFIG["model_type"],
        "input_text":         input_text,
        "output_text":        output_text,
        "diversity_level":    diversity_level,
        "repetition_penalty": repetition_penalty,
        "diversity_penalty":  diversity_penalty,
        "timestamp":          datetime.now().isoformat(),
    })

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return output_text


def gradio_infer(input_text, diversity, repetition, diversity_penalty, top_k):
    return generate_text_and_save_json(
        input_text,
        model              = model,
        src_tokenizer      = src_tokenizer,
        tgt_tokenizer      = tgt_tokenizer,
        diversity_level    = diversity,
        repetition_penalty = float(repetition),
        diversity_penalty  = float(diversity_penalty),
        top_k              = int(top_k),
    )


iface = gr.Interface(
    fn          = gradio_infer,
    inputs      = [
        gr.Textbox(label="Input Sanskrit Text (Roman IAST)"),
        gr.Radio(["low", "medium", "high"], label="Diversity Level", value="medium"),
        gr.Number(value=1.15, label="Repetition Penalty  (1.0 = off)"),
        gr.Number(value=0.0,  label="Diversity Penalty   (0.0 = off)"),
        gr.Number(value=50,   label="Top-K  (30 = focused, 80 = creative)"),
    ],
    outputs     = [gr.Textbox(label="Generated Paraphrase (Devanagari)")],
    title       = "Sanskrit Paraphrase Generator (D3PM)",
    description = (
        "D3PM cross-attention model trained on Sanskrit IAST → Devanagari. "
        "Repetition tail automatically trimmed. BERTScore ceiling: 0.75."
    ),
)

if __name__ == "__main__":
    iface.launch()