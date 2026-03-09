"""
app1.py — Final Correct Version (Stable)
"""

import os
import sys
import json
import random
import torch
from datetime import datetime
import gradio as gr

# ----------------------------
# CONFIG
# ----------------------------

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

device = torch.device(CONFIG["training"]["device"])
dtype = torch.float32


# ----------------------------
# SEED
# ----------------------------

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seed(42)


# ----------------------------
# IMPORT MODEL + TOKENIZERS
# ----------------------------

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.tokenizer import SanskritSourceTokenizer, SanskritTargetTokenizer
from model.sanskrit_model import SanskritModel


# ----------------------------
# TEXT CLEAN
# ----------------------------
#
# def clean_text(text: str):
#
#     text = text.replace("<pad>", "")
#     text = text.replace("<s>", "")
#     text = text.replace("</s>", "")
#     text = text.replace("[MASK]", "")
#     text = text.replace("।।", "।")
#
#     return " ".join(text.split()).strip()
import re

import re

def clean_text(text: str):

    # remove special tokens
    text = text.replace("<pad>", "")
    text = text.replace("<s>", "")
    text = text.replace("</s>", "")
    text = text.replace("[MASK]", "")

    text = text.replace("।।", "।")

    # normalize spaces
    text = " ".join(text.split())

    # ---------------------------------
    # RULE 1: join virama consonants
    # क् + ष → क्ष
    # ---------------------------------
    text = re.sub(r'([क-ह])्\s+([क-ह])', r'\1्\2', text)

    # ---------------------------------
    # RULE 2: join matras
    # क + ा → का
    # ---------------------------------
    text = re.sub(r'([क-ह])\s+([ािीुूृॄेैोौंः])', r'\1\2', text)

    # ---------------------------------
    # RULE 3: join consonant clusters
    # क + त → कत
    # ---------------------------------
    text = re.sub(r'([क-ह])\s+([क-ह])', r'\1\2', text)

    # ---------------------------------
    # RULE 4: join visarga
    # ---------------------------------
    text = re.sub(r'\s+ः', 'ः', text)

    # ---------------------------------
    # RULE 5: remove repetition
    # ---------------------------------
    words = text.split()
    cleaned = []

    for w in words:
        if not cleaned or cleaned[-1] != w:
            cleaned.append(w)

    return " ".join(cleaned).strip()

# ----------------------------
# REPETITION TRIM
# ----------------------------

# def trim_token_ids(token_ids, run_thresh=4):
#
#     if len(token_ids) <= 4:
#         return token_ids
#
#     count = 1
#
#     for i in range(1, len(token_ids)):
#
#         if token_ids[i] == token_ids[i - 1]:
#             count += 1
#
#             if count >= run_thresh:
#
#                 cut = max(4, i - run_thresh + 2)
#
#                 return token_ids[:cut]
#
#         else:
#             count = 1
#
#     return token_ids
def trim_token_ids(token_ids, run_thresh=4):

    cleaned = []
    repeat = 1

    for i in range(len(token_ids)):

        if i > 0 and token_ids[i] == token_ids[i-1]:
            repeat += 1
        else:
            repeat = 1

        if repeat >= run_thresh:
            break

        cleaned.append(token_ids[i])

    return cleaned

# ----------------------------
# LOAD MODEL
# ----------------------------

def load_model():

    src_tokenizer = SanskritSourceTokenizer(
        vocab_size=CONFIG["model"]["src_vocab_size"],
        max_len=CONFIG["model"]["max_seq_len"],
    )

    tgt_tokenizer = SanskritTargetTokenizer(
        vocab_size=CONFIG["model"]["tgt_vocab_size"],
        max_len=CONFIG["model"]["max_seq_len"],
    )

    model = SanskritModel(CONFIG)

    model_path = (
        "/Users/bhsingh/Documents/Final_Paraphrase/Exclude_Negative/"
        "results7/d3pm_cross_attention_neg_True/best_model.pt"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model.load_state_dict(
        torch.load(model_path, map_location=device),
        strict=False,
    )

    model.to(device, dtype=dtype)
    model.eval()

    print("✅ Model loaded")

    return model, src_tokenizer, tgt_tokenizer


model, src_tokenizer, tgt_tokenizer = load_model()


# ----------------------------
# RESULTS JSON
# ----------------------------

RESULTS_DIR = "generated_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    f"{CONFIG['model_type']}_results_{timestamp}.json",
)

with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    json.dump([], f, ensure_ascii=False, indent=4)


# ----------------------------
# GENERATION
# ----------------------------

@torch.no_grad()
def generate_text(
    input_text,
    diversity_level="medium",
    repetition_penalty=1.15,
    diversity_penalty=0.0,
    top_k=50,
):

    input_ids = src_tokenizer.encode(input_text)

    input_tensor = torch.tensor(
        [input_ids],
        device=device,
        dtype=torch.long,
    )

    # temperature control

    if diversity_level == "low":
        temperature = 0.65
    elif diversity_level == "high":
        temperature = 0.90
    else:
        temperature = 0.75

    # FIX: use UI top_k value
    top_k_eff = int(top_k)

    gen_ids = model.generate(
        input_tensor,
        num_steps=CONFIG["model"]["diffusion_steps"],
        temperature=temperature,
        top_k=top_k_eff,
        repetition_penalty=max(1.0, repetition_penalty),
        diversity_penalty=max(0.0, diversity_penalty),
    )

    raw_ids = [x for x in gen_ids[0].tolist() if x > 4]

    trimmed_ids = trim_token_ids(raw_ids)

    raw_text = clean_text(tgt_tokenizer.decode(raw_ids))

    output_text = clean_text(tgt_tokenizer.decode(trimmed_ids))

    if not output_text:
        output_text = raw_text

    print("INPUT :", input_text)
    print("RAW   :", raw_text[:100])
    print("FINAL :", output_text)

    return output_text


# ----------------------------
# SAVE JSON
# ----------------------------

def generate_text_and_save_json(
    input_text,
    diversity,
    repetition,
    diversity_penalty,
    top_k,
):

    output_text = generate_text(
        input_text,
        diversity_level=diversity,
        repetition_penalty=float(repetition),
        diversity_penalty=float(diversity_penalty),
        top_k=int(top_k),
    )

    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    data.append({
        "input_text": input_text,
        "output_text": output_text,
        "diversity_level": diversity,
        "repetition_penalty": repetition,
        "diversity_penalty": diversity_penalty,
        "top_k": top_k,
        "timestamp": datetime.now().isoformat(),
    })

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return output_text


# ----------------------------
# GRADIO
# ----------------------------

def gradio_infer(input_text, diversity, repetition, diversity_penalty, top_k):

    return generate_text_and_save_json(
        input_text,
        diversity,
        repetition,
        diversity_penalty,
        top_k,
    )


iface = gr.Interface(
    fn=gradio_infer,
    inputs=[
        gr.Textbox(label="Input Sanskrit Text (Roman IAST)"),
        gr.Radio(["low", "medium", "high"], label="Diversity Level", value="medium"),
        gr.Number(value=1.15, label="Repetition Penalty"),
        gr.Number(value=0.0, label="Diversity Penalty"),
        gr.Number(value=50, label="Top-K"),
    ],
    outputs=gr.Textbox(label="Generated Paraphrase (Devanagari)"),
    title="Sanskrit Paraphrase Generator (D3PM)",
    description="IAST → Devanagari paraphrase using diffusion model",
)


if __name__ == "__main__":
    iface.launch()