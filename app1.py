
import os
import sys
import json
import random
import torch
import torch.nn.functional as F
from types import MethodType
from datetime import datetime
import gradio as gr

# ----------------------------
# CONFIG
# ----------------------------
CONFIG = {
    "model_type": "d3pm_cross_attention",
    "model": {
        "vocab_size": 16000,
        "max_seq_len": 80,
        "diffusion_steps": 128,#40,
        "d_model": 1024,#512,
        "n_layers": 8,
        "n_heads": 8,
        "d_ff": 4096,#2048,
        "dropout": 0.2
    },
    "diffusion": {"mask_token_id": 0},
    "training": {"precision": "float32", "device": "mps"}
}

device = torch.device(CONFIG["training"]["device"])
dtype = torch.float32


# ----------------------------
# Set Seed
# ----------------------------
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"🔒 Seed set to {seed}")


set_seed(42)

# ----------------------------
# Project modules
# ----------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.tokenizer import SanskritTokenizer
from model.sanskrit_model import SanskritModel
from diffusion.reverse_process1 import ReverseDiffusion
from diffusion.scheduler import OptimizedCosineScheduler


# ----------------------------
# Clean Sanskrit Text
# ----------------------------
# def clean_text(text):
#     text = text.replace("<pad>", "").replace("<s>", "").replace("</s>", "")
#     text = text.replace("[MASK]", "")
#     while "।।" in text: text = text.replace("।।", "।")
#     text = text.replace(" ।", "।")
#     return " ".join(text.split()).strip()
def clean_text(text):

    text = text.replace("<pad>", "")
    text = text.replace("<s>", "")
    text = text.replace("</s>", "")
    text = text.replace("[MASK]", "")

    text = text.replace("।।", "।")

    text = " ".join(text.split())

    return text.strip()


# ----------------------------
# Patch P_SAMPLE_STEP
# ----------------------------
def patch_p_sample_step(reverse_diffusion):
    old_p_sample_step = reverse_diffusion.p_sample_step

    def p_sample_step_override(self, *args, **kwargs):
        result = old_p_sample_step(*args, **kwargs)
        # Ensure output is never empty
        if isinstance(result, list) and len(result) > 0:
            for i, (tokens, score) in enumerate(result):
                if all(t == CONFIG["diffusion"]["mask_token_id"] for t in tokens.view(-1).tolist()):
                    # replace all-mask sequence with random token (safe fallback)
                    fallback_tokens = torch.randint(1, CONFIG["model"]["vocab_size"], tokens.shape,
                                                    device=tokens.device)
                    result[i] = (fallback_tokens, score)
        return result

    # reverse_diffusion.p_sample_step = MethodType(p_sample_step_override, reverse_diffusion)


# ----------------------------
# Load model
# ----------------------------
def load_model(model_type="d3pm_cross_attention"):
    CONFIG["model_type"] = model_type
    tokenizer = SanskritTokenizer(CONFIG["model"]["vocab_size"])
    model = SanskritModel(CONFIG)

    # model_path = "/Users/bhsingh/Documents/Generation/production_model3/best_d3pm_cross_attention.pt"
    # model_path = "/Users/bhsingh/Documents/Generation/NBaseline/production_model4/best_d3pm_cross_attention.pt" #bekar h
    # model_path = f"/Users/bhsingh/Documents/Generation/NBaseline/production_model4/best_d3pm_cross_attention_neg.pt" #bekar h
    # model_path = f"/Users/bhsingh/Documents/Generation/production_model3/best_d3pm_encoder_decoder_neg.pt"
    # model_path = f"/Users/bhsingh/Documents/Generation/results1/best_d3pm_cross_attention.pt"
    # model_path = f"/Users/bhsingh/Documents/Final_Paraphrase/Exclude_Negative/results5/d3pm_cross_attention_neg_False/best_model.pt"
    model_path = f"/Users/bhsingh/Documents/Final_Paraphrase/Exclude_Negative/results7/d3pm_cross_attention_neg_True/best_model.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.to(device, dtype=dtype)
    model.eval()
    model.mask_token_id = CONFIG["diffusion"]["mask_token_id"]

    scheduler = OptimizedCosineScheduler(CONFIG)
    reverse_diffusion = ReverseDiffusion(scheduler)
    patch_p_sample_step(reverse_diffusion)
    return model, tokenizer, reverse_diffusion


model, tokenizer, reverse_diffusion = load_model("d3pm_cross_attention")

# ----------------------------
# JSON file
# ----------------------------
RESULTS_DIR = "generated_results"
os.makedirs(RESULTS_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_FILE = os.path.join(RESULTS_DIR, f"{CONFIG['model_type']}_results_{timestamp}.json")
if not os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=4)


# ----------------------------
# Generate text function
# ----------------------------
# @torch.no_grad()
# def generate_text(input_text: str,
#                   model, tokenizer, reverse_diffusion,
#                   diversity_level="medium",
#                   repetition_penalty=1.15,
#                   diversity_penalty=0.0,
#                   length_penalty=1.0):
#     input_ids = tokenizer.encode(input_text)
#     input_tensor = torch.tensor([input_ids], device=device)
#     #
#     # if diversity_level == "low":
#     #     beam_width, temperature, sampling_mode = 2, 0.6, False
#     # elif diversity_level == "high":
#     #     beam_width, temperature, sampling_mode = 4, 0.95, True
#     # else:
#     #     beam_width, temperature, sampling_mode = 3, 0.75, False
#     if diversity_level == "low":
#         beam_width, temperature = 2, 0.65
#     elif diversity_level == "high":
#         beam_width, temperature = 5, 0.9
#     else:
#         beam_width, temperature = 3, 0.75
#
#     reverse_diffusion.temperature = temperature
#     reverse_diffusion.repetition_penalty = max(1.0, repetition_penalty)
#     reverse_diffusion.diversity_penalty = max(0.0, diversity_penalty)
#     # reverse_diffusion.sampling_mode = sampling_mode
#     reverse_diffusion.length_penalty = max(0.0, length_penalty)
#
#     # # Generate using beam
#     # generated_ids = reverse_diffusion.generate_beam(
#     #     model, condition=input_tensor, beam_width=beam_width, num_steps=CONFIG["model"]["diffusion_steps"]
#     # )
#     generated_ids = reverse_diffusion.generate_beam(
#         model=model,
#         condition=input_tensor,
#         beam_width=beam_width,
#         num_steps=CONFIG["model"]["diffusion_steps"]
#     )
#
#     mask_id = CONFIG["diffusion"]["mask_token_id"]
#     tokens = [tid for tid in generated_ids[0].tolist() if tid != mask_id]
#
#     # Fallback if all mask
#     if len(tokens) == 0:
#         tokens = [random.randint(1, CONFIG["model"]["vocab_size"] - 1) for _ in range(5)]
#
#     output_text = clean_text(tokenizer.decode(tokens))
#     if len(output_text) == 0:
#         output_text = "<MODEL DID NOT GENERATE ANY TOKENS>"
#
#     print("✅ INPUT :", input_text)
#     print("✅ OUTPUT:", output_text)
#
#     return output_text

# ----------------------------
# Robust Generate Text Function
# ----------------------------
@torch.no_grad()
def generate_text(input_text: str,
                  model, tokenizer, reverse_diffusion,
                  diversity_level="medium",
                  repetition_penalty=1.15,
                  diversity_penalty=0.0,
                  length_penalty=1.0,
                  fallback_length=5):
    """
    Generates a Sanskrit paraphrase using reverse diffusion.
    Ensures output is never empty by fallback mechanism.
    """

    # Encode input
    input_ids = tokenizer.encode(input_text)
    input_tensor = torch.tensor([input_ids], device=device)

    # Map diversity level to beam width and temperature
    if diversity_level == "low":
        beam_width, temperature = 2, 0.65
    elif diversity_level == "high":
        beam_width, temperature = 5, 0.9
    else:
        beam_width, temperature = 3, 0.75

    # Set diffusion parameters
    reverse_diffusion.temperature = temperature
    reverse_diffusion.repetition_penalty = max(1.0, repetition_penalty)
    reverse_diffusion.diversity_penalty = max(0.0, diversity_penalty)
    reverse_diffusion.length_penalty = max(0.0, length_penalty)

    # Generate tokens using beam search
    generated_ids = reverse_diffusion.generate_beam(
        model=model,
        condition=input_tensor,
        beam_width=beam_width,
        num_steps=CONFIG["model"]["diffusion_steps"]
    )

    mask_id = CONFIG["diffusion"]["mask_token_id"]
    tokens = [tid for tid in generated_ids[0].tolist() if tid != mask_id]

    # Fallback if all tokens are mask
    if len(tokens) == 0:
        print("⚠️ WARNING: Model generated all MASK tokens. Using random fallback.")
        tokens = [random.randint(1, CONFIG["model"]["vocab_size"] - 1) for _ in range(fallback_length)]

    # Decode tokens to text
    output_text = clean_text(tokenizer.decode(tokens))

    # Extra safeguard for empty string
    if len(output_text) == 0:
        print("⚠️ WARNING: Decoded output is empty. Using random fallback tokens.")
        fallback_tokens = [random.randint(1, CONFIG["model"]["vocab_size"] - 1) for _ in range(fallback_length)]
        output_text = clean_text(tokenizer.decode(fallback_tokens))

    print("✅ INPUT :", input_text)
    print("✅ OUTPUT:", output_text)

    return output_text


# ----------------------------
# Generate text + save JSON
# ----------------------------
@torch.no_grad()
def generate_text_and_save_json(input_text: str,
                                model, tokenizer, reverse_diffusion,
                                diversity_level="medium",
                                repetition_penalty=1.15,
                                diversity_penalty=0.0,
                                length_penalty=1.0):
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

    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

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

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return output_text


# ----------------------------
# Gradio function
# ----------------------------
def gradio_infer(input_text, diversity, repetition, diversity_penalty, length_penalty):
    return generate_text_and_save_json(
        input_text,
        model=model,
        tokenizer=tokenizer,
        reverse_diffusion=reverse_diffusion,
        diversity_level=diversity,
        repetition_penalty=float(repetition),
        diversity_penalty=float(diversity_penalty),
        length_penalty=float(length_penalty)
    )


# ----------------------------
# Gradio Interface
# ----------------------------
iface = gr.Interface(
    fn=gradio_infer,
    inputs=[
        gr.Textbox(label="Input Sanskrit Text"),
        gr.Radio(["low", "medium", "high"], label="Diversity", value="medium"),
        gr.Number(value=1.15, label="Repetition Penalty"),
        gr.Number(value=0.0, label="Diversity Penalty"),
        gr.Number(value=1.0, label="Length Penalty")
    ],
    outputs=[gr.Textbox(label="Output Text")],
    title="Sanskrit Paraphrase Generator",
    description="Generates paraphrases and saves results to JSON automatically."
)

# ----------------------------
# Launch
# ----------------------------
if __name__ == "__main__":
    iface.launch()