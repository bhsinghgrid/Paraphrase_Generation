import torch
import torch.nn.functional as F
import os
import sys

# ==========================================
# IMPORT YOUR MODEL + TOKENIZER
# (adjust paths if needed)
# ==========================================

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.new_d3pm_model import SanskritModel   # change if your model class name differs
from model.tokenizer import SanskritTokenizer


# ==========================================
# Repetition Penalty
# ==========================================
def apply_repetition_penalty(logits, generated_ids, penalty=1.15):
    unique_tokens = set(generated_ids.tolist())
    for token_id in unique_tokens:
        logits[:, token_id] /= penalty
    return logits


# ==========================================
# Block Repeated Bigrams
# ==========================================
def block_repeated_bigrams(logits, generated_ids):
    if len(generated_ids) < 2:
        return logits

    bigrams = set()
    for i in range(len(generated_ids) - 1):
        bigrams.add((generated_ids[i], generated_ids[i + 1]))

    last_token = generated_ids[-1].item()

    for (a, b) in bigrams:
        if a == last_token:
            logits[:, b] = -float("inf")

    return logits


# ==========================================
# Beam Reverse Diffusion
# ==========================================
@torch.no_grad()
def beam_reverse_diffusion(
    model,
    input_ids,
    device="cuda",
    num_steps=6,          # 🔥 reduced steps
    beam_size=4,
    temperature=0.7,
    repetition_penalty=1.15,
):

    model.eval()
    input_ids = input_ids.to(device)

    beams = [(input_ids, 0.0)]

    for step in reversed(range(num_steps)):
        print(f"Beam Reverse step {step}/{num_steps}")

        new_beams = []

        for seq, score in beams:

            timestep_tensor = torch.tensor([step]).to(device)

            logits = model(seq, timestep=timestep_tensor)
            logits = logits[:, -1, :]

            # Temperature scaling
            logits = logits / temperature

            # Repetition penalty
            logits = apply_repetition_penalty(
                logits, seq[0], repetition_penalty
            )

            # Bigram blocking
            logits = block_repeated_bigrams(logits, seq[0])

            log_probs = F.log_softmax(logits, dim=-1)

            topk_log_probs, topk_ids = torch.topk(
                log_probs, beam_size, dim=-1
            )

            for i in range(beam_size):
                next_token = topk_ids[0, i].unsqueeze(0).unsqueeze(0)
                new_seq = torch.cat([seq, next_token], dim=1)
                new_score = score + topk_log_probs[0, i].item()
                new_beams.append((new_seq, new_score))

        new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]

    best_seq = beams[0][0]
    return best_seq


# ==========================================
# Text Generation
# ==========================================
def generate_text(model, tokenizer, text, device="cuda"):

    encoded = tokenizer.encode(text)
    input_ids = torch.tensor(encoded).unsqueeze(0)

    output_ids = beam_reverse_diffusion(
        model=model,
        input_ids=input_ids,
        device=device,
        num_steps=6,          # 🔥 important
        beam_size=4,
        temperature=0.7,
        repetition_penalty=1.15,
    )

    decoded = tokenizer.decode(output_ids[0].tolist())

    # Clean BPE artifacts
    decoded = decoded.replace(" ##", "")
    decoded = decoded.replace("▁", " ")
    decoded = decoded.strip()

    return decoded


# ==========================================
# Load Model
# ==========================================
def load_model(model_path, device="cuda"):

    print("📖 Loading tokenizer...")
    tokenizer = SanskritTokenizer()

    print("📦 Loading model...")
    model = SanskritModel(
        vocab_size=tokenizer.vocab_size
    )

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)

    model.to(device)
    model.eval()

    print("✅ Model loaded successfully")
    return model, tokenizer


# ==========================================
# MAIN LOOP
# ==========================================
if __name__ == "__main__":

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = "production_model3/model.pt"   # adjust path

    model, tokenizer = load_model(MODEL_PATH, DEVICE)

    print("\n📝 Enter Sanskrit input (type 'quit' to exit):")

    while True:
        user_input = input("> ")

        if user_input.lower() == "quit":
            break

        try:
            output = generate_text(
                model,
                tokenizer,
                user_input,
                device=DEVICE
            )
            print("✅ Output:", output)

        except Exception as e:
            print("❌ Inference Error:", str(e))