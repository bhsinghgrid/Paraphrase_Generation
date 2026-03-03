# # """
# # Final Model Test Script
# # Evaluates trained Sanskrit model on 5K test dataset
# # """
# #
# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, Subset
# from tqdm import tqdm
# import os
# import json
# import random
# import numpy as np
# import sys
#
# # Optional BERTScore
# try:
#     from bert_score import score
#     BERTSCORE_AVAILABLE = True
# except ImportError:
#     BERTSCORE_AVAILABLE = False
#
# # Optional BLEU
# try:
#     import sacrebleu
#     BLEU_AVAILABLE = True
# except ImportError:
#     BLEU_AVAILABLE = False
#
# # Your project modules
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from data.dataset import OptimizedSanskritDataset
# from model.tokenizer import SanskritTokenizer
# from model.sanskrit_model import SanskritModel
# #
# #
# # # =========================================================
# # # CONFIG (same as training)
# # # =========================================================
# #
# CONFIG = {
#     "model_type": "d3pm_cross_attention",  # change if needed
#
#     # "model": {
#     #     "vocab_size": 16000,
#     #     "max_seq_len": 80,
#     #     "diffusion_steps": 8,
#     #     "d_model": 512,
#     #     "n_layers": 6,
#     #     "n_heads": 8,
#     #     "d_ff": 1536,
#     #     "dropout": 0.2
#     # },
#     #
#     # "diffusion": {
#     #     "mask_token_id": 0
#     # },
#     #
#     # "training": {
#     #     "batch_size": 8,
#     #     "device": "mps",
#     #     "precision": "float32"
#     # }
# "model": {
#         "vocab_size": 16000,
#         "max_seq_len": 80,
#         "diffusion_steps": 12,
#         "d_model": 512,
#         "n_layers": 6,
#         "n_heads": 8,
#         "d_ff": 1536,
#         "dropout": 0.2
#     },
#
#     "diffusion": {
#         "mask_token_id": 0
#     },
#
#     "training": {
#         "batch_size": 8,
#         "epochs": 7,
#         "lr": 3e-4,
#         "label_smoothing": 0.05,
#         "precision": "float32",
#         "device": "mps",
#         "dataset_size": 5000,
#         "early_stopping_patience": 3
#     }
# }
#
#
# # =========================================================
# # Seed & Deterministic
# # =========================================================
#
# def set_seed(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     torch.use_deterministic_algorithms(True)
#     os.environ["PYTHONHASHSEED"] = str(seed)
#     print(f"🔒 Seed set to {seed} (Deterministic mode enabled)")
# import os
# import json
# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, Subset
# from tqdm import tqdm
# from bert_score import score
# import sacrebleu
#
# def run_test(test_size=5000):
#     print("🧪 Starting Test Evaluation...\n")
#
#     device = torch.device(CONFIG["training"]["device"])
#     dtype = torch.float16 if CONFIG["training"]["precision"] == "float16" else torch.float32
#
#     # -----------------------------------------------------
#     # Load Tokenizer
#     # -----------------------------------------------------
#     tokenizer = SanskritTokenizer(CONFIG["model"]["vocab_size"])
#
#     # -----------------------------------------------------
#     # Load Model
#     # -----------------------------------------------------
#     model = SanskritModel(CONFIG)
#     model.to(device, dtype=dtype)
#
#     # model_path = f"production_model4/best2_{CONFIG['model_type']}.pt"
#     # model_path = f"NBaseline/production_model4/best_{CONFIG['model_type']}.pt"
#     model_path = f"/Users/bhsingh/Documents/Generation/NBaseline/production_model4/best_d3pm_cross_attention.pt"
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Model not found: {model_path}")
#
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.eval()
#     print(f"✅ Loaded model: {CONFIG['model_type']}")
#
#     # -----------------------------------------------------
#     # Load Test Dataset
#     # -----------------------------------------------------
#     test_dataset = OptimizedSanskritDataset(
#         split="test",
#         tokenizer=tokenizer,
#         max_len=CONFIG["model"]["max_seq_len"]
#     )
#
#     size = min(test_size, len(test_dataset))
#     indices = list(range(size))
#
#     def collate(batch):
#         return {
#             "input_ids": torch.stack([b["input_ids"].long() for b in batch]),
#             "target_ids": torch.stack([b["target_ids"].long() for b in batch]),
#             "target_text": [b["target_text"] for b in batch]
#         }
#
#     test_loader = DataLoader(
#         Subset(test_dataset, indices),
#         batch_size=CONFIG["training"]["batch_size"],
#         shuffle=False,
#         collate_fn=collate
#     )
#
#     print(f"📦 Test Samples: {size}\n")
#
#     # -----------------------------------------------------
#     # Metrics
#     # -----------------------------------------------------
#     total_loss = 0
#     true_positive = 0
#     total_tokens = 0
#     exact_match = 0
#
#     predictions = []
#     references = []
#
#     mask_token_id = CONFIG["diffusion"]["mask_token_id"]
#
#     with torch.no_grad():
#         for batch in tqdm(test_loader, desc="Testing"):
#             input_ids = batch["input_ids"].to(device)
#             target_ids = batch["target_ids"].to(device)
#
#             # Diffusion vs Baseline
#             if "d3pm" in CONFIG["model_type"]:
#                 t = torch.randint(
#                     0,
#                     CONFIG["model"]["diffusion_steps"],
#                     (input_ids.size(0),),
#                     device=device
#                 )
#                 logits, _ = model(input_ids, target_ids, t)
#             else:
#                 logits = model(input_ids, target_ids)
#
#             # Loss
#             loss = F.cross_entropy(
#                 logits.view(-1, logits.size(-1)),
#                 target_ids.view(-1),
#                 ignore_index=mask_token_id
#             )
#             total_loss += loss.item()
#
#             # Predictions
#             preds = torch.argmax(logits, dim=-1)
#             mask = target_ids != mask_token_id
#
#             tp = ((preds == target_ids) & mask).sum().item()
#             true_positive += tp
#             total_tokens += mask.sum().item()
#
#             # Decode for BERTScore, BLEU, Exact Match
#             for p, t_ref in zip(preds, target_ids):
#                 decoded_pred = tokenizer.decode([id for id in p.tolist() if id != mask_token_id]).strip()
#                 decoded_ref  = tokenizer.decode([id for id in t_ref.tolist() if id != mask_token_id]).strip()
#                 predictions.append(decoded_pred)
#                 references.append(decoded_ref)
#
#                 if decoded_pred == decoded_ref:
#                     exact_match += 1
#
#     # -----------------------------------------------------
#     # Final Metrics
#     # -----------------------------------------------------
#     avg_loss = total_loss / len(test_loader)
#     token_accuracy = true_positive / (total_tokens + 1e-8)
#     sentence_accuracy = exact_match / len(predictions) if predictions else 0.0
#     #
#     # # BERTScore
#     # if BERTSCORE_AVAILABLE and predictions:
#     #     P, R, F1 = score(predictions, references, lang="hi", verbose=False)
#     #     bert_f1 = F1.mean().item()
#     # else:
#     #     bert_f1 = 0.0
#     #
#     # # BLEU
#     # if BLEU_AVAILABLE and predictions:
#     #     bleu = sacrebleu.corpus_bleu(predictions, [references])
#     #     bleu_score = bleu.score
#     # else:
#     #     bleu_score = 0.0
#     bert_f1 = 0.0
#     if BERTSCORE_AVAILABLE and len(predictions) > 0:
#         try:
#             P, R, F1 = score(
#                 predictions,
#                 references,
#                 lang="hi",  # Sanskrit close to Hindi model
#                 rescale_with_baseline=True
#             )
#             bert_f1 = F1.mean().item()
#         except Exception as e:
#             print("⚠️ BERTScore failed:", e)
#             bert_f1 = 0.0
#
#     # -----------------------------------------------------
#     # BLEU (SacreBLEU Safe)
#     # -----------------------------------------------------
#     bleu_score = 0.0
#     if BLEU_AVAILABLE and len(predictions) > 0:
#         try:
#             bleu = sacrebleu.corpus_bleu(predictions, [references])
#             bleu_score = bleu.score
#         except Exception as e:
#             print("⚠️ BLEU failed:", e)
#             bleu_score = 0.0
#
#     results = {
#         "model_type": CONFIG["model_type"],
#         "test_size": size,
#         "test_loss": round(avg_loss, 4),
#         "token_accuracy": round(token_accuracy, 4),
#         "sentence_exact_match": round(sentence_accuracy, 4),
#         "bert_f1": round(bert_f1, 4),
#         "bleu": round(bleu_score, 4)
#     }
#
#     # Save JSON
#     os.makedirs("results", exist_ok=True)
#     with open("results/test_metrics_NBaseC.json", "w") as f:
#         json.dump(results, f, indent=4)
#
#     # Print Results
#     print("\n📊 FINAL TEST RESULTS")
#     for k, v in results.items():
#         print(f"{k}: {v}")
#
#     print("\n✅ Results saved to results/test_metrics.json")
#
# # =========================================================
# # MAIN
# # =========================================================
# if __name__ == "__main__":
#     set_seed(42)
#     run_test(test_size=5000)

"""
Final Generative Test Script
Evaluates trained Sanskrit model using reverse diffusion sampling
"""

"""
Final Generative Test Script
Evaluates trained Sanskrit model using ReverseDiffusion Beam Search
"""

import os
import sys
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Optional Metrics
try:
    from bert_score import score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False

try:
    import sacrebleu
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False


# =========================================================
# Add Project Path
# =========================================================
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import OptimizedSanskritDataset
from model.tokenizer import SanskritTokenizer
from model.sanskrit_model import SanskritModel
from diffusion.reverse_process import ReverseDiffusion
from diffusion.scheduler import OptimizedCosineScheduler


# =========================================================
# CONFIG
# =========================================================
CONFIG = {
    "model_type": "d3pm_cross_attention",

    "model": {
        "vocab_size": 16000,
        "max_seq_len": 80,
        "diffusion_steps": 12,
        "d_model": 512,
        "n_layers": 6,
        "n_heads": 8,
        "d_ff": 1536,
        "dropout": 0.2
    },

    "diffusion": {
        "mask_token_id": 0
    },

    "training": {
        "batch_size": 8,
        "precision": "float32",
        "device": "mps"
    }
}


# =========================================================
# Deterministic Setup
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"🔒 Seed set to {seed}")


# =========================================================
# CLEANING (Improved for Sanskrit BLEU)
# =========================================================
def clean_text(text):
    # Remove special tokens
    text = text.replace("<pad>", "")
    text = text.replace("<s>", "")
    text = text.replace("</s>", "")

    # Remove mask tokens if present
    text = text.replace("[MASK]", "")

    # Fix repeated danda
    while "।।" in text:
        text = text.replace("।।", "।")

    # Fix spacing before punctuation
    text = text.replace(" ।", "।")

    # Collapse extra spaces
    text = " ".join(text.split())

    return text.strip()


# =========================================================
# TEST FUNCTION
# =========================================================
def run_test(test_size=5000):

    print("🧪 Starting TRUE Generative Evaluation...\n")

    device = torch.device(CONFIG["training"]["device"])

    # -----------------------------------------------------
    # Tokenizer
    # -----------------------------------------------------
    tokenizer = SanskritTokenizer(CONFIG["model"]["vocab_size"])

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------
    model = SanskritModel(CONFIG)
    model.to(device)

    model_path = "/Users/bhsingh/Documents/Generation/NBaseline/production_model4/best_d3pm_cross_attention.pt"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    model.mask_token_id = CONFIG["diffusion"]["mask_token_id"]

    print(f"✅ Loaded model: {CONFIG['model_type']}")

    # -----------------------------------------------------
    # Reverse Diffusion
    # -----------------------------------------------------
    # -----------------------------------------------------
    # Reverse Diffusion
    # -----------------------------------------------------
    scheduler = OptimizedCosineScheduler(CONFIG, device=device)
    # reverse_diffusion = ReverseDiffusion(scheduler)
    reverse_diffusion = ReverseDiffusion(scheduler)

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------
    test_dataset = OptimizedSanskritDataset(
        split="test",
        tokenizer=tokenizer,
        max_len=CONFIG["model"]["max_seq_len"]
    )

    size = min(test_size, len(test_dataset))
    indices = list(range(size))

    def collate(batch):
        return {
            "input_ids": torch.stack([b["input_ids"].long() for b in batch]),
            "target_text": [b["target_text"] for b in batch]
        }

    test_loader = DataLoader(
        Subset(test_dataset, indices),
        batch_size=CONFIG["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate
    )

    print(f"📦 Test Samples: {size}\n")

    predictions = []
    references = []
    exact_match = 0

    # =====================================================
    # GENERATION LOOP
    # =====================================================
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating"):

            input_ids = batch["input_ids"].to(device)

            # 🔥 Improved Beam Generation
            generated_ids = reverse_diffusion.generate_beam(
                model=model,
                condition=input_ids,
                beam_width=5,
                num_steps=CONFIG["model"]["diffusion_steps"],
                temperature=0.8,        # sharper distribution
                repetition_penalty=1.3, # stronger anti-repeat
                diversity_penalty=0.2
            )

            if generated_ids.dim() == 1:
                generated_ids = generated_ids.unsqueeze(0)

            # -------------------------------------------------
            # Decode
            # -------------------------------------------------
            for gen_ids, ref_text in zip(generated_ids, batch["target_text"]):

                decoded_pred = tokenizer.decode(gen_ids.tolist())
                decoded_pred = clean_text(decoded_pred)

                decoded_ref = clean_text(ref_text)

                # Skip extremely short collapsed outputs
                if len(decoded_pred.split()) < 3:
                    decoded_pred = ""

                predictions.append(decoded_pred)
                references.append(decoded_ref)

                if decoded_pred == decoded_ref:
                    exact_match += 1

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------
    sentence_accuracy = exact_match / len(predictions)

    bert_f1 = 0.0
    if BERTSCORE_AVAILABLE and predictions:
        P, R, F1 = score(
            predictions,
            references,
            lang="hi",
            rescale_with_baseline=True
        )
        bert_f1 = F1.mean().item()

    bleu_score = 0.0
    if BLEU_AVAILABLE and predictions:
        bleu = sacrebleu.corpus_bleu(
            predictions,
            [references],
            tokenize="intl"  # better unicode tokenization
        )
        bleu_score = bleu.score

    results = {
        "model_type": CONFIG["model_type"],
        "test_size": size,
        "sentence_exact_match": round(sentence_accuracy, 4),
        "bert_f1": round(bert_f1, 4),
        "bleu": round(bleu_score, 4)
    }

    os.makedirs("results", exist_ok=True)
    with open("results/test_metrics_generative.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\n📊 GENERATIVE TEST RESULTS")
    for k, v in results.items():
        print(f"{k}: {v}")

    print("\n✅ Results saved to results/test_metrics_generative.json")


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    set_seed(42)
    run_test(test_size=5000)