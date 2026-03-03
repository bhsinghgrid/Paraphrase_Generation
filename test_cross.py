# """
# Final Model Test Script
# Evaluates trained Sanskrit model on 5K test dataset
# """
#
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import os
import json
import random
import numpy as np

# Optional BERTScore
try:
    from bert_score import score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False

# Optional BLEU
try:
    import sacrebleu
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False

# Your project modules
from data.dataset import OptimizedSanskritDataset
from model.tokenizer import SanskritTokenizer
from model.new_d3pm_model import SanskritModel
#
#
# # =========================================================
# # CONFIG (same as training)
# # =========================================================
#
CONFIG = {
    "model_type": "d3pm_cross_attention",  # change if needed

    "model": {
        "vocab_size": 16000,
        "max_seq_len": 80,
        "diffusion_steps": 8,
        "d_model": 512,
        "n_layers": 6,
        "n_heads": 8,
        "d_ff": 1536,
        "dropout": 0.1
    },

    "diffusion": {
        "mask_token_id": 0
    },

    "training": {
        "batch_size": 8,
        "device": "mps",
        "precision": "float32"
    }
}


# =========================================================
# Seed & Deterministic
# =========================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"🔒 Seed set to {seed} (Deterministic mode enabled)")
#
#
# # =========================================================
# # TEST FUNCTION
# # =========================================================
#
# def run_test(test_size=5000):
#
#     print("🧪 Starting Test Evaluation...\n")
#
#     device = torch.device(CONFIG["training"]["device"])
#     dtype = torch.float16 if CONFIG["training"]["precision"] == "float16" else torch.float32
#
#     # -----------------------------------------------------
#     # Load Tokenizer
#     # -----------------------------------------------------
#
#     tokenizer = SanskritTokenizer(CONFIG["model"]["vocab_size"])
#
#     # -----------------------------------------------------
#     # Load Model
#     # -----------------------------------------------------
#
#     model = SanskritModel(CONFIG)
#     model.to(device, dtype=dtype)
#
#     model_path = f"production_model3/best_{CONFIG['model_type']}.pt"
#
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Model not found: {model_path}")
#
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.eval()
#
#     print(f"✅ Loaded model: {CONFIG['model_type']}")
#
#     # -----------------------------------------------------
#     # Load Test Dataset
#     # -----------------------------------------------------
#
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
#
#     total_loss = 0
#
#     true_positive = 0
#     false_positive = 0
#     false_negative = 0
#
#     exact_match = 0
#     total_sentences = 0
#
#     predictions = []
#     references = []
#
#     with torch.no_grad():
#         for batch in tqdm(test_loader, desc="Testing"):
#
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
#                 ignore_index=CONFIG["diffusion"]["mask_token_id"]
#             )
#             total_loss += loss.item()
#
#             # Predictions
#             preds = torch.argmax(logits, dim=-1)
#             mask = target_ids != CONFIG["diffusion"]["mask_token_id"]
#
#             tp = ((preds == target_ids) & mask).sum().item()
#             fp = ((preds != target_ids) & mask).sum().item()
#             fn = ((preds != target_ids) & mask).sum().item()
#
#             true_positive += tp
#             false_positive += fp
#             false_negative += fn
#
#             # Decode for BERTScore, BLEU, Exact Match
#             for p, t in zip(preds, target_ids):
#                 decoded_pred = tokenizer.decode(p.tolist())
#                 decoded_ref = tokenizer.decode(t.tolist())
#                 predictions.append(decoded_pred)
#                 references.append(decoded_ref)
#
#                 if decoded_pred.strip() == decoded_ref.strip():
#                     exact_match += 1
#                 total_sentences += 1
#
#     # -----------------------------------------------------
#     # Final Metrics
#     # -----------------------------------------------------
#
#     avg_loss = total_loss / len(test_loader)
#     token_accuracy = true_positive / (true_positive + false_negative + 1e-8)
#     precision = true_positive / (true_positive + false_positive + 1e-8)
#     recall = true_positive / (true_positive + false_negative + 1e-8)
#     sentence_accuracy = exact_match / total_sentences if total_sentences > 0 else 0
#
#     if BERTSCORE_AVAILABLE:
#         P, R, F1 = score(predictions[:200], references[:200], lang="hi", verbose=False)
#         bert_f1 = F1.mean().item()
#     else:
#         bert_f1 = 0.0
#
#     if BLEU_AVAILABLE:
#         bleu = sacrebleu.corpus_bleu(predictions, [references])
#         bleu_score = bleu.score
#     else:
#         bleu_score = 0.0
#
#     results = {
#         "model_type": CONFIG["model_type"],
#         "test_size": size,
#         "test_loss": round(avg_loss, 4),
#         "token_accuracy": round(token_accuracy, 4),
#         "precision": round(precision, 4),
#         "recall": round(recall, 4),
#         "sentence_exact_match": round(sentence_accuracy, 4),
#         "bert_f1": round(bert_f1, 4),
#         "bleu": round(bleu_score, 4)
#     }
#
#     # Save JSON
#     os.makedirs("results", exist_ok=True)
#     with open("results/test_metrics.json", "w") as f:
#         json.dump(results, f, indent=4)
#
#     # Print Results
#     print("\n📊 FINAL TEST RESULTS")
#     for k, v in results.items():
#         print(f"{k}: {v}")
#
#     print("\n✅ Results saved to results/test_metrics.json")
#
#
# # =========================================================
# # MAIN
# # =========================================================
#
# if __name__ == "__main__":
#     set_seed(42)
#     run_test(test_size=5000)
import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from bert_score import score
import sacrebleu

import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from bert_score import score
import sacrebleu

def run_test(test_size=5000):
    print("🧪 Starting Test Evaluation...\n")

    device = torch.device(CONFIG["training"]["device"])
    dtype = torch.float16 if CONFIG["training"]["precision"] == "float16" else torch.float32

    # -----------------------------------------------------
    # Load Tokenizer
    # -----------------------------------------------------
    tokenizer = SanskritTokenizer(CONFIG["model"]["vocab_size"])

    # -----------------------------------------------------
    # Load Model
    # -----------------------------------------------------
    model = SanskritModel(CONFIG)
    model.to(device, dtype=dtype)

    # model_path = f"production_model4/best2_{CONFIG['model_type']}.pt"
    model_path = f"NBaseline/production_model4/best_{CONFIG['model_type']}.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ Loaded model: {CONFIG['model_type']}")

    # -----------------------------------------------------
    # Load Test Dataset
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
            "target_ids": torch.stack([b["target_ids"].long() for b in batch]),
            "target_text": [b["target_text"] for b in batch]
        }

    test_loader = DataLoader(
        Subset(test_dataset, indices),
        batch_size=CONFIG["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate
    )

    print(f"📦 Test Samples: {size}\n")

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------
    total_loss = 0
    true_positive = 0
    total_tokens = 0
    exact_match = 0

    predictions = []
    references = []

    mask_token_id = CONFIG["diffusion"]["mask_token_id"]

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)

            # Diffusion vs Baseline
            if "d3pm" in CONFIG["model_type"]:
                t = torch.randint(
                    0,
                    CONFIG["model"]["diffusion_steps"],
                    (input_ids.size(0),),
                    device=device
                )
                logits, _ = model(input_ids, target_ids, t)
            else:
                logits = model(input_ids, target_ids)

            # Loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                ignore_index=mask_token_id
            )
            total_loss += loss.item()

            # Predictions
            preds = torch.argmax(logits, dim=-1)
            mask = target_ids != mask_token_id

            tp = ((preds == target_ids) & mask).sum().item()
            true_positive += tp
            total_tokens += mask.sum().item()

            # Decode for BERTScore, BLEU, Exact Match
            for p, t_ref in zip(preds, target_ids):
                decoded_pred = tokenizer.decode([id for id in p.tolist() if id != mask_token_id]).strip()
                decoded_ref  = tokenizer.decode([id for id in t_ref.tolist() if id != mask_token_id]).strip()
                predictions.append(decoded_pred)
                references.append(decoded_ref)

                if decoded_pred == decoded_ref:
                    exact_match += 1

    # -----------------------------------------------------
    # Final Metrics
    # -----------------------------------------------------
    avg_loss = total_loss / len(test_loader)
    token_accuracy = true_positive / (total_tokens + 1e-8)
    sentence_accuracy = exact_match / len(predictions) if predictions else 0.0

    # BERTScore
    if BERTSCORE_AVAILABLE and predictions:
        P, R, F1 = score(predictions, references, lang="hi", verbose=False)
        bert_f1 = F1.mean().item()
    else:
        bert_f1 = 0.0

    # BLEU
    if BLEU_AVAILABLE and predictions:
        bleu = sacrebleu.corpus_bleu(predictions, [references])
        bleu_score = bleu.score
    else:
        bleu_score = 0.0

    results = {
        "model_type": CONFIG["model_type"],
        "test_size": size,
        "test_loss": round(avg_loss, 4),
        "token_accuracy": round(token_accuracy, 4),
        "sentence_exact_match": round(sentence_accuracy, 4),
        "bert_f1": round(bert_f1, 4),
        "bleu": round(bleu_score, 4)
    }

    # Save JSON
    os.makedirs("results", exist_ok=True)
    with open("results/test_metrics_NBaseC.json", "w") as f:
        json.dump(results, f, indent=4)

    # Print Results
    print("\n📊 FINAL TEST RESULTS")
    for k, v in results.items():
        print(f"{k}: {v}")

    print("\n✅ Results saved to results/test_metrics.json")

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    set_seed(42)
    run_test(test_size=5000)