# import torch
# import torch.nn.functional as F
# from torch.optim.lr_scheduler import OneCycleLR
# from torch.utils.data import DataLoader, Subset
# from sklearn.model_selection import train_test_split
# from tqdm import tqdm
# import os
# import logging
# import random
# import numpy as np
# import json
# import sys
#
# # Hugging Face BERTScore
# import evaluate
# BERTSCORE_AVAILABLE = True
#
# # Your modules
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from data.dataset import OptimizedSanskritDataset
# from model.tokenizer import SanskritTokenizer
# from model.sanskrit_model import SanskritModel
#
#
# CONFIG = {
#     "model_type": "baseline_cross_attention",  # or baseline_encoder_decoder / d3pm_encoder_decoder
#     "model": {
#         "vocab_size": 16000,
#         "max_seq_len": 80,
#         "diffusion_steps": 12,
#         "d_model": 384,
#         "n_layers": 6,
#         "n_heads": 8,
#         "d_ff": 1536,
#         "dropout": 0.2
#     },
#     "diffusion": {"mask_token_id": 0},
#     "training": {
#         "batch_size": 16,
#         "epochs": 7,
#         "lr": 3e-4,
#         "label_smoothing": 0.05,
#         "precision": "float32",
#         "device": "mps",
#         "dataset_size": 60000,
#         "early_stopping_patience": 3
#     }
# }
#
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
#
#
# class SanskritTrainer:
#
#     def __init__(self, config=CONFIG):
#         self.cfg = config
#         self.device = torch.device(self.cfg["training"]["device"])
#         self.dtype = torch.float16 if self.cfg["training"]["precision"] == "float16" else torch.float32
#
#         if self.device.type == "mps":
#             torch.mps.empty_cache()
#
#         self._setup_logging()
#         try:
#             self.bertscore = evaluate.load("bertscore")
#         except Exception as e:
#             logging.warning(f"Could not load BERTScore: {e}")
#             self.bertscore = None
#
#     # -----------------------------
#     def _setup_logging(self):
#         os.makedirs("production_model4", exist_ok=True)
#         os.makedirs("results", exist_ok=True)
#         logging.basicConfig(
#             level=logging.INFO,
#             format="%(asctime)s - %(message)s",
#             handlers=[
#                 logging.FileHandler("results/training.log"),
#                 logging.StreamHandler()
#             ]
#         )
#
#     # -----------------------------
#     def _collate(self, batch):
#         return {
#             "input_ids": torch.stack([b["input_ids"].long() for b in batch]),
#             "target_ids": torch.stack([b["target_ids"].long() for b in batch]),
#             "input_text": [b["input_text"] for b in batch],
#             "target_text": [b["target_text"] for b in batch]
#         }
#
#     def create_datasets(self):
#         tokenizer = SanskritTokenizer(self.cfg["model"]["vocab_size"])
#         full_dataset = OptimizedSanskritDataset(split="train", tokenizer=tokenizer,
#                                                 max_len=self.cfg["model"]["max_seq_len"])
#         size = min(self.cfg["training"]["dataset_size"], len(full_dataset))
#         indices = list(range(size))
#         train_idx, val_idx = train_test_split(indices, train_size=0.8, random_state=42)
#
#         self.train_loader = DataLoader(Subset(full_dataset, train_idx),
#                                        batch_size=self.cfg["training"]["batch_size"],
#                                        shuffle=True, drop_last=True, collate_fn=self._collate)
#         self.val_loader = DataLoader(Subset(full_dataset, val_idx),
#                                      batch_size=self.cfg["training"]["batch_size"],
#                                      shuffle=False, collate_fn=self._collate)
#
#         self.tokenizer = tokenizer
#         print(f"✅ Train: {len(train_idx)} | Val: {len(val_idx)}")
#
#     # -----------------------------
#     def create_model(self):
#         self.model = SanskritModel(self.cfg)
#         self.model.to(self.device, dtype=self.dtype)
#         self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg["training"]["lr"], eps=1e-6)
#         self.scheduler = OneCycleLR(self.optimizer, max_lr=self.cfg["training"]["lr"],
#                                     epochs=self.cfg["training"]["epochs"], steps_per_epoch=len(self.train_loader))
#         print(f"✅ Model params: {sum(p.numel() for p in self.model.parameters()):,}")
#         print(f"🔥 Model type: {self.cfg['model_type']}")
#
#     # -----------------------------
#     def train_epoch(self):
#         self.model.train()
#         total_loss = 0
#         for batch in tqdm(self.train_loader, desc="Train"):
#             input_ids = batch["input_ids"].to(self.device)
#             target_ids = batch["target_ids"].to(self.device)
#             self.optimizer.zero_grad()
#
#             if "d3pm" in self.cfg["model_type"]:
#                 t = torch.randint(0, self.cfg["model"]["diffusion_steps"], (input_ids.size(0),), device=self.device)
#                 logits, _ = self.model(input_ids, target_ids, t)
#             else:
#                 logits = self.model(input_ids, target_ids)
#
#             loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
#                                    target_ids.view(-1),
#                                    ignore_index=self.tokenizer.mask_token_id,
#                                    label_smoothing=self.cfg["training"]["label_smoothing"])
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
#             self.optimizer.step()
#             self.scheduler.step()
#             total_loss += loss.item()
#         return total_loss / len(self.train_loader)
#
#     # -----------------------------
#     def validate_epoch(self):
#         self.model.eval()
#         total_loss = 0
#         with torch.no_grad():
#             for batch in tqdm(self.val_loader, desc="Val"):
#                 input_ids = batch["input_ids"].to(self.device)
#                 target_ids = batch["target_ids"].to(self.device)
#
#                 if "d3pm" in self.cfg["model_type"]:
#                     t = torch.randint(0, self.cfg["model"]["diffusion_steps"], (input_ids.size(0),), device=self.device)
#                     logits, _ = self.model(input_ids, target_ids, t)
#                 else:
#                     logits = self.model(input_ids, target_ids)
#
#                 loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
#                                        target_ids.view(-1),
#                                        ignore_index=self.tokenizer.mask_token_id)
#                 total_loss += loss.item()
#         return total_loss / len(self.val_loader)
#
#     # -----------------------------
#     # Robust BERTScore
#     # -----------------------------
#     # def compute_bertscore(self, max_samples=1000):
#     #     if self.bertscore is None:
#     #         return 0.0
#     #     self.model.eval()
#     #     predictions, references = [], []
#     #
#     #     with torch.no_grad():
#     #         for batch in self.val_loader:
#     #             input_ids = batch["input_ids"].to(self.device)
#     #             refs = batch["target_text"]
#     #
#     #             # Generate predictions
#     #             try:
#     #                 if "d3pm" in self.cfg.get("model_type", ""):
#     #                     generated = self.model.model.generate(input_ids, num_steps=8)
#     #                 else:
#     #                     generated = self.model.generate(input_ids)
#     #             except Exception as e:
#     #                 logging.error(f"Generation failed: {e}")
#     #                 break
#     #
#     #             # Decode predictions
#     #             for pred in generated:
#     #                 if isinstance(pred, torch.Tensor):
#     #                     ids = [i for i in pred.tolist() if i != self.cfg["diffusion"]["mask_token_id"]]
#     #                     decoded = self.tokenizer.decode(ids).strip()
#     #                     if decoded:
#     #                         predictions.append(decoded)
#     #
#     #             references.extend(refs)
#     #
#     #             if max_samples and len(predictions) >= max_samples:
#     #                 predictions = predictions[:max_samples]
#     #                 references = references[:max_samples]
#     #                 break
#     #
#     #     if len(predictions) == 0:
#     #         return 0.0
#     #
#     #     results = self.bertscore.compute(predictions=predictions, references=references, lang="hi", batch_size=16)
#     #     f1 = float(np.mean(results["f1"]))
#     #     return f1
#
#     def compute_bertscore(self, max_samples=500, debug=True):
#         """
#         Compute BERTScore for validation set, with debug info.
#         Works for baseline and diffusion models.
#
#         Args:
#             max_samples (int): limit number of samples to speed up.
#             debug (bool): whether to print debug info.
#         Returns:
#             float: mean F1 score
#         """
#         if not BERTSCORE_AVAILABLE:
#             if debug:
#                 print("⚠️ BERTScore not available. Returning 0.0")
#             return 0.0
#
#         from bert_score import score
#
#         self.model.eval()
#         predictions, references = [], []
#
#         with torch.no_grad():
#             for batch in self.val_loader:
#                 input_ids = batch["input_ids"].to(self.device)
#                 refs = batch["target_text"]
#
#                 # Generate predictions
#                 if "d3pm" in self.cfg["model_type"]:
#                     generated = self.model.model.generate(input_ids, num_steps=8)
#                 else:
#                     generated = self.model.generate(input_ids)
#
#                 # Decode predictions safely
#                 decoded_preds = [
#                     self.tokenizer.decode([id for id in seq.tolist()
#                                            if id != CONFIG["diffusion"]["mask_token_id"]]).strip()
#                     for seq in generated
#                 ]
#
#                 # Filter out empty predictions
#                 nonempty = [(p, r) for p, r in zip(decoded_preds, refs) if p.strip()]
#                 if not nonempty:
#                     continue
#                 p_f, r_f = zip(*nonempty)
#
#                 predictions.extend(p_f)
#                 references.extend(r_f)
#
#                 if max_samples and len(predictions) >= max_samples:
#                     predictions = predictions[:max_samples]
#                     references = references[:max_samples]
#                     break
#
#         if debug:
#             print(f"Collected {len(predictions)} predictions / {len(references)} references")
#             if len(predictions) > 0:
#                 print("Sample predictions:", predictions[:3])
#                 print("Sample references:", references[:3])
#
#         if not predictions:
#             return 0.0
#
#         # Compute BERTScore
#         P, R, F1 = score(predictions, references, lang="hi", batch_size=16, device=self.device)
#         f1_mean = F1.mean().item()
#
#         # Fallback semantic similarity if F1 is zero
#         if f1_mean == 0.0:
#             if debug:
#                 print("⚠️ BERTScore F1=0. Using semantic fallback via sentence-transformers...")
#             try:
#                 from sentence_transformers import SentenceTransformer, util
#                 sem_model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
#                 emb_preds = sem_model.encode(predictions, convert_to_tensor=True)
#                 emb_refs = sem_model.encode(references, convert_to_tensor=True)
#                 f1_mean = util.cos_sim(emb_preds, emb_refs).diag().mean().item()
#                 if debug:
#                     print(f"Semantic fallback score: {f1_mean:.4f}")
#             except Exception as e:
#                 if debug:
#                     print(f"Semantic fallback failed: {e}")
#
#         return f1_mean
#
#     # -----------------------------
#     def train(self):
#         print("🔥 Training Started")
#         self.create_datasets()
#         self.create_model()
#
#         best_val_loss = float("inf")
#         patience = self.cfg["training"]["early_stopping_patience"]
#         patience_counter = 0
#         self.history = []
#
#         for epoch in range(self.cfg["training"]["epochs"]):
#             print(f"\n🔥 EPOCH {epoch + 1}")
#             train_loss = self.train_epoch()
#             val_loss = self.validate_epoch()
#             bert_f1 = self.compute_bertscore(max_samples=128)  # small subset for speed
#
#             print(f"📊 Train: {train_loss:.4f} | Val: {val_loss:.4f} | BERT F1: {bert_f1:.4f}")
#
#             self.history.append({
#                 "epoch": epoch + 1,
#                 "train_loss": float(train_loss),
#                 "val_loss": float(val_loss),
#                 "bert_f1": float(bert_f1),
#                 "lr": float(self.optimizer.param_groups[0]["lr"])
#             })
#
#             if val_loss < best_val_loss:
#                 best_val_loss = val_loss
#                 patience_counter = 0
#                 torch.save(self.model.state_dict(), f"production_model4/best_{self.cfg['model_type']}.pt")
#                 print("🎉 Best model saved.")
#             else:
#                 patience_counter += 1
#                 if patience_counter >= patience:
#                     print("🛑 Early stopping triggered.")
#                     break
#
#         with open("production_model4/training_metadata.json", "w") as f:
#             json.dump(self.history, f, indent=4)
#         print("✅ Training Complete")
#
#
# if __name__ == "__main__":
#     set_seed(42)
#     trainer = SanskritTrainer()
#     trainer.train()
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import logging
import random
import numpy as np
import json
import sys
import traceback
from bert_score import  score
# Hugging Face evaluate for BERTScore
import evaluate
BERTSCORE_AVAILABLE = True

# Your modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import OptimizedSanskritDataset
from model.tokenizer import SanskritTokenizer
from model.sanskrit_model import SanskritModel

# ------------------------------
# CONFIG
# ------------------------------
CONFIG = {
    "model_type": "baseline_cross_attention",
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
    "diffusion": {"mask_token_id": 0},
    "training": {
        "batch_size": 16,
        "epochs": 7,
        "lr": 3e-4,
        "label_smoothing": 0.05,
        "precision": "float32",
        "device": "mps",
        "dataset_size": 60000,
        "early_stopping_patience": 3
    }
}

# ------------------------------
# Set Seed
# ------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"🔒 Seed set to {seed} (Deterministic mode enabled)")

# ------------------------------
# Trainer Class
# ------------------------------
class SanskritTrainer:
    def __init__(self, config=CONFIG, debug=True):
        self.cfg = config
        self.debug = debug
        self.device = torch.device(self.cfg["training"]["device"])
        self.dtype = torch.float16 if self.cfg["training"]["precision"] == "float16" else torch.float32

        if self.device.type == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

        self._setup_logging()

        # Load BERTScore metric
        try:
            self.bertscore = evaluate.load("bertscore")
            logging.info("Loaded evaluate.bertscore successfully.")
        except Exception as e:
            logging.warning(f"Could not load evaluate.bertscore: {e}")
            self.bertscore = None

    # --------------------------
    # Logging
    # --------------------------
    def _setup_logging(self):
        os.makedirs("production_model_debug", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        logging.basicConfig(
            level=logging.INFO if not self.debug else logging.DEBUG,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("results/training_debug.log"),
                logging.StreamHandler()
            ]
        )

    # --------------------------
    # Dataset
    # --------------------------
    def _collate(self, batch):
        return {
            "input_ids": torch.stack([b["input_ids"].long() for b in batch]),
            "target_ids": torch.stack([b["target_ids"].long() for b in batch]),
            "input_text": [b["input_text"] for b in batch],
            "target_text": [b["target_text"] for b in batch]
        }

    def create_datasets(self):
        logging.info("Creating datasets & tokenizer...")
        self.tokenizer = SanskritTokenizer(self.cfg["model"]["vocab_size"])
        full_dataset = OptimizedSanskritDataset(split="train", tokenizer=self.tokenizer,
                                                max_len=self.cfg["model"]["max_seq_len"])
        size = min(self.cfg["training"]["dataset_size"], len(full_dataset))
        indices = list(range(size))
        train_idx, val_idx = train_test_split(indices, train_size=0.8, random_state=42)

        self.train_loader = DataLoader(Subset(full_dataset, train_idx),
                                       batch_size=self.cfg["training"]["batch_size"],
                                       shuffle=True, drop_last=True, collate_fn=self._collate)
        self.val_loader = DataLoader(Subset(full_dataset, val_idx),
                                     batch_size=self.cfg["training"]["batch_size"],
                                     shuffle=False, collate_fn=self._collate)
        logging.info(f"✅ Train: {len(train_idx)} | Val: {len(val_idx)}")

    # --------------------------
    # Model
    # --------------------------
    def create_model(self):
        logging.info("Creating model...")
        self.model = SanskritModel(self.cfg)
        self.model.to(self.device, dtype=self.dtype)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg["training"]["lr"], eps=1e-6)
        self.scheduler = OneCycleLR(self.optimizer, max_lr=self.cfg["training"]["lr"],
                                    epochs=self.cfg["training"]["epochs"],
                                    steps_per_epoch=len(self.train_loader))
        logging.info(f"✅ Model params: {sum(p.numel() for p in self.model.parameters()):,}")
        logging.info(f"🔥 Model type: {self.cfg['model_type']}")

    # --------------------------
    # Train / Validation Epoch
    # --------------------------
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        for batch in tqdm(self.train_loader, desc="Train"):
            input_ids = batch["input_ids"].to(self.device)
            target_ids = batch["target_ids"].to(self.device)
            self.optimizer.zero_grad()

            logits = self.model(input_ids, target_ids)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   target_ids.view(-1),
                                   ignore_index=self.tokenizer.mask_token_id,
                                   label_smoothing=self.cfg["training"]["label_smoothing"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate_epoch(self):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Val"):
                input_ids = batch["input_ids"].to(self.device)
                target_ids = batch["target_ids"].to(self.device)

                logits = self.model(input_ids, target_ids)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                       target_ids.view(-1),
                                       ignore_index=self.tokenizer.mask_token_id)
                total_loss += loss.item()
        return total_loss / len(self.val_loader)

    # --------------------------
    # Baseline Generation (argmax)
    # --------------------------
    def generate_baseline_output(self, input_ids, max_len=None):
        """Generates non-empty token ID sequences for baseline Transformer."""
        self.model.eval()
        with torch.no_grad():
            batch_size = input_ids.size(0)
            device = input_ids.device
            max_len = max_len or self.cfg["model"]["max_seq_len"]
            start_id = getattr(self.tokenizer, "bos_token_id", 0)

            generated = torch.full((batch_size, max_len), start_id, dtype=torch.long, device=device)
            for t in range(max_len):
                logits = self.model(input_ids, generated)
                token_ids = torch.argmax(logits[:, t, :], dim=-1)
                generated[:, t] = token_ids
            return [generated[i].tolist() for i in range(batch_size)]

    # --------------------------
    # Safe Decode
    # --------------------------
    def safe_decode(self, id_seq):
        if isinstance(id_seq, torch.Tensor):
            ids = id_seq.tolist()
        else:
            ids = list(id_seq)
        ids = [t for t in ids if t != self.cfg["diffusion"]["mask_token_id"]]
        if len(ids) == 0:
            return ""
        try:
            return self.tokenizer.decode(ids).strip()
        except Exception:
            return " ".join(map(str, ids))

    # --------------------------
    # Robust BERTScore
    # --------------------------
    # def compute_bertscore(self, max_samples=128, batch_score_size=16):
    #     if self.bertscore is None:
    #         logging.warning("BERTScore metric not loaded. Returning 0.0")
    #         return 0.0
    #
    #     self.model.eval()
    #     predictions, references = [], []
    #
    #     with torch.no_grad():
    #         for batch in self.val_loader:
    #             input_ids = batch["input_ids"].to(self.device)
    #             refs = batch["target_text"]
    #             generated_raw = self.generate_baseline_output(input_ids)
    #             decoded = [self.safe_decode(seq) for seq in generated_raw]
    #
    #             # collect non-empty
    #             for pred, ref in zip(decoded, refs):
    #                 if pred.strip():
    #                     predictions.append(pred)
    #                     references.append(ref)
    #             if max_samples and len(predictions) >= max_samples:
    #                 predictions = predictions[:max_samples]
    #                 references = references[:max_samples]
    #                 break
    #
    #     if len(predictions) == 0:
    #         logging.warning("No valid predictions for BERTScore. Returning 0.0")
    #         return 0.0
    #
    #     try:
    #         results = self.bertscore.compute(predictions=predictions, references=references,
    #                                          lang="hi", batch_size=batch_score_size)
    #         mean_f1 = float(np.mean(results["f1"]))
    #         return mean_f1
    #     except Exception as e:
    #         logging.error(f"BERTScore computation failed: {e}")
    #         return 0.0
    # def compute_bertscore(self, max_samples=128, batch_score_size=16):
    #     """
    #     Compute semantic accuracy for validation set.
    #     Uses BERTScore (preferred), fallback to sentence-transformers if needed.
    #     """
    #     self.model.eval()
    #     predictions, references = [], []
    #
    #     with torch.no_grad():
    #         for batch in self.val_loader:
    #             input_ids = batch["input_ids"].to(self.device)
    #             refs = batch["target_text"]
    #             # Generate predictions
    #             generated_raw = self.generate_baseline_output(input_ids)
    #             decoded = [self.safe_decode(seq) for seq in generated_raw]
    #
    #             # collect non-empty
    #             for pred, ref in zip(decoded, refs):
    #                 if pred.strip():
    #                     predictions.append(pred)
    #                     references.append(ref)
    #             if max_samples and len(predictions) >= max_samples:
    #                 predictions = predictions[:max_samples]
    #                 references = references[:max_samples]
    #                 break
    #
    #     if len(predictions) == 0:
    #         logging.warning("No valid predictions for semantic similarity. Returning 0.0")
    #         return 0.0
    #
    #     # Try BERTScore first
    #     try:
    #         if BERTSCORE_AVAILABLE:
    #             results = self.bertscore.compute(predictions=predictions,
    #                                              references=references,
    #                                              lang="hi",
    #                                              batch_size=batch_score_size)
    #             f1_mean = float(np.mean(results["f1"]))
    #             if f1_mean > 0:
    #                 return f1_mean
    #     except Exception as e:
    #         logging.warning(f"BERTScore failed: {e}")
    #
    #     # Fallback: Sentence-BERT cosine similarity
    #     try:
    #         from sentence_transformers import SentenceTransformer, util
    #         sem_model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
    #         emb_preds = sem_model.encode(predictions, convert_to_tensor=True)
    #         emb_refs = sem_model.encode(references, convert_to_tensor=True)
    #         f1_mean = util.cos_sim(emb_preds, emb_refs).diag().mean().item()
    #         logging.info(f"Semantic fallback score: {f1_mean:.4f}")
    #         return f1_mean
    #     except Exception as e:
    #         logging.error(f"Sentence-BERT fallback failed: {e}")
    #         return 0.0
    def compute_bertscore(self, max_samples=500, debug=True):
        """
        Compute semantic BERTScore on validation dataset safely for SanskritTokenizer.
        Uses plain string decoding to avoid tokenizer API errors.
        """
        if not BERTSCORE_AVAILABLE:
            if debug:
                print("⚠️ BERTScore not available. Returning 0.0")
            return 0.0

        self.model.eval()
        predictions, references = [], []
        mask_token_id = self.cfg["diffusion"]["mask_token_id"]

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                target_ids = batch["target_ids"].to(self.device)
                refs = batch["target_text"]

                # Generate logits
                if "d3pm" in self.cfg["model_type"]:
                    t = torch.randint(0, self.cfg["model"]["diffusion_steps"], (input_ids.size(0),), device=self.device)
                    logits, _ = self.model(input_ids, target_ids, t)
                else:
                    logits = self.model(input_ids, target_ids)

                # Predicted tokens
                preds_ids = torch.argmax(logits, dim=-1)

                # Decode to plain text
                for pred_seq, ref_text in zip(preds_ids, refs):
                    pred_text = self.tokenizer.decode([id for id in pred_seq.tolist() if id != mask_token_id]).strip()
                    if pred_text:
                        predictions.append(pred_text)
                        references.append(ref_text)

                # Stop early if we have enough samples
                if max_samples and len(predictions) >= max_samples:
                    predictions = predictions[:max_samples]
                    references = references[:max_samples]
                    break

        if debug:
            print(f"Collected {len(predictions)} predictions / {len(references)} references")
            for i, (p, r) in enumerate(
                    zip(predictions[:min(5, len(predictions))], references[:min(5, len(references))])):
                print(f"Sample {i + 1}:")
                print(f"  Prediction: {p}")
                print(f"  Reference : {r}")

        if not predictions:
            return 0.0

        # Compute BERTScore F1
        P, R, F1 = score(predictions, references, lang="hi", verbose=False)
        return F1.mean().item()


    # --------------------------
    # Train Loop
    # --------------------------
    def train(self):
        logging.info("🔥 Training Started")
        self.create_datasets()
        self.create_model()
        best_val_loss = float("inf")
        patience_counter = 0
        self.history = []

        for epoch in range(self.cfg["training"]["epochs"]):
            logging.info(f"\n🔥 EPOCH {epoch + 1}")
            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()
            bert_f1 = self.compute_bertscore(max_samples=128, batch_score_size=16)
            logging.info(f"📊 Train: {train_loss:.4f} | Val: {val_loss:.4f} | BERT F1: {bert_f1:.4f}")

            self.history.append({
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "bert_f1": float(bert_f1),
                "lr": float(self.optimizer.param_groups[0]["lr"])
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(),
                           f"production_model_debug/best_{self.cfg['model_type']}.pt")
                logging.info("🎉 Best model saved.")
            else:
                patience_counter += 1
                if patience_counter >= self.cfg["training"]["early_stopping_patience"]:
                    logging.info("🛑 Early stopping triggered.")
                    break

        with open("production_model_debug/training_metadata.json", "w") as f:
            json.dump(self.history, f, indent=4)
        logging.info("✅ Training Complete")

# --------------------------
# Run
# --------------------------
if __name__ == "__main__":
    set_seed(42)
    trainer = SanskritTrainer(CONFIG, debug=True)
    trainer.train()