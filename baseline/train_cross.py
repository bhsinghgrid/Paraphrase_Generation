import torch
import torch.nn.functional as F
from bert_score import score
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
import evaluate

# Optional BERTScore via Hugging Face evaluate
try:
    import evaluate
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False

# Your modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import OptimizedSanskritDataset
from model.tokenizer import SanskritTokenizer
from model.sanskrit_model import SanskritModel


# =========================================================
# CONFIG
# =========================================================
CONFIG = {
    "model_type": "baseline_cross_attention",  # or baseline_cross_attention / baseline_encoder_decoder / d3pm_encoder_decoder
    "model": {
        "vocab_size": 16000,
        "max_seq_len": 80,
        "diffusion_steps": 12,
        "d_model": 384,
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
        "epochs": 7,
        "lr": 3e-4,
        "label_smoothing": 0.05,
        "precision": "float32",
        "device": "mps",
        "dataset_size": 60000,
        "early_stopping_patience": 3
    }
}


# =========================================================
# SET SEED
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


# =========================================================
# TRAINER
# =========================================================
class SanskritTrainer:

    def __init__(self, config=CONFIG):
        self.cfg = config
        self.device = torch.device(self.cfg["training"]["device"])
        self.dtype = torch.float16 if self.cfg["training"]["precision"] == "float16" else torch.float32

        if self.device.type == "mps":
            torch.mps.empty_cache()

        self._setup_logging()

        # Initialize Hugging Face BERTScore
        if BERTSCORE_AVAILABLE:
            self.bertscore = evaluate.load("bertscore")

    # -----------------------------
    # Logging
    # -----------------------------
    def _setup_logging(self):
        os.makedirs("production_model4", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
            handlers=[
                logging.FileHandler("results/training.log"),
                logging.StreamHandler()
            ]
        )

    # -----------------------------
    # Dataset & Collate
    # -----------------------------
    def _collate(self, batch):
        return {
            "input_ids": torch.stack([b["input_ids"].long() for b in batch]),
            "target_ids": torch.stack([b["target_ids"].long() for b in batch]),
            "input_text": [b["input_text"] for b in batch],
            "target_text": [b["target_text"] for b in batch]
        }

    def create_datasets(self):
        tokenizer = SanskritTokenizer(self.cfg["model"]["vocab_size"])
        full_dataset = OptimizedSanskritDataset(split="train", tokenizer=tokenizer, max_len=self.cfg["model"]["max_seq_len"])

        size = min(self.cfg["training"]["dataset_size"], len(full_dataset))
        indices = list(range(size))
        train_idx, val_idx = train_test_split(indices, train_size=0.8, random_state=42)

        self.train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=self.cfg["training"]["batch_size"],
                                       shuffle=True, drop_last=True, collate_fn=self._collate)
        self.val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=self.cfg["training"]["batch_size"],
                                     shuffle=False, collate_fn=self._collate)

        self.tokenizer = tokenizer
        print(f"✅ Train: {len(train_idx)} | Val: {len(val_idx)}")

    # -----------------------------
    # Model
    # -----------------------------
    def create_model(self):
        self.model = SanskritModel(self.cfg)
        self.model.to(self.device, dtype=self.dtype)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg["training"]["lr"], eps=1e-6)
        self.scheduler = OneCycleLR(self.optimizer, max_lr=self.cfg["training"]["lr"],
                                    epochs=self.cfg["training"]["epochs"],
                                    steps_per_epoch=len(self.train_loader))

        print(f"✅ Model params: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"🔥 Model type: {self.cfg['model_type']}")

    # -----------------------------
    # Training
    # -----------------------------
    def train_epoch(self):
        self.model.train()
        total_loss = 0

        for batch in tqdm(self.train_loader, desc="Train"):
            input_ids = batch["input_ids"].to(self.device)
            target_ids = batch["target_ids"].to(self.device)
            self.optimizer.zero_grad()

            if "d3pm" in self.cfg["model_type"]:
                t = torch.randint(0, self.cfg["model"]["diffusion_steps"], (input_ids.size(0),), device=self.device)
                logits, _ = self.model(input_ids, target_ids, t)
            else:
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

    # -----------------------------
    # Validation
    # -----------------------------
    def validate_epoch(self):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Val"):
                input_ids = batch["input_ids"].to(self.device)
                target_ids = batch["target_ids"].to(self.device)

                if "d3pm" in self.cfg["model_type"]:
                    t = torch.randint(0, self.cfg["model"]["diffusion_steps"], (input_ids.size(0),), device=self.device)
                    logits, _ = self.model(input_ids, target_ids, t)
                else:
                    logits = self.model(input_ids, target_ids)

                loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                       target_ids.view(-1),
                                       ignore_index=self.tokenizer.mask_token_id)
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    # -----------------------------
    # BERTScore
    # -----------------------------
    # def compute_bertscore(self, max_samples=500, batch_size=16):
    #     if not BERTSCORE_AVAILABLE:
    #         return 0.0
    #
    #     self.model.eval()
    #     predictions, references = [], []
    #
    #     with torch.no_grad():
    #         for batch in self.val_loader:
    #             input_ids = batch["input_ids"].to(self.device)
    #             refs = batch["target_text"]
    #
    #             # Generate outputs
    #             if "d3pm" in self.cfg["model_type"]:
    #                 generated = self.model.model.generate(input_ids, num_steps=8)
    #             else:
    #                 generated = self.model.model.generate(input_ids)
    #
    #             preds = [
    #                 self.tokenizer.decode([id for id in ids.tolist() if id != self.cfg["diffusion"]["mask_token_id"]]).strip()
    #                 for ids in generated
    #             ]
    #             predictions.extend(preds)
    #             references.extend(refs)
    #
    #             if len(predictions) >= max_samples:
    #                 predictions = predictions[:max_samples]
    #                 references = references[:max_samples]
    #                 break
    #
    #     if not predictions:
    #         return 0.0
    #
    #     results = self.bertscore.compute(predictions=predictions, references=references, lang="hi", batch_size=batch_size)
    #     return sum(results["f1"]) / len(results["f1"])

    # def compute_bertscore(self, max_samples=1000):
    #     """
    #     Compute BERTScore on the validation dataset.
    #     Works for both diffusion and baseline models.
    #     Args:
    #         max_samples (int, optional): limit number of samples for speed.
    #     Returns:
    #         F1 score (float)
    #     """
    #     if not BERTSCORE_AVAILABLE:
    #         return 0.0
    #
    #     self.model.eval()
    #     predictions, references = [], []
    #
    #     bertscore_metric = evaluate.load("bertscore")
    #
    #     with torch.no_grad():
    #         for batch in self.val_loader:
    #             input_ids = batch["input_ids"].to(self.device)
    #             refs = batch["target_text"]  # list of reference strings
    #
    #             # Generate outputs
    #             if "d3pm" in self.cfg["model_type"]:
    #                 generated = self.model.model.generate(input_ids, num_steps=8)
    #             else:
    #                 generated = self.model.model.generate(input_ids)
    #
    #             # Decode each sequence individually, ignoring mask token
    #             preds = [
    #                 self.tokenizer.decode(
    #                     [id for id in ids.tolist() if id != self.cfg["diffusion"]["mask_token_id"]]
    #                 ).strip()
    #                 for ids in generated
    #             ]
    #
    #             # Remove empty strings
    #             filtered_preds = [p for p in preds if len(p) > 0]
    #             filtered_refs = refs[:len(filtered_preds)]
    #
    #             predictions.extend(filtered_preds)
    #             references.extend(filtered_refs)
    #
    #             # Stop if max_samples is reached
    #             if max_samples and len(predictions) >= max_samples:
    #                 predictions = predictions[:max_samples]
    #                 references = references[:max_samples]
    #                 break
    #
    #     if len(predictions) == 0:
    #         return 0.0
    #
    #     # Compute BERTScore using evaluate
    #     results = bertscore_metric.compute(predictions=predictions, references=references, lang="hi")
    #     f1_score = float(np.mean(results["f1"]))
    #
    #     return f1_score
    from bert_score import score
    # def compute_bertscore(self, max_samples=500, debug=True):
    #     """
    #     Compute BERTScore on the entire validation/test dataset.
    #     Handles proper decoding, device placement, and special tokens.
    #
    #     Args:
    #         max_samples (int): number of sample predictions to print for debugging
    #         debug (bool): whether to print sample predictions
    #     """
    #     if not BERTSCORE_AVAILABLE:
    #         if debug:
    #             print("⚠️ BERTScore not available. Returning 0.0")
    #         return 0.0
    #
    #     self.model.eval()
    #     predictions, references = [], []
    #
    #     with torch.no_grad():
    #         for batch in self.val_loader:
    #             input_ids = batch["input_ids"].to(self.device)
    #             refs = batch["target_text"]  # list of reference strings
    #
    #             # Generate outputs
    #             if "d3pm" in self.cfg["model_type"]:
    #                 generated = self.model.model.generate(input_ids, num_steps=8)
    #             else:
    #                 generated = self.model.model.generate(input_ids)
    #
    #             # Decode each sequence individually, ignoring mask token
    #             preds = [
    #                 self.tokenizer.decode([id for id in ids.tolist()
    #                                        if id != CONFIG["diffusion"]["mask_token_id"]]).strip()
    #                 for ids in generated
    #             ]
    #
    #             predictions.extend(preds)
    #             references.extend(refs)
    #
    #     if debug:
    #         print(f"Collected {len(predictions)} predictions / {len(references)} references")
    #         # Print first few samples
    #         for i, (p, r) in enumerate(zip(predictions[:max_samples], references[:max_samples])):
    #             print(f"Sample {i + 1}:")
    #             print(f"  Prediction: {p}")
    #             print(f"  Reference : {r}\n")
    #
    #     if not predictions:
    #         return 0.0
    #
    #     # Compute BERTScore
    #     P, R, F1 = score(predictions, references, lang="hi", verbose=False)
    #     return F1.mean().item()

    # -----------------------------
    # Robust BERTScore for validation
    # -----------------------------
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

    # -----------------------------
    # TRAIN LOOP
    # -----------------------------
    def train(self):
        print("🔥 Training Started")
        self.create_datasets()
        self.create_model()

        best_val_loss = float("inf")
        patience = self.cfg["training"]["early_stopping_patience"]
        patience_counter = 0

        self.history = []

        for epoch in range(self.cfg["training"]["epochs"]):
            print(f"\n🔥 EPOCH {epoch + 1}")

            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()
            bert_f1 = self.compute_bertscore()

            print(f"📊 Train: {train_loss:.4f} | Val: {val_loss:.4f} | BERT F1: {bert_f1:.4f}")

            # Save per-epoch history
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
                torch.save(self.model.state_dict(), f"production_model4/best_{self.cfg['model_type']}.pt")
                print("🎉 Best model saved.")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("🛑 Early stopping triggered.")
                    break

        with open("production_model4/training_metadata.json", "w") as f:
            json.dump(self.history, f, indent=4)

        print("✅ Training Complete")


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    set_seed(42)
    trainer = SanskritTrainer()
    trainer.train()