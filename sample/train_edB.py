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
import os

# Optional BERTScore
try:
    from bert_score import score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False

# Your modules
from data.dataset import OptimizedSanskritDataset
from model.tokenizer import SanskritTokenizer
from model.new_d3pm_model import SanskritModel


# =========================================================
# CONFIG
# =========================================================

CONFIG = {
    # "model_type": "d3pm_cross_attention",  # 🔥 Change here to switch models
    # "model_type": "baseline_cross_attention",
    # "model_type": "baseline_encoder_decoder",
    "model_type": "d3pm_encoder_decoder",
    "model": {
        "vocab_size": 16000,
        "max_seq_len": 80,
        "diffusion_steps": 12, #8
        "d_model": 384,
        "n_layers": 6,
        "n_heads": 8,
        "d_ff": 1536,
        "dropout": 0.2 # 0.1
    },

    "diffusion": {
        "mask_token_id": 0
    },

    "training": {
        "batch_size": 16,
        "epochs": 9,
        "lr": 3e-4,
        "label_smoothing": 0.05,
        "precision": "float32",
        "device": "mps",
        "dataset_size": 60000, # 50000
        "early_stopping_patience": 3
    }
}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic flags
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For PyTorch >= 1.12
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

    # -----------------------------------------------------
    # LOGGING
    # -----------------------------------------------------

    def _setup_logging(self):
        os.makedirs("../production_model3", exist_ok=True)
        os.makedirs("../results1", exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
            handlers=[
                logging.FileHandler("../results1/training.log"),
                logging.StreamHandler()
            ]
        )

    # -----------------------------------------------------
    # DATA
    # -----------------------------------------------------

    def _collate(self, batch):
        return {
            "input_ids": torch.stack([b["input_ids"].long() for b in batch]),
            "target_ids": torch.stack([b["target_ids"].long() for b in batch]),
            "input_text": [b["input_text"] for b in batch],
            "target_text": [b["target_text"] for b in batch]
        }

    def create_datasets(self):
        tokenizer = SanskritTokenizer(self.cfg["model"]["vocab_size"])

        full_dataset = OptimizedSanskritDataset(
            split="train",
            tokenizer=tokenizer,
            max_len=self.cfg["model"]["max_seq_len"]
        )

        size = min(self.cfg["training"]["dataset_size"], len(full_dataset))
        indices = list(range(size))
        train_idx, val_idx = train_test_split(indices, train_size=0.8, random_state=42)

        self.train_loader = DataLoader(
            Subset(full_dataset, train_idx),
            batch_size=self.cfg["training"]["batch_size"],
            shuffle=True,
            drop_last=True,
            collate_fn=self._collate
        )

        self.val_loader = DataLoader(
            Subset(full_dataset, val_idx),
            batch_size=self.cfg["training"]["batch_size"],
            shuffle=False,
            collate_fn=self._collate
        )

        self.tokenizer = tokenizer

        print(f"✅ Train: {len(train_idx)} | Val: {len(val_idx)}")

    # -----------------------------------------------------
    # MODEL
    # -----------------------------------------------------

    def create_model(self):
        self.model = SanskritModel(self.cfg)
        self.model.to(self.device, dtype=self.dtype)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg["training"]["lr"],
            eps=1e-6
        )

        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.cfg["training"]["lr"],
            epochs=self.cfg["training"]["epochs"],
            steps_per_epoch=len(self.train_loader)
        )

        print(f"✅ Model params: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"🔥 Model type: {self.cfg['model_type']}")

    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        for batch in tqdm(self.train_loader, desc="Train"):
            input_ids = batch["input_ids"].to(self.device)
            target_ids = batch["target_ids"].to(self.device)

            self.optimizer.zero_grad()

            # 🔥 Handle Diffusion vs Baseline Automatically
            if "d3pm" in self.cfg["model_type"]:
                t = torch.randint(
                    0,
                    self.cfg["model"]["diffusion_steps"],
                    (input_ids.size(0),),
                    device=self.device
                )

                logits, _ = self.model(input_ids, target_ids, t)

            else:
                logits = self.model(input_ids, target_ids)

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                ignore_index=self.tokenizer.mask_token_id,
                label_smoothing=self.cfg["training"]["label_smoothing"]
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    # -----------------------------------------------------
    # VALIDATE
    # -----------------------------------------------------

    def validate_epoch(self):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Val"):
                input_ids = batch["input_ids"].to(self.device)
                target_ids = batch["target_ids"].to(self.device)

                if "d3pm" in self.cfg["model_type"]:
                    t = torch.randint(
                        0,
                        self.cfg["model"]["diffusion_steps"],
                        (input_ids.size(0),),
                        device=self.device
                    )
                    logits, _ = self.model(input_ids, target_ids, t)
                else:
                    logits = self.model(input_ids, target_ids)

                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target_ids.view(-1),
                    ignore_index=self.tokenizer.mask_token_id
                )

                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    # -----------------------------------------------------
    # BERTSCORE
    # -----------------------------------------------------

    # def compute_bertscore(self, max_samples=20):
    #     if not BERTSCORE_AVAILABLE:
    #         return 0.0
    #
    #     self.model.eval()
    #     predictions, references = [], []
    #
    #     with torch.no_grad():
    #         for i, batch in enumerate(self.val_loader):
    #             if i >= max_samples:
    #                 break
    #
    #             input_ids = batch["input_ids"].to(self.device)[:2]
    #             refs = batch["target_text"][:2]
    #
    #             if "d3pm" in self.cfg["model_type"]:
    #                 generated = self.model.model.generate(input_ids, num_steps=8)
    #             else:
    #                 generated = self.model.model.generate(input_ids)
    #
    #             preds = [self.tokenizer.decode(ids) for ids in generated]
    #
    #             predictions.extend(preds)
    #             references.extend(refs)
    #
    #     if not predictions:
    #         return 0.0
    #
    #     P, R, F1 = score(predictions, references, lang="hi", verbose=False)
    #     return F1.mean().item()
    # def compute_bertscore(self):
    #     """
    #     Compute BERTScore on the entire validation/test dataset.
    #     Handles proper decoding, device placement, and special tokens.
    #     """
    #     if not BERTSCORE_AVAILABLE:
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
    #             # Decode each sequence individually
    #             preds = [
    #                 # self.tokenizer.decode(ids.tolist(), skip_special_tokens=True)
    #                 self.tokenizer.decode(
    #                     [id for id in ids.tolist() if id != CONFIG["diffusion"]["mask_token_id"]]).strip()
    #                 for ids in generated
    #             ]
    #
    #             predictions.extend(preds)
    #             references.extend(refs)
    #
    #     if not predictions:
    #         return 0.0
    #
    #     # Compute BERTScore
    #     P, R, F1 = score(predictions, references, lang="hi", verbose=False)
    #     return F1.mean().item()
    def compute_bertscore(self, max_samples=None, batch_size=16):
        """
        Stable + Fast BERTScore
        - Uses multilingual BERT (better for Sanskrit/Devanagari)
        - Filters empty predictions
        - Runs safely on CPU (avoids MPS crash)
        - Supports max_samples for speed control
        """

        if not BERTSCORE_AVAILABLE:
            return 0.0

        from bert_score import score

        self.model.eval()
        predictions, references = [], []

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                refs = batch["target_text"]

                # Generate outputs
                if "d3pm" in self.cfg["model_type"]:
                    generated = self.model.model.generate(input_ids, num_steps=8)
                else:
                    generated = self.model.model.generate(input_ids)

                # Decode safely
                for ids, ref in zip(generated, refs):
                    decoded = self.tokenizer.decode(
                        [i for i in ids.tolist()
                         if i != self.cfg["diffusion"]["mask_token_id"]]
                    ).strip()

                    # Skip empty predictions (fixes your crash)
                    if decoded:
                        predictions.append(decoded)
                        references.append(ref)

                    if max_samples and len(predictions) >= max_samples:
                        break

                if max_samples and len(predictions) >= max_samples:
                    break

        if len(predictions) == 0:
            return 0.0

        # 🚨 IMPORTANT: Run BERTScore on CPU (MPS safer)
        P, R, F1 = score(
            predictions,
            references,
            model_type="bert-base-multilingual-cased",
            lang="hi",
            batch_size=batch_size,
            device="cpu",  # safer than MPS
            verbose=False
        )

        return F1.mean().item()

    # -----------------------------------------------------
    # MAIN TRAIN LOOP
    # -----------------------------------------------------

    def train(self):
        print("🔥 Training Started")
        self.create_datasets()
        self.create_model()

        best_val_loss = float("inf")
        patience = self.cfg["training"]["early_stopping_patience"]
        patience_counter = 0

        for epoch in range(self.cfg["training"]["epochs"]):

            print(f"\n🔥 EPOCH {epoch + 1}")

            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()
            bert_f1 = self.compute_bertscore()

            print(f"📊 Train: {train_loss:.4f} | "
                  f"Val: {val_loss:.4f} | "
                  f"BERT: {bert_f1:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                torch.save(
                    self.model.state_dict(),
                    f"production_model3/best_{self.cfg['model_type']}.pt"
                )

                print("🎉 Best model saved.")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("🛑 Early stopping triggered.")
                    break

        print("✅ Training Complete")


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    set_seed(42)
    trainer = SanskritTrainer()
    trainer.train()