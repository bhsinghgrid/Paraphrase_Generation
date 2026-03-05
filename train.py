import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import sys
import logging
import json
import evaluate
import random
import numpy as np

from config import CONFIG

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data.dataset import OptimizedSanskritDataset
from model.tokenizer import SanskritTokenizer
from model.sanskrit_model import SanskritModel


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior (slight speed penalty)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if torch.backends.mps.is_available():
        # For your M4 Pro
        torch.mps.manual_seed(seed)

class FinalTrainer:
    def __init__(self):
        self.cfg = CONFIG
        self.device = torch.device(self.cfg["training"]["device"])
        self.dtype = torch.float32  # Stability for M4 Pro

        model_name = self.cfg['model_type']
        has_neg = "True" if self.cfg['data'].get('include_negative_examples') else "False"
        self.exp_dir = f"results1/{model_name}_neg_{has_neg}"
        os.makedirs(self.exp_dir, exist_ok=True)

        self._setup_logging()

        # Metrics
        self.bleu_metric = evaluate.load("sacrebleu")
        self.bert_metric = evaluate.load("bertscore")

    def _setup_logging(self):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        if logger.hasHandlers(): logger.handlers.clear()
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        file_handler = logging.FileHandler(f"{self.exp_dir}/train.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        stream_handler = logging.StreamHandler();
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    def _collate(self, batch):
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "target_ids": torch.stack([b["target_ids"] for b in batch])
        }

    def train(self):
        logging.info(f"🚀 RUN START | Arch: {self.cfg['model_type']} | Self-Conditioning: ON")

        tokenizer = SanskritTokenizer(self.cfg["model"]["vocab_size"])
        dataset = OptimizedSanskritDataset("train", tokenizer, self.cfg["model"]["max_seq_len"])
        size = min(self.cfg["data"]["dataset_size"], len(dataset))

        train_idx, val_idx = train_test_split(list(range(size)), train_size=0.8, random_state=42)
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=self.cfg["training"]["batch_size"],
                                  shuffle=True, collate_fn=self._collate)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=self.cfg["training"]["batch_size"], shuffle=False,
                                collate_fn=self._collate)

        model = SanskritModel(self.cfg).to(self.device, dtype=self.dtype)

        # REGULARIZATION & OPTIMIZER
        target_lr = 4e-4
        # optimizer = torch.optim.AdamW(model.parameters(), lr=target_lr, weight_decay=0.01)
        #
        # accum_steps = 4
        # updates_per_epoch = max(1, len(train_loader) // accum_steps)
        # scheduler = OneCycleLR(optimizer, max_lr=target_lr, epochs=self.cfg["training"]["epochs"],
        #                        steps_per_epoch=updates_per_epoch)
        accum_steps = 4
        # Use total steps across the entire training duration for precision
        total_steps = (len(train_loader) // accum_steps) * self.cfg["training"]["epochs"]

        optimizer = torch.optim.AdamW(model.parameters(), lr=target_lr, weight_decay=0.01)

        scheduler = OneCycleLR(
            optimizer,
            max_lr=target_lr,
            total_steps=total_steps,  # Let the scheduler manage the full timeline
            pct_start=0.3,  # Spend 20% of time warming up
            div_factor=25,  # Start at lr = max_lr / 25
            final_div_factor=1e4  # End at lr = max_lr / 10000 (critical for precision)
        )

        pad_id = 1
        best_val_loss = float('inf')
        patience = 3
        counter = 0
        l1_lambda = 1e-5

        for epoch in range(self.cfg["training"]["epochs"]):
            model.train()
            train_loss = 0.0
            optimizer.zero_grad()

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1} Train")
            for i, batch in enumerate(pbar):
                input_ids, target_ids = batch["input_ids"].to(self.device), batch["target_ids"].to(self.device)
                t = model.model.scheduler.sample_timestep(input_ids.size(0)).to(self.device)

                # 🔥 SELF-CONDITIONING: 50% of the time, provide previous prediction as a hint
                prev_x0 = None
                if random.random() < 0.5:
                    with torch.no_grad():
                        # Get a quick "hint" prediction
                        prev_logits = model(input_ids, target_ids, t)[0]
                        prev_x0 = torch.argmax(prev_logits, dim=-1)

                # FORWARD PASS
                logits = model(input_ids, target_ids, t, x0_hint=prev_x0)[0]

                # LOSS CALCULATION
                ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1), ignore_index=pad_id,
                                         label_smoothing=CONFIG['training']['label_smoothing'], reduction='none')
                ce_loss = ce_loss.view(input_ids.size(0), -1).mean(dim=1)

                # 🔥 TIME-WEIGHTED LOSS: Pay more attention to the final denoising steps
                # When t is small (near 0), weights are closer to 1.0
                weights = 1.0 - (t.float() / self.cfg['model']['diffusion_steps']) * 0.4
                loss = (ce_loss * weights).mean()

                # ADD L1 REGULARIZATION
                l1_reg = sum(p.abs().sum() for p in model.parameters())
                total_loss = (loss + l1_lambda * l1_reg) / accum_steps

                total_loss.backward()

                if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                train_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

            # --- Validation & Live Scoring ---
            model.eval()
            val_loss, val_preds, val_targets = 0.0, [], []
            with torch.no_grad():
                # for j, batch in enumerate(tqdm(val_loader, desc="Validating")):
                #     input_ids, target_ids = batch["input_ids"].to(self.device), batch["target_ids"].to(self.device)
                #     t = model.model.scheduler.sample_timestep(input_ids.size(0)).to(self.device)
                #     logits = model(input_ids, target_ids, t)[0]
                #     v_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1), ignore_index=pad_id)
                #     val_loss += v_loss.item()
                #
                #     if j < 2:  # Check BERTScore on first 64 samples
                #         gen_ids = model.model.generate(input_ids, beam_width=3)
                #         # val_preds.extend(tokenizer.decode(gen_ids.tolist()))
                #         val_preds.extend([tokenizer.decode(ids) for ids in gen_ids.tolist()])
                #         val_targets.extend(tokenizer.decode(target_ids.tolist()))
                for j, batch in enumerate(tqdm(val_loader, desc="Validating")):
                    input_ids, target_ids = batch["input_ids"].to(self.device), batch["target_ids"].to(self.device)
                    t = model.model.scheduler.sample_timestep(input_ids.size(0)).to(self.device)
                    logits = model(input_ids, target_ids, t)[0]
                    v_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1), ignore_index=pad_id)
                    val_loss += v_loss.item()

                    if j < 2:  # Check BERTScore on first 64 samples
                        gen_ids = model.model.generate(input_ids, beam_width=3)
                        # val_preds.extend(tokenizer.decode(gen_ids.tolist()))
                        val_preds.extend([tokenizer.decode(ids) for ids in gen_ids.tolist()])
                        # val_targets.extend(tokenizer.decode(target_ids.tolist()))
                        val_targets.extend([tokenizer.decode(ids) for ids in target_ids.tolist()])


            avg_val_loss = val_loss / len(val_loader)
            bleu = self.bleu_metric.compute(predictions=val_preds, references=[[t] for t in val_targets])['score']
            bert = self.bert_metric.compute(predictions=val_preds, references=val_targets, lang="sa")
            avg_bert = sum(bert['f1']) / len(bert['f1'])

            logging.info(
                f"📊 Epoch {epoch + 1} | Loss: {avg_val_loss:.4f} | BLEU: {bleu:.2f} | BERTScore: {avg_bert:.4f}")

            # EARLY STOPPING
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), f"{self.exp_dir}/best_model.pt")
                logging.info(f"🎉 New Best Model Saved (BERTScore Improved)!")
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    logging.info("🛑 Early Stopping triggered. Training halted.")
                    break

        logging.info("✅ Training Complete")


if __name__ == "__main__":
    set_seed(42)
    FinalTrainer().train()