"""
train.py  — Cross-Script Edition
==================================
INPUT  : quote_text       (Roman/IAST Sanskrit)
OUTPUT : quote_devanagari (Devanagari script)

Changes from previous version:
  1. Uses SanskritSourceTokenizer + SanskritTargetTokenizer separately
  2. Passes src_vocab_size and tgt_vocab_size so the model builds
     right embedding table sizes for each script independently
  3. Dataset receives both tokenizers
  4. BERTScore decodes predictions with tgt_tokenizer only (Devanagari)
  5. Validation sample printer shows:
        Roman input → Devanagari prediction → Devanagari reference
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os, sys, logging, random, copy, math
import numpy as np

from config import CONFIG

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data.dataset import OptimizedSanskritDataset
from model.tokenizer import SanskritSourceTokenizer, SanskritTargetTokenizer
from model.sanskrit_model import SanskritModel


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay  = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.data.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()


def build_lr_lambda(warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return lr_lambda


class Trainer:
    def __init__(self):
        self.cfg    = CONFIG
        self.device = torch.device(self.cfg['training']['device'])
        model_name  = self.cfg['model_type']
        has_neg     = self.cfg['data']['include_negative_examples']
        self.exp_dir = f"results7/{model_name}_neg_{has_neg}"
        os.makedirs(self.exp_dir, exist_ok=True)
        self._setup_logging()

    def _setup_logging(self):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        if logger.hasHandlers():
            logger.handlers.clear()
        fmt = logging.Formatter("%(asctime)s  %(message)s")
        for h in [logging.FileHandler(f"{self.exp_dir}/train.log"),
                  logging.StreamHandler(sys.stdout)]:
            h.setFormatter(fmt)
            logger.addHandler(h)

    def _collate(self, batch):
        out = {
            'input_ids':   torch.stack([b['input_ids']  for b in batch]),
            'target_ids':  torch.stack([b['target_ids'] for b in batch]),
            'input_text':  [b['input_text']  for b in batch],
            'target_text': [b['target_text'] for b in batch],
        }
        if self.cfg['data']['include_negative_examples'] and 'negative_target_ids' in batch[0]:
            out['negative_target_ids'] = torch.stack(
                [b['negative_target_ids'] for b in batch]
            )
        return out

    def _masked_ce_loss(self, logits, target_ids, x_t_ids, pad_id, label_smoothing):
        """CE only on positions actually masked by q_sample."""
        mask_id    = self.cfg['diffusion']['mask_token_id']
        was_masked = (x_t_ids == mask_id) & (target_ids != pad_id)
        if was_masked.sum() == 0:
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                ignore_index=pad_id,
                label_smoothing=label_smoothing,
            )
        return F.cross_entropy(
            logits[was_masked], target_ids[was_masked],
            label_smoothing=label_smoothing,
        )

    def _negative_loss(self, model, input_ids, neg_ids, t, margin=2.0):
        with torch.no_grad():
            neg_logits, _ = model(input_ids, neg_ids, t)
        neg_ce = F.cross_entropy(
            neg_logits.view(-1, neg_logits.size(-1)),
            neg_ids.view(-1), ignore_index=1, reduction='mean',
        )
        return torch.clamp(margin - neg_ce, min=0.0) * 0.1

    def _print_val_samples(self, model, val_batch, tgt_tokenizer, n=2):
        """Print n samples: Roman INPUT → Devanagari PRED → Devanagari REF."""
        inf_cfg = self.cfg['inference']
        mask_id = self.cfg['diffusion']['mask_token_id']
        pad_id  = 1

        input_ids  = val_batch['input_ids'][:n].to(self.device)
        target_ids = val_batch['target_ids'][:n].to(self.device)

        model.eval()
        with torch.no_grad():
            gen_ids = model.generate(
                input_ids,
                num_steps          = inf_cfg['num_steps'],
                temperature        = inf_cfg['temperature'],
                top_k              = inf_cfg['top_k'],
                repetition_penalty = inf_cfg['repetition_penalty'],
                diversity_penalty  = inf_cfg['diversity_penalty'],
            )

        logging.info("─" * 72)
        logging.info("  📝  VALIDATION SAMPLES   (Roman Sanskrit → Devanagari)")
        for i in range(n):
            src_text  = val_batch['input_text'][i].strip()
            pred_ids  = [x for x in gen_ids[i].tolist()    if x > 4]
            ref_ids   = [x for x in target_ids[i].tolist() if x > 4]
            pred_text = tgt_tokenizer.decode(pred_ids).strip()
            ref_text  = tgt_tokenizer.decode(ref_ids).strip()

            logging.info(f"  [{i+1}] INPUT (Roman)      : {src_text[:100]}")
            logging.info(f"       REF  (Devanagari)  : {ref_text[:100]}")
            logging.info(f"       PRED (Devanagari)  : {pred_text[:100]}")
        logging.info("─" * 72)

    # ── Main train loop ───────────────────────────────────────────────

    def train(self):
        cfg    = self.cfg
        pad_id = 1

        # ── Dual tokenizers ───────────────────────────────────────────
        src_vocab = cfg['model'].get('src_vocab_size', 8000)
        tgt_vocab = cfg['model'].get('tgt_vocab_size', 8000)

        src_tokenizer = SanskritSourceTokenizer(
            vocab_size=src_vocab, max_len=cfg['model']['max_seq_len'])
        tgt_tokenizer = SanskritTargetTokenizer(
            vocab_size=tgt_vocab, max_len=cfg['model']['max_seq_len'])

        # Inject into config so model builds correct embedding sizes
        cfg['model']['src_vocab_size'] = src_vocab
        cfg['model']['tgt_vocab_size'] = tgt_vocab
        cfg['model']['vocab_size']     = tgt_vocab   # output head = Devanagari vocab

        logging.info(
            f"🚀  model={cfg['model_type']}  T={cfg['model']['diffusion_steps']}  "
            f"task=quote_text→quote_devanagari  "
            f"src_vocab={src_vocab}  tgt_vocab={tgt_vocab}"
        )

        # ── Dataset ───────────────────────────────────────────────────
        dataset = OptimizedSanskritDataset(
            'train',
            max_len       = cfg['model']['max_seq_len'],
            cfg           = cfg,
            src_tokenizer = src_tokenizer,
            tgt_tokenizer = tgt_tokenizer,
        )
        size = min(cfg['data']['dataset_size'], len(dataset))
        train_idx, val_idx = train_test_split(
            list(range(size)), train_size=0.8, random_state=42
        )
        train_loader = DataLoader(
            Subset(dataset, train_idx),
            batch_size=cfg['training']['batch_size'],
            shuffle=True, collate_fn=self._collate, num_workers=0,
        )
        val_loader = DataLoader(
            Subset(dataset, val_idx),
            batch_size=cfg['training']['batch_size'],
            shuffle=False, collate_fn=self._collate, num_workers=0,
        )

        # ── Model ─────────────────────────────────────────────────────
        model = SanskritModel(cfg).to(self.device)
        ema   = ModelEMA(model, decay=0.999)
        inner = model.model

        accum           = cfg['training']['accum_steps']
        target_lr       = cfg['training']['lr']
        epochs          = cfg['training']['epochs']
        steps_per_epoch = max(1, len(train_loader) // accum)
        total_steps     = steps_per_epoch * epochs
        warmup_steps    = max(1, int(0.06 * total_steps))

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=target_lr,
            betas=(0.9, 0.98), weight_decay=0.01, eps=1e-8,
        )
        lr_sched = torch.optim.lr_scheduler.LambdaLR(
            optimizer, build_lr_lambda(warmup_steps, total_steps)
        )

        try:
            import evaluate as hf_eval
            bert_metric = hf_eval.load('bertscore')
            USE_BERT    = True
        except Exception:
            USE_BERT    = False
            logging.warning("BERTScore unavailable — using -val_loss.")

        best_metric     = -float('inf')
        patience        = cfg['training']['patience']
        counter         = 0
        T_steps         = cfg['model']['diffusion_steps']
        l1_lambda       = cfg['training']['l1_lambda']
        fixed_val_batch = next(iter(val_loader))

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            optimizer.zero_grad()

            pbar = tqdm(train_loader, desc=f"Ep {epoch+1:02d} train", dynamic_ncols=True)
            for i, batch in enumerate(pbar):
                input_ids  = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                B          = input_ids.size(0)
                t          = inner.scheduler.sample_timestep(B).to(self.device)

                prev_x0 = None
                if random.random() < 0.75:
                    with torch.no_grad():
                        prev_logits, _ = model(input_ids, target_ids, t)
                        prev_x0 = torch.argmax(prev_logits, dim=-1)

                logits, _ = model(input_ids, target_ids, t, x0_hint=prev_x0)

                with torch.no_grad():
                    _, x_t_ids = inner.forward_process.q_sample(target_ids, t)

                ce      = self._masked_ce_loss(logits, target_ids, x_t_ids,
                                               pad_id, cfg['training']['label_smoothing'])
                weights = 1.0 + (t.float() / T_steps).sqrt() * 0.5
                loss    = ce * weights.mean()

                if cfg['data']['include_negative_examples'] and 'negative_target_ids' in batch:
                    loss = loss + self._negative_loss(
                        model, input_ids, batch['negative_target_ids'].to(self.device), t)

                l1         = sum(p.abs().sum() for p in model.parameters())
                total_loss = (loss + l1_lambda * l1) / accum
                total_loss.backward()

                if (i + 1) % accum == 0 or (i + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_sched.step()
                    optimizer.zero_grad()
                    ema.update(model)

                train_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}",
                                 lr=f"{lr_sched.get_last_lr()[0]:.2e}")

            # ── Validation ────────────────────────────────────────────
            ema.shadow.eval()
            val_loss = 0.0
            val_preds, val_refs = [], []

            with torch.no_grad():
                for j, batch in enumerate(tqdm(val_loader, desc=f"Ep {epoch+1:02d} val",
                                               dynamic_ncols=True)):
                    input_ids  = batch['input_ids'].to(self.device)
                    target_ids = batch['target_ids'].to(self.device)
                    t = inner.scheduler.sample_timestep(input_ids.size(0)).to(self.device)

                    logits, _ = ema.shadow(input_ids, target_ids, t)
                    val_loss += F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        target_ids.view(-1), ignore_index=pad_id,
                    ).item()

                    if j < 3:
                        inf_cfg = cfg['inference']
                        gen_ids = ema.shadow.generate(
                            input_ids,
                            num_steps          = inf_cfg['num_steps'],
                            temperature        = inf_cfg['temperature'],
                            top_k              = inf_cfg['top_k'],
                            repetition_penalty = inf_cfg['repetition_penalty'],
                            diversity_penalty  = inf_cfg['diversity_penalty'],
                        )
                        # Decode with TARGET tokenizer (Devanagari only)
                        for ids in gen_ids.tolist():
                            val_preds.append(tgt_tokenizer.decode([x for x in ids if x > 4]))
                        for ids in target_ids.tolist():
                            val_refs.append(tgt_tokenizer.decode([x for x in ids if x > 4]))

            avg_val  = val_loss / len(val_loader)
            avg_train = train_loss / len(train_loader)

            bert_f1 = 0.0
            if USE_BERT and val_preds:
                try:
                    res     = bert_metric.compute(
                        predictions=val_preds, references=val_refs, lang='hi'
                    )
                    bert_f1 = sum(res['f1']) / len(res['f1'])
                except Exception as e:
                    logging.warning(f"BERTScore error: {e}")

            logging.info(
                f"Ep {epoch+1:02d}  train={avg_train:.4f}  val={avg_val:.4f}  "
                f"bert={bert_f1:.4f}  lr={lr_sched.get_last_lr()[0]:.2e}"
            )

            # Print 2 val samples every epoch
            self._print_val_samples(ema.shadow, fixed_val_batch, tgt_tokenizer, n=2)

            metric = bert_f1 if USE_BERT else -avg_val
            if metric > best_metric:
                best_metric = metric
                torch.save(ema.state_dict(), f"{self.exp_dir}/best_model.pt")
                logging.info(f"   ✅ New best: {metric:.4f}")
                counter = 0
            else:
                counter += 1
                logging.info(f"   No improvement {counter}/{patience}")
                if counter >= patience:
                    logging.info("🛑 Early stopping.")
                    break

        logging.info("✅ Training complete.")
        logging.info(f"   Best model → {self.exp_dir}/best_model.pt")


if __name__ == '__main__':
    set_seed(42)
    Trainer().train()