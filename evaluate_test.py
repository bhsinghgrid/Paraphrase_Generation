"""
evaluate_test.py
================
Final held-out test evaluation.
Uses the same run_inference() as inference.py — no divergence.
"""

import torch
import os, sys
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import CONFIG
from inference import load_model, run_inference
from data.dataset import OptimizedSanskritDataset
from model.tokenizer import SanskritTokenizer


class Evaluator:
    def __init__(self):
        self.cfg    = CONFIG
        self.device = torch.device(self.cfg['training']['device'])

        model_name = self.cfg['model_type']
        has_neg    = self.cfg['data']['include_negative_examples']
        self.exp_dir = f"results/{model_name}_neg_{has_neg}"
        self.ckpt    = f"{self.exp_dir}/best_model.pt"

        if not os.path.exists(self.ckpt):
            raise FileNotFoundError(
                f"No checkpoint at {self.ckpt}. Train first."
            )

    def _collate(self, batch):
        return {
            'input_ids':   torch.stack([b['input_ids'].long() for b in batch]),
            'target_text': [b['target_text'] for b in batch],
            'input_text':  [b['input_text']  for b in batch],
        }

    def evaluate(self, sample_size=1000):
        model, cfg = load_model(self.ckpt, self.cfg, self.device)
        model.eval()

        tokenizer = SanskritTokenizer(cfg['model']['vocab_size'])
        PAD_ID    = tokenizer.tokenizer.token_to_id('[PAD]') or 1
        MASK_ID   = cfg['diffusion']['mask_token_id']

        dataset = OptimizedSanskritDataset(
            'test', tokenizer, cfg['model']['max_seq_len'], cfg
        )
        indices = list(range(min(sample_size, len(dataset))))
        loader  = DataLoader(
            Subset(dataset, indices),
            batch_size=cfg['training']['batch_size'],
            shuffle=False, collate_fn=self._collate
        )

        all_preds, all_refs, all_inputs = [], [], []
        print(f"⏳ Evaluating {len(indices)} samples …")

        for batch in tqdm(loader):
            ids = batch['input_ids'].to(self.device)
            out = run_inference(model, ids, cfg)
            for i in range(out.size(0)):
                clean = [x for x in out[i].tolist() if x not in (MASK_ID, PAD_ID)]
                all_preds.append(tokenizer.decode(clean).strip())
                all_refs.append(batch['target_text'][i].strip())
                all_inputs.append(batch['input_text'][i].strip())

        # ── Metrics ───────────────────────────────────────────────────
        bleu_score, bert_f1 = 0.0, 0.0

        try:
            from nltk.translate.bleu_score import corpus_bleu
            bleu_score = corpus_bleu(
                [[r.split()] for r in all_refs],
                [p.split() for p in all_preds]
            )
        except Exception:
            print("⚠️  BLEU unavailable (install nltk)")

        try:
            import evaluate as hf_eval
            print("⏳ Computing BERTScore …")
            res     = hf_eval.load('bertscore').compute(
                predictions=all_preds, references=all_refs, lang='hi'
            )
            bert_f1 = sum(res['f1']) / len(res['f1'])
        except Exception:
            print("⚠️  BERTScore unavailable (install evaluate)")

        # ── Save ──────────────────────────────────────────────────────
        inf = cfg['inference']
        out_path = f"{self.exp_dir}/test_results.txt"
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(f"=== TEST RESULTS ===\n")
            f.write(f"Model      : {cfg['model_type']}\n")
            f.write(f"Negatives  : {cfg['data']['include_negative_examples']}\n")
            f.write(f"Steps      : {inf['num_steps']}\n")
            f.write(f"Temp       : {inf['temperature']}\n")
            f.write(f"RepPen     : {inf['repetition_penalty']}\n")
            f.write(f"DivPen     : {inf['diversity_penalty']}\n")
            f.write(f"BLEU       : {bleu_score:.4f}\n")
            f.write(f"BERTScore  : {bert_f1:.4f}\n\n")
            f.write("=== SAMPLES ===\n")
            for i in range(min(20, len(all_preds))):
                f.write(f"IN  : {all_inputs[i]}\n")
                f.write(f"REF : {all_refs[i]}\n")
                f.write(f"PRED: {all_preds[i]}\n")
                f.write("-" * 60 + "\n")

        print(f"\n✅ Test complete → {out_path}")
        print(f"📊 BLEU: {bleu_score:.4f}  |  BERTScore: {bert_f1:.4f}")
        return bleu_score, bert_f1


if __name__ == '__main__':
    Evaluator().evaluate(sample_size=1000)