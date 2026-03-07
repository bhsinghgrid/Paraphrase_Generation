"""
dataset.py  — Cross-Script Translation Fix
==========================================
INPUT  : quote_text       (Roman/IAST transliteration of Sanskrit)
TARGET : quote_devanagari (Devanagari script)

This is the CORRECT task: the model learns to transliterate / translate
Roman Sanskrit → Devanagari, which is a meaningful, learnable mapping
(far better than devanagari→devanagari reconstruction which teaches nothing).

KEY CHANGES from original:
  1. _input_field  = 'quote_text'        (was 'quote_devanagari')
  2. _target_field = 'quote_devanagari'  (unchanged)
  3. Separate source/target tokenizers — Roman and Devanagari have
     completely different character sets; a shared BPE vocab forces the
     model to learn both scripts in one embedding table, which wastes
     capacity and confuses the attention mechanism.
  4. Negative example generation fixed — reversal now operates on
     DEVANAGARI target only (not accidentally on Roman source).
  5. curriculum_sort uses target length (Devanagari) for difficulty proxy.
"""

from datasets import load_dataset
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import random


class OptimizedSanskritDataset(Dataset):
    def __init__(self, split='train', tokenizer=None, max_len=80, cfg=None,
                 src_tokenizer=None, tgt_tokenizer=None):
        """
        Args:
            tokenizer     : shared tokenizer (legacy — used if src/tgt not provided)
            src_tokenizer : tokenizer for quote_text  (Roman script)
            tgt_tokenizer : tokenizer for quote_devanagari (Devanagari script)
                            If None, falls back to shared `tokenizer`.
        """
        from config import CONFIG
        self.cfg = cfg or CONFIG
        self.max_len = max_len
        self.pad_id  = 1
        self.mask_id = self.cfg['diffusion']['mask_token_id']
        self.include_negatives = self.cfg['data']['include_negative_examples']

        # ── Tokenizer setup ───────────────────────────────────────────
        # Support both legacy (shared) and new (separate src/tgt) tokenizers
        self.src_tokenizer = src_tokenizer or tokenizer
        self.tgt_tokenizer = tgt_tokenizer or tokenizer

        if self.src_tokenizer is None:
            raise ValueError("Provide at least one tokenizer.")

        print(f"📥 Loading '{split}' split …")
        raw = load_dataset("paws/sanskrit-verses-gretil", split=split)
        cols = raw.column_names

        # ── Field selection ───────────────────────────────────────────
        if 'quote_text' in cols and 'quote_devanagari' in cols:
            # CORRECT setup: Roman input → Devanagari output
            self._input_field  = 'quote_text'
            self._target_field = 'quote_devanagari'
            print("   Format: quote_text (Roman) → quote_devanagari (Devanagari) ✓")
        elif 'sentence1' in cols and 'sentence2' in cols:
            # PAWS paraphrase pairs fallback
            self._input_field  = 'sentence1'
            self._target_field = 'sentence2'
            print("   Format: PAWS sentence pairs ✓")
        else:
            # Last resort: same field both sides
            self._input_field  = 'quote_devanagari'
            self._target_field = 'quote_devanagari'
            print("   ⚠️  Format: Devanagari→Devanagari (suboptimal — no quote_text found)")

        # ── Filter empty rows ─────────────────────────────────────────
        # Some rows have empty quote_text — skip them
        raw = raw.filter(
            lambda ex: (
                bool(ex[self._input_field].strip()) and
                bool(ex[self._target_field].strip())
            )
        )
        print(f"   After empty-filter: {len(raw)} samples")

        self.dataset = raw

        if split == 'train':
            self.dataset = self._curriculum_sort()

        print(f"✅ {len(self.dataset)} samples loaded.")

    # ── Encoding ──────────────────────────────────────────────────────

    def _encode_src(self, text):
        """Encode source (Roman) text."""
        ids = self.src_tokenizer.encode(text)[:self.max_len]
        t   = torch.tensor(ids, dtype=torch.long)
        t   = F.pad(t, (0, max(0, self.max_len - len(t))), value=self.pad_id)
        return t

    def _encode_tgt(self, text):
        """Encode target (Devanagari) text."""
        ids = self.tgt_tokenizer.encode(text)[:self.max_len]
        t   = torch.tensor(ids, dtype=torch.long)
        t   = F.pad(t, (0, max(0, self.max_len - len(t))), value=self.pad_id)
        return t

    # ── Curriculum ────────────────────────────────────────────────────

    def _curriculum_sort(self):
        """Short, common Devanagari targets first → long, rare targets last."""
        scores = []
        for s in self.dataset:
            text         = s[self._target_field]
            length       = len(text.split())
            rarity_score = len(set(text)) / max(1, len(text))
            scores.append(length * (1 - rarity_score))
        order = sorted(range(len(self.dataset)), key=lambda i: scores[i])
        return self.dataset.select(order)

    # ── Item ──────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        src_text = sample[self._input_field].strip()
        tgt_text = sample[self._target_field].strip()

        input_ids  = self._encode_src(src_text)   # Roman encoded with src_tokenizer
        target_ids = self._encode_tgt(tgt_text)   # Devanagari encoded with tgt_tokenizer

        out = {
            'input_ids':   input_ids,
            'target_ids':  target_ids,
            'input_text':  src_text,
            'target_text': tgt_text,
        }

        if self.include_negatives:
            neg_ids = target_ids.clone()
            # Reverse a random chunk of the DEVANAGARI target
            non_pad = (neg_ids != self.pad_id).sum().item()
            if non_pad > 4:
                i1, i2 = sorted(random.sample(range(non_pad), 2))
                neg_ids[i1:i2] = torch.flip(neg_ids[i1:i2], dims=[0])
            out['negative_target_ids'] = neg_ids

        return out