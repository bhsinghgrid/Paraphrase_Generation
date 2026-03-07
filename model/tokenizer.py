"""
tokenizer.py  — Dual Tokenizer Fix
====================================
Two separate BPE tokenizers:

  SanskritSourceTokenizer  — trained on quote_text (Roman/IAST script)
  SanskritTargetTokenizer  — trained on quote_devanagari (Devanagari script)

WHY SEPARATE?
  Roman Sanskrit and Devanagari are fundamentally different character sets.
  Roman uses a-z + diacritics (~60 unique chars), Devanagari uses ā-ह + matras
  (~100+ unique chars). A shared BPE tokenizer wastes half its vocab on
  character combos that never cross scripts, and forces the embedding table
  to encode both scripts in one space — confusing the model's cross-attention.

  With separate tokenizers:
  - src vocab captures Roman subwords cleanly (ā, ś, ṭ, ṃ etc.)
  - tgt vocab captures Devanagari akshara clusters cleanly (क्ष, त्र, etc.)
  - The model learns a true cross-script mapping in its cross-attention

SPECIAL TOKENS (same IDs in both):
  [MASK] = 0   ← required by absorbing diffusion
  [PAD]  = 1
  [UNK]  = 2
  [CLS]  = 3
  [SEP]  = 4
"""

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from datasets import load_dataset
from pathlib import Path


SPECIAL_TOKENS = ["[MASK]", "[PAD]", "[UNK]", "[CLS]", "[SEP]"]


def _build_bpe(texts, vocab_size):
    """Build a BPE tokenizer from an iterator of strings."""
    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,   # [MASK] MUST be first → id=0
        min_frequency=2,
    )
    tok.train_from_iterator(texts, trainer)
    return tok


def _validate(tok, name):
    mask_id = tok.token_to_id("[MASK]")
    pad_id  = tok.token_to_id("[PAD]")
    assert mask_id == 0, f"{name}: [MASK] must be id=0, got {mask_id}"
    assert pad_id  == 1, f"{name}: [PAD] must be id=1, got {pad_id}"
    print(f"✅ {name}: [MASK]=0, [PAD]=1 confirmed. Vocab size={tok.get_vocab_size()}")


# ── Source tokenizer (Roman/IAST Sanskrit) ────────────────────────────

class SanskritSourceTokenizer:
    """
    Tokenizer for quote_text — Roman transliteration of Sanskrit.
    Examples: "dharmo rakṣati rakṣitaḥ", "yatra nāryastu pūjyante"
    """
    MODEL_PATH = "sanskrit_src_tokenizer.json"

    def __init__(self, vocab_size=8000, max_len=80, n_train_samples=50000):
        self.vocab_size = vocab_size
        self.max_len    = max_len
        self.mask_token_id = 0

        if Path(self.MODEL_PATH).exists():
            print(f"📖 Loading source tokenizer from {self.MODEL_PATH} …")
            self.tokenizer = Tokenizer.from_file(self.MODEL_PATH)
        else:
            print("🎓 Training source tokenizer on quote_text …")
            self._train(vocab_size, n_train_samples)

        _validate(self.tokenizer, "SrcTokenizer")

    def _train(self, vocab_size, n_samples):
        dataset = load_dataset("paws/sanskrit-verses-gretil", split="train")
        n = min(n_samples, len(dataset))
        texts = [s["quote_text"] for s in dataset.select(range(n))
                 if s["quote_text"].strip()]
        self.tokenizer = _build_bpe(texts, vocab_size)
        self.tokenizer.save(self.MODEL_PATH)
        print(f"✅ Source tokenizer trained on {len(texts)} Roman texts.")

    def encode(self, text):
        ids = self.tokenizer.encode(text).ids[:self.max_len]
        pad = self.tokenizer.token_to_id("[PAD]")
        ids += [pad] * max(0, self.max_len - len(ids))
        return ids[:self.max_len]

    def decode(self, ids):
        clean = [i for i in ids if i > 4]   # skip special tokens
        return self.tokenizer.decode(clean)

    def __len__(self):
        return self.vocab_size


# ── Target tokenizer (Devanagari Sanskrit) ───────────────────────────

class SanskritTargetTokenizer:
    """
    Tokenizer for quote_devanagari — Devanagari script.
    Examples: "धर्मो रक्षति रक्षितः", "यत्र नार्यस्तु पूज्यन्ते"
    """
    MODEL_PATH = "sanskrit_tgt_tokenizer.json"

    def __init__(self, vocab_size=8000, max_len=80, n_train_samples=50000):
        self.vocab_size = vocab_size
        self.max_len    = max_len
        self.mask_token_id = 0

        if Path(self.MODEL_PATH).exists():
            print(f"📖 Loading target tokenizer from {self.MODEL_PATH} …")
            self.tokenizer = Tokenizer.from_file(self.MODEL_PATH)
        else:
            print("🎓 Training target tokenizer on quote_devanagari …")
            self._train(vocab_size, n_train_samples)

        _validate(self.tokenizer, "TgtTokenizer")

    def _train(self, vocab_size, n_samples):
        dataset = load_dataset("paws/sanskrit-verses-gretil", split="train")
        n = min(n_samples, len(dataset))
        texts = [s["quote_devanagari"] for s in dataset.select(range(n))
                 if s["quote_devanagari"].strip()]
        self.tokenizer = _build_bpe(texts, vocab_size)
        self.tokenizer.save(self.MODEL_PATH)
        print(f"✅ Target tokenizer trained on {len(texts)} Devanagari texts.")

    def encode(self, text):
        ids = self.tokenizer.encode(text).ids[:self.max_len]
        pad = self.tokenizer.token_to_id("[PAD]")
        ids += [pad] * max(0, self.max_len - len(ids))
        return ids[:self.max_len]

    def decode(self, ids):
        clean = [i for i in ids if i > 4]
        return self.tokenizer.decode(clean)

    # Methods required by BERTScore
    def build_inputs_with_special_tokens(self, token_ids):
        return list(token_ids)

    def get_vocab(self):
        return {str(i): i for i in range(self.vocab_size)}

    def convert_ids_to_tokens(self, ids):
        return [str(i) for i in ids]

    def __len__(self):
        return self.vocab_size


# ── Legacy shared tokenizer (kept for backward compat) ───────────────

class SanskritTokenizer:
    """
    LEGACY: single shared tokenizer trained on BOTH scripts.
    Still works but suboptimal — use SanskritSourceTokenizer +
    SanskritTargetTokenizer for the quote_text → quote_devanagari task.
    """
    MODEL_PATH = "sanskrit_tokenizer_m4pro.json"

    def __init__(self, vocab_size=16000, max_len=80):
        self.vocab_size    = vocab_size
        self.max_len       = max_len
        self.mask_token_id = 0

        if Path(self.MODEL_PATH).exists():
            print("📖 Loading shared tokenizer …")
            self.tokenizer = Tokenizer.from_file(self.MODEL_PATH)
        else:
            print("🎓 Training shared tokenizer on both scripts …")
            self._train(vocab_size)

        _validate(self.tokenizer, "SharedTokenizer")

    def _train(self, vocab_size):
        dataset = load_dataset("paws/sanskrit-verses-gretil", split="train")
        n = min(50000, len(dataset))
        texts = []
        for s in dataset.select(range(n)):
            if s["quote_text"].strip():
                texts.append(s["quote_text"])
            if s["quote_devanagari"].strip():
                texts.append(s["quote_devanagari"])
        self.tokenizer = _build_bpe(texts, vocab_size)
        self.tokenizer.save(self.MODEL_PATH)
        print(f"✅ Shared tokenizer trained ({len(texts)} texts).")

    def encode(self, text):
        ids = self.tokenizer.encode(text).ids[:self.max_len]
        pad = self.tokenizer.token_to_id("[PAD]")
        ids += [pad] * max(0, self.max_len - len(ids))
        return ids[:self.max_len]

    def decode(self, ids):
        if ids and isinstance(ids[0], list):
            raise TypeError("decode() got 2D list — pass a 1D list.")
        clean = [i for i in ids if i > 4]
        return self.tokenizer.decode(clean)

    def build_inputs_with_special_tokens(self, token_ids):
        return list(token_ids)

    def get_vocab(self):
        return {str(i): i for i in range(self.vocab_size)}

    def convert_ids_to_tokens(self, ids):
        return [str(i) for i in ids]

    def __len__(self):
        return self.vocab_size