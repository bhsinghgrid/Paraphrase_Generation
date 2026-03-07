"""
char_tokenizer.py
─────────────────────────────────────────────────────────────────
Drop-in replacement for your BPE sanskrit_src/tgt_tokenizer.json.

Builds deterministic CHARACTER-LEVEL vocabularies for:
  SRC : Roman-script Sanskrit  (ISO 15919 / IAST diacritics)
  TGT : Devanagari Sanskrit

Special tokens match your existing convention:
  [MASK] = 0
  [PAD]  = 1
  [BOS]  = 2
  [EOS]  = 3
  [UNK]  = 4

Usage
─────
from char_tokenizer import CharTokenizer

src_tok = CharTokenizer.build_or_load("char_src_tokenizer.json", script="roman")
tgt_tok = CharTokenizer.build_or_load("char_tgt_tokenizer.json", script="devanagari")

# Encode (returns List[int])
ids = src_tok.encode("baler atulavīryasya siṃhāsanagatasya vai //")

# Decode (returns str)
text = tgt_tok.decode(ids)

# Batch encode with padding (returns torch.Tensor [B, T])
tensor = src_tok.batch_encode(sentences, max_len=128)
"""

import json
import re
from pathlib import Path
from typing import List, Optional

import torch


# ──────────────────────────────────────────────────────────────
# 1.  CHARACTER INVENTORIES
#     These cover virtually every char in IAST/ISO-15919 Roman
#     Sanskrit and standard printed Devanagari.
#     The tokenizer is also DATA-DRIVEN: any char seen in your
#     corpus that is NOT in these lists gets added automatically.
# ──────────────────────────────────────────────────────────────

# Roman Sanskrit: ASCII + full IAST diacritic set
ROMAN_BASE_CHARS = list(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " \t\n"
    ".,;:!?/\\|()[]{}\"'`~@#$%^&*-_+=<>"
    # IAST / ISO-15919 diacritics (precomposed)
    "āĀīĪūŪṛṚṝṜḷḸ"           # vowels with macron / dot-below
    "ṃṁḥ"                      # anusvara / visarga
    "śŚṣṢṭṬḍḌṇṆḻḺ"            # retroflex & palatal sibilants
    "ñÑ"
    # combining diacritics (decomposed forms) — rare but present
    "\u0304\u0331\u0323\u0325\u0307\u0308"
)

# Devanagari Unicode block: U+0900–U+097F  +  Vedic extensions U+1CD0–U+1CFF
# We enumerate every codepoint rather than listing glyphs
DEVANAGARI_BASE_CHARS = (
    [chr(c) for c in range(0x0900, 0x0980)]   # core Devanagari block
    + [chr(c) for c in range(0x1CD0, 0x1D00)]  # Vedic extensions
    + [chr(c) for c in range(0xA8E0, 0xA900)]  # Devanagari Extended
    + list(" \t\n.,;:!?/|()[]{}\"'`~0123456789")  # shared punctuation
    # Special Vedic / manuscript symbols that appear in GRETIL
    + list("।॥ॐ₹")
    + ["\u0952", "\u0951", "\u1CDA"]           # Vedic accents
)


SPECIAL_TOKENS = {
    "[MASK]": 0,
    "[PAD]":  1,
    "[BOS]":  2,
    "[EOS]":  3,
    "[UNK]":  4,
}


class CharTokenizer:
    """
    Lightweight character-level tokenizer that mirrors the HuggingFace
    tokenizer interface your training loop already uses.
    """

    def __init__(self, vocab: dict[str, int]):
        self.token2id: dict[str, int] = vocab
        self.id2token: dict[int, str] = {v: k for k, v in vocab.items()}

        self.mask_token_id = vocab["[MASK]"]
        self.pad_token_id  = vocab["[PAD]"]
        self.bos_token_id  = vocab["[BOS]"]
        self.eos_token_id  = vocab["[EOS]"]
        self.unk_token_id  = vocab["[UNK]"]

    # ── core API ──────────────────────────────────────────────

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        max_length: Optional[int] = None,
    ) -> List[int]:
        """
        Encode a string to a list of token ids.
        Each Unicode character → one id.
        Devanagari: virama sequences are kept as separate chars
        so the model learns the exact composition rules.
        """
        ids = [self.token2id.get(ch, self.unk_token_id) for ch in text]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        if max_length is not None:
            ids = ids[:max_length]
        return ids

    def decode(
        self,
        ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        special = set(SPECIAL_TOKENS.values()) if skip_special_tokens else set()
        return "".join(
            self.id2token.get(i, "?")
            for i in ids
            if i not in special
        )

    def batch_encode(
        self,
        texts: List[str],
        max_len: int = 128,
        add_special_tokens: bool = False,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Returns LongTensor [B, max_len] padded with pad_token_id.
        """
        encoded = [
            self.encode(t, add_special_tokens=add_special_tokens, max_length=max_len)
            for t in texts
        ]
        out = torch.full(
            (len(encoded), max_len),
            fill_value=self.pad_token_id,
            dtype=torch.long,
        )
        for i, ids in enumerate(encoded):
            out[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        if device is not None:
            out = out.to(device)
        return out

    def __len__(self) -> int:
        return len(self.token2id)

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

    # ── build / save / load ───────────────────────────────────

    @classmethod
    def build_from_corpus(
        cls,
        texts: List[str],
        script: str = "roman",
        save_path: Optional[str] = None,
    ) -> "CharTokenizer":
        """
        Build vocab from a list of strings.
        script: "roman"  → seeds with ROMAN_BASE_CHARS
                "devanagari" → seeds with DEVANAGARI_BASE_CHARS
        """
        base = ROMAN_BASE_CHARS if script == "roman" else DEVANAGARI_BASE_CHARS

        # Start with special tokens at fixed positions
        vocab: dict[str, int] = dict(SPECIAL_TOKENS)
        next_id = max(SPECIAL_TOKENS.values()) + 1  # = 5

        # Add base chars
        for ch in base:
            if ch not in vocab:
                vocab[ch] = next_id
                next_id += 1

        # Add any chars seen in corpus that aren't already present
        seen = sorted(set("".join(texts)))
        for ch in seen:
            if ch not in vocab:
                vocab[ch] = next_id
                next_id += 1

        tok = cls(vocab)
        if save_path:
            tok.save(save_path)
        return tok

    @classmethod
    def build_or_load(
        cls,
        path: str,
        script: str = "roman",
        corpus_texts: Optional[List[str]] = None,
    ) -> "CharTokenizer":
        """
        Load from JSON if it exists, otherwise build from corpus_texts.
        """
        p = Path(path)
        if p.exists():
            return cls.load(path)
        if corpus_texts is None:
            raise ValueError(f"Tokenizer file {path} not found and no corpus_texts provided.")
        tok = cls.build_from_corpus(corpus_texts, script=script, save_path=path)
        print(f"✅ Built {script} tokenizer: vocab={len(tok)}  saved → {path}")
        return tok

    def save(self, path: str) -> None:
        Path(path).write_text(
            json.dumps({"vocab": self.token2id}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls(data["vocab"])
        print(
            f"📖 Loaded tokenizer from {path}  "
            f"vocab_size={len(tok)}  "
            f"[MASK]={tok.mask_token_id}  [PAD]={tok.pad_token_id}"
        )
        return tok


# ──────────────────────────────────────────────────────────────
# 2.  DEVANAGARI-SPECIFIC UTILITIES
# ──────────────────────────────────────────────────────────────

# Virama (halant) = U+094D — joins consonants
VIRAMA = "\u094D"

# Optional: split Devanagari into grapheme clusters
# (consonant + optional virama + optional vowel sign)
# This gives the model a "syllable-level" view if desired.
GRAPHEME_RE = re.compile(
    r"[\u0900-\u097F\u1CD0-\u1CFF]"   # base char
    r"[\u094D[\u0900-\u097F]]*"         # optional virama+consonant
    r"[\u0900-\u097F\u0902\u0903]?",    # optional vowel sign / anusvara
    re.UNICODE,
)


def devanagari_graphemes(text: str) -> List[str]:
    """
    Split Devanagari text into grapheme clusters (akṣaras).
    Useful for CER evaluation and as an alternative encoding unit.
    """
    return GRAPHEME_RE.findall(text)


# ──────────────────────────────────────────────────────────────
# 3.  EVALUATION HELPERS
# ──────────────────────────────────────────────────────────────

def character_error_rate(pred: str, ref: str) -> float:
    """
    CER = edit_distance(pred_chars, ref_chars) / len(ref_chars)
    Target: < 0.05 for 95 %+ character accuracy.
    Requires: pip install editdistance
    """
    try:
        import editdistance
        return editdistance.eval(list(pred), list(ref)) / max(len(ref), 1)
    except ImportError:
        # Pure-python fallback (slower)
        return _levenshtein(list(pred), list(ref)) / max(len(ref), 1)


def _levenshtein(s: list, t: list) -> int:
    m, n = len(s), len(t)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            dp[j] = (
                prev[j - 1]
                if s[i - 1] == t[j - 1]
                else 1 + min(prev[j], dp[j - 1], prev[j - 1])
            )
    return dp[n]


def batch_cer(preds: List[str], refs: List[str]) -> float:
    """Average CER over a batch."""
    return sum(character_error_rate(p, r) for p, r in zip(preds, refs)) / len(refs)


# ──────────────────────────────────────────────────────────────
# 4.  TRAINING LOOP PATCH
#     Replace the two lines that load your BPE tokenizers with:
# ──────────────────────────────────────────────────────────────
INTEGRATION_SNIPPET = '''
# ── In your training script ──────────────────────────────────
from char_tokenizer import CharTokenizer, batch_cer

# Load (or build on first run) char-level tokenizers
src_tok = CharTokenizer.build_or_load(
    "char_src_tokenizer.json",
    script="roman",
    corpus_texts=train_roman_texts,   # list[str] from your dataset
)
tgt_tok = CharTokenizer.build_or_load(
    "char_tgt_tokenizer.json",
    script="devanagari",
    corpus_texts=train_devanagari_texts,
)

# Encoding  (replaces tokenizer(text, ...) calls)
src_ids = src_tok.batch_encode(batch_roman,      max_len=MAX_LEN, device=device)
tgt_ids = tgt_tok.batch_encode(batch_devanagari, max_len=MAX_LEN, device=device)

# Validation CER (add alongside your BERT score)
cer_score = batch_cer(decoded_preds, decoded_refs)   # target < 0.05

# Recommended hyperparameter changes alongside char tokenizer:
#   T            : 32  (was 128 — char mapping is low-entropy)
#   use_negatives: True
#   lr           : 3e-4 with cosine schedule, no aggressive decay
#   epochs       : 15+
'''


# ──────────────────────────────────────────────────────────────
# 5.  QUICK SANITY TEST
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Roman side
    roman_samples = [
        "baler atulavīryasya siṃhāsanagatasya vai //",
        "iti stutvā jagannātham uvāca munisattamān /",
    ]
    dev_samples = [
        "बलेर् अतुलवीर्यस्य सिंहासनगतस्य वै ॥",
        "इति स्तुत्वा जगन्नाथम् उवाच मुनिसत्तमान् ।",
    ]

    src_tok = CharTokenizer.build_from_corpus(roman_samples, script="roman")
    tgt_tok = CharTokenizer.build_from_corpus(dev_samples,   script="devanagari")

    print(f"\n🔤 SRC vocab size : {len(src_tok)}  (vs BPE 16 000)")
    print(f"🔤 TGT vocab size : {len(tgt_tok)}  (vs BPE 16 000)")

    # Encode → decode round-trip
    for s, d in zip(roman_samples, dev_samples):
        enc = src_tok.encode(s)
        rt  = src_tok.decode(enc)
        assert rt == s, f"Round-trip fail: {repr(rt)} != {repr(s)}"

        enc_d = tgt_tok.encode(d)
        rt_d  = tgt_tok.decode(enc_d)
        assert rt_d == d, f"Round-trip fail Devanagari: {repr(rt_d)}"

    print("✅ All round-trip encode/decode tests passed.")

    # CER demo
    pred = "बलेर् अतुलवीर्यस्य सिंहासनगतस्य वै ।"   # minor typo (॥→।)
    ref  = "बलेर् अतुलवीर्यस्य सिंहासनगतस्य वै ॥"
    print(f"\n📊 Demo CER (1 char error): {character_error_rate(pred, ref):.4f}")

    # Batch tensor
    tensor = src_tok.batch_encode(roman_samples, max_len=64)
    print(f"📦 Batch tensor shape: {tensor.shape}  (pad_id={src_tok.pad_token_id})")

    print("\n── Integration snippet ──────────────────────────────")
    print(INTEGRATION_SNIPPET)