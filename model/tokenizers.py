"""
tokenizer.py — FINAL
=====================
Uses the original sanskrit_tokenizer_m4pro.json — the exact one the model
was trained with. Hard-coded absolute path as primary, with fallbacks.

This tokenizer has NO </w> end-of-word markers and NO decoder set.
decode() returns space-separated BPE pieces — this is the format the
model was trained and evaluated on (BERTScore 0.71). Do NOT add a decoder
or retrain: that would break alignment with the checkpoint.
"""

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from datasets import load_dataset
from pathlib import Path
import os

# Hard-coded absolute path — update if you move the project
TOKENIZER_PATH = "/Users/bhsingh/Documents/Final_Paraphrase/sanskrit_tokenizer_m4pro.json"


def build_tokenizer(texts, vocab_size=16000):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[MASK]", "[PAD]", "[UNK]", "[CLS]", "[SEP]"],
        min_frequency=2,
    )
    tokenizer.train_from_iterator(texts, trainer)
    return tokenizer


class SanskritTokenizer:
    def __init__(self, vocab_size=16000, max_len=80):
        self.vocab_size = vocab_size
        self.max_len    = max_len
        self.mask_token_id = 0

        script_dir = Path(__file__).resolve().parent
        candidates = [
            os.environ.get("SANSKRIT_TOKENIZER_PATH", ""),
            TOKENIZER_PATH,
            str(script_dir.parent / "sanskrit_tokenizer_m4pro.json"),
            str(script_dir / "sanskrit_tokenizer_m4pro.json"),
            str(Path.cwd() / "sanskrit_tokenizer_m4pro.json"),
        ]

        self.model_path = None
        for c in candidates:
            if c and Path(c).exists():
                self.model_path = c
                break

        if self.model_path:
            print(f"📖 Loading tokenizer from: {self.model_path}")
            self.tokenizer = Tokenizer.from_file(self.model_path)
            self._validate_mask_token()
        else:
            print(f"⚠️  Tokenizer not found at any candidate path.")
            print(f"    Expected: {TOKENIZER_PATH}")
            print("    Retraining — WARNING: output will not match existing checkpoint!")
            self.model_path = TOKENIZER_PATH
            self._train_tokenizer()

    def _validate_mask_token(self):
        mask_id = self.tokenizer.token_to_id("[MASK]")
        assert mask_id == 0, f"[MASK] must be ID 0, got {mask_id}"
        print("✅ [MASK] token confirmed at ID=0")

    def _train_tokenizer(self):
        dataset = load_dataset("paws/sanskrit-verses-gretil", split="train")
        texts = []
        for sample in dataset.select(range(50000)):
            texts.extend([sample["quote_text"], sample["quote_devanagari"]])
        tokenizer = build_tokenizer(texts, self.vocab_size)
        tokenizer.save(self.model_path)
        self.tokenizer = tokenizer
        self._validate_mask_token()
        print(f"✅ Tokenizer saved to: {self.model_path}")

    def encode(self, text):
        encoded   = self.tokenizer.encode(text)
        token_ids = encoded.ids[:self.max_len]
        pad_id    = self.tokenizer.token_to_id("[PAD]")
        if len(token_ids) < self.max_len:
            token_ids += [pad_id] * (self.max_len - len(token_ids))
        return token_ids[:self.max_len]

    def decode(self, ids):
        if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], list):
            raise TypeError("decode() expects 1D list of IDs, not 2D.")
        # Filter special tokens: 0=MASK 1=PAD 2=UNK 3=CLS 4=SEP
        clean = [i for i in ids if isinstance(i, int) and i > 4]
        if not clean:
            return ""
        return self.tokenizer.decode(clean, skip_special_tokens=True).strip()

    def build_inputs_with_special_tokens(self, token_ids):
        return list(token_ids)

    def get_vocab(self):
        return {str(i): i for i in range(self.vocab_size)}

    def convert_ids_to_tokens(self, ids):
        return [str(i) for i in ids]

    def __len__(self):
        return self.vocab_size