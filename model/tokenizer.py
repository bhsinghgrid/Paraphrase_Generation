"""
Sanskrit BPE Tokenizer with [MASK] token (ID = 0)
M4 Pro Optimized – Fast training on subset of dataset
"""

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from datasets import load_dataset
from pathlib import Path


# ============================================================
# 🔹 Generic BPE Builder (Reusable)
# ============================================================

def build_tokenizer(texts, vocab_size=16000):
    """
    Generic BPE tokenizer builder.
    Ensures [MASK] is the first special token → ID 0
    """

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[MASK]", "[PAD]", "[UNK]", "[CLS]", "[SEP]"],  # MASK first!
        min_frequency=2
    )

    tokenizer.train_from_iterator(texts, trainer)
    return tokenizer


# ============================================================
# 🔹 SanskritTokenizer Class
# ============================================================

class SanskritTokenizer:
    """
    Sanskrit BPE tokenizer wrapper.
    - Fast training (50k samples)
    - [MASK] = ID 0 (required for absorbing diffusion)
    - Supports padding + decoding
    """

    def __init__(self, vocab_size=16000, max_len=80):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.mask_token_id = 0
        self.model_path = "sanskrit_tokenizer_m4pro.json"

        if Path(self.model_path).exists():
            print("📖 Loading existing Sanskrit tokenizer...")
            self.tokenizer = Tokenizer.from_file(self.model_path)
            self._validate_mask_token()
        else:
            print("🎓 Training new Sanskrit tokenizer...")
            self._train_tokenizer()

    # --------------------------------------------------------
    # 🔹 Validation
    # --------------------------------------------------------
    def _validate_mask_token(self):
        mask_id = self.tokenizer.token_to_id("[MASK]")
        assert mask_id == 0, f"[MASK] must be ID 0, got {mask_id}"
        print("✅ [MASK] token confirmed at ID=0")

    # --------------------------------------------------------
    # 🔹 Fast Training (M4 Optimized)
    # --------------------------------------------------------
    def _train_tokenizer(self):
        """
        Train on subset of Sanskrit dataset for speed.
        """

        dataset = load_dataset(
            "paws/sanskrit-verses-gretil",
            split="train"
        )

        texts = []

        # Use 50k samples for fast training
        for sample in dataset.select(range(50000)):
            texts.extend([
                sample["quote_text"],
                sample["quote_devanagari"]
            ])

        # Build tokenizer using reusable function
        tokenizer = build_tokenizer(texts, self.vocab_size)

        tokenizer.save(self.model_path)
        self.tokenizer = tokenizer

        self._validate_mask_token()

        print("✅ Tokenizer trained (50k samples → 16k vocab)")

    # --------------------------------------------------------
    # 🔹 Encode
    # --------------------------------------------------------
    def encode(self, text):
        """
        Encode text → padded token IDs
        """
        encoded = self.tokenizer.encode(text)
        token_ids = encoded.ids[:self.max_len]

        pad_id = self.tokenizer.token_to_id("[PAD]")

        if len(token_ids) < self.max_len:
            token_ids += [pad_id] * (self.max_len - len(token_ids))

        return token_ids[:self.max_len]

    # --------------------------------------------------------
    # 🔹 Decode
    # --------------------------------------------------------
    def decode(self, token_ids):
        """
        Decode token IDs → text (removes padding)
        """
        pad_id = self.tokenizer.token_to_id("[PAD]")

        clean_ids = [t for t in token_ids if t != pad_id]

        return self.tokenizer.decode(clean_ids)

    def build_inputs_with_special_tokens(self, token_ids):
        """
        Required by BERTScore internally.
        Since your tokenizer may not have special tokens (bos/eos),
        we just return the token_ids as-is.
        """
        return list(token_ids)

    # Optional helper methods that BERTScore may call
    def get_vocab(self):
        """
        Return a dict mapping token string -> id.
        This can be simple if your tokenizer has vocab attribute.
        """
        if hasattr(self, "vocab_dict"):
            return self.vocab_dict
        return {str(i): i for i in range(self.vocab_size)}

    def convert_ids_to_tokens(self, ids):
        """
        Convert list of ids to list of strings.
        BERTScore may call this.
        """
        return [str(i) for i in ids]

    def __len__(self):
        return self.vocab_size