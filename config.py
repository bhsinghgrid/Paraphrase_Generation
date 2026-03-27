import os
import torch

# 🎛️ BASH-CONTROLLED SWITCHES (Defaults if run manually)
MODEL = os.environ.get("MODEL_TYPE", "d3pm_cross_attention")
NEGATIVES = os.environ.get("INCLUDE_NEG", "False") == "True"

CONFIG = {
    "model_type": MODEL,

    "data": {
        "include_negative_examples": NEGATIVES,
        "dataset_size": 60000,
    },
'diffusion': {
        'mask_token_id': 0,          # [MASK] = ID 0, fixed by tokenizer
    },

    # ── Model architecture ────────────────────────────────────────────
    'model': {
        # 'vocab_size':       16000,
'src_vocab_size': 16000,   # Roman/IAST BPE vocab
'tgt_vocab_size': 16000,   # Devanagari BPE vocab
        'd_model':          1024,#512,     # was 384 — kept same, shared embeds save params
        'n_heads':          8,       # 384 / 6 = 64 head_dim
        'd_ff':            4096, #2048, #1536,    # 4 × d_model
        'n_layers':         8,#4,
        'dropout':          0.2,
        'max_seq_len':      80,
        'diffusion_steps':  128,      # CHANGED from 16 → 64. Richer curriculum.
    },

    # ── Training ──────────────────────────────────────────────────────
    'training': {
        'epochs':           20,       # Target: 0.71→0.83-0.85 in 5 epochs
        'batch_size':       32,
        'accum_steps':      2,       # effective batch = 64
        'lr':               7e-5,#6e-4,    # raised from 3e-4; warmup protects first steps
        'label_smoothing':  0.1,     # was 0.0; reduces overconfidence (gap 1.7 nats)
        'patience':         4,       # early stop after 4 non-improving epochs
        'l1_lambda':        1e-7,    # very light L1
        'device':           'mps',   # change to 'cuda' or 'cpu' as needed
    },

    # ── Inference (used during val BERTScore and generate()) ──────────
    'inference': {
        'num_steps':          64,    # must equal diffusion_steps
        'temperature':        0.7,   # slightly lower = more confident output
        'top_k':              40,
        'repetition_penalty': 1.2,
        'diversity_penalty':  0.5,   # keep off; global-mean penalty is conservative
    },
}