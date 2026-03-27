# Ablation config: T=32 diffusion steps
import os
import torch


def _get_env_int(name, default):
    value = os.environ.get(name)
    return int(value) if value is not None else default


def _get_env_str(name, default):
    return os.environ.get(name, default)


# 🎛️ BASH-CONTROLLED SWITCHES (Defaults if run manually)
MODEL = os.environ.get("MODEL_TYPE", "d3pm_encoder_decoder")
NEGATIVES = os.environ.get("INCLUDE_NEG", "False") == "True"
DIFFUSION_STEPS = _get_env_int("DIFFUSION_STEPS", 128)
INFERENCE_STEPS = _get_env_int("INFERENCE_NUM_STEPS", min(64, DIFFUSION_STEPS))
TRAIN_DEVICE = _get_env_str(
    "TRAIN_DEVICE",
    "mps" if torch.backends.mps.is_available() else "cpu",
)

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
        'diffusion_steps':  DIFFUSION_STEPS,
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
        'device':           TRAIN_DEVICE,
    },

    # ── Inference (used during val BERTScore and generate()) ──────────
    'inference': {
        'num_steps':          INFERENCE_STEPS,
        'temperature':        0.7,   # slightly lower = more confident output
        'top_k':              40,
        'repetition_penalty': 1.2,
        'diversity_penalty':  0.5,   # keep off; global-mean penalty is conservative
    },
}
