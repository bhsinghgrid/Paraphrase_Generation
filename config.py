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

    # "model": {
    #     "vocab_size": 16000,
    #     "max_seq_len": 80,
    #     "diffusion_steps": 10,
    #     "d_model": 384,
    #     "n_layers": 6,
    #     "n_heads": 6,
    #     "d_ff": 1536,
    #     "dropout": 0.15
    # },
    #
    # "diffusion": {
    #     "mask_token_id": 0
    # },
    #
    # "training": {
    #     "batch_size": 32,
    #     "epochs": 10,
    #     "lr": 2e-4,
    #     "label_smoothing": 0.05,
    #     "precision": "float32",
    #     "device": "mps" if torch.backends.mps.is_available() else "cpu",
    #     "early_stopping_patience": 3
    # }
"model": {
        "vocab_size": 16000,
        "max_seq_len": 96,       # Optimized for GRETIL slokas
        "diffusion_steps": 16,   # Use 16 steps (better than 8)
        "d_model": 512,          # Wider model learns faster
        "n_layers": 8,
        "n_heads": 8,
        "d_ff": 2048,
        "dropout": 0.1
    },

    "diffusion": {
        "mask_token_id": 0
    },

    "training": {
        "batch_size": 32,
        "epochs": 20,            # 20 is enough with these tweaks
        "lr": 4e-4,              # Higher LR + Warmup for speed
        "label_smoothing": 0.15, # Increased for 16k vocab stability
        "precision": "float32",
        "device": "mps" if torch.backends.mps.is_available() else "cpu",
        "early_stopping_patience": 5
    }
}