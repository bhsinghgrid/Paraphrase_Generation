import torch
import torch.nn as nn

# ============================================================
# 🔹 Model Factory
# ============================================================

class SanskritModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        model_type = cfg['model_type']

        if model_type == "d3pm_encoder_decoder":
            # Encoder–Decoder without cross-attention (Vanilla)
            from model.d3pm_model_encoder_decoder import D3PMEncoderDecoder
            self.model = D3PMEncoderDecoder(cfg)

        elif model_type == "d3pm_cross_attention":
            # Encoder–Decoder with cross-attention (Improved)
            from model.d3pm_model_cross_attention import D3PMCrossAttention
            self.model = D3PMCrossAttention(cfg)

        elif model_type == "baseline_encoder_decoder":
            # Baseline Vanilla Transformer
            from model.d3pm_model_encoder_decoder import BaselineEncoderDecoder
            self.model = BaselineEncoderDecoder(cfg)

        elif model_type == "baseline_cross_attention":
            # Baseline Cross-Attention
            from model.d3pm_model_cross_attention import BaselineCrossAttention
            self.model = BaselineCrossAttention(cfg)

        else:
            raise ValueError(f"Invalid model_type: {model_type}")

    def forward(self, *args):
        return self.model(*args)

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)