import torch
import torch.nn as nn
import inspect

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

    # def forward(self, *args):
    #     return self.model(*args)

    # Inside your SanskritModel class:
    def forward(self, input_ids, target_ids, t, x0_hint=None):
        sig = inspect.signature(self.model.forward).parameters

        if 'x0_hint' in sig:
            # If it's your updated Cross-Attention model, pass the hint
            return self.model(input_ids, target_ids, t, x0_hint=x0_hint)
        else:
            # If it's the Encoder-Decoder or an older model, ignore the hint safely
            return self.model(input_ids, target_ids, t)
        # We just pass the hint down to the internal model
        # return self.model(input_ids, target_ids, t, x0_hint=x0_hint)

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        # import inspect

        # 1. Look at the blueprint of the underlying model's generate method
        sig = inspect.signature(self.model.generate).parameters

        # 2. Automatically filter out any keywords it doesn't support
        # (e.g., it will drop 'temperature' if the EncoderDecoder doesn't use it)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig}
        return self.model.generate(*args, **kwargs)
