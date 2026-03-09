"""
sanskrit_model.py  — Fixed
===========================
Added inference_mode parameter to forward() so reverse_process.py can
pass inference_mode=True without a TypeError.

The wrapper introspects each inner model's signature and only passes
kwargs that model actually accepts — safe across all four architectures.
"""

import torch
import torch.nn as nn
import inspect


class SanskritModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        model_type = cfg['model_type']

        if model_type == 'd3pm_cross_attention':
            from model.d3pm_model_cross_attention import D3PMCrossAttention
            self.model = D3PMCrossAttention(cfg)

        elif model_type == 'd3pm_encoder_decoder':
            from model.d3pm_model_encoder_decoder import D3PMEncoderDecoder
            self.model = D3PMEncoderDecoder(cfg)

        elif model_type == 'baseline_cross_attention':
            from model.d3pm_model_cross_attention import BaselineCrossAttention
            self.model = BaselineCrossAttention(cfg)

        elif model_type == 'baseline_encoder_decoder':
            from model.d3pm_model_encoder_decoder import BaselineEncoderDecoder
            self.model = BaselineEncoderDecoder(cfg)

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def forward(self, input_ids, target_ids, t, x0_hint=None, inference_mode=False):
        """
        Forward pass.  Introspects the inner model's signature so only
        supported kwargs are passed — works with all four architectures.
        """
        sig    = inspect.signature(self.model.forward).parameters
        kwargs = {}
        if 'x0_hint'        in sig:
            kwargs['x0_hint']        = x0_hint
        if 'inference_mode' in sig:
            kwargs['inference_mode'] = inference_mode

        if 't' in sig:
            return self.model(input_ids, target_ids, t, **kwargs)
        else:
            return self.model(input_ids, target_ids, **kwargs)

    @torch.no_grad()
    def generate(self, src, **kwargs):
        sig      = inspect.signature(self.model.generate).parameters
        filtered = {k: v for k, v in kwargs.items() if k in sig}
        return self.model.generate(src, **filtered)