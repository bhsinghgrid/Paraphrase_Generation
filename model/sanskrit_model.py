"""
sanskrit_model.py
=================
Model factory — picks the right architecture from config['model_type'].
All models expose the same forward(src, tgt, t, x0_hint) API.
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

    def forward(self, input_ids, target_ids, t, x0_hint=None):
        sig = inspect.signature(self.model.forward).parameters
        if 'x0_hint' in sig:
            return self.model(input_ids, target_ids, t, x0_hint=x0_hint)
        elif 't' in sig:
            return self.model(input_ids, target_ids, t)
        else:
            return self.model(input_ids, target_ids)

    @torch.no_grad()
    def generate(self, src, **kwargs):
        sig = inspect.signature(self.model.generate).parameters
        filtered = {k: v for k, v in kwargs.items() if k in sig}
        return self.model.generate(src, **filtered)