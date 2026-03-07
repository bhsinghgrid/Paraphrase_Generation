"""
forward_process.py  — Verified Correct (no changes needed)
===========================================================
Absorbing (mask) diffusion. PAD never masked. At t=0 alpha=1.0 exactly
so x_t == x_0 (nothing masked). Works correctly with the fixed scheduler.
"""
import torch

class AbsorbingForwardProcess:
    def __init__(self, scheduler, mask_id=0, pad_id=1):
        self.scheduler = scheduler
        self.mask_id   = mask_id
        self.pad_id    = pad_id

    def q_sample(self, x_0, t):
        alpha_t = self.scheduler.get_alpha(t).to(x_0.device).view(-1, 1)
        r   = torch.rand(x_0.shape, device=x_0.device)
        x_t = x_0.clone()
        x_t[r > alpha_t]          = self.mask_id
        x_t[x_0 == self.pad_id]   = self.pad_id   # PAD stays PAD always
        return x_0, x_t