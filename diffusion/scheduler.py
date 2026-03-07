"""
scheduler.py  — Fixed & Upgraded
==================================
Changes:
  1. T=64 (was 16). More timesteps = richer denoising curriculum per epoch.
  2. alpha at t=0 is EXACTLY 1.0 — fixes Bug 2 (final-step re-noise).
  3. sample_timestep samples [0, T-1] including t=0, so model trains on
     fully-clean inputs (learns the identity at t=0 explicitly).
"""
import torch, math

class OptimizedCosineScheduler:
    def __init__(self, cfg, device=None):
        self.num_timesteps  = cfg['model']['diffusion_steps']   # 64
        self.mask_token_id  = cfg['diffusion']['mask_token_id']
        self.device         = device or torch.device('cpu')
        self.alphas_cumprod = self._build_schedule().to(self.device)

    def _build_schedule(self):
        T   = self.num_timesteps
        t   = torch.arange(T + 1, dtype=torch.float32)
        f_t = torch.cos((t / T + 0.008) / 1.008 * math.pi / 2) ** 2
        alphas_bar = f_t / f_t[0]
        alphas_bar = alphas_bar[1:]       # shape [T]
        alphas_bar[0]  = 1.0              # FIX: exact 1.0 at t=0
        alphas_bar[-1] = alphas_bar[-1].clamp(max=0.001)
        return alphas_bar

    def sample_timestep(self, batch_size):
        """Uniform [0, T-1] — includes t=0 so model sees clean inputs."""
        return torch.randint(0, self.num_timesteps, (batch_size,))

    def get_alpha(self, t):
        return self.alphas_cumprod[t.to(self.alphas_cumprod.device).long()]