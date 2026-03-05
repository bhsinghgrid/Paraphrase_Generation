# import torch
# import math
#
# class OptimizedCosineScheduler:
#     def __init__(self, cfg, device=None):
#         self.num_timesteps = cfg['model']['diffusion_steps']
#         self.mask_token_id = cfg['diffusion']['mask_token_id']
#         self.device = device or torch.device('cpu')
#         self.alphas = self._cosine_alphas().to(self.device)
#
#     def sample_timestep(self, batch_size):
#         return torch.randint(1, self.num_timesteps, (batch_size,))
#
#     def _cosine_alphas(self):
#         steps = self.num_timesteps
#         t = torch.arange(steps + 1, dtype=torch.float32)
#         f_t = torch.cos((t / steps + 0.008) / 1.008 * math.pi / 2) ** 2
#         alphas_bar = f_t / f_t[0]
#         return alphas_bar[1:]
#
#     def get_alpha(self, t):
#         # This ensures t is an index for the alphas_cumprod tensor
#         return self.alphas_cumprod[t]
import torch
import math

class OptimizedCosineScheduler:
    def __init__(self, cfg, device=None):
        self.num_timesteps = cfg['model']['diffusion_steps']
        self.mask_token_id = cfg['diffusion']['mask_token_id']
        self.device = device or torch.device('cpu')
        # self.alphas = self._cosine_alphas().to(self.device)
        self.alphas_cumprod = self._cosine_alphas().to(self.device)

    def sample_timestep(self, batch_size):
        return torch.randint(1, self.num_timesteps, (batch_size,))

    def _cosine_alphas(self):
        steps = self.num_timesteps
        t = torch.arange(steps + 1, dtype=torch.float32)
        f_t = torch.cos((t / steps + 0.001) / 1.001 * math.pi / 2) ** 2 ## 1.008 to 1.0001
        alphas_bar = f_t / f_t[0]
        return alphas_bar[1:]

    def get_alpha(self, t):
        # This ensures t is an index for the alphas_cumprod tensor
        t_index = t.to(self.alphas_cumprod.device).long()
        return self.alphas_cumprod[t_index]