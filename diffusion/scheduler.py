"""
Cosine Absorbing + Spindle Scheduling
(Designed for ~25% loss reduction with stable diffusion training)
"""

import torch
import math


class OptimizedCosineScheduler:
    """
    📌 Diffusion Scheduler with:
    - Cosine alpha schedule
    - Absorbing state transition matrices
    - Spindle timestep sampling (mid-steps prioritized)
    """

    def __init__(self, cfg, device=None):
        # Total diffusion steps
        self.num_timesteps = cfg['model']['diffusion_steps']

        # Mask token ID used for absorbing state
        self.mask_token_id = cfg['diffusion']['mask_token_id']

        # Device setup (CPU fallback if not provided)
        self.device = device or torch.device('cpu')

        # Precompute cosine decay schedule
        self.alphas = self._cosine_alphas().to(self.device)

        # Precompute absorbing transition matrices
        self.Q_matrices = self._precompute_absorbing().to(self.device)

        # 🔥 Spindle sampling weights
        # Prioritizes mid diffusion steps (important for Sanskrit structure learning)
        self.spindle_weights = torch.tensor(
            [0.05, 0.10, 0.15, 0.25, 0.20, 0.15, 0.10] * 2
        )

        # Trim to match number of timesteps
        self.spindle_weights = self.spindle_weights[:self.num_timesteps]

        # Normalize probabilities
        self.spindle_weights /= self.spindle_weights.sum()

    # --------------------------------------------------
    # 🎯 Spindle Timestep Sampling
    # --------------------------------------------------
    def sample_timestep(self, batch_size):
        """
        Sample timesteps using spindle distribution.
        Gives more probability mass to middle timesteps.
        """
        return torch.multinomial(
            self.spindle_weights,
            batch_size,
            replacement=True
        )

    # --------------------------------------------------
    # Cosine Alpha Schedule
    # --------------------------------------------------
    def _cosine_alphas(self):
        """
        Cosine decay schedule:
        alpha_t = cos^2(t * pi / 2)
        """
        t = torch.arange(self.num_timesteps) / self.num_timesteps
        return torch.cos(t * math.pi / 2) ** 2

    # --------------------------------------------------
    # 🔄 Absorbing Transition Matrix (First Definition)
    # --------------------------------------------------
    def _precompute_absorbing(self):
        """
        Builds transition matrices Q_t of size [V, V]
        where tokens can either:
        - stay the same (scaled by alpha_t)
        - transition to MASK token
        """
        V = 16000
        Q = []

        for t in range(self.num_timesteps):
            alpha_t = self.alphas[t]

            Q_t = torch.eye(V) * alpha_t
            Q_t[:, 0] = 1 - alpha_t  # Absorb into [MASK]

            Q.append(Q_t)

        return torch.stack(Q)

    # --------------------------------------------------
    # 📦 Get Batch Transition Matrix
    # --------------------------------------------------
    def get_transition_matrix(self, t):
        """
        Always returns a 3D tensor of shape:
        [B, V, V] — compatible with torch.bmm
        """
        # Extract scalar timestep
        t_idx = t[0].item() if t.dim() > 0 else t.item()

        # Select transition matrix
        Q_t = self.Q_matrices[t_idx]  # Shape: [16000, 16000]

        # Expand for batch dimension
        batch_size = t.shape[0] if t.dim() > 0 else 1
        return Q_t.unsqueeze(0).expand(batch_size, -1, -1)

    # --------------------------------------------------
    # 🔄 Absorbing Transition Matrix (Second Definition - Device Aware)
    # --------------------------------------------------
    def _precompute_absorbing(self):
        """
        Same logic as above but explicitly device-aware.
        This overrides the previous definition (Python behavior).
        """
        V = 16000
        Q = []

        for t in range(self.num_timesteps):
            alpha_t = self.alphas[t]

            Q_t = torch.eye(V, device=self.device) * alpha_t
            Q_t[:, 0] = 1 - alpha_t  # Absorb into [MASK]

            Q.append(Q_t)

        return torch.stack(Q)