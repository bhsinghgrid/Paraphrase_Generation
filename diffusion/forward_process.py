"""
Absorbing Forward Diffusion - M4 Pro Memory Optimized
x₀ → xₜ using precomputed absorbing transition matrices
Designed to avoid MPS memory explosion.
"""

import torch
import torch.nn.functional as F


class AbsorbingForwardProcess:
    """
    Implements forward diffusion for absorbing D3PM.

    Instead of performing massive batch matrix multiplications
    (which can explode memory on M4/MPS),
    this version safely processes tokens one-by-one using CPU Q-matrices.
    """

    def __init__(self, scheduler):
        # Scheduler contains precomputed Q matrices and timestep logic
        self.scheduler = scheduler

    def q_sample(self, x0, t):
        """
        Sample x_t from q(x_t | x_0)

        Args:
            x0 : [B, L] LongTensor
                Clean token IDs.
            t  : [B] LongTensor
                Diffusion timesteps per batch sample.

        Returns:
            x_t_probs : [B, L, V]
                Forward probability distributions.
            x_t_ids   : [B, L]
                Noisy token IDs after diffusion.
        """

        B, L = x0.shape
        V = self.scheduler.Q_matrices.shape[-1]

        # Ensure t is a tensor on the correct device
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=x0.device, dtype=torch.long)

        # ---- 🔥 FIX START ----
        # Force t to shape (B,)
        if t.dim() == 0:
            t = t.unsqueeze(0)

        if t.dim() == 1 and t.shape[0] != B:
            t = t.expand(B)

        t = t.to(x0.device).long()
        # If t is scalar (0-dim), expand to batch size
        # if t.dim() == 0:
        #     t = t.unsqueeze(0).expand(B)


        # Allocate output tensors on the same device as input
        x_t_probs = torch.zeros(
            B, L, V,
            device=x0.device,
            dtype=torch.float32
        )

        x_t_ids = torch.zeros(
            B, L,
            device=x0.device,
            dtype=torch.long
        )

        # 🔥 Move transition matrices to CPU for safety (M4 memory optimization)
        Q_matrices_cpu = self.scheduler.Q_matrices.cpu()  # Shape: [T, V, V]

        # Process batch sample-by-sample
        for b in range(B):
            t_idx = t[b].item()
            Q_t = Q_matrices_cpu[t_idx]  # [V, V] (CPU)

            # Process each token in the sequence
            for l in range(L):
                x_token = x0[b, l].item()

                # Get transition probabilities for this token
                probs = Q_t[x_token]  # [V]

                # Move only small probability vector back to device
                x_t_probs[b, l] = probs.to(x0.device)

                # Discrete sampling via argmax
                x_t_ids[b, l] = probs.argmax().to(x0.device)

        return x_t_probs, x_t_ids