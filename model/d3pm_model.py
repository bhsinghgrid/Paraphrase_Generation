"""
🔥 PRODUCTION Sanskrit D3PM
REAL Absorbing Diffusion Implementation

✔ Forward Process (Absorbing)
✔ Cosine Scheduler
✔ Reverse Sampling
✔ M4 Pro Optimized
✔ Target: BERTScore 0.90+
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.scheduler import OptimizedCosineScheduler
from diffusion.forward import AbsorbingForwardProcess


class OptimizedSanskritD3PM(nn.Module):
    """
    Sanskrit D3PM Model

    Architecture:
    IAST (source) → Forward Diffusion → Noisy Target →
    Transformer Decoder → Clean Devanagari Prediction
    """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.device = torch.device("mps")

        # --------------------------------------------------
        # 🔄 Diffusion Components
        # --------------------------------------------------
        self.scheduler = OptimizedCosineScheduler(cfg)
        self.forward_process = AbsorbingForwardProcess(self.scheduler)
        self.mask_token_id = cfg['diffusion']['mask_token_id']

        # --------------------------------------------------
        # 📌 Token & Positional Embeddings
        # --------------------------------------------------
        self.embed = nn.Embedding(
            cfg['model']['vocab_size'],
            cfg['model']['d_model']
        )

        self.pos_emb = nn.Embedding(
            cfg['model']['max_seq_len'],
            cfg['model']['d_model']
        )

        # --------------------------------------------------
        # ⏳ Timestep Embedding (CRITICAL for Diffusion)
        # --------------------------------------------------
        self.time_mlp = nn.Sequential(
            nn.Linear(1, cfg['model']['d_model'] // 4),
            nn.SiLU(),
            nn.Linear(cfg['model']['d_model'] // 4, cfg['model']['d_model'])
        )

        # --------------------------------------------------
        # 🧠 Transformer Decoder (Conditioned on Source)
        # --------------------------------------------------
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=cfg['model']['d_model'],
                nhead=cfg['model']['n_heads'],
                dim_feedforward=cfg['model']['d_ff'],
                dropout=0.1,
                batch_first=True,
                norm_first=True
            ),
            num_layers=cfg['model']['n_layers']
        )

        # --------------------------------------------------
        # 🎯 Output Projection
        # --------------------------------------------------
        self.head = nn.Linear(
            cfg['model']['d_model'],
            cfg['model']['vocab_size']
        )

        # Mask embedding buffer (absorbing state placeholder)
        self.register_buffer(
            "mask_token_emb",
            torch.zeros(cfg['model']['d_model'])
        )

        print(f"✅ D3PM Parameters: {sum(p.numel() for p in self.parameters()):,}")
        print(f"✅ Diffusion Steps: {self.scheduler.num_timesteps}")

    # ==================================================
    # 🔥 Forward Pass (Training)
    # ==================================================
    def forward(self, src, tgt, t):
        """
        REAL D3PM Training Forward

        src : [B, L] IAST tokens
        tgt : [B, L] Clean Devanagari tokens
        t   : [B]    Diffusion timesteps

        Returns:
            logits      : [B, L, V]
            x_t_probs   : Forward diffusion probabilities
        """

        B, L = tgt.shape

        # --------------------------------------------------
        # 1️⃣ Source Conditioning (IAST Encoder)
        # --------------------------------------------------
        positions = torch.arange(L, device=self.device)

        src_emb = self.embed(src) + self.pos_emb(positions)
        memory = src_emb  # Simplified encoder memory

        # --------------------------------------------------
        # 2️⃣ Forward Diffusion (x₀ → xₜ)
        # --------------------------------------------------
        x_t_probs, x_t_ids = self.forward_process.q_sample(tgt, t)

        # --------------------------------------------------
        # 3️⃣ Noisy Target Embedding
        # --------------------------------------------------
        x_t_emb = self.embed(x_t_ids) + self.pos_emb(positions)

        # --------------------------------------------------
        # 4️⃣ Timestep Conditioning (CRITICAL)
        # --------------------------------------------------
        t_emb = self.time_mlp(t.float().unsqueeze(-1))  # [B, D]
        t_emb = t_emb.unsqueeze(1)  # [B, 1, D]

        x_t_emb = x_t_emb + t_emb.expand(-1, L, -1)

        # --------------------------------------------------
        # 5️⃣ Denoising Transformer
        # --------------------------------------------------
        decoder_out = self.transformer(x_t_emb, memory)

        # --------------------------------------------------
        # 6️⃣ Vocabulary Prediction
        # --------------------------------------------------
        logits = self.head(decoder_out)

        return logits, x_t_probs

    # ==================================================
    # 🔄 Reverse Diffusion (Single Step)
    # ==================================================
    def generate_step(self, noisy, cond, t, penalties=None):
        """
        Single reverse diffusion step for inference.
        """
        from diffusion.reverse import ReverseDiffusion
        reverse = ReverseDiffusion(self.scheduler)
        return reverse.p_sample_step(self, noisy, t, cond)

    # ==================================================
    # 🔥 Full Reverse Diffusion Generation
    # ==================================================
    def generate(self, condition, num_steps=None):
        """
        Full generation: Pure noise → Clean Sanskrit verse
        """
        from diffusion.reverse import ReverseDiffusion
        reverse = ReverseDiffusion(self.scheduler)
        return reverse.generate(self, condition, num_steps)

    # ==================================================
    # 🎨 Optional Penalty Logic
    # ==================================================
    def _apply_penalties(self, logits, prev_tokens, penalties):
        """
        Apply repetition and diversity penalties.
        """

        # Repetition penalty
        for i in range(logits.size(1)):
            repeated = (prev_tokens == prev_tokens[:, :i + 1]).any(dim=1)
            logits[repeated, i] /= penalties.repetition_penalty

        # Diversity penalty
        logits_var = logits.var(dim=-1, keepdim=True)
        logits += penalties.diversity_penalty * logits_var

        return logits