import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from diffusion.scheduler import OptimizedCosineScheduler
from diffusion.forward_process import AbsorbingForwardProcess
from diffusion.reverse_process import ReverseDiffusion


# ============================================================
# 🔹 Embeddings
# ============================================================

class SanskritEmbeddings(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        nn.init.xavier_uniform_(self.token_emb.weight)
        nn.init.xavier_uniform_(self.pos_emb.weight)

        # ✅ Alias to prevent attribute error
        self.token_embedding = self.token_emb

    def forward(self, tokens):
        B, L = tokens.shape
        positions = torch.arange(L, device=tokens.device)
        return self.token_emb(tokens) + self.pos_emb(positions).unsqueeze(0)


# ============================================================
# 🔹 Encoder Block
# ============================================================

class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Linear(d_ff, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)

        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x


# ============================================================
# 🔹 Decoder Block (Full Encoder–Decoder)
# ============================================================

class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Linear(d_ff, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, memory):
        sa_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + sa_out)

        ca_out, _ = self.cross_attn(x, memory, memory)
        x = self.norm2(x + ca_out)

        ff_out = self.ff(x)
        x = self.norm3(x + ff_out)

        return x


# ============================================================
# 🔹 Cross-Attention Block (Decoder-only style)
# ============================================================
#
# class CrossAttentionBlock(nn.Module):
#     def __init__(self, d_model, n_heads, d_ff):
#         super().__init__()
#
#         self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
#         self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
#
#         self.ff = nn.Sequential(
#             nn.Linear(d_model, d_ff),
#             nn.SiLU(),
#             nn.Linear(d_ff, d_model)
#         )
#
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.norm3 = nn.LayerNorm(d_model)
#
#     def forward(self, x, memory):
#         sa_out, _ = self.self_attn(x, x, x)
#         x = self.norm1(x + sa_out)
#
#         ca_out, _ = self.cross_attn(x, memory, memory)
#         x = self.norm2(x + ca_out)
#
#         ff_out = self.ff(x)
#         x = self.norm3(x + ff_out)
#
#         return x
import torch
import torch.nn as nn

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )

        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm3 = nn.LayerNorm(d_model, eps=1e-5)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_pad_mask=None, src_pad_mask=None):

        # ---- Self Attention (Pre-Norm) ----
        x_norm = self.norm1(x)
        sa_out, _ = self.self_attn(
            x_norm, x_norm, x_norm,
            key_padding_mask=tgt_pad_mask
        )
        x = x + self.dropout(sa_out)

        # ---- Cross Attention (Pre-Norm) ----
        x_norm = self.norm2(x)
        ca_out, _ = self.cross_attn(
            x_norm, memory, memory,
            key_padding_mask=src_pad_mask
        )
        x = x + self.dropout(ca_out)

        # ---- Feed Forward (Pre-Norm) ----
        x_norm = self.norm3(x)
        ff_out = self.ff(x_norm)
        x = x + self.dropout(ff_out)

        return x


# ============================================================
# 🔥 1️⃣ Diffusion Encoder–Decoder Model
# ============================================================

class D3PMEncoderDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.mask_token_id = cfg['diffusion']['mask_token_id']

        # self.src_embed = SanskritEmbeddings(**cfg['model'])
        # self.tgt_embed = SanskritEmbeddings(**cfg['model'])

        # self.src_embed = SanskritEmbeddings(**cfg['model'])
        self.src_embed = SanskritEmbeddings(
            vocab_size=cfg['model']['vocab_size'],
            d_model=cfg['model']['d_model'],
            max_seq_len=cfg['model']['max_seq_len']
        )
        # self.tgt_embed = SanskritEmbeddings(**cfg['model'])
        self.tgt_embed = SanskritEmbeddings(
            vocab_size=cfg['model']['vocab_size'],
            d_model=cfg['model']['d_model'],
            max_seq_len=cfg['model']['max_seq_len']
        )

        self.scheduler = OptimizedCosineScheduler(cfg)
        self.forward_process = AbsorbingForwardProcess(self.scheduler)

        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(cfg['model']['d_model'],
                         cfg['model']['n_heads'],
                         cfg['model']['d_ff'])
            for _ in range(cfg['model']['n_layers'])
        ])

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(cfg['model']['d_model'],
                         cfg['model']['n_heads'],
                         cfg['model']['d_ff'])
            for _ in range(cfg['model']['n_layers'])
        ])

        self.time_mlp = nn.Sequential(
            nn.Linear(1, cfg['model']['d_model']//4),
            nn.SiLU(),
            nn.Linear(cfg['model']['d_model']//4,
                      cfg['model']['d_model'])
        )

        self.head = nn.Linear(cfg['model']['d_model'],
                              cfg['model']['vocab_size'])

    def forward(self, src, tgt, t):
        B, L = tgt.shape

        memory = self.src_embed(src)
        for block in self.encoder_blocks:
            memory = block(memory)

        x_t_probs, x_t_ids = self.forward_process.q_sample(tgt, t)
        x = self.tgt_embed(x_t_ids)

        t_emb = self.time_mlp(t.float().unsqueeze(-1)).unsqueeze(1)
        x = x + t_emb.expand(-1, L, -1)

        for block in self.decoder_blocks:
            x = block(x, memory)

        logits = self.head(x)
        return logits, x_t_probs

    def generate(self, src, num_steps=None, beam_width=1):
        # src: [B, L] tensor
        scheduler = OptimizedCosineScheduler(self.cfg)  # or self.scheduler
        reverse_diffusion = ReverseDiffusion(scheduler)
        return reverse_diffusion.generate_beam(
            self,  # pass the model
            condition=src,
            beam_width=beam_width,
            num_steps=num_steps or self.forward_process.scheduler.num_timesteps
        )

    # @torch.no_grad()
    # def generate(self, src, num_steps=None, beam_width=1, start_token_id=2, device=None):
    #     """
    #     Greedy / beam generation using reverse diffusion.
    #
    #     Args:
    #         src           : [B, L] LongTensor input tokens
    #         num_steps     : int, number of reverse diffusion steps
    #         beam_width    : int, beam size (currently only 1 supported for greedy)
    #         start_token_id: int, initial decoder token
    #         device        : torch.device (optional, defaults to src.device)
    #
    #     Returns:
    #         generated_ids : [B, L_gen] LongTensor of generated token IDs
    #     """
    #     device = device or src.device
    #     B, L_src = src.shape
    #
    #     # Use model's scheduler and forward_process
    #     scheduler = getattr(self, "scheduler", None)
    #     if scheduler is None:
    #         raise ValueError("Model must have a scheduler for reverse diffusion.")
    #
    #     # Default num_steps from scheduler if not provided
    #     num_steps = num_steps or scheduler.num_timesteps
    #
    #     # Initialize reverse diffusion helper
    #     from diffusion.reverse_process import ReverseDiffusion
    #     reverse_diffusion = ReverseDiffusion(scheduler)
    #
    #     # Run beam / greedy generation
    #     generated_ids = reverse_diffusion.generate_beam(
    #         model=self,  # pass the model itself
    #         condition=src,  # [B, L_src] conditioning
    #         beam_width=beam_width,  # 1 = greedy
    #         num_steps=num_steps,  # diffusion steps
    #         start_token_id=start_token_id,
    #         device=device
    #     )
    #     return generated_ids


# ============================================================
# 🔥 2️⃣ Diffusion Cross-Attention Model
# ============================================================

# class D3PMCrossAttention(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#
#         self.mask_token_id = cfg['diffusion']['mask_token_id']
#
#         # self.src_embed = SanskritEmbeddings(**cfg['model'])
#         self.src_embed = SanskritEmbeddings(
#             vocab_size=cfg['model']['vocab_size'],
#             d_model=cfg['model']['d_model'],
#             max_seq_len=cfg['model']['max_seq_len']
#         )
#         # self.tgt_embed = SanskritEmbeddings(**cfg['model'])
#         self.tgt_embed = SanskritEmbeddings(
#             vocab_size=cfg['model']['vocab_size'],
#             d_model=cfg['model']['d_model'],
#             max_seq_len=cfg['model']['max_seq_len']
#         )
#
#         self.scheduler = OptimizedCosineScheduler(cfg)
#         self.forward_process = AbsorbingForwardProcess(self.scheduler)
#
#         self.blocks = nn.ModuleList([
#             CrossAttentionBlock(cfg['model']['d_model'],
#                                 cfg['model']['n_heads'],
#                                 cfg['model']['d_ff'])
#             for _ in range(cfg['model']['n_layers'])
#         ])
#
#         self.time_mlp = nn.Sequential(
#             nn.Linear(1, cfg['model']['d_model']//4),
#             nn.SiLU(),
#             nn.Linear(cfg['model']['d_model']//4,
#                       cfg['model']['d_model'])
#         )
#
#         self.head = nn.Linear(cfg['model']['d_model'],
#                               cfg['model']['vocab_size'])
#
#     def forward(self, src, tgt, t):
#         B, L = tgt.shape
#
#         memory = self.src_embed(src)
#
#         x_t_probs, x_t_ids = self.forward_process.q_sample(tgt, t)
#         x = self.tgt_embed(x_t_ids)
#
#         t_emb = self.time_mlp(t.float().unsqueeze(-1)).unsqueeze(1)
#         x = x + t_emb.expand(-1, L, -1)
#
#         for block in self.blocks:
#             x = block(x, memory)
#
#         logits = self.head(x)
#         return logits, x_t_probs
#
#     @torch.no_grad()
#     def generate(self, src, num_steps=None, beam_width=1):
#         """
#         Generate sequences from source input using reverse diffusion.
#         src: [B, L] tensor
#         """
#         # Use the model's scheduler
#         scheduler = self.scheduler
#         reverse_diffusion = ReverseDiffusion(scheduler)
#
#         # If num_steps not provided, use scheduler's default
#         steps = num_steps or scheduler.num_timesteps
#
#         # Pass self to reverse diffusion
#         return reverse_diffusion.generate_beam(
#             self,
#             condition=src,
#             beam_width=beam_width,
#             num_steps=steps
#         )

    # @torch.no_grad()
    # def generate(self, src, num_steps=None, beam_width=1, start_token_id=2, device=None):
    #     """
    #     Greedy / beam generation using reverse diffusion.
    #
    #     Args:
    #         src           : [B, L] LongTensor input tokens
    #         num_steps     : int, number of reverse diffusion steps
    #         beam_width    : int, beam size (currently only 1 supported for greedy)
    #         start_token_id: int, initial decoder token
    #         device        : torch.device (optional, defaults to src.device)
    #
    #     Returns:
    #         generated_ids : [B, L_gen] LongTensor of generated token IDs
    #     """
    #     device = device or src.device
    #     B, L_src = src.shape
    #
    #     # Use model's scheduler and forward_process
    #     scheduler = getattr(self, "scheduler", None)
    #     if scheduler is None:
    #         raise ValueError("Model must have a scheduler for reverse diffusion.")
    #
    #     # Default num_steps from scheduler if not provided
    #     num_steps = num_steps or scheduler.num_timesteps
    #
    #     # Initialize reverse diffusion helper
    #     from diffusion.reverse_process import ReverseDiffusion
    #     reverse_diffusion = ReverseDiffusion(scheduler)
    #
    #     # Run beam / greedy generation
    #     generated_ids = reverse_diffusion.generate_beam(
    #         model=self,  # pass the model itself
    #         condition=src,  # [B, L_src] conditioning
    #         beam_width=beam_width,  # 1 = greedy
    #         num_steps=num_steps,  # diffusion steps
    #         start_token_id=start_token_id,
    #         device=device
    #     )
    #     return generated_ids

class D3PMCrossAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        d_model = cfg['model']['d_model']
        vocab_size = cfg['model']['vocab_size']
        dropout = cfg['model'].get('dropout', 0.1)

        self.pad_token_id = cfg['model'].get('pad_token_id', 0)
        self.mask_token_id = cfg['diffusion']['mask_token_id']

        # Embeddings
        self.src_embed = SanskritEmbeddings(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=cfg['model']['max_seq_len']
        )

        self.tgt_embed = SanskritEmbeddings(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=cfg['model']['max_seq_len']
        )

        # Scheduler + Diffusion
        self.scheduler = OptimizedCosineScheduler(cfg)
        self.forward_process = AbsorbingForwardProcess(self.scheduler)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(
                d_model,
                cfg['model']['n_heads'],
                cfg['model']['d_ff'],
                dropout=dropout
            )
            for _ in range(cfg['model']['n_layers'])
        ])

        # Time embedding (scaled)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, d_model)
        )

        self.final_norm = nn.LayerNorm(d_model, eps=1e-5)

        # Output head
        self.head = nn.Linear(d_model, vocab_size)

        # 🔥 Weight tying (important)
        self.head.weight = self.tgt_embed.token_embedding.weight

    def forward(self, src, tgt, t):
        B, L = tgt.shape

        # Padding masks
        src_pad_mask = (src == self.pad_token_id)
        tgt_pad_mask = (tgt == self.pad_token_id)

        # Encoder memory
        memory = self.src_embed(src)

        # Forward diffusion
        x_t_probs, x_t_ids = self.forward_process.q_sample(tgt, t)
        x = self.tgt_embed(x_t_ids)

        # Time scaling to [0,1]
        t_scaled = t.float() / self.scheduler.num_timesteps
        t_emb = self.time_mlp(t_scaled.unsqueeze(-1)).unsqueeze(1)

        x = x + t_emb

        # Transformer blocks
        for block in self.blocks:
            x = block(
                x,
                memory,
                tgt_pad_mask=tgt_pad_mask,
                src_pad_mask=src_pad_mask
            )

        x = self.final_norm(x)

        logits = self.head(x)

        return logits, x_t_probs

    @torch.no_grad()
    def generate(self, src, num_steps=None, temperature=1.0):
        reverse_diffusion = ReverseDiffusion(self.scheduler)
        return reverse_diffusion.generate_beam(
            model=self,
            condition=src,
            num_steps=num_steps,
            temperature=temperature
        )


# ============================================================
# 🔹 Baseline Encoder–Decoder (No Diffusion)
# ============================================================

class BaselineEncoderDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # self.src_embed = SanskritEmbeddings(**cfg['model'])
        # self.tgt_embed = SanskritEmbeddings(**cfg['model'])

        # self.src_embed = SanskritEmbeddings(**cfg['model'])
        self.src_embed = SanskritEmbeddings(
            vocab_size=cfg['model']['vocab_size'],
            d_model=cfg['model']['d_model'],
            max_seq_len=cfg['model']['max_seq_len']
        )
        # self.tgt_embed = SanskritEmbeddings(**cfg['model'])
        self.tgt_embed = SanskritEmbeddings(
            vocab_size=cfg['model']['vocab_size'],
            d_model=cfg['model']['d_model'],
            max_seq_len=cfg['model']['max_seq_len']
        )

        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(cfg['model']['d_model'],
                         cfg['model']['n_heads'],
                         cfg['model']['d_ff'])
            for _ in range(cfg['model']['n_layers'])
        ])

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(cfg['model']['d_model'],
                         cfg['model']['n_heads'],
                         cfg['model']['d_ff'])
            for _ in range(cfg['model']['n_layers'])
        ])

        self.head = nn.Linear(cfg['model']['d_model'],
                              cfg['model']['vocab_size'])

    def forward(self, src, tgt):
        memory = self.src_embed(src)
        for block in self.encoder_blocks:
            memory = block(memory)

        x = self.tgt_embed(tgt)
        for block in self.decoder_blocks:
            x = block(x, memory)

        return self.head(x)

    @torch.no_grad()
    def generate(self, src, max_len=50, start_token_id=2):
        """
        Greedy generation.
        src: [batch, seq_len] tensor
        returns: [batch, generated_seq_len] tensor
        """
        batch_size = src.size(0)
        device = src.device

        memory = self.src_embed(src)

        # Initialize decoder input with start token
        ys = torch.ones(batch_size, 1, dtype=torch.long, device=device) * start_token_id

        for _ in range(max_len):
            x = self.tgt_embed(ys)
            for block in self.decoder_blocks:  # ✅ use decoder_blocks, not self.blocks
                x = block(x, memory)
            logits = self.head(x)  # [batch, seq_len, vocab_size]
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)

            # Stop if all sequences generate EOS (optional)
            # if (next_token == eos_token_id).all():
            #     break

        return ys[:, 1:]

# ============================================================
# 🔹 Baseline Cross-Attention (No Diffusion)
# ============================================================

class BaselineCrossAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # self.src_embed = SanskritEmbeddings(**cfg['model'])
        # self.tgt_embed = SanskritEmbeddings(**cfg['model'])

        # self.src_embed = SanskritEmbeddings(**cfg['model'])
        self.src_embed = SanskritEmbeddings(
            vocab_size=cfg['model']['vocab_size'],
            d_model=cfg['model']['d_model'],
            max_seq_len=cfg['model']['max_seq_len']
        )
        # self.tgt_embed = SanskritEmbeddings(**cfg['model'])
        self.tgt_embed = SanskritEmbeddings(
            vocab_size=cfg['model']['vocab_size'],
            d_model=cfg['model']['d_model'],
            max_seq_len=cfg['model']['max_seq_len']
        )

        self.blocks = nn.ModuleList([
            CrossAttentionBlock(cfg['model']['d_model'],
                                cfg['model']['n_heads'],
                                cfg['model']['d_ff'])
            for _ in range(cfg['model']['n_layers'])
        ])

        self.head = nn.Linear(cfg['model']['d_model'],
                              cfg['model']['vocab_size'])

    def forward(self, src, tgt):
        memory = self.src_embed(src)
        x = self.tgt_embed(tgt)

        for block in self.blocks:
            x = block(x, memory)

        return self.head(x)
        # 🔹 Add this method for inference / generation

    @torch.no_grad()
    def generate(self, src, max_len=50, start_token_id=2):
        """
        Greedy generation.
        src: [batch, seq_len] tensor
        returns: [batch, generated_seq_len] tensor
        """
        batch_size = src.size(0)
        device = src.device

        memory = self.src_embed(src)

        # Initialize decoder input with start token
        ys = torch.ones(batch_size, 1, dtype=torch.long, device=device) * start_token_id

        for _ in range(max_len):
            x = self.tgt_embed(ys)
            for block in self.blocks:
                x = block(x, memory)
            logits = self.head(x)  # [batch, seq_len, vocab_size]
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)

            # Stop if all sequences generate EOS (optional)
            # if (next_token == eos_token_id).all():
            #     break

        return ys[:, 1:]


# ============================================================
# 🔥 MODEL FACTORY
# ============================================================

class SanskritModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        model_type = cfg['model_type']

        if model_type == "d3pm_encoder_decoder":
            self.model = D3PMEncoderDecoder(cfg)

        elif model_type == "d3pm_cross_attention":
            self.model = D3PMCrossAttention(cfg)

        elif model_type == "baseline_encoder_decoder":
            self.model = BaselineEncoderDecoder(cfg)

        elif model_type == "baseline_cross_attention":
            self.model = BaselineCrossAttention(cfg)

        else:
            raise ValueError("Invalid model_type")

    def forward(self, *args):
        return self.model(*args)