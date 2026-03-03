import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.scheduler import OptimizedCosineScheduler
from diffusion.forward_process import AbsorbingForwardProcess
from diffusion.reverse_process import ReverseDiffusion


# ============================================================
# 🔹 Embedding Layer
# ============================================================

class SanskritEmbeddings(nn.Module):
    """
    Token + Positional embeddings for Sanskrit sequences.
    Everything is created on CPU — model.to(device) will move it later.
    """

    def __init__(self, vocab_size, d_model, max_len):
        super().__init__()
        self.d_model = d_model

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.token_emb.weight)
        nn.init.xavier_uniform_(self.pos_emb.weight)

    def forward(self, tokens):
        """
        tokens: [B, L]
        returns: [B, L, d_model]
        """
        B, L = tokens.shape

        token_embeddings = self.token_emb(tokens)

        positions = torch.arange(
            L,
            device=tokens.device,
            dtype=torch.long
        )
        position_embeddings = self.pos_emb(positions)

        embeddings = token_embeddings + position_embeddings.unsqueeze(0)

        return embeddings


# ============================================================
# 🔹 Cross-Attention Transformer Block
# ============================================================

class CrossAttentionBlock(nn.Module):
    """
    Decoder block containing:
    - Self Attention
    - Cross Attention (encoder memory)
    - Feed Forward Network
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            batch_first=True
        )

        self.cross_attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            batch_first=True
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Linear(d_ff, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory):
        """
        x: decoder states
        memory: encoder outputs
        """

        # ---- Self Attention ----
        self_attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(self_attn_out))

        # ---- Cross Attention ----
        cross_attn_out, _ = self.cross_attn(x, memory, memory)
        x = self.norm2(x + self.dropout(cross_attn_out))

        # ---- Feed Forward ----
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_out))

        return x


# ============================================================
# 🔹 Sanskrit Cross-Attention D3PM Model
# ============================================================

class SanskritCrossAttentionTransformer(nn.Module):
    """
    Encoder-Decoder style D3PM model using
    cross-attention blocks for reverse diffusion.
    """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.dtype = torch.float32   # MPS-safe
        self.mask_token_id = cfg['diffusion']['mask_token_id']

        # ----------------------------------------------------
        # Embeddings
        # ----------------------------------------------------
        self.src_embed = SanskritEmbeddings(
            cfg['model']['vocab_size'],
            cfg['model']['d_model'],
            cfg['model']['max_seq_len']
        )

        self.tgt_embed = SanskritEmbeddings(
            cfg['model']['vocab_size'],
            cfg['model']['d_model'],
            cfg['model']['max_seq_len']
        )

        # ----------------------------------------------------
        # Diffusion Modules
        # ----------------------------------------------------
        self.scheduler = OptimizedCosineScheduler(cfg)
        self.forward_process = AbsorbingForwardProcess(self.scheduler)
        self.reverse_process = ReverseDiffusion(self.scheduler)

        # ----------------------------------------------------
        # Time Embedding
        # ----------------------------------------------------
        self.time_mlp = nn.Sequential(
            nn.Linear(1, cfg['model']['d_model'] // 4),
            nn.SiLU(),
            nn.Linear(cfg['model']['d_model'] // 4,
                      cfg['model']['d_model'])
        )

        # ----------------------------------------------------
        # Cross-Attention Transformer Blocks
        # ----------------------------------------------------
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(
                cfg['model']['d_model'],
                cfg['model']['n_heads'],
                cfg['model']['d_ff'],
                dropout=0.1
            )
            for _ in range(cfg['model']['n_layers'])
        ])

        # ----------------------------------------------------
        # Output Projection
        # ----------------------------------------------------
        self.head = nn.Linear(
            cfg['model']['d_model'],
            cfg['model']['vocab_size']
        )

        total_params = sum(p.numel() for p in self.parameters())
        print(f"✅ Cross-Attention D3PM: {total_params:,} params | MPS Ready")

    # ========================================================
    # 🔹 Forward (Training)
    # ========================================================
    def forward(self, src, tgt, t):
        """
        src: [B, L]
        tgt: [B, L]
        t:   [B] diffusion step
        """

        B, L = tgt.shape

        # ---- Source Embedding ----
        src_emb = self.src_embed(src)

        # ---- Forward Diffusion ----
        x_t_probs, x_t_ids = self.forward_process.q_sample(tgt, t)

        # ---- Target Embedding ----
        tgt_emb = self.tgt_embed(x_t_ids)

        # ---- Time Embedding ----
        t_emb = self.time_mlp(
            t.float().unsqueeze(-1)
        ).unsqueeze(1)  # [B, 1, d_model]

        x = tgt_emb + t_emb.expand(-1, L, -1)

        # ---- Cross-Attention Blocks ----
        for block in self.blocks:
            x = block(x, src_emb)

        logits = self.head(x)

        return logits, t_emb

    # ========================================================
    # 🔹 Reverse Diffusion Generation
    # ========================================================
    @torch.no_grad()
    def generate(self, src, num_steps=8, max_len=None):
        """
        Iterative reverse diffusion generation.
        """

        self.eval()

        device = src.device

        if max_len is None:
            max_len = src.size(1)

        B = src.size(0)

        # ---- Encode Source ----
        src_emb = self.src_embed(src)

        # ---- Initialize with MASK tokens ----
        x_t = torch.full(
            (B, max_len),
            self.mask_token_id,
            dtype=torch.long,
            device=device
        )

        # ---- Reverse Diffusion Loop ----
        for step in reversed(range(num_steps)):

            t = torch.full(
                (B,),
                step,
                dtype=torch.long,
                device=device
            )

            # Embed target
            x_emb = self.tgt_embed(x_t)

            # Add time embedding
            t_emb = self.time_mlp(
                t.float().unsqueeze(-1)
            ).unsqueeze(1)

            x_emb = x_emb + t_emb.expand(-1, max_len, -1)

            # Cross-attention blocks
            for block in self.blocks:
                x_emb = block(x_emb, src_emb)

            # Predict tokens
            logits = self.head(x_emb)
            x0_pred = torch.argmax(logits, dim=-1)

            x_t = x0_pred

        return x_t
    
# ========================================================
# 🔹 BaseLine EncoderDecore
# ========================================================  

class BaselineEncoderDecoder(nn.Module):
    """
    🔹 Pure Transformer Encoder–Decoder
    🔹 No diffusion
    🔹 Direct sequence-to-sequence
    """

    def __init__(self, cfg):
        super().__init__()

        self.src_embed = SanskritEmbeddings(
            cfg['model']['vocab_size'],
            cfg['model']['d_model'],
            cfg['model']['max_seq_len']
        )

        self.tgt_embed = SanskritEmbeddings(
            cfg['model']['vocab_size'],
            cfg['model']['d_model'],
            cfg['model']['max_seq_len']
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg['model']['d_model'],
            nhead=cfg['model']['n_heads'],
            dim_feedforward=cfg['model']['d_ff'],
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, cfg['model']['n_layers'])

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg['model']['d_model'],
            nhead=cfg['model']['n_heads'],
            dim_feedforward=cfg['model']['d_ff'],
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, cfg['model']['n_layers'])

        self.head = nn.Linear(cfg['model']['d_model'], cfg['model']['vocab_size'])

    def forward(self, src, tgt):
        memory = self.encoder(self.src_embed(src))
        output = self.decoder(self.tgt_embed(tgt), memory)
        return self.head(output)
    
# ========================================================
# 🔹 BaseLine Cross - Attention 
# ========================================================  

class BaselineCrossAttention(nn.Module):
    """
    🔹 Cross-Attention Decoder-only style
    🔹 No diffusion
    """

    def __init__(self, cfg):
        super().__init__()

        self.src_embed = SanskritEmbeddings(
            cfg['model']['vocab_size'],
            cfg['model']['d_model'],
            cfg['model']['max_seq_len']
        )

        self.tgt_embed = SanskritEmbeddings(
            cfg['model']['vocab_size'],
            cfg['model']['d_model'],
            cfg['model']['max_seq_len']
        )

        self.blocks = nn.ModuleList([
            CrossAttentionBlock(
                cfg['model']['d_model'],
                cfg['model']['n_heads'],
                cfg['model']['d_ff']
            )
            for _ in range(cfg['model']['n_layers'])
        ])

        self.head = nn.Linear(cfg['model']['d_model'], cfg['model']['vocab_size'])

    def forward(self, src, tgt):
        src_emb = self.src_embed(src)
        x = self.tgt_embed(tgt)

        for block in self.blocks:
            x = block(x, src_emb)

        return self.head(x)