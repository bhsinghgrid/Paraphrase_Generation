import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.reverse_process import ReverseDiffusion
from diffusion.scheduler import OptimizedCosineScheduler
from diffusion.forward_process import AbsorbingForwardProcess

# ============================================================
# 🔹 Sinusoidal Positional Encoding
# ============================================================
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# ============================================================
# 🔹 Token Embeddings
# ============================================================
class SanskritEmbeddings(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_seq_len)
        self.token_embedding = self.token_emb  # for weight tying

    def forward(self, tokens):
        x = self.token_emb(tokens)
        x = self.pos_enc(x)
        return x

# ============================================================
# 🔹 Multi-Head Attention
# ============================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B, Lq, _ = q.size()
        Lk = k.size(1)

        # Project and reshape for multi-heads
        Q = self.q_proj(q).view(B, Lq, self.n_heads, self.head_dim).transpose(1,2)  # [B, heads, Lq, head_dim]
        K = self.k_proj(k).view(B, Lk, self.n_heads, self.head_dim).transpose(1,2)  # [B, heads, Lk, head_dim]
        V = self.v_proj(v).view(B, Lk, self.n_heads, self.head_dim).transpose(1,2)  # [B, heads, Lk, head_dim]

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2,-1)) / (self.head_dim ** 0.5)  # [B, heads, Lq, Lk]

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,Lk]
            scores = scores.masked_fill(mask, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)  # [B, heads, Lq, head_dim]
        out = out.transpose(1,2).contiguous().view(B, Lq, self.d_model)
        return self.out_proj(out)

# ============================================================
# 🔹 Feed-Forward Layer
# ============================================================
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# ============================================================
# 🔹 Encoder Block
# ============================================================
class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, pad_mask=None):
        x2 = self.mha(x, x, x, mask=pad_mask)
        x = self.norm1(x + x2)
        x2 = self.ff(x)
        x = self.norm2(x + x2)
        return x

# ============================================================
# 🔹 Decoder Block (Baseline & Diffusion Compatible)
# ============================================================
class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, memory=None, tgt_pad_mask=None, src_pad_mask=None):
        # Self-Attention
        x2 = self.self_attn(x, x, x, mask=tgt_pad_mask)
        x = self.norm1(x + x2)
        # Cross-Attention if memory is provided
        if memory is not None:
            x2 = self.cross_attn(x, memory, memory, mask=src_pad_mask)
            x = self.norm2(x + x2)
        # Feed Forward
        x2 = self.ff(x)
        x = self.norm3(x + x2)
        return x

# ============================================================
# 🔹 Baseline Encoder-Decoder (No Diffusion)
# ============================================================
class BaselineEncoderDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_model = cfg['model']['d_model']
        vocab_size = cfg['model']['vocab_size']
        n_layers = cfg['model']['n_layers']
        n_heads = cfg['model']['n_heads']
        d_ff = cfg['model']['d_ff']

        self.src_embed = SanskritEmbeddings(vocab_size, d_model, cfg['model']['max_seq_len'])
        self.tgt_embed = SanskritEmbeddings(vocab_size, d_model, cfg['model']['max_seq_len'])

        self.encoder_blocks = nn.ModuleList([EncoderBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.decoder_blocks = nn.ModuleList([DecoderBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])

        self.head = nn.Linear(d_model, vocab_size)
        self.head.weight = self.tgt_embed.token_embedding.weight  # weight tying

    def forward(self, src, tgt):
        src_pad_mask = (src == 0)
        tgt_pad_mask = (tgt == 0)

        # Encoder
        memory = self.src_embed(src)
        for block in self.encoder_blocks:
            memory = block(memory, pad_mask=src_pad_mask)

        # Decoder
        x = self.tgt_embed(tgt)
        for block in self.decoder_blocks:
            x = block(x, memory, tgt_pad_mask=tgt_pad_mask, src_pad_mask=src_pad_mask)

        return self.head(x)

    @torch.no_grad()
    def generate(self, src, max_len=50, start_token_id=2):
        batch_size = src.size(0)
        device = src.device
        src_pad_mask = (src == 0)

        memory = self.src_embed(src)
        for block in self.encoder_blocks:
            memory = block(memory, pad_mask=src_pad_mask)

        ys = torch.ones(batch_size, 1, dtype=torch.long, device=device) * start_token_id
        for _ in range(max_len):
            x = self.tgt_embed(ys)
            for block in self.decoder_blocks:
                x = block(x, memory, tgt_pad_mask=None, src_pad_mask=src_pad_mask)
            logits = self.head(x)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)
        return ys[:, 1:]

# ============================================================
# 🔹 D3PM Encoder-Decoder (With Diffusion)
# ============================================================
class D3PMEncoderDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.mask_token_id = cfg['diffusion']['mask_token_id']

        self.src_embed = SanskritEmbeddings(cfg['model']['vocab_size'], cfg['model']['d_model'], cfg['model']['max_seq_len'])
        self.tgt_embed = SanskritEmbeddings(cfg['model']['vocab_size'], cfg['model']['d_model'], cfg['model']['max_seq_len'])

        self.scheduler = OptimizedCosineScheduler(cfg)
        self.forward_process = AbsorbingForwardProcess(self.scheduler)

        self.encoder_blocks = nn.ModuleList([EncoderBlock(cfg['model']['d_model'], cfg['model']['n_heads'], cfg['model']['d_ff']) for _ in range(cfg['model']['n_layers'])])
        self.decoder_blocks = nn.ModuleList([DecoderBlock(cfg['model']['d_model'], cfg['model']['n_heads'], cfg['model']['d_ff']) for _ in range(cfg['model']['n_layers'])])

        self.time_mlp = nn.Sequential(nn.Linear(1, cfg['model']['d_model']//4), nn.SiLU(), nn.Linear(cfg['model']['d_model']//4, cfg['model']['d_model']))
        self.head = nn.Linear(cfg['model']['d_model'], cfg['model']['vocab_size'])
        self.head.weight = self.tgt_embed.token_embedding.weight

    def forward(self, src, tgt, t):
        src_pad_mask = (src == 0)
        tgt_pad_mask = (tgt == 0)

        # Encoder
        memory = self.src_embed(src)
        for block in self.encoder_blocks:
            memory = block(memory, pad_mask=src_pad_mask)

        # Forward diffusion sample
        x_t_probs, x_t_ids = self.forward_process.q_sample(tgt, t)
        x = self.tgt_embed(x_t_ids)

        # Time embedding
        t_emb = self.time_mlp(t.float().unsqueeze(-1)).unsqueeze(1)
        x = x + t_emb.expand(-1, tgt.shape[1], -1)

        # Decoder
        for block in self.decoder_blocks:
            x = block(x, memory, tgt_pad_mask=tgt_pad_mask, src_pad_mask=src_pad_mask)

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