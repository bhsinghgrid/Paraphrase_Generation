import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion.reverse_process import ReverseDiffusion
from diffusion.scheduler import OptimizedCosineScheduler
from diffusion.forward_process import AbsorbingForwardProcess

# ============================================================
# 🔹 Sanskrit Embeddings (Token + Sinusoidal Positional)
# ============================================================
class SanskritEmbeddings(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        nn.init.xavier_uniform_(self.pos_enc)
        self.token_embedding = self.token_emb  # for weight tying

    def forward(self, tokens):
        x = self.token_emb(tokens) + self.pos_enc[:, :tokens.size(1), :]
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

        Q = self.q_proj(q).view(B, Lq, self.n_heads, self.head_dim).transpose(1,2)
        K = self.k_proj(k).view(B, Lk, self.n_heads, self.head_dim).transpose(1,2)
        V = self.v_proj(v).view(B, Lk, self.n_heads, self.head_dim).transpose(1,2)

        scores = torch.matmul(Q, K.transpose(-2,-1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V).transpose(1,2).contiguous().view(B, Lq, self.d_model)
        return self.out_proj(out)

# ============================================================
# 🔹 Feed Forward
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
# 🔹 Decoder Block with Cross-Attention
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

    def forward(self, x, memory, tgt_pad_mask=None, src_pad_mask=None):
        # Self-Attention
        x2 = self.self_attn(x, x, x, mask=tgt_pad_mask)
        x = self.norm1(x + x2)
        # Cross-Attention
        x2 = self.cross_attn(x, memory, memory, mask=src_pad_mask)
        x = self.norm2(x + x2)
        # Feed Forward
        x2 = self.ff(x)
        x = self.norm3(x + x2)
        return x

# ============================================================
# 🔹 D3PM Encoder–Decoder with Cross-Attention
# ============================================================
class D3PMCrossAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.mask_token_id = cfg['diffusion']['mask_token_id']

        self.src_embed = SanskritEmbeddings(cfg['model']['vocab_size'], cfg['model']['d_model'], cfg['model']['max_seq_len'])
        self.tgt_embed = SanskritEmbeddings(cfg['model']['vocab_size'], cfg['model']['d_model'], cfg['model']['max_seq_len'])

        self.scheduler = OptimizedCosineScheduler(cfg)
        self.forward_process = AbsorbingForwardProcess(self.scheduler)

        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(cfg['model']['d_model'], cfg['model']['n_heads'], cfg['model']['d_ff'])
            for _ in range(cfg['model']['n_layers'])
        ])

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(cfg['model']['d_model'], cfg['model']['n_heads'], cfg['model']['d_ff'])
            for _ in range(cfg['model']['n_layers'])
        ])

        self.time_mlp = nn.Sequential(
            nn.Linear(1, cfg['model']['d_model']//4),
            nn.SiLU(),
            nn.Linear(cfg['model']['d_model']//4, cfg['model']['d_model'])
        )

        self.head = nn.Linear(cfg['model']['d_model'], cfg['model']['vocab_size'])
        self.head.weight = self.tgt_embed.token_embedding.weight  # weight tying

    def forward(self, src, tgt, t):
        src_pad_mask = (src == 0)
        tgt_pad_mask = (tgt == 0)

        memory = self.src_embed(src)
        for block in self.encoder_blocks:
            memory = block(memory, pad_mask=src_pad_mask)

        x_t_probs, x_t_ids = self.forward_process.q_sample(tgt, t)
        x = self.tgt_embed(x_t_ids)

        t_emb = self.time_mlp(t.float().unsqueeze(-1)).unsqueeze(1)
        x = x + t_emb.expand(-1, tgt.shape[1], -1)

        for block in self.decoder_blocks:
            x = block(x, memory, tgt_pad_mask=tgt_pad_mask, src_pad_mask=src_pad_mask)

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
# 🔹 Baseline Cross-Attention (No Diffusion)
# ============================================================
class BaselineCrossAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d_model = cfg['model']['d_model']
        vocab_size = cfg['model']['vocab_size']
        n_layers = cfg['model']['n_layers']
        n_heads = cfg['model']['n_heads']
        d_ff = cfg['model']['d_ff']

        # Embeddings
        self.src_embed = SanskritEmbeddings(vocab_size, d_model, cfg['model']['max_seq_len'])
        self.tgt_embed = SanskritEmbeddings(vocab_size, d_model, cfg['model']['max_seq_len'])

        # Encoder blocks
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])

        # Decoder blocks
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])

        # Output head with weight tying
        self.head = nn.Linear(d_model, vocab_size)
        self.head.weight = self.tgt_embed.token_embedding.weight

    def forward(self, src, tgt):
        # src, tgt: [B, L] LongTensor
        src_pad_mask = (src == 0)  # assuming 0 is pad token
        tgt_pad_mask = (tgt == 0)

        # Encoder
        memory = self.src_embed(src)
        for block in self.encoder_blocks:
            memory = block(memory, pad_mask=src_pad_mask)

        # Decoder
        x = self.tgt_embed(tgt)
        for block in self.decoder_blocks:
            x = block(x, memory, tgt_pad_mask=tgt_pad_mask, src_pad_mask=src_pad_mask)

        # Output logits
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def generate(self, src, max_len=50, start_token_id=2):
        """
        Greedy generation.
        src: [B, L] tensor
        returns: [B, generated_seq_len] tensor
        """
        batch_size = src.size(0)
        device = src.device
        src_pad_mask = (src == 0)

        memory = self.src_embed(src)
        for block in self.encoder_blocks:
            memory = block(memory, pad_mask=src_pad_mask)

        # Initialize decoder input with start token
        ys = torch.ones(batch_size, 1, dtype=torch.long, device=device) * start_token_id

        for _ in range(max_len):
            x = self.tgt_embed(ys)
            for block in self.decoder_blocks:
                x = block(x, memory, tgt_pad_mask=None, src_pad_mask=src_pad_mask)
            logits = self.head(x)  # [B, seq_len, vocab_size]
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)

            # Optional: break if all sequences generate EOS token
            # if (next_token == eos_token_id).all():
            #     break

        return ys[:, 1:]  # exclude start token