import torch
import torch.nn as nn
from diffusion.scheduler import OptimizedCosineScheduler
from diffusion.forward_process import AbsorbingForwardProcess
from diffusion.reverse_process import ReverseDiffusion # 🔥 FIXED IMPORT

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

class SanskritEmbeddings(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_seq_len)
        self.token_embedding = self.token_emb

    def forward(self, tokens):
        return self.pos_enc(self.token_emb(tokens))

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.15):
        super().__init__()
        self.d_model, self.n_heads, self.head_dim = d_model, n_heads, d_model // n_heads
        self.q_proj, self.k_proj, self.v_proj = nn.Linear(d_model, d_model), nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

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

class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.15):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model), nn.Dropout(dropout))
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)

    def forward(self, x, pad_mask=None):
        x = self.norm1(x + self.mha(x, x, x, mask=pad_mask))
        return self.norm2(x + self.ff(x))

class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.15):
        super().__init__()
        self.self_attn, self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout), MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model), nn.Dropout(dropout))
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)

    def forward(self, x, memory, tgt_pad_mask=None, src_pad_mask=None):
        x = self.norm1(x + self.self_attn(x, x, x, mask=tgt_pad_mask))
        x = self.norm2(x + self.cross_attn(x, memory, memory, mask=src_pad_mask))
        return self.norm3(x + self.ff(x))

# ============================================================
# 🔥 1. D3PM Cross-Attention Model
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
        self.encoder_blocks = nn.ModuleList([EncoderBlock(cfg['model']['d_model'], cfg['model']['n_heads'], cfg['model']['d_ff'], cfg['model']['dropout']) for _ in range(cfg['model']['n_layers'])])
        self.decoder_blocks = nn.ModuleList([DecoderBlock(cfg['model']['d_model'], cfg['model']['n_heads'], cfg['model']['d_ff'], cfg['model']['dropout']) for _ in range(cfg['model']['n_layers'])])
        self.time_mlp = nn.Sequential(nn.Linear(1, cfg['model']['d_model']//4), nn.SiLU(), nn.Linear(cfg['model']['d_model']//4, cfg['model']['d_model']))
        self.head = nn.Linear(cfg['model']['d_model'], cfg['model']['vocab_size'])
        self.head.weight = self.tgt_embed.token_embedding.weight

        self.hint_gate = nn.Sequential(
            nn.Linear(cfg['model']['d_model'], cfg['model']['d_model']),
            nn.Sigmoid()
        )

    # def forward(self, src, tgt, t, x0_hint=None):
    #     # 1. Setup masks
    #     src_pad_mask, tgt_pad_mask = (src == 1), (tgt == 1)
    #
    #     # 2. Encoder Path (Conditioning)
    #     memory = self.src_embed(src)
    #     for block in self.encoder_blocks:
    #         memory = block(memory, pad_mask=src_pad_mask)
    #
    #     # 3. Diffusion Forward Process (Add noise to target)
    #     _, x_t_ids = self.forward_process.q_sample(tgt, t)
    #
    #     # 4. Target Embedding
    #     x = self.tgt_embed(x_t_ids)
    #
    #     # 🔥 SELF-CONDITIONING LOGIC
    #     # If a hint is provided, embed it and add it to the noisy target embedding
    #     if x0_hint is not None:
    #         # Note: We use tgt_embed because x0_hint is in the target language
    #         hint_emb = self.tgt_embed(x0_hint)
    #         x = x + hint_emb  # Merge the noisy input with the model's previous guess
    #
    #     # 5. Time Embedding
    #     t_emb = self.time_mlp(t.float().unsqueeze(-1)).unsqueeze(1)
    #     x = x + t_emb.expand(-1, tgt.shape[1], -1)
    #
    #     # 6. Decoder Path (Denoising)
    #     for block in self.decoder_blocks:
    #         x = block(x, memory, tgt_pad_mask=tgt_pad_mask, src_pad_mask=src_pad_mask)
    #
    #     return self.head(x), None
    def forward(self, src, tgt, t, x0_hint=None):
        src_pad_mask, tgt_pad_mask = (src == 1), (tgt == 1)

        # 1. Encoder Path
        memory = self.src_embed(src)
        for block in self.encoder_blocks:
            memory = block(memory, pad_mask=src_pad_mask)

        # 2. Diffusion Process
        _, x_t_ids = self.forward_process.q_sample(tgt, t)
        x = self.tgt_embed(x_t_ids)

        # 3. 🔥 FIXED SELF-CONDITIONING (Gated approach)
        if x0_hint is not None:
            hint_emb = self.tgt_embed(x0_hint)
            # Instead of raw addition, we "gate" it.
            # This allows the model to gradually trust the hint more as 't' decreases.
            gate = self.hint_gate(x)
            x = x + (gate * hint_emb)

            # 4. Time Embedding
        t_emb = self.time_mlp(t.float().unsqueeze(-1)).unsqueeze(1)
        x = x + t_emb.expand(-1, tgt.shape[1], -1)

        # 5. Decoder Path (Strict masking for precision)
        for block in self.decoder_blocks:
            x = block(x, memory, tgt_pad_mask=tgt_pad_mask, src_pad_mask=src_pad_mask)

        return self.head(x), None

    # 🔥 ADDED GENERATE METHOD TO LINK TO YOUR REVERSE DIFFUSION
    @torch.no_grad()
    def generate(self, src, num_steps=None, beam_width=3):
        reverse_diffusion = ReverseDiffusion(self.scheduler)
        return reverse_diffusion.generate_beam(
            model=self,
            condition=src,
            beam_width=beam_width,
            num_steps=num_steps or self.scheduler.num_timesteps
        )

# ============================================================
# 🔹 2. Baseline Cross-Attention Model (Standard Auto-Regressive)
# ============================================================
# class BaselineCrossAttention(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.cfg = cfg
#         self.src_embed = SanskritEmbeddings(cfg['model']['vocab_size'], cfg['model']['d_model'], cfg['model']['max_seq_len'])
#         self.tgt_embed = SanskritEmbeddings(cfg['model']['vocab_size'], cfg['model']['d_model'], cfg['model']['max_seq_len'])
#         self.encoder_blocks = nn.ModuleList([EncoderBlock(cfg['model']['d_model'], cfg['model']['n_heads'], cfg['model']['d_ff'], cfg['model']['dropout']) for _ in range(cfg['model']['n_layers'])])
#         self.decoder_blocks = nn.ModuleList([DecoderBlock(cfg['model']['d_model'], cfg['model']['n_heads'], cfg['model']['d_ff'], cfg['model']['dropout']) for _ in range(cfg['model']['n_layers'])])
#         self.head = nn.Linear(cfg['model']['d_model'], cfg['model']['vocab_size'])
#         self.head.weight = self.tgt_embed.token_embedding.weight
#
#     def forward(self, src, tgt):
#         src_pad_mask, tgt_pad_mask = (src == 1), (tgt == 1)
#         memory = self.src_embed(src)
#         for block in self.encoder_blocks: memory = block(memory, pad_mask=src_pad_mask)
#         x = self.tgt_embed(tgt)
#         for block in self.decoder_blocks: x = block(x, memory, tgt_pad_mask=tgt_pad_mask, src_pad_mask=src_pad_mask)
#         return self.head(x)
#
#     @torch.no_grad()
#     def generate(self, src, max_len=80, start_token_id=2):
#         batch_size, device = src.size(0), src.device
#         src_pad_mask = (src == 1)
#         memory = self.src_embed(src)
#         for block in self.encoder_blocks: memory = block(memory, pad_mask=src_pad_mask)
#
#         ys = torch.ones(batch_size, 1, dtype=torch.long, device=device) * start_token_id
#         for _ in range(max_len):
#             x = self.tgt_embed(ys)
#             for block in self.decoder_blocks: x = block(x, memory, tgt_pad_mask=None, src_pad_mask=src_pad_mask)
#             logits = self.head(x)
#             next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
#             ys = torch.cat([ys, next_token], dim=1)
#         return ys[:, 1:]

class BaselineCrossAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.src_embed = SanskritEmbeddings(cfg['model']['vocab_size'], cfg['model']['d_model'],
                                            cfg['model']['max_seq_len'])
        self.tgt_embed = SanskritEmbeddings(cfg['model']['vocab_size'], cfg['model']['d_model'],
                                            cfg['model']['max_seq_len'])
        self.encoder_blocks = nn.ModuleList([EncoderBlock(cfg['model']['d_model'], cfg['model']['n_heads'],
                                                          cfg['model']['d_ff'], cfg['model']['dropout']) for _ in
                                             range(cfg['model']['n_layers'])])
        self.decoder_blocks = nn.ModuleList([DecoderBlock(cfg['model']['d_model'], cfg['model']['n_heads'],
                                                          cfg['model']['d_ff'], cfg['model']['dropout']) for _ in
                                             range(cfg['model']['n_layers'])])
        self.head = nn.Linear(cfg['model']['d_model'], cfg['model']['vocab_size'])
        self.head.weight = self.tgt_embed.token_embedding.weight

    def forward(self, src, tgt):
        # 1. Padding Masks
        src_pad_mask, tgt_pad_mask = (src == 1), (tgt == 1)

        # 2. 🔥 Generate Causal Mask for the Decoder (prevents looking into the future)
        seq_len = tgt.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=tgt.device), diagonal=1).bool()

        # 3. Encoder Pass
        memory = self.src_embed(src)
        for block in self.encoder_blocks:
            memory = block(memory, pad_mask=src_pad_mask)

        # 4. Decoder Pass
        x = self.tgt_embed(tgt)
        for block in self.decoder_blocks:
            # Pass BOTH the padding mask and the causal mask to the decoder
            x = block(x, memory, tgt_pad_mask=tgt_pad_mask, src_pad_mask=src_pad_mask, causal_mask=causal_mask)

        # 5. 🔥 Return as a tuple so train.py's [0] indexing works perfectly!
        return (self.head(x),)

    @torch.no_grad()
    def generate(self, src, max_len=80, start_token_id=2):
        batch_size, device = src.size(0), src.device
        src_pad_mask = (src == 1)

        # 1. Encode the source text once
        memory = self.src_embed(src)
        for block in self.encoder_blocks:
            memory = block(memory, pad_mask=src_pad_mask)

        # 2. Autoregressive loop
        ys = torch.ones(batch_size, 1, dtype=torch.long, device=device) * start_token_id
        for _ in range(max_len):
            # Create a causal mask for the current generation length
            seq_len = ys.size(1)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

            x = self.tgt_embed(ys)
            for block in self.decoder_blocks:
                x = block(x, memory, tgt_pad_mask=None, src_pad_mask=src_pad_mask, causal_mask=causal_mask)

            logits = self.head(x)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)

        return ys[:, 1:]