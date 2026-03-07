"""
d3pm_model_cross_attention.py  — Cross-Script + Generation-Fixed
=================================================================
INPUT  : quote_text       tokens  (Roman script, src_vocab_size)
OUTPUT : quote_devanagari tokens  (Devanagari script, tgt_vocab_size)

src_embed  uses src_vocab_size  (Roman BPE)
tgt_embed  uses tgt_vocab_size  (Devanagari BPE)
head       outputs tgt_vocab_size  (predict Devanagari tokens)
Weight tying: head <-> tgt_embed only (NOT src_embed)

Generation bugs fixed:
  BUG 1 - tgt_pad_mask suppressed during inference
  BUG 2 - q_sample skipped at t=0
  BUG 3 - time embedding before hint_gate
  BUG 4 - diversity penalty uses global mean not var
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.scheduler import OptimizedCosineScheduler
from diffusion.forward_process import AbsorbingForwardProcess


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class SanskritEmbeddings(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len):
        super().__init__()
        self.token_emb       = nn.Embedding(vocab_size, d_model)
        self.pos_enc         = SinusoidalPositionalEncoding(d_model, max_seq_len)
        self.token_embedding = self.token_emb
    def forward(self, tokens):
        return self.pos_enc(self.token_emb(tokens))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj   = nn.Linear(d_model, d_model)
        self.k_proj   = nn.Linear(d_model, d_model)
        self.v_proj   = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B, Lq, _ = q.size()
        Lk = k.size(1)
        Q = self.q_proj(q).view(B, Lq, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(k).view(B, Lk, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(v).view(B, Lk, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out  = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        return self.out_proj(out)


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.mha   = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff    = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
                                   nn.Linear(d_ff, d_model), nn.Dropout(dropout))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    def forward(self, x, pad_mask=None):
        x = self.norm1(x + self.mha(x, x, x, mask=pad_mask))
        return self.norm2(x + self.ff(x))


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff         = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
                                        nn.Linear(d_ff, d_model), nn.Dropout(dropout))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
    def forward(self, x, memory, tgt_pad_mask=None, src_pad_mask=None):
        x = self.norm1(x + self.self_attn(x, x, x, mask=tgt_pad_mask))
        x = self.norm2(x + self.cross_attn(x, memory, memory, mask=src_pad_mask))
        return self.norm3(x + self.ff(x))


class D3PMCrossAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg           = cfg
        self.mask_token_id = cfg['diffusion']['mask_token_id']
        d      = cfg['model']['d_model']
        nhead  = cfg['model']['n_heads']
        d_ff   = cfg['model']['d_ff']
        drop   = cfg['model']['dropout']
        seqlen = cfg['model']['max_seq_len']
        nlayer = cfg['model']['n_layers']
        src_vocab = cfg['model'].get('src_vocab_size', cfg['model']['vocab_size'])
        tgt_vocab = cfg['model'].get('tgt_vocab_size', cfg['model']['vocab_size'])

        # Separate embeddings: Roman src, Devanagari tgt
        self.src_embed = SanskritEmbeddings(src_vocab, d, seqlen)
        self.tgt_embed = SanskritEmbeddings(tgt_vocab, d, seqlen)

        self.scheduler       = OptimizedCosineScheduler(cfg)
        self.forward_process = AbsorbingForwardProcess(self.scheduler)

        self.encoder_blocks = nn.ModuleList([EncoderBlock(d, nhead, d_ff, drop) for _ in range(nlayer)])
        self.decoder_blocks = nn.ModuleList([DecoderBlock(d, nhead, d_ff, drop) for _ in range(nlayer)])

        self.time_mlp  = nn.Sequential(nn.Linear(1, d//4), nn.SiLU(), nn.Linear(d//4, d))
        self.hint_gate = nn.Sequential(nn.Linear(d, d), nn.Sigmoid())

        # Output head: predict Devanagari tokens, tied to tgt_embed
        self.head = nn.Linear(d, tgt_vocab, bias=False)
        self.head.weight = self.tgt_embed.token_embedding.weight

    def forward(self, src, tgt, t, x0_hint=None, inference_mode=False):
        PAD = 1
        src_pad_mask = (src == PAD)
        # BUG 1 FIX: no tgt mask during inference
        tgt_pad_mask = None if inference_mode else (tgt == PAD)

        # Encode Roman source
        memory = self.src_embed(src)
        for block in self.encoder_blocks:
            memory = block(memory, pad_mask=src_pad_mask)

        # BUG 2 FIX: skip q_sample at final step t=0
        if inference_mode and (t == 0).all():
            x_t_ids = tgt
        else:
            _, x_t_ids = self.forward_process.q_sample(tgt, t)

        x = self.tgt_embed(x_t_ids)

        # BUG 3 FIX: time embedding BEFORE hint gate
        t_norm = t.float() / self.scheduler.num_timesteps
        t_emb  = self.time_mlp(t_norm.unsqueeze(-1))
        x      = x + t_emb.unsqueeze(1)

        if x0_hint is not None:
            hint_emb = self.tgt_embed(x0_hint)
            gate     = self.hint_gate(x)   # time-aware gate
            x        = x + gate * hint_emb

        for block in self.decoder_blocks:
            x = block(x, memory, tgt_pad_mask=tgt_pad_mask, src_pad_mask=src_pad_mask)

        return self.head(x), None

    @torch.no_grad()
    def generate(self, src, num_steps=None, temperature=0.8, top_k=50,
                 repetition_penalty=1.2, diversity_penalty=0.0):
        if src.dim() == 1:
            src = src.unsqueeze(0)
        device = src.device
        B, L   = src.shape
        T      = self.scheduler.num_timesteps
        steps  = num_steps or T
        step_size = max(1, T // steps)
        timesteps = list(range(T - 1, -1, -step_size))
        if timesteps[-1] != 0:
            timesteps.append(0)

        mask_id = self.mask_token_id
        x0_est  = torch.full((B, L), mask_id, dtype=torch.long, device=device)
        hint    = None

        self.eval()
        with torch.no_grad():
            for step_idx, t_val in enumerate(timesteps):
                t       = torch.full((B,), t_val, dtype=torch.long, device=device)
                is_last = (step_idx == len(timesteps) - 1)
                logits, _ = self.forward(src, x0_est, t, x0_hint=hint, inference_mode=True)
                if repetition_penalty != 1.0:
                    logits = _apply_repetition_penalty(logits, x0_est, repetition_penalty)
                if diversity_penalty > 0.0:
                    logits = _apply_diversity_penalty_fixed(logits, diversity_penalty)  # BUG 4 FIX
                logits = logits / max(temperature, 1e-5)
                if top_k > 0:
                    logits = _top_k_filter(logits, top_k)
                probs = F.softmax(logits, dim=-1)
                x0_est = torch.argmax(probs, dim=-1) if is_last else _batch_multinomial(probs)
                hint = x0_est
        return x0_est


class BaselineCrossAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg['model']['d_model']; nhead = cfg['model']['n_heads']
        d_ff = cfg['model']['d_ff']; drop = cfg['model']['dropout']
        seqlen = cfg['model']['max_seq_len']; nlayer = cfg['model']['n_layers']
        src_vocab = cfg['model'].get('src_vocab_size', cfg['model']['vocab_size'])
        tgt_vocab = cfg['model'].get('tgt_vocab_size', cfg['model']['vocab_size'])
        self.src_embed = SanskritEmbeddings(src_vocab, d, seqlen)
        self.tgt_embed = SanskritEmbeddings(tgt_vocab, d, seqlen)
        self.encoder_blocks = nn.ModuleList([EncoderBlock(d, nhead, d_ff, drop) for _ in range(nlayer)])
        self.decoder_blocks = nn.ModuleList([DecoderBlock(d, nhead, d_ff, drop) for _ in range(nlayer)])
        self.head = nn.Linear(d, tgt_vocab, bias=False)
        self.head.weight = self.tgt_embed.token_embedding.weight

    def forward(self, src, tgt, t=None, x0_hint=None):
        PAD = 1
        memory = self.src_embed(src)
        for b in self.encoder_blocks: memory = b(memory, pad_mask=(src==PAD))
        x = self.tgt_embed(tgt)
        for b in self.decoder_blocks: x = b(x, memory, tgt_pad_mask=(tgt==PAD), src_pad_mask=(src==PAD))
        return (self.head(x),)

    @torch.no_grad()
    def generate(self, src, max_len=None, start_token_id=2, **kwargs):
        if max_len is None: max_len = src.size(1)
        B, device = src.size(0), src.device
        memory = self.src_embed(src)
        for b in self.encoder_blocks: memory = b(memory, pad_mask=(src==1))
        ys = torch.full((B, 1), start_token_id, dtype=torch.long, device=device)
        for _ in range(max_len):
            x = self.tgt_embed(ys)
            for b in self.decoder_blocks: x = b(x, memory, tgt_pad_mask=None, src_pad_mask=(src==1))
            ys = torch.cat([ys, torch.argmax(self.head(x)[:,-1,:], dim=-1, keepdim=True)], dim=1)
        return ys[:, 1:max_len+1]


# helpers
def _top_k_filter(logits, k):
    B, L, V = logits.shape
    if k >= V: return logits
    topk_vals, _ = torch.topk(logits, k, dim=-1)
    return logits.masked_fill(logits < topk_vals[..., -1].unsqueeze(-1), float('-inf'))

def _batch_multinomial(probs):
    B, L, V = probs.shape
    flat = probs.view(B*L, V) + 1e-9
    return torch.multinomial(flat/flat.sum(-1,keepdim=True), 1).squeeze(-1).view(B, L)

def _apply_repetition_penalty(logits, prev_tokens, penalty):
    for b in range(logits.shape[0]):
        for tid in set(prev_tokens[b].tolist()):
            if tid > 4: logits[b, :, tid] = logits[b, :, tid] / penalty
    return logits

def _apply_diversity_penalty(logits, penalty):          # legacy wrong version
    return logits + penalty * logits.var(dim=-1, keepdim=True)

def _apply_diversity_penalty_fixed(logits, penalty):    # correct version
    return logits - penalty * logits.mean(dim=1, keepdim=True)