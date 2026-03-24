import torch
import torch.nn as nn
from diffusion.scheduler import OptimizedCosineScheduler
from diffusion.forward_process import AbsorbingForwardProcess
# Import shared classes to guarantee identical architectures
from model.d3pm_model_cross_attention import SanskritEmbeddings, EncoderBlock, MultiHeadAttention
class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.15):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)  # ← restored
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)  # ← restored (for cross-attn residual)

    def forward(self, x, memory, tgt_pad_mask=None):
        # 1. Masked self-attention on target
        x = self.norm1(x + self.self_attn(x, x, x, mask=tgt_pad_mask))
        # 2. Cross-attention: queries from decoder, keys/values from encoder memory
        x = self.norm2(x + self.cross_attn(x, memory, memory))
        # 3. Feed-forward
        return self.norm3(x + self.ff(x))


class DecoderBlockNoCrossAttn(nn.Module):
    """Kept for reference — NOT used by D3PMEncoderDecoder."""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.15):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)

    def forward(self, x, tgt_pad_mask=None, causal_mask=None):
        combined_mask = None
        if tgt_pad_mask is not None and causal_mask is not None:
            combined_mask = tgt_pad_mask | causal_mask
        elif causal_mask is not None:
            combined_mask = causal_mask
        elif tgt_pad_mask is not None:
            combined_mask = tgt_pad_mask
        x = self.norm1(x + self.self_attn(x, x, x, mask=combined_mask))
        return self.norm2(x + self.ff(x))


# ============================================================
# 1. D3PM Encoder-Decoder Model
# ============================================================
class D3PMEncoderDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg           = cfg
        self.mask_token_id = cfg['diffusion']['mask_token_id']

        src_vocab = cfg['model'].get('src_vocab_size', cfg['model']['vocab_size'])
        tgt_vocab = cfg['model'].get('tgt_vocab_size', cfg['model']['vocab_size'])
        d_model   = cfg['model']['d_model']
        n_heads   = cfg['model']['n_heads']
        d_ff      = cfg['model']['d_ff']
        dropout   = cfg['model']['dropout']
        n_layers  = cfg['model']['n_layers']
        max_len   = cfg['model']['max_seq_len']

        self.src_embed = SanskritEmbeddings(src_vocab, d_model, max_len)
        self.tgt_embed = SanskritEmbeddings(tgt_vocab, d_model, max_len)

        self.scheduler       = OptimizedCosineScheduler(cfg)
        self.forward_process = AbsorbingForwardProcess(self.scheduler)

        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        # DecoderBlock now has cross-attention — matches saved checkpoint
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])

        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_model // 4), nn.SiLU(),
            nn.Linear(d_model // 4, d_model),
        )
        self.head        = nn.Linear(d_model, tgt_vocab)
        self.head.weight = self.tgt_embed.token_embedding.weight

    def forward(self, src, tgt, t, x0_hint=None):
        src_pad_mask = (src == 1)
        tgt_pad_mask = (tgt == 1)

        # Encode source (Roman IAST)
        memory = self.src_embed(src)
        for block in self.encoder_blocks:
            memory = block(memory, pad_mask=src_pad_mask)

        # Corrupt target with forward diffusion
        _, x_t_ids = self.forward_process.q_sample(tgt, t)

        # Optionally blend in x0_hint (self-conditioning)
        if x0_hint is not None:
            hint_prob  = 0.5
            blend_mask = (torch.rand(x_t_ids.shape, device=x_t_ids.device) < hint_prob)
            still_mask = (x_t_ids == self.mask_token_id)
            x_t_ids    = torch.where(blend_mask & still_mask, x0_hint, x_t_ids)

        x     = self.tgt_embed(x_t_ids)
        t_emb = self.time_mlp(t.float().unsqueeze(-1)).unsqueeze(1)
        x     = x + t_emb.expand(-1, tgt.shape[1], -1)

        # Decode with cross-attention over encoder memory
        for block in self.decoder_blocks:
            x = block(x, memory, tgt_pad_mask=tgt_pad_mask)

        return self.head(x), None

    @torch.no_grad()
    def generate(
        self,
        src,
        num_steps          = None,
        temperature        = 0.75,
        top_k              = 50,
        repetition_penalty = 1.15,
        diversity_penalty  = 0.0,
    ):
        """
        Iterative D3PM reverse diffusion — same signature as
        D3PMCrossAttention.generate() so SanskritModel.generate() works
        identically for both model types.
        """
        device   = src.device
        B, L     = src.shape[0], self.cfg['model']['max_seq_len']
        T        = num_steps or self.scheduler.num_timesteps
        mask_id  = self.mask_token_id
        pad_id   = 1

        x0_est = torch.full((B, L), mask_id, dtype=torch.long, device=device)

        for step in range(T - 1, -1, -1):
            t_tensor = torch.full((B,), step, dtype=torch.long, device=device)
            hint     = x0_est.clone()

            logits, _ = self.forward(src, x0_est, t_tensor, x0_hint=hint)

            # Repetition penalty
            if repetition_penalty != 1.0:
                for b in range(B):
                    for tok in set(x0_est[b].tolist()):
                        if tok > pad_id:
                            logits[b, :, tok] /= repetition_penalty

            # Diversity penalty (suppress common tokens)
            if diversity_penalty > 0.0:
                logits = logits - diversity_penalty * logits.mean(dim=1, keepdim=True)

            # Temperature + top-k sampling
            logits = logits / max(temperature, 1e-8)
            if top_k > 0:
                vals, _ = torch.topk(logits, top_k, dim=-1)
                logits  = logits.masked_fill(logits < vals[..., -1:], float('-inf'))

            probs  = torch.softmax(logits, dim=-1)
            # Only update positions that are still masked
            still  = (x0_est == mask_id)
            sample = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(B, L)
            x0_est = torch.where(still, sample, x0_est)

        return x0_est


# ============================================================
# 2. Baseline Encoder-Decoder Model (unchanged)
# ============================================================
class BaselineEncoderDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg       = cfg
        self.src_embed = SanskritEmbeddings(cfg['model']['vocab_size'], cfg['model']['d_model'],
                                            cfg['model']['max_seq_len'])
        self.tgt_embed = SanskritEmbeddings(cfg['model']['vocab_size'], cfg['model']['d_model'],
                                            cfg['model']['max_seq_len'])
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(cfg['model']['d_model'], cfg['model']['n_heads'],
                         cfg['model']['d_ff'], cfg['model']['dropout'])
            for _ in range(cfg['model']['n_layers'])
        ])
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(cfg['model']['d_model'], cfg['model']['n_heads'],
                         cfg['model']['d_ff'], cfg['model']['dropout'])
            for _ in range(cfg['model']['n_layers'])
        ])
        self.head        = nn.Linear(cfg['model']['d_model'], cfg['model']['vocab_size'])
        self.head.weight = self.tgt_embed.token_embedding.weight

    def forward(self, src, tgt):
        src_pad_mask, tgt_pad_mask = (src == 1), (tgt == 1)
        memory = self.src_embed(src)
        for block in self.encoder_blocks:
            memory = block(memory, pad_mask=src_pad_mask)
        x = self.tgt_embed(tgt)
        for block in self.decoder_blocks:
            x = block(x, memory, tgt_pad_mask=tgt_pad_mask)
        return self.head(x)

    @torch.no_grad()
    def generate(self, src, max_len=80, start_token_id=2):
        batch_size, device = src.size(0), src.device
        src_pad_mask = (src == 1)
        memory = self.src_embed(src)
        for block in self.encoder_blocks:
            memory = block(memory, pad_mask=src_pad_mask)
        ys = torch.ones(batch_size, 1, dtype=torch.long, device=device) * start_token_id
        for _ in range(max_len):
            x = self.tgt_embed(ys)
            for block in self.decoder_blocks:
                x = block(x, memory, tgt_pad_mask=None)
            logits     = self.head(x)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            ys         = torch.cat([ys, next_token], dim=1)
        return ys[:, 1:]