import torch
import torch.nn as nn
from diffusion.scheduler import OptimizedCosineScheduler
from diffusion.forward_process import AbsorbingForwardProcess
from diffusion.reverse_process import ReverseDiffusion  # 🔥 FIXED IMPORT

# Import shared classes to guarantee identical architectures
from model.d3pm_model_cross_attention import SanskritEmbeddings, EncoderBlock, MultiHeadAttention


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.15):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, tgt_pad_mask=None):
        # Position 1: x
        # Position 2: tgt_pad_mask
        x = self.norm1(x + self.self_attn(x, x, x, mask=tgt_pad_mask))
        return self.norm2(x + self.ff(x))

class DecoderBlockNoCrossAttn(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.15):
        super().__init__()
        # ONLY Self-Attention!
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model), nn.Dropout(dropout))
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)

    def forward(self, x, tgt_pad_mask=None, causal_mask=None):
        # Combine the masks
        combined_mask = None
        if tgt_pad_mask is not None and causal_mask is not None:
            combined_mask = tgt_pad_mask | causal_mask
        elif causal_mask is not None:
            combined_mask = causal_mask
        elif tgt_pad_mask is not None:
            combined_mask = tgt_pad_mask

        # Self Attention Only (No Cross Attention step!)
        x = self.norm1(x + self.self_attn(x, x, x, mask=combined_mask))
        return self.norm2(x + self.ff(x))
# ============================================================
# 🔥 1. D3PM Encoder-Decoder Model
# ============================================================
class D3PMEncoderDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.mask_token_id = cfg['diffusion']['mask_token_id']
        self.src_embed = SanskritEmbeddings(cfg['model']['vocab_size'], cfg['model']['d_model'],
                                            cfg['model']['max_seq_len'])
        self.tgt_embed = SanskritEmbeddings(cfg['model']['vocab_size'], cfg['model']['d_model'],
                                            cfg['model']['max_seq_len'])
        self.scheduler = OptimizedCosineScheduler(cfg)
        self.forward_process = AbsorbingForwardProcess(self.scheduler)
        self.encoder_blocks = nn.ModuleList([EncoderBlock(cfg['model']['d_model'], cfg['model']['n_heads'],
                                                          cfg['model']['d_ff'], cfg['model']['dropout']) for _ in
                                             range(cfg['model']['n_layers'])])
        self.decoder_blocks = nn.ModuleList([DecoderBlock(cfg['model']['d_model'], cfg['model']['n_heads'],
                                                          cfg['model']['d_ff'], cfg['model']['dropout']) for _ in
                                             range(cfg['model']['n_layers'])])
        self.time_mlp = nn.Sequential(nn.Linear(1, cfg['model']['d_model'] // 4), nn.SiLU(),
                                      nn.Linear(cfg['model']['d_model'] // 4, cfg['model']['d_model']))
        self.head = nn.Linear(cfg['model']['d_model'], cfg['model']['vocab_size'])
        self.head.weight = self.tgt_embed.token_embedding.weight

    def forward(self, src, tgt, t):
        src_pad_mask, tgt_pad_mask = (src == 1), (tgt == 1)
        memory = self.src_embed(src)
        for block in self.encoder_blocks: memory = block(memory, pad_mask=src_pad_mask)

        _, x_t_ids = self.forward_process.q_sample(tgt, t)
        x = self.tgt_embed(x_t_ids)
        t_emb = self.time_mlp(t.float().unsqueeze(-1)).unsqueeze(1)
        x = x + t_emb.expand(-1, tgt.shape[1], -1)

        # Omit src_pad_mask in vanilla Enc-Dec to force reliance on pooled memory
        for block in self.decoder_blocks: x = block(x, memory, tgt_pad_mask=tgt_pad_mask)
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
# 🔹 2. Baseline Encoder-Decoder Model
# ============================================================
class BaselineEncoderDecoder(nn.Module):
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
        src_pad_mask, tgt_pad_mask = (src == 1), (tgt == 1)
        memory = self.src_embed(src)
        for block in self.encoder_blocks: memory = block(memory, pad_mask=src_pad_mask)

        x = self.tgt_embed(tgt)
        for block in self.decoder_blocks: x = block(x, memory, tgt_pad_mask=tgt_pad_mask)
        return self.head(x)

    @torch.no_grad()
    def generate(self, src, max_len=80, start_token_id=2):
        batch_size, device = src.size(0), src.device
        src_pad_mask = (src == 1)
        memory = self.src_embed(src)
        for block in self.encoder_blocks: memory = block(memory, pad_mask=src_pad_mask)

        ys = torch.ones(batch_size, 1, dtype=torch.long, device=device) * start_token_id
        for _ in range(max_len):
            x = self.tgt_embed(ys)
            for block in self.decoder_blocks: x = block(x, memory, tgt_pad_mask=None)
            logits = self.head(x)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)
        return ys[:, 1:]