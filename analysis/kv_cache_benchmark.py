import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 🔧 MODEL (PATCHED WITH PROJECTION + KV CACHE)
# ============================================================

class D3PMCrossAttention(nn.Module):
    def __init__(self, d_model=512, vocab_size=500, max_seq_len=64, T=128):
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.mask_token_id = 0

        # Dummy encoder/decoder (replace with yours)
        self.encoder = nn.Embedding(vocab_size, d_model)
        self.tgt_embed = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, vocab_size)

        self.time_mlp = nn.Linear(1, d_model)
        self.hint_gate = nn.Linear(d_model, d_model)

        # Fake scheduler
        class Scheduler:
            def __init__(self, T):
                self.num_timesteps = T
        self.scheduler = Scheduler(T)

        # 🔥 Projection layer (Task 1 requirement)
        self.semantic_proj = nn.Linear(d_model, d_model // 2)
        self.semantic_up   = nn.Linear(d_model // 2, d_model)

    # ========================================================
    # ✅ ENCODER WITH PROJECTION
    # ========================================================
    def encode_source(self, src):
        memory = self.encoder(src)   # [B, L, d]

        # 🔥 Compress → Expand
        compressed = self.semantic_proj(memory)
        memory     = self.semantic_up(compressed)

        src_pad_mask = None
        return memory, src_pad_mask

    # ========================================================
    # ✅ STANDARD (NO CACHE)
    # ========================================================
    def forward(self, src, x, t):
        memory, mask = self.encode_source(src)
        return self.forward_cached(memory, mask, x, t)

    # ========================================================
    # ✅ CACHED FORWARD
    # ========================================================
    def forward_cached(self, memory, src_pad_mask, x, t, hint=None):
        x = self.tgt_embed(x)

        t_emb = self.time_mlp((t.float()/self.scheduler.num_timesteps).unsqueeze(-1))
        x = x + t_emb.unsqueeze(1)

        if hint is not None:
            x = x + self.hint_gate(x) * self.tgt_embed(hint)

        logits = self.head(x)

        self._last_hidden = x
        return logits, None

    # ========================================================
    # ❌ OLD GENERATE (SLOW)
    # ========================================================
    @torch.no_grad()
    def generate(self, src):
        B = src.shape[0]
        device = src.device
        T = self.scheduler.num_timesteps

        x = torch.zeros((B, self.max_seq_len), dtype=torch.long, device=device)

        for t_val in range(T - 1, -1, -1):
            t = torch.full((B,), t_val, device=device)

            logits, _ = self.forward(src, x, t)
            probs = F.softmax(logits, dim=-1)

            x = torch.argmax(probs, dim=-1)

        return x

    # ========================================================
    # ✅ FAST GENERATE (KV CACHE)
    # ========================================================
    @torch.no_grad()
    def generate_cached(self, src):
        B = src.shape[0]
        device = src.device
        T = self.scheduler.num_timesteps

        # 🔥 Encode once
        memory, mask = self.encode_source(src)

        x = torch.zeros((B, self.max_seq_len), dtype=torch.long, device=device)
        hint = None

        for t_val in range(T - 1, -1, -1):
            t = torch.full((B,), t_val, device=device)

            logits, _ = self.forward_cached(memory, mask, x, t, hint)
            probs = F.softmax(logits, dim=-1)

            x = torch.argmax(probs, dim=-1)
            hint = x

        return x


# ============================================================
# 📊 BENCHMARK + MEMORY + GRAPHS
# ============================================================

def benchmark(model, device):
    model.to(device)
    model.eval()

    vocab = 500
    src_lens = [16, 32, 64]

    standard_times = []
    cached_times   = []
    speedups       = []
    memory_savings = []

    for src_len in src_lens:
        print(f"\n🔹 src_len = {src_len}")

        src = torch.randint(5, vocab, (1, src_len)).to(device)

        # -------- STANDARD --------
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        model.generate(src)
        torch.cuda.synchronize()
        t_std = time.time() - start
        mem_std = torch.cuda.max_memory_allocated() / 1024**2

        # -------- CACHED --------
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        model.generate_cached(src)
        torch.cuda.synchronize()
        t_cache = time.time() - start
        mem_cache = torch.cuda.max_memory_allocated() / 1024**2

        speedup = t_std / t_cache
        mem_red = 100 * (mem_std - mem_cache) / mem_std

        print(f"Time: {t_std:.2f}s → {t_cache:.2f}s  |  {speedup:.2f}x")
        print(f"Memory: {mem_std:.0f}MB → {mem_cache:.0f}MB  |  {mem_red:.1f}%")

        standard_times.append(t_std)
        cached_times.append(t_cache)
        speedups.append(speedup)
        memory_savings.append(mem_red)

    # ==========================
    # 📈 PLOT: TIME
    # ==========================
    plt.figure()
    plt.plot(src_lens, standard_times, marker='o', label="Standard")
    plt.plot(src_lens, cached_times, marker='o', label="Cached")
    plt.xlabel("Source Length")
    plt.ylabel("Time (s)")
    plt.title("Generation Time")
    plt.legend()
    plt.grid()
    plt.show()

    # ==========================
    # 📈 PLOT: SPEEDUP
    # ==========================
    plt.figure()
    plt.plot(src_lens, speedups, marker='o')
    plt.xlabel("Source Length")
    plt.ylabel("Speedup (x)")
    plt.title("KV Cache Speedup")
    plt.grid()
    plt.show()

    # ==========================
    # 📈 PLOT: MEMORY
    # ==========================
    plt.figure()
    plt.plot(src_lens, memory_savings, marker='o')
    plt.xlabel("Source Length")
    plt.ylabel("Memory Reduction (%)")
    plt.title("Memory Savings")
    plt.grid()
    plt.show()


# ============================================================
# 🚀 RUN
# ============================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = D3PMCrossAttention()
    benchmark(model, device)
