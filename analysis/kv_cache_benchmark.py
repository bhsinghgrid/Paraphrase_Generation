# """
# analysis/kv_cache_benchmark.py
# ================================
# Task 1: Benchmark KV cache vs standard generate().
#
# Measures:
#   - Wall-clock time for generate() vs generate_cached()
#   - Encoder time as % of total generation time (before/after)
#   - Speedup ratio at src_len = 16, 32, 64 tokens
#
# How it works:
#   Standard generate():
#     For each of T=128 steps:
#       src → encoder → memory → decoder → logits    (encoder runs 128 times)
#
#   generate_cached():
#     src → encoder → memory (once)
#     For each of T=128 steps:
#       cached_memory → decoder → logits              (encoder runs 1 time)
#
#   Expected speedup:
#     If encoder = 30% of per-step time:
#       Saved = 127/128 * 30% ≈ 29.7% of total time
#     If encoder = 50% of per-step time:
#       Saved ≈ 49.6% of total time
#
# Usage:
#     python -m analysis.kv_cache_benchmark
#     or:
#     from analysis.kv_cache_benchmark import run_benchmark
#     results = run_benchmark(model, src_tokenizer, device)
# """
#
# import torch
# import time
# import numpy as np
# from typing import Dict, List
#
#
# def _make_src(src_len: int, src_vocab: int, device: torch.device, batch_size: int = 1):
#     """Create a random source tensor of given length."""
#     # Random real tokens (ids 5..src_vocab-1), padded to src_len
#     ids = torch.randint(5, src_vocab, (batch_size, src_len), device=device)
#     return ids
#
#
# def _time_fn(fn, n_warmup: int = 2, n_runs: int = 5) -> float:
#     """
#     Time a zero-argument callable.
#     Returns mean wall-clock seconds over n_runs after n_warmup warmup calls.
#     """
#     # Warmup
#     for _ in range(n_warmup):
#         fn()
#         if torch.cuda.is_available():
#             torch.cuda.synchronize()
#         elif torch.backends.mps.is_available():
#             torch.mps.synchronize()
#
#     times = []
#     for _ in range(n_runs):
#         start = time.perf_counter()
#         fn()
#         if torch.cuda.is_available():
#             torch.cuda.synchronize()
#         elif torch.backends.mps.is_available():
#             torch.mps.synchronize()
#         times.append(time.perf_counter() - start)
#
#     return float(np.mean(times))
#
#
# def benchmark_encoder_cost(
#     model,
#     src:    torch.Tensor,
# ) -> Dict[str, float]:
#     """
#     Measure encoder time as a fraction of one full forward pass.
#
#     Returns:
#         encoder_s   : seconds for one encoder call
#         full_step_s : seconds for one full forward_cached decoder step
#         encoder_pct : encoder_s / (encoder_s + full_step_s) * 100
#     """
#     inner = model.model
#     if not hasattr(inner, 'encode_source'):
#         raise ValueError("Model does not support KV cache (not D3PMCrossAttention).")
#
#     device = src.device
#     B      = src.shape[0]
#     T      = inner.scheduler.num_timesteps
#     tgt_len = inner.max_seq_len
#     mask_id = inner.mask_token_id
#
#     x0_est = torch.full((B, tgt_len), mask_id, dtype=torch.long, device=device)
#     t      = torch.zeros(B, dtype=torch.long, device=device)
#
#     # Time encoder alone
#     encoder_s = _time_fn(lambda: inner.encode_source(src))
#
#     # Pre-compute memory for decoder timing
#     memory, src_pad_mask = inner.encode_source(src)
#
#     # Time one decoder step (cached)
#     decoder_s = _time_fn(
#         lambda: inner.forward_cached(memory, src_pad_mask, x0_est, t,
#                                      inference_mode=True)
#     )
#
#     # Time one full step (non-cached = encoder + decoder)
#     full_s = _time_fn(
#         lambda: inner.forward(src, x0_est, t, inference_mode=True)
#     )
#
#     encoder_pct = 100.0 * encoder_s / max(full_s, 1e-9)
#
#     return {
#         "encoder_s":   encoder_s,
#         "decoder_s":   decoder_s,
#         "full_step_s": full_s,
#         "encoder_pct": encoder_pct,
#     }
#
#
# def run_benchmark(
#     model,
#     src_tokenizer,
#     device:        torch.device,
#     src_lens:      List[int] = [16, 32, 64],
#     n_runs:        int       = 5,
# ) -> Dict:
#     """
#     Full benchmark: compare generate() vs generate_cached() at multiple src lengths.
#
#     Args:
#         model         : SanskritModel (D3PMCrossAttention)
#         src_tokenizer : SanskritSourceTokenizer
#         device        : torch.device
#         src_lens      : list of source lengths to benchmark
#         n_runs        : number of timing runs per condition
#
#     Returns:
#         results dict with timing and speedup for each src_len
#     """
#     inner = model.model
#     if not hasattr(inner, 'generate_cached'):
#         raise ValueError("Model does not support KV cache (not D3PMCrossAttention).")
#
#     src_vocab = inner.src_embed.token_emb.weight.shape[0]
#     results   = {}
#
#     print("\n" + "=" * 65)
#     print("  KV CACHE BENCHMARK")
#     print("=" * 65)
#     print(f"  {'src_len':>8}  {'standard(s)':>12}  {'cached(s)':>10}  "
#           f"{'speedup':>8}  {'encoder%':>9}")
#     print("-" * 65)
#
#     for src_len in src_lens:
#         src = _make_src(src_len, src_vocab, device)
#
#         # Encoder cost breakdown
#         enc_cost = benchmark_encoder_cost(model, src)
#
#         # Time standard generate() — encoder runs T times
#         def run_standard():
#             return inner.generate(src, temperature=0.8, top_k=40)
#
#         # Time generate_cached() — encoder runs once
#         def run_cached():
#             return inner.generate_cached(src, temperature=0.8, top_k=40)
#
#         t_standard = _time_fn(run_standard, n_warmup=1, n_runs=n_runs)
#         t_cached   = _time_fn(run_cached,   n_warmup=1, n_runs=n_runs)
#         speedup    = t_standard / max(t_cached, 1e-9)
#
#         results[src_len] = {
#             "standard_s":  t_standard,
#             "cached_s":    t_cached,
#             "speedup":     speedup,
#             "encoder_pct": enc_cost["encoder_pct"],
#         }
#
#         print(f"  {src_len:>8}  {t_standard:>12.3f}  {t_cached:>10.3f}  "
#               f"{speedup:>7.2f}x  {enc_cost['encoder_pct']:>8.1f}%")
#
#     print("=" * 65)
#     print(f"\n  Encoder cost = % of one full forward pass")
#     print(f"  Speedup = standard_time / cached_time")
#     print(f"  Expected: speedup ≈ 1 / (1 - encoder_pct/100 * (T-1)/T)")
#
#     return results
#
#
# def print_summary(results: Dict):
#     """Print a human-readable summary of benchmark results."""
#     print("\n  SUMMARY")
#     print("  -------")
#     for src_len, r in results.items():
#         saved_pct = (1.0 - 1.0 / r["speedup"]) * 100
#         print(f"  src_len={src_len}: {r['speedup']:.2f}x speedup "
#               f"({saved_pct:.1f}% time saved, "
#               f"encoder was {r['encoder_pct']:.1f}% of total)")
#
#
# if __name__ == "__main__":
#     import sys, os
#     sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#     from config import CONFIG
#     from inference import load_model
#     from models.tokenizer import SanskritSourceTokenizer
#
#     cfg    = CONFIG
#     device = torch.device(cfg['training']['device'])
#
#     model_name = cfg['model_type']
#     has_neg    = cfg['data']['include_negative_examples']
#     ckpt       = f"results7/{model_name}_neg_{has_neg}/best_model.pt"
#
#     if not os.path.exists(ckpt):
#         print(f"No checkpoint at {ckpt}. Train first.")
#         sys.exit(1)
#
#     model, cfg = load_model(ckpt, cfg, device)
#     model.eval()
#
#     src_tokenizer = SanskritSourceTokenizer(
#         vocab_size = cfg['model'].get('src_vocab_size', 500),
#         max_len    = cfg['model']['max_seq_len'],
#     )
#
#     results = run_benchmark(model, src_tokenizer, device)
#     print_summary(results)
# ============================================================
# FULL TASK 1: KV CACHE + PROJECTION + BENCHMARK + GRAPHS
# ============================================================

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
