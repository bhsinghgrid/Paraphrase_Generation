"""
analysis/run_analysis.py
=========================
Entry point for all 5 tasks.

Tasks:
  Task 1 — KV Cache benchmark          (no retraining)
  Task 2 — Attention viz + drift        (no retraining)
  Task 3 — Concept vectors + PCA steer  (no retraining)
  Task 4 — Step ablation                (REQUIRES retraining for each T)
  Task 5 — Classifier-free guidance     (trains small 10k-param classifier)

Usage:
  python analysis/run_analysis.py --task 1
  python analysis/run_analysis.py --task 2 --input "dharmo rakṣati rakṣitaḥ"
  python analysis/run_analysis.py --task 3
  python analysis/run_analysis.py --task 4 --phase generate_configs
  python analysis/run_analysis.py --task 4 --phase analyze
  python analysis/run_analysis.py --task 5
  python analysis/run_analysis.py --task all --input "satyameva jayate"

Output files: analysis/outputs/
"""

import copy
import torch
import os, sys, argparse, json
import numpy as np
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG
from inference import load_model
from model.tokenizer import SanskritSourceTokenizer, SanskritTargetTokenizer

OUTPUT_DIR = "analysis/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Shared loader ─────────────────────────────────────────────────────

def infer_model_type_from_checkpoint(ckpt_path: str) -> str:
    name = ckpt_path.lower()
    if "ablation_results/t" in name or "d3pm_cross_attention" in name:
        return "d3pm_cross_attention"
    if "d3pm_encoder_decoder" in name:
        return "d3pm_encoder_decoder"
    if "baseline_cross_attention" in name:
        return "baseline_cross_attention"
    if "baseline_encoder_decoder" in name:
        return "baseline_encoder_decoder"
    return CONFIG["model_type"]


def infer_include_negative_from_checkpoint(ckpt_path: str) -> bool:
    name = ckpt_path.lower()
    if "_neg_true" in name:
        return True
    if "_neg_false" in name:
        return False
    if "ablation_results/t" in name:
        return False
    return CONFIG["data"]["include_negative_examples"]


def load_everything(cfg, device, ckpt_override=None):
    model_name = cfg['model_type']
    has_neg    = cfg['data']['include_negative_examples']
    candidates = [
        f"results7/{model_name}_neg_{has_neg}/best_model.pt",
        f"results/{model_name}_neg_{has_neg}/best_model.pt",
        f"results7/{model_name}_neg_True/best_model.pt",
        f"results/{model_name}_neg_True/best_model.pt",
        f"results7/{model_name}_neg_False/best_model.pt",
        f"results/{model_name}_neg_False/best_model.pt",
        "ablation_results/T4/best_model.pt",
        "ablation_results/T8/best_model.pt",
    ]
    ckpt = ckpt_override if ckpt_override else next((p for p in candidates if os.path.exists(p)), None)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"No checkpoint found. Checked: {candidates}")
    model, cfg = load_model(ckpt, cfg, device)
    model.eval()
    src_tok = SanskritSourceTokenizer(
        vocab_size=cfg['model'].get('src_vocab_size', 500),
        max_len=cfg['model']['max_seq_len'])
    tgt_tok = SanskritTargetTokenizer(
        vocab_size=cfg['model'].get('tgt_vocab_size', 500),
        max_len=cfg['model']['max_seq_len'])
    return model, src_tok, tgt_tok, cfg


def load_val_data(cfg, src_tok, tgt_tok, n=500):
    """Load validation set as (src_tensors, ref_strings, input_strings)."""
    from data.dataset import OptimizedSanskritDataset
    from torch.utils.data import Subset
    from sklearn.model_selection import train_test_split

    dataset = OptimizedSanskritDataset(
        'train', max_len=cfg['model']['max_seq_len'],
        cfg=cfg, src_tokenizer=src_tok, tgt_tokenizer=tgt_tok)
    total = min(cfg['data']['dataset_size'], len(dataset))
    _, val_idx = train_test_split(list(range(total)), train_size=0.8, random_state=42)
    val_idx = val_idx[:n]

    src_list, ref_list, inp_list = [], [], []
    for i in val_idx:
        item = dataset[i]
        src_list.append(item['input_ids'].unsqueeze(0))
        ref_list.append(item['target_text'])
        inp_list.append(item['input_text'])
    return src_list, ref_list, inp_list


def _generate_ids_compat(model, src, num_steps=None, temperature=0.8, top_k=40,
                         repetition_penalty=1.2, diversity_penalty=0.0):
    kwargs = dict(temperature=temperature, top_k=top_k)
    if num_steps is not None:
        kwargs["num_steps"] = int(num_steps)
    if repetition_penalty is not None:
        kwargs["repetition_penalty"] = float(repetition_penalty)
    if diversity_penalty is not None:
        kwargs["diversity_penalty"] = float(diversity_penalty)
    try:
        return model.generate(src, **kwargs)
    except TypeError:
        # Some model variants expose reduced generate() kwargs.
        slim = {k: kwargs[k] for k in ["temperature", "top_k", "num_steps"] if k in kwargs}
        try:
            return model.generate(src, **slim)
        except TypeError:
            return model.generate(src)


def _decode_ids(tgt_tok, out_ids):
    ids = []
    for x in out_ids[0].tolist():
        # stop at PAD/SEP once decoding started
        if x in (1, 4) and ids:
            break
        if x > 4:
            ids.append(x)
    return tgt_tok.decode(ids).strip(), ids


def _cer(a: str, b: str) -> float:
    if not b:
        return 1.0 if a else 0.0
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            tmp = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = tmp
    return dp[n] / max(1, n)


# ── Task 1 ────────────────────────────────────────────────────────────

def run_task1(model, src_tok, device):
    print("\n" + "="*65)
    print("  TASK 1 — KV Cache Benchmark")
    print("="*65)
    src_vocab = model.model.src_embed.token_emb.weight.shape[0]
    src_lens = [16, 32, 64]
    n_runs = 3
    has_cached = hasattr(model, "generate_cached")
    if not has_cached:
        print("  Compatibility mode: generate_cached() unavailable; running standard benchmark only.")

    def _timeit(fn, runs=n_runs):
        vals = []
        for _ in range(runs):
            t0 = time.perf_counter()
            fn()
            vals.append(time.perf_counter() - t0)
        return float(np.mean(vals))

    results = {}
    for L in src_lens:
        src = torch.randint(5, src_vocab, (1, L), device=device)
        t_std = _timeit(lambda: _generate_ids_compat(model, src, temperature=0.8, top_k=40))

        if has_cached:
            t_cache = _timeit(
                lambda: model.generate_cached(
                    src, num_steps=64, temperature=0.8, top_k=40,
                    repetition_penalty=1.2, diversity_penalty=0.0
                )
            )
            speedup = t_std / max(t_cache, 1e-9)
        else:
            t_cache = t_std
            speedup = 1.0

        # Encoder cost estimate: one encode_source pass vs one cached step.
        if hasattr(model.model, "encode_source") and hasattr(model.model, "forward_cached"):
            memory, src_pad = model.model.encode_source(src)
            x = torch.full((1, L), model.model.mask_token_id, dtype=torch.long, device=device)
            t = torch.full((1,), max(0, model.model.scheduler.num_timesteps - 1), dtype=torch.long, device=device)
            t_enc = _timeit(lambda: model.model.encode_source(src))
            t_step = _timeit(lambda: model.model.forward_cached(memory, src_pad, x, t, x0_hint=None, inference_mode=True))
            encoder_pct = (t_enc / max(t_enc + t_step, 1e-9)) * 100.0
        else:
            encoder_pct = 0.0

        results[L] = dict(
            standard_s=t_std,
            cached_s=t_cache,
            speedup=speedup,
            encoder_pct=encoder_pct,
        )
        print(f"  src_len={L:>3d}  standard={t_std:.3f}s  cached={t_cache:.3f}s  speedup={speedup:.2f}x  encoder%={encoder_pct:.1f}")

    # Memory profiling (GPU only). CPU memory differs by allocator and is noisy.
    mem_note = "N/A (CPU/MPS)"
    mem_red = None
    if torch.cuda.is_available():
        L = 64
        src = torch.randint(5, src_vocab, (1, L), device=device)
        torch.cuda.reset_peak_memory_stats(device)
        _ = _generate_ids_compat(model, src, temperature=0.8, top_k=40)
        m_std = torch.cuda.max_memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        _ = model.generate_cached(src, num_steps=64, temperature=0.8, top_k=40,
                                  repetition_penalty=1.2, diversity_penalty=0.0)
        m_cache = torch.cuda.max_memory_allocated(device)
        mem_red = 100.0 * (m_std - m_cache) / max(m_std, 1)
        mem_note = f"{mem_red:.1f}% reduction @ src_len=64"
        print(f"  Memory reduction: {mem_note}")

    # Subtask graphs
    lens = sorted(results.keys())
    std_vals = [results[L]["standard_s"] for L in lens]
    cache_vals = [results[L]["cached_s"] for L in lens]
    speed_vals = [results[L]["speedup"] for L in lens]
    enc_vals = [results[L]["encoder_pct"] for L in lens]

    plt.figure(figsize=(7, 4))
    plt.plot(lens, std_vals, marker="o", label="standard")
    plt.plot(lens, cache_vals, marker="o", label="cached")
    plt.xlabel("Source length")
    plt.ylabel("Time (s)")
    plt.title("Task1: Generation Time (Standard vs Cached)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "task1_time_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(lens, speed_vals, marker="o")
    plt.xlabel("Source length")
    plt.ylabel("Speedup (x)")
    plt.title("Task1: KV-Cache Speedup")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "task1_speedup.png"), dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(lens, enc_vals, marker="o")
    plt.xlabel("Source length")
    plt.ylabel("Encoder cost (%)")
    plt.title("Task1: Encoder Cost Share")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "task1_encoder_cost.png"), dpi=150, bbox_inches="tight")
    plt.close()

    path = os.path.join(OUTPUT_DIR, "task1_kv_cache.txt")
    with open(path, "w") as f:
        f.write("TASK 1 — KV CACHE BENCHMARK\n" + "="*40 + "\n\n")
        f.write(f"has_generate_cached={has_cached}\n")
        f.write(f"memory_profile={mem_note}\n\n")
        f.write(f"{'src_len':>8}  {'standard(s)':>12}  {'cached(s)':>10}  "
                f"{'speedup':>8}  {'encoder%':>9}\n")
        for src_len, r in results.items():
            f.write(f"{src_len:>8}  {r['standard_s']:>12.3f}  {r['cached_s']:>10.3f}  "
                    f"{r['speedup']:>7.2f}x  {r['encoder_pct']:>8.1f}%\n")
        f.write("\nSaved graphs:\n")
        f.write("  - task1_time_comparison.png\n")
        f.write("  - task1_speedup.png\n")
        f.write("  - task1_encoder_cost.png\n")
    print(f"  Saved: {path}")


# ── Task 2 ────────────────────────────────────────────────────────────

def run_task2(model, src_tok, tgt_tok, device, input_text, corpus_inputs=None):
    print("\n" + "="*65)
    print("  TASK 2 — Attention Visualization + Semantic Drift")
    print("="*65)
    print(f"  Input: {input_text}")
    if not hasattr(model.model, 'encode_source'):
        print("  Compatibility mode: attention hooks unavailable; running semantic-drift-only analysis.")
        src_ids = src_tok.encode(input_text)
        src = torch.tensor([src_ids], dtype=torch.long, device=device)
        # Keep steps <= scheduler horizon for this checkpoint to avoid backend aborts.
        t_sched = int(getattr(getattr(model.model, "scheduler", object()), "num_timesteps", 64))
        # Stability guard for some checkpoints/backends: keep sweep moderate.
        t_max = min(t_sched, 64)
        candidates = [t_max, 48, 32, 24, 16, 8, 4, 1]
        step_list = []
        seen = set()
        for s in candidates:
            s = max(1, min(int(s), t_max))
            if s not in seen:
                step_list.append(s)
                seen.add(s)
        outs = {}
        for s in step_list:
            out = _generate_ids_compat(model, src, num_steps=s, temperature=0.8, top_k=40)
            txt, _ = _decode_ids(tgt_tok, out)
            outs[s] = txt
        final = outs[1]
        drift = [(_cer(outs[s], final), s) for s in step_list]
        # Plot drift
        xs = [s for _, s in drift]
        ys = [c for c, _ in drift]
        plt.figure(figsize=(8, 4))
        plt.plot(xs, ys, marker='o')
        plt.gca().invert_xaxis()
        plt.xlabel("Generation steps")
        plt.ylabel("CER to 1-step output")
        plt.title("Task2 Semantic Drift (Compatibility Mode)")
        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, "task2_semantic_drift.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        report = os.path.join(OUTPUT_DIR, "task2_report.txt")
        with open(report, "w", encoding="utf-8") as f:
            f.write("TASK 2 — COMPATIBILITY REPORT\n")
            f.write("="*40 + "\n")
            f.write("Cross-attention capture unavailable for this checkpoint.\n")
            f.write(f"Input: {input_text}\n")
            f.write(f"Reference final (1 step): {final}\n\n")
            for cer_v, s in drift:
                f.write(f"steps={s:>3d}  CER_to_final={cer_v:.4f}  output={outs[s][:120]}\n")
        print(f"  Output(final@1): {final}")
        print(f"  Report: {report}")
        print(f"  Saved: {plot_path}")
        return

    src_ids    = src_tok.encode(input_text)
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_chars  = list(input_text.strip())

    from analysis.attention_viz import (
        AttentionCapture,
        compute_trajectory_metrics,
        analyze_token_stability,
        tfidf_attention_correlation,
    )

    # Attention capture
    print("  Capturing attention weights...")
    capturer = AttentionCapture(model)
    step_weights, step_outputs_ids = capturer.run(src_tensor)

    def _decode_tensor_ids(t):
        out = []
        for x in t[0].tolist():
            if x in (1, 4) and out:
                break
            if x > 4:
                out.append(x)
        return tgt_tok.decode(out).strip(), out

    decoded = {}
    for t_val, ids_t in step_outputs_ids.items():
        txt, ids = _decode_tensor_ids(ids_t)
        decoded[t_val] = (txt, ids)
    final_out = decoded[min(decoded.keys())][0]
    tgt_chars = list(final_out)
    print(f"  Output: {final_out}")

    # Heatmap t=max, layer 0
    first_t = max(step_weights.keys())
    w_first = step_weights[first_t][0][0]
    w0 = step_weights[0][0][0]
    n_src = min(len(src_chars), w_first.shape[1], 20)
    n_tgt = min(len(tgt_chars), w_first.shape[0], 20)
    plt.figure(figsize=(max(8, n_src * 0.35), max(6, n_tgt * 0.3)))
    plt.imshow(w_first[:n_tgt, :n_src], aspect="auto", cmap="YlOrRd")
    plt.xticks(range(n_src), src_chars[:n_src], rotation=45, ha="right", fontsize=8)
    plt.yticks(range(n_tgt), tgt_chars[:n_tgt], fontsize=8)
    plt.title(f"Attention t={first_t} Layer 0")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"task2_attn_t{first_t}.png"), dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(max(8, n_src * 0.35), max(6, n_tgt * 0.3)))
    plt.imshow(w0[:n_tgt, :n_src], aspect="auto", cmap="YlOrRd")
    plt.xticks(range(n_src), src_chars[:n_src], rotation=45, ha="right", fontsize=8)
    plt.yticks(range(n_tgt), tgt_chars[:n_tgt], fontsize=8)
    plt.title("Attention t=0 Layer 0")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "task2_attn_t0.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # All layers at t=0
    layers = step_weights[0]
    n_layers = len(layers)
    n_cols = min(4, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.2))
    axes = np.array(axes).reshape(-1)
    for i, layer_w in enumerate(layers):
        ax = axes[i]
        w = layer_w[0][:n_tgt, :n_src]
        ax.imshow(w, aspect="auto", cmap="YlOrRd")
        ax.set_title(f"Layer {i}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for i in range(n_layers, len(axes)):
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "task2_all_layers_t0.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Attention evolution for src[0] -> tgt[0]
    t_vals_desc = sorted(step_weights.keys(), reverse=True)
    evo = []
    for t_val in t_vals_desc:
        w = step_weights[t_val][0][0]
        evo.append(float(w[0, 0]) if w.shape[0] > 0 and w.shape[1] > 0 else 0.0)
    plt.figure(figsize=(10, 3.5))
    plt.plot(range(len(t_vals_desc)), evo, marker="o")
    plt.xlabel("Captured step index (T→0)")
    plt.ylabel("Attention weight")
    plt.title("Attention Evolution (src0→tgt0)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "task2_attn_evolution.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Drift (CER to final across steps)
    t_vals = sorted(decoded.keys(), reverse=True)
    cer_vals = [_cer(decoded[t][0], final_out) for t in t_vals]
    plt.figure(figsize=(8, 4))
    plt.plot(t_vals, cer_vals, marker="o")
    plt.gca().invert_xaxis()
    plt.xlabel("Diffusion step")
    plt.ylabel("CER to final")
    plt.title("Task2 Semantic Drift")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "task2_semantic_drift.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Source alignment proxy (avg attention on source positions at t=0, last layer)
    last_layer_t0 = step_weights[0][-1][0]
    src_align = last_layer_t0.mean(axis=0)[:n_src]
    plt.figure(figsize=(8, 3))
    plt.bar(np.arange(len(src_align)), src_align)
    plt.xticks(range(n_src), src_chars[:n_src], rotation=45, ha="right", fontsize=8)
    plt.title("Source Alignment Importance (t=0, last layer)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "task2_source_alignment.png"), dpi=150, bbox_inches="tight")
    plt.close()

    stability = analyze_token_stability(step_weights)
    n_locked = sum(1 for v in stability.values() if v == "LOCKED")
    n_flex = sum(1 for v in stability.values() if v == "FLEXIBLE")
    tfidf_info = tfidf_attention_correlation(input_text, step_weights, corpus_texts=corpus_inputs)
    tfidf_corr = tfidf_info.get("corr")
    tfidf_status = tfidf_info.get("status", "UNKNOWN")
    traj = compute_trajectory_metrics(step_outputs_ids, tgt_tok, reference_text=input_text)

    # TF-IDF vs attention graph (subtask visualization)
    tfidf_vec = np.asarray(tfidf_info.get("tfidf_scores", []), dtype=np.float32)
    attn_vec = np.asarray(tfidf_info.get("attn_scores", []), dtype=np.float32)
    labels = list(tfidf_info.get("tokens", []))
    m = min(len(tfidf_vec), len(attn_vec), len(labels), 20)
    if m > 0:
        x = np.arange(m)
        plt.figure(figsize=(8, 3.5))
        tf_part = tfidf_vec[:m]
        at_part = attn_vec[:m]
        tf_norm = tf_part / (np.max(np.abs(tf_part)) + 1e-9)
        at_norm = at_part / (np.max(np.abs(at_part)) + 1e-9)
        w = 0.4
        plt.bar(x - w/2, tf_norm, width=w, label="tfidf(norm)")
        plt.bar(x + w/2, at_norm, width=w, label="attn(norm)")
        plt.xlabel("Source token")
        plt.ylabel("Normalized score")
        plt.title("Task2: TF-IDF vs Attention Stability")
        plt.xticks(x, labels[:m], rotation=45, ha="right", fontsize=8)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "task2_tfidf_vs_attention.png"), dpi=150, bbox_inches="tight")
        plt.close()

    lock_in_t = next((t for t, c in zip(t_vals[::-1], cer_vals[::-1]) if c <= 0.05), t_vals[-1])
    report = os.path.join(OUTPUT_DIR, "task2_report.txt")
    with open(report, "w", encoding="utf-8") as f:
        f.write("TASK 2 — ATTENTION + DRIFT REPORT\n" + "=" * 50 + "\n\n")
        f.write(f"Input : {input_text}\n")
        f.write(f"Output: {final_out}\n\n")
        f.write(f"Captured steps: {len(t_vals)}\n")
        f.write(f"Lock-in step (CER<=0.05): t={lock_in_t}\n")
        f.write(f"Locked tokens: {n_locked}  Flexible tokens: {n_flex}\n")
        corr_txt = f"{tfidf_corr:.4f}" if tfidf_corr is not None else "N/A"
        f.write(f"TF-IDF vs attention stability corr: {corr_txt}\n")
        f.write(f"TF-IDF status: {tfidf_status}\n\n")
        f.write("Saved graphs:\n")
        f.write("  - task2_attn_t*.png / task2_all_layers_t0.png\n")
        f.write("  - task2_attn_evolution.png\n")
        f.write("  - task2_semantic_drift.png\n")
        f.write("  - task2_source_alignment.png\n")
        f.write("  - task2_tfidf_vs_attention.png\n\n")
        f.write("Step trajectory (first 10 rows)\n")
        f.write("-" * 60 + "\n")
        for row in traj[:10]:
            f.write(f"t={row['step']:>3d}  bert={row['bert']:.4f}  drift={row['drift']:.4f}  text={row['text'][:60]}\n")

    print(f"  Lock-in timestep: t={lock_in_t}")
    print(f"  Locked/Flexible: {n_locked}/{n_flex}")
    corr_txt = f"{tfidf_corr:.4f}" if tfidf_corr is not None else "N/A"
    print(f"  TF-IDF corr: {corr_txt} ({tfidf_status})")
    print(f"  Report: {report}")


# ── Task 3 ────────────────────────────────────────────────────────────

def run_task3(model, src_tok, tgt_tok, device, src_list, ref_list, n_samples=500):
    print("\n" + "="*65)
    print("  TASK 3 — Concept Vectors + PCA Steering")
    print("="*65)
    if not hasattr(model.model, 'encode_source'):
        print("  Compatibility mode: using output-token statistics for PCA concept proxy.")
        # Keep compatibility run lightweight/stable on constrained backends.
        n = min(60, len(src_list))
        feats, lens = [], []
        for i, src in enumerate(src_list[:n]):
            out = _generate_ids_compat(model, src.to(device), num_steps=8, temperature=0.8, top_k=40)
            txt, ids = _decode_ids(tgt_tok, out)
            arr = np.array(ids[:64] + [0] * max(0, 64 - len(ids[:64])), dtype=np.float32)
            feats.append(arr)
            lens.append(len(txt))
        from sklearn.decomposition import PCA
        X = np.stack(feats)
        pca = PCA(n_components=min(10, X.shape[0]-1, X.shape[1]))
        Z = pca.fit_transform(X)
        plt.figure(figsize=(6, 5))
        sc = plt.scatter(Z[:, 0], Z[:, 1] if Z.shape[1] > 1 else np.zeros_like(Z[:, 0]),
                         c=lens, cmap="viridis", s=14)
        plt.colorbar(sc, label="Output length")
        plt.title("Task3 Concept Proxy Space (Compatibility Mode)")
        plt.tight_layout()
        img = os.path.join(OUTPUT_DIR, "task3_concept_space.png")
        plt.savefig(img, dpi=150, bbox_inches="tight")
        plt.close()
        rep = os.path.join(OUTPUT_DIR, "task3_report.txt")
        corr = float(np.corrcoef(Z[:, 0], np.array(lens))[0, 1]) if len(lens) > 2 else 0.0
        with open(rep, "w", encoding="utf-8") as f:
            f.write("TASK 3 — COMPATIBILITY REPORT\n")
            f.write("="*40 + "\n")
            f.write("Hidden-state capture unavailable; used output-token vector proxy.\n")
            f.write(f"Samples: {n}\n")
            f.write(f"PC1-length correlation: {corr:.4f}\n")
        print(f"  Saved: {img}")
        print(f"  Report: {rep}")
        return

    from analysis.concept_vectors import (
        collect_hidden_states, fit_pca, find_diversity_direction, generate_diversity_spectrum
    )

    # Collect hidden states from val set
    n = min(max(1, int(n_samples)), len(src_list))
    print(f"  Collecting hidden states from {n} examples...")
    hidden, texts, lengths = collect_hidden_states(
        model, src_list[:n], tgt_tok, t_capture=0, max_samples=n
    )

    # Fit PCA + find diversity direction
    pca = fit_pca(hidden, n_components=min(50, n-1))
    direction = find_diversity_direction(hidden, lengths, pca)
    proj = pca.transform(hidden)
    corr = float(np.corrcoef(proj[:, 0], np.array(lengths))[0, 1]) if len(lengths) > 2 else 0.0
    best_pc = 0

    # Plot concept space
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(proj[:, 0], proj[:, 1] if proj.shape[1] > 1 else np.zeros_like(proj[:, 0]),
                     c=lengths, cmap="viridis", s=14)
    plt.colorbar(sc, label="Output length")
    plt.title("Task3 Concept Space")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "task3_concept_space.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Subtask graph: explained variance by PCA components
    ev = pca.explained_variance_ratio_
    k = min(20, len(ev))
    plt.figure(figsize=(8, 3.5))
    plt.bar(np.arange(k), ev[:k])
    plt.xlabel("PC index")
    plt.ylabel("Explained variance ratio")
    plt.title("Task3: PCA Explained Variance (Top Components)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "task3_pca_explained_variance.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Generate diversity spectrum for first example
    print("\n  Diversity spectrum for first example:")
    src0  = src_list[0]
    inp0  = src_tok.decode([x for x in src0[0].tolist() if x > 4])
    print(f"  Input: {inp0}")
    spectrum = generate_diversity_spectrum(
        model, src0.to(device), direction, tgt_tok,
        alphas=[-2.0, -1.0, 0.0, 1.0, 2.0])

    # Subtask graph: alpha vs decoded length
    a_vals = sorted(spectrum.keys())
    l_vals = [len(spectrum[a]) for a in a_vals]
    plt.figure(figsize=(7, 3.5))
    plt.plot(a_vals, l_vals, marker="o")
    plt.xlabel("Steering alpha")
    plt.ylabel("Output length")
    plt.title("Task3: Diversity Steering Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "task3_diversity_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Save diversity direction + results
    np.save(os.path.join(OUTPUT_DIR, "task3_diversity_direction.npy"), direction)

    report = os.path.join(OUTPUT_DIR, "task3_report.txt")
    with open(report, "w", encoding="utf-8") as f:
        f.write("TASK 3 — CONCEPT VECTORS + PCA STEERING\n" + "="*50 + "\n\n")
        f.write(f"PCA: {pca.n_components_} components, "
                f"{pca.explained_variance_ratio_.sum()*100:.1f}% variance\n")
        f.write(f"Diversity PC: {best_pc}  (|r|={corr:.3f} with output length)\n\n")
        f.write("Saved graphs:\n")
        f.write("  - task3_concept_space.png\n")
        f.write("  - task3_pca_explained_variance.png\n")
        f.write("  - task3_diversity_curve.png\n\n")
        f.write("Diversity spectrum:\n")
        for alpha, text in sorted(spectrum.items()):
            f.write(f"  alpha={alpha:+.1f}  →  {text}\n")
    print(f"  Report: {report}")


# ── Task 4 ────────────────────────────────────────────────────────────

def run_task4(phase, model, src_tok, tgt_tok, device, cfg,
              src_list, ref_list, n_samples=200):
    print("\n" + "="*65)
    print(f"  TASK 4 — Step Ablation  (phase={phase})")
    print("="*65)

    import analysis.step_ablation as step_ablation

    # Legacy API
    has_legacy = all(hasattr(step_ablation, fn) for fn in [
        "generate_ablation_configs", "run_ablation_analysis", "plot_ablation_3d"
    ])

    # New API
    has_new = hasattr(step_ablation, "run_task4")

    if phase == "generate_configs":
        if has_legacy:
            print("  Generating ablation configs...")
            step_ablation.generate_ablation_configs(output_dir="ablation_configs")
            print("\n  NEXT STEPS:")
            print("  1. bash ablation_configs/train_all.sh")
            print("  2. python analysis/run_analysis.py --task 4 --phase analyze")
            return
        print("  This step_ablation version does not expose config generation helpers.")
        print("  Use your latest ablation training script/config pipeline directly.")
        return

    if phase == "analyze":
        existing = [T for T in [4, 8, 16, 32, 64]
                    if os.path.exists(f"ablation_results/T{T}/best_model.pt")]
        if not existing:
            print("  No ablation models found at ablation_results/T*/best_model.pt")
            return
        print(f"  Found models for T={existing}")

        if has_legacy:
            results = step_ablation.run_ablation_analysis(
                ablation_dir="ablation_results", base_cfg=cfg,
                src_list=src_list[:200], ref_list=ref_list[:200],
                tgt_tokenizer=tgt_tok, device=device,
                output_dir=OUTPUT_DIR)
            step_ablation.plot_ablation_3d(
                results, save_path=os.path.join(OUTPUT_DIR, "task4_ablation_3d.png"))
        elif has_new:
            from inference import load_model as _load_model
            models = {}
            for T in existing:
                ckpt = f"ablation_results/T{T}/best_model.pt"
                cfg_t = copy.deepcopy(cfg)
                cfg_t["model"]["diffusion_steps"] = T
                cfg_t["inference"]["num_steps"] = T
                m_t, _ = _load_model(ckpt, cfg_t, device)
                m_t.eval()
                models[T] = m_t
            knee_t = step_ablation.run_task4(
                models, src_list[:n_samples], ref_list[:n_samples], tgt_tok,
                output_dir=OUTPUT_DIR, n_samples=n_samples)
            print(f"  New pipeline suggested optimal T={knee_t}")
        else:
            print("  Unsupported step_ablation API; please sync analysis/step_ablation.py")
            return

    # Optional adversarial robustness (legacy helper only)
    if hasattr(step_ablation, "run_adversarial_test"):
        print("\n  Running adversarial robustness test...")
        inp_texts = [src_tok.decode([x for x in s[0].tolist() if x > 4])
                     for s in src_list[:50]]
        step_ablation.run_adversarial_test(
            model, src_tok, tgt_tok,
            test_inputs=inp_texts, test_refs=ref_list[:50],
            device=device, output_dir=OUTPUT_DIR)


# ── Task 5 ────────────────────────────────────────────────────────────

def run_task5(model, src_tok, tgt_tok, device, cfg, src_list, ref_list):
    print("\n" + "="*65)
    print("  TASK 5 — Classifier-Free Guidance")
    print("="*65)
    if not hasattr(model.model, 'encode_source'):
        print("  Compatibility mode: classifier-guidance unavailable; sweeping decoding controls.")
        n = min(100, len(src_list), len(ref_list))
        lambdas = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
        results = []
        for lam in lambdas:
            rep_pen = 1.0 + 0.15 * lam
            cer_vals, uniq_vals = [], []
            for src, ref in zip(src_list[:n], ref_list[:n]):
                out = _generate_ids_compat(
                    model, src.to(device), num_steps=8, temperature=0.8, top_k=40,
                    repetition_penalty=rep_pen, diversity_penalty=0.0
                )
                txt, ids = _decode_ids(tgt_tok, out)
                cer_vals.append(_cer(txt, ref))
                uniq_vals.append(len(set(ids)) / max(1, len(ids)))
            results.append((lam, float(np.mean(cer_vals)), float(np.mean(uniq_vals))))
            print(f"  λ={lam:.1f}  CER={results[-1][1]:.4f}  diversity={results[-1][2]:.3f}")
        # Subtask graph: quality-diversity tradeoff
        x = [r[1] for r in results]
        y = [r[2] for r in results]
        labels = [r[0] for r in results]
        plt.figure(figsize=(6, 4))
        plt.plot(x, y, marker="o")
        for xi, yi, la in zip(x, y, labels):
            plt.text(xi, yi, f"λ={la:.1f}", fontsize=8)
        plt.xlabel("CER (lower is better)")
        plt.ylabel("Diversity")
        plt.title("Task5: Quality-Diversity Tradeoff")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "task5_quality_diversity_tradeoff.png"), dpi=150, bbox_inches="tight")
        plt.close()
        rep = os.path.join(OUTPUT_DIR, "task5_report.txt")
        with open(rep, "w", encoding="utf-8") as f:
            f.write("TASK 5 — COMPATIBILITY REPORT\n")
            f.write("="*40 + "\n")
            f.write("Guidance classifier path unavailable; λ mapped to repetition penalty.\n\n")
            for lam, cer_v, div_v in results:
                f.write(f"lambda={lam:.1f}  CER={cer_v:.4f}  diversity={div_v:.3f}\n")
            f.write("\nSaved graphs:\n")
            f.write("  - task5_quality_diversity_tradeoff.png\n")
        print(f"  Report: {rep}")
        return

    try:
        from analysis.quality_classifier import (
            QualityClassifier, collect_quality_data,
            train_quality_classifier, sweep_guidance_scales)
    except Exception:
        print("  Quality-classifier API mismatch; using compatibility sweep.")
        n = min(50, len(src_list))
        scales = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
        results = []
        for lam in scales:
            rep_pen = 1.0 + 0.2 * lam
            cer_vals, uniq_vals = [], []
            for src, ref in zip(src_list[:n], ref_list[:n]):
                out = _generate_ids_compat(
                    model, src.to(device), num_steps=8, temperature=0.8, top_k=40,
                    repetition_penalty=rep_pen, diversity_penalty=0.0
                )
                txt, ids = _decode_ids(tgt_tok, out)
                cer_vals.append(_cer(txt, ref))
                uniq_vals.append(len(set(ids)) / max(1, len(ids)))
            results.append((lam, float(np.mean(cer_vals)), float(np.mean(uniq_vals))))
            print(f"  λ={lam:.1f}  CER={results[-1][1]:.4f}  diversity={results[-1][2]:.3f}")
        # Subtask graph: quality-diversity tradeoff
        x = [r[1] for r in results]
        y = [r[2] for r in results]
        labels = [r[0] for r in results]
        plt.figure(figsize=(6, 4))
        plt.plot(x, y, marker="o")
        for xi, yi, la in zip(x, y, labels):
            plt.text(xi, yi, f"λ={la:.1f}", fontsize=8)
        plt.xlabel("CER (lower is better)")
        plt.ylabel("Diversity")
        plt.title("Task5: Quality-Diversity Tradeoff")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "task5_quality_diversity_tradeoff.png"), dpi=150, bbox_inches="tight")
        plt.close()
        rep = os.path.join(OUTPUT_DIR, "task5_report.txt")
        with open(rep, "w", encoding="utf-8") as f:
            f.write("TASK 5 — COMPATIBILITY REPORT\n")
            f.write("="*40 + "\n")
            f.write("Guidance classifier path unavailable; λ mapped to repetition penalty.\n\n")
            for lam, cer_v, div_v in results:
                f.write(f"lambda={lam:.1f}  CER={cer_v:.4f}  diversity={div_v:.3f}\n")
            f.write("\nSaved graphs:\n")
            f.write("  - task5_quality_diversity_tradeoff.png\n")
        print(f"  Report: {rep}")
        return

    clf_path = os.path.join(OUTPUT_DIR, "task5_quality_classifier.pt")
    d_model  = cfg['model']['d_model']

    # Step 1: collect or load training data
    data_path = os.path.join(OUTPUT_DIR, "task5_quality_data.npz")
    if os.path.exists(data_path):
        print("  Loading cached quality data...")
        data    = np.load(data_path)
        hidden  = data["hidden"]
        quality = data["quality"]
    else:
        print("  Collecting quality data (this takes a few minutes)...")
        n       = min(2000, len(src_list))
        hidden, quality = collect_quality_data(
            model, src_list[:n], ref_list[:n], tgt_tok,
            t_capture=0, max_samples=n)
        np.savez(data_path, hidden=hidden, quality=quality)
        print(f"  Saved quality data: {data_path}")

    # Step 2: train or load classifier
    if os.path.exists(clf_path):
        print(f"  Loading cached classifier: {clf_path}")
        clf = QualityClassifier(d_model)
        clf.load_state_dict(torch.load(clf_path, map_location='cpu'))
        clf.eval()
    else:
        print("  Training quality classifier...")
        clf = train_quality_classifier(
            hidden, quality, d_model=d_model,
            epochs=30, batch_size=64, lr=1e-3,
            save_path=clf_path)
        clf.eval()

    # Step 3: guidance scale sweep
    print("\n  Guidance scale sweep (λ ∈ {0.0, 0.5, 1.0, 1.5, 2.0, 3.0})...")
    n_sweep = min(50, len(src_list))
    results = sweep_guidance_scales(
        model, clf, src_list[:n_sweep], ref_list[:n_sweep],
        tgt_tok, scales=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
        n_samples=n_sweep, device=device, output_dir=OUTPUT_DIR)

    # Find optimal scale
    best_scale = min(results, key=lambda s: results[s]["mean_cer"])
    print(f"\n  Optimal guidance scale: λ={best_scale:.1f}  "
          f"CER={results[best_scale]['mean_cer']:.4f}")

    report = os.path.join(OUTPUT_DIR, "task5_report.txt")
    with open(report, "w") as f:
        f.write("TASK 5 — CLASSIFIER-FREE GUIDANCE\n" + "="*50 + "\n\n")
        f.write(f"Classifier params: {sum(p.numel() for p in clf.parameters())}\n")
        f.write(f"Training samples : {len(hidden)}\n\n")
        f.write("Guidance scale sweep:\n")
        f.write(f"  {'λ':>6}  {'CER':>8}  {'diversity':>10}\n")
        f.write("  " + "-"*28 + "\n")
        for s in sorted(results.keys()):
            r = results[s]
            marker = " ← optimal" if s == best_scale else ""
            f.write(f"  {s:>6.1f}  {r['mean_cer']:>8.4f}  {r['diversity']:>10.3f}{marker}\n")
    print(f"  Report: {report}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    global OUTPUT_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--task",
        choices=["1","2","3","4","5","all"], default="all")
    parser.add_argument("--input",
        default="dharmo rakṣati rakṣitaḥ",
        help="IAST input text for Task 2")
    parser.add_argument("--phase",
        choices=["generate_configs", "analyze"], default="analyze",
        help="Task 4 phase: generate_configs (before training) or analyze (after)")
    parser.add_argument("--checkpoint", default=None,
        help="Optional explicit checkpoint path")
    parser.add_argument("--output_dir", default="analysis/outputs",
        help="Output directory for reports/figures")
    parser.add_argument("--task4_samples", type=int, default=50,
        help="Samples for Task 4 dry/full evaluation")
    parser.add_argument("--task3_samples", type=int, default=500,
        help="Samples for Task 3 hidden-state collection")
    args = parser.parse_args()

    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cfg = copy.deepcopy(CONFIG)
    if args.checkpoint:
        cfg["model_type"] = infer_model_type_from_checkpoint(args.checkpoint)
        cfg["data"]["include_negative_examples"] = infer_include_negative_from_checkpoint(args.checkpoint)
        ckpt_name = os.path.basename(os.path.dirname(args.checkpoint))
        if ckpt_name.startswith("T") and ckpt_name[1:].isdigit():
            t_val = int(ckpt_name[1:])
            cfg["model"]["diffusion_steps"] = t_val
            cfg["inference"]["num_steps"] = t_val

    requested = cfg["training"]["device"]
    if requested == "mps" and not torch.backends.mps.is_available():
        requested = "cpu"
    elif requested == "cuda" and not torch.cuda.is_available():
        requested = "cpu"
    cfg["training"]["device"] = requested
    device = torch.device(requested)

    print("Loading model and tokenizers...")
    model, src_tok, tgt_tok, cfg = load_everything(cfg, device, ckpt_override=args.checkpoint)

    # Load val data for tasks that need corpus/context (Tasks 2, 3, 4, 5)
    needs_data = args.task in ("2", "3", "4", "5", "all")
    if needs_data:
        print("Loading validation data...")
        src_list, ref_list, inp_list = load_val_data(cfg, src_tok, tgt_tok, n=500)
    else:
        src_list, ref_list, inp_list = [], [], []

    tasks = (["1","2","3","4","5"] if args.task == "all"
             else [args.task])

    for task in tasks:
        if task == "1":
            run_task1(model, src_tok, device)
        elif task == "2":
            run_task2(model, src_tok, tgt_tok, device, args.input, corpus_inputs=inp_list)
        elif task == "3":
            run_task3(model, src_tok, tgt_tok, device, src_list, ref_list, n_samples=args.task3_samples)
        elif task == "4":
            run_task4(args.phase, model, src_tok, tgt_tok, device, cfg,
                      src_list, ref_list, n_samples=args.task4_samples)
        elif task == "5":
            run_task5(model, src_tok, tgt_tok, device, cfg, src_list, ref_list)

    print(f"\n{'='*65}")
    print(f"  All outputs saved to: {OUTPUT_DIR}/")
    print("="*65)


if __name__ == "__main__":
    main()
