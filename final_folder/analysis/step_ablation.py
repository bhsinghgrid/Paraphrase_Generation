# """
# analysis/step_ablation.py
# ==========================
# Task 4: Semantic Robustness — Ablation of Diffusion Steps vs Meaning Preservation
#
# Two-phase workflow (retraining IS required for different T values):
#
#   PHASE 1 — Generate configs + train (run once per T value):
#     python analysis/step_ablation.py --phase generate_configs
#     # Creates configs: ablation_configs/T4.py, T8.py, T16.py, T32.py, T64.py
#     # Then train each: MODEL_TYPE=d3pm_cross_attention python train.py  (for each config)
#
#   PHASE 2 — Analyze trained models (no retraining needed):
#     python analysis/step_ablation.py --phase analyze
#     # Loads each trained model, generates 200 paraphrases, computes CER
#     # Produces 3D plot: X=steps, Y=generation_speed, Z=CER
#
# Why retraining is needed:
#   A model trained with T=128 learns to denoise from x_t~Uniform[0,128].
#   Running it with T=4 means the model only sees t∈{0,1,2,3} — which it
#   was never trained on at those scales. Outputs are meaningless.
#   You must train a separate model for each T value.
#
# Also implements adversarial robustness test (no retraining):
#   Takes your existing T=128 model and tests whether corrupted IAST
#   inputs (typos, character swaps) cause proportional output degradation.
# """
#
# import torch
# import torch.nn.functional as F
# import numpy as np
# import os
# import sys
# import time
# import json
# import copy
# from typing import List, Dict, Optional
#
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#
#
# # ── Phase 1: Config generation ────────────────────────────────────────
#
# T_VALUES = [4, 8, 16, 32, 64]
#
# def generate_ablation_configs(base_config_path: str = "config.py",
#                                output_dir: str = "ablation_configs"):
#     """
#     Generate one config file per T value.
#     Each config is a copy of the base config with diffusion_steps changed.
#
#     After running this, train each model:
#         for T in 4 8 16 32 64; do
#             cp ablation_configs/config_T${T}.py config.py
#             python train.py
#             mv results7/d3pm_cross_attention_neg_False \
#                ablation_results/T${T}
#         done
#     """
#     os.makedirs(output_dir, exist_ok=True)
#
#     # Read base config
#     with open(base_config_path, "r") as f:
#         base_src = f.read()
#
#     for T in T_VALUES:
#         # Replace diffusion_steps and num_steps
#         cfg_src = base_src
#         cfg_src = cfg_src.replace(
#             '"diffusion_steps": 128',
#             f'"diffusion_steps": {T}'
#         )
#         cfg_src = cfg_src.replace(
#             "'diffusion_steps': 128",
#             f"'diffusion_steps': {T}"
#         )
#         cfg_src = cfg_src.replace(
#             '"num_steps": 128',
#             f'"num_steps": {T}'
#         )
#         cfg_src = cfg_src.replace(
#             "'num_steps': 128",
#             f"'num_steps': {T}"
#         )
#         out_path = os.path.join(output_dir, f"config_T{T}.py")
#         with open(out_path, "w") as f:
#             f.write(f"# Ablation config: T={T} diffusion steps\n")
#             f.write(cfg_src)
#         print(f"  Wrote: {out_path}")
#
#     # Write a shell script to train all
#     shell_script = os.path.join(output_dir, "train_all.sh")
#     with open(shell_script, "w") as f:
#         f.write("#!/bin/bash\n")
#         f.write("# Run this script to train all ablation models\n\n")
#         for T in T_VALUES:
#             f.write(f"echo '=== Training T={T} ==='\n")
#             f.write(f"cp {output_dir}/config_T{T}.py config.py\n")
#             f.write(f"python train.py\n")
#             f.write(f"mkdir -p ablation_results/T{T}\n")
#             f.write(f"cp -r results7/d3pm_cross_attention_neg_False/best_model.pt "
#                     f"ablation_results/T{T}/best_model.pt\n")
#             f.write(f"cp -r results7/d3pm_cross_attention_neg_False/train.log "
#                     f"ablation_results/T{T}/train.log\n\n")
#     os.chmod(shell_script, 0o755)
#     print(f"\nTraining script: {shell_script}")
#     print(f"Run: bash {shell_script}")
#
#
# # ── Phase 2: Analysis (after models are trained) ──────────────────────
#
# def compute_cer(pred: str, ref: str) -> float:
#     if not ref:
#         return 1.0
#
#     def edit_distance(s1, s2):
#         m, n = len(s1), len(s2)
#         dp = list(range(n + 1))
#         for i in range(1, m + 1):
#             prev, dp[0] = dp[0], i
#             for j in range(1, n + 1):
#                 temp = dp[j]
#                 dp[j] = prev if s1[i-1] == s2[j-1] else 1 + min(prev, dp[j], dp[j-1])
#                 prev = temp
#         return dp[n]
#
#     return edit_distance(pred, ref) / max(len(ref), 1)
#
#
# def evaluate_model(
#     model,
#     src_list:      List[torch.Tensor],
#     ref_list:      List[str],
#     tgt_tokenizer,
#     n_samples:     int   = 200,
#     temperature:   float = 0.8,
#     top_k:         int   = 40,
# ) -> Dict:
#     """
#     Generate n_samples outputs and compute CER + generation speed.
#
#     Returns dict with:
#         mean_cer      : average CER over samples
#         generation_s  : total wall-clock seconds for all generations
#         speed_per_sample: seconds per sample
#         cer_list      : per-sample CER values
#     """
#     device   = next(model.parameters()).device
#     n        = min(n_samples, len(src_list))
#     cer_list = []
#
#     start = time.perf_counter()
#     for i, (src, ref) in enumerate(zip(src_list[:n], ref_list[:n])):
#         if src.dim() == 1:
#             src = src.unsqueeze(0)
#
#         with torch.no_grad():
#             if hasattr(model.model, 'generate_cached'):
#                 out = model.model.generate_cached(
#                     src.to(device), temperature=temperature, top_k=top_k
#                 )
#             else:
#                 out = model.generate(
#                     src.to(device), temperature=temperature, top_k=top_k
#                 )
#
#         ids  = [x for x in out[0].tolist() if x > 4]
#         pred = tgt_tokenizer.decode(ids).strip()
#         cer  = compute_cer(pred, ref)
#         cer_list.append(cer)
#
#     elapsed = time.perf_counter() - start
#
#     return {
#         "mean_cer":          float(np.mean(cer_list)),
#         "std_cer":           float(np.std(cer_list)),
#         "generation_s":      elapsed,
#         "speed_per_sample":  elapsed / max(n, 1),
#         "cer_list":          cer_list,
#         "n_samples":         n,
#     }
#
#
# def run_ablation_analysis(
#     ablation_dir:  str = "ablation_results",
#     base_cfg:      dict = None,
#     src_list:      List[torch.Tensor] = None,
#     ref_list:      List[str] = None,
#     tgt_tokenizer  = None,
#     device:        torch.device = None,
#     output_dir:    str = "analysis/outputs",
# ) -> Dict:
#     """
#     Load each trained model and evaluate.
#     Produces results dict and 3D plot.
#
#     Expects ablation_results/T{N}/best_model.pt for each T in T_VALUES.
#     """
#     from inference import load_model
#
#     results = {}
#     for T in T_VALUES:
#         ckpt = os.path.join(ablation_dir, f"T{T}", "best_model.pt")
#         if not os.path.exists(ckpt):
#             print(f"  SKIP T={T}: no checkpoint at {ckpt}")
#             continue
#
#         print(f"\nEvaluating T={T}...")
#         cfg_T = copy.deepcopy(base_cfg)
#         cfg_T['model']['diffusion_steps'] = T
#         cfg_T['inference']['num_steps']   = T
#
#         model, cfg_T = load_model(ckpt, cfg_T, device)
#         model.eval()
#
#         metrics = evaluate_model(
#             model, src_list, ref_list, tgt_tokenizer, n_samples=200
#         )
#         results[T] = metrics
#         print(f"  T={T}  CER={metrics['mean_cer']:.4f}  "
#               f"speed={metrics['speed_per_sample']:.3f}s/sample")
#
#         del model
#
#     # Save results
#     os.makedirs(output_dir, exist_ok=True)
#     results_path = os.path.join(output_dir, "ablation_results.json")
#     with open(results_path, "w") as f:
#         json.dump({str(k): {kk: vv for kk, vv in v.items() if kk != 'cer_list'}
#                    for k, v in results.items()}, f, indent=2)
#     print(f"\nResults saved: {results_path}")
#
#     return results
#
#
# def plot_ablation_3d(
#     results:   Dict,
#     save_path: Optional[str] = None,
# ):
#     """
#     3D plot: X=diffusion_steps, Y=generation_speed(s/sample), Z=CER.
#     Also produces a 2D summary plot.
#     """
#     try:
#         import matplotlib.pyplot as plt
#         from mpl_toolkits.mplot3d import Axes3D
#     except ImportError:
#         print("pip install matplotlib.")
#         return
#
#     T_list    = sorted(results.keys())
#     cers      = [results[T]["mean_cer"] for T in T_list]
#     speeds    = [results[T]["speed_per_sample"] for T in T_list]
#
#     # ── 3D plot ───────────────────────────────────────────────────────
#     fig = plt.figure(figsize=(14, 5))
#
#     ax3d = fig.add_subplot(121, projection='3d')
#     ax3d.scatter(T_list, speeds, cers, c=cers, cmap='RdYlGn_r', s=80)
#     for T, s, c in zip(T_list, speeds, cers):
#         ax3d.text(T, s, c, f"T={T}", fontsize=8)
#     ax3d.set_xlabel("Diffusion steps T", fontsize=9)
#     ax3d.set_ylabel("Speed (s/sample)", fontsize=9)
#     ax3d.set_zlabel("CER (↓ better)", fontsize=9)
#     ax3d.set_title("T vs speed vs CER", fontsize=10)
#
#     # ── 2D CER vs T (find the knee) ──────────────────────────────────
#     ax2d = fig.add_subplot(122)
#     ax2d.plot(T_list, cers, 'o-', linewidth=1.8, color='coral', markersize=7)
#     for T, c in zip(T_list, cers):
#         ax2d.annotate(f"{c:.3f}", (T, c), textcoords="offset points",
#                       xytext=(0, 8), fontsize=8, ha='center')
#
#     # Find knee: largest CER drop per unit T (elbow method)
#     if len(T_list) >= 3:
#         drops  = [cers[i] - cers[i+1] for i in range(len(cers)-1)]
#         knee_i = int(np.argmax(drops))
#         knee_T = T_list[knee_i + 1]
#         ax2d.axvline(knee_T, color='steelblue', linestyle='--', linewidth=1.2,
#                      label=f"Knee at T={knee_T}")
#         ax2d.legend(fontsize=9)
#
#     ax2d.set_xlabel("Diffusion steps T", fontsize=10)
#     ax2d.set_ylabel("CER (lower = better)", fontsize=10)
#     ax2d.set_title("CER vs diffusion steps", fontsize=10)
#     ax2d.set_ylim(0, max(cers) * 1.1)
#
#     plt.tight_layout()
#     if save_path:
#         os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
#         plt.savefig(save_path, dpi=150, bbox_inches='tight')
#         print(f"Saved: {save_path}")
#     else:
#         plt.show()
#     plt.close()
#
#
# # ── Adversarial robustness test (no retraining needed) ───────────────
#
# def corrupt_iast(text: str, corruption_rate: float = 0.05) -> str:
#     """
#     Introduce random corruption into IAST text:
#       - Character swap (adjacent chars swapped)
#       - Character deletion
#       - Random character insertion
#
#     Models rate as 5% to 20% corruption to test robustness.
#     """
#     import random
#     chars = list(text)
#     n_corrupt = max(1, int(len(chars) * corruption_rate))
#
#     for _ in range(n_corrupt):
#         op  = random.choice(['swap', 'delete', 'insert'])
#         pos = random.randint(0, len(chars) - 1)
#
#         if op == 'swap' and pos < len(chars) - 1:
#             chars[pos], chars[pos+1] = chars[pos+1], chars[pos]
#         elif op == 'delete' and len(chars) > 1:
#             chars.pop(pos)
#         elif op == 'insert':
#             chars.insert(pos, random.choice('abcdeimnostu'))
#
#     return "".join(chars)
#
#
# @torch.no_grad()
# def run_adversarial_test(
#     model,
#     src_tokenizer,
#     tgt_tokenizer,
#     test_inputs:    List[str],
#     test_refs:      List[str],
#     corruption_rates: List[float] = [0.0, 0.05, 0.10, 0.15, 0.20],
#     device:         torch.device  = None,
#     output_dir:     str           = "analysis/outputs",
# ) -> Dict:
#     """
#     Test if CER degrades proportionally with IAST corruption.
#     Uses existing trained model — no retraining.
#     """
#     device = device or next(model.parameters()).device
#     results = {}
#
#     print("\nAdversarial robustness test...")
#     for rate in corruption_rates:
#         cer_list = []
#         for text, ref in zip(test_inputs, test_refs):
#             corrupted = corrupt_iast(text, rate)
#             ids       = src_tokenizer.encode(corrupted)
#             src       = torch.tensor([ids], dtype=torch.long, device=device)
#
#             if hasattr(model.model, 'generate_cached'):
#                 out = model.model.generate_cached(src)
#             else:
#                 out = model.generate(src)
#
#             pred_ids = [x for x in out[0].tolist() if x > 4]
#             pred     = tgt_tokenizer.decode(pred_ids).strip()
#             cer_list.append(compute_cer(pred, ref))
#
#         mean_cer = float(np.mean(cer_list))
#         results[rate] = mean_cer
#         print(f"  corruption={rate*100:.0f}%  →  CER={mean_cer:.4f}")
#
#     # Save + plot
#     os.makedirs(output_dir, exist_ok=True)
#     try:
#         import matplotlib.pyplot as plt
#         fig, ax = plt.subplots(figsize=(8, 4))
#         rates   = [r * 100 for r in corruption_rates]
#         cers    = [results[r] for r in corruption_rates]
#         ax.plot(rates, cers, 'o-', linewidth=1.8, color='steelblue', markersize=7)
#         ax.set_xlabel("IAST corruption rate (%)", fontsize=11)
#         ax.set_ylabel("CER", fontsize=11)
#         ax.set_title("Model robustness to IAST input corruption", fontsize=11)
#         ax.set_ylim(0, max(cers) * 1.2)
#         plt.tight_layout()
#         plt.savefig(os.path.join(output_dir, "adversarial_robustness.png"),
#                     dpi=150, bbox_inches='tight')
#         plt.close()
#         print(f"  Saved: {output_dir}/adversarial_robustness.png")
#     except ImportError:
#         pass
#
#     with open(os.path.join(output_dir, "adversarial_results.json"), "w") as f:
#         json.dump({str(k): v for k, v in results.items()}, f, indent=2)
#
#     return results
"""
analysis/task4_pipeline.py
================================
Correct Task 4 Pipeline:

PHASE 1 → Evaluate all models
PHASE 2 → Analyze + detect optimal T

NO early decision making.
"""

import torch
import numpy as np
import time
import os
import json
from typing import Dict, List
from difflib import SequenceMatcher


# ─────────────────────────────────────────────
# Load Metrics
# ─────────────────────────────────────────────

def load_metrics():
    try:
        from bert_score import score as bert_score
    except Exception:
        bert_score = None
    from nltk.translate.bleu_score import sentence_bleu
    try:
        from sentence_transformers import SentenceTransformer, util
        st_model = SentenceTransformer('all-MiniLM-L6-v2')
        return bert_score, st_model, util, sentence_bleu
    except Exception:
        # Offline-safe fallback: skip sentence-transformer similarity.
        return bert_score, None, None, sentence_bleu


# ─────────────────────────────────────────────
# PHASE 1 — Evaluate ALL models
# ─────────────────────────────────────────────

def evaluate_all_models(models: Dict[int, object],
                        src_list,
                        ref_list,
                        tgt_tokenizer,
                        n_samples=200,
                        output_dir: str = "analysis/outputs"):

    bert_score_fn, st_model, util, bleu_fn = load_metrics()

    results = {}

    print("\n=== PHASE 1: Evaluating ALL models ===")

    for T, model in sorted(models.items()):
        print(f"\nEvaluating T={T}...")

        device = next(model.parameters()).device
        preds, refs = [], []

        start = time.perf_counter()

        for src, ref in zip(src_list[:n_samples], ref_list[:n_samples]):
            if src.dim() == 1:
                src = src.unsqueeze(0)

            with torch.no_grad():
                if hasattr(model, "model") and hasattr(model.model, "generate_cached"):
                    out = model.model.generate_cached(src.to(device))
                else:
                    # Fallback for wrappers that only expose top-level generate.
                    out = model.generate(src.to(device))

            ids = [x for x in out[0].tolist() if x > 4]
            pred = tgt_tokenizer.decode(ids).strip()

            preds.append(pred)
            refs.append(ref)

        elapsed = time.perf_counter() - start

        # BERTScore (fallback to lexical similarity if unavailable/offline)
        try:
            if bert_score_fn is not None:
                _, _, F1 = bert_score_fn(preds, refs, lang="hi", verbose=False)
                bert_f1 = float(F1.mean())
            else:
                raise RuntimeError("bertscore unavailable")
        except Exception:
            bert_f1 = float(np.mean([SequenceMatcher(None, p, r).ratio() for p, r in zip(preds, refs)]))

        # Sentence similarity
        if st_model is not None:
            emb_p = st_model.encode(preds, convert_to_tensor=True)
            emb_r = st_model.encode(refs, convert_to_tensor=True)
            sim = util.cos_sim(emb_p, emb_r).diagonal().mean().item()
        else:
            sim = float("nan")

        # BLEU
        bleu_scores = [
            bleu_fn([r.split()], p.split())
            for p, r in zip(preds, refs)
        ]

        results[T] = {
            "bertscore_f1": bert_f1,
            "semantic_sim": sim,
            "bleu": float(np.mean(bleu_scores)),
            "speed_per_sample": elapsed / n_samples
        }

        print(f"  BERTScore: {bert_f1:.4f}")
        print(f"  Sim: {sim:.4f}")
        print(f"  BLEU: {results[T]['bleu']:.4f}")
        print(f"  Speed: {results[T]['speed_per_sample']:.4f}s")

    # Save raw results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "task4_raw_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


# ─────────────────────────────────────────────
# PHASE 2 — Analyze results (Knee Detection)
# ─────────────────────────────────────────────

def analyze_results(results: Dict):
    print("\n=== PHASE 2: Analysis ===")

    T_list = sorted(results.keys())
    scores = [results[T]["bertscore_f1"] for T in T_list]

    gains = [scores[i+1] - scores[i] for i in range(len(scores)-1)]

    print("\nMarginal Gains:")
    for i, g in enumerate(gains):
        print(f"  T{T_list[i]} → T{T_list[i+1]}: +{g:.4f}")

    # Knee detection
    threshold = 0.02
    knee_T = T_list[-1]

    for i, g in enumerate(gains):
        if g < threshold:
            knee_T = T_list[i+1]
            break

    print(f"\n✅ Optimal T (knee detected): {knee_T}")

    return knee_T, gains


# ─────────────────────────────────────────────
# 3D Plot (BERTScore)
# ─────────────────────────────────────────────

def plot_3d(results, output_dir: str = "analysis/outputs"):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    T_list = sorted(results.keys())

    X = T_list
    Y = [results[T]["speed_per_sample"] for T in T_list]
    Z = [results[T]["bertscore_f1"] for T in T_list]

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X, Y, Z)

    for x, y, z in zip(X, Y, Z):
        ax.text(x, y, z, f"T={x}", fontsize=8)

    ax.set_xlabel("Diffusion Steps")
    ax.set_ylabel("Speed")
    ax.set_zlabel("BERTScore")

    plt.title("3D Tradeoff: Steps vs Speed vs Quality")

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "task4_3d.png"))
    plt.close()

    print("Saved 3D plot")


# ─────────────────────────────────────────────
# FINAL RUNNER
# ─────────────────────────────────────────────

def run_task4(models, src_list, ref_list, tgt_tokenizer,
              output_dir: str = "analysis/outputs", n_samples: int = 200):

    # Phase 1: Evaluate all
    results = evaluate_all_models(
        models, src_list, ref_list, tgt_tokenizer, n_samples=n_samples, output_dir=output_dir
    )

    # Phase 2: Analyze
    knee_T, gains = analyze_results(results)

    # Plot
    plot_3d(results, output_dir=output_dir)

    # Save report
    with open(os.path.join(output_dir, "task4_report.txt"), "w") as f:
        f.write(f"Optimal diffusion steps = {knee_T}\n")

    return knee_T
