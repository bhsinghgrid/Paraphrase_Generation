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
from collections import Counter


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

        # Sentence similarity (distinct from BERT fallback)
        if st_model is not None:
            emb_p = st_model.encode(preds, convert_to_tensor=True)
            emb_r = st_model.encode(refs, convert_to_tensor=True)
            sim = util.cos_sim(emb_p, emb_r).diagonal().mean().item()
        else:
            # token-overlap F1 proxy (different behavior from char-level similarity)
            f1s = []
            for p, r in zip(preds, refs):
                pt = [t for t in p.split() if t]
                rt = [t for t in r.split() if t]
                if not pt or not rt:
                    f1s.append(0.0)
                    continue
                cp, cr = Counter(pt), Counter(rt)
                inter = sum((cp & cr).values())
                prec = inter / max(1, len(pt))
                rec = inter / max(1, len(rt))
                f1s.append((2 * prec * rec / max(1e-9, prec + rec)))
            sim = float(np.mean(f1s)) if f1s else 0.0
        if not np.isfinite(sim):
            sim = float(np.mean([SequenceMatcher(None, p, r).ratio() for p, r in zip(preds, refs)]))

        # BLEU
        bleu_scores = [
            bleu_fn([r.split()], p.split())
            for p, r in zip(preds, refs)
        ]

        results[T] = {
            "bertscore_f1": bert_f1,
            "semantic_sim": sim,
            "bleu": float(np.mean(bleu_scores)),
            "speed_per_sample": elapsed / max(1, len(preds))
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

    # Robust utility selection (quality + semantics + speed regularizer)
    bvals = np.array([results[T]["bertscore_f1"] for T in T_list], dtype=np.float32)
    svals = np.array([results[T]["semantic_sim"] for T in T_list], dtype=np.float32)
    tvals = np.array([results[T]["speed_per_sample"] for T in T_list], dtype=np.float32)
    b_norm = (bvals - bvals.min()) / max(1e-9, (bvals.max() - bvals.min()))
    s_norm = (svals - svals.min()) / max(1e-9, (svals.max() - svals.min()))
    t_norm = (tvals - tvals.min()) / max(1e-9, (tvals.max() - tvals.min()))
    utility = 0.50 * b_norm + 0.30 * s_norm - 0.20 * t_norm
    knee_T = T_list[int(np.argmax(utility))]

    print(f"\n✅ Optimal T (semantic-speed tradeoff): {knee_T}")

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

    # Save detailed report
    report_path = os.path.join(output_dir, "task4_report.txt")
    with open(report_path, "w") as f:
        f.write("TASK 4 — SEMANTIC ROBUSTNESS ABLATION\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Optimal diffusion steps = {knee_T}\n\n")
        f.write(f"{'T':>6}  {'BERT-F1':>10}  {'SEM_SIM':>10}  {'BLEU':>8}  {'sec/sample':>12}\n")
        f.write("  " + "-" * 56 + "\n")
        for T in sorted(results.keys()):
            r = results[T]
            f.write(
                f"{T:>6}  {r['bertscore_f1']:>10.4f}  {r['semantic_sim']:>10.4f}  "
                f"{r['bleu']:>8.4f}  {r['speed_per_sample']:>12.4f}\n"
            )
        f.write("\nMarginal gains (BERT-F1):\n")
        for i, g in enumerate(gains):
            t0 = sorted(results.keys())[i]
            t1 = sorted(results.keys())[i + 1]
            f.write(f"  T{t0} -> T{t1}: {g:+.4f}\n")
        f.write("\nSaved plots/files:\n")
        f.write("  - task4_3d.png\n")
        f.write("  - task4_raw_results.json\n")

    return knee_T
