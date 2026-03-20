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

import torch
import os, sys, argparse, json
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG
from inference import load_model
from model.tokenizer import SanskritSourceTokenizer, SanskritTargetTokenizer

OUTPUT_DIR = "analysis/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Shared loader ─────────────────────────────────────────────────────

def load_everything(cfg, device):
    model_name = cfg['model_type']
    has_neg    = cfg['data']['include_negative_examples']
    ckpt       = f"results7/{model_name}_neg_{has_neg}/best_model.pt"
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"No checkpoint at {ckpt}. Train first.")
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
    from Data.data import OptimizedSanskritDataset
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


# ── Task 1 ────────────────────────────────────────────────────────────

def run_task1(model, src_tok, device):
    print("\n" + "="*65)
    print("  TASK 1 — KV Cache Benchmark")
    print("="*65)
    if not hasattr(model.model, 'generate_cached'):
        print("  SKIP: not D3PMCrossAttention.")
        return
    from analysis.kv_cache_benchmark import run_benchmark, print_summary
    results = run_benchmark(model, src_tok, device, src_lens=[16, 32, 64])
    print_summary(results)
    path = os.path.join(OUTPUT_DIR, "task1_kv_cache.txt")
    with open(path, "w") as f:
        f.write("TASK 1 — KV CACHE BENCHMARK\n" + "="*40 + "\n\n")
        f.write(f"{'src_len':>8}  {'standard(s)':>12}  {'cached(s)':>10}  "
                f"{'speedup':>8}  {'encoder%':>9}\n")
        for src_len, r in results.items():
            f.write(f"{src_len:>8}  {r['standard_s']:>12.3f}  {r['cached_s']:>10.3f}  "
                    f"{r['speedup']:>7.2f}x  {r['encoder_pct']:>8.1f}%\n")
    print(f"  Saved: {path}")


# ── Task 2 ────────────────────────────────────────────────────────────

def run_task2(model, src_tok, tgt_tok, device, input_text):
    print("\n" + "="*65)
    print("  TASK 2 — Attention Visualization + Semantic Drift")
    print("="*65)
    print(f"  Input: {input_text}")
    if not hasattr(model.model, 'encode_source'):
        print("  SKIP: not D3PMCrossAttention.")
        return

    src_ids    = src_tok.encode(input_text)
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_chars  = list(input_text.strip())

    from analysis.attention_viz import (AttentionCapture, plot_attn_heatmap,
                                         plot_attn_evolution, plot_all_layers)
    from analysis.semantic_drift import (capture_intermediate_outputs,
                                          compute_drift, compute_token_stability,
                                          plot_drift_curve)

    # Attention capture
    print("  Capturing attention weights...")
    capturer     = AttentionCapture(model)
    step_weights = capturer.capture(src_tensor, capture_every=10)

    with torch.no_grad():
        out_ids  = model.generate_cached(src_tensor)
    tgt_ids   = [x for x in out_ids[0].tolist() if x > 4]
    tgt_text  = tgt_tok.decode(tgt_ids).strip()
    tgt_chars = list(tgt_text)
    print(f"  Output: {tgt_text}")

    first_t = max(step_weights.keys())
    plot_attn_heatmap(step_weights, t_val=first_t, layer=0,
        src_tokens=src_chars[:20], tgt_tokens=tgt_chars[:20],
        save_path=os.path.join(OUTPUT_DIR, f"task2_attn_t{first_t}.png"),
        title=f"Attention t={first_t} (noisy)  Layer 0")
    plot_attn_heatmap(step_weights, t_val=0, layer=0,
        src_tokens=src_chars[:20], tgt_tokens=tgt_chars[:20],
        save_path=os.path.join(OUTPUT_DIR, "task2_attn_t0.png"),
        title="Attention t=0 (final)  Layer 0")
    plot_all_layers(step_weights, t_val=0,
        src_tokens=src_chars[:20], tgt_tokens=tgt_chars[:20],
        save_path=os.path.join(OUTPUT_DIR, "task2_all_layers_t0.png"))
    if len(src_chars) > 0 and len(tgt_chars) > 0:
        plot_attn_evolution(step_weights, src_token_idx=0, tgt_token_idx=0,
            layer=0, src_token_str=src_chars[0], tgt_token_str=tgt_chars[0],
            save_path=os.path.join(OUTPUT_DIR, "task2_attn_evolution.png"))

    # Semantic drift
    print("  Computing semantic drift...")
    step_outputs, final_out = capture_intermediate_outputs(
        model, src_tensor, tgt_tok, capture_every=5)
    drift   = compute_drift(step_outputs, final_out)
    stab    = compute_token_stability(step_outputs, final_out, tgt_tok)
    plot_drift_curve(drift, src_text=input_text,
        save_path=os.path.join(OUTPUT_DIR, "task2_semantic_drift.png"))

    print(f"  Lock-in timestep: t={drift['lock_in_t']}")
    print(f"  Mean position lock-in: t={stab['mean_lock_t']:.1f} ± {stab['std_lock_t']:.1f}")

    report = os.path.join(OUTPUT_DIR, "task2_report.txt")
    with open(report, "w", encoding="utf-8") as f:
        f.write("TASK 2 — ATTENTION + DRIFT REPORT\n" + "="*50 + "\n\n")
        f.write(f"Input  : {input_text}\nOutput : {final_out}\n\n")
        f.write(f"Lock-in t : {drift['lock_in_t']}\n")
        f.write(f"Mean pos lock-in : {stab['mean_lock_t']:.1f} ± {stab['std_lock_t']:.1f}\n\n")
        f.write("Step → Output → CER-to-final\n" + "-"*60 + "\n")
        for tv, cer in zip(drift["t_vals"], drift["cer_to_final"]):
            f.write(f"  t={tv:4d}  |  {step_outputs.get(tv,'')[:40]:40s}  |  {cer:.4f}\n")
    print(f"  Report: {report}")


# ── Task 3 ────────────────────────────────────────────────────────────

def run_task3(model, src_tok, tgt_tok, device, src_list, ref_list):
    print("\n" + "="*65)
    print("  TASK 3 — Concept Vectors + PCA Steering")
    print("="*65)
    if not hasattr(model.model, 'encode_source'):
        print("  SKIP: not D3PMCrossAttention.")
        return

    from analysis.concept_vectors import (collect_hidden_states, fit_pca,
        find_diversity_direction, generate_diversity_spectrum, plot_pca_space)

    # Collect hidden states from val set
    n = min(500, len(src_list))
    print(f"  Collecting hidden states from {n} examples...")
    hidden, _ = collect_hidden_states(
        model, src_list[:n], t_capture=0, max_samples=n)

    # Compute output lengths for diversity direction
    lengths = []
    for src in src_list[:n]:
        with torch.no_grad():
            out = model.generate_cached(src.to(device))
        ids = [x for x in out[0].tolist() if x > 4]
        lengths.append(len(tgt_tok.decode(ids)))

    # Fit PCA + find diversity direction
    pca = fit_pca(hidden, n_components=min(50, n-1))
    direction, best_pc, corr = find_diversity_direction(hidden, lengths, pca)

    # Plot concept space
    plot_pca_space(hidden, lengths, pca, best_pc,
        save_path=os.path.join(OUTPUT_DIR, "task3_concept_space.png"))

    # Generate diversity spectrum for first example
    print("\n  Diversity spectrum for first example:")
    src0  = src_list[0]
    inp0  = src_tok.decode([x for x in src0[0].tolist() if x > 4])
    print(f"  Input: {inp0}")
    spectrum = generate_diversity_spectrum(
        model, src0.to(device), direction, tgt_tok,
        alphas=[-2.0, -1.0, 0.0, 1.0, 2.0])

    # Save diversity direction + results
    np.save(os.path.join(OUTPUT_DIR, "task3_diversity_direction.npy"), direction)

    report = os.path.join(OUTPUT_DIR, "task3_report.txt")
    with open(report, "w", encoding="utf-8") as f:
        f.write("TASK 3 — CONCEPT VECTORS + PCA STEERING\n" + "="*50 + "\n\n")
        f.write(f"PCA: {pca.n_components_} components, "
                f"{pca.explained_variance_ratio_.sum()*100:.1f}% variance\n")
        f.write(f"Diversity PC: {best_pc}  (|r|={corr:.3f} with output length)\n\n")
        f.write("Diversity spectrum:\n")
        for alpha, text in sorted(spectrum.items()):
            f.write(f"  alpha={alpha:+.1f}  →  {text}\n")
    print(f"  Report: {report}")


# ── Task 4 ────────────────────────────────────────────────────────────

def run_task4(phase, model, src_tok, tgt_tok, device, cfg,
              src_list, ref_list):
    print("\n" + "="*65)
    print(f"  TASK 4 — Step Ablation  (phase={phase})")
    print("="*65)

    from analysis.step_ablation import (generate_ablation_configs,
        run_ablation_analysis, plot_ablation_3d, run_adversarial_test)

    if phase == "generate_configs":
        print("  Generating ablation configs...")
        generate_ablation_configs(output_dir="ablation_configs")
        print("\n  NEXT STEPS:")
        print("  1. bash ablation_configs/train_all.sh")
        print("  2. python analysis/run_analysis.py --task 4 --phase analyze")

    elif phase == "analyze":
        # Check which models exist
        existing = [T for T in [4, 8, 16, 32, 64]
                    if os.path.exists(f"ablation_results/T{T}/best_model.pt")]
        if not existing:
            print("  No ablation models found at ablation_results/T*/best_model.pt")
            print("  Run: python analysis/run_analysis.py --task 4 --phase generate_configs")
            print("  Then: bash ablation_configs/train_all.sh")
            return

        print(f"  Found models for T={existing}")
        results = run_ablation_analysis(
            ablation_dir="ablation_results", base_cfg=cfg,
            src_list=src_list[:200], ref_list=ref_list[:200],
            tgt_tokenizer=tgt_tok, device=device,
            output_dir=OUTPUT_DIR)
        plot_ablation_3d(results,
            save_path=os.path.join(OUTPUT_DIR, "task4_ablation_3d.png"))

    # Adversarial robustness always runs on existing model (no retraining)
    print("\n  Running adversarial robustness test...")
    inp_texts = [src_tok.decode([x for x in s[0].tolist() if x > 4])
                 for s in src_list[:50]]
    run_adversarial_test(
        model, src_tok, tgt_tok,
        test_inputs=inp_texts, test_refs=ref_list[:50],
        device=device, output_dir=OUTPUT_DIR)


# ── Task 5 ────────────────────────────────────────────────────────────

def run_task5(model, src_tok, tgt_tok, device, cfg, src_list, ref_list):
    print("\n" + "="*65)
    print("  TASK 5 — Classifier-Free Guidance")
    print("="*65)
    if not hasattr(model.model, 'encode_source'):
        print("  SKIP: not D3PMCrossAttention.")
        return

    from analysis.quality_classifier import (
        QualityClassifier, collect_quality_data,
        train_quality_classifier, sweep_guidance_scales)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",
        choices=["1","2","3","4","5","all"], default="all")
    parser.add_argument("--input",
        default="dharmo rakṣati rakṣitaḥ",
        help="IAST input text for Task 2")
    parser.add_argument("--phase",
        choices=["generate_configs", "analyze"], default="analyze",
        help="Task 4 phase: generate_configs (before training) or analyze (after)")
    args = parser.parse_args()

    cfg    = CONFIG
    device = torch.device(cfg['training']['device'])

    print("Loading model and tokenizers...")
    model, src_tok, tgt_tok, cfg = load_everything(cfg, device)

    # Load val data for tasks that need it (Tasks 3, 4, 5)
    needs_data = args.task in ("3", "4", "5", "all")
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
            run_task2(model, src_tok, tgt_tok, device, args.input)
        elif task == "3":
            run_task3(model, src_tok, tgt_tok, device, src_list, ref_list)
        elif task == "4":
            run_task4(args.phase, model, src_tok, tgt_tok, device, cfg,
                      src_list, ref_list)
        elif task == "5":
            run_task5(model, src_tok, tgt_tok, device, cfg, src_list, ref_list)

    print(f"\n{'='*65}")
    print(f"  All outputs saved to: {OUTPUT_DIR}/")
    print("="*65)


if __name__ == "__main__":
    main()
