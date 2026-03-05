"""
inference.py — Correctly Fixed Sanskrit D3PM Inference
=======================================================

ROOT CAUSE OF GARBAGE OUTPUT (now fixed):
------------------------------------------
The model's forward() method internally applies q_sample(tgt, t):

    _, x_t_ids = self.forward_process.q_sample(tgt, t)   # inside D3PMCrossAttention
    x = self.tgt_embed(x_t_ids)

This means:
  • During TRAINING : tgt = clean target  → q_sample adds noise → model denoises
  • During INFERENCE: we were passing x_t (already noisy) as tgt
                      → q_sample added MORE noise → model saw double-noise → garbage

CORRECT INFERENCE LOOP:
  1. Start with x0_estimate = all [MASK]  (we know nothing about the output)
  2. At each timestep t (high → low):
       - Pass x0_estimate as tgt  (model noises it to the right level internally)
       - Model predicts new x0
       - Update x0_estimate ← new x0 prediction
  3. Return x0_estimate at t=0

This exactly mirrors training: the model always sees q_sample(clean_guess, t),
and iteratively refines its clean guess from all-mask → full Devanagari.

Also included: robust checkpoint loading (auto-detects d_model, n_layers,
max_seq_len from weight shapes — no more size-mismatch errors).
"""

import torch
import torch.nn.functional as F
import os
import sys
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from config import CONFIG
except ImportError:
    CONFIG = {}


# ──────────────────────────────────────────────────────────────────────
# Architecture auto-detection from checkpoint
# ──────────────────────────────────────────────────────────────────────

def infer_cfg_from_checkpoint(ckpt_path: str, base_cfg: dict) -> dict:
    """Read weight shapes from checkpoint → build matching architecture config."""
    import copy
    cfg = copy.deepcopy(base_cfg)

    print(f"🔍 Auto-detecting architecture from: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")

    # d_model & vocab_size from embedding weight
    emb_key = "model.src_embed.token_emb.weight"
    if emb_key in state:
        vocab_size, d_model = state[emb_key].shape
        cfg["model"]["vocab_size"] = vocab_size
        cfg["model"]["d_model"]    = d_model
        cfg["model"]["d_ff"]       = d_model * 4
        print(f"   vocab_size  = {vocab_size}")
        print(f"   d_model     = {d_model},  d_ff = {d_model * 4}")

    # n_layers from number of encoder blocks
    enc_indices = {
        int(k.split(".")[2])
        for k in state
        if k.startswith("model.encoder_blocks.")
    }
    if enc_indices:
        n_layers = max(enc_indices) + 1
        cfg["model"]["n_layers"] = n_layers
        print(f"   n_layers    = {n_layers}")

    # max_seq_len from positional encoding
    pe_key = "model.src_embed.pos_enc.pe"
    if pe_key in state:
        max_seq_len = state[pe_key].shape[1]
        cfg["model"]["max_seq_len"] = max_seq_len
        print(f"   max_seq_len = {max_seq_len}")

    # n_heads: find largest divisor of d_model ≤ existing setting
    d = cfg["model"]["d_model"]
    preferred = cfg["model"].get("n_heads", 8)
    if d % preferred != 0:
        for h in [8, 6, 4, 2, 1]:
            if d % h == 0:
                preferred = h
                break
    cfg["model"]["n_heads"] = preferred
    print(f"   n_heads     = {preferred}")
    print(f"   diffusion_steps = {cfg['model'].get('diffusion_steps', '?')} (from CONFIG)")
    return cfg


def load_model_from_checkpoint(ckpt_path: str, base_cfg: dict, device: torch.device):
    """Build model from checkpoint shapes, then load weights (no mismatches)."""
    from model.sanskrit_model import SanskritModel

    cfg   = infer_cfg_from_checkpoint(ckpt_path, base_cfg)
    model = SanskritModel(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device)

    missing, unexpected = model.load_state_dict(state, strict=False)

    # hint_gate is the only key that legitimately may be absent
    allowed_missing   = {"model.hint_gate.0.weight", "model.hint_gate.0.bias"}
    truly_missing     = [k for k in missing if k not in allowed_missing]

    if truly_missing:
        print(f"⚠️  Truly missing keys ({len(truly_missing)}): {truly_missing[:3]} …")
    if unexpected:
        print(f"⚠️  Unexpected keys ({len(unexpected)}): {unexpected[:3]} …")

    # Initialise hint_gate to safe identity-like default
    if hasattr(model.model, "hint_gate"):
        with torch.no_grad():
            w = model.model.hint_gate[0].weight
            torch.nn.init.zeros_(model.model.hint_gate[0].bias)
            if w.shape[0] == w.shape[1]:
                torch.nn.init.eye_(w)
            else:
                torch.nn.init.xavier_uniform_(w)
        if "model.hint_gate.0.weight" in missing:
            print("ℹ️  hint_gate not in checkpoint — initialised to identity default.")

    print("✅ Checkpoint loaded successfully!")
    return model, cfg


# ──────────────────────────────────────────────────────────────────────
# 🔥 CORRECT D3PM inference (matches training exactly)
# ──────────────────────────────────────────────────────────────────────

def run_d3pm_inference(
    model,
    input_ids: torch.Tensor,
    num_steps: int = 20,
    temperature: float = 0.8,
    top_k: int = 50,
    cfg: dict = None,
) -> torch.Tensor:
    """
    Correct D3PM iterative refinement.

    The model's forward() is:
        x_t_ids = q_sample(tgt, t)    ← model noises tgt internally
        logits  = decoder(x_t_ids)    ← predict x_0 from noisy input

    So we must pass our CURRENT BEST ESTIMATE of x_0 as `tgt`,
    NOT the noisy x_t.  The model handles noising itself.

    Algorithm:
        x0_est = all [MASK]                      # know nothing yet
        for t in T-1 → 0:
            logits = model(src, x0_est, t)       # model noises x0_est → predicts x0
            x0_est = sample(softmax(logits))     # refine our estimate
        return x0_est
    """
    cfg    = cfg or CONFIG
    device = input_ids.device
    B, L   = input_ids.shape

    inner   = model.model
    sched   = inner.scheduler
    mask_id = inner.mask_token_id
    T       = sched.num_timesteps

    # Timestep schedule: T-1 → 0
    step_size = max(1, T // num_steps)
    timesteps = list(range(T - 1, -1, -step_size))
    if timesteps[-1] != 0:
        timesteps.append(0)

    # Start: we know nothing → all masks as our initial "clean" estimate
    x0_est = torch.full((B, L), mask_id, dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        for step_idx, t_val in enumerate(timesteps):
            t            = torch.full((B,), t_val, dtype=torch.long, device=device)
            is_last_step = (step_idx == len(timesteps) - 1)

            # 🔥 KEY FIX: pass x0_est as tgt (model noises it internally via q_sample)
            # x0_hint = previous estimate for self-conditioning
            logits = model(input_ids, x0_est, t, x0_hint=x0_est)[0]  # [B, L, V]

            # Temperature scaling + top-k filtering
            logits = logits / max(temperature, 1e-5)
            if top_k > 0:
                logits = _top_k_filter(logits, top_k)
            probs = F.softmax(logits, dim=-1)  # [B, L, V]

            # Sample new x0 estimate
            if is_last_step:
                x0_est = torch.argmax(probs, dim=-1)   # greedy on last step
            else:
                x0_est = _batch_multinomial(probs)     # stochastic otherwise

        # Final cleanup: any remaining mask tokens → force argmax prediction
        # (handles edge case where stochastic sampling left mask tokens)
        still_masked = (x0_est == mask_id)
        if still_masked.any():
            final_t  = torch.zeros(B, dtype=torch.long, device=device)
            logits   = model(input_ids, x0_est, final_t, x0_hint=x0_est)[0]
            best_ids = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
            x0_est   = torch.where(still_masked, best_ids, x0_est)

    return x0_est  # [B, L] — clean predicted Devanagari token IDs


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    B, L, V = logits.shape
    if k >= V:
        return logits
    topk_vals, _ = torch.topk(logits, k, dim=-1)
    threshold     = topk_vals[..., -1].unsqueeze(-1)
    return logits.masked_fill(logits < threshold, float('-inf'))


def _batch_multinomial(probs: torch.Tensor) -> torch.Tensor:
    B, L, V = probs.shape
    flat     = probs.view(B * L, V) + 1e-9
    flat     = flat / flat.sum(dim=-1, keepdim=True)
    return torch.multinomial(flat, num_samples=1).squeeze(-1).view(B, L)


# ──────────────────────────────────────────────────────────────────────
# Interactive demo
# ──────────────────────────────────────────────────────────────────────

def interactive_demo():
    from model.tokenizer import SanskritTokenizer

    base_cfg   = CONFIG
    device     = torch.device(base_cfg["training"]["device"])
    model_name = base_cfg["model_type"]
    has_neg    = "True" if base_cfg["data"]["include_negative_examples"] else "False"
    exp_dir    = f"results/{model_name}_neg_{has_neg}"
    ckpt_path  = f"{exp_dir}/best_model.pt"

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}.\n"
            "Train the model first or correct the path."
        )

    model, cfg = load_model_from_checkpoint(ckpt_path, base_cfg, device)
    model.eval()

    tokenizer = SanskritTokenizer(cfg["model"]["vocab_size"])
    PAD_ID    = tokenizer.tokenizer.token_to_id("[PAD]") or 1
    MASK_ID   = cfg["diffusion"]["mask_token_id"]

    print("\n" + "="*60)
    print("Sanskrit D3PM Inference — IAST → Devanagari")
    print("="*60)
    print("Type IAST transliteration → Devanagari.  'quit' to exit.\n")

    while True:
        try:
            text = input("INPUT (IAST) > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in ("quit", "exit", "q") or not text:
            break

        ids = torch.tensor(
            [tokenizer.encode(text)[:cfg["model"]["max_seq_len"]]],
            dtype=torch.long, device=device
        )

        out   = run_d3pm_inference(
            model, ids,
            num_steps=cfg["model"].get("diffusion_steps", 16),
            temperature=0.8,
            top_k=50,
            cfg=cfg
        )
        clean = [i for i in out[0].tolist() if i not in (MASK_ID, PAD_ID)]
        print(f"PRED → {tokenizer.decode(clean).strip()}\n")


# ──────────────────────────────────────────────────────────────────────
# Batch evaluation
# ──────────────────────────────────────────────────────────────────────

def batch_evaluate(sample_size=1000, num_steps=None, temperature=0.8, top_k=50):
    try:
        import evaluate as hf_evaluate
        BERTSCORE_AVAILABLE = True
    except ImportError:
        BERTSCORE_AVAILABLE = False
        print("⚠️  `evaluate` not installed — BERTScore skipped.")

    try:
        from nltk.translate.bleu_score import corpus_bleu
        BLEU_AVAILABLE = True
    except ImportError:
        BLEU_AVAILABLE = False
        print("⚠️  `nltk` not installed — BLEU skipped.")

    from data.dataset import OptimizedSanskritDataset
    from model.tokenizer import SanskritTokenizer

    base_cfg   = CONFIG
    device     = torch.device(base_cfg["training"]["device"])
    model_name = base_cfg["model_type"]
    has_neg    = "True" if base_cfg["data"]["include_negative_examples"] else "False"
    exp_dir    = f"results/{model_name}_neg_{has_neg}"
    ckpt_path  = f"{exp_dir}/best_model.pt"

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    model, cfg = load_model_from_checkpoint(ckpt_path, base_cfg, device)
    model.eval()

    steps     = num_steps or cfg["model"].get("diffusion_steps", 16)
    tokenizer = SanskritTokenizer(cfg["model"]["vocab_size"])
    PAD_ID    = tokenizer.tokenizer.token_to_id("[PAD]") or 1
    MASK_ID   = cfg["diffusion"]["mask_token_id"]

    def collate(batch):
        return {
            "input_ids":   torch.stack([b["input_ids"].long() for b in batch]),
            "target_text": [b["target_text"] for b in batch],
            "input_text":  [b["input_text"]  for b in batch],
        }

    dataset = OptimizedSanskritDataset("test", tokenizer, cfg["model"]["max_seq_len"])
    indices = list(range(min(sample_size, len(dataset))))
    loader  = DataLoader(
        Subset(dataset, indices),
        batch_size=base_cfg["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate
    )

    all_preds, all_refs, all_inputs = [], [], []
    print(f"⏳ Generating {len(indices)} translations "
          f"(steps={steps}, temp={temperature}, top_k={top_k}) …")

    for batch in tqdm(loader):
        input_ids  = batch["input_ids"].to(device)
        output_ids = run_d3pm_inference(
            model, input_ids,
            num_steps=steps,
            temperature=temperature,
            top_k=top_k,
            cfg=cfg
        )
        for i in range(output_ids.size(0)):
            clean = [id_ for id_ in output_ids[i].tolist()
                     if id_ not in (MASK_ID, PAD_ID)]
            all_preds.append(tokenizer.decode(clean).strip())
            all_refs.append(batch["target_text"][i].strip())
            all_inputs.append(batch["input_text"][i].strip())

    bleu_score, bert_f1 = 0.0, 0.0
    if BLEU_AVAILABLE:
        bleu_score = corpus_bleu(
            [[r.split()] for r in all_refs],
            [p.split() for p in all_preds]
        )
    if BERTSCORE_AVAILABLE:
        print("⏳ Computing BERTScore …")
        results = hf_evaluate.load("bertscore").compute(
            predictions=all_preds, references=all_refs, lang="hi"
        )
        bert_f1 = sum(results["f1"]) / len(results["f1"])

    out_path = f"{exp_dir}/evaluation_results_fixed.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"=== FIXED EVALUATION: {model_name} ===\n")
        f.write(f"Detected arch   : d_model={cfg['model']['d_model']}, "
                f"n_layers={cfg['model']['n_layers']}, "
                f"max_seq_len={cfg['model']['max_seq_len']}\n")
        f.write(f"Inference steps : {steps}\n")
        f.write(f"Temperature     : {temperature}\n")
        f.write(f"Top-k           : {top_k}\n")
        f.write(f"BLEU Score      : {bleu_score:.4f}\n")
        f.write(f"BERTScore (F1)  : {bert_f1:.4f}\n\n")
        f.write("=== SAMPLE PREDICTIONS ===\n")
        for i in range(min(20, len(all_preds))):
            f.write(f"INPUT : {all_inputs[i]}\n")
            f.write(f"REF   : {all_refs[i]}\n")
            f.write(f"PRED  : {all_preds[i]}\n")
            f.write("-" * 60 + "\n")

    print(f"\n✅ Done!  Results → {out_path}")
    print(f"📊 BLEU: {bleu_score:.4f}  |  BERTScore: {bert_f1:.4f}")
    return all_preds, all_refs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="D3PM Sanskrit Inference")
    parser.add_argument("--mode",        choices=["demo", "eval"], default="demo")
    parser.add_argument("--steps",       type=int,   default=None,
                        help="Denoising steps (default: diffusion_steps from CONFIG)")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k",       type=int,   default=50)
    parser.add_argument("--samples",     type=int,   default=200)
    args = parser.parse_args()

    if args.mode == "demo":
        interactive_demo()
    else:
        batch_evaluate(
            sample_size=args.samples,
            num_steps=args.steps,
            temperature=args.temperature,
            top_k=args.top_k,
        )