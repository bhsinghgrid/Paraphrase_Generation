"""
analysis/quality_classifier.py
================================
Task 5: Classifier-Free Guidance for Paraphrase Quality Control

Two steps — only Step 2 requires training a SMALL model (not the main D3PM):

STEP 1 — Collect training data (no training):
  Run existing model on val set, record (hidden_state, CER) pairs.
  Hidden states come from model.model._last_hidden after forward_cached().
  CER score = quality label (lower CER = higher quality).

STEP 2 — Train quality classifier:
  Small 2-layer MLP: d_model → 64 → 1
  Input: pooled decoder hidden state [B, d_model]
  Output: predicted quality score in [0, 1]  (1 = high quality)
  Loss: MSE against normalized CER labels
  Training time: ~5-10 minutes on CPU for 10k examples

STEP 3 — Guided inference (no retraining):
  At each diffusion step, use classifier gradient to shift logits:
    guided_logits = logits + λ * ∂(quality_score)/∂(logits)
  Higher λ → model biased toward high-quality outputs
  λ=0 → standard generation (no guidance)

Key: main D3PM model is FROZEN throughout. Only the 10k-param classifier trains.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
from typing import List, Dict, Optional, Tuple


# ── Quality classifier architecture ──────────────────────────────────

class QualityClassifier(nn.Module):
    """
    Lightweight MLP that predicts transliteration quality from decoder
    hidden states.

    Architecture:
      d_model → 128 → 64 → 1 → Sigmoid

    Input:  mean-pooled decoder hidden state [B, d_model]
    Output: quality score [B, 1] ∈ [0, 1]  (1 = high quality)

    ~10k parameters. Trains in minutes on CPU.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.d_model = d_model

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden : [B, tgt_len, d_model] OR [B, d_model] (already pooled)

        Returns:
            score : [B, 1] quality score in [0, 1]
        """
        if hidden.dim() == 3:
            # Pool over sequence length
            hidden = hidden.mean(dim=1)   # [B, d_model]
        return self.net(hidden)           # [B, 1]


# ── Training data collection ──────────────────────────────────────────

@torch.no_grad()
def collect_quality_data(
    model,
    src_list:      List[torch.Tensor],
    ref_list:      List[str],
    tgt_tokenizer,
    t_capture:     int   = 0,
    temperature:   float = 0.8,
    top_k:         int   = 40,
    max_samples:   int   = 5000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect (hidden_state, quality_score) pairs for classifier training.

    For each sample:
      1. Run generate_cached() on src
      2. Capture decoder hidden state at t=t_capture
      3. Compute CER between output and reference
      4. Quality = 1 - CER  (normalize to [0,1])

    Args:
        model         : SanskritModel
        src_list      : list of [1, src_len] tensors
        ref_list      : list of reference Devanagari strings
        tgt_tokenizer : SanskritTargetTokenizer
        t_capture     : which step to capture hidden states (0 = final)
        max_samples   : cap number of training examples

    Returns:
        hidden_matrix : np.ndarray [N, d_model]
        quality_scores: np.ndarray [N]  values in [0, 1]
    """
    inner  = model.model
    T      = inner.scheduler.num_timesteps
    device = next(inner.parameters()).device

    hidden_list  = []
    quality_list = []
    n            = min(len(src_list), max_samples)

    def cer(pred, ref):
        if not ref:
            return 1.0
        def ed(s1, s2):
            m, n = len(s1), len(s2)
            dp = list(range(n + 1))
            for i in range(1, m + 1):
                prev, dp[0] = dp[0], i
                for j in range(1, n + 1):
                    temp = dp[j]
                    dp[j] = prev if s1[i-1] == s2[j-1] else 1 + min(prev, dp[j], dp[j-1])
                    prev = temp
            return dp[n]
        return ed(pred, ref) / max(len(ref), 1)

    print(f"Collecting quality data from {n} examples...")
    for i, (src, ref) in enumerate(zip(src_list[:n], ref_list[:n])):
        if i % 200 == 0:
            print(f"  {i}/{n}")

        if src.dim() == 1:
            src = src.unsqueeze(0)
        src = src.to(device)

        B       = src.shape[0]
        tgt_len = inner.max_seq_len
        mask_id = inner.mask_token_id

        memory, src_pad_mask = inner.encode_source(src)
        x0_est  = torch.full((B, tgt_len), mask_id, dtype=torch.long, device=device)
        hint    = None
        h_cap   = None

        for t_val in range(T - 1, -1, -1):
            t       = torch.full((B,), t_val, dtype=torch.long, device=device)
            is_last = (t_val == 0)

            logits, _ = inner.forward_cached(
                memory, src_pad_mask, x0_est, t,
                x0_hint=hint, inference_mode=True,
            )

            if t_val == t_capture and hasattr(inner, '_last_hidden'):
                h_cap = inner._last_hidden[0].mean(dim=0).detach().cpu()  # [d_model]

            logits = logits / max(temperature, 1e-8)
            if top_k > 0:
                V = logits.shape[-1]
                if top_k < V:
                    vals, _ = torch.topk(logits, top_k, dim=-1)
                    logits  = logits.masked_fill(logits < vals[..., -1:], float('-inf'))

            probs  = F.softmax(logits, dim=-1)
            x0_est = torch.argmax(probs, dim=-1) if is_last else _sample(probs)
            hint   = x0_est

        if h_cap is None:
            continue

        ids  = [x for x in x0_est[0].tolist() if x > 4]
        pred = tgt_tokenizer.decode(ids).strip()
        q    = max(0.0, 1.0 - cer(pred, ref))   # quality = 1 - CER

        hidden_list.append(h_cap.numpy())
        quality_list.append(q)

    print(f"Collected {len(hidden_list)} quality examples.")
    print(f"Quality stats: mean={np.mean(quality_list):.3f}  "
          f"min={np.min(quality_list):.3f}  max={np.max(quality_list):.3f}")

    return np.stack(hidden_list), np.array(quality_list, dtype=np.float32)


def _sample(probs):
    B, L, V = probs.shape
    flat    = probs.view(B * L, V).clamp(min=1e-9)
    flat    = flat / flat.sum(dim=-1, keepdim=True)
    return torch.multinomial(flat, 1).squeeze(-1).view(B, L)


# ── Training ──────────────────────────────────────────────────────────

def train_quality_classifier(
    hidden_matrix:  np.ndarray,
    quality_scores: np.ndarray,
    d_model:        int,
    epochs:         int   = 30,
    batch_size:     int   = 64,
    lr:             float = 1e-3,
    val_frac:       float = 0.1,
    save_path:      Optional[str] = None,
) -> QualityClassifier:
    """
    Train QualityClassifier on collected (hidden, quality) pairs.

    Args:
        hidden_matrix  : [N, d_model] from collect_quality_data()
        quality_scores : [N] quality labels in [0, 1]
        d_model        : hidden dimension
        epochs         : training epochs
        save_path      : if given, save trained classifier weights here

    Returns:
        trained QualityClassifier
    """
    device = torch.device("cpu")   # classifier is tiny, CPU is fine

    X = torch.tensor(hidden_matrix, dtype=torch.float32)
    y = torch.tensor(quality_scores, dtype=torch.float32).unsqueeze(-1)

    N     = len(X)
    n_val = max(1, int(N * val_frac))
    idx   = torch.randperm(N)
    val_idx   = idx[:n_val]
    train_idx = idx[n_val:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]

    clf       = QualityClassifier(d_model).to(device)
    optimizer = torch.optim.Adam(clf.parameters(), lr=lr)

    print(f"\nTraining QualityClassifier: {sum(p.numel() for p in clf.parameters())} params")
    print(f"Train: {len(X_train)}  Val: {len(X_val)}")

    best_val_loss = float('inf')
    best_state    = None

    for epoch in range(epochs):
        clf.train()
        perm       = torch.randperm(len(X_train))
        train_loss = 0.0
        n_batches  = 0

        for start in range(0, len(X_train), batch_size):
            batch_idx = perm[start:start + batch_size]
            xb, yb    = X_train[batch_idx], y_train[batch_idx]
            pred      = clf(xb)
            loss      = F.mse_loss(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches  += 1

        clf.eval()
        with torch.no_grad():
            val_pred = clf(X_val)
            val_loss = F.mse_loss(val_pred, y_val).item()

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"  Ep {epoch+1:3d}  train={train_loss/n_batches:.4f}  val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in clf.state_dict().items()}

    if best_state:
        clf.load_state_dict(best_state)
        print(f"  Best val loss: {best_val_loss:.4f}")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save(clf.state_dict(), save_path)
        print(f"  Classifier saved: {save_path}")

    return clf


# ── Guided inference ──────────────────────────────────────────────────

def generate_guided(
    model,
    src:        torch.Tensor,
    classifier: QualityClassifier,
    guidance_scale: float = 1.0,
    temperature:    float = 0.8,
    top_k:          int   = 40,
) -> torch.Tensor:
    """
    Classifier-guided generation.

    At each diffusion step:
      1. Run forward_cached() → logits, hidden states
      2. Compute classifier gradient: ∂(quality_score) / ∂(hidden)
      3. Project gradient back to logit space (approximate)
      4. guided_logits = logits + λ * gradient_signal
      5. Sample from guided_logits

    guidance_scale λ:
      0.0 → no guidance (standard generation)
      0.5 → weak guidance
      1.0 → moderate guidance (recommended starting point)
      2.0 → strong guidance (may reduce diversity)
      3.0 → very strong (may collapse to repetitive output)

    Args:
        model           : SanskritModel (frozen)
        src             : [1, src_len] IAST token ids
        classifier      : trained QualityClassifier
        guidance_scale  : λ — guidance strength

    Returns:
        x0_est : [1, tgt_len] generated token ids
    """
    inner  = model.model
    T      = inner.scheduler.num_timesteps
    device = next(inner.parameters()).device
    clf_device = next(classifier.parameters()).device

    if src.dim() == 1:
        src = src.unsqueeze(0)
    src = src.to(device)

    B       = src.shape[0]
    tgt_len = inner.max_seq_len
    mask_id = inner.mask_token_id

    memory, src_pad_mask = inner.encode_source(src)
    x0_est  = torch.full((B, tgt_len), mask_id, dtype=torch.long, device=device)
    hint    = None

    inner.eval()
    classifier.eval()

    for t_val in range(T - 1, -1, -1):
        t       = torch.full((B,), t_val, dtype=torch.long, device=device)
        is_last = (t_val == 0)

        if guidance_scale > 0.0:
            # Need gradients for classifier guidance
            with torch.enable_grad():
                # Run forward_cached and get hidden states
                PAD = 1
                if t_val > 0:
                    _, x_t_ids = inner.forward_process.q_sample(x0_est, t)
                else:
                    x_t_ids = x0_est

                x      = inner.tgt_embed(x_t_ids)
                t_norm = t.float() / T
                t_emb  = inner.time_mlp(t_norm.unsqueeze(-1))
                x      = x + t_emb.unsqueeze(1)

                if hint is not None:
                    hint_emb = inner.tgt_embed(hint)
                    gate     = inner.hint_gate(x)
                    x        = x + gate * hint_emb

                for block in inner.decoder_blocks:
                    x = block(x, memory, tgt_pad_mask=None, src_pad_mask=src_pad_mask)

                # hidden: [B, tgt_len, d_model] — detach from graph for clf
                hidden = x.detach().requires_grad_(True).to(clf_device)

                # Classifier quality score
                quality = classifier(hidden)   # [B, 1]
                quality.sum().backward()

                # Gradient of quality w.r.t. hidden: [B, tgt_len, d_model]
                grad = hidden.grad.to(device)   # [B, tgt_len, d_model]

                # Project gradient to logit space via output head weight
                # logit_grad ≈ grad @ head.weight   [B, tgt_len, tgt_vocab]
                logit_grad = grad @ inner.head.weight.T

                # Compute standard logits (no gradient needed)
                with torch.no_grad():
                    logits = inner.head(x)

                # Apply guidance
                logits = logits + guidance_scale * logit_grad

        else:
            with torch.no_grad():
                logits, _ = inner.forward_cached(
                    memory, src_pad_mask, x0_est, t,
                    x0_hint=hint, inference_mode=True,
                )

        with torch.no_grad():
            logits = logits / max(temperature, 1e-8)
            if top_k > 0:
                V = logits.shape[-1]
                if top_k < V:
                    vals, _ = torch.topk(logits, top_k, dim=-1)
                    logits  = logits.masked_fill(logits < vals[..., -1:], float('-inf'))

            probs  = F.softmax(logits, dim=-1)
            x0_est = torch.argmax(probs, dim=-1) if is_last else _sample_no_grad(probs)
            hint   = x0_est

    return x0_est


def _sample_no_grad(probs):
    B, L, V = probs.shape
    flat    = probs.view(B * L, V).clamp(min=1e-9)
    flat    = flat / flat.sum(dim=-1, keepdim=True)
    return torch.multinomial(flat, 1).squeeze(-1).view(B, L)


# ── Guidance scale sweep ──────────────────────────────────────────────

def sweep_guidance_scales(
    model,
    classifier: QualityClassifier,
    src_list:   List[torch.Tensor],
    ref_list:   List[str],
    tgt_tokenizer,
    scales:     List[float] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
    n_samples:  int         = 50,
    device:     torch.device = None,
    output_dir: str          = "analysis/outputs",
) -> Dict:
    """
    Evaluate CER at each guidance scale.
    Produces quality-diversity tradeoff plot.
    """
    def cer(pred, ref):
        if not ref:
            return 1.0
        def ed(s1, s2):
            m, n = len(s1), len(s2)
            dp = list(range(n + 1))
            for i in range(1, m + 1):
                prev, dp[0] = dp[0], i
                for j in range(1, n + 1):
                    temp = dp[j]
                    dp[j] = prev if s1[i-1] == s2[j-1] else 1 + min(prev, dp[j], dp[j-1])
                    prev = temp
            return dp[n]
        return ed(pred, ref) / max(len(ref), 1)

    device  = device or next(model.parameters()).device
    results = {}
    n       = min(n_samples, len(src_list))

    print("\nGuidance scale sweep...")
    for scale in scales:
        cer_list   = []
        output_set = []
        for src, ref in zip(src_list[:n], ref_list[:n]):
            if src.dim() == 1:
                src = src.unsqueeze(0)
            out      = generate_guided(model, src.to(device), classifier,
                                        guidance_scale=scale)
            ids      = [x for x in out[0].tolist() if x > 4]
            pred     = tgt_tokenizer.decode(ids).strip()
            cer_list.append(cer(pred, ref))
            output_set.append(pred)

        mean_cer = float(np.mean(cer_list))

        # Self-diversity: unique outputs / total (proxy for diversity)
        unique_frac = len(set(output_set)) / max(len(output_set), 1)

        results[scale] = {"mean_cer": mean_cer, "diversity": unique_frac}
        print(f"  λ={scale:.1f}  CER={mean_cer:.4f}  diversity={unique_frac:.3f}")

    # Plot
    os.makedirs(output_dir, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        sc_list  = sorted(results.keys())
        cers     = [results[s]["mean_cer"]   for s in sc_list]
        diversities = [results[s]["diversity"] for s in sc_list]

        ax1.plot(sc_list, cers, 'o-', color='coral', linewidth=1.8, markersize=7)
        ax1.set_xlabel("Guidance scale λ", fontsize=10)
        ax1.set_ylabel("CER (↓ better)", fontsize=10)
        ax1.set_title("Quality vs guidance scale", fontsize=10)

        ax2.plot(sc_list, diversities, 'o-', color='steelblue', linewidth=1.8, markersize=7)
        ax2.set_xlabel("Guidance scale λ", fontsize=10)
        ax2.set_ylabel("Output diversity (unique fraction)", fontsize=10)
        ax2.set_title("Diversity vs guidance scale", fontsize=10)

        plt.suptitle("Quality-Diversity Tradeoff (Guidance Scale Sweep)", fontsize=11)
        plt.tight_layout()
        path = os.path.join(output_dir, "guidance_scale_sweep.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")
    except ImportError:
        pass

    with open(os.path.join(output_dir, "guidance_results.json"), "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)

    return results
