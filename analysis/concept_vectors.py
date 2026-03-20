"""
analysis/concept_vectors.py
============================
Task 3: Concept Vector Extraction + Controlled Paraphrase Diversity

No retraining required. Uses decoder hidden states already computed
during generate_cached() — stored in model.model._last_hidden after
each forward_cached() call.

Steps:
  1. Collect hidden states from N examples at a fixed diffusion step
  2. Pool sequence dimension → [N, d_model] representation per example
  3. PCA → find principal directions in concept space
  4. Identify "diversity direction" (PC that best separates short/long outputs)
  5. Steer: at inference, shift hidden states along diversity direction
     before the output head projection
  6. Generate at 5 points along the direction, measure output diversity

Key insight: the diversity direction is found purely from model outputs
(no human annotation needed). We use output length as a proxy:
  short output  → low diversity (model collapsed to simple token)
  long output   → high diversity (model exploring more of the space)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple


# ── Hidden state collection ───────────────────────────────────────────

@torch.no_grad()
def collect_hidden_states(
    model,
    src_list:    List[torch.Tensor],
    t_capture:   int   = 0,
    temperature: float = 0.8,
    top_k:       int   = 40,
    max_samples: int   = 1000,
) -> Tuple[np.ndarray, List[str]]:
    """
    Run generate_cached() on a list of source tensors, collecting the
    decoder hidden state at timestep t_capture for each sample.

    Args:
        model      : SanskritModel (D3PMCrossAttention)
        src_list   : list of [1, src_len] tensors, one per sample
        t_capture  : which diffusion step to capture hidden states at
                     0 = final (clean), T-1 = noisy start
        temperature: sampling temperature
        top_k      : top-k filter
        max_samples: cap at this many samples

    Returns:
        hidden_matrix : np.ndarray [N, d_model] — pooled hidden states
        output_texts  : list of N decoded output strings (for diversity analysis)
    """
    inner   = model.model
    T       = inner.scheduler.num_timesteps
    device  = next(inner.parameters()).device

    hidden_list  = []
    output_list  = []

    n = min(len(src_list), max_samples)
    print(f"Collecting hidden states from {n} examples at t={t_capture}...")

    for i, src in enumerate(src_list[:n]):
        if i % 100 == 0:
            print(f"  {i}/{n}")

        if src.dim() == 1:
            src = src.unsqueeze(0)
        src = src.to(device)

        B       = src.shape[0]
        tgt_len = inner.max_seq_len
        mask_id = inner.mask_token_id

        # KV cache
        memory, src_pad_mask = inner.encode_source(src)

        x0_est = torch.full((B, tgt_len), mask_id, dtype=torch.long, device=device)
        hint   = None
        captured_hidden = None

        for t_val in range(T - 1, -1, -1):
            t       = torch.full((B,), t_val, dtype=torch.long, device=device)
            is_last = (t_val == 0)

            logits, _ = inner.forward_cached(
                memory, src_pad_mask, x0_est, t,
                x0_hint=hint, inference_mode=True,
            )

            # Capture hidden state at target step
            if t_val == t_capture and hasattr(inner, '_last_hidden'):
                captured_hidden = inner._last_hidden.detach().cpu()

            logits = logits / max(temperature, 1e-8)
            if top_k > 0:
                V = logits.shape[-1]
                if top_k < V:
                    vals, _ = torch.topk(logits, top_k, dim=-1)
                    logits  = logits.masked_fill(logits < vals[..., -1:], float('-inf'))

            probs  = F.softmax(logits, dim=-1)
            x0_est = torch.argmax(probs, dim=-1) if is_last else _sample(probs)
            hint   = x0_est

        # Pool hidden state over non-PAD positions → [d_model]
        if captured_hidden is not None:
            non_pad = (x0_est[0] > 1).cpu()   # [tgt_len] bool
            if non_pad.sum() > 0:
                h = captured_hidden[0][non_pad].mean(dim=0)   # [d_model]
            else:
                h = captured_hidden[0].mean(dim=0)
            hidden_list.append(h.numpy())

        # Decode output
        ids  = [x for x in x0_est[0].tolist() if x > 4]

    print(f"Collected {len(hidden_list)} hidden states.")
    return np.stack(hidden_list), output_list


# ── PCA on hidden states ──────────────────────────────────────────────

def fit_pca(
    hidden_matrix: np.ndarray,
    n_components:  int = 50,
) -> object:
    """
    Fit PCA on hidden state matrix.

    Args:
        hidden_matrix : [N, d_model]
        n_components  : number of PCA components to retain

    Returns:
        fitted sklearn PCA object
    """
    from sklearn.decomposition import PCA
    n_comp = min(n_components, hidden_matrix.shape[0] - 1, hidden_matrix.shape[1])
    pca    = PCA(n_components=n_comp)
    pca.fit(hidden_matrix)
    print(f"PCA fit: {n_comp} components explain "
          f"{pca.explained_variance_ratio_.sum()*100:.1f}% of variance.")
    return pca


def find_diversity_direction(
    hidden_matrix: np.ndarray,
    output_lengths: List[int],
    pca:           object,
) -> np.ndarray:
    """
    Find the PCA direction that best correlates with output diversity
    (measured by output length as proxy).

    Projects hidden states into PCA space, then finds the PC whose
    scores have highest Spearman correlation with output lengths.

    Returns:
        direction : np.ndarray [d_model] — diversity direction in original space
    """
    from scipy.stats import spearmanr

    projected = pca.transform(hidden_matrix)   # [N, n_components]
    lengths   = np.array(output_lengths)

    correlations = []
    for pc_idx in range(projected.shape[1]):
        r, _ = spearmanr(projected[:, pc_idx], lengths)
        correlations.append(abs(r))

    best_pc = int(np.argmax(correlations))
    print(f"Diversity direction: PC {best_pc}  "
          f"(|r|={correlations[best_pc]:.3f} with output length)")

    # Map back to original d_model space
    direction = pca.components_[best_pc]   # [d_model]
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    return direction, best_pc, correlations[best_pc]


# ── Steered generation ────────────────────────────────────────────────

@torch.no_grad()
def generate_steered(
    model,
    src:       torch.Tensor,
    direction: np.ndarray,
    alpha:     float = 0.0,
    temperature: float = 0.8,
    top_k:     int   = 40,
) -> torch.Tensor:
    """
    Generate output while steering hidden states along diversity direction.

    At each diffusion step, after the decoder runs, we shift the hidden state
    by alpha * direction before projecting to logits.

    alpha > 0 → push toward high-diversity output
    alpha < 0 → push toward low-diversity output
    alpha = 0 → standard generation (no steering)

    Args:
        model     : SanskritModel (D3PMCrossAttention)
        src       : [1, src_len] IAST token ids
        direction : [d_model] diversity direction from find_diversity_direction()
        alpha     : steering strength
        temperature / top_k: sampling params

    Returns:
        x0_est : [1, tgt_len] generated token ids
    """
    inner   = model.model
    T       = inner.scheduler.num_timesteps
    device  = next(inner.parameters()).device

    if src.dim() == 1:
        src = src.unsqueeze(0)
    src = src.to(device)

    B       = src.shape[0]
    tgt_len = inner.max_seq_len
    mask_id = inner.mask_token_id

    dir_tensor = torch.tensor(direction, dtype=torch.float32, device=device)

    memory, src_pad_mask = inner.encode_source(src)
    x0_est = torch.full((B, tgt_len), mask_id, dtype=torch.long, device=device)
    hint   = None

    inner.eval()
    for t_val in range(T - 1, -1, -1):
        t       = torch.full((B,), t_val, dtype=torch.long, device=device)
        is_last = (t_val == 0)

        # Standard forward_cached but we intercept hidden states
        PAD = 1
        tgt_pad_mask = None  # inference_mode

        _, x_t_ids = inner.forward_process.q_sample(x0_est, t) if t_val > 0 else \
                     (None, x0_est)
        x      = inner.tgt_embed(x_t_ids)
        t_norm = t.float() / inner.scheduler.num_timesteps
        t_emb  = inner.time_mlp(t_norm.unsqueeze(-1))
        x      = x + t_emb.unsqueeze(1)

        if hint is not None:
            hint_emb = inner.tgt_embed(hint)
            gate     = inner.hint_gate(x)
            x        = x + gate * hint_emb

        for block in inner.decoder_blocks:
            x = block(x, memory, tgt_pad_mask=tgt_pad_mask, src_pad_mask=src_pad_mask)

        # ── STEERING: shift hidden states along diversity direction ───
        if alpha != 0.0:
            x = x + alpha * dir_tensor.unsqueeze(0).unsqueeze(0)

        # Project to logits using the head
        logits = inner.head(x)

        logits = logits / max(temperature, 1e-8)
        if top_k > 0:
            V = logits.shape[-1]
            if top_k < V:
                vals, _ = torch.topk(logits, top_k, dim=-1)
                logits  = logits.masked_fill(logits < vals[..., -1:], float('-inf'))

        probs  = F.softmax(logits, dim=-1)
        x0_est = torch.argmax(probs, dim=-1) if is_last else _sample(probs)
        hint   = x0_est

    return x0_est


def generate_diversity_spectrum(
    model,
    src:           torch.Tensor,
    direction:     np.ndarray,
    tgt_tokenizer,
    alphas:        List[float] = [-2.0, -1.0, 0.0, 1.0, 2.0],
    temperature:   float       = 0.8,
    top_k:         int         = 40,
) -> Dict[float, str]:
    """
    Generate outputs at 5 points along the diversity direction.

    Args:
        alphas : steering strengths (negative = low diversity, positive = high)

    Returns:
        dict mapping alpha → decoded Devanagari string
    """
    results = {}
    for alpha in alphas:
        out_ids  = generate_steered(model, src, direction, alpha, temperature, top_k)
        ids      = [x for x in out_ids[0].tolist() if x > 4]
        text     = tgt_tokenizer.decode(ids).strip()
        results[alpha] = text
        print(f"  alpha={alpha:+.1f}  → {text}")
    return results


def plot_pca_space(
    hidden_matrix:  np.ndarray,
    output_lengths: List[int],
    pca:            object,
    diversity_pc:   int,
    save_path:      Optional[str] = None,
):
    """
    Scatter plot of examples in PC1 vs PC2 space, coloured by output length.
    Highlights the diversity direction.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("pip install matplotlib.")
        return

    projected = pca.transform(hidden_matrix)   # [N, n_pc]
    lengths   = np.array(output_lengths)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: PC0 vs PC1 coloured by length
    ax = axes[0]
    sc = ax.scatter(projected[:, 0], projected[:, 1],
                    c=lengths, cmap='viridis', alpha=0.6, s=15)
    plt.colorbar(sc, ax=ax, label="Output length (chars)")
    ax.set_xlabel(f"PC0 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=10)
    ax.set_ylabel(f"PC1 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=10)
    ax.set_title("Concept space (PC0 vs PC1)", fontsize=11)

    # Right: explained variance
    ax2 = axes[1]
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    ax2.plot(range(1, len(cumvar)+1), cumvar, linewidth=1.5, color='steelblue')
    ax2.axvline(diversity_pc, color='coral', linestyle='--', label=f"Diversity PC={diversity_pc}")
    ax2.set_xlabel("Number of PCs", fontsize=10)
    ax2.set_ylabel("Cumulative variance (%)", fontsize=10)
    ax2.set_title("PCA explained variance", fontsize=11)
    ax2.legend()
    ax2.set_ylim(0, 102)

    plt.tight_layout()
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def _sample(probs):
    B, L, V = probs.shape
    flat    = probs.view(B * L, V).clamp(min=1e-9)
    flat    = flat / flat.sum(dim=-1, keepdim=True)
    return torch.multinomial(flat, 1).squeeze(-1).view(B, L)
