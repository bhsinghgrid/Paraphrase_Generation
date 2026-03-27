"""
Task 3: Concept Vector Extraction + Controlled Paraphrase Diversity
Fully corrected & production-ready version
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional


# ─────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────

def _sample(probs: torch.Tensor) -> torch.Tensor:
    B, L, V = probs.shape
    flat = probs.view(B * L, V).clamp(min=1e-9)
    flat = flat / flat.sum(dim=-1, keepdim=True)
    return torch.multinomial(flat, 1).squeeze(-1).view(B, L)


# ─────────────────────────────────────────────────────────────
# 1. Collect Hidden States
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_hidden_states(
    model,
    src_list: List[torch.Tensor],
    tgt_tokenizer,
    t_capture: int = 0,
    temperature: float = 0.8,
    top_k: int = 40,
    max_samples: int = 1000,
) -> Tuple[np.ndarray, List[str], List[int]]:
    """
    Collect pooled hidden representations + outputs
    """

    inner = model.model
    device = next(inner.parameters()).device
    T = inner.scheduler.num_timesteps

    hidden_list = []
    texts = []
    lengths = []

    print(f"Collecting {min(len(src_list), max_samples)} samples...")

    for i, src in enumerate(src_list[:max_samples]):

        if src.dim() == 1:
            src = src.unsqueeze(0)
        src = src.to(device)

        B = src.shape[0]
        tgt_len = inner.max_seq_len
        mask_id = inner.mask_token_id

        # KV Cache (IMPORTANT)
        memory, src_pad_mask = inner.encode_source(src)

        x0_est = torch.full((B, tgt_len), mask_id, dtype=torch.long, device=device)
        hint = None
        captured_hidden = None

        for t_val in range(T - 1, -1, -1):

            t = torch.full((B,), t_val, dtype=torch.long, device=device)
            is_last = (t_val == 0)

            logits, _ = inner.forward_cached(
                memory,
                src_pad_mask,
                x0_est,
                t,
                x0_hint=hint,
                inference_mode=True,
            )

            # Capture hidden state
            if t_val == t_capture:
                if hasattr(inner, "_last_hidden"):
                    captured_hidden = inner._last_hidden.detach().cpu()

            # Sampling
            logits = logits / max(temperature, 1e-8)

            if top_k > 0:
                vals, _ = torch.topk(logits, top_k, dim=-1)
                logits = logits.masked_fill(logits < vals[..., -1:], float("-inf"))

            probs = F.softmax(logits, dim=-1)
            x0_est = torch.argmax(probs, dim=-1) if is_last else _sample(probs)
            hint = x0_est

        # Pool hidden
        if captured_hidden is not None:
            h = captured_hidden[0].mean(dim=0)  # [d_model]
            hidden_list.append(h.numpy())

        # Decode
        ids = [x for x in x0_est[0].tolist() if x > 4]
        text = tgt_tokenizer.decode(ids).strip()

        texts.append(text)
        lengths.append(len(text))

        if i % 100 == 0:
            print(f"{i} done")

    hidden_matrix = np.stack(hidden_list)

    print("Collected hidden states:", hidden_matrix.shape)
    return hidden_matrix, texts, lengths


# ─────────────────────────────────────────────────────────────
# 2. PCA
# ─────────────────────────────────────────────────────────────

def fit_pca(hidden_matrix: np.ndarray, n_components: int = 50):
    from sklearn.decomposition import PCA

    n_comp = min(n_components, hidden_matrix.shape[0] - 1, hidden_matrix.shape[1])
    pca = PCA(n_components=n_comp)
    pca.fit(hidden_matrix)

    print("Explained variance:", pca.explained_variance_ratio_.sum())
    return pca


# ─────────────────────────────────────────────────────────────
# 3. Find Diversity Direction
# ─────────────────────────────────────────────────────────────

def find_diversity_direction(hidden_matrix, lengths, pca):
    from scipy.stats import spearmanr

    projected = pca.transform(hidden_matrix)
    lengths = np.array(lengths)

    scores = []

    for i in range(projected.shape[1]):
        r, _ = spearmanr(projected[:, i], lengths)
        scores.append(abs(r))

    best_pc = int(np.argmax(scores))

    print(f"Best PC: {best_pc} | corr={scores[best_pc]:.3f}")

    direction = pca.components_[best_pc]
    direction = direction / (np.linalg.norm(direction) + 1e-8)

    return direction


# ─────────────────────────────────────────────────────────────
# 4. Steered Generation
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_steered(
    model,
    src,
    direction,
    alpha=0.0,
    temperature=0.8,
    top_k=40,
):
    inner = model.model
    device = next(inner.parameters()).device
    T = inner.scheduler.num_timesteps

    if src.dim() == 1:
        src = src.unsqueeze(0)
    src = src.to(device)

    B = src.shape[0]
    tgt_len = inner.max_seq_len
    mask_id = inner.mask_token_id

    direction = torch.tensor(direction, dtype=torch.float32, device=device)
    direction = direction / (torch.norm(direction) + 1e-6)

    memory, src_pad_mask = inner.encode_source(src)

    x0_est = torch.full((B, tgt_len), mask_id, dtype=torch.long, device=device)
    hint = None

    for t_val in range(T - 1, -1, -1):

        t = torch.full((B,), t_val, dtype=torch.long, device=device)
        is_last = (t_val == 0)

        logits, _ = inner.forward_cached(
            memory,
            src_pad_mask,
            x0_est,
            t,
            x0_hint=hint,
            inference_mode=True,
        )

        # Inject diversity
        if hasattr(inner, "_last_hidden") and alpha != 0.0:
            h = inner._last_hidden
            h = h + alpha * direction.unsqueeze(0).unsqueeze(0)
            logits = inner.head(h)

        # Sampling
        logits = logits / max(temperature, 1e-8)

        if top_k > 0:
            vals, _ = torch.topk(logits, top_k, dim=-1)
            logits = logits.masked_fill(logits < vals[..., -1:], float("-inf"))

        probs = F.softmax(logits, dim=-1)
        x0_est = torch.argmax(probs, dim=-1) if is_last else _sample(probs)
        hint = x0_est

    return x0_est


# ─────────────────────────────────────────────────────────────
# 5. Diversity Spectrum
# ─────────────────────────────────────────────────────────────

def generate_diversity_spectrum(
    model,
    src,
    direction,
    tgt_tokenizer,
    alphas=[-2, -1, 0, 1, 2],
):
    results = {}

    print("\nDiversity Spectrum:\n")

    for alpha in alphas:
        out_ids = generate_steered(model, src, direction, alpha)

        ids = [x for x in out_ids[0].tolist() if x > 4]
        text = tgt_tokenizer.decode(ids).strip()

        print(f"{alpha:+} → {text}")
        results[alpha] = text

    return results


# ─────────────────────────────────────────────────────────────
# 6. Visualization
# ─────────────────────────────────────────────────────────────

def plot_pca_space(hidden_matrix, lengths, pca):
    import matplotlib.pyplot as plt

    proj = pca.transform(hidden_matrix)

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(proj[:, 0], proj[:, 1], c=lengths)
    plt.colorbar(sc)
    plt.title("Concept Space")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()
