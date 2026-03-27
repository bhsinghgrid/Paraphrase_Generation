# ============================================================
# TASK 2: Source–Paraphrase Semantic Alignment Trajectory
# ============================================================

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from collections import defaultdict

# Optional (install if needed)
# pip install bert-score scikit-learn
from bert_score import score as bertscore
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# ------------------ ATTENTION HOOK --------------------------
# ============================================================

def register_attention_hooks(model):
    """
    Registers forward hooks to capture cross-attention weights
    from each decoder block.

    Assumes each block has attribute `.cross_attn.attn_weights`
    """
    inner = model.model
    attention_maps = []

    def hook_fn(module, input, output):
        if hasattr(module, "attn_weights"):
            attention_maps.append(module.attn_weights.detach().cpu())

    hooks = []
    for block in inner.decoder_blocks:
        if hasattr(block, "cross_attn"):
            h = block.cross_attn.register_forward_hook(hook_fn)
            hooks.append(h)

    return hooks, attention_maps


# ============================================================
# ------------------ CAPTURE TRAJECTORY ----------------------
# ============================================================

@torch.no_grad()
def capture_alignment_trajectory(
    model,
    src_tensor: torch.Tensor,
    src_text: str,
    tgt_tokenizer,
    steps_to_capture: List[int] = None,
):
    """
    Capture:
      - intermediate outputs
      - cross-attention maps
      - BERTScore vs source

    Returns:
      dict with outputs, attention, drift
    """

    inner = model.model
    device = src_tensor.device
    T = inner.scheduler.num_timesteps

    if steps_to_capture is None:
        steps_to_capture = list(range(T - 1, -1, -5)) + [0]

    # Register hooks
    hooks, attn_storage = register_attention_hooks(model)

    memory, src_pad_mask = inner.encode_source(src_tensor)

    B = src_tensor.shape[0]
    tgt_len = inner.max_seq_len
    mask_id = inner.mask_token_id

    x0_est = torch.full((B, tgt_len), mask_id, device=device)
    hint = None

    outputs = {}
    attention_per_step = {}

    for t_val in range(T - 1, -1, -1):
        t = torch.full((B,), t_val, device=device)

        logits, _ = inner.forward_cached(
            memory, src_pad_mask, x0_est, t,
            x0_hint=hint, inference_mode=True
        )

        probs = F.softmax(logits, dim=-1)
        x0_est = torch.argmax(probs, dim=-1)
        hint = x0_est

        if t_val in steps_to_capture:
            ids = [x for x in x0_est[0].tolist() if x > 4]
            text = tgt_tokenizer.decode(ids)

            outputs[t_val] = text

            # Collect attention maps (last layer only for simplicity)
            if len(attn_storage) > 0:
                attention_per_step[t_val] = attn_storage[-1].numpy()

    # Remove hooks
    for h in hooks:
        h.remove()

    # Compute BERTScore trajectory
    bert_scores = compute_bert_alignment(src_text, outputs)

    return {
        "outputs": outputs,
        "attention": attention_per_step,
        "bert_scores": bert_scores,
    }


# ============================================================
# ------------------ BERTScore -------------------------------
# ============================================================

def compute_bert_alignment(src_text: str, outputs: Dict[int, str]):
    """
    Compute BERTScore between source and each intermediate output
    """
    scores = {}

    for t, text in outputs.items():
        P, R, F1 = bertscore([text], [src_text], lang="hi", verbose=False)
        scores[t] = float(F1.mean())

    return scores


# ============================================================
# ------------------ SEMANTIC DRIFT --------------------------
# ============================================================

def compute_semantic_drift(bert_scores: Dict[int, float]):
    """
    Drift = drop from best alignment
    """
    max_score = max(bert_scores.values())
    drift = {t: max_score - s for t, s in bert_scores.items()}
    return drift


# ============================================================
# ------------------ ATTENTION STABILITY ---------------------
# ============================================================

def compute_attention_stability(attention_maps: Dict[int, np.ndarray]):
    """
    Measures if tokens attend consistently across steps.
    """
    steps = sorted(attention_maps.keys(), reverse=True)

    stability_scores = []

    for i in range(len(steps) - 1):
        A = attention_maps[steps[i]]
        B = attention_maps[steps[i+1]]

        diff = np.abs(A - B).mean()
        stability_scores.append(diff)

    return np.mean(stability_scores)


# ============================================================
# ------------------ TF-IDF vs STABILITY ---------------------
# ============================================================

def compute_tfidf_attention_correlation(
    src_texts: List[str],
    attention_maps_list: List[Dict[int, np.ndarray]]
):
    """
    Correlate TF-IDF importance with attention stability
    """

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(src_texts).toarray()

    word_importance = tfidf.mean(axis=0)

    stability = []
    for attn_maps in attention_maps_list:
        stability.append(compute_attention_stability(attn_maps))

    corr = np.corrcoef(word_importance[:len(stability)], stability)[0, 1]
    return corr


# ============================================================
# ------------------ HEATMAP VISUALIZATION -------------------
# ============================================================

def plot_attention_heatmap(attn: np.ndarray, title="Attention"):
    """
    Plot cross-attention heatmap
    attn: [tgt_len, src_len]
    """
    plt.figure(figsize=(6,5))
    plt.imshow(attn, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Source tokens")
    plt.ylabel("Target tokens")
    plt.show()


def visualize_trajectory(attention_maps: Dict[int, np.ndarray]):
    """
    Show attention evolution over time
    """
    steps = sorted(attention_maps.keys(), reverse=True)

    for t in steps[:5]:  # show 5 steps
        plot_attention_heatmap(attention_maps[t], title=f"Step t={t}")


# ============================================================
# ------------------ LOCKED vs FLEXIBLE ----------------------
# ============================================================

def analyze_token_behavior(attention_maps: Dict[int, np.ndarray]):
    """
    Detect whether tokens are locked or flexible
    """
    steps = sorted(attention_maps.keys(), reverse=True)

    first = attention_maps[steps[0]]
    last = attention_maps[steps[-1]]

    diff = np.abs(first - last).mean(axis=1)

    locked = np.where(diff < 0.05)[0]
    flexible = np.where(diff >= 0.05)[0]

    return {
        "locked_tokens": locked.tolist(),
        "flexible_tokens": flexible.tolist()
    }


# ============================================================
# ------------------ MASTER FUNCTION -------------------------
# ============================================================

def run_task2_analysis(
    model,
    src_tensor,
    src_text,
    tgt_tokenizer
):
    result = capture_alignment_trajectory(
        model, src_tensor, src_text, tgt_tokenizer
    )

    drift = compute_semantic_drift(result["bert_scores"])
    stability = compute_attention_stability(result["attention"])
    behavior = analyze_token_behavior(result["attention"])

    print("\nBERTScore trajectory:")
    print(result["bert_scores"])

    print("\nSemantic drift:")
    print(drift)

    print(f"\nAttention stability: {stability:.4f}")

    print("\nToken behavior:")
    print(behavior)

    visualize_trajectory(result["attention"])

    return {
        "trajectory": result,
        "drift": drift,
        "stability": stability,
        "behavior": behavior
    }