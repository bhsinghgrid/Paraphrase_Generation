# """
# analysis/semantic_drift.py
# ===========================
# Task 2: Semantic drift metric — how much does the intermediate generation
# diverge from the final output as we walk through diffusion steps T → 0?
#
# Metric: CER between x0_estimate at each step vs the final x0 at t=0.
#
# A well-trained model should show:
#   - High drift at t=T-1 (near-random initial estimate)
#   - Rapid decrease in drift around t=T//2 (model finds the right structure)
#   - Near-zero drift at t=10 (output is stable, only fine corrections remain)
#
# If drift stays high until t=5 then suddenly collapses → model is doing all
# its work in the last few steps → consider reducing T.
#
# Also measures:
#   - Token stability: fraction of positions that don't change between steps
#   - Lock-in time: first step where each position "commits" to its final token
#
# No retraining required. Uses generate_cached() with intermediate snapshots.
# """
#
# import torch
# import torch.nn.functional as F
# import numpy as np
# from typing import List, Dict, Optional, Tuple
#
#
# def compute_cer_between(pred: str, ref: str) -> float:
#     """CER between two strings."""
#     if not ref:
#         return 1.0 if pred else 0.0
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
#     return edit_distance(pred, ref) / len(ref)
#
#
# @torch.no_grad()
# def capture_intermediate_outputs(
#     model,
#     src:          torch.Tensor,
#     tgt_tokenizer,
#     capture_every: int = 5,
#     temperature:   float = 0.8,
#     top_k:         int   = 40,
# ) -> Tuple[Dict[int, str], str]:
#     """
#     Run generation while recording the decoded x0_estimate at every
#     `capture_every` diffusion steps.
#
#     Args:
#         model         : SanskritModel (D3PMCrossAttention)
#         src           : [1, src_len] IAST token ids (single sample)
#         tgt_tokenizer : SanskritTargetTokenizer for decoding intermediate outputs
#         capture_every : record every N steps
#         temperature   : sampling temperature
#         top_k         : top-k filter
#
#     Returns:
#         step_outputs : dict mapping t_val → decoded Devanagari string at that step
#         final_output : decoded string at t=0 (final result)
#     """
#     if src.dim() == 1:
#         src = src.unsqueeze(0)
#
#     inner  = model.model
#     T      = inner.scheduler.num_timesteps
#     device = src.device
#
#     # Encode source once (KV cache)
#     memory, src_pad_mask = inner.encode_source(src)
#
#     B       = src.shape[0]
#     tgt_len = inner.max_seq_len
#     mask_id = inner.mask_token_id
#
#     x0_est = torch.full((B, tgt_len), mask_id, dtype=torch.long, device=device)
#     hint   = None
#
#     step_outputs: Dict[int, str] = {}
#     inner.eval()
#
#     for t_val in range(T - 1, -1, -1):
#         t       = torch.full((B,), t_val, dtype=torch.long, device=device)
#         is_last = (t_val == 0)
#
#         logits, _ = inner.forward_cached(
#             memory, src_pad_mask, x0_est, t,
#             x0_hint=hint, inference_mode=True,
#         )
#
#         logits = logits / max(temperature, 1e-8)
#         if top_k > 0:
#             V = logits.shape[-1]
#             if top_k < V:
#                 topk_vals, _ = torch.topk(logits, top_k, dim=-1)
#                 threshold    = topk_vals[..., -1].unsqueeze(-1)
#                 logits       = logits.masked_fill(logits < threshold, float('-inf'))
#
#         probs  = F.softmax(logits, dim=-1)
#         x0_est = torch.argmax(probs, dim=-1) if is_last else _sample(probs)
#         hint   = x0_est
#
#         # Capture at this step
#         if (T - 1 - t_val) % capture_every == 0 or is_last:
#             ids  = [x for x in x0_est[0].tolist() if x > 4]
#             text = tgt_tokenizer.decode(ids).strip()
#             step_outputs[t_val] = text
#
#     final_output = step_outputs.get(0, "")
#     return step_outputs, final_output
#
#
# def _sample(probs):
#     B, L, V = probs.shape
#     flat    = probs.view(B * L, V).clamp(min=1e-9)
#     flat    = flat / flat.sum(dim=-1, keepdim=True)
#     return torch.multinomial(flat, 1).squeeze(-1).view(B, L)
#
#
# def compute_drift(
#     step_outputs:  Dict[int, str],
#     final_output:  str,
# ) -> Dict[str, object]:
#     """
#     Compute drift metrics comparing each intermediate output to the final.
#
#     Returns dict with:
#       t_vals      : list of captured timesteps (T-1 → 0)
#       cer_to_final: CER between each step's output and the final output
#                     0.0 = identical to final, 1.0 = completely different
#       lock_in_t   : first t_val where CER drops and stays below 0.1
#                     (step at which output "commits" to final form)
#     """
#     t_vals       = sorted(step_outputs.keys(), reverse=True)   # T-1 → 0
#     cer_to_final = []
#
#     for t_val in t_vals:
#         cer = compute_cer_between(step_outputs[t_val], final_output)
#         cer_to_final.append(cer)
#
#     # Find lock-in: first step where CER stays below threshold for rest of run
#     threshold = 0.1
#     lock_in_t = 0   # default: never locked in early
#     for i, (t_val, cer) in enumerate(zip(t_vals, cer_to_final)):
#         if all(c <= threshold for c in cer_to_final[i:]):
#             lock_in_t = t_val
#             break
#
#     return {
#         "t_vals":       t_vals,
#         "cer_to_final": cer_to_final,
#         "lock_in_t":    lock_in_t,
#         "final_output": final_output,
#     }
#
#
# def compute_token_stability(
#     step_outputs:  Dict[int, str],
#     final_output:  str,
#     tgt_tokenizer,
# ) -> Dict[str, object]:
#     """
#     Token-level stability: for each position, at which diffusion step
#     does it first match its final token and stay matched?
#
#     Returns:
#       position_lock_times: list of t_val at which each position locks in
#       mean_lock_t        : average lock-in timestep across positions
#     """
#     T      = max(step_outputs.keys())
#     t_vals = sorted(step_outputs.keys(), reverse=True)   # T-1 → 0
#
#     # Encode all intermediate outputs and the final
#     def encode(text):
#         return tgt_tokenizer.encode(text)
#
#     final_ids = encode(final_output)
#     L         = len(final_ids)
#
#     # Build matrix: [n_steps, L]
#     step_ids = []
#     for t_val in t_vals:
#         step_ids.append(encode(step_outputs.get(t_val, "")))
#
#     # Pad all to same length
#     max_len = max(len(s) for s in step_ids)
#     step_ids = [s + [1] * (max_len - len(s)) for s in step_ids]   # 1=PAD
#     final_ids_padded = final_ids + [1] * (max_len - len(final_ids))
#
#     step_arr  = np.array(step_ids)                # [n_steps, L]
#     final_arr = np.array(final_ids_padded)         # [L]
#
#     # For each position: find first step index where it matches final
#     # and stays matched for all subsequent steps
#     position_lock_steps = []
#     for pos in range(min(L, max_len)):
#         col = step_arr[:, pos]   # [n_steps]
#         fin = final_arr[pos]
#         locked_at = len(t_vals) - 1   # default: never locks early
#         for i in range(len(t_vals)):
#             if all(col[i:] == fin):
#                 locked_at = i
#                 break
#         position_lock_steps.append(t_vals[locked_at] if locked_at < len(t_vals) else 0)
#
#     return {
#         "position_lock_times": position_lock_steps,
#         "mean_lock_t":         float(np.mean(position_lock_steps)),
#         "std_lock_t":          float(np.std(position_lock_steps)),
#     }
#
#
# def plot_drift_curve(
#     drift_result: Dict,
#     src_text:     str = "",
#     save_path:    Optional[str] = None,
# ):
#     """
#     Plot CER-to-final vs diffusion step.
#     Shows where the model "commits" to the final output.
#     """
#     try:
#         import matplotlib.pyplot as plt
#     except ImportError:
#         print("pip install matplotlib.")
#         return
#
#     t_vals  = drift_result["t_vals"]
#     cers    = drift_result["cer_to_final"]
#     lock_t  = drift_result["lock_in_t"]
#
#     fig, ax = plt.subplots(figsize=(12, 4))
#     ax.plot(range(len(t_vals)), cers, linewidth=1.8, color='coral', label='CER to final')
#     ax.fill_between(range(len(t_vals)), cers, alpha=0.15, color='coral')
#
#     # Mark lock-in point
#     if lock_t in t_vals:
#         lock_idx = t_vals.index(lock_t)
#         ax.axvline(lock_idx, color='steelblue', linestyle='--', linewidth=1.2,
#                    label=f"Lock-in at t={lock_t}")
#
#     ax.axhline(0.1, color='gray', linestyle=':', linewidth=1, alpha=0.7)
#
#     n = len(t_vals)
#     tick_positions = list(range(0, n, max(1, n // 10)))
#     ax.set_xticks(tick_positions)
#     ax.set_xticklabels([str(t_vals[i]) for i in tick_positions], fontsize=8)
#     ax.set_xlabel("Diffusion step t  (T-1 → 0)", fontsize=11)
#     ax.set_ylabel("CER vs final output", fontsize=11)
#     ax.set_ylim(0, 1.05)
#     ax.set_xlim(0, n - 1)
#     ax.legend(fontsize=10)
#
#     title = f"Semantic drift"
#     if src_text:
#         title += f"  |  src: {src_text[:50]}"
#     ax.set_title(title, fontsize=11)
#     plt.tight_layout()
#
#     if save_path:
#         import os
#         os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
#         plt.savefig(save_path, dpi=150, bbox_inches='tight')
#         print(f"Saved: {save_path}")
#     else:
#         plt.show()
#     plt.close()
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