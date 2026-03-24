# """
# analysis/attention_viz.py
# ==========================
# Task 2: Attention weight capture and visualization across diffusion steps.
#
# How it works (no retraining needed):
#   MultiHeadAttention now has two attributes:
#     - capture_weights: bool  — set True to start storing weights
#     - last_attn_weights: Tensor — [B, n_heads, Lq, Lk], updated each forward call
#
#   AttentionCapture:
#     - Sets capture_weights=True on all cross-attention layers
#     - Hooks into generate_cached() to record weights at every diffusion step
#     - Returns a dict: {t_val: [layer_0_weights, layer_1_weights, ...]}
#
#   Visualization:
#     - plot_attn_heatmap(): shows src→tgt alignment at a single step
#     - plot_attn_evolution(): shows how one src→tgt pair evolves over T steps
#     - plot_all_layers(): grid of heatmaps per layer at a given step
#
# Usage:
#     from analysis.attention_viz import AttentionCapture, plot_attn_heatmap
#
#     capturer = AttentionCapture(model)
#     weights  = capturer.capture(src_ids, src_tokens, tgt_tokens)
#     plot_attn_heatmap(weights, step=0, layer=0, src_tokens=..., tgt_tokens=...)
# """
#
# import torch
# import numpy as np
# import os
# from typing import List, Dict, Optional
#
#
# # ── Attention capture ─────────────────────────────────────────────────
#
# class AttentionCapture:
#     """
#     Captures cross-attention weights from all decoder layers at every
#     diffusion step during generate_cached().
#
#     Works by:
#       1. Setting capture_weights=True on each DecoderBlock.cross_attn
#       2. Running generate_cached() (encoder runs once via KV cache)
#       3. After each denoising step, reading last_attn_weights from each layer
#       4. Storing as {t_val: list_of_layer_weights}
#
#     Zero retraining required — uses the flag added to MultiHeadAttention.
#     """
#
#     def __init__(self, model):
#         """
#         Args:
#             model : SanskritModel wrapper (must be D3PMCrossAttention)
#         """
#         self.model       = model
#         self.inner       = model.model   # D3PMCrossAttention
#         self._cross_attns = []
#
#         # Collect all cross-attention modules from decoder blocks
#         if hasattr(self.inner, 'decoder_blocks'):
#             for block in self.inner.decoder_blocks:
#                 if hasattr(block, 'cross_attn'):
#                     self._cross_attns.append(block.cross_attn)
#
#         if not self._cross_attns:
#             raise ValueError(
#                 "No cross-attention layers found. "
#                 "AttentionCapture only works with D3PMCrossAttention."
#             )
#
#         print(f"AttentionCapture: found {len(self._cross_attns)} cross-attention layers.")
#
#     def _enable(self):
#         """Turn on weight capture for all cross-attention layers."""
#         for ca in self._cross_attns:
#             ca.capture_weights = True
#
#     def _disable(self):
#         """Turn off weight capture (restores zero overhead)."""
#         for ca in self._cross_attns:
#             ca.capture_weights = False
#             ca.last_attn_weights = None
#
#     def _read_weights(self) -> List[np.ndarray]:
#         """
#         Read current last_attn_weights from all layers.
#         Returns list of [B, n_heads, Lq, Lk] arrays — one per layer.
#         Averages over heads to produce [B, Lq, Lk].
#         """
#         weights = []
#         for ca in self._cross_attns:
#             if ca.last_attn_weights is not None:
#                 # Average over attention heads → [B, Lq, Lk]
#                 w = ca.last_attn_weights.float().mean(dim=1)
#                 weights.append(w.numpy())
#         return weights
#
#     @torch.no_grad()
#     def capture(
#         self,
#         src:        torch.Tensor,
#         capture_every: int = 10,
#     ) -> Dict[int, List[np.ndarray]]:
#         """
#         Run full generation while capturing attention at every `capture_every` steps.
#
#         Args:
#             src           : [1, src_len] or [B, src_len] IAST token ids
#             capture_every : capture weights every N steps (default 10)
#                             Use 1 to capture every step (slow, high memory).
#
#         Returns:
#             step_weights : dict mapping t_val → list of [B, Lq, Lk] arrays
#                            one array per decoder layer
#                            keys are t values: T-1, T-1-N, ..., 0
#
#         Example:
#             weights = capturer.capture(src_ids, capture_every=10)
#             # weights[127] = layer weights at t=127 (heavy noise)
#             # weights[0]   = layer weights at t=0   (clean output)
#         """
#         if src.dim() == 1:
#             src = src.unsqueeze(0)
#
#         inner  = self.inner
#         T      = inner.scheduler.num_timesteps
#         device = src.device
#
#         # KV cache: encode source once
#         memory, src_pad_mask = inner.encode_source(src)
#
#         B       = src.shape[0]
#         tgt_len = inner.max_seq_len
#         mask_id = inner.mask_token_id
#
#         x0_est = torch.full((B, tgt_len), mask_id, dtype=torch.long, device=device)
#         hint   = None
#
#         step_weights: Dict[int, List[np.ndarray]] = {}
#
#         self._enable()
#         try:
#             inner.eval()
#             for t_val in range(T - 1, -1, -1):
#                 t       = torch.full((B,), t_val, dtype=torch.long, device=device)
#                 is_last = (t_val == 0)
#
#                 logits, _ = inner.forward_cached(
#                     memory, src_pad_mask, x0_est, t,
#                     x0_hint=hint, inference_mode=True,
#                 )
#
#                 # Capture at this step if scheduled or it's the last step
#                 if (T - 1 - t_val) % capture_every == 0 or is_last:
#                     step_weights[t_val] = self._read_weights()
#
#                 import torch.nn.functional as F
#                 probs  = F.softmax(logits / 0.8, dim=-1)
#                 x0_est = torch.argmax(probs, dim=-1) if is_last else \
#                          _multinomial_sample(probs)
#                 hint   = x0_est
#
#         finally:
#             self._disable()   # always restore — even if exception raised
#
#         print(f"Captured attention at {len(step_weights)} steps "
#               f"({len(self._cross_attns)} layers each).")
#         return step_weights
#
#
# def _multinomial_sample(probs: torch.Tensor) -> torch.Tensor:
#     B, L, V = probs.shape
#     flat    = probs.view(B * L, V).clamp(min=1e-9)
#     flat    = flat / flat.sum(dim=-1, keepdim=True)
#     return torch.multinomial(flat, 1).squeeze(-1).view(B, L)
#
#
# # ── Visualization ─────────────────────────────────────────────────────
#
# def plot_attn_heatmap(
#     step_weights:  Dict[int, List[np.ndarray]],
#     t_val:         int,
#     layer:         int,
#     src_tokens:    List[str],
#     tgt_tokens:    List[str],
#     sample_idx:    int  = 0,
#     save_path:     Optional[str] = None,
#     title:         Optional[str] = None,
# ):
#     """
#     Plot cross-attention heatmap for a single step and layer.
#
#     X-axis = source (IAST) tokens
#     Y-axis = target (Devanagari) positions
#     Color  = attention weight (brighter = stronger attention)
#
#     Args:
#         step_weights : output of AttentionCapture.capture()
#         t_val        : which diffusion step to visualize
#         layer        : which decoder layer (0 = first, -1 = last)
#         src_tokens   : list of IAST token strings for x-axis labels
#         tgt_tokens   : list of Devanagari token strings for y-axis labels
#         sample_idx   : which batch item to visualize (default 0)
#         save_path    : if given, save figure to this path
#         title        : custom plot title
#     """
#     try:
#         import matplotlib.pyplot as plt
#         import matplotlib.ticker as ticker
#     except ImportError:
#         print("pip install matplotlib to use visualization functions.")
#         return
#
#     if t_val not in step_weights:
#         available = sorted(step_weights.keys())
#         raise ValueError(
#             f"t_val={t_val} not in captured steps. "
#             f"Available: {available[:5]}...{available[-5:]}"
#         )
#
#     layers  = step_weights[t_val]
#     weights = layers[layer][sample_idx]   # [Lq, Lk]
#
#     # Trim to actual token lengths
#     n_src = min(len(src_tokens), weights.shape[1])
#     n_tgt = min(len(tgt_tokens), weights.shape[0])
#     weights = weights[:n_tgt, :n_src]
#
#     fig, ax = plt.subplots(figsize=(max(8, n_src * 0.4), max(6, n_tgt * 0.35)))
#     im = ax.imshow(weights, aspect='auto', cmap='YlOrRd', interpolation='nearest')
#
#     ax.set_xticks(range(n_src))
#     ax.set_xticklabels(src_tokens[:n_src], rotation=45, ha='right', fontsize=9)
#     ax.set_yticks(range(n_tgt))
#     ax.set_yticklabels(tgt_tokens[:n_tgt], fontsize=9)
#
#     ax.set_xlabel("Source (IAST)", fontsize=11)
#     ax.set_ylabel("Target position (Devanagari)", fontsize=11)
#
#     plot_title = title or f"Cross-Attention  |  t={t_val}  |  Layer {layer}"
#     ax.set_title(plot_title, fontsize=12, pad=10)
#
#     plt.colorbar(im, ax=ax, label="Attention weight")
#     plt.tight_layout()
#
#     if save_path:
#         os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
#         plt.savefig(save_path, dpi=150, bbox_inches='tight')
#         print(f"Saved: {save_path}")
#     else:
#         plt.show()
#     plt.close()
#
#
# def plot_attn_evolution(
#     step_weights:  Dict[int, List[np.ndarray]],
#     src_token_idx: int,
#     tgt_token_idx: int,
#     layer:         int = -1,
#     sample_idx:    int = 0,
#     src_token_str: str = "",
#     tgt_token_str: str = "",
#     save_path:     Optional[str] = None,
# ):
#     """
#     Plot how attention between one specific src↔tgt token pair evolves
#     across all captured diffusion steps (T → 0).
#
#     Reveals whether a token pair is 'locked' (stable from early steps)
#     or 'flexible' (weight fluctuates until final steps).
#
#     Args:
#         step_weights  : output of AttentionCapture.capture()
#         src_token_idx : index of source token to track
#         tgt_token_idx : index of target position to track
#         layer         : decoder layer index
#         sample_idx    : batch item
#         src_token_str : string label for the source token (for plot title)
#         tgt_token_str : string label for the target token (for plot title)
#         save_path     : if given, save figure to this path
#     """
#     try:
#         import matplotlib.pyplot as plt
#     except ImportError:
#         print("pip install matplotlib to use visualization functions.")
#         return
#
#     t_vals  = sorted(step_weights.keys(), reverse=True)  # T-1 → 0
#     weights = []
#
#     for t_val in t_vals:
#         layers = step_weights[t_val]
#         w      = layers[layer][sample_idx]   # [Lq, Lk]
#         if tgt_token_idx < w.shape[0] and src_token_idx < w.shape[1]:
#             weights.append(w[tgt_token_idx, src_token_idx])
#         else:
#             weights.append(0.0)
#
#     fig, ax = plt.subplots(figsize=(12, 4))
#     ax.plot(range(len(t_vals)), weights, linewidth=1.5, color='steelblue')
#     ax.fill_between(range(len(t_vals)), weights, alpha=0.2, color='steelblue')
#
#     # Mark every 10th step on x-axis
#     step_labels = [str(t) if i % max(1, len(t_vals)//10) == 0 else ""
#                    for i, t in enumerate(t_vals)]
#     ax.set_xticks(range(len(t_vals)))
#     ax.set_xticklabels(step_labels, fontsize=8)
#     ax.set_xlabel("Diffusion step (T → 0)", fontsize=11)
#     ax.set_ylabel("Attention weight", fontsize=11)
#
#     pair_str = f"src[{src_token_idx}]={src_token_str!r} → tgt[{tgt_token_idx}]={tgt_token_str!r}"
#     ax.set_title(f"Attention evolution  |  {pair_str}  |  Layer {layer}", fontsize=11)
#     ax.set_xlim(0, len(t_vals) - 1)
#     ax.set_ylim(0, None)
#     plt.tight_layout()
#
#     if save_path:
#         os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
#         plt.savefig(save_path, dpi=150, bbox_inches='tight')
#         print(f"Saved: {save_path}")
#     else:
#         plt.show()
#     plt.close()
#
#
# def plot_all_layers(
#     step_weights: Dict[int, List[np.ndarray]],
#     t_val:        int,
#     src_tokens:   List[str],
#     tgt_tokens:   List[str],
#     sample_idx:   int          = 0,
#     save_path:    Optional[str] = None,
# ):
#     """
#     Plot attention heatmaps for ALL decoder layers at a single diffusion step.
#     Shows how different layers specialize their attention patterns.
#     """
#     try:
#         import matplotlib.pyplot as plt
#     except ImportError:
#         print("pip install matplotlib to use visualization functions.")
#         return
#
#     layers  = step_weights[t_val]
#     n_layers = len(layers)
#     n_cols   = min(4, n_layers)
#     n_rows   = (n_layers + n_cols - 1) // n_cols
#
#     fig, axes = plt.subplots(n_rows, n_cols,
#                               figsize=(n_cols * 5, n_rows * 4))
#     axes = np.array(axes).flatten() if n_layers > 1 else [axes]
#
#     n_src = min(len(src_tokens), layers[0][sample_idx].shape[1])
#     n_tgt = min(len(tgt_tokens), layers[0][sample_idx].shape[0])
#
#     for i, (ax, layer_w) in enumerate(zip(axes, layers)):
#         w  = layer_w[sample_idx][:n_tgt, :n_src]
#         im = ax.imshow(w, aspect='auto', cmap='YlOrRd', interpolation='nearest',
#                        vmin=0, vmax=w.max())
#         ax.set_title(f"Layer {i}", fontsize=10)
#         ax.set_xticks(range(n_src))
#         ax.set_xticklabels(src_tokens[:n_src], rotation=45, ha='right', fontsize=7)
#         ax.set_yticks(range(n_tgt))
#         ax.set_yticklabels(tgt_tokens[:n_tgt], fontsize=7)
#
#     for ax in axes[n_layers:]:
#         ax.set_visible(False)
#
#     fig.suptitle(f"All layers at t={t_val}", fontsize=13, y=1.02)
#     plt.tight_layout()
#
#     if save_path:
#         os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
#         plt.savefig(save_path, dpi=150, bbox_inches='tight')
#         print(f"Saved: {save_path}")
#     else:
#         plt.show()
#     plt.close()
"""
analysis/task2_full.py
=====================

FULL Task 2 implementation:
✔ Attention trajectory (already yours)
✔ BERTScore over diffusion steps
✔ Semantic drift metric
✔ Locked vs flexible token detection
✔ TF-IDF vs attention stability correlation
"""

import torch
import numpy as np
from typing import Dict, List
from collections import defaultdict
import re

# Optional metrics
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import evaluate
    bertscore = evaluate.load("bertscore")
    USE_BERT = True
except:
    USE_BERT = False


# ─────────────────────────────────────────────────────────────
# 1. ATTENTION CAPTURE (FIXED VERSION)
# ─────────────────────────────────────────────────────────────

class AttentionCapture:
    def __init__(self, model):
        self.model = model
        self.inner = model.model
        self.cross_attns = []

        for block in self.inner.decoder_blocks:
            if hasattr(block, "cross_attn"):
                self.cross_attns.append(block.cross_attn)

    def _enable(self):
        for ca in self.cross_attns:
            ca.capture_weights = True

    def _disable(self):
        for ca in self.cross_attns:
            ca.capture_weights = False
            ca.last_attn_weights = None

    def _read(self):
        weights = []
        for ca in self.cross_attns:
            if ca.last_attn_weights is not None:
                w = ca.last_attn_weights.mean(dim=1)  # avg heads
                weights.append(w.cpu().numpy())
        return weights

    @torch.no_grad()
    def run(self, src_ids):
        inner = self.inner
        T = inner.scheduler.num_timesteps
        device = src_ids.device

        memory, mask = inner.encode_source(src_ids)

        x = torch.full(
            (1, inner.max_seq_len),
            inner.mask_token_id,
            dtype=torch.long,
            device=device
        )

        hint = None
        step_weights = {}
        step_outputs = {}

        self._enable()

        try:
            for t_val in range(T - 1, -1, -1):
                t = torch.tensor([t_val], device=device)

                logits, _ = inner.forward_cached(
                    memory, mask, x, t, x0_hint=hint, inference_mode=True
                )

                probs = torch.softmax(logits, dim=-1)
                x = torch.argmax(probs, dim=-1)

                step_weights[t_val] = self._read()
                step_outputs[t_val] = x.clone()

                hint = x

        finally:
            self._disable()

        return step_weights, step_outputs


# ─────────────────────────────────────────────────────────────
# 2. BERTScore + Semantic Drift
# ─────────────────────────────────────────────────────────────

def compute_trajectory_metrics(
    step_outputs,
    tgt_tokenizer,
    reference_text
):
    trajectory = []

    for t, ids in step_outputs.items():
        text = tgt_tokenizer.decode(
            [x for x in ids[0].tolist() if x > 4]
        )

        if USE_BERT:
            score = bertscore.compute(
                predictions=[text],
                references=[reference_text],
                lang="hi"
            )["f1"][0]
        else:
            score = 0.0

        drift = 1.0 - score

        trajectory.append({
            "step": t,
            "text": text,
            "bert": score,
            "drift": drift
        })

    return sorted(trajectory, key=lambda x: -x["step"])


# ─────────────────────────────────────────────────────────────
# 3. LOCKED vs FLEXIBLE TOKENS
# ─────────────────────────────────────────────────────────────

def analyze_token_stability(step_weights):
    """
    Measure variance of attention over time
    """
    token_stability = defaultdict(list)

    for t, layers in step_weights.items():
        last_layer = layers[-1][0]  # [Lq, Lk]

        # max attention source index per target token
        align = np.argmax(last_layer, axis=1)

        for tgt_idx, src_idx in enumerate(align):
            token_stability[tgt_idx].append(src_idx)

    results = {}

    for tgt_idx, src_seq in token_stability.items():
        changes = sum(
            1 for i in range(1, len(src_seq))
            if src_seq[i] != src_seq[i-1]
        )

        if changes <= 2:
            results[tgt_idx] = "LOCKED"
        else:
            results[tgt_idx] = "FLEXIBLE"

    return results


# ─────────────────────────────────────────────────────────────
# 4. TF-IDF vs ATTENTION STABILITY
# ─────────────────────────────────────────────────────────────

_IAST_STOPWORDS = {
    "ca", "eva", "api", "tu", "va", "na", "hi", "atha", "iti",
    "vai", "ha", "khalu", "sma", "yadi", "tatah", "atra", "iva",
    "itiha", "idam", "tat", "etat", "sa", "sah", "te", "tam"
}


def _normalize_iast_token(token: str) -> str:
    t = token.lower().strip()
    t = t.replace("’", "'").replace("`", "'")
    return "".join(ch for ch in t if (ch.isalpha() or ch in ("'", "-")))


def _tokenize_iast(text: str) -> List[str]:
    raw = re.split(r"\s+", (text or "").strip())
    toks = []
    for tok in raw:
        t = _normalize_iast_token(tok)
        if len(t) >= 2 and t not in _IAST_STOPWORDS:
            toks.append(t)
    return toks


def tfidf_attention_correlation(src_text, step_weights, corpus_texts=None):
    """
    Compute TF-IDF/attention correlation using corpus-level TF-IDF.
    Returns a dict so callers can plot and handle undefined cases safely.
    """
    src_tokens = _tokenize_iast(src_text)
    if not src_tokens:
        return {
            "corr": None,
            "status": "NA_NO_SOURCE_TOKENS",
            "tokens": [],
            "tfidf_scores": np.array([], dtype=np.float32),
            "attn_scores": np.array([], dtype=np.float32),
        }

    docs = []
    src_doc = " ".join(src_tokens)
    docs.append(src_doc)
    if corpus_texts:
        for txt in corpus_texts:
            toks = _tokenize_iast(txt)
            if toks:
                docs.append(" ".join(toks))

    # Deduplicate while preserving order to avoid overweighting repeated docs.
    docs = list(dict.fromkeys(docs))
    if len(docs) < 2:
        docs = [src_doc, src_doc + " semantic context"]

    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        norm="l2",
    )
    mat = vectorizer.fit_transform(docs)
    terms = vectorizer.get_feature_names_out()
    src_vec = mat[0].toarray()[0]
    term_to_tfidf = {terms[i]: float(src_vec[i]) for i in range(len(terms))}
    tfidf_by_pos = np.array([term_to_tfidf.get(tok, 0.0) for tok in src_tokens], dtype=np.float32)

    # Average attention over captured steps, last layer.
    attn_scores = None
    for _, layers in step_weights.items():
        w = layers[-1][0]
        avg = w.mean(axis=0).astype(np.float32)
        if attn_scores is None:
            attn_scores = avg
        else:
            attn_scores += avg
    if attn_scores is None:
        attn_scores = np.zeros((0,), dtype=np.float32)
    else:
        attn_scores /= max(len(step_weights), 1)

    min_len = min(len(tfidf_by_pos), len(attn_scores))
    if min_len < 2:
        return {
            "corr": None,
            "status": "NA_INSUFFICIENT_LENGTH",
            "tokens": src_tokens[:min_len],
            "tfidf_scores": tfidf_by_pos[:min_len],
            "attn_scores": attn_scores[:min_len],
        }

    x = tfidf_by_pos[:min_len]
    y = attn_scores[:min_len]
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return {
            "corr": None,
            "status": "NA_ZERO_VARIANCE",
            "tokens": src_tokens[:min_len],
            "tfidf_scores": x,
            "attn_scores": y,
        }

    corr = float(np.corrcoef(x, y)[0, 1])
    if np.isnan(corr):
        return {
            "corr": None,
            "status": "NA_NUMERIC",
            "tokens": src_tokens[:min_len],
            "tfidf_scores": x,
            "attn_scores": y,
        }

    return {
        "corr": corr,
        "status": "OK",
        "tokens": src_tokens[:min_len],
        "tfidf_scores": x,
        "attn_scores": y,
    }


# ─────────────────────────────────────────────────────────────
# 5. FULL PIPELINE
# ─────────────────────────────────────────────────────────────

def run_task2_analysis(
    text,
    model,
    src_tokenizer,
    tgt_tokenizer,
    device
):
    src_ids = torch.tensor(
        [src_tokenizer.encode(text)],
        device=device
    )

    capturer = AttentionCapture(model)

    # Step 1: Capture
    step_weights, step_outputs = capturer.run(src_ids)

    # Step 2: Metrics
    trajectory = compute_trajectory_metrics(
        step_outputs,
        tgt_tokenizer,
        reference_text=text   # transliteration task
    )

    # Step 3: Token stability
    stability = analyze_token_stability(step_weights)

    # Step 4: TF-IDF correlation
    corr_obj = tfidf_attention_correlation(text, step_weights)

    return {
        "trajectory": trajectory,
        "token_stability": stability,
        "tfidf_corr": corr_obj.get("corr"),
        "tfidf_status": corr_obj.get("status"),
    }
