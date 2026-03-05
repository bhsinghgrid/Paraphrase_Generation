# import torch
# import torch.nn.functional as F
# import inspect
#
# class ReverseDiffusion:
#     """
#     Reverse diffusion with Beam Search + Temperature + Proper Penalties + Self-Conditioning.
#     """
#
#     def __init__(self, scheduler):
#         self.scheduler = scheduler
#
#     def p_sample_step(
#         self,
#         model,
#         x_t,
#         t,
#         condition,
#         x0_hint=None, # 🔥 Added x0_hint
#         beam_width=3,
#         temperature=1.0,
#         repetition_penalty=1.2,
#         diversity_penalty=0.3
#     ):
#         with torch.no_grad():
#             # ---- Shape safety ----
#             if x_t.dim() == 1: x_t = x_t.unsqueeze(0)
#             if condition.dim() == 1: condition = condition.unsqueeze(0)
#             if t.dim() == 0: t = t.unsqueeze(0)
#             if t.shape[0] != x_t.shape[0]: t = t.expand(x_t.shape[0])
#
#             # ---- Model forward with Self-Conditioning Hint ----
#             # 🔥 Pass the x0_hint to the model
#             # logits, _ = model(condition, x_t, t, x0_hint=x0_hint)
#             sig = inspect.signature(model.forward).parameters
#
#             if 'x0_hint' in sig:
#                 outputs = model(condition, x_t, t, x0_hint=x0_hint)
#             else:
#                 outputs = model(condition, x_t, t)
#
#             # 🔥 Safely extract logits (handles both tuple returns and single tensor returns)
#             logits = outputs[0] if isinstance(outputs, tuple) else outputs
#
#             # ---- Temperature scaling ----
#             logits = logits / temperature
#
#             # ---- Repetition penalty ----
#             if repetition_penalty != 1.0:
#                 logits = apply_repetition_penalty(logits, x_t, repetition_penalty)
#
#             # ---- Diversity penalty ----
#             if diversity_penalty > 0:
#                 logits = apply_diversity_penalty(logits, diversity_penalty)
#
#             probs = F.softmax(logits, dim=-1)
#             B, L, V = probs.shape
#
#             # ---- Top-k beam expansion ----
#             topk_probs, topk_ids = torch.topk(probs, beam_width, dim=-1)
#
#             candidates = []
#             for k in range(beam_width):
#                 next_tokens = topk_ids[:, :, k]
#                 score = torch.log(topk_probs[:, :, k] + 1e-9).sum()
#                 candidates.append((next_tokens, score))
#
#             return candidates
#
#     def generate_beam(
#         self,
#         model,
#         condition,
#         beam_width=3,
#         num_steps=None,
#         temperature=1.0,
#         repetition_penalty=1.2,
#         diversity_penalty=0.5
#     ):
#         if num_steps is None:
#             num_steps = self.scheduler.num_timesteps
#
#         device = condition.device
#         if condition.dim() == 1:
#             condition = condition.unsqueeze(0)
#
#         B, L = condition.shape
#
#         # Initialize: start from MASK
#         x_init = torch.full((B, L), fill_value=model.mask_token_id, dtype=torch.long, device=device)
#
#         # 🔥 Beam state now includes (current_x_t, cumulative_score, last_x0_prediction)
#         # last_x0_prediction starts as None for the first step
#         # Inside the loop, ensure t_tensor is on the same device
#         t_tensor = torch.full((B,), num_steps, dtype=torch.long, device=device)
#         beams = [(x_init, 0.0, None)]
#
#         for step in reversed(range(num_steps)):
#             new_beams = []
#
#             for x_t, score, last_x0 in beams:
#                 t_tensor = torch.full((B,), step, dtype=torch.long, device=device)
#
#                 # 🔥 Pass last_x0 as the hint
#                 candidates = self.p_sample_step(
#                     model,
#                     x_t,
#                     t_tensor,
#                     condition,
#                     x0_hint=last_x0, # Feed previous prediction back
#                     beam_width=beam_width,
#                     temperature=temperature,
#                     repetition_penalty=repetition_penalty,
#                     diversity_penalty=diversity_penalty
#                 )
#
#                 for tokens, new_score in candidates:
#                     # tokens here becomes the new x0_hint for the next step
#                     new_beams.append((tokens, score + new_score, tokens))
#
#             # ---- Keep top beams ----
#             new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)
#             beams = new_beams[:beam_width]
#
#         best_tokens, best_score, _ = beams[0]
#         return best_tokens
#
# # --- Penalty functions (apply_repetition_penalty and apply_diversity_penalty remain same) ---
#
#
# def apply_repetition_penalty(logits, prev_tokens, penalty=1.2):
#     """
#     Penalize tokens that already appeared in the sequence.
#     """
#
#     B, L, V = logits.shape
#
#     for b in range(B):
#         used_tokens = set(prev_tokens[b].tolist())
#
#         for token_id in used_tokens:
#             logits[b, :, token_id] /= penalty
#
#     return logits
#
# def apply_diversity_penalty(logits, penalty=0.3):
#     """
#     Encourages variance in predictions.
#     """
#     logits_var = logits.var(dim=-1, keepdim=True)
#     return logits + penalty * logits_var

import torch
import torch.nn.functional as F
import inspect


class ReverseDiffusion:
    """
    Proper D3PM Absorbing-State Reverse Diffusion.

    KEY FIX: Uses correct posterior sampling p(x_{t-1} | x_t, x_0_pred)
    instead of a broken multi-step beam search that accumulated scores
    across positions and timesteps (causing repetitive, incoherent output).

    The D3PM absorbing posterior for a single token position is:
      - If x_t[i] != [MASK]: x_{t-1}[i] = x_t[i]  (already unmasked, stays)
      - If x_t[i] == [MASK]: unmask with prob (alpha_{t-1} - alpha_t) / (1 - alpha_t)
                              else stay masked
    """

    def __init__(self, scheduler):
        self.scheduler = scheduler

    # ------------------------------------------------------------------
    # 🔥 PRIMARY METHOD: Proper D3PM reverse diffusion
    # ------------------------------------------------------------------
    def generate(
        self,
        model,
        condition,
        num_steps=20,          # Use fewer steps than training (10-30 works well)
        temperature=0.8,       # < 1.0 = sharper, more confident predictions
        top_k=50,              # Top-k filtering for diversity
        use_argmax_final=True  # Greedy on the last step for stability
    ):
        """
        Proper iterative D3PM denoising.

        Args:
            model      : The inner model (D3PMCrossAttention, etc.)
            condition  : [B, L] source token IDs (transliteration input)
            num_steps  : Number of denoising steps (subset of training timesteps)
            temperature: Sampling temperature (0.7-1.0 recommended)
            top_k      : Vocabulary filtering per position
        """
        device = condition.device
        if condition.dim() == 1:
            condition = condition.unsqueeze(0)
        B, L = condition.shape

        scheduler = self.scheduler
        T = scheduler.num_timesteps

        # --- Build a linearly-spaced subset of timesteps (T → 0) ---
        step_size = max(1, T // num_steps)
        timesteps = list(range(T - 1, -1, -step_size))
        if timesteps[-1] != 0:
            timesteps.append(0)

        # --- Start: all tokens are [MASK] ---
        mask_id = model.mask_token_id
        x_t = torch.full((B, L), fill_value=mask_id, dtype=torch.long, device=device)

        prev_x0_hint = None  # for self-conditioning

        for i, t_val in enumerate(timesteps):
            t = torch.full((B,), t_val, dtype=torch.long, device=device)

            # ── Forward pass ──────────────────────────────────────────
            sig = inspect.signature(model.forward).parameters
            with torch.no_grad():
                if 'x0_hint' in sig:
                    outputs = model(condition, x_t, t, x0_hint=prev_x0_hint)
                else:
                    outputs = model(condition, x_t, t)

            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            # logits: [B, L, V]

            # ── Predict x_0 ───────────────────────────────────────────
            logits = logits / max(temperature, 1e-5)

            # Top-k filtering (improves quality, prevents degenerate tokens)
            if top_k > 0:
                logits = _top_k_filter(logits, top_k)

            probs = F.softmax(logits, dim=-1)  # [B, L, V]

            # ── Sample or argmax x_0_pred ─────────────────────────────
            is_last_step = (i == len(timesteps) - 1)
            if is_last_step and use_argmax_final:
                x_0_pred = torch.argmax(probs, dim=-1)  # [B, L]
            else:
                # Multinomial sample per position
                x_0_pred = _batch_multinomial(probs)     # [B, L]

            # Update self-conditioning hint
            prev_x0_hint = x_0_pred

            # ── D3PM Absorbing Posterior: compute x_{t-1} ────────────
            x_t = self._absorbing_posterior_sample(
                x_t, x_0_pred, t_val, timesteps, i, mask_id, device
            )

        return x_t   # [B, L] — final predicted token IDs

    # ------------------------------------------------------------------
    # 🔥 LEGACY ALIAS: keep generate_beam name for backward compat
    # ------------------------------------------------------------------
    def generate_beam(
        self,
        model,
        condition,
        beam_width=3,       # ignored — kept for API compatibility
        num_steps=None,
        temperature=0.8,
        repetition_penalty=1.2,   # ignored (handled via top-k now)
        diversity_penalty=0.5     # ignored
    ):
        """
        Backward-compatible wrapper.  Delegates to the correct generate().
        The old beam-search logic is replaced entirely because it produced
        degenerate (repetitive, incoherent) outputs due to:
          1. Summing log-probs across ALL positions at EVERY step
          2. No proper D3PM posterior transition — just repeated greedy decoding
          3. Diversity/repetition penalties that amplified degenerate loops
        """
        steps = num_steps if num_steps is not None else self.scheduler.num_timesteps
        # Clamp to a sensible inference budget
        steps = min(steps, 30)
        return self.generate(
            model=model,
            condition=condition,
            num_steps=steps,
            temperature=temperature,
            top_k=50,
        )

    # ------------------------------------------------------------------
    # D3PM Absorbing Posterior helper
    # ------------------------------------------------------------------
    def _absorbing_posterior_sample(
        self, x_t, x_0_pred, t_val, timesteps, step_idx, mask_id, device
    ):
        """
        Compute x_{t-1} from x_t and x_0_pred using the absorbing-state posterior.

        For each position i:
          • If x_t[i] != mask: it was already unmasked → keep it unchanged
          • If x_t[i] == mask:
              With prob  (alpha_{t-1} - alpha_t) / (1 - alpha_t)  → unmask to x_0_pred[i]
              Otherwise                                             → stay masked
        """
        B, L = x_t.shape

        if t_val == 0:
            # Final step: replace ALL remaining masks with prediction
            is_masked = (x_t == mask_id)
            x_new = x_t.clone()
            x_new[is_masked] = x_0_pred[is_masked]
            return x_new

        # alpha at current t
        t_tensor = torch.tensor([t_val], device=self.scheduler.alphas_cumprod.device)
        alpha_t = self.scheduler.get_alpha(t_tensor).item()

        # alpha at previous (smaller) t
        if step_idx + 1 < len(timesteps):
            t_prev_val = timesteps[step_idx + 1]
        else:
            t_prev_val = 0
        t_prev_tensor = torch.tensor([t_prev_val], device=self.scheduler.alphas_cumprod.device)
        alpha_t_prev = self.scheduler.get_alpha(t_prev_tensor).item()

        # Probability of unmasking a currently-masked token at this step
        denom = max(1.0 - alpha_t, 1e-8)
        unmask_prob = (alpha_t_prev - alpha_t) / denom
        unmask_prob = float(torch.clamp(torch.tensor(unmask_prob), 0.0, 1.0))

        is_masked = (x_t == mask_id)   # [B, L] bool
        rand = torch.rand(B, L, device=device)
        should_unmask = is_masked & (rand < unmask_prob)

        x_new = x_t.clone()
        x_new[should_unmask] = x_0_pred[should_unmask]
        return x_new

    # ------------------------------------------------------------------
    # Legacy p_sample_step (kept for compatibility, not used in generate)
    # ------------------------------------------------------------------
    def p_sample_step(self, model, x_t, t, condition, x0_hint=None,
                      beam_width=3, temperature=1.0,
                      repetition_penalty=1.2, diversity_penalty=0.3):
        with torch.no_grad():
            if x_t.dim() == 1: x_t = x_t.unsqueeze(0)
            if condition.dim() == 1: condition = condition.unsqueeze(0)
            if t.dim() == 0: t = t.unsqueeze(0)
            if t.shape[0] != x_t.shape[0]: t = t.expand(x_t.shape[0])

            sig = inspect.signature(model.forward).parameters
            if 'x0_hint' in sig:
                outputs = model(condition, x_t, t, x0_hint=x0_hint)
            else:
                outputs = model(condition, x_t, t)

            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            B, L, V = probs.shape
            topk_probs, topk_ids = torch.topk(probs, beam_width, dim=-1)

            candidates = []
            for k in range(beam_width):
                next_tokens = topk_ids[:, :, k]
                score = torch.log(topk_probs[:, :, k] + 1e-9).sum()
                candidates.append((next_tokens, score))
            return candidates


# ──────────────────────────────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────────────────────────────

def _top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Zero out all vocab entries outside the top-k per position."""
    B, L, V = logits.shape
    if k >= V:
        return logits
    # Find the k-th largest value per position
    topk_vals, _ = torch.topk(logits, k, dim=-1)
    threshold = topk_vals[..., -1].unsqueeze(-1)   # [B, L, 1]
    logits = logits.masked_fill(logits < threshold, float('-inf'))
    return logits


def _batch_multinomial(probs: torch.Tensor) -> torch.Tensor:
    """
    Sample one token per (batch, position) from probs [B, L, V].
    Returns [B, L].
    """
    B, L, V = probs.shape
    flat = probs.view(B * L, V)
    # Guard against all-zero rows (e.g., after top-k on short sequences)
    flat = flat + 1e-9
    flat = flat / flat.sum(dim=-1, keepdim=True)
    sampled = torch.multinomial(flat, num_samples=1).squeeze(-1)  # [B*L]
    return sampled.view(B, L)