"""
reverse_process.py  — Final Correct Version
=============================================

KEY PRINCIPLE: generate() must be byte-for-byte identical to run_inference()
in inference.py, which is what produced BERTScore 0.75 at validation.

CRITICAL BUG IN PREVIOUS VERSION:
  We passed inference_mode=True to the model, but the model was NEVER
  called with inference_mode=True during training or validation.
  run_inference() (the validated path) does:
      model(input_ids, x0_est, t, x0_hint=hint)
  → inference_mode defaults to False.

  With inference_mode=True the model does two things differently:
    1. tgt_pad_mask = None  (training used tgt_pad_mask = tgt==PAD)
    2. Skips q_sample at t=0 (training always called q_sample)
  The model was never trained to handle these conditions → garbage output.

  Fix: do NOT pass inference_mode. Let it default to False, exactly
  as run_inference() did.

BUGS FIXED (vs original reverse_process.py)
--------------------------------------------
BUG 1  generate_beam() used for D3PM → all-Ṛ repetition.
       Use generate() (iterative refinement) from app1.py instead.
BUG 2  apply_diversity_penalty used logits.var() → noise injection.
       Fixed to logits - penalty * logits.mean(dim=1) — global suppression.
BUG 3  x0_hint (self-conditioning) never passed to model.
       Fixed: generate() passes x0_hint=hint every step.
BUG 4  params not forwarded from generate_beam() to p_sample_step().
       Fixed in generate_beam() (kept for reference, not for production use).
"""

import torch
import torch.nn.functional as F


class ReverseDiffusion:

    def __init__(self, scheduler):
        self.scheduler = scheduler

        # Attribute-style defaults for backward compat with any code
        # that sets  reverse_diffusion.temperature = 0.9 etc.
        # generate() prefers explicit kwargs and falls back to these.
        self.temperature        = 0.75
        self.repetition_penalty = 1.15
        self.diversity_penalty  = 0.0
        self.top_k              = 50

    # ------------------------------------------------------------------ #
    #  generate  — CORRECT D3PM iterative refinement                      #
    #  Exact equivalent of run_inference() in inference.py                #
    # ------------------------------------------------------------------ #
    def generate(
        self,
        model,
        condition,
        num_steps          = None,
        temperature        = None,
        top_k              = None,
        repetition_penalty = None,
        diversity_penalty  = None,
    ):
        """
        D3PM iterative refinement — identical to run_inference() in inference.py,
        which is the validated path (BERTScore 0.75).

        Algorithm:
          x0_est = all [MASK]
          for t = T-1 down to 0:
            logits = model(src, x0_est, t, x0_hint=hint)
                     ↑ inference_mode NOT passed (defaults to False)
                     ↑ this exactly matches training/validation
            apply penalties, temperature, top_k
            if t > 0: x0_est = multinomial(softmax(logits))   ← stochastic
            if t = 0: x0_est = argmax(softmax(logits))         ← deterministic
            hint = x0_est
        """
        # Resolve: explicit kwarg > object attribute
        temperature        = temperature        if temperature        is not None else self.temperature
        top_k              = top_k              if top_k              is not None else self.top_k
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.repetition_penalty
        diversity_penalty  = diversity_penalty  if diversity_penalty  is not None else self.diversity_penalty

        if num_steps is None:
            num_steps = self.scheduler.num_timesteps

        device = condition.device
        if condition.dim() == 1:
            condition = condition.unsqueeze(0)
        B, L = condition.shape

        T         = self.scheduler.num_timesteps
        step_size = max(1, T // num_steps)
        timesteps = list(range(T - 1, -1, -step_size))
        if timesteps[-1] != 0:
            timesteps.append(0)

        mask_id = model.mask_token_id
        x0_est  = torch.full((B, L), mask_id, dtype=torch.long, device=device)
        hint    = None

        model.eval()
        with torch.no_grad():
            for step_idx, t_val in enumerate(timesteps):
                t       = torch.full((B,), t_val, dtype=torch.long, device=device)
                is_last = (step_idx == len(timesteps) - 1)

                # ── CRITICAL: do NOT pass inference_mode ──────────────────
                # inference_mode defaults to False inside SanskritModel /
                # D3PMCrossAttention. This matches run_inference() exactly.
                # Passing inference_mode=True changes tgt_pad_mask and
                # q_sample behaviour — the model was never trained for that.
                logits, _ = model(condition, x0_est, t, x0_hint=hint)

                # Repetition penalty
                if repetition_penalty != 1.0:
                    logits = apply_repetition_penalty(
                        logits, x0_est, repetition_penalty
                    )

                # Diversity penalty (correct: global mean suppression)
                if diversity_penalty > 0.0:
                    logits = apply_diversity_penalty(logits, diversity_penalty)

                logits = logits / max(temperature, 1e-5)

                if top_k > 0:
                    logits = top_k_filter(logits, top_k)

                probs = F.softmax(logits, dim=-1)

                # Stochastic at every step except the last (argmax at t=0)
                if is_last:
                    x0_est = torch.argmax(probs, dim=-1)
                else:
                    x0_est = batch_multinomial(probs)

                hint = x0_est

        return x0_est   # (B, L)

    # ------------------------------------------------------------------ #
    #  p_sample_step  — used by generate_beam (not for production)        #
    # ------------------------------------------------------------------ #
    def p_sample_step(
        self,
        model,
        x_t,
        t,
        condition,
        beam_width         = 3,
        temperature        = 1.0,
        repetition_penalty = 1.2,
        diversity_penalty  = 0.3,
        x0_hint            = None,
    ):
        with torch.no_grad():
            if x_t.dim() == 1:       x_t       = x_t.unsqueeze(0)
            if condition.dim() == 1: condition  = condition.unsqueeze(0)
            if t.dim() == 0:         t          = t.unsqueeze(0)
            if t.shape[0] != x_t.shape[0]:
                t = t.expand(x_t.shape[0])

            # No inference_mode — matches training convention
            logits, _ = model(condition, x_t, t, x0_hint=x0_hint)

            logits = logits / max(temperature, 1e-5)

            if repetition_penalty != 1.0:
                logits = apply_repetition_penalty(logits, x_t, repetition_penalty)
            if diversity_penalty > 0.0:
                logits = apply_diversity_penalty(logits, diversity_penalty)

            probs = F.softmax(logits, dim=-1)
            B, L, V = probs.shape

            topk_probs, topk_ids = torch.topk(probs, beam_width, dim=-1)
            candidates = []
            for k in range(beam_width):
                next_tokens = topk_ids[:, :, k]
                score       = torch.log(topk_probs[:, :, k] + 1e-9).sum()
                candidates.append((next_tokens, score))
            return candidates

    # ------------------------------------------------------------------ #
    #  generate_beam  — kept for reference; NOT the correct D3PM method   #
    # ------------------------------------------------------------------ #
    def generate_beam(
        self,
        model,
        condition,
        beam_width         = 3,
        num_steps          = None,
        temperature        = None,
        repetition_penalty = None,
        diversity_penalty  = None,
    ):
        """
        WARNING: do NOT call this from app1.py for D3PM generation.
        generate_beam() forces every position to the same top-k token
        → all-Ṛ / all-rud repetition. Use generate() instead.
        Kept only for experimental reference.
        """
        temperature        = temperature        if temperature        is not None else self.temperature
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.repetition_penalty
        diversity_penalty  = diversity_penalty  if diversity_penalty  is not None else self.diversity_penalty
        if num_steps is None:
            num_steps = self.scheduler.num_timesteps

        device = condition.device
        if condition.dim() == 1: condition = condition.unsqueeze(0)
        B, L = condition.shape

        x_init = torch.full((B, L), fill_value=model.mask_token_id,
                            dtype=torch.long, device=device)
        beams     = [(x_init, 0.0)]
        best_hint = None

        for step in reversed(range(num_steps)):
            t_tensor  = torch.full((B,), step, dtype=torch.long, device=device)
            new_beams = []
            for x_t, score in beams:
                candidates = self.p_sample_step(
                    model, x_t, t_tensor, condition,
                    beam_width         = beam_width,
                    temperature        = temperature,
                    repetition_penalty = repetition_penalty,
                    diversity_penalty  = diversity_penalty,
                    x0_hint            = best_hint,
                )
                for tokens, new_score in candidates:
                    new_beams.append((tokens, score + new_score.item()))

            new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)
            beams     = new_beams[:beam_width]
            best_hint = beams[0][0]

        return beams[0][0]   # (B, L)


# ── Penalty helpers ────────────────────────────────────────────────────────

def apply_repetition_penalty(logits, prev_tokens, penalty=1.2):
    """Down-weight tokens already present in the sequence."""
    for b in range(logits.shape[0]):
        for token_id in set(prev_tokens[b].tolist()):
            if token_id > 4:
                logits[b, :, token_id] = logits[b, :, token_id] / penalty
    return logits


def apply_diversity_penalty(logits, penalty=0.3):
    """
    Correct diversity penalty: suppress globally dominant tokens.
    logits -= penalty * mean(logits, dim=1)  [sequence dimension]
    """
    global_mean = logits.mean(dim=1, keepdim=True)   # [B, 1, V]
    return logits - penalty * global_mean


def top_k_filter(logits, k):
    B, L, V = logits.shape
    if k >= V: return logits
    topk_vals, _ = torch.topk(logits, k, dim=-1)
    return logits.masked_fill(logits < topk_vals[..., -1].unsqueeze(-1), float('-inf'))


def batch_multinomial(probs):
    B, L, V = probs.shape
    flat = probs.view(B * L, V) + 1e-9
    flat = flat / flat.sum(dim=-1, keepdim=True)
    return torch.multinomial(flat, 1).squeeze(-1).view(B, L)