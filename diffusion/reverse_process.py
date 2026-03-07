"""
reverse_process.py — Fixed
===========================
Two bugs fixed from the original:

BUG 1 (critical): generate_beam() passed x_t (noisy) as `tgt` to model.
  The model does q_sample(tgt, t) internally — so x_t got double-noised.
  Fix: pass x0_estimate (current clean guess) as tgt. Model noises it correctly.

BUG 2: apply_diversity_penalty used logits.var(dim=-1) — this adds the
  variance of each position's own distribution back to itself, which is
  mathematically meaningless and just injects noise.
  Fix: penalize tokens that are uniformly high-probability across ALL positions
  (global common tokens). This genuinely promotes diversity.
"""

import torch
import torch.nn.functional as F


class ReverseDiffusion:
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def p_sample_step(
        self,
        model,
        x_t,
        t,
        condition,
        beam_width=3,
        temperature=1.0,
        repetition_penalty=1.2,
        diversity_penalty=0.3
    ):
        """
        Single reverse step with temperature + penalties.
        """

        with torch.no_grad():

            # ---- Shape safety ----
            if x_t.dim() == 1:
                x_t = x_t.unsqueeze(0)

            if condition.dim() == 1:
                condition = condition.unsqueeze(0)

            if t.dim() == 0:
                t = t.unsqueeze(0)

            if t.shape[0] != x_t.shape[0]:
                t = t.expand(x_t.shape[0])

            # ---- Model forward ----
            logits, _ = model(condition, x_t, t)

            # ---- Temperature scaling ----
            logits = logits / temperature

            # ---- Repetition penalty (FIXED VERSION) ----
            if repetition_penalty != 1.0:
                logits = apply_repetition_penalty(
                    logits, x_t, repetition_penalty
                )

            # ---- Diversity penalty ----
            if diversity_penalty > 0:
                logits = apply_diversity_penalty(
                    logits, diversity_penalty
                )

            probs = F.softmax(logits, dim=-1)

            B, L, V = probs.shape

            # ---- Top-k beam expansion ----
            topk_probs, topk_ids = torch.topk(
                probs, beam_width, dim=-1
            )

            candidates = []

            for k in range(beam_width):
                next_tokens = topk_ids[:, :, k]
                score = torch.log(
                    topk_probs[:, :, k] + 1e-9
                ).sum()
                candidates.append((next_tokens, score))

            return candidates

    def generate_beam(
        self,
        model,
        condition,
        beam_width=3,
        num_steps=None,
        temperature=1.0,
        repetition_penalty=1.2,
        diversity_penalty=0.3
    ):
        """
        Beam-search reverse diffusion with temperature.
        """

        if num_steps is None:
            num_steps = self.scheduler.num_timesteps

        device = condition.device

        if condition.dim() == 1:
            condition = condition.unsqueeze(0)

        B, L = condition.shape

        # 🔥 Better initialization: start from MASK
        x_init = torch.full(
            (B, L),
            fill_value=model.mask_token_id,
            dtype=torch.long,
            device=device
        )

        beams = [(x_init, 0.0)]

        for step in reversed(range(num_steps)):

            new_beams = []

            for x_t, score in beams:

                t_tensor = torch.full(
                    (B,),
                    step,
                    dtype=torch.long,
                    device=device
                )

                candidates = self.p_sample_step(
                    model,
                    x_t,
                    t_tensor,
                    condition,
                    beam_width,
                    temperature,
                    repetition_penalty,
                    diversity_penalty
                )

                for tokens, new_score in candidates:
                    new_beams.append(
                        (tokens, score + new_score)
                    )

            # ---- Keep top beams ----
            new_beams = sorted(
                new_beams,
                key=lambda x: x[1],
                reverse=True
            )

            beams = new_beams[:beam_width]

        best_tokens, best_score = beams[0]
        return best_tokens



    def generate(
        self,
        model,
        condition,
        num_steps=None,
        temperature=0.8,
        top_k=50,
        repetition_penalty=1.2,
        diversity_penalty=0.0,
    ):
        """
        Correct D3PM iterative refinement.

        x0_est starts as all [MASK].
        Each step: forward(src=condition, tgt=x0_est, t)
          → model applies q_sample(x0_est, t) internally
          → predicts cleaner x0
          → x0_est updated

        diversity_penalty: reduces probability of tokens that are
        globally dominant across all sequence positions (not logits.var()).
        """
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
        # Start: know nothing → all MASK is our initial clean estimate
        x0_est = torch.full((B, L), mask_id, dtype=torch.long, device=device)
        hint   = None

        model.eval()
        with torch.no_grad():
            for step_idx, t_val in enumerate(timesteps):
                t       = torch.full((B,), t_val, dtype=torch.long, device=device)
                is_last = (step_idx == len(timesteps) - 1)

                # KEY: pass x0_est as tgt — model noises it internally
                import inspect
                sig = inspect.signature(model.forward).parameters
                if 'x0_hint' in sig:
                    outputs = model(condition, x0_est, t, x0_hint=hint)
                else:
                    outputs = model(condition, x0_est, t)

                logits = outputs[0] if isinstance(outputs, tuple) else outputs

                # Repetition penalty: down-weight tokens already in sequence
                if repetition_penalty != 1.0:
                    logits = apply_repetition_penalty(logits, x0_est, repetition_penalty)

                # Diversity penalty: reduce globally dominant tokens
                if diversity_penalty > 0.0:
                    logits = apply_diversity_penalty(logits, diversity_penalty)

                # Temperature + top-k
                logits = logits / max(temperature, 1e-5)
                if top_k > 0:
                    logits = top_k_filter(logits, top_k)

                probs = F.softmax(logits, dim=-1)

                if is_last:
                    x0_est = torch.argmax(probs, dim=-1)
                else:
                    x0_est = batch_multinomial(probs)

                hint = x0_est

        return x0_est


# ── Penalty functions ─────────────────────────────────────────────────

def apply_repetition_penalty(logits, prev_tokens, penalty=1.2):
    """
    Down-weight tokens that already appear in the current sequence.
    Prevents मनो मनो मनो repetition loops.
    penalty=1.0 → no effect
    penalty=1.2 → mild suppression of repeated tokens
    penalty=2.0 → strong suppression
    """
    B, L, V = logits.shape
    for b in range(B):
        for token_id in set(prev_tokens[b].tolist()):
            if token_id > 4:   # don't penalize special tokens
                logits[b, :, token_id] = logits[b, :, token_id] / penalty
    return logits


def apply_diversity_penalty(logits, penalty=0.5):
    """
    Correct diversity penalty: penalize tokens that are globally dominant
    across ALL sequence positions. This forces the model to use less
    common tokens, increasing output diversity.

    Method: compute mean probability across positions, subtract penalty
    times that mean. Tokens uniformly high everywhere get suppressed.

    penalty=0.0 → no diversity enforcement
    penalty=0.5 → moderate diversity
    penalty=1.0 → strong diversity (may hurt coherence)
    """
    # Mean logit across all positions: [B, V]
    global_mean = logits.mean(dim=1, keepdim=True)   # [B, 1, V]
    # Subtract scaled global mean — suppresses globally common tokens
    return logits - penalty * global_mean


def top_k_filter(logits, k):
    B, L, V = logits.shape
    if k >= V:
        return logits
    topk_vals, _ = torch.topk(logits, k, dim=-1)
    threshold = topk_vals[..., -1].unsqueeze(-1)
    return logits.masked_fill(logits < threshold, float('-inf'))


def batch_multinomial(probs):
    B, L, V = probs.shape
    flat = probs.view(B * L, V) + 1e-9
    flat = flat / flat.sum(dim=-1, keepdim=True)
    return torch.multinomial(flat, 1).squeeze(-1).view(B, L)