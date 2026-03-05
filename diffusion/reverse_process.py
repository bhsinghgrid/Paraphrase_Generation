# """
# Reverse Diffusion with Beam Search
# Improves generation quality over pure multinomial sampling.
# """
#
# import torch
# import torch.nn.functional as F
#
#
# class ReverseDiffusion:
#     """
#     Reverse diffusion with optional Beam Search generation.
#     """
#
#     def __init__(self, scheduler):
#         self.scheduler = scheduler
#
#     # --------------------------------------------------
#     # 🔄 Single Reverse Step (Beam Compatible)
#     # --------------------------------------------------
#     def p_sample_step(self, model, x_t, t, condition, beam_width=1):
#         """
#         Single reverse step with beam expansion.
#
#         Returns:
#             candidates: list of (tokens, score)
#         """
#         with torch.no_grad():
#
#             # -------------------------------
#             # ✅ SHAPE SAFETY FIXES
#             # -------------------------------
#             if x_t.dim() == 1:
#                 x_t = x_t.unsqueeze(0)
#
#             if condition.dim() == 1:
#                 condition = condition.unsqueeze(0)
#
#             if t.dim() == 0:
#                 t = t.unsqueeze(0)
#
#             if t.dim() == 1 and t.shape[0] != x_t.shape[0]:
#                 t = t.expand(x_t.shape[0])
#             # -------------------------------
#
#             logits, _ = model(condition, x_t, t)
#
#             logits = apply_diversity_penalty(logits, 0.5)
#             logits = apply_repetition_penalty(logits, x_t, 1.2)
#
#             probs = F.softmax(logits, dim=-1)
#
#             B, L, V = probs.shape
#
#             # Flatten for sampling
#             probs_flat = probs.view(-1, V)
#
#             # Top-k instead of multinomial
#             topk_probs, topk_ids = torch.topk(probs_flat, beam_width, dim=-1)
#
#             candidates = []
#             for k in range(beam_width):
#                 next_tokens = topk_ids[:, k].view(B, L)
#                 score = torch.log(topk_probs[:, k] + 1e-9).sum()
#                 candidates.append((next_tokens, score))
#
#             return candidates
#
#     # --------------------------------------------------
#     # 🔥 Beam Search Generation
#     # --------------------------------------------------
#     def generate_beam(self, model, condition, beam_width=3, num_steps=None):
#         """
#         Beam-search reverse diffusion.
#         """
#
#         if num_steps is None:
#             num_steps = self.scheduler.num_timesteps
#
#         device = condition.device
#
#         if condition.dim() == 1:
#             condition = condition.unsqueeze(0)
#
#         B, L = condition.shape
#
#         # Initialize beam list
#         beams = [(torch.zeros(B, L, dtype=torch.long, device=device), 0.0)]
#
#         for step in reversed(range(num_steps)):
#             print(f"Beam Reverse step {step}/{num_steps}")
#
#             new_beams = []
#
#             for x_t, score in beams:
#
#                 # ✅ FIX: make t (B,) instead of scalar
#                 t_tensor = torch.full(
#                     (B,),
#                     step,
#                     dtype=torch.long,
#                     device=device
#                 )
#
#                 candidates = self.p_sample_step(
#                     model,
#                     x_t,
#                     t_tensor,
#                     condition,
#                     beam_width
#                 )
#
#                 for tokens, new_score in candidates:
#                     new_beams.append((tokens, score + new_score))
#
#             # Keep top beams
#             new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)
#             beams = new_beams[:beam_width]
#
#         # Return best sequence
#         best_tokens, best_score = beams[0]
#         return best_tokens
#
#
# # --------------------------------------------------
# # 🎨 Penalties (unchanged logic)
# # --------------------------------------------------
#
# def apply_diversity_penalty(logits, penalty=0.5):
#     logits_var = logits.var(dim=-1, keepdim=True)
#     return logits + penalty * logits_var
#
#
# def apply_repetition_penalty(logits, prev_tokens, penalty=1.2):
#     B, L, V = logits.shape
#     for i in range(L):
#         for b in range(B):
#             if i > 0 and prev_tokens[b, i] == prev_tokens[b, :i].any():
#                 logits[b, i] *= (1.0 / penalty)
#     return logits

import torch
import torch.nn.functional as F


class ReverseDiffusion:
    """
    Reverse diffusion with Beam Search + Temperature + Proper Penalties.
    """

    def __init__(self, scheduler):
        self.scheduler = scheduler

    # --------------------------------------------------
    # 🔄 Single Reverse Step
    # --------------------------------------------------
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

    # --------------------------------------------------
    # 🔥 Beam Search Generation
    # --------------------------------------------------
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


def apply_repetition_penalty(logits, prev_tokens, penalty=1.2):
    """
    Penalize tokens that already appeared in the sequence.
    """

    B, L, V = logits.shape

    for b in range(B):
        used_tokens = set(prev_tokens[b].tolist())

        for token_id in used_tokens:
            logits[b, :, token_id] /= penalty

    return logits

def apply_diversity_penalty(logits, penalty=0.3):
    """
    Encourages variance in predictions.
    """
    logits_var = logits.var(dim=-1, keepdim=True)
    return logits + penalty * logits_var