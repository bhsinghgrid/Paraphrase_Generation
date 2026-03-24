import torch
import torch.nn.functional as F


class ReverseDiffusion:
    """
    Stable reverse diffusion with:
    - Beam search
    - Self conditioning
    - Temperature sampling
    - Repetition penalty
    - Diversity penalty
    """

    def __init__(self, scheduler):

        self.scheduler = scheduler

        self.temperature = 0.75
        self.repetition_penalty = 1.15
        self.diversity_penalty = 0.0
        self.length_penalty = 1.0

    # ------------------------------------------------
    # penalties
    # ------------------------------------------------

    def apply_repetition_penalty(self, logits, tokens):

        B, L, V = logits.shape

        for b in range(B):

            used = set(tokens[b].tolist())

            for token_id in used:
                logits[b, :, token_id] /= self.repetition_penalty

        return logits

    def apply_diversity_penalty(self, logits):

        if self.diversity_penalty == 0:
            return logits

        logits_var = logits.var(dim=-1, keepdim=True)
        return logits + self.diversity_penalty * logits_var

    # ------------------------------------------------
    # single reverse step
    # ------------------------------------------------

    def p_sample_step(self, model, x_t, t, condition, self_cond=None, beam_width=3):

        with torch.no_grad():

            logits, hidden = model(condition, x_t, t, self_cond)

            logits = logits / self.temperature

            logits = self.apply_repetition_penalty(logits, x_t)
            logits = self.apply_diversity_penalty(logits)

            probs = F.softmax(logits, dim=-1)

            B, L, V = probs.shape

            topk_probs, topk_ids = torch.topk(probs, beam_width, dim=-1)

            candidates = []

            for k in range(beam_width):

                tokens = topk_ids[:, :, k]

                score = torch.log(topk_probs[:, :, k] + 1e-9).sum()

                candidates.append((tokens, score))

            return candidates

    # ------------------------------------------------
    # beam reverse diffusion
    # ------------------------------------------------

    def generate_beam(self, model, condition, beam_width=3, num_steps=None):

        if num_steps is None:
            num_steps = self.scheduler.num_timesteps

        device = condition.device

        if condition.dim() == 1:
            condition = condition.unsqueeze(0)

        B, L = condition.shape

        # ------------------------------------------------
        # BETTER LATENT INITIALIZATION
        # ------------------------------------------------

        x_init = condition.clone()

        mask = torch.rand_like(x_init.float()) < 0.5
        x_init[mask] = model.mask_token_id

        beams = [(x_init, 0.0)]

        self_cond = None

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
                    self_cond,
                    beam_width
                )

                for tokens, new_score in candidates:

                    length_norm = tokens.shape[1] ** self.length_penalty

                    final_score = (score + new_score) / length_norm

                    new_beams.append((tokens, final_score))

            new_beams = sorted(
                new_beams,
                key=lambda x: x[1],
                reverse=True
            )

            beams = new_beams[:beam_width]

            # self conditioning
            self_cond = beams[0][0]

        best_tokens, best_score = beams[0]

        return best_tokens