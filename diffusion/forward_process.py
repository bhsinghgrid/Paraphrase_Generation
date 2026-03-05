import torch


class AbsorbingForwardProcess:
    def __init__(self, scheduler, mask_id=0):
        self.scheduler = scheduler
        self.mask_id = mask_id

    # def q_sample(self, x0, t):
    #     B, L = x0.shape
    #     device = x0.device
    #     mask_token_id = self.scheduler.mask_token_id
    #
    #     if t.dim() == 0: t = t.unsqueeze(0)
    #     if t.shape[0] != B: t = t.expand(B)
    #
    #     # 🔥 THE FIX: Use t.cpu() to index the CPU alphas, then move the result to the target device
    #     alpha_bar_t = self.scheduler.alphas[t.cpu()].to(device).unsqueeze(1)
    #
    #     # Ultra-fast boolean masking on GPU
    #     rand_tensor = torch.rand((B, L), device=device)
    #     mask_condition = rand_tensor > alpha_bar_t
    #
    #     x_t_ids = x0.clone()
    #     x_t_ids[mask_condition] = mask_token_id
    #
    #     return None, x_t_ids
    # def q_sample(self, x_0, t):
    #     # Get the noise schedule (alpha_t)
    #     alpha_t = self.scheduler.get_alpha(t).view(-1, 1)  # Probability of being correct
    #
    #     # Calculate transition probabilities
    #     # alpha_t: stay same
    #     # beta_t: move to [MASK]
    #     # gamma_t: move to a random token (Uniform)
    #
    #     beta_t = (1 - alpha_t) * 0.8  # 80% of noise goes to MASK
    #     gamma_t = (1 - alpha_t) * 0.2  # 20% of noise goes to Random Token
    #
    #     r = torch.rand(x_0.shape, device=x_0.device)
    #
    #     x_t = x_0.clone()
    #
    #     # 1. Masking
    #     mask_indices = (r > alpha_t) & (r <= (alpha_t + beta_t))
    #     x_t[mask_indices] = self.mask_id
    #
    #     # 2. Uniform Random (The "Research" upgrade)
    #     # This prevents the "य य" repetition by introducing entropy
    #     random_indices = (r > (alpha_t + beta_t))
    #     random_tokens = torch.randint(0, self.vocab_size, x_0.shape, device=x_0.device)
    #     x_t[random_indices] = random_tokens[random_indices]
    #
    #     return x_0, x_t
    # def q_sample(self, x_0, t, token_frequencies):
    #     alpha_t = self.scheduler.get_alpha(t).view(-1, 1)
    #
    #     # 🚀 RESEARCH UPGRADE: Modify mask probability based on frequency
    #     # High frequency tokens get masked MORE easily (speeding up their learning)
    #     # Low frequency (important) tokens are preserved LONGER
    #     importance_weights = torch.stack([token_frequencies[ids] for ids in x_0])
    #
    #     # Adjusted alpha (stay probability)
    #     # If importance is high, we increase the chance of staying original
    #     adjusted_alpha = alpha_t * importance_weights
    #
    #     r = torch.rand(x_0.shape, device=x_0.device)
    #     x_t = x_0.clone()
    #
    #     mask_indices = (r > adjusted_alpha)
    #     x_t[mask_indices] = self.mask_id
    #
    #     return x_0, x_t
    def q_sample(self, x_0, t, importance_map=None):
        """
        x_0: [Batch, SeqLen] - Clean Targets
        t: [Batch] - Timesteps
        importance_map: [Vocab] - Tensor of importance weights (0.1 to 1.0)
        """
        # if hasattr(self.scheduler, 'get_alpha'):
        #     alpha_t = self.scheduler.get_alpha(t).view(-1, 1)
        # else:
        #     # standard attribute name in most diffusers
        #     alpha_t = self.scheduler.alphas_cumprod[t].view(-1, 1)
        # # If no map provided, fall back to standard random masking
        # if importance_map is not None:
        #     # Map rare tokens to high probability of staying clean
        #     token_weights = importance_map[x_0]
        #     keep_prob = alpha_t * token_weights
        # else:
        #     keep_prob = alpha_t
        #
        # r = torch.rand(x_0.shape, device=x_0.device)
        # x_t = x_0.clone()
        #
        # # Mask indices where random value is greater than our (weighted) keep probability
        # mask_indices = (r > keep_prob)
        # x_t[mask_indices] = self.mask_id
        #
        # return x_0, x_t
        alpha_t = self.scheduler.get_alpha(t).view(-1, 1)  # Probability of staying original

        # 2. Force it to the same device as x_0 (MPS)
        alpha_t = alpha_t.to(x_0.device)

        if importance_map is not None:
            # Ensure importance_map and its result are on the correct device
            importance_weights = importance_map[x_0].to(x_0.device)
            keep_prob = alpha_t * importance_weights
        else:
            keep_prob = alpha_t

        # 3. Create 'r' on the correct device from the start
        r = torch.rand(x_0.shape, device=x_0.device)

        x_t = x_0.clone()
        mask_indices = (r > keep_prob)  # Now both are on MPS!
        x_t[mask_indices] = self.mask_id

        return x_0, x_t