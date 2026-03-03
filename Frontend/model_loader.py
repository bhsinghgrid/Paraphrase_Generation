# # import torch
# # import os
# # import torch.nn.functional as F
# # import sys
# #
# # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# #
# # from model.tokenizer import SanskritTokenizer
# # from model.new_d3pm_model import SanskritModel
# # from diffusion.reverse_process import ReverseDiffusion
# # from diffusion.scheduler import OptimizedCosineScheduler
# #
# #
# # def top_k_filtering(logits, top_k):
# #     values, _ = torch.topk(logits, top_k)
# #     min_values = values[:, -1].unsqueeze(-1)
# #     return torch.where(logits < min_values, torch.full_like(logits, -1e10), logits)
# #
# #
# # def top_p_filtering(logits, top_p):
# #     sorted_logits, sorted_indices = torch.sort(logits, descending=True)
# #     cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
# #
# #     sorted_indices_to_remove = cumulative_probs > top_p
# #     sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
# #     sorted_indices_to_remove[:, 0] = 0
# #
# #     indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
# #     logits[indices_to_remove] = -1e10
# #     return logits
# #
# #
# # def apply_repetition_penalty(logits, generated, penalty):
# #     for token in set(generated):
# #         logits[:, token] /= penalty
# #     return logits
# #
# #
# # class ModelManager:
# #
# #     def __init__(self, config):
# #         self.config = config
# #         self.device = torch.device(config["training"]["device"])
# #         self.tokenizer = SanskritTokenizer(config["model"]["vocab_size"])
# #         self.model = None
# #         self.reverse_diffusion = None
# #
# #     # def load_model(self, model_type):
# #     #
# #     #     # 🔹 Set model type in config
# #     #     self.config["model_type"] = model_type
# #     #
# #     #     self.model = SanskritModel(self.config)
# #     #
# #     #     if model_type == "baseline_encoder_decoder":
# #     #         self.reverse_diffusion = None
# #     #         model_path = "production_model3/best_baseline_encoder_decoder.pt"
# #     #
# #     #     elif model_type == "baseline_cross_attention":
# #     #         self.reverse_diffusion = None
# #     #         model_path = "/Users/bhsingh/Documents/Generation/production_model3/best_d3pm_cross_attention.pt"
# #     #
# #     #     else:
# #     #         raise ValueError("Only baseline models allowed in UI")
# #     #
# #     #     self.model.load_state_dict(torch.load(model_path, map_location=self.device))
# #     #     self.model.to(self.device)
# #     #     self.model.eval()
# #     def load_model(self, model_type):
# #
# #         self.config["model_type"] = model_type
# #         self.model = SanskritModel(self.config)
# #
# #         if model_type == "baseline_encoder_decoder":
# #             self.reverse_diffusion = None
# #             model_path = "production_model3/best_baseline_encoder_decoder.pt"
# #
# #         elif model_type == "baseline_cross_attention":
# #             self.reverse_diffusion = None
# #             model_path = "/Users/bhsingh/Documents/Generation/production_model3/best_d3pm_cross_attention.pt"
# #
# #         else:
# #             raise ValueError("Only baseline models allowed in UI")
# #
# #         state_dict = torch.load(model_path, map_location=self.device)
# #
# #         # ✅ FIX HERE
# #         self.model.load_state_dict(state_dict, strict=False)
# #
# #         self.model.to(self.device)
# #         self.model.eval()
# #
# #     @torch.no_grad()
# #     def generate(
# #         self,
# #         text,
# #         beam_width=3,
# #         temperature=1.0,
# #         top_k=0,
# #         top_p=1.0,
# #         max_len=80,
# #         repetition_penalty=1.0
# #     ):
# #
# #         input_ids = self.tokenizer.encode(text)
# #         input_tensor = torch.tensor([input_ids], device=self.device)
# #
# #         generated = []
# #
# #         for _ in range(max_len):
# #
# #             logits = self.model(input_tensor, input_tensor)
# #             logits = logits[:, -1, :] / temperature
# #
# #             if repetition_penalty != 1.0:
# #                 logits = apply_repetition_penalty(logits, generated, repetition_penalty)
# #
# #             if top_k > 0:
# #                 logits = top_k_filtering(logits, top_k)
# #
# #             if top_p < 1.0:
# #                 logits = top_p_filtering(logits, top_p)
# #
# #             probs = F.softmax(logits, dim=-1)
# #             next_token = torch.multinomial(probs, 1)
# #
# #             generated.append(next_token.item())
# #             input_tensor = torch.cat([input_tensor, next_token], dim=1)
# #
# #         return self.tokenizer.decode(generated).strip()
#
# import torch
# import os
# import torch.nn.functional as F
# import sys
#
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#
# from model.tokenizer import SanskritTokenizer
# from model.new_d3pm_model import SanskritModel
# from diffusion.reverse_process import ReverseDiffusion
# from diffusion.scheduler import OptimizedCosineScheduler
#
#
# def top_k_filtering(logits, top_k):
#     values, _ = torch.topk(logits, top_k)
#     min_values = values[:, -1].unsqueeze(-1)
#     return torch.where(logits < min_values, torch.full_like(logits, -1e10), logits)
#
#
# def top_p_filtering(logits, top_p):
#     sorted_logits, sorted_indices = torch.sort(logits, descending=True)
#     cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
#
#     sorted_indices_to_remove = cumulative_probs > top_p
#     sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
#     sorted_indices_to_remove[:, 0] = 0
#
#     indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
#     logits[indices_to_remove] = -1e10
#     return logits
#
#
# def apply_repetition_penalty(logits, generated, penalty):
#     for token in set(generated):
#         logits[:, token] /= penalty
#     return logits
#
#
# class ModelManager:
#
#     def __init__(self, config):
#         self.config = config
#         self.device = torch.device(config["training"]["device"])
#         self.tokenizer = None
#         self.model = None
#         self.reverse_diffusion = None
#
#     # ==========================================
#     # Load Model (AUTO FIXED VERSION)
#     # ==========================================
#     def load_model(self, model_type):
#
#         if model_type == "baseline_encoder_decoder":
#             model_path = "production_model3/best_baseline_encoder_decoder.pt"
#
#         elif model_type == "baseline_cross_attention":
#             model_path = "/Users/bhsingh/Documents/Generation/production_model3/best_d3pm_cross_attention.pt"
#
#         else:
#             raise ValueError("Only baseline models allowed")
#
#         # 🔥 Load checkpoint FIRST
#         state_dict = torch.load(model_path, map_location=self.device)
#
#         # 🔥 Automatically detect vocab size from checkpoint
#         embedding_key = "model.src_embed.token_emb.weight"
#         real_vocab_size = state_dict[embedding_key].shape[0]
#
#         # 🔥 Update config to match checkpoint
#         self.config["model"]["vocab_size"] = real_vocab_size
#         self.config["model_type"] = model_type
#
#         # 🔥 Initialize tokenizer AFTER fixing vocab
#         self.tokenizer = SanskritTokenizer(real_vocab_size)
#
#         # 🔥 Build model AFTER fixing vocab
#         self.model = SanskritModel(self.config)
#
#         # 🔥 Load weights safely
#         self.model.load_state_dict(state_dict, strict=False)
#
#         self.model.to(self.device)
#         self.model.eval()
#
#     # ==========================================
#     # Generate
#     # ==========================================
#     @torch.no_grad()
#     def generate(
#         self,
#         text,
#         beam_width=3,
#         temperature=1.0,
#         top_k=0,
#         top_p=1.0,
#         max_len=80,
#         repetition_penalty=1.0
#     ):
#
#         input_ids = self.tokenizer.encode(text)
#
#         # 🔥 Safety check
#         max_token_id = max(input_ids)
#         vocab_size = self.config["model"]["vocab_size"]
#
#         if max_token_id >= vocab_size:
#             raise ValueError(
#                 f"Token id {max_token_id} exceeds vocab size {vocab_size}. "
#                 "Tokenizer and model vocab mismatch."
#             )
#
#         input_tensor = torch.tensor([input_ids], device=self.device)
#
#         generated = []
#
#         for _ in range(max_len):
#
#             logits = self.model(input_tensor, input_tensor)
#             logits = logits[:, -1, :] / temperature
#
#             if repetition_penalty != 1.0:
#                 logits = apply_repetition_penalty(logits, generated, repetition_penalty)
#
#             if top_k > 0:
#                 logits = top_k_filtering(logits, top_k)
#
#             if top_p < 1.0:
#                 logits = top_p_filtering(logits, top_p)
#
#             probs = F.softmax(logits, dim=-1)
#             next_token = torch.multinomial(probs, 1)
#
#             generated.append(next_token.item())
#             input_tensor = torch.cat([input_tensor, next_token], dim=1)
#
#         return self.tokenizer.decode(generated).strip()

# model_loader.py (defensive loader)
import os
import sys
import json
import pickle
import warnings
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.tokenizer import SanskritTokenizer
from model.new_d3pm_model import SanskritModel
from diffusion.reverse_process import ReverseDiffusion
from diffusion.scheduler import OptimizedCosineScheduler


def top_k_filtering(logits, top_k):
    values, _ = torch.topk(logits, top_k)
    min_values = values[:, -1].unsqueeze(-1)
    return torch.where(logits < min_values, torch.full_like(logits, -1e10), logits)


def top_p_filtering(logits, top_p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = -1e10
    return logits


def apply_repetition_penalty(logits, generated, penalty):
    for token in set(generated):
        logits[:, token] /= penalty
    return logits


class ModelManager:
    """
    Defensive ModelManager:
      - Detects vocab size from checkpoint embedding
      - Tries to load the same tokenizer used at training
      - If tokenizer not found, falls back but maps OOR tokens to an UNK/fallback index
      - Loads checkpoint weights with strict=False (tolerate extra keys)
      - Provides detailed debug prints so you can see mismatches
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config["training"]["device"])
        self.tokenizer = None
        self.model = None
        self.reverse_diffusion = None
        # fallback unk id if tokenizer doesn't provide it
        self._fallback_unk_id = None

    # -------------------------
    # Helpers to find files
    # -------------------------
    def _get_checkpoint_path(self, model_type):
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "production_model3")
        if model_type == "baseline_encoder_decoder":
            fname = "best_baseline_encoder_decoder.pt"
        elif model_type == "baseline_cross_attention":
            fname = "best_d3pm_cross_attention.pt"
        else:
            raise ValueError("Only baseline models supported from this loader for UI")
        p = os.path.join(base, fname)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p

    def _find_tokenizer_file(self, base_dir=None):
        if base_dir is None:
            base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "production_model3")
        if not os.path.isdir(base_dir):
            return None
        # Look for likely tokenizer filenames
        for fname in os.listdir(base_dir):
            low = fname.lower()
            if "token" in low or "vocab" in low or fname.endswith(".json") or fname.endswith(".pkl"):
                return os.path.join(base_dir, fname)
        return None

    def _attempt_tokenizer_load(self, path):
        if path is None:
            return None
        # Try loads in several ways
        try:
            # If class has a load/from_json/from_file method
            for method in ("from_json", "load", "from_file", "from_pretrained"):
                if hasattr(SanskritTokenizer, method):
                    try:
                        tok = getattr(SanskritTokenizer, method)(path)
                        if hasattr(tok, "encode"):
                            return tok
                    except Exception:
                        pass
            # Try constructing from json file content if tokenizer expects dict
            if path.endswith(".json"):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if hasattr(SanskritTokenizer, "from_dict"):
                        try:
                            tok = SanskritTokenizer.from_dict(data)
                            if hasattr(tok, "encode"):
                                return tok
                        except Exception:
                            pass
                except Exception:
                    pass
            # Try pickle
            try:
                with open(path, "rb") as f:
                    tok = pickle.load(f)
                    if hasattr(tok, "encode"):
                        return tok
            except Exception:
                pass
        except Exception:
            pass
        return None

    # -------------------------
    # Primary load_model
    # -------------------------
    def load_model(self, model_type):
        ckpt_path = self._get_checkpoint_path(model_type)
        print(f"[ModelManager] Loading checkpoint: {ckpt_path}")

        # load checkpoint (may be state_dict or dict with 'state_dict')
        raw = torch.load(ckpt_path, map_location=self.device)
        if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
            state_dict = raw["state_dict"]
        else:
            state_dict = raw

        # Try to detect token embedding key (common names)
        embedding_key = None
        for k in state_dict.keys():
            lk = k.lower()
            if "token_emb" in lk and k.endswith(".weight"):
                embedding_key = k
                break
            if "src_embed.token_emb.weight" in lk:
                embedding_key = k
                break

        # fallback: find any 2D tensor with large first dim (likely vocab)
        if embedding_key is None:
            for k, v in state_dict.items():
                if isinstance(v, torch.Tensor) and v.dim() == 2 and v.shape[0] > 100:
                    embedding_key = k
                    break

        if embedding_key is None:
            raise RuntimeError("Could not detect embedding weight key in checkpoint. Keys: " + ", ".join(list(state_dict.keys())[:20]))

        real_vocab_size = int(state_dict[embedding_key].shape[0])
        print(f"[ModelManager] Detected embedding key '{embedding_key}' with vocab_size={real_vocab_size}")

        # set config vocab and model_type
        self.config.setdefault("model", {})["vocab_size"] = real_vocab_size
        self.config["model_type"] = model_type

        # try to load the original tokenizer if available
        tok_file = self._find_tokenizer_file()
        tok = self._attempt_tokenizer_load(tok_file)
        if tok is not None:
            self.tokenizer = tok
            print(f"[ModelManager] Loaded tokenizer from: {tok_file}")
            # attempt to get unk id if tokenizer exposes it
            for attr in ("unk_id", "unk_token_id", "unk_index", "oov_token_id"):
                if hasattr(self.tokenizer, attr):
                    try:
                        self._fallback_unk_id = int(getattr(self.tokenizer, attr))
                        break
                    except Exception:
                        pass
        else:
            warnings.warn(
                "[ModelManager] No tokenizer file found (production_model3). "
                "Falling back to constructing a new SanskritTokenizer from vocab_size. "
                "THIS IS A TEMPORARY FALLBACK — results may be incorrect if token->id mappings differ."
            )
            self.tokenizer = SanskritTokenizer(real_vocab_size)
            # set fallback unk id to last token
            self._fallback_unk_id = real_vocab_size - 1

        # Build model after we know vocab size
        self.model = SanskritModel(self.config)

        # Load weights (non-strict to ignore diffusion-only keys)
        try:
            self.model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            warnings.warn(f"[ModelManager] load_state_dict raised: {e}. Continuing with strict=False load attempt.")
            # try to load piecewise if possible
            self.model.load_state_dict(state_dict, strict=False)

        self.model.to(self.device)
        self.model.eval()

        # Try to get embedding size from live model (some models wrap embedding under different attr)
        emb_size = None
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Embedding):
                # heuristics: token emb usually has num_embeddings equal to vocab
                if module.num_embeddings >= 100 and (emb_size is None or module.num_embeddings > emb_size):
                    emb_size = module.num_embeddings
        if emb_size is None:
            emb_size = real_vocab_size
        print(f"[ModelManager] Final model embedding size = {emb_size}")

        # set fallback unk if not already
        if self._fallback_unk_id is None:
            self._fallback_unk_id = emb_size - 1

        # diffusion not used for baseline UI loader
        self.reverse_diffusion = None

        # Save a small summary / debug file so you can inspect easily
        try:
            debug_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "production_model3", "loader_debug.json")
            with open(debug_path, "w", encoding="utf-8") as f:
                json.dump({
                    "checkpoint": ckpt_path,
                    "detected_embedding_key": embedding_key,
                    "detected_vocab_size": real_vocab_size,
                    "model_embedding_size": emb_size,
                    "tokenizer_file": tok_file,
                    "fallback_unk_id": int(self._fallback_unk_id)
                }, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

        print("[ModelManager] Model & tokenizer ready (loaded with strict=False). Debug info written to production_model3/loader_debug.json if possible.")

    # -------------------------
    # Generation (defensive)
    # -------------------------
    @torch.no_grad()
    def generate(
            self,
            text,
            beam_width=3,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            max_len=80,
            repetition_penalty=1.0,
            emergency_clip=True,  # fallback behavior toggle
    ):

        # -------------------------
        # 1) encode (tokenizer must be loaded)
        # -------------------------
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded. Call load_model(...) first.")

        input_ids = self.tokenizer.encode(text)
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()

        if len(input_ids) == 0:
            # avoid empty sequences
            input_ids = [self.config["model"].get("mask_token_id", 0)]

        # -------------------------
        # 2) diagnostics
        # -------------------------
        vocab_size = int(self.config["model"]["vocab_size"])
        max_token = max(input_ids)
        bad_tokens = [t for t in input_ids if t >= vocab_size or t < 0]

        if bad_tokens:
            # detailed log for debugging
            print("=== ModelManager WARNING: tokenizer -> model vocab mismatch ===")
            print(f"model vocab_size = {vocab_size}")
            print(f"max token id from tokenizer = {max_token}")
            print(f"number of OOR tokens = {len(bad_tokens)}; examples: {bad_tokens[:20]}")
            # write to debug file
            try:
                debug_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "..", "production_model3", "tokenizer_mismatch_debug.json")
                with open(debug_path, "w", encoding="utf-8") as df:
                    json.dump({
                        "vocab_size": vocab_size,
                        "max_token": int(max_token),
                        "num_oob": len(bad_tokens),
                        "examples_oob": bad_tokens[:50],
                        "input_text_sample": text[:500]
                    }, df, ensure_ascii=False, indent=2)
                print(f"[ModelManager] wrote tokenizer mismatch debug file to: {debug_path}")
            except Exception:
                pass

            # emergency fallback handling
            if emergency_clip:
                # find fallback id: prefer tokenizer.unk_id or tokenizer.unk_token_id or mask id, else last id
                fallback = None
                for attr in ("unk_id", "unk_token_id", "oov_token_id", "unk_index"):
                    if hasattr(self.tokenizer, attr):
                        try:
                            fallback = int(getattr(self.tokenizer, attr))
                            break
                        except Exception:
                            pass
                if fallback is None:
                    # try mask token id (commonly 0), else last embedding index
                    fallback = int(
                        self.config.get("diffusion", {}).get("mask_token_id", 0)) if "diffusion" in self.config else 0
                    if fallback >= vocab_size or fallback < 0:
                        fallback = vocab_size - 1

                print(f"[ModelManager] emergency clipping: mapping OOR tokens -> fallback id {fallback}")

                input_ids = [t if 0 <= t < vocab_size else fallback for t in input_ids]
            else:
                raise ValueError(
                    f"Tokenizer produced token id {max_token} >= model vocab_size {vocab_size}. "
                    "Set emergency_clip=True to auto-map OOR tokens, or load the exact tokenizer used during training."
                )

        # -------------------------
        # 3) prepare tensor (long dtype)
        # -------------------------
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        # -------------------------
        # 4) generation loop (autoregressive baseline)
        # -------------------------
        generated = []
        for _ in range(max_len):
            out = self.model(input_tensor, input_tensor)
            logits = out[0] if isinstance(out, tuple) else out
            logits = logits[:, -1, :] / float(temperature)

            if repetition_penalty != 1.0:
                logits = apply_repetition_penalty(logits, generated, repetition_penalty)

            if top_k > 0:
                logits = top_k_filtering(logits, top_k)

            if top_p < 1.0:
                logits = top_p_filtering(logits, top_p)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

            token_id = int(next_token.item())
            generated.append(token_id)
            input_tensor = torch.cat([input_tensor, next_token], dim=1)

        return self.tokenizer.decode(generated).strip()