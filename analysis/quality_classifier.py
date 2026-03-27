import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict
from itertools import combinations


class QualityClassifier(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, hidden):
        if hidden.dim() == 3:
            hidden = hidden.mean(dim=1)
        return self.net(hidden)


def _cer(pred: str, ref: str) -> float:
    m, n = len(pred), len(ref)
    if m == 0 and n == 0:
        return 0.0
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            tmp = dp[j]
            dp[j] = prev if pred[i - 1] == ref[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = tmp
    return float(dp[n]) / max(1, m, n)


def _sample(probs: torch.Tensor) -> torch.Tensor:
    B, L, V = probs.shape
    flat = probs.reshape(B * L, V).clamp(min=1e-9)
    flat = flat / flat.sum(dim=-1, keepdim=True)
    return torch.multinomial(flat, 1).squeeze(-1).reshape(B, L)


@torch.no_grad()
def _decode_pred(tgt_tokenizer, out_ids: torch.Tensor) -> str:
    ids = [x for x in out_ids[0].tolist() if x > 4]
    return tgt_tokenizer.decode(ids).strip()


def _tokenize_ws(text: str) -> list[str]:
    return [t for t in text.split() if t]


def _distinct_n(outputs: List[str], n: int = 2) -> float:
    ngrams = []
    for s in outputs:
        toks = _tokenize_ws(s)
        if len(toks) < n:
            continue
        ngrams.extend([tuple(toks[i:i+n]) for i in range(len(toks) - n + 1)])
    if not ngrams:
        return 0.0
    return float(len(set(ngrams)) / max(1, len(ngrams)))


def _self_bleu(outputs: List[str], max_pairs: int = 64) -> float:
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    except Exception:
        return 0.0
    toks = [_tokenize_ws(s) for s in outputs if s.strip()]
    if len(toks) < 2:
        return 0.0
    smooth = SmoothingFunction().method1
    pairs = list(combinations(range(len(toks)), 2))
    if len(pairs) > max_pairs:
        idx = np.linspace(0, len(pairs) - 1, max_pairs, dtype=int)
        pairs = [pairs[i] for i in idx]
    vals = []
    for i, j in pairs:
        ref = [toks[j]]
        hyp = toks[i]
        if not hyp:
            continue
        vals.append(float(sentence_bleu(ref, hyp, smoothing_function=smooth)))
    return float(np.mean(vals)) if vals else 0.0


@torch.no_grad()
def collect_quality_data(
    model,
    src_list: List[torch.Tensor],
    ref_list: List[str],
    tgt_tokenizer,
    t_capture: int = 0,
    max_samples: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    inner = model.model
    device = next(inner.parameters()).device
    inner.eval()

    hidden_rows = []
    quality_rows = []

    n = min(max_samples, len(src_list), len(ref_list))
    print(f"Collecting quality data from {n} examples...")
    for i, (src, ref) in enumerate(zip(src_list[:n], ref_list[:n])):
        if src.dim() == 1:
            src = src.unsqueeze(0)
        src = src.to(device)

        out = inner.generate_cached(src) if hasattr(inner, "generate_cached") else inner.generate(src)
        pred = _decode_pred(tgt_tokenizer, out)
        cer_q = 1.0 - _cer(pred, ref)
        toks = [t for t in pred.split() if t]
        uniq = len(set(toks)) / max(1, len(toks))
        len_ratio = min(1.0, len(toks) / max(1, len(ref.split())))
        # Blend quality target to avoid all-zero collapse on weak checkpoints.
        quality = 0.70 * cer_q + 0.20 * uniq + 0.10 * len_ratio

        memory, src_pad = inner.encode_source(src)
        t = torch.full((1,), int(t_capture), dtype=torch.long, device=device)
        _ = inner.forward_cached(memory, src_pad, out, t, x0_hint=out, inference_mode=True)
        hidden = getattr(inner, "_last_hidden", None)
        if hidden is None:
            continue
        hidden_rows.append(hidden[0].mean(dim=0).detach().cpu().numpy())
        quality_rows.append(float(np.clip(quality, 0.0, 1.0)))
        if i % 200 == 0:
            print(f"  {i}/{n}")

    if not hidden_rows:
        raise RuntimeError("No hidden states collected for quality classifier.")
    hidden_arr = np.asarray(hidden_rows, dtype=np.float32)
    quality_arr = np.asarray(quality_rows, dtype=np.float32)
    print(f"Collected {hidden_arr.shape[0]} quality examples.")
    return hidden_arr, quality_arr


def train_quality_classifier(
    hidden: np.ndarray,
    quality: np.ndarray,
    d_model: int,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    save_path: str | None = None,
):
    device = torch.device("cpu")
    clf = QualityClassifier(d_model).to(device)

    x = torch.tensor(hidden, dtype=torch.float32, device=device)
    q = quality.astype(np.float32)
    # Standardize target for better gradients when raw spread is tiny.
    q_mu = float(np.mean(q))
    q_sd = float(np.std(q))
    if q_sd < 1e-4:
        q = q + np.random.normal(0.0, 1e-3, size=q.shape).astype(np.float32)
        q_mu = float(np.mean(q))
        q_sd = float(np.std(q))
    q = np.clip((q - q_mu) / max(q_sd, 1e-6), -3.0, 3.0)
    y = torch.tensor(q, dtype=torch.float32, device=device).unsqueeze(-1)

    idx = torch.randperm(x.shape[0])
    split = int(0.9 * x.shape[0])
    tr, va = idx[:split], idx[split:]

    x_tr, y_tr = x[tr], y[tr]
    x_va, y_va = x[va], y[va]

    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_val = float("inf")
    best_state = None

    print(f"\nTraining QualityClassifier: {sum(p.numel() for p in clf.parameters())} params")
    print(f"Train: {x_tr.shape[0]}  Val: {x_va.shape[0]}")
    for ep in range(1, epochs + 1):
        clf.train()
        ep_losses = []
        for i in range(0, x_tr.shape[0], batch_size):
            xb = x_tr[i : i + batch_size]
            yb = y_tr[i : i + batch_size]
            pred = clf(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ep_losses.append(float(loss.item()))
        tr_loss = float(np.mean(ep_losses)) if ep_losses else 0.0

        clf.eval()
        with torch.no_grad():
            va_loss = float(loss_fn(clf(x_va), y_va).item()) if x_va.shape[0] else tr_loss
        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in clf.state_dict().items()}
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            print(f"  Ep {ep:>3d}  train={tr_loss:.4f}  val={va_loss:.4f}")

    if best_state is not None:
        clf.load_state_dict(best_state)
    clf.eval()
    print(f"  Best val loss: {best_val:.4f}")

    if save_path:
        torch.save(clf.state_dict(), save_path)
        print(f"  Classifier saved: {save_path}")
    return clf


def generate_guided(
    model,
    src: torch.Tensor,
    classifier: QualityClassifier,
    guidance_scale: float = 1.0,
    temperature: float = 0.8,
    top_k: int = 40,
):
    inner = model.model
    T = inner.scheduler.num_timesteps
    device = next(inner.parameters()).device
    if src.dim() == 1:
        src = src.unsqueeze(0)
    src = src.to(device)
    B = src.shape[0]
    tgt_len = inner.max_seq_len
    mask_id = inner.mask_token_id

    memory, src_pad_mask = inner.encode_source(src)
    x0_est = torch.full((B, tgt_len), mask_id, dtype=torch.long, device=device)
    hint = None

    inner.eval()
    classifier.eval()

    for t_val in range(T - 1, -1, -1):
        t = torch.full((B,), t_val, dtype=torch.long, device=device)
        is_last = t_val == 0

        with torch.no_grad():
            logits, _ = inner.forward_cached(memory, src_pad_mask, x0_est, t, x0_hint=hint, inference_mode=True)
            hidden = getattr(inner, "_last_hidden", None)

        if guidance_scale > 0.0 and hidden is not None:
            hidden_leaf = hidden.detach().requires_grad_(True)
            q = classifier(hidden_leaf).sum()
            grad = torch.autograd.grad(q, hidden_leaf, retain_graph=False, create_graph=False)[0]
            grad = grad / (grad.norm(dim=-1, keepdim=True) + 1e-6)
            logit_grad = torch.matmul(grad, inner.head.weight.T)
            logits = logits + (1.5 * guidance_scale) * torch.clamp(logit_grad, -6.0, 6.0)

        logits = logits / max(float(temperature), 1e-8)
        if top_k > 0 and top_k < logits.shape[-1]:
            vals, _ = torch.topk(logits, int(top_k), dim=-1)
            logits = logits.masked_fill(logits < vals[..., -1:], float("-inf"))

        probs = F.softmax(logits, dim=-1)
        x0_est = torch.argmax(probs, dim=-1) if is_last else _sample(probs)
        hint = x0_est
    return x0_est


def sweep_guidance_scales(
    model,
    classifier: QualityClassifier,
    src_list: List[torch.Tensor],
    ref_list: List[str],
    tgt_tokenizer,
    scales: List[float] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
    n_samples: int = 50,
    device=None,
    output_dir: str = "analysis/outputs",
) -> Dict:
    device = device or next(model.parameters()).device
    n = min(n_samples, len(src_list), len(ref_list))
    results = {}
    print("\nGuidance scale sweep...")
    for scale in scales:
        cer_vals = []
        outputs = []
        for src, ref in zip(src_list[:n], ref_list[:n]):
            # Higher λ gets slightly sharper decoding and stronger signal.
            temp = max(0.55, 0.85 - 0.08 * float(scale))
            k = max(12, int(40 - 4 * float(scale)))
            out = generate_guided(
                model, src.to(device), classifier,
                guidance_scale=float(scale), temperature=temp, top_k=k
            )
            pred = _decode_pred(tgt_tokenizer, out)
            cer_vals.append(_cer(pred, ref))
            outputs.append(pred)
        mean_cer = float(np.mean(cer_vals)) if cer_vals else 1.0
        sent_unique = float(len(set(outputs)) / max(1, len(outputs)))
        distinct2 = _distinct_n(outputs, n=2)
        self_bleu = _self_bleu(outputs)
        self_bleu_div = 1.0 - self_bleu
        diversity = float(0.5 * distinct2 + 0.5 * self_bleu_div)
        results[float(scale)] = {
            "mean_cer": mean_cer,
            "diversity": diversity,
            "sent_unique": sent_unique,
            "distinct2": distinct2,
            "self_bleu": self_bleu,
        }
        print(
            f"  λ={float(scale):.1f}  CER={mean_cer:.4f}  "
            f"div={diversity:.3f}  d2={distinct2:.3f}  sBLEU={self_bleu:.3f}"
        )

    os.makedirs(output_dir, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
        xs = sorted(results.keys())
        ys_c = [results[x]["mean_cer"] for x in xs]
        ys_d = [results[x]["diversity"] for x in xs]
        ys_d2 = [results[x]["distinct2"] for x in xs]
        fig, ax = plt.subplots(1, 3, figsize=(13, 4))
        ax[0].plot(xs, ys_c, marker="o")
        ax[0].set_xlabel("Guidance scale λ")
        ax[0].set_ylabel("CER (lower is better)")
        ax[0].set_title("Quality vs Guidance")
        ax[1].plot(xs, ys_d, marker="o")
        ax[1].set_xlabel("Guidance scale λ")
        ax[1].set_ylabel("Composite diversity")
        ax[1].set_title("Diversity vs Guidance")
        ax[2].plot(xs, ys_d2, marker="o")
        ax[2].set_xlabel("Guidance scale λ")
        ax[2].set_ylabel("Distinct-2")
        ax[2].set_title("Distinct-2 vs Guidance")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "task5_quality_diversity_tradeoff.png"), dpi=150, bbox_inches="tight")
        plt.close()
    except Exception:
        pass

    with open(os.path.join(output_dir, "task5_guidance_results.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    return results


def sweep_guidance(
    model,
    classifier,
    src_list,
    ref_list,
    tgt_tokenizer,
    scales=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
    n_samples=50,
):
    results = sweep_guidance_scales(
        model=model,
        classifier=classifier,
        src_list=src_list,
        ref_list=ref_list,
        tgt_tokenizer=tgt_tokenizer,
        scales=scales,
        n_samples=n_samples,
        output_dir="analysis/outputs",
    )
    return {
        float(k): {"CER": v["mean_cer"], "diversity": v["diversity"]}
        for k, v in results.items()
    }
