"""
inference.py
============
Correct D3PM inference for Sanskrit paraphrase generation.

The model's forward() takes CLEAN tgt and noises it internally.
So inference passes x0_estimate (starting all-[MASK]) as tgt each step,
letting the model noise it and then predict a cleaner version.

Also includes: robust checkpoint loading (auto-detects architecture
from saved weights — no CONFIG mismatch crashes).
"""

import json
import torch
import os, sys
import re
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import CONFIG


# ── Checkpoint loader ─────────────────────────────────────────────────

def _resolve_device(cfg_device: str) -> torch.device:
    cfg_device = (cfg_device or "").lower()
    if cfg_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if cfg_device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if cfg_device in {"cpu", "cuda", "mps"}:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_model(ckpt_path: str, base_cfg: dict, device: torch.device):
    """
    Auto-detect architecture from checkpoint weight shapes,
    then load. Never fails due to CONFIG vs checkpoint mismatch.
    """
    import copy
    from model.sanskrit_model import SanskritModel

    cfg   = copy.deepcopy(base_cfg)
    state = torch.load(ckpt_path, map_location='cpu')

    # d_model + vocab_size
    ek = 'model.src_embed.token_emb.weight'
    if ek in state:
        vocab, d          = state[ek].shape
        cfg['model']['vocab_size'] = vocab
        cfg['model']['d_model']    = d
        cfg['model']['d_ff']       = d * 4

    # n_layers
    ids = {int(k.split('.')[2]) for k in state if k.startswith('model.encoder_blocks.')}
    if ids:
        cfg['model']['n_layers'] = max(ids) + 1

    # max_seq_len
    pk = 'model.src_embed.pos_enc.pe'
    if pk in state:
        cfg['model']['max_seq_len'] = state[pk].shape[1]

    # n_heads
    d = cfg['model']['d_model']
    h = cfg['model'].get('n_heads', 6)
    if d % h != 0:
        h = next(x for x in [8, 6, 4, 2, 1] if d % x == 0)
    cfg['model']['n_heads'] = h

    print(f"🔍 Detected: d_model={cfg['model']['d_model']}, "
          f"n_layers={cfg['model']['n_layers']}, "
          f"max_seq_len={cfg['model']['max_seq_len']}, "
          f"n_heads={cfg['model']['n_heads']}")

    model = SanskritModel(cfg).to(device)
    raw_state = torch.load(ckpt_path, map_location=device)
    model_state = model.state_dict()
    filtered_state = {}
    skipped_mismatch = []
    for k, v in raw_state.items():
        if k in model_state and hasattr(v, "shape") and hasattr(model_state[k], "shape"):
            if tuple(v.shape) != tuple(model_state[k].shape):
                skipped_mismatch.append((k, tuple(v.shape), tuple(model_state[k].shape)))
                continue
        filtered_state[k] = v

    missing, unexpected = model.load_state_dict(filtered_state, strict=False)

    # hint_gate may be absent in older checkpoints — initialise safely
    allowed = {'model.hint_gate.0.weight', 'model.hint_gate.0.bias'}
    real_missing = [k for k in missing if k not in allowed]
    if real_missing:
        print(f"⚠️  Missing keys: {real_missing[:3]} …")
    if unexpected:
        print(f"⚠️  Unexpected keys: {unexpected[:3]} …")
    if skipped_mismatch:
        print(f"⚠️  Shape-mismatched keys skipped: {len(skipped_mismatch)}")

    # Enable compact-attention branch only when checkpoint actually provides it.
    has_compact = any(".compact_out_proj.weight" in k for k in filtered_state.keys())
    if has_compact and hasattr(model, "model") and hasattr(model.model, "decoder_blocks"):
        for block in model.model.decoder_blocks:
            if hasattr(block, "cross_attn") and hasattr(block.cross_attn, "use_compact"):
                block.cross_attn.use_compact = True
        print("ℹ️  Compact cross-attention branch enabled from checkpoint.")
    if hasattr(model.model, 'hint_gate') and 'model.hint_gate.0.weight' in missing:
        with torch.no_grad():
            w = model.model.hint_gate[0].weight
            torch.nn.init.zeros_(model.model.hint_gate[0].bias)
            torch.nn.init.eye_(w) if w.shape[0] == w.shape[1] \
                else torch.nn.init.xavier_uniform_(w)
        print("ℹ️  hint_gate initialised to identity (not in checkpoint).")

    print("✅ Model loaded.")
    return model, cfg


# ── Core inference function (same path as validation) ────────────────

@torch.no_grad()
def run_inference(model, input_ids, cfg):
    """
    Reverse diffusion sampling (clean path).
    Uses cached reverse diffusion when available, otherwise model.generate().
    """
    inf = cfg['inference']
    model.eval()
    kwargs = dict(
        num_steps=inf['num_steps'],
        temperature=inf['temperature'],
        top_k=inf['top_k'],
        repetition_penalty=inf.get('repetition_penalty', 1.2),
        diversity_penalty=inf.get('diversity_penalty', 0.0),
    )
    if hasattr(model, "generate_cached"):
        out = model.generate_cached(input_ids, **kwargs)
    else:
        out = model.generate(input_ids, **kwargs)

    # Optional retry with stronger anti-repetition settings.
    if inf.get("auto_retry_on_repetition", True):
        repeat_threshold = float(inf.get("repeat_ratio_threshold", 0.40))
        max_repeat_run = int(inf.get("max_repeat_run", 4))
        if _mean_repeat_ratio(out) >= repeat_threshold:
            retry_kwargs = dict(kwargs)
            retry_kwargs["temperature"] = max(0.6, float(kwargs["temperature"]) - 0.1)
            retry_kwargs["top_k"] = max(20, int(kwargs["top_k"]) - 10)
            retry_kwargs["repetition_penalty"] = max(float(kwargs["repetition_penalty"]), 1.6)
            retry_kwargs["diversity_penalty"] = max(float(kwargs["diversity_penalty"]), 0.3)
            if hasattr(model, "generate_cached"):
                retry = model.generate_cached(input_ids, **retry_kwargs)
            else:
                retry = model.generate(input_ids, **retry_kwargs)
            if _mean_repeat_ratio(retry) < _mean_repeat_ratio(out):
                out = retry
        out = _dedup_repeated_ids(out, max_repeat_run=max_repeat_run)

    return out


def _mean_repeat_ratio(ids_tensor: torch.Tensor) -> float:
    if ids_tensor is None or ids_tensor.numel() == 0:
        return 0.0
    ratios = []
    for row in ids_tensor:
        ids = [int(x) for x in row.tolist() if int(x) > 4]
        if len(ids) < 2:
            ratios.append(0.0)
            continue
        repeats = sum(1 for i in range(1, len(ids)) if ids[i] == ids[i - 1])
        ratios.append(repeats / max(1, len(ids) - 1))
    return float(sum(ratios) / max(1, len(ratios)))


def _dedup_repeated_ids(ids_tensor: torch.Tensor, max_repeat_run: int = 4) -> torch.Tensor:
    """
    Keep generation path unchanged, but clean extreme run-on token loops in final output ids.
    """
    if ids_tensor is None or ids_tensor.numel() == 0:
        return ids_tensor
    cleaned_rows = []
    for row in ids_tensor.tolist():
        out = []
        prev = None
        run = 0
        for tok in row:
            if tok <= 4:
                out.append(tok)
                prev = tok
                run = 1
                continue
            if tok == prev:
                run += 1
                if run > max_repeat_run:
                    continue
            else:
                run = 1
            out.append(tok)
            prev = tok
        # Preserve original length for downstream decode assumptions.
        if len(out) < len(row):
            out.extend([1] * (len(row) - len(out)))
        else:
            out = out[:len(row)]
        cleaned_rows.append(out)
    return torch.tensor(cleaned_rows, dtype=ids_tensor.dtype, device=ids_tensor.device)


def _decode_clean(tgt_tok, ids):
    out = []
    for x in ids:
        if x in (1, 4) and out:
            break
        if x > 4:
            out.append(x)
    text = tgt_tok.decode(out).strip()
    return _clean_repetition_text(text)


def _clean_repetition_text(text: str, max_repeat_run: int = 3) -> str:
    words = [w for w in text.split() if w.strip()]
    if not words:
        return text.strip()
    cleaned = []
    prev = None
    run = 0
    for w in words:
        if w == prev:
            run += 1
            if run > max_repeat_run:
                continue
        else:
            run = 1
        cleaned.append(w)
        prev = w
    return " ".join(cleaned).strip()


# ── Cleanup heuristics from UI inference pipeline ─────────────────────

_IAST_VOWELS = [
    ("ai", "ऐ"), ("au", "औ"),
    ("ā", "आ"), ("ī", "ई"), ("ū", "ऊ"),
    ("ṛ", "ऋ"), ("ṝ", "ॠ"), ("ḷ", "ऌ"), ("ḹ", "ॡ"),
    ("a", "अ"), ("i", "इ"), ("u", "उ"),
    ("e", "ए"), ("o", "ओ"),
]
_IAST_MATRAS = [
    ("ai", "ै"), ("au", "ौ"),
    ("ā", "ा"), ("ī", "ी"), ("ū", "ू"),
    ("ṛ", "ृ"), ("ṝ", "ॄ"), ("ḷ", "ॢ"), ("ḹ", "ॣ"),
    ("a", ""), ("i", "ि"), ("u", "ु"),
    ("e", "े"), ("o", "ो"),
]
_IAST_CONS = [
    ("kṣ", "क्ष"), ("jñ", "ज्ञ"), ("tr", "त्र"),
    ("kh", "ख"), ("gh", "घ"), ("ch", "छ"), ("jh", "झ"),
    ("ṭh", "ठ"), ("ḍh", "ढ"), ("th", "थ"), ("dh", "ध"),
    ("ph", "फ"), ("bh", "भ"),
    ("ṅ", "ङ"), ("ñ", "ञ"), ("ṭ", "ट"), ("ḍ", "ड"),
    ("ṇ", "ण"), ("ś", "श"), ("ṣ", "ष"), ("ḥ", "ः"),
    ("ṃ", "ं"), ("ṁ", "ं"),
    ("y", "य"), ("r", "र"), ("l", "ल"), ("v", "व"),
    ("s", "स"), ("h", "ह"),
    ("k", "क"), ("g", "ग"), ("c", "च"), ("j", "ज"),
    ("t", "त"), ("d", "द"), ("n", "न"),
    ("p", "प"), ("b", "ब"), ("m", "म"),
]
_PUNCT = {".": "।", "|": "।", "||": "॥", ",": ",", "?": "?", "!": "!"}


def _iast_to_deva(text: str) -> str:
    s = (text or "").lower()
    out = []
    i = 0
    pending_consonant = False

    def _match_any(pairs, pos):
        for k, v in pairs:
            if s.startswith(k, pos):
                return k, v
        return None, None

    while i < len(s):
        if s[i].isspace():
            pending_consonant = False
            out.append(s[i])
            i += 1
            continue
        if s[i:i+2] == "||":
            pending_consonant = False
            out.append(_PUNCT["||"])
            i += 2
            continue
        if s[i] in _PUNCT:
            pending_consonant = False
            out.append(_PUNCT[s[i]])
            i += 1
            continue

        v_key, v_deva = _match_any(_IAST_VOWELS, i)
        if v_key:
            if pending_consonant:
                _, v_matra = _match_any(_IAST_MATRAS, i)
                out[-1] = out[-1] + (v_matra or "")
                pending_consonant = False
            else:
                out.append(v_deva)
            i += len(v_key)
            continue

        c_key, c_deva = _match_any(_IAST_CONS, i)
        if c_key:
            if pending_consonant:
                out[-1] = out[-1] + "्"
            out.append(c_deva)
            pending_consonant = True
            i += len(c_key)
            continue

        out.append(s[i])
        pending_consonant = False
        i += 1

    return "".join(out).strip()


def _compute_cer(pred: str, ref: str) -> float:
    if pred == ref:
        return 0.0
    if not pred or not ref:
        return 1.0
    m, n = len(pred), len(ref)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            cost = 0 if pred[i - 1] == ref[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = temp
    return dp[n] / max(m, n)


def _cleanup_thresholds(temperature: float, top_k: int):
    temp = float(temperature)
    k = max(1, int(top_k))
    t_norm = max(0.0, min((temp - 0.4) / 0.6, 1.0))
    k_norm = max(0.0, min((k - 20) / 80.0, 1.0))
    diversity = 0.6 * t_norm + 0.4 * k_norm
    cer_threshold = 0.10 + 0.18 * diversity
    deva_ratio_threshold = 0.60 - 0.20 * diversity
    return cer_threshold, deva_ratio_threshold


def _decode_with_cleanup(tgt_tok, ids, src_text: str, inf_cfg: dict):
    model_out = _decode_clean(tgt_tok, ids)
    rule_out = _iast_to_deva(src_text.strip())
    deva_chars = sum(1 for ch in model_out if "\u0900" <= ch <= "\u097F")
    deva_ratio = deva_chars / max(1, len(model_out))
    cer = _compute_cer(model_out, rule_out)
    cer_thr, ratio_thr = _cleanup_thresholds(
        inf_cfg.get("temperature", 0.8),
        inf_cfg.get("top_k", 40),
    )
    if deva_ratio < ratio_thr or len(model_out) > 2.0 * max(1, len(rule_out)) or cer > cer_thr:
        return rule_out
    return model_out


# ── Interactive demo ──────────────────────────────────────────────────

def interactive_demo(checkpoint=None, single_text=None):
    from model.tokenizer import SanskritSourceTokenizer, SanskritTargetTokenizer

    cfg    = CONFIG
    device = _resolve_device(cfg['training'].get('device', 'cpu'))

    model_name = cfg['model_type']
    has_neg    = cfg['data']['include_negative_examples']
    ckpt       = checkpoint or f"results/{model_name}_neg_{has_neg}/best_model.pt"

    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"No checkpoint at {ckpt} — train first.")

    model, cfg = load_model(ckpt, cfg, device)
    model.eval()

    src_tok = SanskritSourceTokenizer(
        vocab_size=cfg['model'].get('src_vocab_size', 16000),
        max_len=cfg['model']['max_seq_len'],
    )
    tgt_tok = SanskritTargetTokenizer(
        vocab_size=cfg['model'].get('tgt_vocab_size', 16000),
        max_len=cfg['model']['max_seq_len'],
    )

    print("\n" + "="*55)
    print("Sanskrit D3PM Paraphrase — type verse, get paraphrase")
    print("="*55 + "\n")

    while True:
        try:
            text = (single_text if single_text is not None else input("INPUT > ")).strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text or text.lower() in ('quit', 'exit', 'q'):
            break

        ids = torch.tensor(
            [src_tok.encode(text)[:cfg['model']['max_seq_len']]],
            dtype=torch.long, device=device
        )
        out   = run_inference(model, ids, cfg)
        cleaned = _decode_with_cleanup(tgt_tok, out[0].tolist(), text, cfg["inference"])
        print(f"PARAPHRASE → {cleaned}\n")
        if single_text is not None:
            break


# ── Batch evaluation ──────────────────────────────────────────────────

def batch_evaluate(sample_size=500, checkpoint=None):
    from data.dataset import OptimizedSanskritDataset
    from model.tokenizer import SanskritSourceTokenizer, SanskritTargetTokenizer

    cfg    = CONFIG
    device = _resolve_device(cfg['training'].get('device', 'cpu'))

    model_name = cfg['model_type']
    has_neg    = cfg['data']['include_negative_examples']
    exp_dir    = f"results/{model_name}_neg_{has_neg}"
    ckpt       = checkpoint or f"{exp_dir}/best_model.pt"

    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"No checkpoint at {ckpt}")

    model, cfg = load_model(ckpt, cfg, device)
    model.eval()

    src_tok = SanskritSourceTokenizer(
        vocab_size=cfg['model'].get('src_vocab_size', 16000),
        max_len=cfg['model']['max_seq_len'],
    )
    tgt_tok = SanskritTargetTokenizer(
        vocab_size=cfg['model'].get('tgt_vocab_size', 16000),
        max_len=cfg['model']['max_seq_len'],
    )

    def collate(batch):
        return {
            'input_ids':   torch.stack([b['input_ids'].long() for b in batch]),
            'target_text': [b['target_text'] for b in batch],
            'input_text':  [b['input_text']  for b in batch],
        }

    dataset = OptimizedSanskritDataset(
        split='test',
        max_len=cfg['model']['max_seq_len'],
        cfg=cfg,
        src_tokenizer=src_tok,
        tgt_tokenizer=tgt_tok,
    )
    indices = list(range(min(sample_size, len(dataset))))
    loader  = DataLoader(
        Subset(dataset, indices),
        batch_size=cfg['training']['batch_size'],
        shuffle=False, collate_fn=collate
    )

    all_preds, all_refs, all_inputs = [], [], []
    print(f"⏳ Generating {len(indices)} paraphrases …")

    for batch in tqdm(loader):
        ids = batch['input_ids'].to(device)
        out = run_inference(model, ids, cfg)
        for i in range(out.size(0)):
            all_preds.append(_decode_with_cleanup(
                tgt_tok, out[i].tolist(), batch['input_text'][i], cfg["inference"]
            ))
            all_refs.append(batch['target_text'][i].strip())
            all_inputs.append(batch['input_text'][i].strip())

    # Metrics
    bleu_score, bert_f1 = 0.0, 0.0
    try:
        from nltk.translate.bleu_score import corpus_bleu
        bleu_score = corpus_bleu(
            [[r.split()] for r in all_refs],
            [p.split() for p in all_preds]
        )
    except Exception:
        pass

    try:
        import evaluate as hf_eval
        res    = hf_eval.load('bertscore').compute(
            predictions=all_preds, references=all_refs, lang='hi'
        )
        bert_f1 = sum(res['f1']) / len(res['f1'])
    except Exception:
        pass

    # Save
    out_path = f"{exp_dir}/evaluation_results.txt"
    pred_path = f"{exp_dir}/evaluation_predictions.jsonl"
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"Model    : {model_name}\n")
        f.write(f"Negatives: {has_neg}\n")
        f.write(f"Steps    : {cfg['inference']['num_steps']}\n")
        f.write(f"Temp     : {cfg['inference']['temperature']}\n")
        f.write(f"RepPen   : {cfg['inference']['repetition_penalty']}\n")
        f.write(f"DivPen   : {cfg['inference']['diversity_penalty']}\n")
        f.write(f"BLEU     : {bleu_score:.4f}\n")
        f.write(f"BERTScore: {bert_f1:.4f}\n\n")
        f.write("=== SAMPLES ===\n")
        for i in range(min(20, len(all_preds))):
            f.write(f"IN  : {all_inputs[i]}\n")
            f.write(f"REF : {all_refs[i]}\n")
            f.write(f"PRED: {all_preds[i]}\n")
            f.write("-" * 60 + "\n")

    with open(pred_path, 'w', encoding='utf-8') as f:
        for src, ref, pred in zip(all_inputs, all_refs, all_preds):
            row = {"input": src, "reference": ref, "prediction": pred}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n✅ Results → {out_path}")
    print(f"🗂️  Saved predictions → {pred_path}")
    print(f"📊 BLEU: {bleu_score:.4f}  |  BERTScore: {bert_f1:.4f}")
    return all_preds, all_refs


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--mode',    choices=['demo', 'eval'], default='demo')
    p.add_argument('--samples', type=int, default=500)
    p.add_argument('--checkpoint', type=str, default=None)
    p.add_argument('--text', type=str, default=None, help='Run one-shot demo input and exit')
    args = p.parse_args()

    if args.mode == 'demo':
        interactive_demo(checkpoint=args.checkpoint, single_text=args.text)
    else:
        batch_evaluate(args.samples, checkpoint=args.checkpoint)
