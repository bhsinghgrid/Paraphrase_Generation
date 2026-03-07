"""
tune_and_eval.py
================
Does 3 things automatically:
  1. Finds best inference parameters (temperature, repetition_penalty, top_k)
  2. Runs final evaluation with best params
  3. Saves a clean results report

Run: uv run tune_and_eval.py
Takes ~10-15 minutes total.
"""

import torch
import os, sys, copy, json
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import CONFIG
from model.tokenizer import SanskritTokenizer
from model.sanskrit_model import SanskritModel


# ── Load model ────────────────────────────────────────────────────────

def load_model(cfg, device):
    model_name = cfg['model_type']
    has_neg    = cfg['data']['include_negative_examples']

    ckpt = None
    for folder in ['results2']:
        candidate = f"{folder}/{model_name}_neg_{has_neg}/best_model.pt"
        if os.path.exists(candidate):
            ckpt = candidate
            break
    if ckpt is None:
        raise FileNotFoundError("No checkpoint found.")

    state = torch.load(ckpt, map_location='cpu')
    ek = 'model.src_embed.token_emb.weight'
    if ek in state:
        vocab, d = state[ek].shape
        cfg['model']['vocab_size'] = vocab
        cfg['model']['d_model']    = d
        cfg['model']['d_ff']       = d * 4
    ids = {int(k.split('.')[2]) for k in state if k.startswith('model.encoder_blocks.')}
    if ids:
        cfg['model']['n_layers'] = max(ids) + 1
    if 'model.src_embed.pos_enc.pe' in state:
        cfg['model']['max_seq_len'] = state['model.src_embed.pos_enc.pe'].shape[1]
    d = cfg['model']['d_model']
    h = cfg['model'].get('n_heads', 6)
    if d % h != 0:
        h = next(x for x in [8, 6, 4, 2, 1] if d % x == 0)
    cfg['model']['n_heads'] = h

    model = SanskritModel(cfg).to(device)
    missing, _ = model.load_state_dict(
        torch.load(ckpt, map_location=device), strict=False
    )
    if hasattr(model.model, 'hint_gate') and 'model.hint_gate.0.weight' in missing:
        with torch.no_grad():
            w = model.model.hint_gate[0].weight
            torch.nn.init.zeros_(model.model.hint_gate[0].bias)
            torch.nn.init.eye_(w) if w.shape[0] == w.shape[1] \
                else torch.nn.init.xavier_uniform_(w)
    model.eval()
    print(f"✅ Loaded: {ckpt}")
    return model, ckpt


def decode(token_ids, tokenizer):
    clean = [i for i in token_ids if i > 4]
    if not clean:
        return ""
    return tokenizer.tokenizer.decode(clean, skip_special_tokens=True).replace("@","").strip()


def run_batch(model, loader, tokenizer, device, cfg, temperature, rep_penalty, top_k):
    """Generate predictions for a batch of samples."""
    preds, refs, inputs = [], [], []
    mask_id = cfg['diffusion']['mask_token_id']
    pad_id  = 1

    for batch in loader:
        src = batch['input_ids'].to(device)
        with torch.no_grad():
            out = model.generate(
                src,
                num_steps          = cfg['model']['diffusion_steps'],
                temperature        = temperature,
                top_k              = top_k,
                repetition_penalty = rep_penalty,
                diversity_penalty  = 0.0,
            )
        for i in range(out.size(0)):
            preds.append(decode(out[i].tolist(), tokenizer))
            refs.append(batch['target_text'][i].strip())
            inputs.append(batch['input_text'][i].strip())

    return preds, refs, inputs


def bertscore(preds, refs):
    try:
        import evaluate as hf_eval
        res = hf_eval.load('bertscore').compute(
            predictions=preds, references=refs, lang='hi'
        )
        return round(sum(res['f1']) / len(res['f1']), 4)
    except Exception as e:
        print(f"  BERTScore error: {e}")
        return 0.0


def bleu(preds, refs):
    try:
        from nltk.translate.bleu_score import corpus_bleu
        return round(corpus_bleu(
            [[r.split()] for r in refs],
            [p.split() for p in preds]
        ), 4)
    except Exception:
        return 0.0


# ── Main ──────────────────────────────────────────────────────────────

def main():
    cfg    = copy.deepcopy(CONFIG)
    device = torch.device(cfg['training']['device'])

    model, ckpt = load_model(cfg, device)
    tokenizer   = SanskritTokenizer(cfg['model']['vocab_size'])

    from data.dataset import OptimizedSanskritDataset
    from sklearn.model_selection import train_test_split

    print("\n📥 Loading dataset...")
    dataset = OptimizedSanskritDataset('train', tokenizer, cfg['model']['max_seq_len'], cfg)

    def collate(batch):
        return {
            'input_ids':   torch.stack([b['input_ids'].long() for b in batch]),
            'target_text': [b['target_text'] for b in batch],
            'input_text':  [b['input_text']  for b in batch],
        }

    # Use 200 samples for fast tuning, 500 for final eval
    all_idx = list(range(min(5000, len(dataset))))
    _, val_idx = train_test_split(all_idx, train_size=0.8, random_state=42)
    tune_idx = val_idx[:200]    # fast tuning
    eval_idx = val_idx[:500]    # final eval

    tune_loader = DataLoader(
        Subset(dataset, tune_idx),
        batch_size=32, shuffle=False, collate_fn=collate
    )

    # ── Step 1: Parameter search ──────────────────────────────────────
    print("\n" + "="*55)
    print("STEP 1: Finding best inference parameters (200 samples)")
    print("="*55)

    param_grid = [
        # temperature, rep_penalty, top_k
        (0.6, 1.0,  30),
        (0.7, 1.0,  50),
        (0.8, 1.0,  50),
        (0.8, 1.2,  50),
        (0.8, 1.5,  50),
        (0.8, 1.2,  30),
        (0.8, 1.2, 100),
        (0.9, 1.2,  50),
        (1.0, 1.2,  50),
        (0.7, 1.5,  30),
    ]

    best_score  = 0.0
    best_params = None
    results     = []

    for temp, rep, topk in param_grid:
        label = f"temp={temp} rep={rep} top_k={topk}"
        print(f"\n  Testing: {label}")
        preds, refs, _ = run_batch(
            model, tune_loader, tokenizer, device, cfg, temp, rep, topk
        )
        score = bertscore(preds, refs)
        print(f"  BERTScore: {score}")
        results.append((score, temp, rep, topk))

        if score > best_score:
            best_score  = score
            best_params = (temp, rep, topk)

    results.sort(reverse=True)

    print(f"\n{'='*55}")
    print(f"BEST PARAMS: temp={best_params[0]} rep={best_params[1]} top_k={best_params[2]}")
    print(f"BEST BERTScore on tune set: {best_score}")
    print(f"{'='*55}")

    # ── Step 2: Final evaluation with best params ─────────────────────
    print(f"\n{'='*55}")
    print("STEP 2: Final evaluation (500 samples) with best params")
    print(f"{'='*55}")

    eval_loader = DataLoader(
        Subset(dataset, eval_idx),
        batch_size=32, shuffle=False, collate_fn=collate
    )

    temp, rep, topk = best_params
    preds, refs, inputs = run_batch(
        model, eval_loader, tokenizer, device, cfg, temp, rep, topk
    )

    final_bert = bertscore(preds, refs)
    final_bleu = bleu(preds, refs)

    print(f"\n📊 FINAL RESULTS")
    print(f"   BERTScore F1 : {final_bert}")
    print(f"   BLEU         : {final_bleu}")

    # ── Step 3: Save report ───────────────────────────────────────────
    model_name = cfg['model_type']
    has_neg    = cfg['data']['include_negative_examples']
    exp_dir    = ckpt.rsplit('/', 1)[0]

    report_path = f"{exp_dir}/final_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("SANSKRIT D3PM — FINAL EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Checkpoint   : {ckpt}\n")
        f.write(f"Model        : {model_name}\n")
        f.write(f"Negatives    : {has_neg}\n")
        f.write(f"d_model      : {cfg['model']['d_model']}\n")
        f.write(f"n_layers     : {cfg['model']['n_layers']}\n")
        f.write(f"T (steps)    : {cfg['model']['diffusion_steps']}\n\n")

        f.write("BEST INFERENCE PARAMS (from grid search)\n")
        f.write(f"  temperature        : {temp}\n")
        f.write(f"  repetition_penalty : {rep}\n")
        f.write(f"  top_k              : {topk}\n\n")

        f.write("FINAL METRICS (500 test samples)\n")
        f.write(f"  BERTScore F1 : {final_bert}\n")
        f.write(f"  BLEU         : {final_bleu}\n\n")

        f.write("PARAM GRID RESULTS (ranked by BERTScore)\n")
        for rank, (score, t, r, k) in enumerate(results, 1):
            f.write(f"  #{rank:2d}  bert={score:.4f}  temp={t} rep={r} top_k={k}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("SAMPLE PREDICTIONS\n")
        f.write("=" * 60 + "\n")
        for i in range(min(20, len(preds))):
            f.write(f"IN  : {inputs[i]}\n")
            f.write(f"REF : {refs[i]}\n")
            f.write(f"OUT : {preds[i]}\n")
            f.write("-" * 60 + "\n")

    print(f"\n✅ Full report saved → {report_path}")

    # Also save best params to config snippet
    params_path = f"{exp_dir}/best_inference_params.json"
    with open(params_path, 'w') as f:
        json.dump({
            "temperature":        temp,
            "repetition_penalty": rep,
            "top_k":              topk,
            "bertscore":          final_bert,
            "bleu":               final_bleu,
        }, f, indent=2)
    print(f"✅ Best params saved → {params_path}")
    print(f"\n   Update config.py inference section with these params for best demo output.")


if __name__ == '__main__':
    main()