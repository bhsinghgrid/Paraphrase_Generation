"""
inference.py — ROOT CAUSE FIXED
=================================
ROOT CAUSE of garbage output:
  The model's hint_gate (self-conditioning) was added to the code AFTER the
  checkpoint was trained. So checkpoint has NO hint_gate weights.
  load_model() initialises it randomly (identity-ish), but sigmoid(random_init)
  ≈ 0.5-0.8, meaning 50-80% of a random embedding is ADDED to the decoder
  input at every step → pure noise → garbage output.

FIX: run_inference() passes x0_hint=None at ALL steps.
  Self-conditioning is disabled since the gate was never trained.
  The model forward() already handles x0_hint=None correctly (skips the gate).

All other code is unchanged.
"""

import torch
import os, sys
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import CONFIG


def load_model(ckpt_path, base_cfg, device):
    import copy
    from model.sanskrit_model import SanskritModel

    cfg   = copy.deepcopy(base_cfg)
    state = torch.load(ckpt_path, map_location='cpu')

    ek = 'model.src_embed.token_emb.weight'
    if ek in state:
        vocab, d                   = state[ek].shape
        cfg['model']['vocab_size'] = vocab
        cfg['model']['d_model']    = d
        cfg['model']['d_ff']       = d * 4

    ids = {int(k.split('.')[2]) for k in state if k.startswith('model.encoder_blocks.')}
    if ids:
        cfg['model']['n_layers'] = max(ids) + 1

    pk = 'model.src_embed.pos_enc.pe'
    if pk in state:
        cfg['model']['max_seq_len'] = state[pk].shape[1]

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
    missing, unexpected = model.load_state_dict(
        torch.load(ckpt_path, map_location=device), strict=False
    )
    allowed = {'model.hint_gate.0.weight', 'model.hint_gate.0.bias'}
    real_missing = [k for k in missing if k not in allowed]
    if real_missing:
        print(f"⚠️  Missing keys: {real_missing[:3]} …")
    if unexpected:
        print(f"⚠️  Unexpected keys: {unexpected[:3]} …")

    hint_gate_missing = 'model.hint_gate.0.weight' in missing
    if hint_gate_missing:
        print("ℹ️  hint_gate NOT in checkpoint — self-conditioning DISABLED at inference.")
        # Zero out the gate bias so sigmoid(0)=0.5 but we won't use it anyway
        if hasattr(model.model, 'hint_gate'):
            with torch.no_grad():
                model.model.hint_gate[0].weight.zero_()
                model.model.hint_gate[0].bias.fill_(-10.0)  # sigmoid(-10) ≈ 0 → gate off

    print("✅ Model loaded.")
    return model, cfg, hint_gate_missing


def run_inference(model, input_ids, cfg, disable_hint=True):
    """
    Fixed inference loop.

    disable_hint=True (default): never pass x0_hint.
    This is correct when hint_gate was not in the checkpoint (i.e., was not trained).
    Passing a random-init hint gate corrupts decoder input at every step.
    """
    inf       = cfg['inference']
    device    = input_ids.device
    B, L      = input_ids.shape
    inner     = model.model
    T         = inner.scheduler.num_timesteps
    steps     = inf['num_steps']
    step_size = max(1, T // steps)
    timesteps = list(range(T - 1, -1, -step_size))
    if timesteps[-1] != 0:
        timesteps.append(0)

    mask_id = inner.mask_token_id
    x0_est  = torch.full((B, L), mask_id, dtype=torch.long, device=device)

    import torch.nn.functional as F
    model.eval()
    with torch.no_grad():
        for step_idx, t_val in enumerate(timesteps):
            t       = torch.full((B,), t_val, dtype=torch.long, device=device)
            is_last = (step_idx == len(timesteps) - 1)

            # KEY FIX: x0_hint=None always — hint_gate not in checkpoint
            logits, _ = model(input_ids, x0_est, t, x0_hint=None)

            # Temperature scaling
            logits = logits / max(inf['temperature'], 1e-5)

            # Top-k filtering
            if inf['top_k'] > 0:
                from model.d3pm_model_cross_attention import _top_k_filter
                logits = _top_k_filter(logits, inf['top_k'])

            probs = F.softmax(logits, dim=-1)

            if is_last:
                x0_est = torch.argmax(probs, dim=-1)
            else:
                from model.d3pm_model_cross_attention import _batch_multinomial
                x0_est = _batch_multinomial(probs)

    return x0_est


def _decode(tokenizer, ids, mask_id, pad_id):
    clean = [i for i in ids if isinstance(i, int) and i > 4
             and i not in (mask_id, pad_id)]
    if not clean:
        return ""
    return tokenizer.tokenizer.decode(clean, skip_special_tokens=True).strip()


def interactive_demo():
    from model.tokenizer import SanskritTokenizer

    cfg    = CONFIG
    device = torch.device(cfg['training']['device'])

    model_name = cfg['model_type']
    has_neg    = cfg['data']['include_negative_examples']
    ckpt       = f"results/{model_name}_neg_{has_neg}/best_model.pt"

    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"No checkpoint at {ckpt} — train first.")

    model, cfg, hint_missing = load_model(ckpt, cfg, device)
    model.eval()

    tokenizer = SanskritTokenizer(cfg['model']['vocab_size'])
    PAD_ID    = tokenizer.tokenizer.token_to_id('[PAD]') or 1
    MASK_ID   = cfg['diffusion']['mask_token_id']

    print("\n" + "="*60)
    print("Sanskrit D3PM Paraphrase")
    if hint_missing:
        print("ℹ️  Self-conditioning disabled (hint_gate not in checkpoint)")
    print("="*60 + "\n")

    while True:
        try:
            text = input("INPUT > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text or text.lower() in ('quit', 'exit', 'q'):
            break

        ids = torch.tensor(
            [tokenizer.encode(text)[:cfg['model']['max_seq_len']]],
            dtype=torch.long, device=device
        )
        out    = run_inference(model, ids, cfg, disable_hint=hint_missing)
        result = _decode(tokenizer, out[0].tolist(), MASK_ID, PAD_ID)
        print(f"PARAPHRASE → {result}\n")


def batch_evaluate(sample_size=500):
    from data.dataset import OptimizedSanskritDataset
    from model.tokenizer import SanskritTokenizer

    cfg    = CONFIG
    device = torch.device(cfg['training']['device'])

    model_name = cfg['model_type']
    has_neg    = cfg['data']['include_negative_examples']
    exp_dir    = f"results5/{model_name}_neg_{has_neg}"
    ckpt       = f"{exp_dir}/best_model.pt"

    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"No checkpoint at {ckpt}")

    model, cfg, hint_missing = load_model(ckpt, cfg, device)
    model.eval()

    tokenizer = SanskritTokenizer(cfg['model']['vocab_size'])
    PAD_ID    = tokenizer.tokenizer.token_to_id('[PAD]') or 1
    MASK_ID   = cfg['diffusion']['mask_token_id']

    def collate(batch):
        return {
            'input_ids':   torch.stack([b['input_ids'].long()  for b in batch]),
            'target_text': [b['target_text'] for b in batch],
            'input_text':  [b['input_text']  for b in batch],
        }

    dataset = OptimizedSanskritDataset('test', tokenizer, cfg['model']['max_seq_len'], cfg)
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
        out = run_inference(model, ids, cfg, disable_hint=hint_missing)
        for i in range(out.size(0)):
            pred = _decode(tokenizer, out[i].tolist(), MASK_ID, PAD_ID)
            all_preds.append(pred.strip())
            all_refs.append(batch['target_text'][i].strip())
            all_inputs.append(batch['input_text'][i].strip())

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
        res     = hf_eval.load('bertscore').compute(
            predictions=all_preds, references=all_refs, lang='hi'
        )
        bert_f1 = sum(res['f1']) / len(res['f1'])
    except Exception:
        pass

    out_path = f"{exp_dir}/evaluation_results.txt"
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"Model    : {model_name}\n")
        f.write(f"Negatives: {has_neg}\n")
        f.write(f"Steps    : {cfg['inference']['num_steps']}\n")
        f.write(f"Temp     : {cfg['inference']['temperature']}\n")
        f.write(f"RepPen   : disabled\n")
        f.write(f"DivPen   : disabled\n")
        f.write(f"HintGate : {'disabled (not in ckpt)' if hint_missing else 'enabled'}\n")
        f.write(f"BLEU     : {bleu_score:.4f}\n")
        f.write(f"BERTScore: {bert_f1:.4f}\n\n")
        f.write("=== SAMPLES ===\n")
        for i in range(min(20, len(all_preds))):
            f.write(f"IN  : {all_inputs[i]}\n")
            f.write(f"REF : {all_refs[i]}\n")
            f.write(f"PRED: {all_preds[i]}\n")
            f.write("-" * 60 + "\n")

    print(f"\n✅ Results → {out_path}")
    print(f"📊 BLEU: {bleu_score:.4f}  |  BERTScore: {bert_f1:.4f}")
    return all_preds, all_refs


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--mode',    choices=['demo', 'eval'], default='demo')
    p.add_argument('--samples', type=int, default=500)
    args = p.parse_args()
    if args.mode == 'demo':
        interactive_demo()
    else:
        batch_evaluate(args.samples)