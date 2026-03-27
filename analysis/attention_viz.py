import torch
import numpy as np
from typing import Dict, List
from collections import defaultdict
import re

# Optional metrics
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import evaluate
    bertscore = evaluate.load("bertscore")
    USE_BERT = True
except:
    USE_BERT = False


# ─────────────────────────────────────────────────────────────
# 1. ATTENTION CAPTURE (FIXED VERSION)
# ─────────────────────────────────────────────────────────────

class AttentionCapture:
    def __init__(self, model):
        self.model = model
        self.inner = model.model
        self.cross_attns = []

        for block in self.inner.decoder_blocks:
            if hasattr(block, "cross_attn"):
                self.cross_attns.append(block.cross_attn)

    def _enable(self):
        for ca in self.cross_attns:
            ca.capture_weights = True

    def _disable(self):
        for ca in self.cross_attns:
            ca.capture_weights = False
            ca.last_attn_weights = None

    def _read(self):
        weights = []
        for ca in self.cross_attns:
            if ca.last_attn_weights is not None:
                w = ca.last_attn_weights.mean(dim=1)  # avg heads
                weights.append(w.cpu().numpy())
        return weights

    @torch.no_grad()
    def run(self, src_ids):
        inner = self.inner
        T = inner.scheduler.num_timesteps
        device = src_ids.device

        memory, mask = inner.encode_source(src_ids)

        x = torch.full(
            (1, inner.max_seq_len),
            inner.mask_token_id,
            dtype=torch.long,
            device=device
        )

        hint = None
        step_weights = {}
        step_outputs = {}

        self._enable()

        try:
            for t_val in range(T - 1, -1, -1):
                t = torch.tensor([t_val], device=device)

                logits, _ = inner.forward_cached(
                    memory, mask, x, t, x0_hint=hint, inference_mode=True
                )

                probs = torch.softmax(logits, dim=-1)
                x = torch.argmax(probs, dim=-1)

                step_weights[t_val] = self._read()
                step_outputs[t_val] = x.clone()

                hint = x

        finally:
            self._disable()

        return step_weights, step_outputs


# ─────────────────────────────────────────────────────────────
# 2. BERTScore + Semantic Drift
# ─────────────────────────────────────────────────────────────

def compute_trajectory_metrics(
    step_outputs,
    tgt_tokenizer,
    reference_text
):
    trajectory = []

    for t, ids in step_outputs.items():
        text = tgt_tokenizer.decode(
            [x for x in ids[0].tolist() if x > 4]
        )

        if USE_BERT:
            score = bertscore.compute(
                predictions=[text],
                references=[reference_text],
                lang="hi"
            )["f1"][0]
        else:
            score = 0.0

        drift = 1.0 - score

        trajectory.append({
            "step": t,
            "text": text,
            "bert": score,
            "drift": drift
        })

    return sorted(trajectory, key=lambda x: -x["step"])


# ─────────────────────────────────────────────────────────────
# 3. LOCKED vs FLEXIBLE TOKENS
# ─────────────────────────────────────────────────────────────

def analyze_token_stability(step_weights):
    """
    Measure variance of attention over time
    """
    token_stability = defaultdict(list)

    for t, layers in step_weights.items():
        last_layer = layers[-1][0]  # [Lq, Lk]

        # max attention source index per target token
        align = np.argmax(last_layer, axis=1)

        for tgt_idx, src_idx in enumerate(align):
            token_stability[tgt_idx].append(src_idx)

    results = {}

    for tgt_idx, src_seq in token_stability.items():
        changes = sum(
            1 for i in range(1, len(src_seq))
            if src_seq[i] != src_seq[i-1]
        )

        if changes <= 2:
            results[tgt_idx] = "LOCKED"
        else:
            results[tgt_idx] = "FLEXIBLE"

    return results


# ─────────────────────────────────────────────────────────────
# 4. TF-IDF vs ATTENTION STABILITY
# ─────────────────────────────────────────────────────────────

_IAST_STOPWORDS = {
    "ca", "eva", "api", "tu", "va", "na", "hi", "atha", "iti",
    "vai", "ha", "khalu", "sma", "yadi", "tatah", "atra", "iva",
    "itiha", "idam", "tat", "etat", "sa", "sah", "te", "tam"
}


def _normalize_iast_token(token: str) -> str:
    t = token.lower().strip()
    t = t.replace("’", "'").replace("`", "'")
    return "".join(ch for ch in t if (ch.isalpha() or ch in ("'", "-")))


def _tokenize_iast(text: str) -> List[str]:
    raw = re.split(r"\s+", (text or "").strip())
    toks = []
    for tok in raw:
        t = _normalize_iast_token(tok)
        if len(t) >= 2 and t not in _IAST_STOPWORDS:
            toks.append(t)
    return toks


def tfidf_attention_correlation(src_text, step_weights, corpus_texts=None):
    """
    Compute TF-IDF/attention correlation using corpus-level TF-IDF.
    Returns a dict so callers can plot and handle undefined cases safely.
    """
    src_tokens = _tokenize_iast(src_text)
    if not src_tokens:
        return {
            "corr": None,
            "status": "NA_NO_SOURCE_TOKENS",
            "tokens": [],
            "tfidf_scores": np.array([], dtype=np.float32),
            "attn_scores": np.array([], dtype=np.float32),
        }

    docs = []
    src_doc = " ".join(src_tokens)
    docs.append(src_doc)
    if corpus_texts:
        for txt in corpus_texts:
            toks = _tokenize_iast(txt)
            if toks:
                docs.append(" ".join(toks))

    # Deduplicate while preserving order to avoid overweighting repeated docs.
    docs = list(dict.fromkeys(docs))
    if len(docs) < 2:
        docs = [src_doc, src_doc + " semantic context"]

    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        norm="l2",
    )
    mat = vectorizer.fit_transform(docs)
    terms = vectorizer.get_feature_names_out()
    src_vec = mat[0].toarray()[0]
    term_to_tfidf = {terms[i]: float(src_vec[i]) for i in range(len(terms))}
    tfidf_by_pos = np.array([term_to_tfidf.get(tok, 0.0) for tok in src_tokens], dtype=np.float32)

    # Average attention over captured steps, last layer.
    attn_scores = None
    for _, layers in step_weights.items():
        w = layers[-1][0]
        avg = w.mean(axis=0).astype(np.float32)
        if attn_scores is None:
            attn_scores = avg
        else:
            attn_scores += avg
    if attn_scores is None:
        attn_scores = np.zeros((0,), dtype=np.float32)
    else:
        attn_scores /= max(len(step_weights), 1)

    min_len = min(len(tfidf_by_pos), len(attn_scores))
    if min_len < 2:
        return {
            "corr": None,
            "status": "NA_INSUFFICIENT_LENGTH",
            "tokens": src_tokens[:min_len],
            "tfidf_scores": tfidf_by_pos[:min_len],
            "attn_scores": attn_scores[:min_len],
        }

    x = tfidf_by_pos[:min_len]
    y = attn_scores[:min_len]
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return {
            "corr": None,
            "status": "NA_ZERO_VARIANCE",
            "tokens": src_tokens[:min_len],
            "tfidf_scores": x,
            "attn_scores": y,
        }

    corr = float(np.corrcoef(x, y)[0, 1])
    if np.isnan(corr):
        return {
            "corr": None,
            "status": "NA_NUMERIC",
            "tokens": src_tokens[:min_len],
            "tfidf_scores": x,
            "attn_scores": y,
        }

    return {
        "corr": corr,
        "status": "OK",
        "tokens": src_tokens[:min_len],
        "tfidf_scores": x,
        "attn_scores": y,
    }


# ─────────────────────────────────────────────────────────────
# 5. FULL PIPELINE
# ─────────────────────────────────────────────────────────────

def run_task2_analysis(
    text,
    model,
    src_tokenizer,
    tgt_tokenizer,
    device
):
    src_ids = torch.tensor(
        [src_tokenizer.encode(text)],
        device=device
    )

    capturer = AttentionCapture(model)

    # Step 1: Capture
    step_weights, step_outputs = capturer.run(src_ids)

    # Step 2: Metrics
    trajectory = compute_trajectory_metrics(
        step_outputs,
        tgt_tokenizer,
        reference_text=text   # transliteration task
    )

    # Step 3: Token stability
    stability = analyze_token_stability(step_weights)

    # Step 4: TF-IDF correlation
    corr_obj = tfidf_attention_correlation(text, step_weights)

    return {
        "trajectory": trajectory,
        "token_stability": stability,
        "tfidf_corr": corr_obj.get("corr"),
        "tfidf_status": corr_obj.get("status"),
    }
