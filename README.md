# 🕉️ Sanskrit Neural Translation System

### Transformer & D3PM-based Encoder–Decoder Architectures

---

## 1️⃣ Problem Fit & Scope (5 Points)

### 🎯 Problem Statement

Sanskrit is a morphologically rich, low-resource classical language. Existing neural translation and text generation systems struggle due to:

* Complex sandhi transformations
* Free word order
* Rich inflectional morphology
* Limited parallel corpora

This project builds a **robust Sanskrit sequence-to-sequence generation system** using:

* ✅ Baseline Transformer (Encoder–Decoder)
* ✅ Cross-Attention Transformer
* ✅ D3PM (Diffusion-based Discrete Probabilistic Model)
* ✅ D3PM with Cross-Attention

### 📌 Scope of the Project

This system:

* Translates or reconstructs Sanskrit sequences
* Compares baseline vs diffusion architectures
* Evaluates generation quality using modern NLP metrics
* Supports structured experimentation
* Enables demo-based inference on new user inputs

Out of scope:

* Large-scale multilingual translation
* Production-scale deployment
* Extremely large foundation models

---

## 2️⃣ Data Acquisition & Quality (8 Points)

### 📦 Dataset Source

The project uses:

`OptimizedSanskritDataset`

Dataset includes:

* Input transliterated Sanskrit
* Target Devanagari Sanskrit
* Train / Validation / Test splits

### 📊 Dataset Properties

* Max sequence length: 80
* Vocabulary size: 16,000
* Dataset size used in experiments: 50,000 samples
* Test set evaluation: 5,000 samples

### 🧹 Preprocessing Pipeline

* Custom `SanskritTokenizer`
* Padding & masking
* Controlled truncation
* Mask token handling
* Collate function batching

### ✅ Data Quality Measures

* Train/Val split: 80/20
* Mask token ignored in loss
* Label smoothing applied
* Controlled dataset size for stability

---

## 3️⃣ Baseline & Experiments (8 Points)

We implemented and compared 4 architectures:

| Model Type                 | Description                 |
| -------------------------- | --------------------------- |
| `baseline_encoder_decoder` | Vanilla Transformer         |
| `baseline_cross_attention` | Cross-attention Transformer |
| `d3pm_encoder_decoder`     | Diffusion Transformer       |
| `d3pm_cross_attention`     | Diffusion + Cross Attention |

### 🧪 Experimental Setup

* Same dataset
* Same tokenizer
* Same hyperparameters
* Same optimizer
* Same scheduler
* Same early stopping

This ensures fair comparison.

### ⚙️ Hyperparameters

* d_model: 384
* Layers: 6
* Heads: 8
* FFN dim: 1536
* Dropout: 0.1
* LR: 2e-4
* Diffusion steps: 8

---

## 4️⃣ Training Correctness & Efficiency (7 Points)

### 🔁 Training Strategy

* Cross Entropy Loss
* Label smoothing (0.05)
* Gradient clipping (0.5)
* AdamW optimizer
* OneCycleLR scheduler
* Early stopping (patience = 3)

### 🧠 Diffusion Handling

For D3PM models:

* Random timestep sampling
* Mask-based corruption
* Model predicts clean tokens

Baseline models:

* Standard teacher forcing

### ⚡ Efficiency Measures

* MPS device support (Mac GPU)
* Mixed precision option
* Subset dataset for controlled training
* Gradient clipping prevents explosion

---

## 5️⃣ Evaluation & Metrics (7 Points)

Evaluation performed on:

**5,000 Test Samples**

Metrics:

### 📉 Test Loss

Cross-entropy on test set.

### 🎯 Token-Level Accuracy

Correct tokens / total tokens.

### 🔁 Precision & Recall

Token-level evaluation.

### 🧠 BERTScore (Semantic Similarity)

Using multilingual BERT (Hindi mode for Sanskrit similarity).

### 📊 JSON Output Example

```json
{
    "model_type": "d3pm_cross_attention",
    "test_size": 5000,
    "test_loss": 0.8421,
    "accuracy": 0.9134,
    "precision": 0.9134,
    "recall": 0.9134,
    "bert_f1": 0.8945
}
```

---

## 6️⃣ Minimal UI / Demo (5 Points)

We provide a simple inference demo using **Gradio**.

### 🎛️ Features

* Input Sanskrit transliteration
* Model selection dropdown
* Generated output
* Diffusion or baseline auto-handled

### ▶ Run Demo

```bash
python app.py
```

### Example Interface

Input:

```
udelaḍi
```

Output:

```
उदेलडि ॥
```

This allows new data inference without retraining.

---

## 7️⃣ Success Criteria & Insights (5 Points)

### ✔ Success Criteria

* Stable training without divergence
* Early stopping triggers correctly
* Diffusion models converge
* BERTScore > baseline
* Cross-attention improves alignment

### 🔍 Key Insights

1. Cross-attention improves structural alignment.
2. Diffusion models provide better robustness to noise.
3. Early stopping prevents overfitting.
4. Token-level accuracy alone is insufficient — BERTScore adds semantic insight.
5. Diffusion models require careful timestep sampling.

---

## 8️⃣ Reproducibility & Documentation (5 Points)

### 📂 Project Structure

```
Sanskrit_Translator/
│
├── data/
│   └── dataset.py
│
├── model/
│   ├── tokenizer.py
│   ├── transformer.py
│   ├── diffusion.py
│   └── d3pm_model.py
│
├── production_model3/
│   └── best_*.pt
│
├── results/
│   ├── training.log
│   └── test_metrics.json
│
├── train.py
├── test_model.py
├── app.py
└── README.md
```

### 🔁 How to Reproduce

1. Install requirements:

```bash
pip install -r requirements.txt
```

2. Train model:

```bash
python train.py
```

3. Evaluate:

```bash
python test_model.py
```

4. Run demo:

```bash
python app.py
```

### 🔒 Determinism

* Fixed random seed
* Fixed dataset split
* Same hyperparameters across experiments

---

# 🙏 Final Note

This project demonstrates:

* Modern transformer architectures
* Diffusion-based discrete modeling
* Controlled experimentation
* Reproducible training
* Research-style evaluation
* Real-world deployable demo

---

### Author

Bhanu Pratap Singh

---
