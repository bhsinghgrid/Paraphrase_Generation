# Paraphrase Generation — Sanskrit IAST → Devanagari Transliteration

> **Language:** Python (98.2%), Shell (1.8%)

---

## 📌 Executive Summary

This project addresses the challenge of **automatically converting Sanskrit text written in IAST (International Alphabet of Sanskrit Transliteration) format into accurate Devanagari script** using deep learning. It provides an end-to-end pipeline — from data preparation to GPU-accelerated training and production-ready inference — making it suitable for large-scale automated script conversion tasks.

---

## 🔴 Problem

Sanskrit texts are widely represented in the **IAST romanization** standard across digital archives, academic publications, and linguistic databases. However, most Sanskrit scholars, readers, and downstream NLP tools require the native **Devanagari script**. Manual transliteration is:

- Time-consuming and error-prone at scale
- Inconsistent across different human translators
- A bottleneck for digitization and accessibility of Sanskrit literature

---

## ✅ Solution

This repository implements a **sequence-to-sequence deep learning model** (leveraging diffusion-based architecture) that learns the character-level mapping from IAST to Devanagari. Key highlights:

| Feature | Detail |
|---|---|
| **Architecture** | Sequence-to-Sequence with Diffusion module |
| **Training** | GPU-accelerated, scalable to large corpora |
| **Inference** | Production-ready, batch and single-sample support |
| **Script** | Python with Shell utilities |

The model is structured across four clean modules:

- **`data/`** — Data loading, preprocessing, and formatting pipelines
- **`diffusion/`** — Diffusion-based generation/transformation logic
- **`model/`** — Core neural network architecture and training scripts
- **`inference/`** — Ready-to-use inference scripts for deployment
- **`sample/`** — Example inputs/outputs to validate model behavior

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/bhsinghgrid/Paraphrase_Generation.git
cd Paraphrase_Generation
git checkout olb_branch
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **GPU recommended.** Ensure CUDA is properly configured for training.

### 3. Prepare Your Data

Place your IAST-format Sanskrit text files into the `data/` directory following the expected format (refer to `sample/` for examples).

### 4. Train the Model

```bash
python model/train.py
```

### 5. Run Inference

```bash
python inference/infer.py --input "your IAST text here"
```

---

## 📁 Repository Structure

```
Paraphrase_Generation/
├── data/           # Data loading and preprocessing
├── diffusion/      # Diffusion model components
├── model/          # Model architecture and training
├── inference/      # Inference pipeline
├── sample/         # Sample inputs and expected outputs
├── requirements.txt
└── .gitignore
```

---

## 🛠️ Requirements

All Python dependencies are listed in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

---

## 🤝 Contributing

Contributions are welcome! Please fork the repository, make your changes on a new branch, and open a pull request against `olb_branch`.

---

## 📬 Contact

For questions or issues, open a GitHub Issue in this repository.
