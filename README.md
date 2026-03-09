---

# 🕉️ Sanskrit AI Paraphrasing Engine (D3PM)

Welcome to the **Sanskrit Paraphrase Lab**. This project is a state-of-the-art text-to-text generation tool designed specifically for the Sanskrit language. It provides researchers, linguists, and data scientists with an interactive environment to test custom-trained AI models, generate paraphrases, and automatically log metadata for accuracy grading (e.g., BERTScore).
<img width="1378" height="679" alt="image" src="https://github.com/user-attachments/assets/5fe68b86-f405-4a74-8e31-a6e5e22b1f0d" />


---

## 🧠 The Science: How it Works (A Deeper Dive)

Unlike standard AI models (like ChatGPT) that guess words one by one from left to right, this project uses a cutting-edge architecture called **Discrete Denoising Diffusion Probabilistic Models (D3PM)** with Cross-Attention.

1. **The Starting Point:** When you input a Sanskrit verse, the AI doesn't translate it directly. Instead, it creates a blank sequence of `[MASK]` tokens (digital "noise").
2. **The Denoising Process:** Over a series of mathematical steps (Diffusion Steps), the AI iteratively refines this noise. It uses **Cross-Attention** to constantly look back at your original verse, ensuring the meaning remains intact.
3. **Self-Conditioning:** At each step, the model feeds its previous guess back into itself as a "hint." This forces the AI to maintain strict grammatical consistency — a crucial requirement for a highly inflected language like Sanskrit.
4. **The Penalties:** We implemented custom mathematical penalties (Diversity and Repetition) into the beam search. This gives the user granular control to push the AI toward finding rare synonyms or to keep it strictly literal.

---

## 📂 Understanding the Project Structure

```
Paraphrase_Generation/
│
├── app2.py                        # ✅ Main web app — run this to launch the UI
├── train.py                       # Model training script (called by .sh files)
├── inference.py                   # Standalone inference (command-line, no UI)
├── inference1.py                  # Alternate inference variant for testing
├── evaluate_test.py               # Evaluation script — computes BERTScore & metrics
├── main.py                        # Orchestrator / experiment runner
├── config.py                      # Central config — edit hyperparameters here
│
├── model/
│   ├── sanskrit_model.py          # Core D3PM model architecture
│   └── tokenizer.py               # Custom 16,000-word Sanskrit tokenizer
│
├── diffusion/
│   └── reverse_process.py         # Denoising math, beam search & generation penalties
│
├── data/                          # Training data (Sanskrit verse pairs)
│
├── results/                       # ⚠️ Created after training — stores .pt model weights
│                                  #    e.g., d3pm_cross_attention_neg_False.pt
│
├── backend_generation_log.csv     # Master log — every generation is saved here
├── sanskrit_results.csv           # Sample evaluation results
├── sanskrit_tokenizer_m4pro.json  # Tokenizer vocab file (used by model/)
│
├── requirements.txt               # Python dependencies
├── pyproject.toml                 # Project metadata (built with uv)
├── uv.lock                        # Locked dependency versions
├── .python-version                # Pins Python version (3.11 recommended)
│
├── run_all_experiments.sh         # Runs ALL training configurations sequentially
├── run_cross_att_no_neg.sh        # Train: Cross-Attention, no diversity penalty
├── run_cross_attn_with_neg.sh     # Train: Cross-Attention, with diversity penalty
├── run_enc_dec_no_neg.sh          # Train: Encoder-Decoder, no diversity penalty
├── run_enc_dec_with_neg.sh        # Train: Encoder-Decoder, with diversity penalty
├── run_base_cross_att_no_neg.sh   # Train: Base + Cross-Attention, no penalty
├── run_base_cross_att_with_neg.sh # Train: Base + Cross-Attention, with penalty
├── run_base_enc_dec_no_neg.sh     # Train: Base + Enc-Dec, no penalty
└── run_base_enc_dec_with_neg.sh   # Train: Base + Enc-Dec, with penalty
```

---

## 📥 How to Access & Run the Entire Project

You do not need to be a software developer to run this tool on your computer. Follow these steps to get the Paraphrase Lab running locally.

> **⚠️ Requirements:**
> - **Python 3.11** (check with `python --version` in your terminal)
> - A machine with a **GPU is strongly recommended** for training. CPU-only machines can run inference but training will be extremely slow.

---

### Step 1: Download the Project

1. Navigate to the top of this GitHub repository.
2. Click the green **`<> Code`** button.
3. Click **"Download ZIP"** and extract the folder to your computer (e.g., your Desktop).

---

### Step 2: Set Up the Environment

To ensure the AI runs smoothly without affecting your computer's other settings, we will create a **Virtual Environment**.

1. Open your computer's **Terminal** (Mac/Linux) or **Command Prompt** (Windows).
2. Navigate into the downloaded project folder:

```bash
cd /path/to/Paraphrase_Generation
```

> **Tip (Mac):** Type `cd ` (with a space), drag the unzipped folder into the Terminal, then press Enter.

3. Create a virtual environment:

```bash
python -m venv .venv
```

4. Activate it:

| Platform | Command |
|---|---|
| Mac / Linux | `source .venv/bin/activate` |
| Windows | `.venv\Scripts\activate` |

5. Install all required libraries:

```bash
pip install -r requirements.txt
```

---

### Step 3: (Optional) Edit Configuration

Before training, you may want to review `config.py` to adjust hyperparameters such as:
- Number of diffusion timesteps
- Batch size
- Learning rate
- Maximum sequence length

Open it in any text editor. The default values work well for a first run.

---

### Step 4: Train the Models (Run `.sh` Scripts)

Before launching the web interface, you need to train the models. Each `.sh` script trains a different model configuration. The trained weights are saved to a `results/` folder that is automatically created.

> **⚠️ Warning:** Training from scratch is computationally intensive. A single configuration can take **several hours** on a GPU. CPU training will be significantly slower. Plan accordingly.

Run a single configuration (recommended to start):

```bash
bash run_cross_att_no_neg.sh
```

Or run **all configurations** back-to-back (will take a long time):

```bash
bash run_all_experiments.sh
```

Once complete, trained weights (`.pt` files) will appear in the `results/` folder. For example:

```
results/
└── d3pm_cross_attention_neg_False/
    └── best_model.pt
```

---

### Step 5: Launch the Paraphrase Lab (Web App)

With models trained and saved in `results/`, launch the interactive interface:

```bash
python app2.py
```

After a few seconds, a local link will appear in the terminal — usually:

```
http://127.0.0.1:7860
```

**Click that link** to open the Paraphrase Lab in your web browser.

---

## 🖥️ How to Use the Paraphrase Lab Interface

Once the app is open, you will see a clean, interactive dashboard:

1. **Select Experiment:** Use the dropdown at the top left to pick a trained model (e.g., `d3pm_cross_attention_neg_False`).

2. **Load Model (Crucial!):** Click **"🔄 Load Model"** and wait until the status shows ✅ *Loaded*.

3. **Choose a Strategy:**
   - **Manual Tuning** — Unlock the Advanced Controls accordion to adjust Temperature, Beam Width, and Penalties yourself.
   - **High Diversity** — Auto-sets sliders to encourage creative, synonym-rich paraphrases.
   - **Low Diversity** — Auto-sets sliders to keep output strict and literal.

4. **Generate:** Paste your source Sanskrit verse into the text box and click **"✨ Paraphrase"**.

5. **Save Your Work:**
   - Click **"Download Current Session"** to export results from this session.
   - All generations are permanently saved to `backend_generation_log.csv` automatically — nothing is ever lost.

---

## 📊 Evaluating Results

To compute quantitative metrics (BERTScore, etc.) on a test set, run:

```bash
python evaluate_test.py
```

Results will be saved to `sanskrit_results.csv`.

---

## 🧪 Running Inference Without the UI

If you prefer the command line over the web interface, you can use:

```bash
python inference.py
```

or the alternate variant:

```bash
python inference1.py
```

---

## 📋 Experiment Configurations at a Glance

| Script | Architecture | Diversity Penalty |
|---|---|---|
| `run_cross_att_no_neg.sh` | Cross-Attention | ❌ No |
| `run_cross_attn_with_neg.sh` | Cross-Attention | ✅ Yes |
| `run_enc_dec_no_neg.sh` | Encoder-Decoder | ❌ No |
| `run_enc_dec_with_neg.sh` | Encoder-Decoder | ✅ Yes |
| `run_base_cross_att_no_neg.sh` | Base + Cross-Attention | ❌ No |
| `run_base_cross_att_with_neg.sh` | Base + Cross-Attention | ✅ Yes |
| `run_base_enc_dec_no_neg.sh` | Base + Encoder-Decoder | ❌ No |
| `run_base_enc_dec_with_neg.sh` | Base + Encoder-Decoder | ✅ Yes |
| `run_all_experiments.sh` | All of the above | Both |
