---

# 🕉️ Sanskrit AI Paraphrasing Engine (D3PM)

Welcome to the **Sanskrit Paraphrase Lab**. This project is a state-of-the-art text-to-text generation tool designed specifically for the Sanskrit language. It provides researchers, linguists, and data scientists with an interactive environment to test custom-trained AI models, generate paraphrases, and automatically log metadata for accuracy grading (e.g., BERTScore).

---

## 🧠 The Science: How it Works (A Deeper Dive)

Unlike standard AI models (like ChatGPT) that guess words one by one from left to right, this project uses a cutting-edge architecture called **Discrete Denoising Diffusion Probabilistic Models (D3PM)** with Cross-Attention.

1. **The Starting Point:** When you input a Sanskrit verse, the AI doesn't translate it directly. Instead, it creates a blank sequence of `[MASK]` tokens (digital "noise").
2. **The Denoising Process:** Over a series of mathematical steps (Diffusion Steps), the AI iteratively refines this noise. It uses **Cross-Attention** to constantly look back at your original verse, ensuring the meaning remains intact.
3. **Self-Conditioning:** At each step, the model feeds its previous guess back into itself as a "hint." This forces the AI to maintain strict grammatical consistency—a crucial requirement for a highly inflected language like Sanskrit.
4. **The Penalties:** We implemented custom mathematical penalties (Diversity and Repetition) into the beam search. This gives the user granular control to push the AI toward finding rare synonyms or to keep it strictly literal.

---

## 📂 Understanding the Project Structure

When you download this project, you will see several folders and files. Here is what they do:

* **`inference.py`**: The main application file. Running this launches the user-friendly web interface.
* **`model/`**: Contains the core AI architecture (`sanskrit_model.py`) and the custom 16,000-word Sanskrit Tokenizer (`tokenizer.py`).
* **`diffusion/`**: Contains `reverse_process.py`, which holds the complex mathematics for the denoising steps and generation penalties.
* **`results/`**: This is where your trained AI weights (the `.pt` files) live. (e.g., `d3pm_cross_attention_neg_False`). The app automatically scans this folder to let you switch between different models.
* **`run_all_experiments.sh`**: Automated shell scripts for developers who want to train the models from scratch.
* **`backend_generation_log.csv`**: The master database. Every generation is automatically saved here to ensure zero data loss.

---

Here is the updated section for your `README.md`. I have added a clear new step showing exactly how to run the `.sh` files right after installing the requirements, explaining that this is necessary to generate the model results.

You can replace the **"How to Access & Run the Entire Project"** section of your README with this updated version:

---

## 📥 How to Access & Run the Entire Project

You do not need to be a software developer to run this tool on your computer. Follow these steps to get the Paraphrase Lab running locally:

### Step 1: Download the Project

1. Navigate to the top of this GitHub repository.
2. Click the green **"<> Code"** button.
3. Click **"Download ZIP"** and extract the folder to your computer (e.g., your Desktop).

### Step 2: Set Up the Environment

To ensure the AI runs smoothly without messing up your computer's other settings, we will create a "Virtual Environment". You will need Python installed on your computer.

1. Open your computer's **Terminal** (Mac) or **Command Prompt** (Windows).
2. Tell the Terminal to go inside your downloaded folder. Type `cd ` (with a space), drag the unzipped project folder into the Terminal, and hit **Enter**.
3. Create a virtual environment by typing:
```bash
python -m venv .venv

```


4. Activate the environment:
* **Mac/Linux:** `source .venv/bin/activate`
* **Windows:** `.venv\Scripts\activate`


5. Install the required AI libraries:
```bash
pip install -r requirements.txt

```



### Step 3: Run the Training Experiments (.sh files)

Before you can use the web interface, you need to run the experiment scripts to train the models and generate the weights.

1. While still in your Terminal with the virtual environment activated, run one of the provided bash scripts by typing `bash` followed by the file name. For example:
```bash
bash run_cross_att_no_neg.sh

```


2. **Want to run everything?** If you want to run all the different model configurations sequentially, simply run the master script:
```bash
bash run_all_experiments.sh

```



*(Note: This process will begin training the AI. Once it finishes, it will automatically save the trained weights into the `results/` folder so the web app can use them.)*

### Step 4: Launch the Paraphrase Lab (Web App)

Once your `.sh` scripts have finished running and your models are saved in the `results/` folder, you can launch the interactive interface!

1. In your terminal, type:
```bash
python inference.py

```


2. After a few seconds, a local web link (usually `http://127.0.0.1:7860`) will appear in the Terminal. **Click that link** to open the Paraphrase Lab in your web browser!
## 🖥️ How to Use the Paraphrase Lab Interface

Once the app is open, you will see a clean, interactive dashboard. Here is how to use it:

1. **Select Experiment:** Use the dropdown menu at the top left to select your AI model (e.g., `d3pm_cross_attention_neg_False`).
2. **Load Model (Crucial!):** Click the **"🔄 Load Model"** button. Wait until the status changes to *✅ Loaded*.
3. **Choose a Strategy:**
* **Manual Tuning:** Unlocks the "Advanced Controls" accordion so you can adjust the Temperature, Beam Width, and Penalties yourself.
* **High Diversity:** Automatically sets the sliders to encourage the AI to be highly creative with its synonyms.
* **Low Diversity:** Automatically sets the sliders to keep the AI strict and literal.


4. **Generate:** Paste your source verse into the text box and click **"✨ Paraphrase"**.
5. **Save Your Work:** Your generations will appear in the "Current Session History" table. Click **"Download Current Session"** to save your immediate results, or check the `backend_generation_log.csv` in your project folder for a permanent record of everything you have ever generated.

---
