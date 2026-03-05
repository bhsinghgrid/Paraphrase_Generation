import torch
import os
import sys
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from config import CONFIG

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data.dataset import OptimizedSanskritDataset
from model.tokenizer import SanskritTokenizer
from model.sanskrit_model import SanskritModel

try:
    import evaluate

    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False

try:
    from nltk.translate.bleu_score import corpus_bleu

    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False


class FinalEvaluator:
    def __init__(self):
        self.cfg = CONFIG
        self.device = torch.device(self.cfg["training"]["device"])

        # 📂 AUTOMATICALLY LOCATE THE CORRECT EXPERIMENT FOLDER
        model_name = self.cfg['model_type']
        has_neg = "True" if self.cfg['data']['include_negative_examples'] else "False"
        self.exp_dir = f"results/{model_name}_neg_{has_neg}"

        if not os.path.exists(f"{self.exp_dir}/best_model.pt"):
            raise FileNotFoundError(f"🚨 Model not found at {self.exp_dir}/best_model.pt! Train this config first.")

    def _collate(self, batch):
        return {
            "input_ids": torch.stack([b["input_ids"].long() for b in batch]),
            "target_text": [b["target_text"] for b in batch],
            "input_text": [b["input_text"] for b in batch]
        }

    def evaluate(self, sample_size=1000):
        print(f"🚀 Evaluating: {self.exp_dir}")

        tokenizer = SanskritTokenizer(self.cfg["model"]["vocab_size"])
        PAD_ID = tokenizer.tokenizer.token_to_id("[PAD]") if tokenizer.tokenizer.token_to_id("[PAD]") is not None else 1
        MASK_ID = self.cfg["diffusion"]["mask_token_id"]

        dataset = OptimizedSanskritDataset("test", tokenizer, self.cfg["model"]["max_seq_len"])
        indices = list(range(min(sample_size, len(dataset))))
        test_loader = DataLoader(Subset(dataset, indices), batch_size=self.cfg["training"]["batch_size"], shuffle=False,
                                 collate_fn=self._collate)

        # Load Model
        model = SanskritModel(self.cfg).to(self.device)
        model.load_state_dict(torch.load(f"{self.exp_dir}/best_model.pt", map_location=self.device))
        model.eval()

        all_predictions, all_references, all_inputs = [], [], []

        print(f"⏳ Generating translations for {len(indices)} samples...")
        for batch in tqdm(test_loader):
            input_ids = batch["input_ids"].to(self.device)
            refs = batch["target_text"]
            inputs = batch["input_text"]

            with torch.no_grad():
                # 🔥 This calls YOUR generate_beam logic from reverse_process.py
                # (via the model's wrapper methods)
                if "d3pm" in self.cfg["model_type"]:
                    output_ids = model.generate(input_ids, num_steps=self.cfg["model"]["diffusion_steps"], beam_width=3)
                else:
                    output_ids = model.generate(input_ids)

            for i in range(output_ids.size(0)):
                clean_ids = [id for id in output_ids[i].tolist() if id not in [MASK_ID, PAD_ID]]
                all_predictions.append(tokenizer.decode(clean_ids).strip())
                all_references.append(refs[i].strip())
                all_inputs.append(inputs[i].strip())

        # Calculate Metrics
        bleu_score = 0.0
        if BLEU_AVAILABLE:
            refs_tokenized = [[ref.split()] for ref in all_references]
            preds_tokenized = [pred.split() for pred in all_predictions]
            bleu_score = corpus_bleu(refs_tokenized, preds_tokenized)

        bert_f1 = 0.0
        if BERTSCORE_AVAILABLE:
            print("⏳ Calculating BERTScore...")
            bertscore = evaluate.load("bertscore")
            results = bertscore.compute(predictions=all_predictions, references=all_references, lang="hi")
            bert_f1 = sum(results['f1']) / len(results['f1'])

        # 💾 SAVE RESULTS TO FILE
        results_path = f"{self.exp_dir}/evaluation_results.txt"
        with open(results_path, "w", encoding="utf-8") as f:
            f.write(
                f"=== EVALUATION RESULTS: {self.cfg['model_type']} (Negatives: {self.cfg['data']['include_negative_examples']}) ===\n")
            f.write(f"BLEU Score : {bleu_score:.4f}\n")
            f.write(f"BERTScore  : {bert_f1:.4f}\n\n")
            f.write("=== SAMPLE PREDICTIONS ===\n")
            for i in range(min(10, len(all_predictions))):
                f.write(f"INPUT : {all_inputs[i]}\n")
                f.write(f"REF   : {all_references[i]}\n")
                f.write(f"PRED  : {all_predictions[i]}\n")
                f.write("-" * 50 + "\n")

        print(f"✅ Evaluation complete! Results saved to {results_path}")
        print(f"📊 BLEU: {bleu_score:.4f} | BERTScore: {bert_f1:.4f}")


if __name__ == "__main__":
    FinalEvaluator().evaluate(sample_size=1000)