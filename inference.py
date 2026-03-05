# import gradio as gr
# import torch
# import torch.nn.functional as F
# import pandas as pd
# import csv
# import os
# from datetime import datetime
# from pathlib import Path
#
# # Internal project imports
# from model.sanskrit_model import SanskritModel
# from model.tokenizer import SanskritTokenizer
#
#
# # --- Integrated Reverse Diffusion Penalty Logic ---
# def apply_repetition_penalty(logits, prev_tokens, penalty=1.2):
#     B, L, V = logits.shape
#     for b in range(B):
#         # Prevent penalizing the [MASK] or [PAD] tokens heavily if they are ID 0 or 1
#         used_tokens = set(prev_tokens[b].tolist()) - {0, 1, 2, 3}
#         for token_id in used_tokens:
#             logits[b, :, token_id] /= penalty
#     return logits
#
#
# def apply_diversity_penalty(logits, penalty=0.3):
#     logits_var = logits.var(dim=-1, keepdim=True)
#     return logits + penalty * logits_var
#
#
# class SanskritLab:
#     def __init__(self):
#         self.cfg = {
#             "model_type": "d3pm_cross_attention",
#             "model": {
#                 "vocab_size": 16000, "max_seq_len": 80, "diffusion_steps": 10,
#                 "d_model": 384, "n_layers": 6, "n_heads": 6, "d_ff": 1536, "dropout": 0.15
#             },
#             "diffusion": {"mask_token_id": 0}
#         }
#         self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#         self.tokenizer = SanskritTokenizer(self.cfg)
#         self.model = None
#         self.history = []
#         self.backend_log_file = "backend_generation_log.csv"
#
#     def get_models(self):
#         path = Path("results")
#         return sorted([p.parent.name for p in path.glob("**/best_model.pt")]) if path.exists() else []
#
#     def load_model(self, folder_name):
#         if not folder_name: return "⚠️ Select a folder"
#         ckpt_path = Path("results") / folder_name / "best_model.pt"
#         try:
#             if torch.backends.mps.is_available(): torch.mps.empty_cache()
#
#             self.model = SanskritModel(self.cfg).to(self.device)
#             ckpt = torch.load(ckpt_path, map_location=self.device)
#
#             state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
#             self.model.load_state_dict(state_dict)
#             self.model.eval()
#
#             return f"✅ Loaded: {folder_name}"
#         except Exception as e:
#             return f"❌ Load Error: {str(e)}"
#
#     def custom_generate(self, input_ids, num_steps, temperature, repetition_penalty, diversity_penalty):
#         B, L = input_ids.shape
#         mask_id = self.cfg["diffusion"]["mask_token_id"]
#         target_ids = torch.full((B, L), fill_value=mask_id, dtype=torch.long, device=self.device)
#
#         for step in reversed(range(num_steps)):
#             t_tensor = torch.full((B,), step, dtype=torch.long, device=self.device)
#
#             with torch.no_grad():
#                 outputs = self.model(input_ids, target_ids, t_tensor)
#                 logits = outputs[0] if isinstance(outputs, tuple) else outputs
#
#             logits = logits / temperature
#             if repetition_penalty != 1.0:
#                 logits = apply_repetition_penalty(logits, target_ids, repetition_penalty)
#             if diversity_penalty > 0.0:
#                 logits = apply_diversity_penalty(logits, diversity_penalty)
#
#             probs = F.softmax(logits, dim=-1)
#             target_ids = torch.argmax(probs, dim=-1)
#
#         return target_ids
#
#     # 🔥 NEW: Automatically appends to a backend file instantly
#     def append_to_backend_log(self, record):
#         file_exists = os.path.isfile(self.backend_log_file)
#
#         with open(self.backend_log_file, mode='a', newline='', encoding='utf-8') as f:
#             fieldnames = [
#                 "Timestamp", "Model_Name", "Strategy", "Temperature",
#                 "Repetition_Penalty", "Diversity_Penalty", "Diffusion_Steps",
#                 "Input_Text", "Generated_Output"
#             ]
#             writer = csv.DictWriter(f, fieldnames=fieldnames)
#
#             if not file_exists:
#                 writer.writeheader()
#             writer.writerow(record)
#
#     def generate(self, model_name, text, temp, rep_pen, div_pen, steps, mode):
#         if self.model is None:
#             return "⚠️ Load a model first!", pd.DataFrame(self.history)
#
#         # Mode overrides
#         if mode == "High Diversity":
#             div_pen, rep_pen, temp = 0.8, 1.4, 1.2
#         elif mode == "Low Diversity":
#             div_pen, rep_pen, temp = 0.0, 1.1, 0.7
#
#         try:
#             tokens = self.tokenizer.encode(text)
#             input_ids = torch.tensor(tokens).unsqueeze(0).to(self.device)
#
#             output_ids = self.custom_generate(
#                 input_ids=input_ids,
#                 num_steps=int(steps),
#                 temperature=float(temp),
#                 repetition_penalty=float(rep_pen),
#                 diversity_penalty=float(div_pen)
#             )
#
#             result = self.tokenizer.decode(output_ids[0].tolist())
#             timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#
#             # 🔥 NEW: Build the comprehensive metadata record
#             full_record = {
#                 "Timestamp": timestamp,
#                 "Model_Name": model_name,
#                 "Strategy": mode,
#                 "Temperature": temp,
#                 "Repetition_Penalty": rep_pen,
#                 "Diversity_Penalty": div_pen,
#                 "Diffusion_Steps": steps,
#                 "Input_Text": text,
#                 "Generated_Output": result
#             }
#
#             # 🔥 NEW: Instantly save to the backend master file
#             self.append_to_backend_log(full_record)
#
#             # Update UI History (Keep this brief for the on-screen table)
#             self.history.insert(0, {
#                 "Time": datetime.now().strftime("%H:%M:%S"),
#                 "Strategy": mode,
#                 "Input": text[:40] + "...",
#                 "Output": result
#             })
#
#             return result, pd.DataFrame(self.history)
#
#         except Exception as e:
#             return f"Generation Error: {str(e)}", pd.DataFrame(self.history)
#
#     # Legacy export for the UI download button (exports the session only)
#     def export_csv(self):
#         if not self.history: return None
#         path = "session_export.csv"
#         pd.DataFrame(self.history).to_csv(path, index=False)
#         return path
#
#
# # --- Clean Gradio UI ---
# app = SanskritLab()
#
# with gr.Blocks(theme=gr.themes.Soft()) as demo:
#     gr.Markdown("# 🕉️ Sanskrit Paraphrase Diffusion Lab")
#
#     with gr.Row():
#         with gr.Column(scale=1, variant="panel"):
#             m_drop = gr.Dropdown(choices=app.get_models(), label="1. Select Experiment")
#             load_btn = gr.Button("🔄 Load Model", variant="secondary")
#             status = gr.Markdown("*Status: Not Loaded*")
#
#             gr.HTML("<hr>")
#             mode = gr.Radio(["Manual Tuning", "High Diversity", "Low Diversity"], value="Manual Tuning",
#                             label="2. Generation Strategy")
#
#             with gr.Accordion("3. Advanced Controls", open=False):
#                 temp = gr.Slider(0.1, 2.0, 0.8, label="Temperature (Randomness)")
#                 div = gr.Slider(0.0, 1.0, 0.3, label="Diversity Penalty (Synonyms)")
#                 rep = gr.Slider(1.0, 2.0, 1.2, label="Repetition Penalty")
#                 steps = gr.Slider(1, 50, 10, step=1, label="Diffusion Steps")
#
#         with gr.Column(scale=2):
#             src_in = gr.Textbox(label="Source Verse", lines=4, placeholder="यदा यदा हि धर्मस्य...")
#             res_out = gr.Textbox(label="Paraphrase Result", lines=4, interactive=False)
#
#             with gr.Row():
#                 run_btn = gr.Button("✨ Paraphrase", variant="primary")
#                 clear_btn = gr.Button("🗑️ Clear")
#
#             gr.HTML("<hr>")
#             gr.Markdown(
#                 "*Note: All generations and metadata are automatically saved to `backend_generation_log.csv` on the server.*")
#             hist_df = gr.Dataframe(label="Current Session History")
#
#             with gr.Row():
#                 dl_btn = gr.Button("📂 Download Current Session", variant="secondary")
#                 file_dl = gr.File(label="Downloaded File")
#
#     # Connect UI to logic
#     load_btn.click(app.load_model, inputs=[m_drop], outputs=[status])
#     run_btn.click(app.generate, inputs=[m_drop, src_in, temp, rep, div, steps, mode], outputs=[res_out, hist_df])
#     clear_btn.click(lambda: ("", pd.DataFrame()), outputs=[src_in, hist_df])
#     dl_btn.click(app.export_csv, outputs=[file_dl])
#
# if __name__ == "__main__":
#     demo.launch(share=True)

import torch
import torch.nn.functional as F
import inspect
import gradio as gr
import pandas as pd
import os
import csv
from datetime import datetime
from pathlib import Path

# Internal project imports
from model.sanskrit_model import SanskritModel
from model.tokenizer import SanskritTokenizer

# --- 1. Advanced Penalty Logic ---
def apply_repetition_penalty(logits, prev_tokens, penalty=1.5):
    B, L, V = logits.shape
    for b in range(B):
        counts = torch.bincount(prev_tokens[b], minlength=V)
        repeated_mask = counts > 0
        applied_penalty = torch.pow(penalty, counts[repeated_mask].float())
        logits[b, :, repeated_mask] /= applied_penalty.to(logits.device)
    return logits

def apply_diversity_penalty(logits, penalty=0.3):
    logits_var = logits.var(dim=-1, keepdim=True)
    return logits + penalty * logits_var

# --- 2. Reverse Diffusion Logic ---
class ReverseDiffusion:
    def __init__(self, mask_token_id):
        self.mask_token_id = mask_token_id

    def p_sample_step(self, model, x_t, t, condition, x0_hint, beam_width, temp, rep_pen, div_pen):
        with torch.no_grad():
            sig = inspect.signature(model.forward).parameters
            # 🔥 Pass x0_hint for Self-Conditioning
            if 'x0_hint' in sig:
                outputs = model(condition, x_t, t, x0_hint=x0_hint)
            else:
                outputs = model(condition, x_t, t)

            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            logits = logits / temp

            if rep_pen != 1.0:
                logits = apply_repetition_penalty(logits, x_t, rep_pen)
            if div_pen > 0:
                logits = apply_diversity_penalty(logits, div_pen)

            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_ids = torch.topk(probs, beam_width, dim=-1)

            candidates = []
            for k in range(beam_width):
                next_tokens = topk_ids[:, :, k]
                score = torch.log(topk_probs[:, :, k] + 1e-9).mean()
                candidates.append((next_tokens, score))
            return candidates

    def generate_beam(self, model, condition, beam_width, num_steps, temp, rep_pen, div_pen):
        device = condition.device
        B, L = condition.shape
        x_init = torch.full((B, L), fill_value=self.mask_token_id, dtype=torch.long, device=device)
        beams = [(x_init, 0.0, None)]

        for step in reversed(range(num_steps)):
            new_beams = []
            t_tensor = torch.full((B,), step, dtype=torch.long, device=device)
            for x_t, score, last_x0 in beams:
                candidates = self.p_sample_step(model, x_t, t_tensor, condition, last_x0, beam_width, temp, rep_pen, div_pen)
                for tokens, new_score in candidates:
                    new_beams.append((tokens, score + new_score, tokens))
            new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_width]
        return beams[0][0]

# --- 3. Gradio Interface Wrapper ---
class SanskritLab:
    def __init__(self):
        self.cfg = {
            "model_type": "d3pm_cross_attention",
            "model": {
                "vocab_size": 16000, "max_seq_len": 80, "diffusion_steps": 10,
                "d_model": 384, "n_layers": 6, "n_heads": 6, "d_ff": 1536, "dropout": 0.15
            },
            "diffusion": {"mask_token_id": 0}
        }
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = SanskritTokenizer(self.cfg)
        self.model = None
        self.current_model_name = "None"
        self.rd = ReverseDiffusion(mask_token_id=0)
        self.history = []
        self.master_log = "backend_generation_log.csv"

    def load_model(self, folder_name):
        if not folder_name: return "⚠️ Select a folder"
        ckpt_path = Path("results") / folder_name / "best_model.pt"
        try:
            self.model = SanskritModel(self.cfg).to(self.device)
            ckpt = torch.load(ckpt_path, map_location=self.device)
            state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
            self.current_model_name = folder_name
            return f"✅ Loaded: {folder_name}"
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def save_to_master_log(self, record):
        """ 🔥 PERMANENT SAVING LOGIC """
        file_exists = os.path.isfile(self.master_log)
        with open(self.master_log, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)

    def generate(self, text, temp, rep_pen, div_pen, steps, beam_width):
        if self.model is None: return "⚠️ Load model first!", pd.DataFrame()

        tokens = self.tokenizer.encode(text)
        condition = torch.tensor(tokens).unsqueeze(0).to(self.device)

        output_ids = self.rd.generate_beam(self.model, condition, beam_width, steps, temp, rep_pen, div_pen)
        result = self.tokenizer.decode(output_ids[0].tolist())

        # 🔥 Prepare Record for CSV
        record = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Model": self.current_model_name,
            "Input": text,
            "Output": result,
            "Temp": temp,
            "Rep_Pen": rep_pen,
            "Div_Pen": div_pen,
            "Steps": steps,
            "Beams": beam_width
        }
        self.save_to_master_log(record)

        # Update UI history
        self.history.insert(0, {"Time": record["Timestamp"], "Input": text[:30], "Output": result})
        return result, pd.DataFrame(self.history)

# --- 4. UI ---
app = SanskritLab()
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🕉️ Sanskrit Reverse Diffusion Lab")
    with gr.Row():
        with gr.Column(scale=1):
            m_drop = gr.Dropdown(choices=sorted([p.parent.name for p in Path("results").glob("**/best_model.pt")]) if Path("results").exists() else [], label="Select Model")
            load_btn = gr.Button("🔄 Load")
            temp = gr.Slider(0.1, 1.5, 0.8, label="Temperature")
            rep = gr.Slider(1.0, 2.0, 1.5, label="Repetition Penalty")
            div = gr.Slider(0.0, 1.0, 0.3, label="Diversity Penalty")
            beams = gr.Slider(1, 5, 3, step=1, label="Beam Width")
            steps = gr.Slider(1, 20, 10, step=1, label="Steps")
        with gr.Column(scale=2):
            src_in = gr.Textbox(label="Source IAST", placeholder="yadā yadā hi dharmasya...")
            res_out = gr.Textbox(label="Result", interactive=False)
            run_btn = gr.Button("✨ Generate", variant="primary")
            hist_df = gr.Dataframe()

    load_btn.click(app.load_model, inputs=[m_drop], outputs=[gr.Markdown()])
    run_btn.click(app.generate, inputs=[src_in, temp, rep, div, steps, beams], outputs=[res_out, hist_df])

if __name__ == "__main__":
    demo.launch()