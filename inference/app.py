import gradio as gr
from fast_main import load_model, generate_text_and_save_json  # use the JSON-saving version

# ----------------------------
# Load model once
# ----------------------------
model, tokenizer, reverse_diffusion = load_model("d3pm_encoder_decoder")

# ----------------------------
# Gradio inference function
# ----------------------------
def gradio_infer(input_text, diversity, repetition, diversity_penalty, length_penalty):
    """
    Generate Sanskrit paraphrase and save all details to JSON.
    """
    return generate_text_and_save_json(
        input_text,
        model=model,
        tokenizer=tokenizer,
        reverse_diffusion=reverse_diffusion,
        diversity_level=diversity,
        repetition_penalty=float(repetition),
        diversity_penalty=float(diversity_penalty),
        length_penalty=float(length_penalty)
    )

# ----------------------------
# Gradio Interface
# ----------------------------
iface = gr.Interface(
    fn=gradio_infer,
    inputs=[
        gr.Textbox(label="Input Sanskrit Text"),
        gr.Radio(["low", "medium", "high"], label="Diversity", value="medium"),
        gr.Number(value=1.15, label="Repetition Penalty"),
        gr.Number(value=0.0, label="Diversity Penalty"),
        gr.Number(value=1.0, label="Length Penalty")
    ],
    outputs=[gr.Textbox(label="Output Text")],
    title="Sanskrit Paraphrase Generator",
    description="Generates paraphrases and saves all results1 automatically to a JSON file."
)

# ----------------------------
# Launch Gradio app
# ----------------------------
if __name__ == "__main__":
    iface.launch()