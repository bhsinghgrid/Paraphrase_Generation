import gradio as gr
import json
import os
from datetime import datetime
from model_loader import ModelManager


CONFIG = {
    "model": {
        "vocab_size": 12000,
        "max_seq_len": 80,
        "diffusion_steps": 12,
        "d_model": 384,
        "n_layers": 6,
        "n_heads": 8,
        "d_ff": 1536,
        "dropout": 0.1
    },
    "diffusion": {
        "mask_token_id": 0
    },
    "training": {
        "precision": "float32",
        "device": "cpu"
    }
}


manager = ModelManager(CONFIG)
current_model = None
os.makedirs("results", exist_ok=True)


def chat(
    user_input,
    history,
    model_choice,
    beam,
    temp,
    top_k,
    top_p,
    max_len,
    rep_penalty
):

    global current_model

    # Map UI choice → internal model_type
    if model_choice == "Encoder-Decoder":
        model_type = "baseline_encoder_decoder"
    else:
        model_type = "baseline_cross_attention"

    if current_model != model_type:
        manager.load_model(model_type)
        current_model = model_type

    output = manager.generate(
        user_input,
        beam_width=beam,
        temperature=temp,
        top_k=top_k,
        top_p=top_p,
        max_len=max_len,
        repetition_penalty=rep_penalty
    )

    history.append((user_input, output))

    file_path = f"results/inference_{model_type}.json"

    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    data.append({
        "time": str(datetime.now()),
        "model": model_type,
        "input": user_input,
        "output": output
    })

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    return "", history


with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 🕉 Sanskrit Neural Generator")

    model_dropdown = gr.Dropdown(
        ["Encoder-Decoder", "Cross Attention"],
        value="Encoder-Decoder",
        label="Model Architecture"
    )

    with gr.Accordion("⚙️ Advanced Controls", open=False):

        beam_slider = gr.Slider(1, 10, value=3, step=1, label="Beam Width")
        temp_slider = gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="Temperature")
        top_k_slider = gr.Slider(0, 100, value=0, step=1, label="Top-k")
        top_p_slider = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="Top-p")
        max_len_slider = gr.Slider(10, 200, value=80, step=5, label="Max Length")
        rep_slider = gr.Slider(1.0, 2.0, value=1.0, step=0.1, label="Repetition Penalty")

    chatbot = gr.Chatbot(height=450)
    msg = gr.Textbox(placeholder="Type Sanskrit input here...")
    clear = gr.Button("Clear Chat")

    msg.submit(
        chat,
        inputs=[
            msg,
            chatbot,
            model_dropdown,
            beam_slider,
            temp_slider,
            top_k_slider,
            top_p_slider,
            max_len_slider,
            rep_slider
        ],
        outputs=[msg, chatbot]
    )

    clear.click(lambda: [], None, chatbot)

demo.launch()