import os
import torch
import torch.onnx
from model.d3pm_model_cross_attention import D3PMModel
from config import CONFIG
from env_utils import load_local_env

def export_to_onnx(checkpoint_path, output_path="sanskrit_model.onnx"):
    load_local_env(__file__)
    device = "cpu"
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize model
    model = D3PMModel(CONFIG)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Dummy inputs for tracing
    # x_t: (batch, seq_len), t: (batch,), cond: (batch, cond_len)
    batch_size = 1
    seq_length = 16
    cond_length = 64
    
    dummy_x = torch.randint(0, CONFIG['vocab_size'], (batch_size, seq_length)).to(device)
    dummy_t = torch.randint(0, CONFIG['timesteps'], (batch_size,)).to(device)
    dummy_cond = torch.randint(0, CONFIG['vocab_size'], (batch_size, cond_length)).to(device)
    
    print(f"Exporting model to {output_path}...")
    torch.onnx.export(
        model,
        (dummy_x, dummy_t, dummy_cond),
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['x_t', 't', 'condition'],
        output_names=['logits'],
        dynamic_axes={
            'x_t': {0: 'batch_size', 1: 'seq_len'},
            'condition': {0: 'batch_size', 1: 'cond_len'},
            'logits': {0: 'batch_size', 1: 'seq_len'}
        }
    )
    print("Export successful!")

if __name__ == "__main__":
    # Default to the T4 best model if environment is set
    ckpt = os.environ.get("HF_MODEL_CHECKPOINT", "ablation_results/T4/best_model.pt")
    if os.path.exists(ckpt):
        export_to_onnx(ckpt, "sanskrit_d3pm_T4.onnx")
    else:
        print(f"Checkpoint not found at {ckpt}. Please check your .env or paths.")
