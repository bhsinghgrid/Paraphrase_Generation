# Task 2 Report: Attention Visualization and Semantic Drift

## 1. Objective

Task 2 investigates how the diffusion model behaves internally during generation. It has two goals:

- capture cross-attention patterns between source and generated target tokens
- measure how intermediate generations converge toward the final output over diffusion steps

This task is important for evaluation because it gives interpretability evidence. Instead of only showing the final prediction, it examines whether the model gradually stabilizes its output and whether attention is distributed in a meaningful way.

## 2. Implementation Approach

The implementation uses two analysis modules:

- [analysis/attention_viz.py](/Users/bhsingh/Documents/Final_Paraphrase/Exclude_Negative/analysis/attention_viz.py)
- [analysis/semantic_drift.py](/Users/bhsingh/Documents/Final_Paraphrase/Exclude_Negative/analysis/semantic_drift.py)

To support this, the cross-attention layer stores attention weights during decoding. The model also exposes a cached inference path so per-step diagnostics can be collected efficiently.

### Attention Capture Snippet

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        ...
        self.capture_weights = False
        self.last_attn_weights = None

    def forward(self, q, k, v, mask=None):
        ...
        attn = self.dropout(torch.softmax(scores, dim=-1))
        if self.capture_weights:
            self.last_attn_weights = attn.detach().cpu()
```

### Drift Computation Snippet

```python
def compute_drift(step_outputs, final_output):
    t_vals = sorted(step_outputs.keys(), reverse=True)
    cer_to_final = []
    for t_val in t_vals:
        cer = compute_cer_between(step_outputs[t_val], final_output)
        cer_to_final.append(cer)
```

The metric used is character error rate between each intermediate output and the final output.

## 3. Experimental Setup

The task was run with:

```bash
uv run --active analysis/run_analysis.py --task 2 --input "dharmo rakṣati rakṣitaḥ"
```

Generated outputs:

- [analysis/outputs/task2_attn_t127.png](/Users/bhsingh/Documents/Final_Paraphrase/Exclude_Negative/analysis/outputs/task2_attn_t127.png)
- [analysis/outputs/task2_attn_t0.png](/Users/bhsingh/Documents/Final_Paraphrase/Exclude_Negative/analysis/outputs/task2_attn_t0.png)
- [analysis/outputs/task2_all_layers_t0.png](/Users/bhsingh/Documents/Final_Paraphrase/Exclude_Negative/analysis/outputs/task2_all_layers_t0.png)
- [analysis/outputs/task2_attn_evolution.png](/Users/bhsingh/Documents/Final_Paraphrase/Exclude_Negative/analysis/outputs/task2_attn_evolution.png)
- [analysis/outputs/task2_semantic_drift.png](/Users/bhsingh/Documents/Final_Paraphrase/Exclude_Negative/analysis/outputs/task2_semantic_drift.png)
- [analysis/outputs/task2_report.txt](/Users/bhsingh/Documents/Final_Paraphrase/Exclude_Negative/analysis/outputs/task2_report.txt)

## 4. Results

The saved report shows:

- lock-in timestep: `t = 22`
- mean token-position lock-in: `53.6 ± 28.4`

This indicates that the generated sequence becomes relatively stable before the final denoising step. In other words, the model is not making all of its decisions only at the very end.

However, the actual generated Sanskrit output is low quality and strongly repetitive. That matters for interpretation: the drift curve is still valid as a measure of convergence, but it is convergence toward a weak final output.

## 5. Interpretation

For mentor evaluation, this task should be presented as a diagnostic analysis rather than a quality claim.

What the task supports:

- the model’s output evolves gradually over time
- the diffusion process shows an identifiable stabilization region
- attention weights can now be inspected layer by layer

What the task does not yet support:

- strong semantic alignment
- trustworthy linguistic paraphrase quality
- meaningful claim that attention maps correspond to correct Sanskrit transformation

## 6. Benefits

This task has practical value even with imperfect outputs:

- helps identify when the model stabilizes
- supports debugging of the denoising trajectory
- provides visual artifacts for discussing model internals
- can guide reduction of unnecessary inference steps in future work

## 7. Limitations

There are two important limitations:

1. The output quality is weak, so the interpretability evidence is about model behavior, not model correctness.
2. Matplotlib on the current machine does not render Devanagari fonts well, so the generated figures contain font warnings and may not display labels cleanly.

## 8. Conclusion

Task 2 is partially suitable for evaluation. It is strong as an interpretability and debugging report, but weak as proof of semantic paraphrase quality. For mentor review, it should be framed as evidence that the diffusion generation process can now be inspected and analyzed step by step.
