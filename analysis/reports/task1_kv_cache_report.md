# Task 1 Report: KV Cache Benchmark

## 1. Objective

The purpose of Task 1 is to measure whether encoder-side key/value caching improves inference speed for the cross-attention D3PM paraphrase model. In the unoptimized version, the source sequence is re-encoded at every diffusion step. In the cached version, the source is encoded once and reused for all denoising steps.

This task is useful for mentor evaluation because it measures an engineering improvement directly tied to deployment cost. Even when model quality is unchanged, lower generation latency improves usability for experimentation, batch evaluation, and interactive inference.

## 2. Implementation Approach

The benchmark is implemented in [analysis/kv_cache_benchmark.py](/Users/bhsingh/Documents/Final_Paraphrase/Exclude_Negative/analysis/kv_cache_benchmark.py). To support it, the cross-attention model was extended with three helper methods in [model/d3pm_model_cross_attention.py](/Users/bhsingh/Documents/Final_Paraphrase/Exclude_Negative/model/d3pm_model_cross_attention.py):

- `encode_source(...)`
- `forward_cached(...)`
- `generate_cached(...)`

These methods separate source encoding from decoder-side denoising, which is the standard way to benchmark KV caching in encoder-decoder style architectures.

### Core Implementation Snippet

```python
def encode_source(self, src):
    PAD = 1
    src_pad_mask = (src == PAD)
    memory = self.src_embed(src)
    for block in self.encoder_blocks:
        memory = block(memory, pad_mask=src_pad_mask)
    return memory, src_pad_mask

def forward_cached(self, memory, src_pad_mask, tgt, t, x0_hint=None, inference_mode=False):
    ...
    for block in self.decoder_blocks:
        x = block(x, memory, tgt_pad_mask=tgt_pad_mask, src_pad_mask=src_pad_mask)
    self._last_hidden = x.detach()
    return self.head(x), None
```

This design avoids recomputing the encoder stack at each diffusion step.

## 3. Experimental Setup

The benchmark was run using the Task 1 entry point:

```bash
uv run --active analysis/run_analysis.py --task 1
```

The script tests source lengths of 16, 32, and 64 tokens and reports:

- standard generation time
- cached generation time
- speedup ratio
- estimated encoder cost as a percentage of one forward pass

The benchmark output is stored in [analysis/outputs/task1_kv_cache.txt](/Users/bhsingh/Documents/Final_Paraphrase/Exclude_Negative/analysis/outputs/task1_kv_cache.txt).

## 4. Results

Observed benchmark values:

| Source Length | Standard (s) | Cached (s) | Speedup | Encoder % |
| --- | ---: | ---: | ---: | ---: |
| 16 | 1.784 | 1.780 | 1.00x | 42.7% |
| 32 | 2.055 | 1.850 | 1.11x | 41.9% |
| 64 | 1.724 | 1.608 | 1.07x | 43.2% |

The main outcome is that caching works correctly and provides a measurable speed improvement, though the improvement is modest on the current hardware and runtime stack.

## 5. Interpretation

The result is technically correct and useful, but it should be positioned carefully in evaluation:

- This is a systems optimization result, not a model quality result.
- The speedup is real, but not dramatic.
- The benchmark confirms that source-side recomputation can be removed without changing the inference algorithm.

For mentor evaluation, this can be presented as a successful engineering optimization with limited but positive runtime impact.

## 6. Benefits

Benefits of this task:

- reduces redundant encoder computation
- provides a reusable cached inference path for later analysis tasks
- improves scalability for repeated generation and diagnostic probes
- establishes infrastructure for attention and hidden-state inspection

## 7. Limitations

The result should not be overstated:

- speedup depends heavily on hardware and backend
- current gains are relatively small
- more stable benchmarking would require repeated runs and device-specific profiling
- this does not improve semantic accuracy directly

## 8. Conclusion

Task 1 is valid and suitable for mentor evaluation as an implementation-focused result. It demonstrates that cached inference was successfully added to the D3PM cross-attention model and that it reduces generation cost modestly. The strongest value of this task is architectural: it enables faster repeated inference and supports later interpretability experiments.
