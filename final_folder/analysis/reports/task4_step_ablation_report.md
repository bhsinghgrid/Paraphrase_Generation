# Task 4 Report: Diffusion Step Ablation

## 1. Objective

Task 4 studies how the number of diffusion steps affects meaning preservation, speed, and robustness. The hypothesis is that fewer denoising steps may improve speed, but too few steps may reduce output quality. This type of ablation is important for mentor evaluation because it tests a core design parameter of the D3PM model.

Unlike the earlier tasks, this one requires retraining separate checkpoints for each step count. This is not optional. A model trained at `T=128` cannot be evaluated fairly at `T=4` or `T=8` without retraining, because the timestep distribution seen during training changes fundamentally.

## 2. Implementation Approach

The implementation is in [analysis/step_ablation.py](/Users/bhsingh/Documents/Final_Paraphrase/Exclude_Negative/analysis/step_ablation.py). I patched the workflow so it is safe for this repository:

- it no longer overwrites `config.py`
- it uses environment variables for `DIFFUSION_STEPS`
- each training run writes directly to `ablation_results/T*`

### Training Script Generation Snippet

```python
f.write(
    f"MODEL_TYPE=\"$MODEL_TYPE\" INCLUDE_NEG=\"$INCLUDE_NEG\" "
    f"TRAIN_DEVICE=\"$TRAIN_DEVICE\" "
    f"DIFFUSION_STEPS={T} INFERENCE_NUM_STEPS={T} "
    f"TRAIN_OUTPUT_DIR=\"ablation_results/T{T}\" "
    f"python train.py\n\n"
)
```

This makes the ablation workflow reproducible without mutating repository files between runs.

## 3. Current Workflow

Task 4 now supports the following sequence:

```bash
uv run --active analysis/run_analysis.py --task 4 --phase generate_configs
bash ablation_configs/train_all.sh
uv run --active analysis/run_analysis.py --task 4 --phase analyze
```

Generated script:

- [ablation_configs/train_all.sh](/Users/bhsingh/Documents/Final_Paraphrase/Exclude_Negative/ablation_configs/train_all.sh)

This script trains:

- `T=4`
- `T=8`
- `T=16`
- `T=32`
- `T=64`

with outputs saved to `ablation_results/T4`, `T8`, `T16`, `T32`, and `T64`.

## 4. Current Result Status

At the moment, no trained ablation checkpoints exist in `ablation_results/T*/best_model.pt`. Therefore, the analysis phase has no quantitative result yet. That means Task 4 currently has a correct implementation pipeline, but not a completed experiment.

This distinction matters for evaluation:

- the workflow is correct
- the experiment has not yet produced final numbers

## 5. Evaluation Value

For mentor evaluation, Task 4 can still be included, but it should be presented as:

- a completed experimental setup
- a validated retraining workflow
- pending final quantitative results

This is still useful because ablation design is part of research rigor. It shows that the project is set up to test the effect of a critical modeling choice instead of assuming the default step count is optimal.

## 6. Benefits

Once the checkpoints are trained, this task will answer:

- how much generation speed improves as diffusion steps decrease
- how meaning preservation changes with fewer steps
- where the best quality-speed tradeoff lies
- whether the current choice of diffusion steps is over- or under-provisioned

## 7. Limitations

The limitation is straightforward: there are no ablation checkpoints yet, so there are no real results to defend. It should not be presented as a finished evaluation experiment at this stage.

## 8. Conclusion

Task 4 is structurally correct and now safe to run in this repository. It is suitable for mentor evaluation as an experimental design and workflow contribution, but not yet as a result section. The next milestone is to train the five ablation checkpoints and run the analysis phase to generate the actual CER-speed comparison.
