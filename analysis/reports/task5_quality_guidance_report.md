# Task 5 Report: Quality Classifier and Guidance-Based Decoding

## 1. Objective

Task 5 attempts to guide generation using a lightweight quality classifier trained on decoder hidden states. The idea is to predict a quality score from hidden states and then use the classifier gradient to bias inference toward higher-quality outputs.

This is an ambitious extension because it adds a second learned component on top of the main D3PM model without retraining the core paraphrase model itself.

## 2. Implementation Approach

The implementation is in [analysis/quality_classifier.py](/Users/bhsingh/Documents/Final_Paraphrase/Exclude_Negative/analysis/quality_classifier.py). It has three stages:

1. collect `(hidden_state, quality_score)` pairs
2. train a small MLP quality classifier
3. use classifier gradients during decoding

### Classifier Definition Snippet

```python
class QualityClassifier(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
```

### Guidance Snippet

```python
hidden = x.detach().to(clf_device).requires_grad_(True)
hidden.retain_grad()
quality = classifier(hidden)
quality.sum().backward()
grad = hidden.grad.to(device)
logit_grad = grad @ inner.head.weight.T
logits = logits + guidance_scale * logit_grad
```

This turns hidden-state quality prediction into a differentiable decoding signal.

## 3. Current Status

Task 5 originally failed for two reasons:

- the gradient was taken from a non-leaf tensor, causing `hidden.grad` to be `None`
- the cached quality labels collapsed to all zeros, so the classifier had no meaningful learning signal

These implementation bugs were patched. However, the existing saved quality cache in [analysis/outputs/task5_quality_data.npz](/Users/bhsingh/Documents/Final_Paraphrase/Exclude_Negative/analysis/outputs/task5_quality_data.npz) still contains degenerate labels from the earlier failed run.

Observed cache statistics:

- count: `500`
- mean: `0.0`
- std: `0.0`
- min: `0.0`
- max: `0.0`

That means the current classifier result is not valid for evaluation.

## 4. Why the Current Result Is Not Reliable

Because all quality labels are zero:

- the classifier is effectively trained on a constant target
- low validation loss is meaningless
- guidance behavior cannot be interpreted as quality-aware control

So although the code path now exists, the saved run should not be used in mentor evaluation as a finished result.

## 5. What Was Fixed

Two concrete corrections were made:

- a bounded quality transform was introduced so very large CER values do not collapse everything to zero
- the Task 5 runner now refreshes cached quality data when it detects degenerate labels

This means Task 5 is closer to being experimentally sound, but it still needs to be rerun from scratch after the patch.

## 6. Expected Benefits

If Task 5 works as intended after rerunning, it could provide:

- a lightweight mechanism for improving generation quality
- a controllable quality-diversity tradeoff
- a reusable framework for guidance without retraining the full D3PM model
- a more research-oriented extension beyond standard training and inference

## 7. Limitations

At present, this task has one decisive limitation: the saved outputs are not valid evaluation artifacts. The infrastructure is promising, but the experimental evidence is not yet strong enough to defend.

## 8. Conclusion

Task 5 should be presented only as a partially completed advanced experiment. The implementation framework is now in place and the core bugs have been addressed, but the current cached run is still invalid for evaluation. Before showing this task to a mentor as a result, the quality data and guidance sweep should be rerun after patching so that the classifier is trained on non-degenerate labels.
