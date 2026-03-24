# Task 3 Report: Concept Vectors and PCA-Based Steering

## 1. Objective

Task 3 explores whether decoder hidden states contain a measurable direction corresponding to paraphrase diversity. The idea is:

1. collect hidden states from many validation samples
2. fit PCA to the hidden-state space
3. find a principal direction correlated with output diversity
4. steer generation along that direction

This is an advanced representation-learning experiment. Its value for mentor evaluation lies in showing that the project is not limited to training and inference, but also investigates controllable generation.

## 2. Implementation Approach

The implementation is in [analysis/concept_vectors.py](/Users/bhsingh/Documents/Final_Paraphrase/Exclude_Negative/analysis/concept_vectors.py). Hidden states are captured from the decoder during cached inference and pooled across sequence positions.

### PCA Fitting Snippet

```python
def fit_pca(hidden_matrix, n_components=50):
    from sklearn.decomposition import PCA
    n_comp = min(n_components, hidden_matrix.shape[0] - 1, hidden_matrix.shape[1])
    pca = PCA(n_components=n_comp)
    pca.fit(hidden_matrix)
    return pca
```

### Steering Snippet

```python
if alpha != 0.0:
    x = x + alpha * dir_tensor.unsqueeze(0).unsqueeze(0)

logits = inner.head(x)
```

The steering mechanism adds a learned direction in hidden-state space before projection to logits.

## 3. Experimental Setup

Task 3 was run from the shared analysis driver and generated:

- [analysis/outputs/task3_concept_space.png](/Users/bhsingh/Documents/Final_Paraphrase/Exclude_Negative/analysis/outputs/task3_concept_space.png)
- [analysis/outputs/task3_diversity_direction.npy](/Users/bhsingh/Documents/Final_Paraphrase/Exclude_Negative/analysis/outputs/task3_diversity_direction.npy)
- [analysis/outputs/task3_report.txt](/Users/bhsingh/Documents/Final_Paraphrase/Exclude_Negative/analysis/outputs/task3_report.txt)

The run used 500 validation examples for hidden-state extraction.

## 4. Results

Observed summary:

- PCA components retained: `50`
- total explained variance: `96.1%`
- selected diversity principal component: `PC 1`
- absolute correlation with output length: `0.303`

On paper, these values suggest that hidden-state variation is structured and that at least one direction correlates with output-length changes. That is a positive sign from a representation-analysis standpoint.

However, the actual diversity spectrum outputs are not semantically convincing. The steered generations are highly repetitive and mostly malformed token sequences rather than clear paraphrases with controlled variation.

## 5. Interpretation

This task should be presented carefully.

What is supported:

- hidden states are rich enough for PCA analysis
- the representation space is not random noise
- controllable steering infrastructure has been implemented successfully

What is not yet supported:

- interpretable semantic control
- high-quality paraphrase diversity
- evidence that the identified direction reflects useful linguistic variation

For mentor evaluation, this is best framed as a promising exploratory experiment rather than a finished result.

## 6. Benefits

Benefits of the task include:

- opens a path toward controllable paraphrase generation
- demonstrates hidden-state instrumentation beyond standard inference
- provides a research direction for future work on style and diversity control
- connects model analysis with possible user-facing controllability

## 7. Limitations

The main limitation is output quality. Even though the PCA statistics look reasonable, the steered generations are not linguistically strong enough to claim meaningful semantic control. This makes the current result more useful as a prototype than as a validated research finding.

## 8. Conclusion

Task 3 is not yet strong enough as a final evaluation result, but it is valuable as research evidence of advanced model analysis. For mentor discussion, it should be described as an experimental controllability framework that has been implemented successfully but still requires better base model quality before the steering outputs become persuasive.
