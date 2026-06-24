# Fractional Information Gain

**Fractional Information Gain (FIG)** is a performance metric for knowledge-tracing models
in educational assessment. This library implements it in Python (NumPy), PyTorch, and R.

Knowledge-tracing models are commonly evaluated with metrics like AUC, accuracy, and F1.
These metrics can be dominated by item difficulty: a hard item is hard for everyone, an easy item is easy for everyone. FIG
isolates the part of a model's performance that reflects actual student-level knowledge, by
comparing predictions against per-item baselines computed from training data. This library
currently handles binary (correct/incorrect) response data; support for other response
types may be added in the future.

FIG = 0 means the model adds nothing beyond item base rates. FIG = 1 means perfect
predictions. FIG < 0 means the model is worse than the baseline.

FIG comes in two variants, both of which require training data to compute per-item
baselines (success rates):

- **FIG-C** (Confidence) -- measures whether the model is *confident*. Does not need
  evaluation-set ground truth. For well-calibrated models, FIG-C approximates FIG-V in
  expectation.
- **FIG-V** (Validation) -- measures whether the model is confident *and correct*.
  Requires ground truth labels on the evaluation set. Can also be used as a differentiable
  training loss (see [PyTorch loss](getting-started.md#using-fig-as-a-pytorch-loss)).

Both metrics are reported as observation-weighted (pooled) and student-weighted
(macro-average) variants.

## Quick Example

```python
from fig import (
    fractional_information_gain_validation,
    fractional_information_gain_confidence,
)

# FIG-C: no ground truth on eval needed
results = fractional_information_gain_confidence(
    y_pred_eval=y_pred,        # model probabilities, shape (n_eval,)
    item_id_eval=items,        # item IDs for eval observations
    student_id_eval=students,  # student IDs for eval observations
    y_train=y_train,           # ground truth on train split (for baseline)
    item_id_train=items_train, # item IDs on train split
)
print(results["fig_c"])         # student-weighted FIG-C

# FIG-V: also requires ground truth on the eval split
results = fractional_information_gain_validation(
    y_pred_eval=y_pred,
    y_eval=y_true,             # ground truth (0/1), shape (n_eval,)
    item_id_eval=items,
    student_id_eval=students,
    y_train=y_train,
    item_id_train=items_train,
)
print(results["fig_v"])         # student-weighted FIG-V
print(results["fig_v_pooled"])  # observation-weighted FIG-V
```

## Implementations

| Implementation | Language | Differentiable | Import |
|---|---|---|---|
| NumPy | Python | No | `from fig import ...` |
| PyTorch | Python | Yes | `from fig.fig_torch import ...` |
| R | R | No | `library(figmetric)` |

The NumPy implementation is the primary reference. The PyTorch version can be used as a
differentiable training loss. The R package mirrors the Python API.

## Next Steps

- [Installation](installation.md) -- install via pip or from source
- [Getting Started](getting-started.md) -- walkthrough with realistic examples
- [Concepts](concepts.md) -- the math and intuition behind FIG
- [API Reference](api/fig.md) -- full function and type documentation
