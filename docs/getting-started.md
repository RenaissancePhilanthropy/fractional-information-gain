# Getting Started

## Data Format

FIG expects data in **long format**: one row per student-item observation. An **item** is a
question or task (e.g. "question 5 on the algebra test") that multiple students can respond
to.

| Array | Description | Example |
|---|---|---|
| `y_train` | Binary outcomes on the training split (0/1) | `[1, 0, 1, 1, 0, ...]` |
| `item_id_train` | Item IDs for training observations | `[5, 3, 5, 2, 3, ...]` |
| `y_pred_eval` | Model-predicted probabilities on the eval split | `[0.82, 0.35, ...]` |
| `y_eval` | Binary outcomes on the eval split (FIG-V only) | `[1, 0, ...]` |
| `item_id_eval` | Item IDs for eval observations | `[5, 2, ...]` |
| `student_id_eval` | Student IDs for eval observations | `[1, 1, ...]` |

Train and eval arrays must each be the same length internally, but can differ from each
other. Student IDs are only needed on the eval side. Item and student IDs can be any
sortable type (integers, strings, etc.).

The training data is used **only** to compute per-item baseline success rates. It does not
need to come from the same model training run.

## Example Data

```python
import numpy as np

# Simulate training data: 500 observations across 10 items
rng = np.random.default_rng(42)
n_train = 500
y_train = rng.choice([0, 1], size=n_train, p=[0.4, 0.6])
item_id_train = rng.integers(1, 11, size=n_train)

# Simulate eval data: 200 observations, 20 students, 10 items
n_eval = 200
y_eval = rng.choice([0, 1], size=n_eval, p=[0.4, 0.6])
item_id_eval = rng.integers(1, 11, size=n_eval)
student_id_eval = rng.integers(1, 21, size=n_eval)

# Simulate a decent model: predictions correlated with ground truth.
# (In practice y_pred_eval comes from your model, not from y_eval.)
y_pred_eval = np.where(
    y_eval == 1,
    rng.beta(4, 2, size=n_eval),   # higher predictions for correct
    rng.beta(2, 4, size=n_eval),   # lower predictions for incorrect
)
```

## Computing FIG-C

FIG-C does not require ground truth on the eval split. For well-calibrated models, it
approximates FIG-V in expectation.

```python
from fig import fractional_information_gain_confidence

results = fractional_information_gain_confidence(
    y_pred_eval=y_pred_eval,
    item_id_eval=item_id_eval,
    student_id_eval=student_id_eval,
    y_train=y_train,
    item_id_train=item_id_train,
)

print(f"FIG-C (student-weighted): {results['fig_c']:.4f}")
print(f"FIG-C (observation-weighted): {results['fig_c_pooled']:.4f}")
```

!!! warning
    FIG-C is only trustworthy when the model is well-calibrated. A miscalibrated model
    can produce high FIG-C by being confidently wrong. See
    [Concepts](concepts.md#calibration-dependence) for details.

## Computing FIG-V

FIG-V additionally requires ground truth labels on the eval split:

```python
from fig import fractional_information_gain_validation

results = fractional_information_gain_validation(
    y_pred_eval=y_pred_eval,
    y_eval=y_eval,
    item_id_eval=item_id_eval,
    student_id_eval=student_id_eval,
    y_train=y_train,
    item_id_train=item_id_train,
)

print(f"FIG-V (student-weighted): {results['fig_v']:.4f}")
print(f"FIG-V (observation-weighted): {results['fig_v_pooled']:.4f}")
print(f"Number of students: {len(results['student_ids'])}")
```

## Understanding the Output

Both functions return typed dictionaries:

=== "FIG-C (FigCResult)"

    ```python
    {
        "fig_c": 0.18,             # Student-weighted FIG-C (macro average)
        "fig_c_pooled": 0.17,      # Observation-weighted FIG-C
        "fig_c_by_student": [...],  # Per-student FIG-C values (np.ndarray)
        "student_ids": [...],       # Student IDs in sorted order (np.ndarray)
    }
    ```

=== "FIG-V (FigVResult)"

    ```python
    {
        "fig_v": 0.15,             # Student-weighted FIG-V (macro average)
        "fig_v_pooled": 0.14,      # Observation-weighted FIG-V
        "fig_v_by_student": [...],  # Per-student FIG-V values (np.ndarray)
        "student_ids": [...],       # Student IDs in sorted order (np.ndarray)
        "calibration": None,        # CalibrationResult if calibration=True
    }
    ```

**`fig_v` / `fig_c`** (student-weighted): FIG computed per student, then averaged. Each
student gets equal weight. This is the primary metric.

**`fig_v_pooled` / `fig_c_pooled`** (observation-weighted): All observations pooled.
Students with more items contribute more. Equal to the student-weighted value when every
student answers the same number of items.

**`fig_v_by_student` / `fig_c_by_student`**: Per-student values, aligned with
`student_ids`.

## Optional Parameters

### Shrinkage

By default, per-item base rates are stabilized via empirical Bayes shrinkage (Beta(2, 2)
prior). You can disable or customize this:

```python
# Without shrinkage (raw item means)
results = fractional_information_gain_validation(
    ...,  # same args as above
    use_shrinkage=False,
)

# With weaker shrinkage
results = fractional_information_gain_validation(
    ...,  # same args as above
    alpha=1.0, beta=1.0,  # uniform prior (Laplace smoothing)
)
```

See [Concepts](concepts.md#baseline) for details on how the baseline is computed.

### Calibration Checking

FIG-V can optionally compute Expected Calibration Error (ECE):

```python
results = fractional_information_gain_validation(
    ...,  # same args as above
    calibration=True,  # default: 10 bins, warn if ECE > 0.1
)

calib = results["calibration"]
print(f"ECE: {calib['ece']:.4f}")
```

Checking ECE is especially useful before trusting FIG-C.

## Using FIG as a PyTorch Loss

The PyTorch module provides differentiable versions for use during training:

```python
import torch
from fig.fig_torch import fig_v_loss, precompute_baseline_torch

# Pre-compute baseline once (runs on CPU, no gradients needed)
baseline = precompute_baseline_torch(
    y_train=y_train,
    item_id_train=item_id_train,
)

# In your training loop:
for y_batch, item_ids, student_ids, x_batch in dataloader:
    optimizer.zero_grad()
    y_pred = model(x_batch)  # must be probabilities in [0, 1]

    loss = fig_v_loss(
        y_pred_eval=y_pred,
        y_eval=y_batch,
        item_id_eval=item_ids,
        student_id_eval=student_ids,
        baseline=baseline,  # reuse pre-computed baseline
    )
    loss.backward()
    optimizer.step()
```

`fig_v_loss` returns `-fractional_information_gain_validation_torch(...)["fig_v"]`.

Unlike standard BCE loss, FIG-V loss places **equal weight on each student** regardless
of how many items they answered.

!!! tip
    Always use `precompute_baseline_torch` to compute the baseline once before training.
    Passing `baseline=` avoids recomputing per-item statistics on every batch.

## Next Steps

- [Concepts](concepts.md) -- the math and intuition behind FIG
- [API Reference](api/fig.md) -- full parameter documentation
