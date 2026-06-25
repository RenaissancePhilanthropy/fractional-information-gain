# fractional-information-gain

**Fractional Information Gain (FIG)** is a performance metric for knowledge-tracing models
in educational assessment. This library implements it in Python (NumPy), PyTorch, and R.

Knowledge-tracing models are commonly evaluated with metrics like AUC, accuracy, and F1.
These metrics can be dominated by item difficulty: a hard item is hard for everyone, an
easy item is easy for everyone. FIG isolates the part of a model's performance that
reflects actual student-level knowledge, by comparing predictions against per-item
baselines computed from training data. This library currently handles binary
(correct/incorrect) response data.

FIG = 0 means the model adds nothing beyond item base rates. FIG = 1 means perfect
predictions. FIG < 0 means the model is worse than the baseline.

FIG comes in two variants, both of which require training data to compute per-item
baselines (success rates):

- **FIG-C** (Confidence) — measures whether the model is *confident*. Does not need
  evaluation-set ground truth. For well-calibrated models, FIG-C approximates FIG-V in
  expectation.
- **FIG-V** (Validation) — measures whether the model is confident *and correct*.
  Requires ground truth labels on the evaluation set. Can also be used as a differentiable
  training loss (PyTorch).

Both metrics are reported as observation-weighted (pooled) and student-weighted
(macro-average) variants.

## Installation

```bash
pip install fractional-information-gain
```

The PyTorch (differentiable) implementation is an optional extra:

```bash
pip install "fractional-information-gain[torch]"
```

## Quick example

The package is imported as `fig`:

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
differentiable training loss. The R package (`r/` subdirectory) mirrors the Python API.

## Development

The project uses [uv](https://docs.astral.sh/uv/) for dependency management:

```bash
uv sync --dev --extra torch
uv run pytest
```

To build the docs locally:

```bash
uv sync --dev --extra docs --extra torch
uv run mkdocs serve
```

## Releasing

Releases are published to PyPI by the `.github/workflows/publish.yml` workflow using
[PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC) — no API
token is stored. One-time setup is required before the first release: register a
"pending publisher" on PyPI (and TestPyPI for dry runs) and create the matching GitHub
Environments; the exact values are documented in the header of `publish.yml`.

To release: bump `version` in `pyproject.toml`, then publish a GitHub Release tagged
`vX.Y.Z`. The workflow builds the sdist/wheel and uploads them to PyPI. Running the
workflow manually (Actions → Publish → Run workflow) instead uploads to TestPyPI for a
dry run.
