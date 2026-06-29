# Installation

## Requirements

- Python 3.12 through 3.14
- NumPy 2.4+

For the differentiable PyTorch implementation, PyTorch 2.10+ is also required.

## Install

```bash
pip install fractional-information-gain
```

With PyTorch support (for differentiable loss functions):

```bash
pip install "fractional-information-gain[torch]"
```

To install the latest development version directly from GitHub instead:

```bash
pip install git+https://github.com/RenaissancePhilanthropy/fractional-information-gain.git
```

## Import Name

The package is installed as `fractional-information-gain` but imported as `fig`:

```python
import fig
# or
from fig import fractional_information_gain_validation
```

The PyTorch module is not re-exported from the top-level package. Import it directly:

```python
from fig.fig_torch import fractional_information_gain_validation_torch
```

## Development Setup

The project uses [uv](https://docs.astral.sh/uv/) for dependency management:

```bash
git clone https://github.com/RenaissancePhilanthropy/fractional-information-gain.git
cd fractional-information-gain
uv sync --dev --extra torch
```

This installs all development dependencies (pytest, ruff, pre-commit) and the PyTorch
extra.

To run the test suite:

```bash
uv run python -m pytest tests/
```

## R Package

The R implementation (`figmetric`) lives in the `r/` subdirectory. See the
[R Package](r-package.md) page for installation instructions.
