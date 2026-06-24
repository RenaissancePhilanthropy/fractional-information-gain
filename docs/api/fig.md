# fig -- NumPy Implementation

The primary implementation of FIG-V and FIG-C. Functions in this module accept NumPy
arrays (or any array-like input) and return Python floats and NumPy arrays.

This module is the reference implementation. The PyTorch version in
[`fig.fig_torch`](fig_torch.md) mirrors this API but returns tensors and supports
gradient computation.

## Main Functions

### fractional_information_gain_validation

::: fig.fig.fractional_information_gain_validation

---

### fractional_information_gain_confidence

::: fig.fig.fractional_information_gain_confidence

## Result Types

### FigVResult

::: fig.fig.FigVResult

---

### FigCResult

::: fig.fig.FigCResult
