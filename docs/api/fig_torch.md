# fig.fig_torch -- PyTorch Implementation

Differentiable versions of FIG-V and FIG-C that accept and return PyTorch tensors.
Gradients flow through `y_pred_eval` only; all other inputs (training data, item/student
IDs) are treated as constants. Baseline computation always runs in NumPy.

The API mirrors [`fig.fig`](fig.md) with the following differences:

- Inputs can be tensors, NumPy arrays, or sequences.
- Outputs are tensors (except `student_ids`, which is always a NumPy array).
- A `baseline` parameter allows reusing pre-computed baselines across batches (strongly
  recommended -- use `precompute_baseline_torch` once before the training loop).
- Device is inferred from `y_pred_eval` if it is a tensor, or defaults to CPU.

!!! note
    Import this module directly -- it is not re-exported from the top-level `fig` package:
    ```python
    from fig.fig_torch import fractional_information_gain_validation_torch
    ```

## Main Functions

### fractional_information_gain_validation_torch

::: fig.fig_torch.fractional_information_gain_validation_torch

---

### fractional_information_gain_confidence_torch

::: fig.fig_torch.fractional_information_gain_confidence_torch

## Loss Wrappers

Convenience functions that negate the FIG metric for use as a minimization objective
with standard optimizers.

### fig_v_loss

::: fig.fig_torch.fig_v_loss

---

### fig_c_loss

::: fig.fig_torch.fig_c_loss

## Baseline Pre-computation

### precompute_baseline_torch

::: fig.fig_torch.precompute_baseline_torch

## Result Types

### FigVTorchResult

::: fig.fig_torch.FigVTorchResult

---

### FigCTorchResult

::: fig.fig_torch.FigCTorchResult
