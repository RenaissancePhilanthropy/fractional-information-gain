"""PyTorch implementation of Fractional Information Gain (FIG-V and FIG-C).

This module provides differentiable versions of FIG that can be used as loss functions
during model training. The API mirrors fig.py but accepts and returns torch tensors.

- FIG-V (Validation): Requires ground truth, uses cross-entropy
- FIG-C (Confidence): No ground truth needed, uses entropy (model confidence)

For users without PyTorch, use the numpy version in fig.py instead.
"""

# PyTorch's type stubs expose public symbols (tensor, float32, all, etc.) via
# wildcard imports from private submodules (torch._C._VariableFunctions) and
# build __all__ dynamically at runtime. Pyright cannot evaluate this statically,
# so it flags every use of torch.tensor, torch.float32, etc. as a private import.
# See https://github.com/pytorch/pytorch/issues/134985
# Remove this override if PyTorch ships a proper __init__.pyi or static __all__.
# pyright: reportPrivateImportUsage=false

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypedDict

import numpy as np
import torch

from fig._utils import (
    BaselineData,
    CalibrationResult,
    TrainData,
    check_calibration,
    compute_baseline,
)


def _to_tensor(
    x: torch.Tensor | np.ndarray | Sequence[Any],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Coerce an array-like or tensor to a flat tensor with the given dtype and device.

    If x is already a torch.Tensor it is cast and moved via .to(); otherwise a new
    tensor is created from x. The result is always 1-D (flattened).

    Parameters
    ----------
    x : array-like or torch.Tensor
        Input to coerce.
    dtype : torch.dtype
        Target dtype (e.g. torch.float32, torch.long).
    device : torch.device
        Target device.

    Returns
    -------
    torch.Tensor
        1-D tensor with the requested dtype on the requested device.
    """
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype, device=device).flatten()
    return torch.tensor(x, dtype=dtype, device=device).flatten()


def _to_numpy(x: torch.Tensor | np.ndarray | Sequence[Any]) -> np.ndarray:
    """Coerce an array-like or tensor to a flat numpy array.

    If x is a torch.Tensor, gradients are detached and data is moved to CPU before
    conversion. Otherwise np.asarray is used directly. Dtype is preserved from input.

    Parameters
    ----------
    x : array-like or torch.Tensor
        Input to coerce.

    Returns
    -------
    np.ndarray
        1-D numpy array. Dtype is preserved from input.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().ravel()
    return np.asarray(x).ravel()


def _binary_entropy_torch(p: torch.Tensor) -> torch.Tensor:
    """Compute binary entropy ``H(p) = -[p*log(p) + (1-p)*log(1-p)]``.

    Parameters
    ----------
    p : torch.Tensor
        Probabilities (should already be clamped to avoid log(0)).

    Returns
    -------
    torch.Tensor
        Binary entropy for each probability.
    """
    return -(p * torch.log(p) + (1 - p) * torch.log(1 - p))


@dataclass(frozen=True)
class EvalDataTorch:
    """Validated, encoded eval observations for PyTorch FIG computation.

    Produced by prepare_eval_data_torch(). Do not construct manually.

    Attributes
    ----------
    y_pred : torch.Tensor
        float32, clipped to [eps, 1-eps], shape (n_eval,). Retains gradients
        if the input tensor required them.
    y : torch.Tensor or None
        float32, binary {0, 1}, shape (n_eval,). None for FIG-C.
    encoded_item_id : np.ndarray
        int64, shape (n_eval,). Values in 0..N where N is the sentinel for
        items unseen in training.
    encoded_student_id : torch.Tensor
        int64, shape (n_eval,). Values in 0..S-1.
    student_labels : np.ndarray
        Original student IDs in sorted order, length S. Used to reconstruct
        the student_ids key in the output dict.
    """

    y_pred: torch.Tensor
    y: torch.Tensor | None
    encoded_item_id: np.ndarray
    encoded_student_id: torch.Tensor
    student_labels: np.ndarray


def prepare_eval_data_torch(
    y_pred: torch.Tensor | np.ndarray | Sequence[float],
    y: torch.Tensor | np.ndarray | Sequence[float] | None,
    item_id: torch.Tensor | np.ndarray | Sequence[Any],
    student_id: torch.Tensor | np.ndarray | Sequence[Any],
    baseline: BaselineData,
    eps: float = 1e-12,
    device: torch.device | None = None,
) -> EvalDataTorch:
    """Validate, encode, and coerce eval inputs to an EvalDataTorch.

    Encodes item_id against the vocabulary in baseline, encodes student_id
    independently, clips y_pred, and places tensors on device.

    Parameters
    ----------
    y_pred : array-like or torch.Tensor
        Predicted probabilities (1D). Values must be in [0, 1]. This tensor
        retains gradients if requires_grad=True.
    y : array-like or torch.Tensor or None
        Ground truth (1D), binary. None for FIG-C.
    item_id : array-like or torch.Tensor
        Item identifiers for eval observations (1D). Any sortable type.
    student_id : array-like or torch.Tensor
        Student identifiers for eval observations (1D). Any sortable type.
    baseline : BaselineData
        Pre-computed baseline from compute_baseline().
    eps : float, default=1e-12
        Clipping constant for y_pred.
    device : torch.device, optional
        Target device. Inferred from y_pred if it is already a tensor.

    Returns
    -------
    EvalDataTorch
    """
    if device is None:
        device = (
            y_pred.device if isinstance(y_pred, torch.Tensor) else torch.device("cpu")
        )

    # Coerce y_pred to tensor, validate, then clip.
    y_pred_t = _to_tensor(y_pred, torch.float32, device)

    n_eval = len(y_pred_t)
    if n_eval == 0:
        raise ValueError("y_pred must not be empty")
    if not torch.all(torch.isfinite(y_pred_t)):
        raise ValueError("y_pred contains NaN or inf values")
    if not (torch.all(y_pred_t >= 0) and torch.all(y_pred_t <= 1)):
        raise ValueError("y_pred values must be in [0, 1]")
    y_pred_t = torch.clamp(y_pred_t, eps, 1 - eps)

    # Encode item_id against the baseline vocabulary (numpy, no gradients needed).
    item_id_np = _to_numpy(item_id)
    if len(item_id_np) != n_eval:
        raise ValueError(
            f"Length mismatch: item_id ({len(item_id_np)}) != y_pred ({n_eval})"
        )
    n_items = len(baseline.item_vocab)
    encoded_item = np.fromiter(
        (baseline.item_vocab.get(x, n_items) for x in item_id_np),
        dtype=np.int64,
        count=n_eval,
    )

    # Encode student_id independently from eval.
    student_id_np = _to_numpy(student_id)
    if len(student_id_np) != n_eval:
        raise ValueError(
            f"Length mismatch: student_id ({len(student_id_np)}) != y_pred ({n_eval})"
        )
    student_labels, encoded_student = np.unique(student_id_np, return_inverse=True)
    encoded_student_t = torch.tensor(encoded_student, dtype=torch.long, device=device)

    # Coerce and validate y if provided.
    y_t: torch.Tensor | None = None
    if y is not None:
        y_t = _to_tensor(y, torch.float32, device)
        if len(y_t) != n_eval:
            raise ValueError(f"Length mismatch: y ({len(y_t)}) != y_pred ({n_eval})")
        unique_y = set(np.unique(y_t.detach().cpu().numpy()))
        bad_y = unique_y - {0.0, 1.0}
        if bad_y:
            raise ValueError(f"y must contain only 0 and 1, got: {bad_y}")

    return EvalDataTorch(
        y_pred=y_pred_t,
        y=y_t,
        encoded_item_id=encoded_item,
        encoded_student_id=encoded_student_t,
        student_labels=student_labels,
    )


def _compute_fig_torch(
    numerator: torch.Tensor,
    entropy_baseline: torch.Tensor,
    student_id_eval: torch.Tensor,
    metric_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shared aggregation core for FIG-C and FIG-V (torch version).

    Analogous to _compute_fig in fig.py. The two FIG variants differ only in
    how numerator is computed: FIG-C uses ``H(y_pred)``; FIG-V uses ``BCE(y_pred, y)``.

    Parameters
    ----------
    numerator : torch.Tensor
        Per-observation entropy array (1D, float32).
        For FIG-C: binary entropy of model predictions.
        For FIG-V: binary cross-entropy of predictions against ground truth.
    entropy_baseline : torch.Tensor
        Per-observation baseline entropy (1D, float32). Computed from
        train-derived item base rates.
    student_id_eval : torch.Tensor
        Encoded student IDs for eval observations (1D, int64). Values in 0..S-1.
    metric_name : str
        Name used in warning messages, e.g. "FIG-C" or "FIG-V".

    Returns
    -------
    tuple
        (fig_pooled, fig, fig_by_student) where:
        - fig_pooled: observation-weighted FIG (scalar tensor)
        - fig: student-weighted FIG (scalar tensor)
        - fig_by_student: per-student FIG values (1D tensor)
    """
    device = numerator.device
    fig_pooled = 1 - numerator.sum() / entropy_baseline.sum()

    n_students = int(student_id_eval.max().item()) + 1

    student_numerator_sum = torch.zeros(n_students, dtype=torch.float32, device=device)
    student_baseline_sum = torch.zeros(n_students, dtype=torch.float32, device=device)
    student_numerator_sum.scatter_add_(0, student_id_eval, numerator)
    student_baseline_sum.scatter_add_(0, student_id_eval, entropy_baseline)

    fig_by_student = 1 - (student_numerator_sum / student_baseline_sum)
    fig = fig_by_student.mean()

    if fig_pooled > 1 + 1e-6:
        warnings.warn(
            f"{metric_name} pooled {fig_pooled.item():.3f} > 1; "
            f"possible leakage or model issues.",
            stacklevel=2,
        )
    if fig > 1 + 1e-6:
        warnings.warn(
            f"{metric_name} {fig.item():.3f} > 1; possible leakage or model issues.",
            stacklevel=2,
        )

    return fig_pooled, fig, fig_by_student


def precompute_baseline_torch(
    y_train: torch.Tensor | np.ndarray | Sequence[float],
    item_id_train: torch.Tensor | np.ndarray | Sequence[Any],
    use_shrinkage: bool = True,
    alpha: float = 2.0,
    beta: float = 2.0,
) -> BaselineData:
    """Pre-compute baseline statistics from training data for reuse across calls.

    This avoids recomputing per-item baselines on every call to
    ``fractional_information_gain_validation_torch`` or
    ``fractional_information_gain_confidence_torch``. Useful when evaluating the
    same training set against multiple eval batches (e.g., during a training loop).

    The computations in this function always happen on the CPU.

    Parameters
    ----------
    y_train : array-like or torch.Tensor
        Ground truth on train split. Shape: (n_train,)
    item_id_train : array-like or torch.Tensor
        Item identifiers for train observations. Shape: (n_train,)
    use_shrinkage : bool, default=True
        If True, apply empirical Bayes shrinkage to item baselines.
    alpha : float, default=2.0
        Prior pseudo-count for successes in shrinkage.
    beta : float, default=2.0
        Prior pseudo-count for failures in shrinkage.

    Returns
    -------
    BaselineData
        Pre-computed baseline. Pass this as the ``baseline`` argument to the
        FIG-V/FIG-C functions to avoid recomputing per training batch.
    """
    y_train = _to_numpy(y_train)
    item_id_train = _to_numpy(item_id_train)
    train = TrainData(y=y_train, item_id=item_id_train)
    return compute_baseline(train, use_shrinkage=use_shrinkage, alpha=alpha, beta=beta)


class FigCTorchResult(TypedDict):
    """Return type of ``fractional_information_gain_confidence_torch``.

    Attributes
    ----------
    fig_c_pooled : torch.Tensor
        Observation-weighted FIG-C (scalar tensor).
    fig_c : torch.Tensor
        Student-weighted FIG-C (scalar tensor).
    fig_c_by_student : torch.Tensor
        Per-student FIG-C values, aligned with ``student_ids``.
    student_ids : np.ndarray
        Unique student IDs in sorted order.
    """

    fig_c_pooled: torch.Tensor
    fig_c: torch.Tensor
    fig_c_by_student: torch.Tensor
    student_ids: np.ndarray


def fractional_information_gain_confidence_torch(
    y_pred_eval: torch.Tensor | np.ndarray | Sequence[float],
    item_id_eval: torch.Tensor | np.ndarray | Sequence[Any],
    student_id_eval: torch.Tensor | np.ndarray | Sequence[Any],
    y_train: torch.Tensor | np.ndarray | Sequence[float] | None = None,
    item_id_train: torch.Tensor | np.ndarray | Sequence[Any] | None = None,
    eps: float = 1e-12,
    use_shrinkage: bool = True,
    alpha: float = 2.0,
    beta: float = 2.0,
    baseline: BaselineData | None = None,
) -> FigCTorchResult:
    """Compute FIG-C (Confidence) using PyTorch.

    FIG-C measures the reduction in uncertainty about student responses based on
    model confidence, without requiring ground truth labels. Differentiable
    with respect to ``y_pred_eval``.

    ``FIG-C = 1 - H(y_pred) / H(baseline[item_id])``

    where:

    - ``H(baseline[item_id])`` = sum of binary entropies of per-item base rates
      (prior uncertainty), looked up by item ID
    - ``H(y_pred)`` = sum of binary entropies of model predictions
      (remaining uncertainty)
    - ``H(p) = -[p*log(p) + (1-p)*log(1-p)]``

    Parameters
    ----------
    y_pred_eval : array-like or torch.Tensor
        Predicted probabilities (0 to 1). Shape: (n_eval,)
        This is the only input that retains gradients.
    item_id_eval : array-like or torch.Tensor
        Item identifiers for eval observations. Shape: (n_eval,)
    student_id_eval : array-like or torch.Tensor
        Student identifiers for eval observations. Shape: (n_eval,)
    y_train : array-like or torch.Tensor, optional
        Ground truth on train split. Shape: (n_train,)
        Required if baseline is not provided.
    item_id_train : array-like or torch.Tensor, optional
        Item identifiers for train observations. Shape: (n_train,)
        Required if baseline is not provided.
    eps : float, default=1e-12
        Small constant for clipping probabilities to avoid log(0).
    use_shrinkage : bool, default=True
        If True, apply empirical Bayes shrinkage to item baselines.
    alpha : float, default=2.0
        Prior pseudo-count for successes in shrinkage.
    beta : float, default=2.0
        Prior pseudo-count for failures in shrinkage.
    baseline : BaselineData, optional
        Pre-computed baseline from ``precompute_baseline_torch()``.
        If provided, ``y_train``, ``item_id_train``, ``use_shrinkage``, ``alpha``,
        and ``beta`` are ignored. If not provided, ``y_train`` and ``item_id_train``
        are required and the baseline is computed on each call.

    Returns
    -------
    FigCTorchResult
        Dictionary with keys:

        - ``fig_c_pooled``: Observation-weighted FIG-C (scalar tensor)
        - ``fig_c``: Student-weighted FIG-C (scalar tensor)
        - ``fig_c_by_student``: Per-student FIG-C values (1D tensor)
        - ``student_ids``: Unique student IDs in sorted order (np.ndarray)

    Notes
    -----
    - ``FIG-C = 0`` when model predictions match item base rates (no information gain).
    - ``FIG-C = 1`` when model is perfectly confident (entropy = 0).
    - FIG-C can be negative if model adds uncertainty beyond the baseline.
    - For well-calibrated models, FIG-C approximates FIG-V in expectation.
    - Items in eval that don't appear in train use the global training mean as baseline.
    """
    if baseline is None:
        msg = "Either baseline or both y_train and item_id_train must be provided."
        if y_train is None:
            raise ValueError(msg)
        if item_id_train is None:
            raise ValueError(msg)
        baseline = precompute_baseline_torch(
            y_train,
            item_id_train,
            use_shrinkage=use_shrinkage,
            alpha=alpha,
            beta=beta,
        )

    eval_data = prepare_eval_data_torch(
        y_pred=y_pred_eval,
        y=None,
        item_id=item_id_eval,
        student_id=student_id_eval,
        baseline=baseline,
        eps=eps,
    )

    # Look up per-eval baseline probabilities and compute entropy.
    baseline_probs = torch.tensor(
        np.clip(baseline.item_baselines[eval_data.encoded_item_id], eps, 1 - eps),
        dtype=torch.float32,
        device=eval_data.y_pred.device,
    )
    entropy_baseline = _binary_entropy_torch(baseline_probs)
    entropy_model = _binary_entropy_torch(eval_data.y_pred)

    fig_c_pooled, fig_c, fig_c_by_student = _compute_fig_torch(
        entropy_model,
        entropy_baseline,
        eval_data.encoded_student_id,
        "FIG-C",
    )

    return {
        "fig_c_pooled": fig_c_pooled,
        "fig_c": fig_c,
        "fig_c_by_student": fig_c_by_student,
        "student_ids": eval_data.student_labels,
    }


def fig_c_loss(*args: Any, **kwargs: Any) -> torch.Tensor:  # noqa: ANN401
    """Return FIG-C negated for use as a minimization loss.

    Equivalent to: ``-fractional_information_gain_confidence_torch(...)['fig_c']``
    """
    return -fractional_information_gain_confidence_torch(*args, **kwargs)["fig_c"]


def _check_calibration_torch(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    n_bins: int,
    warn_threshold: float,
) -> CalibrationResult:
    """Compute Expected Calibration Error (ECE) using torch tensors.

    Parameters
    ----------
    y_pred : torch.Tensor
        Predicted probabilities (1D).
    y_true : torch.Tensor
        True binary outcomes (1D).
    n_bins : int
        Number of bins for binning predictions.
    warn_threshold : float
        Warn if ECE exceeds this value.

    Returns
    -------
    CalibrationResult
        Calibration metrics. See check_calibration for key descriptions.
    """
    return check_calibration(
        y_pred.detach().cpu().numpy(),
        y_true.detach().cpu().numpy(),
        n_bins,
        warn_threshold,
    )


class FigVTorchResult(TypedDict):
    """Return type of ``fractional_information_gain_validation_torch``.

    Attributes
    ----------
    fig_v_pooled : torch.Tensor
        Observation-weighted FIG-V (scalar tensor).
    fig_v : torch.Tensor
        Student-weighted FIG-V (scalar tensor).
    fig_v_by_student : torch.Tensor
        Per-student FIG-V values, aligned with ``student_ids``.
    student_ids : np.ndarray
        Unique student IDs in sorted order.
    calibration : CalibrationResult or None
        Calibration metrics. None unless ``calibration=True`` was passed.
    """

    fig_v_pooled: torch.Tensor
    fig_v: torch.Tensor
    fig_v_by_student: torch.Tensor
    student_ids: np.ndarray
    calibration: CalibrationResult | None


def fractional_information_gain_validation_torch(
    y_pred_eval: torch.Tensor | np.ndarray | Sequence[float],
    y_eval: torch.Tensor | np.ndarray | Sequence[float],
    item_id_eval: torch.Tensor | np.ndarray | Sequence[Any],
    student_id_eval: torch.Tensor | np.ndarray | Sequence[Any],
    y_train: torch.Tensor | np.ndarray | Sequence[float] | None = None,
    item_id_train: torch.Tensor | np.ndarray | Sequence[Any] | None = None,
    eps: float = 1e-12,
    use_shrinkage: bool = True,
    alpha: float = 2.0,
    beta: float = 2.0,
    calibration: bool = False,
    calib_n_bins: int = 10,
    calib_warn_threshold: float = 0.1,
    baseline: BaselineData | None = None,
) -> FigVTorchResult:
    """Compute FIG-V (Validation) using PyTorch.

    FIG-V measures how much better a model's predictions are than per-item base
    rates, using cross-entropy against ground truth. Differentiable with respect
    to ``y_pred_eval``.

    ``FIG-V = 1 - BCE(y_pred, y) / H(baseline[item_id])``

    where:

    - ``H(baseline[item_id])`` = sum of binary entropies of per-item base rates
      (prior uncertainty), looked up by item ID
    - ``BCE(y_pred, y)`` = sum of binary cross-entropies of model predictions
      against ground truth
    - ``BCE(p, y) = -[y*log(p) + (1-y)*log(1-p)]``

    Parameters
    ----------
    y_pred_eval : array-like or torch.Tensor
        Predicted probabilities (0 to 1). Shape: (n_eval,)
        This is the only input that retains gradients.
    y_eval : array-like or torch.Tensor
        Ground truth on eval split. Shape: (n_eval,)
        Values: 1=correct, 0=incorrect.
    item_id_eval : array-like or torch.Tensor
        Item identifiers for eval observations. Shape: (n_eval,)
    student_id_eval : array-like or torch.Tensor
        Student identifiers for eval observations. Shape: (n_eval,)
    y_train : array-like or torch.Tensor, optional
        Ground truth on train split. Shape: (n_train,)
        Required if baseline is not provided.
    item_id_train : array-like or torch.Tensor, optional
        Item identifiers for train observations. Shape: (n_train,)
        Required if baseline is not provided.
    eps : float, default=1e-12
        Small constant for clipping probabilities to avoid log(0).
    use_shrinkage : bool, default=True
        If True, apply empirical Bayes shrinkage to item baselines.
    alpha : float, default=2.0
        Prior pseudo-count for successes in shrinkage.
    beta : float, default=2.0
        Prior pseudo-count for failures in shrinkage.
    calibration : bool, default=False
        If True, attach calibration metrics (ECE) to output.
    calib_n_bins : int, default=10
        Number of bins for ECE computation.
    calib_warn_threshold : float, default=0.1
        Warn if ECE exceeds this value.
    baseline : BaselineData, optional
        Pre-computed baseline from ``precompute_baseline_torch()``.
        If provided, ``y_train``, ``item_id_train``, ``use_shrinkage``, ``alpha``,
        and ``beta`` are ignored. If not provided, ``y_train`` and ``item_id_train``
        are required and the baseline is computed on each call.

    Returns
    -------
    FigVTorchResult
        Dictionary with keys:

        - ``fig_v_pooled``: Observation-weighted FIG-V (scalar tensor)
        - ``fig_v``: Student-weighted FIG-V (scalar tensor)
        - ``fig_v_by_student``: Per-student FIG-V values (1D tensor)
        - ``student_ids``: Unique student IDs in sorted order (np.ndarray)
        - ``calibration``: None unless ``calibration=True``, in which case a
          dict of ECE metrics from ``_check_calibration_torch``

    Notes
    -----
    - ``FIG-V = 0`` when model predictions match item base rates (no information gain).
    - ``FIG-V = 1`` when model predictions are perfect (zero cross-entropy).
    - FIG-V can be negative if model is worse than the baseline.
    - ``FIG-V > 1`` usually indicates data leakage; the library warns when this happens.
    - Items in eval that don't appear in train use the global training mean as baseline.
    """
    if baseline is None:
        msg = "Either baseline or both y_train and item_id_train must be provided."
        if y_train is None:
            raise ValueError(msg)
        if item_id_train is None:
            raise ValueError(msg)
        baseline = precompute_baseline_torch(
            y_train,
            item_id_train,
            use_shrinkage=use_shrinkage,
            alpha=alpha,
            beta=beta,
        )

    eval_data = prepare_eval_data_torch(
        y_pred=y_pred_eval,
        y=y_eval,
        item_id=item_id_eval,
        student_id=student_id_eval,
        baseline=baseline,
        eps=eps,
    )

    # Look up per-eval baseline probabilities and compute entropy.
    baseline_probs = torch.tensor(
        np.clip(baseline.item_baselines[eval_data.encoded_item_id], eps, 1 - eps),
        dtype=torch.float32,
        device=eval_data.y_pred.device,
    )

    # Bernoulli cross-entropy: -[y*log(p) + (1-y)*log(1-p)]
    assert eval_data.y is not None, (
        "eval_data.y must not be None for FIG-V (this is a bug)"
    )
    ce_model = -(
        eval_data.y * torch.log(eval_data.y_pred)
        + (1 - eval_data.y) * torch.log(1 - eval_data.y_pred)
    )
    entropy_baseline = _binary_entropy_torch(baseline_probs)

    fig_v_pooled, fig_v, fig_v_by_student = _compute_fig_torch(
        ce_model,
        entropy_baseline,
        eval_data.encoded_student_id,
        "FIG-V",
    )

    result: FigVTorchResult = {
        "fig_v_pooled": fig_v_pooled,
        "fig_v": fig_v,
        "fig_v_by_student": fig_v_by_student,
        "student_ids": eval_data.student_labels,
        "calibration": None,
    }

    if calibration:
        assert eval_data.y is not None, (
            "eval_data.y must not be None for calibration (this is a bug)"
        )
        result["calibration"] = _check_calibration_torch(
            eval_data.y_pred, eval_data.y, calib_n_bins, calib_warn_threshold
        )

    return result


def fig_v_loss(*args: Any, **kwargs: Any) -> torch.Tensor:  # noqa: ANN401
    """Return FIG-V negated for use as a minimization loss.

    Equivalent to: ``-fractional_information_gain_validation_torch(...)['fig_v']``
    """
    return -fractional_information_gain_validation_torch(*args, **kwargs)["fig_v"]


# TODO: This example should probably be moved somewhere else.
if __name__ == "__main__":
    # Example usage
    torch.manual_seed(42)  # pyright: ignore[reportUnknownMemberType]

    n_train = 3000
    n_eval = 1000
    n_items = 20
    n_students = 100

    # Training data
    y_train = torch.randint(0, 2, (n_train,)).float()
    item_id_train = torch.randint(1, n_items + 1, (n_train,))

    # Eval data
    y_eval = torch.randint(0, 2, (n_eval,)).float()
    item_id_eval = torch.randint(1, n_items + 1, (n_eval,))
    student_id_eval = torch.randint(1, n_students + 1, (n_eval,))

    # Predictions (with gradient tracking)
    y_pred_eval = torch.sigmoid(torch.randn(n_eval, requires_grad=True))

    # Compute FIG-V (requires ground truth)
    results_v = fractional_information_gain_validation_torch(
        y_pred_eval=y_pred_eval,
        y_eval=y_eval,
        item_id_eval=item_id_eval,
        student_id_eval=student_id_eval,
        y_train=y_train,
        item_id_train=item_id_train,
    )

    print("FIG-V Results (PyTorch, requires ground truth):")  # noqa: T201
    print(f"  FIG-V pooled: {results_v['fig_v_pooled'].item():.4f}")  # noqa: T201
    print(f"  FIG-V: {results_v['fig_v'].item():.4f}")  # noqa: T201

    # Test gradient flow for FIG-V
    loss_v = -results_v["fig_v"]
    loss_v.backward()  # pyright: ignore[reportUnknownMemberType]
    print(f"\n  FIG-V Gradient exists: {y_pred_eval.grad is not None}")  # noqa: T201
    assert y_pred_eval.grad is not None
    print(  # noqa: T201
        f"  FIG-V Gradient mean abs: {y_pred_eval.grad.abs().mean().item():.6f}"
    )

    # Reset gradients for FIG-C test
    y_pred_eval_c = torch.sigmoid(torch.randn(n_eval, requires_grad=True))

    # Compute FIG-C (does NOT require ground truth)
    results_c = fractional_information_gain_confidence_torch(
        y_pred_eval=y_pred_eval_c,
        item_id_eval=item_id_eval,
        student_id_eval=student_id_eval,
        y_train=y_train,
        item_id_train=item_id_train,
    )

    print("\nFIG-C Results (PyTorch, no ground truth needed):")  # noqa: T201
    print(f"  FIG-C pooled: {results_c['fig_c_pooled'].item():.4f}")  # noqa: T201
    print(f"  FIG-C: {results_c['fig_c'].item():.4f}")  # noqa: T201

    # Test gradient flow for FIG-C
    loss_c = -results_c["fig_c"]
    loss_c.backward()  # pyright: ignore[reportUnknownMemberType]
    print(f"\n  FIG-C Gradient exists: {y_pred_eval_c.grad is not None}")  # noqa: T201
    assert y_pred_eval_c.grad is not None
    print(  # noqa: T201
        f"  FIG-C Gradient mean abs: {y_pred_eval_c.grad.abs().mean().item():.6f}"
    )

    print(  # noqa: T201
        "\nNote: For well-calibrated models, FIG-C approximates FIG-V in expectation."
    )
