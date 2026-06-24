"""NumPy implementation of Fractional Information Gain (FIG-V and FIG-C).

This module provides the primary, non-differentiable implementations of FIG.
For a differentiable PyTorch version suitable for use as a training loss,
see fig_torch.py.
"""

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypedDict

import numpy as np

from fig._utils import (
    BaselineData,
    CalibrationResult,
    TrainData,
    binary_entropy,
    check_calibration,
    compute_baseline,
)


@dataclass(frozen=True)
class EvalData:
    """Validated, encoded eval observations for numpy FIG computation.

    Produced by prepare_eval_data(). Do not construct manually.

    Attributes
    ----------
    y_pred : np.ndarray
        float64, clipped to [eps, 1-eps], shape (n_eval,).
    y : np.ndarray or None
        float64, binary {0, 1}, shape (n_eval,). None for FIG-C.
    encoded_item_id : np.ndarray
        int64, shape (n_eval,). Values in 0..N where N is the sentinel for
        items unseen in training.
    encoded_student_id : np.ndarray
        int64, shape (n_eval,). Values in 0..S-1.
    student_labels : np.ndarray
        Original student IDs in sorted order, length S. Used to reconstruct
        the student_ids key in the output dict.
    """

    y_pred: np.ndarray
    y: np.ndarray | None
    encoded_item_id: np.ndarray
    encoded_student_id: np.ndarray
    student_labels: np.ndarray


def prepare_eval_data(
    y_pred: np.ndarray | Sequence[float],
    y: np.ndarray | Sequence[float] | None,
    item_id: np.ndarray | Sequence[Any],
    student_id: np.ndarray | Sequence[Any],
    baseline: BaselineData,
    eps: float = 1e-12,
) -> EvalData:
    """Validate, encode, and coerce eval inputs to an EvalData.

    Encodes item_id against the vocabulary in baseline, encodes student_id
    independently, clips y_pred, and validates y if provided.

    Parameters
    ----------
    y_pred : array-like
        Predicted probabilities (1D). Values must be in [0, 1].
    y : array-like or None
        Ground truth (1D), binary. None for FIG-C.
    item_id : array-like
        Item identifiers for eval observations (1D). Any sortable type.
    student_id : array-like
        Student identifiers for eval observations (1D). Any sortable type.
    baseline : BaselineData
        Pre-computed baseline from compute_baseline().
    eps : float, default=1e-12
        Clipping constant for y_pred.

    Returns
    -------
    EvalData
    """
    if not isinstance(baseline, BaselineData):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(
            f"baseline must be a BaselineData produced by compute_baseline(), "
            f"got {type(baseline).__name__}"
        )
    y_pred_arr = np.asarray(y_pred, dtype=np.float64).ravel()
    n_eval = len(y_pred_arr)
    if n_eval == 0:
        raise ValueError("y_pred must not be empty")
    if not np.all(np.isfinite(y_pred_arr)):
        raise ValueError("y_pred contains NaN or inf values")
    if not (np.all(y_pred_arr >= 0) and np.all(y_pred_arr <= 1)):
        raise ValueError("y_pred values must be in [0, 1]")
    y_pred_arr = np.clip(y_pred_arr, eps, 1 - eps)

    item_id_np = np.asarray(item_id).ravel()
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

    student_id_np = np.asarray(student_id).ravel()
    if len(student_id_np) != n_eval:
        raise ValueError(
            f"Length mismatch: student_id ({len(student_id_np)}) != y_pred ({n_eval})"
        )
    student_labels, encoded_student = np.unique(student_id_np, return_inverse=True)
    encoded_student = encoded_student.astype(np.int64)

    y_arr: np.ndarray | None = None
    if y is not None:
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        if len(y_arr) != n_eval:
            raise ValueError(f"Length mismatch: y ({len(y_arr)}) != y_pred ({n_eval})")
        unique_y = set(np.unique(y_arr))
        bad = unique_y - {0.0, 1.0}
        if bad:
            raise ValueError(f"y must contain only 0 and 1, got: {bad}")

    return EvalData(
        y_pred=y_pred_arr,
        y=y_arr,
        encoded_item_id=encoded_item,
        encoded_student_id=encoded_student,
        student_labels=student_labels,
    )


def _compute_fig(
    numerator: np.ndarray,
    entropy_baseline: np.ndarray,
    student_id_eval: np.ndarray,
    metric_name: str,
) -> tuple[float, float, np.ndarray]:
    """Compute pooled and student-weighted FIG from per-observation entropy arrays.

    This is the shared aggregation core used by both FIG-C and FIG-V. The two
    variants differ only in how `numerator` is computed: FIG-C uses the binary
    entropy of model predictions; FIG-V uses the binary cross-entropy of
    predictions against ground truth.

    Parameters
    ----------
    numerator : np.ndarray
        Per-observation entropy array (1D, float64).
        For FIG-C: ``H(y_pred)``. For FIG-V: ``BCE(y_pred, y)``.
    entropy_baseline : np.ndarray
        Per-observation baseline entropy (1D, float64). Computed from
        train-derived item base rates.
    student_id_eval : np.ndarray
        Encoded student IDs for eval observations (1D, integer). Values in 0..S-1.
    metric_name : str
        Name used in warning messages, e.g. "FIG-C" or "FIG-V".

    Returns
    -------
    tuple
        (fig_pooled, fig, fig_by_student) where:
        - fig_pooled: observation-weighted FIG (float)
        - fig: student-weighted FIG (float)
        - fig_by_student: per-student FIG values (np.ndarray)
    """
    fig_pooled = 1 - numerator.sum() / entropy_baseline.sum()

    student_numerator_sum = np.bincount(student_id_eval, weights=numerator)
    student_baseline_sum = np.bincount(student_id_eval, weights=entropy_baseline)

    fig_by_student = 1 - (student_numerator_sum / student_baseline_sum)
    fig = fig_by_student.mean()

    if fig_pooled > 1 + 1e-6:
        warnings.warn(
            f"{metric_name} pooled {fig_pooled:.3f} > 1; "
            f"possible leakage or model issues.",
            stacklevel=2,
        )
    if fig > 1 + 1e-6:
        warnings.warn(
            f"{metric_name} {fig:.3f} > 1; possible leakage or model issues.",
            stacklevel=2,
        )

    return float(fig_pooled), float(fig), fig_by_student


class FigCResult(TypedDict):
    """Return type of ``fractional_information_gain_confidence``.

    Attributes
    ----------
    fig_c_pooled : float
        Observation-weighted FIG-C.
    fig_c : float
        Student-weighted FIG-C (macro average).
    fig_c_by_student : np.ndarray
        Per-student FIG-C values, aligned with ``student_ids``.
    student_ids : np.ndarray
        Unique student IDs in sorted order.
    """

    fig_c_pooled: float
    fig_c: float
    fig_c_by_student: np.ndarray
    student_ids: np.ndarray


def fractional_information_gain_confidence(
    y_pred_eval: np.ndarray | Sequence[float],
    item_id_eval: np.ndarray | Sequence[Any],
    student_id_eval: np.ndarray | Sequence[Any],
    y_train: np.ndarray | Sequence[float],
    item_id_train: np.ndarray | Sequence[Any],
    eps: float = 1e-12,
    use_shrinkage: bool = True,
    alpha: float = 2.0,
    beta: float = 2.0,
) -> FigCResult:
    """Compute Fractional Information Gain for Confidence (FIG-C).

    FIG-C measures the reduction in uncertainty about student responses based on
    model confidence, without requiring ground truth labels.

    ``FIG-C = 1 - H(y_pred) / H(baseline[item_id])``

    where:

    - ``H(baseline[item_id])`` = sum of binary entropies of per-item base rates
      (prior uncertainty), looked up by item ID
    - ``H(y_pred)`` = sum of binary entropies of model predictions
      (remaining uncertainty)
    - ``H(p) = -[p*log(p) + (1-p)*log(1-p)]``

    Parameters
    ----------
    y_pred_eval : array-like
        Predicted probabilities (0 to 1). Shape: (n_eval,)
    item_id_eval : array-like
        Item identifiers for eval observations. Shape: (n_eval,)
    student_id_eval : array-like
        Student identifiers for eval observations. Shape: (n_eval,)
    y_train : array-like
        Ground truth on train split. Shape: (n_train,)
        Values: 1=correct, 0=incorrect.
    item_id_train : array-like
        Item identifiers for train observations. Shape: (n_train,)
    eps : float, default=1e-12
        Small constant for clipping probabilities to avoid log(0).
    use_shrinkage : bool, default=True
        If True, apply empirical Bayes shrinkage to item baselines.
    alpha : float, default=2.0
        Prior pseudo-count for successes in shrinkage.
    beta : float, default=2.0
        Prior pseudo-count for failures in shrinkage.

    Returns
    -------
    FigCResult
        Dictionary with keys:

        - ``fig_c_pooled``: Observation-weighted FIG-C (scalar)
        - ``fig_c``: Student-weighted FIG-C (scalar)
        - ``fig_c_by_student``: Per-student FIG-C values (np.ndarray)
        - ``student_ids``: Unique student IDs in sorted order (np.ndarray)

    Notes
    -----
    - ``FIG-C = 0`` when model predictions match item base rates (no information gain).
    - ``FIG-C = 1`` when model is perfectly confident (entropy = 0).
    - FIG-C can be negative if model adds uncertainty beyond the baseline.
    - For well-calibrated models, FIG-C approximates FIG-V in expectation.
    - Items in eval that don't appear in train use the global training mean as baseline.

    Examples
    --------
    >>> y_pred = [0.95, 0.05, 0.9, 0.1, 0.8]
    >>> items = [1, 1, 2, 2, 1]
    >>> students = [1, 1, 2, 2, 3]
    >>> y_train = [1, 1, 0, 1, 0]
    >>> items_train = [1, 1, 2, 2, 2]
    >>>
    >>> results = fractional_information_gain_confidence(
    ...     y_pred_eval=y_pred,
    ...     item_id_eval=items,
    ...     student_id_eval=students,
    ...     y_train=y_train,
    ...     item_id_train=items_train,
    ... )
    >>> print(f"FIG-C: {results['fig_c']:.3f}")
    """
    # Validate and prepare inputs
    train = TrainData(y=y_train, item_id=item_id_train)
    baseline = compute_baseline(
        train, use_shrinkage=use_shrinkage, alpha=alpha, beta=beta
    )
    eval_data = prepare_eval_data(
        y_pred=y_pred_eval,
        y=None,
        item_id=item_id_eval,
        student_id=student_id_eval,
        baseline=baseline,
        eps=eps,
    )

    # Build per-eval baseline probabilities from pre-computed baseline
    baseline_probs = np.clip(
        baseline.item_baselines[eval_data.encoded_item_id], eps, 1 - eps
    )

    # H_D(Y): Prior uncertainty based on item base rates
    entropy_baseline = binary_entropy(baseline_probs)

    # H_M(Y|X): Remaining uncertainty based on model predictions
    entropy_model = binary_entropy(eval_data.y_pred)

    fig_c_pooled, fig_c, fig_c_by_student = _compute_fig(
        numerator=entropy_model,
        entropy_baseline=entropy_baseline,
        student_id_eval=eval_data.encoded_student_id,
        metric_name="FIG-C",
    )

    return {
        "fig_c_pooled": fig_c_pooled,
        "fig_c": fig_c,
        "fig_c_by_student": fig_c_by_student,
        "student_ids": eval_data.student_labels,
    }


class FigVResult(TypedDict):
    """Return type of ``fractional_information_gain_validation``.

    Attributes
    ----------
    fig_v_pooled : float
        Observation-weighted FIG-V.
    fig_v : float
        Student-weighted FIG-V (macro average).
    fig_v_by_student : np.ndarray
        Per-student FIG-V values, aligned with ``student_ids``.
    student_ids : np.ndarray
        Unique student IDs in sorted order.
    calibration : CalibrationResult or None
        Calibration metrics. None unless ``calibration=True`` was passed.
    """

    fig_v_pooled: float
    fig_v: float
    fig_v_by_student: np.ndarray
    student_ids: np.ndarray
    calibration: CalibrationResult | None


def fractional_information_gain_validation(
    # These are from the split of the dataset you want to evaluate the metric for
    # (usually val/test)
    y_pred_eval: np.ndarray | Sequence[float],
    y_eval: np.ndarray | Sequence[float],
    item_id_eval: np.ndarray | Sequence[Any],
    student_id_eval: np.ndarray | Sequence[Any],
    # These are from the training data and are used to calculate the baseline item
    # difficulty. It is important that these are from the training data, since the
    # model internalizes item difficulty and we are trying to correct for that
    y_train: np.ndarray | Sequence[float],
    item_id_train: np.ndarray | Sequence[Any],
    # These are optional and you can probably ignore them
    eps: float = 1e-12,
    use_shrinkage: bool = True,
    alpha: float = 2.0,
    beta: float = 2.0,
    calibration: bool = False,
    calib_n_bins: int = 10,
    calib_warn_threshold: float = 0.1,
) -> FigVResult:
    """Compute Fractional Information Gain for Validation (FIG-V).

    FIG-V measures how much better a model's predictions are than per-item base
    rates, using cross-entropy against ground truth.

    ``FIG-V = 1 - BCE(y_pred, y) / H(baseline[item_id])``

    where:

    - ``H(baseline[item_id])`` = sum of binary entropies of per-item base rates
      (prior uncertainty), looked up by item ID
    - ``BCE(y_pred, y)`` = sum of binary cross-entropies of model predictions
      against ground truth
    - ``BCE(p, y) = -[y*log(p) + (1-y)*log(1-p)]``

    Parameters
    ----------
    y_pred_eval : array-like
        Predicted probabilities (0 to 1). Shape: (n_eval,)
    y_eval : array-like
        Ground truth on eval split. Shape: (n_eval,)
        Values: 1=correct, 0=incorrect.
    item_id_eval : array-like
        Item identifiers for eval observations. Shape: (n_eval,)
    student_id_eval : array-like
        Student identifiers for eval observations. Shape: (n_eval,)
    y_train : array-like
        Ground truth on train split. Shape: (n_train,)
        Values: 1=correct, 0=incorrect.
    item_id_train : array-like
        Item identifiers for train observations. Shape: (n_train,)
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

    Returns
    -------
    FigVResult
        Dictionary with keys:

        - ``fig_v_pooled``: Observation-weighted FIG-V (scalar)
        - ``fig_v``: Student-weighted FIG-V (scalar)
        - ``fig_v_by_student``: Per-student FIG-V values (np.ndarray)
        - ``student_ids``: Unique student IDs in sorted order (np.ndarray)
        - ``calibration``: None unless ``calibration=True``, in which case a
          dict of ECE metrics from ``check_calibration``

    Notes
    -----
    - ``FIG-V = 0`` when model predictions match item base rates (no information gain).
    - ``FIG-V = 1`` when model predictions are perfect (zero cross-entropy).
    - FIG-V can be negative if model is worse than the baseline.
    - ``FIG-V > 1`` usually indicates data leakage; the library warns when this happens.
    - Items in eval that don't appear in train use the global training mean as baseline.

    Examples
    --------
    >>> y_pred = [0.8, 0.6, 0.9, 0.5, 0.7]
    >>> y_eval = [1, 0, 1, 0, 1]
    >>> items = [1, 1, 2, 2, 1]
    >>> students = [1, 1, 2, 2, 3]
    >>> y_train = [1, 1, 0, 1, 0]
    >>> items_train = [1, 1, 2, 2, 2]
    >>>
    >>> results = fractional_information_gain_validation(
    ...     y_pred_eval=y_pred,
    ...     y_eval=y_eval,
    ...     item_id_eval=items,
    ...     student_id_eval=students,
    ...     y_train=y_train,
    ...     item_id_train=items_train,
    ... )
    >>> print(f"FIG-V: {results['fig_v']:.3f}")
    """
    # Validate and prepare inputs
    train = TrainData(y=y_train, item_id=item_id_train)
    baseline = compute_baseline(
        train, use_shrinkage=use_shrinkage, alpha=alpha, beta=beta
    )
    eval_data = prepare_eval_data(
        y_pred=y_pred_eval,
        y=y_eval,
        item_id=item_id_eval,
        student_id=student_id_eval,
        baseline=baseline,
        eps=eps,
    )

    # --- Build per-eval baseline probabilities from pre-computed baseline ---
    baseline_probs_eval = np.clip(
        baseline.item_baselines[eval_data.encoded_item_id], eps, 1 - eps
    )

    # --- Compute cross-entropy for model (numerator) ---
    # Bernoulli cross-entropy: -[y*log(p) + (1-y)*log(1-p)]
    assert eval_data.y is not None, (
        "eval_data.y must not be None for FIG-V (this is a bug)"
    )
    ce_model = -(
        eval_data.y * np.log(eval_data.y_pred)
        + (1 - eval_data.y) * np.log(1 - eval_data.y_pred)
    )

    entropy_baseline = binary_entropy(baseline_probs_eval)

    fig_v_pooled, fig_v, fig_v_by_student = _compute_fig(
        numerator=ce_model,
        entropy_baseline=entropy_baseline,
        student_id_eval=eval_data.encoded_student_id,
        metric_name="FIG-V",
    )

    result: FigVResult = {
        "fig_v_pooled": fig_v_pooled,
        "fig_v": fig_v,
        "fig_v_by_student": fig_v_by_student,
        "student_ids": eval_data.student_labels,
        "calibration": None,
    }

    # --- Optional calibration ---
    if calibration:
        result["calibration"] = check_calibration(
            eval_data.y_pred, eval_data.y, calib_n_bins, calib_warn_threshold
        )

    return result


# Example usage
if __name__ == "__main__":
    # Create synthetic data in "long format"
    rng = np.random.default_rng(42)

    # Simulate 100 students, 20 items, ~30 responses per student on average
    n_train = 3000
    n_eval = 1000
    n_items = 20
    n_students = 100

    # Training data
    y_train = rng.choice([1, 0], size=n_train, p=[0.6, 0.4])
    item_id_train = rng.integers(1, n_items + 1, size=n_train)

    # Eval data with student IDs
    y_eval = rng.choice([1, 0], size=n_eval, p=[0.65, 0.35])
    item_id_eval = rng.integers(1, n_items + 1, size=n_eval)
    student_id_eval = rng.integers(1, n_students + 1, size=n_eval)

    # Generate predictions (somewhat correlated with true values)
    y_pred_eval = rng.beta(2, 1, size=n_eval)
    # Make predictions better for correct answers
    y_pred_eval = np.where(
        y_eval == 1, np.clip(y_pred_eval + 0.2, 0, 1), np.clip(y_pred_eval - 0.1, 0, 1)
    )

    # Compute FIG-V (requires ground truth)
    results_v = fractional_information_gain_validation(
        y_pred_eval=y_pred_eval,
        y_eval=y_eval,
        item_id_eval=item_id_eval,
        y_train=y_train,
        item_id_train=item_id_train,
        student_id_eval=student_id_eval,
        use_shrinkage=True,
        calibration=True,
    )

    print("FIG-V Results (requires ground truth):")  # noqa: T201
    print(f"  FIG-V pooled: {results_v['fig_v_pooled']:.4f}")  # noqa: T201
    print(f"  FIG-V: {results_v['fig_v']:.4f}")  # noqa: T201

    # Compute FIG-C (does NOT require ground truth)
    results_c = fractional_information_gain_confidence(
        y_pred_eval=y_pred_eval,
        item_id_eval=item_id_eval,
        student_id_eval=student_id_eval,
        y_train=y_train,
        item_id_train=item_id_train,
        use_shrinkage=True,
    )

    print("\nFIG-C Results (no ground truth needed):")  # noqa: T201
    print(f"  FIG-C pooled: {results_c['fig_c_pooled']:.4f}")  # noqa: T201
    print(f"  FIG-C: {results_c['fig_c']:.4f}")  # noqa: T201
    print(  # noqa: T201
        "\nNote: For well-calibrated models, FIG-C approximates FIG-V in expectation."
    )
