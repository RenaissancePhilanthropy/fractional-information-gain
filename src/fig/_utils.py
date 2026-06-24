"""Internal shared utilities for the fig package.

This module contains functions shared between fig.py (numpy) and fig_torch.py
(PyTorch). Public names: CalibrationResult, TrainData, BaselineData, compute_baseline,
check_calibration. All other names are private (_-prefixed).
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, TypedDict

import numpy as np

try:
    import torch as _torch_opt
except ImportError:
    _torch_opt = None  # type: ignore[assignment]


@dataclass(frozen=True, init=False)
class TrainData:
    """Validated, encoded training observations.

    Accepts raw array-likes for y and item_id. Validation and integer encoding
    of item IDs are performed in ``__init__``. After construction the object
    is immutable.

    Parameters
    ----------
    y : array-like
        Binary training outcomes (1D). Values must be 0 or 1.
    item_id : array-like
        Item identifiers for training observations (1D). Any sortable type.

    Attributes
    ----------
    y : np.ndarray
        float64, shape (n_train,).
    item_id : np.ndarray
        Raw item IDs, shape (n_train,). Dtype preserved from input.
    encoded_item_id : np.ndarray
        int64, contiguous integers 0..N-1, shape (n_train,).
    item_labels : np.ndarray
        Unique training item IDs in sorted order, length N.
    """

    y: np.ndarray
    item_id: np.ndarray
    encoded_item_id: np.ndarray = field(repr=False)
    item_labels: np.ndarray = field(repr=False)

    def __init__(self, y: Any, item_id: Any) -> None:  # noqa: ANN401
        # Coerce inputs - use object.__setattr__ because the dataclass is frozen.
        # torch.Tensor inputs must be detached before conversion; np.asarray()
        # raises on tensors that require grad. _torch_opt is None when PyTorch
        # is not installed.
        def _to_np(x: object, dtype: type[Any] | None = None) -> np.ndarray:
            if _torch_opt is not None and isinstance(x, _torch_opt.Tensor):
                arr: np.ndarray = x.detach().cpu().numpy().ravel()  # pyright: ignore[reportUnknownMemberType]
            else:
                arr = np.asarray(x).ravel()
            return arr.astype(dtype) if dtype is not None else arr

        y_arr = _to_np(y, np.float64)
        item_id_arr = _to_np(item_id)
        object.__setattr__(self, "y", y_arr)
        object.__setattr__(self, "item_id", item_id_arr)

        n = len(y_arr)
        if n == 0:
            raise ValueError("y must not be empty")
        if len(item_id_arr) != n:
            raise ValueError(
                f"Length mismatch: item_id ({len(item_id_arr)}) != y ({n})"
            )
        unique_vals = set(np.unique(y_arr))
        bad = unique_vals - {0.0, 1.0}
        if bad:
            raise ValueError(f"y must contain only 0 and 1, got: {bad}")

        labels, encoded = np.unique(item_id_arr, return_inverse=True)
        object.__setattr__(self, "item_labels", labels)
        object.__setattr__(self, "encoded_item_id", encoded.astype(np.int64))


@dataclass(frozen=True)
class BaselineData:
    """Pre-computed baseline statistics derived from training data.

    Produced by compute_baseline(). Treat as an opaque token: construct it once
    per training split and pass it to the FIG functions. Do not construct
    manually.

    Notes
    -----
    ``frozen=True`` prevents attribute *reassignment* but does not prevent
    in-place mutation of the mutable fields (``item_baselines[i] = x`` or
    ``item_vocab[k] = v`` will succeed silently). Do not mutate either field
    after construction.

    Attributes
    ----------
    item_vocab : dict
        Maps each training item ID to its integer index (0..N-1).
    item_baselines : np.ndarray
        float64, length N+1. Indices 0..N-1 hold per-item baseline
        probabilities (not yet clipped). Index N holds the global training
        mean and serves as the fallback for unseen eval items.
    """

    item_vocab: dict[Any, int]
    item_baselines: np.ndarray


def compute_baseline(
    train: TrainData,
    use_shrinkage: bool = True,
    alpha: float = 2.0,
    beta: float = 2.0,
) -> BaselineData:
    """Compute per-item baseline statistics from a TrainData object.

    Parameters
    ----------
    train : TrainData
        Training data
    use_shrinkage : bool, default=True
        If True, apply empirical Bayes shrinkage toward the global mean.
    alpha : float, default=2.0
        Prior pseudo-count for successes in EB shrinkage.
    beta : float, default=2.0
        Prior pseudo-count for failures in EB shrinkage.

    Returns
    -------
    BaselineData
        Pre-computed baseline. Pass this to FIG functions as the baseline
        argument to avoid recomputing per training batch.
    """
    item_baselines, global_mean = _compute_item_baselines(
        train.y, train.encoded_item_id, use_shrinkage, alpha, beta
    )
    # Append global_mean as sentinel at index N.
    item_baselines_with_sentinel = np.append(item_baselines, global_mean)
    item_vocab = {label: i for i, label in enumerate(train.item_labels)}
    return BaselineData(
        item_vocab=item_vocab,
        item_baselines=item_baselines_with_sentinel,
    )


def binary_entropy(p: np.ndarray) -> np.ndarray:
    """Compute binary entropy ``H(p) = -[p*log(p) + (1-p)*log(1-p)]``.

    Parameters
    ----------
    p : np.ndarray
        Probabilities (should already be clipped to avoid log(0)).

    Returns
    -------
    np.ndarray
        Binary entropy for each probability.
    """
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def _compute_item_baselines(
    y_train: np.ndarray,
    encoded_item_id_train: np.ndarray,
    use_shrinkage: bool = True,
    alpha: float = 2.0,
    beta: float = 2.0,
) -> tuple[np.ndarray, float]:
    """Compute per-item baseline probabilities from pre-encoded training data.

    Parameters
    ----------
    y_train : np.ndarray
        Binary training outcomes (1D, float64).
    encoded_item_id_train : np.ndarray
        Pre-encoded item IDs (1D). Values must be in 0..N-1 where N is the
        number of unique training items.
    use_shrinkage : bool, default=True
        If True, apply empirical Bayes shrinkage toward the global mean.
    alpha : float, default=2.0
        Prior pseudo-count for successes in EB shrinkage.
    beta : float, default=2.0
        Prior pseudo-count for failures in EB shrinkage.

    Returns
    -------
    item_baselines : np.ndarray
        Per-item baseline probability (1D, float64), indexed by encoded ID.
        Length N (number of unique training items). Not clipped.
    global_mean : float
        Global mean correctness. Used as the fallback for unseen eval items.
    """
    if len(y_train) == 0:
        raise ValueError("_compute_item_baselines requires non-empty training data")
    if len(y_train) != len(encoded_item_id_train):
        raise ValueError(
            f"Length mismatch: y_train ({len(y_train)}) != "
            f"encoded_item_id_train ({len(encoded_item_id_train)})"
        )
    global_mean = float(y_train.mean())
    n_items = int(encoded_item_id_train.max()) + 1
    item_sums = np.bincount(encoded_item_id_train, weights=y_train, minlength=n_items)
    item_counts = np.bincount(encoded_item_id_train, minlength=n_items)
    if use_shrinkage:
        item_baselines = (item_sums + alpha) / (item_counts + alpha + beta)
    else:
        item_baselines = item_sums / item_counts
    return item_baselines, global_mean


class CalibrationResult(TypedDict):
    """Return type of check_calibration.

    Bin lists have length <= n_bins; empty bins are omitted.
    """

    ece: float
    bin_accuracies: list[float]
    bin_confidences: list[float]
    bin_counts: list[int]


def check_calibration(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    n_bins: int,
    warn_threshold: float,
) -> CalibrationResult:
    """Compute Expected Calibration Error (ECE).

    This is done by binning predictions and comparing average confidence to average
    accuracy in each bin.

    Parameters
    ----------
    y_pred : np.ndarray
        Predicted probabilities (1D).
    y_true : np.ndarray
        True binary outcomes (1D).
    n_bins : int
        Number of bins for binning predictions.
    warn_threshold : float
        Warn if ECE exceeds this value.

    Returns
    -------
    CalibrationResult
        Dictionary of calibration metrics including ECE, with the keys:
        - 'ece': Expected Calibration Error (scalar float)
        - 'bin_accuracies': Average accuracy per non-empty bin (list[float])
        - 'bin_confidences': Average confidence per non-empty bin (list[float])
        - 'bin_counts': Observation count per non-empty bin (list[int])
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred, bin_edges[1:-1])

    ece = 0.0
    bin_accs: list[float] = []
    bin_confs: list[float] = []
    bin_counts: list[int] = []

    for bin_idx in range(n_bins):
        in_bin = bin_indices == bin_idx
        bin_count = in_bin.sum()
        if bin_count > 0:
            bin_acc = y_true[in_bin].mean()
            bin_conf = y_pred[in_bin].mean()

            ece += (bin_count / len(y_pred)) * abs(bin_acc - bin_conf)

            bin_accs.append(float(bin_acc))
            bin_confs.append(float(bin_conf))
            bin_counts.append(int(bin_count))

    if ece > warn_threshold:
        warnings.warn(
            f"ECE {ece:.3f} exceeds threshold {warn_threshold:.3f}", stacklevel=2
        )

    return {
        "ece": float(ece),
        "bin_accuracies": bin_accs,
        "bin_confidences": bin_confs,
        "bin_counts": bin_counts,
    }
