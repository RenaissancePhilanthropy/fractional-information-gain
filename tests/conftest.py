"""
Shared test helpers for the fig test suite.

pytest loads conftest.py automatically, but plain constants and functions defined
here still require an explicit `from conftest import ...` in each test file.
"""

from typing import Any, TypedDict

import numpy as np

from fig._utils import CalibrationResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Tolerance for comparing the production implementation against the reference
# implementation. This is float64 machine epsilon multiplied by a few orders of
# magnitude for safety.
FLOAT_TOL = 1e-10


class FigCInputs(TypedDict):
    """Keyword arguments accepted by fractional_information_gain_confidence."""

    y_pred_eval: np.ndarray
    item_id_eval: np.ndarray
    student_id_eval: np.ndarray
    y_train: np.ndarray
    item_id_train: np.ndarray


class FigVInputs(TypedDict):
    """Keyword arguments accepted by fractional_information_gain_validation."""

    y_pred_eval: np.ndarray
    y_eval: np.ndarray
    item_id_eval: np.ndarray
    student_id_eval: np.ndarray
    y_train: np.ndarray
    item_id_train: np.ndarray


# ---------------------------------------------------------------------------
# Canonical fixtures
# ---------------------------------------------------------------------------

# Canonical inputs for FIG-C tests (2 items, 2 students, 4 observations).
INPUTS_C: FigCInputs = {
    "y_pred_eval": np.array([0.8, 0.3, 0.7, 0.2]),
    "item_id_eval": np.array([1, 1, 2, 2]),
    "student_id_eval": np.array([1, 1, 2, 2]),
    "y_train": np.array([1, 0, 1, 0]),
    "item_id_train": np.array([1, 1, 2, 2]),
}

# Same as INPUTS_C but with ground-truth labels for FIG-V tests.
INPUTS_V: FigVInputs = {**INPUTS_C, "y_eval": np.array([1, 0, 1, 0])}

# String-ID variants of the canonical fixtures (same numeric values, string IDs).
INPUTS_C_STR: FigCInputs = {
    "y_pred_eval": np.array([0.8, 0.3, 0.7, 0.2]),
    "item_id_eval": np.array(["item_1", "item_1", "item_2", "item_2"]),
    "student_id_eval": np.array(["student_1", "student_1", "student_2", "student_2"]),
    "y_train": np.array([1, 0, 1, 0], dtype=float),
    "item_id_train": np.array(["item_1", "item_1", "item_2", "item_2"]),
}
INPUTS_V_STR: FigVInputs = {
    **INPUTS_C_STR,
    "y_eval": np.array([1, 0, 1, 0], dtype=float),
}


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------


def fig_c_reference(
    y_pred_eval: np.ndarray,
    item_id_eval: np.ndarray,
    student_id_eval: np.ndarray,
    y_train: np.ndarray,
    item_id_train: np.ndarray,
) -> dict[str, Any]:
    """Reference implementation of FIG-C.

    This is a maximally simple and readable implementation of FIG-C, without any
    optimisations or any of the finer features, like clipping extreme probabilities
    or shrinkage. It is intended to be used as a reference for testing the more complex
    and optimised implementation in fractional_information_gain_confidence.
    """
    # Per-item base rates from training data (raw means, no shrinkage).
    item_base_rates = {
        item: y_train[item_id_train == item].mean() for item in np.unique(item_id_train)
    }
    global_mean = y_train.mean()

    # One baseline probability per eval observation.
    baseline = np.array(
        [item_base_rates.get(item, global_mean) for item in item_id_eval]
    )

    # Vectorised binary entropy.
    def _h(p: np.ndarray) -> np.ndarray:
        return -(p * np.log(p) + (1 - p) * np.log(1 - p))

    h_pred = _h(y_pred_eval)
    h_base = _h(baseline)

    # Per-student FIG-C, then average.
    unique_students = np.unique(student_id_eval)
    fig_c_by_student = np.array(
        [
            1 - h_pred[student_id_eval == s].sum() / h_base[student_id_eval == s].sum()
            for s in unique_students
        ]
    )
    fig_c = float(np.mean(fig_c_by_student))

    # Observation-weighted FIG-C.
    fig_c_pooled = float(1 - h_pred.sum() / h_base.sum())

    return {
        "fig_c": fig_c,
        "fig_c_pooled": fig_c_pooled,
        "fig_c_by_student": fig_c_by_student,
        "student_ids": unique_students,
    }


def fig_v_reference(
    y_pred_eval: np.ndarray,
    y_eval: np.ndarray,
    item_id_eval: np.ndarray,
    student_id_eval: np.ndarray,
    y_train: np.ndarray,
    item_id_train: np.ndarray,
) -> dict[str, Any]:
    """Reference implementation of FIG-V.

    This is a maximally simple and readable implementation of FIG-V, without any
    optimisations or any of the finer features, like clipping extreme probabilities
    or shrinkage. It is intended to be used as a reference for testing the more complex
    and optimised implementation in fractional_information_gain_validation.
    """
    # Per-item base rates from training data (raw means, no shrinkage).
    item_base_rates = {
        item: y_train[item_id_train == item].mean() for item in np.unique(item_id_train)
    }
    global_mean = y_train.mean()

    # One baseline probability per eval observation.
    baseline = np.array(
        [item_base_rates.get(item, global_mean) for item in item_id_eval]
    )

    # Vectorised binary entropy (denominator: prior uncertainty from baseline).
    def _h(p: np.ndarray) -> np.ndarray:
        return -(p * np.log(p) + (1 - p) * np.log(1 - p))

    # Vectorised binary cross-entropy (numerator: model's remaining uncertainty).
    def _bce(y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    ce_model = _bce(y_eval, y_pred_eval)
    h_base = _h(baseline)

    # Per-student FIG-V, then average.
    unique_students = np.unique(student_id_eval)
    fig_v_by_student = np.array(
        [
            1
            - ce_model[student_id_eval == s].sum() / h_base[student_id_eval == s].sum()
            for s in unique_students
        ]
    )
    fig_v = float(np.mean(fig_v_by_student))

    # Observation-weighted FIG-V.
    fig_v_pooled = float(1 - ce_model.sum() / h_base.sum())

    return {
        "fig_v": fig_v,
        "fig_v_pooled": fig_v_pooled,
        "fig_v_by_student": fig_v_by_student,
        "student_ids": unique_students,
    }


# ---------------------------------------------------------------------------
# Shared invariant checkers
# ---------------------------------------------------------------------------


def assert_calibration_result_invariants(
    result: CalibrationResult, n_bins: int
) -> None:
    """Assert invariants that must hold for any valid calibration result dict."""
    assert "ece" in result
    assert "bin_accuracies" in result
    assert "bin_confidences" in result
    assert "bin_counts" in result
    assert isinstance(result["ece"], float)
    assert np.isfinite(result["ece"])
    assert result["ece"] >= 0
    # The comparison is <= because empty bins are omitted from the results.
    assert len(result["bin_accuracies"]) <= n_bins
    assert len(result["bin_confidences"]) <= n_bins
    assert len(result["bin_counts"]) <= n_bins
