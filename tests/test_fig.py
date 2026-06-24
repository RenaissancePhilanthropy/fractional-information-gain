"""
Unit tests for the public API of fig.py
"""

from typing import Any

import numpy as np
import pytest
from conftest import (
    FLOAT_TOL,
    INPUTS_C,
    INPUTS_C_STR,
    INPUTS_V,
    INPUTS_V_STR,
    FigCInputs,
    FigVInputs,
    assert_calibration_result_invariants,
    fig_c_reference,
    fig_v_reference,
)

from fig.fig import (
    FigCResult,
    FigVResult,
    binary_entropy,
    check_calibration,
    fractional_information_gain_confidence,
    fractional_information_gain_validation,
)


def assert_fig_c_invariants(result: FigCResult) -> None:
    """Assert invariants that must hold for any valid FIG-C result dict."""
    # Required keys are present.
    assert "fig_c" in result
    assert "fig_c_pooled" in result
    assert "fig_c_by_student" in result
    assert "student_ids" in result

    # Scalar types.
    assert isinstance(result["fig_c"], float)
    assert isinstance(result["fig_c_pooled"], float)

    # Per-student array has one entry per student.
    assert isinstance(result["fig_c_by_student"], np.ndarray)
    assert len(result["fig_c_by_student"]) == len(result["student_ids"])

    # FIG-C is bounded above by 1 (a perfectly confident model achieves exactly 1).
    assert result["fig_c"] <= 1 + 1e-6
    assert result["fig_c_pooled"] <= 1 + 1e-6
    assert all(v <= 1 + 1e-6 for v in result["fig_c_by_student"])

    # All values are finite.
    assert np.isfinite(result["fig_c"])
    assert np.isfinite(result["fig_c_pooled"])
    assert all(np.isfinite(v) for v in result["fig_c_by_student"])

    # student_ids is a numpy array with one entry per student, in sorted order.
    assert isinstance(result["student_ids"], np.ndarray)
    assert np.all(result["student_ids"] == np.sort(result["student_ids"]))


def assert_fig_c_close(
    result: FigCResult, ref: dict[str, Any], tol: float = FLOAT_TOL
) -> None:
    """Assert that a FIG-C result dict agrees with a reference dict within tol.

    Compares fig_c, fig_c_pooled, and per-student values. Student ordering in
    result and ref may differ; values are matched by student ID.
    """
    np.testing.assert_allclose(result["fig_c"], ref["fig_c"], atol=tol)
    np.testing.assert_allclose(result["fig_c_pooled"], ref["fig_c_pooled"], atol=tol)
    for s, ref_val in zip(ref["student_ids"], ref["fig_c_by_student"], strict=False):
        idx = list(result["student_ids"]).index(s)
        np.testing.assert_allclose(result["fig_c_by_student"][idx], ref_val, atol=tol)


def assert_fig_v_invariants(
    result: FigVResult, calibration: bool = False, n_bins: int = 10
) -> None:
    """Assert invariants that must hold for any valid FIG-V result dict."""
    assert "fig_v" in result
    assert "fig_v_pooled" in result
    assert "fig_v_by_student" in result
    assert "student_ids" in result
    assert isinstance(result["fig_v"], float)
    assert isinstance(result["fig_v_pooled"], float)
    assert np.isfinite(result["fig_v"])
    assert np.isfinite(result["fig_v_pooled"])
    assert result["fig_v"] <= 1 + 1e-6
    assert result["fig_v_pooled"] <= 1 + 1e-6
    assert isinstance(result["fig_v_by_student"], np.ndarray)
    assert len(result["fig_v_by_student"]) == len(result["student_ids"])
    assert all(v <= 1 + 1e-6 for v in result["fig_v_by_student"])
    assert all(np.isfinite(v) for v in result["fig_v_by_student"])
    assert isinstance(result["student_ids"], np.ndarray)
    assert np.all(result["student_ids"] == np.sort(result["student_ids"]))
    assert "calibration" in result
    if calibration:
        assert result["calibration"] is not None
        assert_calibration_result_invariants(result["calibration"], n_bins=n_bins)
    else:
        assert result["calibration"] is None


def assert_fig_v_close(
    result: FigVResult, ref: dict[str, Any], tol: float = FLOAT_TOL
) -> None:
    """Assert that a FIG-V result dict agrees with a reference dict within tol.

    Compares fig_v, fig_v_pooled, and per-student values. Student ordering in
    result and ref may differ; values are matched by student ID.
    """
    np.testing.assert_allclose(result["fig_v"], ref["fig_v"], atol=tol)
    np.testing.assert_allclose(result["fig_v_pooled"], ref["fig_v_pooled"], atol=tol)
    for s, ref_val in zip(ref["student_ids"], ref["fig_v_by_student"], strict=False):
        idx = list(result["student_ids"]).index(s)
        np.testing.assert_allclose(result["fig_v_by_student"][idx], ref_val, atol=tol)


class TestCheckCalibration:
    """Tests for check_calibration function."""

    def test_perfectly_calibrated(self) -> None:
        """Test ECE with perfectly calibrated predictions."""
        # Predictions exactly match true rates
        np.random.seed(42)
        n = 1000
        y_pred = np.random.uniform(0, 1, n)
        y_true = (np.random.uniform(0, 1, n) < y_pred).astype(float)

        result = check_calibration(y_pred, y_true, n_bins=10, warn_threshold=0.5)
        assert_calibration_result_invariants(result, n_bins=10)
        assert result["ece"] < 0.1

    def test_overconfident_predictions(self) -> None:
        """Test ECE with overconfident predictions."""
        # All predictions are 0.9, but only 50% are correct.
        # All observations fall in one bin, so ECE = |confidence - accuracy|
        # = |0.9 - 0.5|.
        y_pred = np.ones(100) * 0.9
        y_true = np.concatenate([np.ones(50), np.zeros(50)])
        result = check_calibration(y_pred, y_true, n_bins=10, warn_threshold=0.5)
        assert_calibration_result_invariants(result, n_bins=10)
        assert result["ece"] == pytest.approx(abs(0.9 - 0.5))  # pyright: ignore[reportUnknownMemberType]

    def test_underconfident_predictions(self) -> None:
        """Test ECE with underconfident predictions."""
        # All predictions are 0.5, but 90% are correct.
        # All observations fall in one bin, so ECE = |confidence - accuracy|
        # = |0.5 - 0.9|.
        y_pred = np.ones(100) * 0.5
        y_true = np.concatenate([np.ones(90), np.zeros(10)])
        result = check_calibration(y_pred, y_true, n_bins=10, warn_threshold=0.5)
        assert_calibration_result_invariants(result, n_bins=10)
        assert result["ece"] == pytest.approx(abs(0.5 - 0.9))  # pyright: ignore[reportUnknownMemberType]

    def test_warning_threshold(self) -> None:
        """Test that a warning is raised when ECE exceeds threshold."""
        y_pred = np.ones(100) * 0.9
        y_true = np.concatenate([np.ones(50), np.zeros(50)])
        with pytest.warns(UserWarning, match="ECE"):
            check_calibration(y_pred, y_true, n_bins=10, warn_threshold=0.1)

    def test_different_bin_counts(self) -> None:
        """More bins inflates ECE; result structure is valid for each bin count."""
        np.random.seed(42)
        n = 1000
        y_pred = np.random.uniform(0, 1, n)
        y_true = (np.random.uniform(0, 1, n) < y_pred).astype(float)
        result_low = check_calibration(y_pred, y_true, n_bins=5, warn_threshold=0.5)
        result_high = check_calibration(y_pred, y_true, n_bins=100, warn_threshold=0.5)
        assert_calibration_result_invariants(result_low, n_bins=5)
        assert_calibration_result_invariants(result_high, n_bins=100)
        # Using an excessive number of bins should inflate ECE due to many
        # empty/low-count bins.
        assert result_low["ece"] < result_high["ece"]


class TestBinaryEntropy:
    """Tests for binary_entropy function."""

    def test_entropy_at_half(self) -> None:
        """Binary entropy should be maximum (log(2)) at p=0.5."""
        p = np.array([0.5])
        result = binary_entropy(p)
        expected = np.log(2)  # Maximum entropy for binary
        assert result[0] == pytest.approx(expected)  # pyright: ignore[reportUnknownMemberType]

    def test_entropy_at_extremes(self) -> None:
        """Binary entropy should be near 0 at extreme probabilities."""
        eps = 1e-12
        p = np.array([eps, 1 - eps])
        result = binary_entropy(p)
        assert len(result) == len(p)
        assert result[0] == pytest.approx(0, abs=1e-9)  # pyright: ignore[reportUnknownMemberType]
        assert result[1] == pytest.approx(0, abs=1e-9)  # pyright: ignore[reportUnknownMemberType]

    def test_entropy_symmetry(self) -> None:
        """Binary entropy should be symmetric: H(p) = H(1-p)."""
        p = np.array([0.2, 0.3, 0.4])
        result_p = binary_entropy(p)
        result_1_minus_p = binary_entropy(1 - p)
        assert result_p == pytest.approx(result_1_minus_p)  # pyright: ignore[reportUnknownMemberType]


class TestFractionalInformationGainConfidence:
    """Tests for fractional_information_gain_confidence function (FIG-C)."""

    @pytest.mark.parametrize("seed", [1, 42, 123, 456, 789])
    @pytest.mark.parametrize("use_shrinkage", [False, True])
    def test_matches_reference(self, seed: int, use_shrinkage: bool) -> None:
        """FIG-C output matches the reference implementation on random inputs.

        With use_shrinkage=False, values are compared to the reference implementation.
        With use_shrinkage=True, only invariants are checked (reference does not support
        shrinkage).
        """
        rng = np.random.default_rng(seed)
        n = rng.integers(20, 80)
        num_students = 7
        inputs: FigCInputs = {
            "y_pred_eval": rng.uniform(0.05, 0.95, n),
            "item_id_eval": rng.integers(1, 5, n),
            "student_id_eval": rng.integers(1, num_students + 1, n),
            "y_train": rng.integers(0, 2, n * 2).astype(float),
            "item_id_train": rng.integers(1, 5, n * 2),
        }
        result = fractional_information_gain_confidence(
            **inputs, use_shrinkage=use_shrinkage
        )
        assert_fig_c_invariants(result)
        if not use_shrinkage:
            ref = fig_c_reference(**inputs)
            assert_fig_c_close(result, ref)

    @pytest.mark.parametrize(
        "inputs_c", [INPUTS_C, INPUTS_C_STR], ids=["int_ids", "str_ids"]
    )
    def test_matches_reference_canonical(self, inputs_c: FigCInputs) -> None:
        """FIG-C matches the reference implementation for both integer and string ID
        fixtures."""
        result = fractional_information_gain_confidence(**inputs_c, use_shrinkage=False)
        ref = fig_c_reference(**inputs_c)
        assert_fig_c_invariants(result)
        assert_fig_c_close(result, ref)

    def test_confident_model_high_fig_c(self) -> None:
        """A model with near-certain predictions should have FIG-C close to 1."""
        # Items with 50% base rate; model is maximally confident on each.
        inputs: FigCInputs = {
            **INPUTS_C,
            "y_pred_eval": np.array([1 - 1e-9, 1e-9, 1 - 1e-9, 1e-9]),
            "y_train": np.array([1, 0, 1, 0, 1, 0], dtype=np.float64),
            "item_id_train": np.array([1, 1, 1, 2, 2, 2]),
        }
        result = fractional_information_gain_confidence(**inputs, use_shrinkage=False)
        assert_fig_c_invariants(result)
        ref = fig_c_reference(**inputs)
        assert_fig_c_close(result, ref)
        assert result["fig_c"] > 0.9
        assert result["fig_c_pooled"] > 0.9

    def test_uninformative_model_low_fig_c(self) -> None:
        """A model predicting 0.5 against a 50/50 baseline should have FIG-C of 0."""
        inputs: FigCInputs = {**INPUTS_C, "y_pred_eval": np.array([0.5, 0.5, 0.5, 0.5])}
        result = fractional_information_gain_confidence(**inputs, use_shrinkage=False)
        assert_fig_c_invariants(result)
        assert result["fig_c"] == pytest.approx(0)  # pyright: ignore[reportUnknownMemberType]
        assert result["fig_c_pooled"] == pytest.approx(0)  # pyright: ignore[reportUnknownMemberType]

    def test_single_student(self) -> None:
        """With one student, fig_c equals fig_c_pooled and fig_c_by_student has one
        entry."""
        inputs: FigCInputs = {**INPUTS_C, "student_id_eval": np.array([1, 1, 1, 1])}
        result = fractional_information_gain_confidence(**inputs)
        assert_fig_c_invariants(result)
        assert result["fig_c_pooled"] == result["fig_c"]
        assert len(result["fig_c_by_student"]) == 1
        assert result["fig_c_by_student"][0] == pytest.approx(result["fig_c"])  # pyright: ignore[reportUnknownMemberType]

    def test_unseen_item_integration_matches_reference(self) -> None:
        """End-to-end: eval item absent from training uses global-mean fallback and
        the result matches the reference implementation."""
        inputs: FigCInputs = {
            **INPUTS_C,
            "item_id_eval": np.array([1, 1, 999, 999]),
        }
        result = fractional_information_gain_confidence(**inputs, use_shrinkage=False)
        ref = fig_c_reference(**inputs)
        assert_fig_c_invariants(result)
        assert_fig_c_close(result, ref)

    def test_invalid_y_train_raises(self) -> None:
        """Non-binary y_train raises ValueError."""
        inputs: FigCInputs = {
            **INPUTS_C,
            "y_train": np.array([1, -1]),
            "item_id_train": np.array([1, 2]),
        }
        with pytest.raises(ValueError, match="must contain only 0 and 1"):
            fractional_information_gain_confidence(**inputs)

    def test_empty_train_raises(self) -> None:
        """Empty y_train raises ValueError."""
        inputs: FigCInputs = {
            **INPUTS_C,
            "y_train": np.array([], dtype=float),
            "item_id_train": np.array([], dtype=int),
        }
        with pytest.raises(ValueError, match="must not be empty"):
            fractional_information_gain_confidence(**inputs)

    def test_empty_eval_raises(self) -> None:
        """Empty eval arrays raise ValueError."""
        inputs: FigCInputs = {
            **INPUTS_C,
            "y_pred_eval": np.array([], dtype=float),
            "item_id_eval": np.array([], dtype=int),
            "student_id_eval": np.array([], dtype=int),
        }
        with pytest.raises(ValueError, match="must not be empty"):
            fractional_information_gain_confidence(**inputs)

    def test_mismatched_item_id_train_length_raises(self) -> None:
        """item_id_train length != y_train length raises ValueError."""
        inputs: FigCInputs = {
            **INPUTS_C,
            "item_id_train": np.array([1, 2]),  # y_train has length 4
        }
        with pytest.raises(ValueError, match="Length mismatch"):
            fractional_information_gain_confidence(**inputs)

    def test_y_pred_out_of_range_raises(self) -> None:
        """y_pred_eval > 1 raises ValueError."""
        inputs: FigCInputs = {**INPUTS_C, "y_pred_eval": np.array([0.8, 1.5, 0.7, 0.3])}
        with pytest.raises(ValueError, match="values must be in"):
            fractional_information_gain_confidence(**inputs)

    def test_nan_predictions_raises(self) -> None:
        """NaN in y_pred_eval raises ValueError."""
        inputs: FigCInputs = {
            **INPUTS_C,
            "y_pred_eval": np.array([0.8, np.nan, 0.7, 0.3]),
        }
        with pytest.raises(ValueError, match="NaN or inf"):
            fractional_information_gain_confidence(**inputs)

    def test_mismatched_item_id_eval_length_raises(self) -> None:
        """item_id_eval with wrong length raises ValueError."""
        inputs: FigCInputs = {
            **INPUTS_C,
            "item_id_eval": np.array([1, 2]),
        }  # should be length 4
        with pytest.raises(ValueError, match="Length mismatch"):
            fractional_information_gain_confidence(**inputs)

    def test_mismatched_student_id_eval_length_raises(self) -> None:
        """student_id_eval with wrong length raises ValueError."""
        inputs: FigCInputs = {
            **INPUTS_C,
            "student_id_eval": np.array([1]),
        }  # should be length 4
        with pytest.raises(ValueError, match="Length mismatch"):
            fractional_information_gain_confidence(**inputs)

    def test_probability_clipping(self) -> None:
        """y_pred_eval values of 0.0 or 1.0 do not crash (clipped before log)."""
        inputs: FigCInputs = {**INPUTS_C, "y_pred_eval": np.array([0.0, 1.0, 0.5, 0.5])}
        result = fractional_information_gain_confidence(**inputs)
        assert_fig_c_invariants(result)

    @pytest.mark.parametrize(
        "cast",
        [list, lambda x: np.asarray(x).reshape(-1, 1)],  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType]
        ids=["list", "2d_column"],
    )
    def test_input_types_accepted(self, cast: Any) -> None:  # noqa: ANN401
        """Lists and 2-D column arrays are accepted without error."""
        inputs = {k: cast(v) for k, v in INPUTS_C.items()}
        result = fractional_information_gain_confidence(**inputs)  # pyright: ignore[reportUnknownArgumentType]
        assert_fig_c_invariants(result)


class TestFractionalInformationGainValidation:
    """Tests for fractional_information_gain_validation function (FIG-V)."""

    @pytest.mark.parametrize("seed", [1, 42, 123, 456, 789])
    @pytest.mark.parametrize("use_shrinkage", [False, True])
    def test_matches_reference(self, seed: int, use_shrinkage: bool) -> None:
        """FIG-V output matches the reference implementation on random inputs.

        With use_shrinkage=False, values are compared to the reference implementation.
        With use_shrinkage=True, only invariants are checked (reference does not support
        shrinkage).
        """
        rng = np.random.default_rng(seed)
        n = rng.integers(20, 80)
        inputs: FigVInputs = {
            "y_pred_eval": rng.uniform(0.05, 0.95, n),
            "y_eval": rng.integers(0, 2, n).astype(float),
            "item_id_eval": rng.integers(1, 5, n),
            "student_id_eval": rng.integers(1, 8, n),
            "y_train": rng.integers(0, 2, n * 2).astype(float),
            "item_id_train": rng.integers(1, 5, n * 2),
        }
        result = fractional_information_gain_validation(
            **inputs, use_shrinkage=use_shrinkage
        )
        assert_fig_v_invariants(result)
        if not use_shrinkage:
            ref = fig_v_reference(**inputs)
            assert_fig_v_close(result, ref)

    @pytest.mark.parametrize(
        "inputs_v", [INPUTS_V, INPUTS_V_STR], ids=["int_ids", "str_ids"]
    )
    def test_matches_reference_canonical(self, inputs_v: FigVInputs) -> None:
        """FIG-V matches the reference implementation for both integer and string ID
        fixtures."""
        result = fractional_information_gain_validation(**inputs_v, use_shrinkage=False)
        ref = fig_v_reference(**inputs_v)
        assert_fig_v_invariants(result)
        assert_fig_v_close(result, ref)

    def test_perfect_model_high_fig_v(self) -> None:
        """A model that predicts perfectly should have FIG-V close to 1."""
        # Items with 50% base rate; model is perfectly correct on each.
        inputs: FigVInputs = {
            **INPUTS_V,
            "y_pred_eval": np.array([1 - 1e-9, 1e-9, 1 - 1e-9, 1e-9]),
            "y_train": np.array([1, 0, 1, 0, 1, 0], dtype=np.float64),
            "item_id_train": np.array([1, 1, 1, 2, 2, 2]),
        }
        result = fractional_information_gain_validation(**inputs, use_shrinkage=False)
        assert_fig_v_invariants(result)
        ref = fig_v_reference(**inputs)
        assert_fig_v_close(result, ref)
        assert result["fig_v"] > 0.9
        assert result["fig_v_pooled"] > 0.9

    def test_uninformative_model_low_fig_v(self) -> None:
        """A model predicting 0.5 against a 50/50 baseline should have FIG-V near 0."""
        inputs: FigVInputs = {**INPUTS_V, "y_pred_eval": np.array([0.5, 0.5, 0.5, 0.5])}
        result = fractional_information_gain_validation(**inputs, use_shrinkage=False)
        assert_fig_v_invariants(result)
        ref = fig_v_reference(**inputs)
        assert_fig_v_close(result, ref)
        assert result["fig_v"] == pytest.approx(0)  # pyright: ignore[reportUnknownMemberType]
        assert result["fig_v_pooled"] == pytest.approx(0)  # pyright: ignore[reportUnknownMemberType]

    def test_adversarial_model_negative_fig_v(self) -> None:
        """A model that is confidently wrong should have a strongly negative FIG-V."""
        # Items have 50% base rate. Model predicts ~0 for all correct answers,
        # producing cross-entropy >> baseline entropy.
        inputs: FigVInputs = {
            **INPUTS_V,
            "y_eval": np.array([1, 1, 1, 1]),
            "y_pred_eval": np.array([0.01, 0.01, 0.01, 0.01]),
        }
        result = fractional_information_gain_validation(**inputs, use_shrinkage=False)
        assert_fig_v_invariants(result)
        ref = fig_v_reference(**inputs)
        assert_fig_v_close(result, ref)
        assert result["fig_v"] < -3
        assert result["fig_v_pooled"] < -3

    def test_student_weighted_vs_pooled_differ_with_unequal_students(self) -> None:
        """fig_v and fig_v_pooled differ when students have unequal observation
        counts."""
        # Student 1 has 1 observation, student 2 has 3. Pooled weights by obs count;
        # student-weighted gives them equal weight.
        inputs: FigVInputs = {
            **INPUTS_V,
            "y_eval": np.array([1, 1, 0, 1]),
            "y_pred_eval": np.array([0.9, 0.8, 0.3, 0.7]),
            "student_id_eval": np.array([1, 2, 2, 2]),
        }
        result = fractional_information_gain_validation(**inputs, use_shrinkage=False)
        assert_fig_v_invariants(result)
        assert not np.isclose(result["fig_v"], result["fig_v_pooled"])

    def test_single_student_pooled_equals_student_weighted(self) -> None:
        """With a single student, fig_v_pooled and fig_v should be equal."""
        inputs: FigVInputs = {**INPUTS_V, "student_id_eval": np.array([1, 1, 1, 1])}
        result = fractional_information_gain_validation(**inputs, use_shrinkage=False)
        assert_fig_v_invariants(result)
        assert result["fig_v"] == pytest.approx(result["fig_v_pooled"])  # pyright: ignore[reportUnknownMemberType]

    def test_unseen_item_integration_matches_reference(self) -> None:
        """End-to-end: eval item absent from training uses global-mean fallback and
        the result matches the reference implementation."""
        inputs: FigVInputs = {
            **INPUTS_V,
            "item_id_eval": np.array([1, 1, 999, 999]),
        }
        result = fractional_information_gain_validation(**inputs, use_shrinkage=False)
        ref = fig_v_reference(**inputs)
        assert_fig_v_invariants(result)
        assert_fig_v_close(result, ref)

    def test_single_item_type(self) -> None:
        """FIG-V should be finite and valid when all eval observations share one item
        type."""
        inputs: FigVInputs = {**INPUTS_V, "item_id_eval": np.array([1, 1, 1, 1])}
        result = fractional_information_gain_validation(**inputs, use_shrinkage=False)
        ref = fig_v_reference(**inputs)
        assert_fig_v_invariants(result)
        assert_fig_v_close(result, ref)

    def test_all_correct_responses(self) -> None:
        """FIG-V should be finite and valid when all eval responses are correct."""
        inputs: FigVInputs = {**INPUTS_V, "y_eval": np.array([1, 1, 1, 1], dtype=float)}
        result = fractional_information_gain_validation(**inputs, use_shrinkage=False)
        ref = fig_v_reference(**inputs)
        assert_fig_v_invariants(result)
        assert_fig_v_close(result, ref)

    def test_all_incorrect_responses(self) -> None:
        """FIG-V should be finite and valid when all eval responses are incorrect."""
        inputs: FigVInputs = {**INPUTS_V, "y_eval": np.array([0, 0, 0, 0], dtype=float)}
        result = fractional_information_gain_validation(**inputs, use_shrinkage=False)
        ref = fig_v_reference(**inputs)
        assert_fig_v_invariants(result)
        assert_fig_v_close(result, ref)

    def test_calibration_results(self) -> None:
        """Check that calibration results are found if and only if requested."""
        result_default = fractional_information_gain_validation(**INPUTS_V)
        assert_fig_v_invariants(result_default)
        result_true = fractional_information_gain_validation(
            **INPUTS_V, calibration=True
        )
        assert_fig_v_invariants(result_true, calibration=True)
        result_false = fractional_information_gain_validation(
            **INPUTS_V, calibration=False
        )
        assert_fig_v_invariants(result_false, calibration=False)

    def test_shrinkage_changes_result(self) -> None:
        """use_shrinkage=True and use_shrinkage=False should produce different
        results."""
        # Override training data to give items clearly asymmetric base rates
        # (0.75 and 0.25).
        # Shrinkage pulls these toward the global mean (0.5), changing the baseline
        # and FIG-V.
        inputs: FigVInputs = {
            **INPUTS_V,
            "y_train": np.array([1, 1, 1, 0, 1, 0, 0, 0], dtype=float),
            "item_id_train": np.array([1, 1, 1, 1, 2, 2, 2, 2]),
        }
        result_shrink = fractional_information_gain_validation(
            **inputs, use_shrinkage=True
        )
        result_no_shrink = fractional_information_gain_validation(
            **inputs, use_shrinkage=False
        )
        assert_fig_v_invariants(result_shrink)
        assert_fig_v_invariants(result_no_shrink)
        assert result_shrink["fig_v"] != pytest.approx(result_no_shrink["fig_v"])  # pyright: ignore[reportUnknownMemberType]

    def test_shrinkage_values_match_formula(self) -> None:
        """With known item counts and alpha/beta, FIG-V matches a hand-calculated value.

        Item 1: 3 correct, 0 wrong  → raw = 1.0, shrunk = (3+2)/(3+2+2) = 5/7
        Item 2: 0 correct, 3 wrong  → raw = 0.0, shrunk = (0+2)/(3+2+2) = 2/7

        Eval: one obs per item, y_eval=[1, 0], one student.
        Model predictions: y_pred=[0.9, 0.1] (both correct).

        fig_v_pooled = 1 - (BCE_1 + BCE_2) / (H(5/7) + H(2/7))
        """
        alpha, beta = 2.0, 2.0
        y_train = np.array([1, 1, 1, 0, 0, 0], dtype=float)
        item_id_train = np.array([1, 1, 1, 2, 2, 2])
        y_pred_eval = np.array([0.9, 0.1])
        y_eval = np.array([1.0, 0.0])
        item_id_eval = np.array([1, 2])
        student_id_eval = np.array([1, 1])

        result = fractional_information_gain_validation(
            y_pred_eval=y_pred_eval,
            y_eval=y_eval,
            item_id_eval=item_id_eval,
            student_id_eval=student_id_eval,
            y_train=y_train,
            item_id_train=item_id_train,
            use_shrinkage=True,
            alpha=alpha,
            beta=beta,
        )
        assert_fig_v_invariants(result)

        # Hand-calculate expected fig_v_pooled.
        p1 = (3 + alpha) / (3 + alpha + beta)  # 5/7
        p2 = (0 + alpha) / (3 + alpha + beta)  # 2/7

        def h(p: float) -> float:
            return -(p * np.log(p) + (1 - p) * np.log(1 - p))

        h_base = h(p1) + h(p2)
        bce = -np.log(0.9) - np.log(0.9)
        expected_fig_v_pooled = 1 - bce / h_base

        assert result["fig_v_pooled"] == pytest.approx(expected_fig_v_pooled, abs=1e-10)  # pyright: ignore[reportUnknownMemberType]
        # With one student, fig_v == fig_v_pooled.
        assert result["fig_v"] == pytest.approx(expected_fig_v_pooled, abs=1e-10)  # pyright: ignore[reportUnknownMemberType]

    def test_better_model_has_higher_fig_v(self) -> None:
        """An oracle model should have higher FIG-V than a uniform baseline model."""
        rng = np.random.default_rng(123)
        n_students, n_items, responses_per_student = 50, 10, 30
        n_eval = n_students * responses_per_student

        student_abilities = rng.normal(0, 1.5, n_students)
        item_difficulties = rng.uniform(-1, 1, n_items)

        n_train = 2000
        item_id_train = rng.integers(1, n_items + 1, n_train)
        y_train = rng.binomial(
            1, 1 / (1 + np.exp(-item_difficulties[item_id_train - 1]))
        )

        student_id_eval = np.repeat(np.arange(1, n_students + 1), responses_per_student)
        item_id_eval = rng.integers(1, n_items + 1, n_eval)
        true_probs = 1 / (
            1
            + np.exp(
                -(
                    student_abilities[student_id_eval - 1]
                    + item_difficulties[item_id_eval - 1]
                )
            )
        )
        y_eval = rng.binomial(1, true_probs).astype(float)

        # Create dictionaries of inputs. inputs_good predicts true probabilities;
        # inputs_bad predicts 0.5 for all.
        inputs_good: FigVInputs = {
            "y_pred_eval": true_probs,
            "y_eval": y_eval,
            "item_id_eval": item_id_eval,
            "student_id_eval": student_id_eval,
            "y_train": y_train,
            "item_id_train": item_id_train,
        }
        inputs_bad: FigVInputs = {
            "y_pred_eval": np.ones(n_eval) * 0.5,
            "y_eval": y_eval,
            "item_id_eval": item_id_eval,
            "student_id_eval": student_id_eval,
            "y_train": y_train,
            "item_id_train": item_id_train,
        }

        result_good = fractional_information_gain_validation(
            **inputs_good, use_shrinkage=False
        )
        result_bad = fractional_information_gain_validation(
            **inputs_bad, use_shrinkage=False
        )
        ref_good = fig_v_reference(**inputs_good)
        ref_bad = fig_v_reference(**inputs_bad)

        assert_fig_v_invariants(result_good)
        assert_fig_v_invariants(result_bad)
        assert_fig_v_close(result_good, ref_good)
        assert_fig_v_close(result_bad, ref_bad)
        assert result_good["fig_v"] > result_bad["fig_v"]

    def test_y_pred_out_of_range_raises(self) -> None:
        """y_pred_eval > 1 raises ValueError."""
        inputs: FigVInputs = {**INPUTS_V, "y_pred_eval": np.array([0.8, 1.5, 0.7, 0.3])}
        with pytest.raises(ValueError, match="values must be in"):
            fractional_information_gain_validation(**inputs)

    def test_nan_predictions_raises(self) -> None:
        """NaN in y_pred_eval raises ValueError."""
        inputs: FigVInputs = {
            **INPUTS_V,
            "y_pred_eval": np.array([0.8, np.nan, 0.7, 0.3]),
        }
        with pytest.raises(ValueError, match="NaN or inf"):
            fractional_information_gain_validation(**inputs)

    def test_mismatched_item_id_eval_length_raises(self) -> None:
        """item_id_eval with wrong length raises ValueError."""
        inputs: FigVInputs = {
            **INPUTS_V,
            "item_id_eval": np.array([1, 2]),
        }  # should be length 4
        with pytest.raises(ValueError, match="Length mismatch"):
            fractional_information_gain_validation(**inputs)

    def test_mismatched_student_id_eval_length_raises(self) -> None:
        """student_id_eval with wrong length raises ValueError."""
        inputs: FigVInputs = {
            **INPUTS_V,
            "student_id_eval": np.array([1]),
        }  # should be length 4
        with pytest.raises(ValueError, match="Length mismatch"):
            fractional_information_gain_validation(**inputs)

    def test_mismatched_y_eval_length_raises(self) -> None:
        """y_eval with wrong length raises ValueError."""
        inputs: FigVInputs = {
            **INPUTS_V,
            "y_eval": np.array([1, 0]),
        }  # should be length 4
        with pytest.raises(ValueError, match="Length mismatch"):
            fractional_information_gain_validation(**inputs)

    def test_invalid_y_eval_raises(self) -> None:
        """Non-binary y_eval raises ValueError."""
        inputs: FigVInputs = {**INPUTS_V, "y_eval": np.array([1, 2, 1, 0])}
        with pytest.raises(ValueError, match="must contain only 0 and 1"):
            fractional_information_gain_validation(**inputs)

    def test_invalid_y_train_raises(self) -> None:
        """Non-binary y_train raises ValueError."""
        inputs: FigVInputs = {
            **INPUTS_V,
            "y_train": np.array([1, -1]),
            "item_id_train": np.array([1, 2]),
        }
        with pytest.raises(ValueError, match="must contain only 0 and 1"):
            fractional_information_gain_validation(**inputs)

    def test_empty_train_raises(self) -> None:
        """Empty y_train raises ValueError."""
        inputs: FigVInputs = {
            **INPUTS_V,
            "y_train": np.array([], dtype=float),
            "item_id_train": np.array([], dtype=int),
        }
        with pytest.raises(ValueError, match="must not be empty"):
            fractional_information_gain_validation(**inputs)

    def test_empty_eval_raises(self) -> None:
        """Empty eval arrays raise ValueError."""
        inputs: FigVInputs = {
            **INPUTS_V,
            "y_pred_eval": np.array([], dtype=float),
            "y_eval": np.array([], dtype=float),
            "item_id_eval": np.array([], dtype=int),
            "student_id_eval": np.array([], dtype=int),
        }
        with pytest.raises(ValueError, match="must not be empty"):
            fractional_information_gain_validation(**inputs)

    def test_mismatched_item_id_train_length_raises(self) -> None:
        """item_id_train length != y_train length raises ValueError."""
        inputs: FigVInputs = {
            **INPUTS_V,
            "item_id_train": np.array([1, 2]),  # y_train has length 4
        }
        with pytest.raises(ValueError, match="Length mismatch"):
            fractional_information_gain_validation(**inputs)

    def test_probability_clipping(self) -> None:
        """y_pred_eval values of 0.0 or 1.0 do not crash (clipped before log)."""
        inputs: FigVInputs = {**INPUTS_V, "y_pred_eval": np.array([0.0, 1.0, 0.5, 0.5])}
        result = fractional_information_gain_validation(**inputs)
        assert_fig_v_invariants(result)

    @pytest.mark.parametrize(
        "cast",
        [list, lambda x: np.asarray(x).reshape(-1, 1)],  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType]
        ids=["list", "2d_column"],
    )
    def test_input_types_accepted(self, cast: Any) -> None:  # noqa: ANN401
        """Lists and 2-D column arrays are accepted without error."""
        inputs = {k: cast(v) for k, v in INPUTS_V.items()}
        result = fractional_information_gain_validation(**inputs)  # pyright: ignore[reportUnknownArgumentType]
        assert_fig_v_invariants(result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
