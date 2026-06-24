"""
Tests for the PyTorch implementation of FIG-V and FIG-C (fig_torch.py).
"""

# See comment in src/fig/fig_torch.py for why this is needed.
# pyright: reportPrivateImportUsage=false

from typing import Any

import numpy as np
import pytest
import torch
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

from fig._utils import BaselineData, TrainData
from fig.fig_torch import (
    FigCTorchResult,
    FigVTorchResult,
    fig_c_loss,
    fig_v_loss,
    fractional_information_gain_confidence_torch,
    fractional_information_gain_validation_torch,
    precompute_baseline_torch,
)

# All symbols above are used by the test classes added below.

# Tolerance for comparisons between float32 (torch) and float64 (numpy reference).
# float32 machine epsilon is ~1.2e-7; 1e-6 gives a safe margin.
FLOAT32_TOL = 1e-6


# ---------------------------------------------------------------------------
# Invariant checkers (torch versions)
# ---------------------------------------------------------------------------


def assert_fig_c_invariants_torch(result: FigCTorchResult) -> None:
    """Assert invariants that must hold for any valid FIG-C torch result dict."""
    # Required keys are present.
    assert "fig_c" in result
    assert "fig_c_pooled" in result
    assert "fig_c_by_student" in result
    assert "student_ids" in result

    # Scalar types are tensors.
    assert isinstance(result["fig_c"], torch.Tensor)
    assert isinstance(result["fig_c_pooled"], torch.Tensor)
    assert result["fig_c"].ndim == 0, "fig_c must be a scalar tensor"
    assert result["fig_c_pooled"].ndim == 0, "fig_c_pooled must be a scalar tensor"

    # All values are finite.
    assert torch.isfinite(result["fig_c"]).item()
    assert torch.isfinite(result["fig_c_pooled"]).item()

    # Per-student array has one entry per student.
    assert isinstance(result["fig_c_by_student"], torch.Tensor)
    assert isinstance(result["student_ids"], np.ndarray)
    assert len(result["fig_c_by_student"]) == len(result["student_ids"])
    assert torch.all(torch.isfinite(result["fig_c_by_student"]))

    # FIG-C is bounded above by 1.
    assert result["fig_c"].item() <= 1 + 1e-6
    assert result["fig_c_pooled"].item() <= 1 + 1e-6
    assert all(v <= 1 + 1e-6 for v in result["fig_c_by_student"].tolist())  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportUnknownVariableType]

    # student_ids is sorted.
    assert np.all(result["student_ids"] == np.sort(result["student_ids"]))


def assert_fig_v_invariants_torch(
    result: FigVTorchResult, calibration: bool = False, n_bins: int = 10
) -> None:
    """Assert invariants that must hold for any valid FIG-V torch result dict."""
    assert "fig_v" in result
    assert "fig_v_pooled" in result
    assert "fig_v_by_student" in result
    assert "student_ids" in result

    assert isinstance(result["fig_v"], torch.Tensor)
    assert isinstance(result["fig_v_pooled"], torch.Tensor)
    assert result["fig_v"].ndim == 0
    assert result["fig_v_pooled"].ndim == 0

    assert torch.isfinite(result["fig_v"]).item()
    assert torch.isfinite(result["fig_v_pooled"]).item()

    assert isinstance(result["fig_v_by_student"], torch.Tensor)
    assert isinstance(result["student_ids"], np.ndarray)
    assert len(result["fig_v_by_student"]) == len(result["student_ids"])
    assert torch.all(torch.isfinite(result["fig_v_by_student"]))

    assert result["fig_v"].item() <= 1 + 1e-6
    assert result["fig_v_pooled"].item() <= 1 + 1e-6
    assert all(v <= 1 + 1e-6 for v in result["fig_v_by_student"].tolist())  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportUnknownVariableType]

    assert np.all(result["student_ids"] == np.sort(result["student_ids"]))

    assert "calibration" in result
    if calibration:
        assert result["calibration"] is not None
        assert_calibration_result_invariants(result["calibration"], n_bins=n_bins)
    else:
        assert result["calibration"] is None


def assert_fig_c_close_torch(
    result: FigCTorchResult,
    ref: FigCTorchResult | dict[str, Any],
    tol: float = FLOAT_TOL,
) -> None:
    """Assert that a FIG-C torch result dict agrees with a reference result dict.

    ref may be a numpy reference dict or another torch result dict. Converts
    tensor values to numpy scalars before comparing. Student ordering in
    result and ref may differ; values are matched by student ID.
    """
    np.testing.assert_allclose(result["fig_c"].item(), ref["fig_c"], atol=tol)
    np.testing.assert_allclose(
        result["fig_c_pooled"].item(), ref["fig_c_pooled"], atol=tol
    )
    result_by_student = result["fig_c_by_student"].numpy()
    result_student_ids = result["student_ids"]
    for s, ref_val in zip(ref["student_ids"], ref["fig_c_by_student"], strict=False):
        idx = list(result_student_ids).index(s)
        np.testing.assert_allclose(result_by_student[idx], ref_val, atol=tol)


def assert_fig_v_close_torch(
    result: FigVTorchResult,
    ref: FigVTorchResult | dict[str, Any],
    tol: float = FLOAT_TOL,
) -> None:
    """Assert that a FIG-V torch result dict agrees with a reference result dict.

    ref may be a numpy reference dict or another torch result dict. Converts
    tensor values to numpy scalars before comparing. Student ordering in
    result and ref may differ; values are matched by student ID.
    """
    np.testing.assert_allclose(result["fig_v"].item(), ref["fig_v"], atol=tol)
    np.testing.assert_allclose(
        result["fig_v_pooled"].item(), ref["fig_v_pooled"], atol=tol
    )
    result_by_student = result["fig_v_by_student"].numpy()
    result_student_ids = result["student_ids"]
    for s, ref_val in zip(ref["student_ids"], ref["fig_v_by_student"], strict=False):
        idx = list(result_student_ids).index(s)
        np.testing.assert_allclose(result_by_student[idx], ref_val, atol=tol)


class TestValidateAndPrepareInputsTorch:
    """Error-path tests for validation in the torch public functions."""

    def test_length_mismatch_raises(self) -> None:
        """Mismatched y_pred_eval / item_id_eval lengths raise ValueError."""
        inputs: FigVInputs = {
            **INPUTS_V,
            "y_pred_eval": np.array([0.8, 0.6]),  # length 2, others are 4
        }
        with pytest.raises(ValueError, match="Length mismatch"):
            fractional_information_gain_validation_torch(**inputs)

    def test_prediction_out_of_range_raises(self) -> None:
        """y_pred_eval > 1 raises ValueError."""
        inputs: FigVInputs = {
            **INPUTS_V,
            "y_pred_eval": np.array([0.8, 1.5, 0.7, 0.3]),
        }
        with pytest.raises(ValueError, match="values must be in"):
            fractional_information_gain_validation_torch(**inputs)

    def test_invalid_y_train_raises(self) -> None:
        """Non-binary y_train raises ValueError."""
        inputs: FigVInputs = {
            **INPUTS_V,
            "y_train": np.array([1, -1]),
            "item_id_train": np.array([1, 2]),
        }
        with pytest.raises(ValueError, match="must contain only 0 and 1"):
            fractional_information_gain_validation_torch(**inputs)

    def test_empty_eval_raises(self) -> None:
        """Empty eval arrays raise ValueError."""
        inputs: FigVInputs = {
            **INPUTS_V,
            "y_pred_eval": np.array([]),
            "y_eval": np.array([]),
            "item_id_eval": np.array([]),
            "student_id_eval": np.array([]),
        }
        with pytest.raises(ValueError, match="must not be empty"):
            fractional_information_gain_validation_torch(**inputs)

    def test_empty_train_raises(self) -> None:
        """Empty train arrays raise ValueError."""
        inputs: FigVInputs = {
            **INPUTS_V,
            "y_train": np.array([]),
            "item_id_train": np.array([]),
        }
        with pytest.raises(ValueError, match="must not be empty"):
            fractional_information_gain_validation_torch(**inputs)

    def test_invalid_y_eval_raises(self) -> None:
        """Non-binary y_eval raises ValueError."""
        inputs: FigVInputs = {**INPUTS_V, "y_eval": np.array([1, 2, 1, 0])}
        with pytest.raises(ValueError, match="must contain only 0 and 1"):
            fractional_information_gain_validation_torch(**inputs)

    def test_nan_predictions_raises(self) -> None:
        """NaN in y_pred_eval raises ValueError (finiteness check)."""
        inputs: FigVInputs = {
            **INPUTS_V,
            "y_pred_eval": np.array([0.8, np.nan, 0.7, 0.3]),
        }
        with pytest.raises(ValueError, match="NaN or inf"):
            fractional_information_gain_validation_torch(**inputs)

    def test_input_types_accepted(self) -> None:
        """numpy arrays, Python lists, and torch.Tensors are all accepted."""
        for cast in [np.array, list, torch.tensor]:
            inputs = {k: cast(v) for k, v in INPUTS_V.items()}  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType, reportArgumentType]
            # Should not raise.
            fractional_information_gain_validation_torch(**inputs)  # pyright: ignore[reportUnknownArgumentType, reportArgumentType]

    def test_tensor_item_and_student_ids_accepted(self) -> None:
        """torch.Tensor item_id and student_id are accepted and produce a
        valid result."""
        # Should not raise and should produce a valid result.
        result = fractional_information_gain_validation_torch(
            y_pred_eval=INPUTS_V["y_pred_eval"],
            y_eval=INPUTS_V["y_eval"],
            item_id_eval=torch.tensor(INPUTS_V["item_id_eval"]),
            student_id_eval=torch.tensor(INPUTS_V["student_id_eval"]),
            y_train=INPUTS_V["y_train"],
            item_id_train=torch.tensor(INPUTS_V["item_id_train"]),
        )
        assert torch.isfinite(result["fig_v"])


class TestPrecomputeBaselineTorch:
    """Tests for precompute_baseline_torch."""

    @pytest.mark.parametrize(
        "inputs_c", [INPUTS_C, INPUTS_C_STR], ids=["int_ids", "str_ids"]
    )
    def test_result_structure(self, inputs_c: FigCInputs) -> None:
        """Returns a BaselineData with item_vocab (dict) and item_baselines (float64
        ndarray, length N+1)."""
        baseline = precompute_baseline_torch(
            y_train=inputs_c["y_train"],
            item_id_train=inputs_c["item_id_train"],
            use_shrinkage=False,
        )
        assert isinstance(baseline, BaselineData)
        assert isinstance(baseline.item_vocab, dict)
        assert isinstance(baseline.item_baselines, np.ndarray)
        assert baseline.item_baselines.dtype == np.float64
        assert len(baseline.item_baselines) == len(baseline.item_vocab) + 1

    def test_train_data_accepts_torch_tensors(self) -> None:
        """TrainData correctly coerces torch.Tensor inputs to numpy arrays."""
        t = TrainData(
            y=torch.tensor([1.0, 0.0, 1.0]),
            item_id=torch.tensor([1, 2, 1]),
        )
        assert isinstance(t.y, np.ndarray)
        assert isinstance(t.item_id, np.ndarray)
        assert t.y.dtype == np.float64
        assert list(t.y) == [1.0, 0.0, 1.0]
        assert list(t.item_id) == [1, 2, 1]

        # Tensors with requires_grad=True must also work — np.asarray() raises
        # on these; .detach() is required.
        t2 = TrainData(
            y=torch.tensor([1.0, 0.0, 1.0], requires_grad=True),
            item_id=torch.tensor([1, 2, 1]),
        )
        assert isinstance(t2.y, np.ndarray)
        assert isinstance(t2.item_id, np.ndarray)
        assert t2.y.dtype == np.float64
        assert list(t2.y) == [1.0, 0.0, 1.0]
        assert list(t2.item_id) == [1, 2, 1]

    @pytest.mark.parametrize(
        ("inputs_c", "unseen_item_id"),
        [(INPUTS_C, 99), (INPUTS_C_STR, "item_unseen")],
        ids=["int_ids", "str_ids"],
    )
    def test_unseen_item_falls_back_to_global_mean(
        self, inputs_c: FigCInputs, unseen_item_id: int | str
    ) -> None:
        """An eval item not seen in training gets the global mean as its baseline."""
        baseline = precompute_baseline_torch(
            y_train=inputs_c["y_train"],
            item_id_train=inputs_c["item_id_train"],
            use_shrinkage=False,
        )
        item_id_eval = np.array(
            [
                inputs_c["item_id_eval"][0],
                inputs_c["item_id_eval"][0],
                unseen_item_id,
                unseen_item_id,
            ]
        )
        result = fractional_information_gain_confidence_torch(
            y_pred_eval=inputs_c["y_pred_eval"],
            item_id_eval=item_id_eval,
            student_id_eval=inputs_c["student_id_eval"],
            baseline=baseline,
        )
        assert torch.isfinite(result["fig_c"])
        assert torch.isfinite(result["fig_c_pooled"])

    @pytest.mark.parametrize(
        "inputs_c", [INPUTS_C, INPUTS_C_STR], ids=["int_ids", "str_ids"]
    )
    def test_global_mean_equals_training_mean(self, inputs_c: FigCInputs) -> None:
        """The sentinel entry item_baselines[-1] equals the mean of y_train
        (no shrinkage)."""
        baseline = precompute_baseline_torch(
            y_train=inputs_c["y_train"],
            item_id_train=inputs_c["item_id_train"],
            use_shrinkage=False,
        )
        expected = float(np.mean(inputs_c["y_train"]))
        assert baseline.item_baselines[-1] == pytest.approx(expected)  # pyright: ignore[reportUnknownMemberType]


class TestFractionalInformationGainConfidenceTorch:
    """Tests for fractional_information_gain_confidence_torch (FIG-C)."""

    @pytest.mark.parametrize("seed", [1, 42, 123, 456, 789])
    @pytest.mark.parametrize("use_shrinkage", [False, True])
    def test_matches_reference(self, seed: int, use_shrinkage: bool) -> None:
        """FIG-C torch output matches the numpy reference on random inputs.

        With use_shrinkage=False, values are compared to the reference implementation.
        With use_shrinkage=True, only invariants are checked (reference does not support
        shrinkage).
        """
        rng = np.random.default_rng(seed)
        n = rng.integers(20, 80)
        inputs: FigCInputs = {
            "y_pred_eval": rng.uniform(0.05, 0.95, n),
            "item_id_eval": rng.integers(1, 5, n),
            "student_id_eval": rng.integers(1, 8, n),
            "y_train": rng.integers(0, 2, n * 2).astype(float),
            "item_id_train": rng.integers(1, 5, n * 2),
        }
        result = fractional_information_gain_confidence_torch(
            **inputs, use_shrinkage=use_shrinkage
        )
        assert_fig_c_invariants_torch(result)
        if not use_shrinkage:
            ref = fig_c_reference(**inputs)
            assert_fig_c_close_torch(result, ref, tol=FLOAT32_TOL)

    def test_confident_model_high_fig_c(self) -> None:
        """A model with near-certain predictions should have FIG-C close to 1."""
        inputs: FigCInputs = {
            **INPUTS_C,
            "y_pred_eval": np.array([1 - 1e-6, 1e-6, 1 - 1e-6, 1e-6]),
        }
        result = fractional_information_gain_confidence_torch(
            **inputs, use_shrinkage=False
        )
        ref = fig_c_reference(**inputs)
        assert_fig_c_invariants_torch(result)
        assert_fig_c_close_torch(result, ref, tol=FLOAT32_TOL)
        assert result["fig_c"].item() > 0.9
        assert result["fig_c_pooled"].item() > 0.9

    def test_uninformative_model_low_fig_c(self) -> None:
        """A model predicting 0.5 against a 50/50 baseline should have FIG-C of 0."""
        inputs: FigCInputs = {**INPUTS_C, "y_pred_eval": np.array([0.5, 0.5, 0.5, 0.5])}
        result = fractional_information_gain_confidence_torch(
            **inputs, use_shrinkage=False
        )
        ref = fig_c_reference(**inputs)
        assert_fig_c_invariants_torch(result)
        assert_fig_c_close_torch(result, ref)
        assert result["fig_c"].item() == pytest.approx(0)  # pyright: ignore[reportUnknownMemberType]
        assert result["fig_c_pooled"].item() == pytest.approx(0)  # pyright: ignore[reportUnknownMemberType]

    def test_single_student(self) -> None:
        """With one student, fig_c equals fig_c_pooled and fig_c_by_student has one
        entry."""
        inputs: FigCInputs = {**INPUTS_C, "student_id_eval": np.array([1, 1, 1, 1])}
        result = fractional_information_gain_confidence_torch(**inputs)
        assert_fig_c_invariants_torch(result)
        assert result["fig_c_pooled"].item() == pytest.approx(result["fig_c"].item())  # pyright: ignore[reportUnknownMemberType]
        assert len(result["fig_c_by_student"]) == 1
        assert result["fig_c_by_student"][0].item() == pytest.approx(  # pyright: ignore[reportUnknownMemberType]
            result["fig_c"].item()
        )

    def test_unseen_item_fallback(self) -> None:
        """Items absent from training should fall back to global mean without
        crashing."""
        inputs: FigCInputs = {**INPUTS_C, "item_id_eval": np.array([1, 1, 99, 99])}
        result = fractional_information_gain_confidence_torch(**inputs)
        assert_fig_c_invariants_torch(result)

    @pytest.mark.parametrize(
        "inputs_c", [INPUTS_C, INPUTS_C_STR], ids=["int_ids", "str_ids"]
    )
    def test_precomputed_baseline_matches_inline(self, inputs_c: FigCInputs) -> None:
        """Passing a precomputed baseline gives the same result as inline
        computation."""
        baseline = precompute_baseline_torch(
            y_train=inputs_c["y_train"],
            item_id_train=inputs_c["item_id_train"],
            use_shrinkage=False,
        )
        result_precomputed = fractional_information_gain_confidence_torch(
            y_pred_eval=inputs_c["y_pred_eval"],
            item_id_eval=inputs_c["item_id_eval"],
            student_id_eval=inputs_c["student_id_eval"],
            baseline=baseline,
        )
        result_inline = fractional_information_gain_confidence_torch(
            **inputs_c, use_shrinkage=False
        )
        assert_fig_c_invariants_torch(result_precomputed)
        assert_fig_c_close_torch(result_precomputed, result_inline, tol=FLOAT32_TOL)

    def test_loss_and_grad(self) -> None:
        """fig_c_loss returns negated fig_c; gradients flow through y_pred_eval."""
        y_pred_eval = torch.tensor(
            INPUTS_C["y_pred_eval"], dtype=torch.float32, requires_grad=True
        )
        shared_kwargs = {
            "item_id_eval": INPUTS_C["item_id_eval"],
            "student_id_eval": INPUTS_C["student_id_eval"],
            "y_train": INPUTS_C["y_train"],
            "item_id_train": INPUTS_C["item_id_train"],
        }
        result_dict = fractional_information_gain_confidence_torch(
            y_pred_eval=y_pred_eval,
            **shared_kwargs,  # pyright: ignore[reportArgumentType]
        )
        loss = fig_c_loss(
            y_pred_eval=y_pred_eval,
            **shared_kwargs,  # pyright: ignore[reportArgumentType]
        )
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        np.testing.assert_allclose(loss.item(), -result_dict["fig_c"].item())
        # Gradients flow back through y_pred_eval.
        loss.backward()  # pyright: ignore[reportUnknownMemberType]
        assert y_pred_eval.grad is not None
        assert torch.all(torch.isfinite(y_pred_eval.grad))


class TestFractionalInformationGainValidationTorch:
    """Tests for fractional_information_gain_validation_torch (FIG-V)."""

    @pytest.mark.parametrize("seed", [1, 42, 123, 456, 789])
    @pytest.mark.parametrize("use_shrinkage", [False, True])
    def test_matches_reference(self, seed: int, use_shrinkage: bool) -> None:
        """FIG-V torch output matches the numpy reference on random inputs.

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
        result = fractional_information_gain_validation_torch(
            **inputs, use_shrinkage=use_shrinkage
        )
        assert_fig_v_invariants_torch(result)
        if not use_shrinkage:
            ref = fig_v_reference(**inputs)
            assert_fig_v_close_torch(result, ref, tol=FLOAT32_TOL)

    def test_perfect_model_high_fig_v(self) -> None:
        """A model that predicts perfectly should have FIG-V close to 1."""
        inputs: FigVInputs = {
            **INPUTS_V,
            "y_pred_eval": np.array([1 - 1e-6, 1e-6, 1 - 1e-6, 1e-6]),
        }
        result = fractional_information_gain_validation_torch(
            **inputs, use_shrinkage=False
        )
        ref = fig_v_reference(**inputs)
        assert_fig_v_invariants_torch(result)
        assert_fig_v_close_torch(result, ref, tol=FLOAT32_TOL)
        assert result["fig_v"].item() > 0.9
        assert result["fig_v_pooled"].item() > 0.9

    def test_uninformative_model_low_fig_v(self) -> None:
        """A model predicting 0.5 against a 50/50 baseline should have FIG-V near 0."""
        inputs: FigVInputs = {**INPUTS_V, "y_pred_eval": np.array([0.5, 0.5, 0.5, 0.5])}
        result = fractional_information_gain_validation_torch(
            **inputs, use_shrinkage=False
        )
        ref = fig_v_reference(**inputs)
        assert_fig_v_invariants_torch(result)
        assert_fig_v_close_torch(result, ref)
        assert result["fig_v"].item() == pytest.approx(0)  # pyright: ignore[reportUnknownMemberType]
        assert result["fig_v_pooled"].item() == pytest.approx(0)  # pyright: ignore[reportUnknownMemberType]

    def test_adversarial_model_negative_fig_v(self) -> None:
        """A model that is confidently wrong should have a strongly negative FIG-V."""
        inputs: FigVInputs = {
            **INPUTS_V,
            "y_eval": np.array([1, 1, 1, 1]),
            "y_pred_eval": np.array([0.01, 0.01, 0.01, 0.01]),
        }
        result = fractional_information_gain_validation_torch(
            **inputs, use_shrinkage=False
        )
        ref = fig_v_reference(**inputs)
        assert_fig_v_invariants_torch(result)
        assert_fig_v_close_torch(result, ref)
        assert result["fig_v"].item() < -3
        assert result["fig_v_pooled"].item() < -3

    def test_student_weighted_vs_pooled_differ(self) -> None:
        """fig_v and fig_v_pooled differ when students have unequal observation
        counts."""
        inputs: FigVInputs = {
            **INPUTS_V,
            "y_eval": np.array([1, 1, 0, 1]),
            "y_pred_eval": np.array([0.9, 0.8, 0.3, 0.7]),
            "student_id_eval": np.array([1, 2, 2, 2]),
        }
        # Student 1 has 1 observation, student 2 has 3. Pooled weights by obs count;
        # student-weighted gives equal weight per student.
        result = fractional_information_gain_validation_torch(
            **inputs, use_shrinkage=False
        )
        assert_fig_v_invariants_torch(result)
        assert not np.isclose(result["fig_v"].item(), result["fig_v_pooled"].item())

    def test_single_student_pooled_equals_student_weighted(self) -> None:
        """With a single student, fig_v_pooled and fig_v should be equal."""
        inputs: FigVInputs = {**INPUTS_V, "student_id_eval": np.array([1, 1, 1, 1])}
        result = fractional_information_gain_validation_torch(
            **inputs, use_shrinkage=False
        )
        assert_fig_v_invariants_torch(result)
        assert result["fig_v"].item() == pytest.approx(result["fig_v_pooled"].item())  # pyright: ignore[reportUnknownMemberType]

    def test_unseen_eval_items_use_global_mean(self) -> None:
        """Items absent from training should fall back to global mean without
        crashing."""
        inputs: FigVInputs = {**INPUTS_V, "item_id_eval": np.array([1, 1, 99, 99])}
        result = fractional_information_gain_validation_torch(
            **inputs, use_shrinkage=False
        )
        assert_fig_v_invariants_torch(result)

    def test_single_item_type(self) -> None:
        """FIG-V is finite and valid when all eval observations share one item type."""
        inputs: FigVInputs = {**INPUTS_V, "item_id_eval": np.array([1, 1, 1, 1])}
        result = fractional_information_gain_validation_torch(
            **inputs, use_shrinkage=False
        )
        ref = fig_v_reference(**inputs)
        assert_fig_v_invariants_torch(result)
        assert_fig_v_close_torch(result, ref)

    def test_all_correct_responses(self) -> None:
        """FIG-V is finite and valid when all eval responses are correct."""
        inputs: FigVInputs = {**INPUTS_V, "y_eval": np.array([1, 1, 1, 1], dtype=float)}
        result = fractional_information_gain_validation_torch(
            **inputs, use_shrinkage=False
        )
        ref = fig_v_reference(**inputs)
        assert_fig_v_invariants_torch(result)
        assert_fig_v_close_torch(result, ref, tol=FLOAT32_TOL)

    def test_all_incorrect_responses(self) -> None:
        """FIG-V is finite and valid when all eval responses are incorrect."""
        inputs: FigVInputs = {**INPUTS_V, "y_eval": np.array([0, 0, 0, 0], dtype=float)}
        result = fractional_information_gain_validation_torch(
            **inputs, use_shrinkage=False
        )
        ref = fig_v_reference(**inputs)
        assert_fig_v_invariants_torch(result)
        assert_fig_v_close_torch(result, ref, tol=FLOAT32_TOL)

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

        result_good = fractional_information_gain_validation_torch(
            **inputs_good, use_shrinkage=False
        )
        result_bad = fractional_information_gain_validation_torch(
            **inputs_bad, use_shrinkage=False
        )
        ref_good = fig_v_reference(**inputs_good)
        ref_bad = fig_v_reference(**inputs_bad)
        assert_fig_v_invariants_torch(result_good)
        assert_fig_v_invariants_torch(result_bad)
        assert_fig_v_close_torch(result_good, ref_good, tol=FLOAT32_TOL)
        assert_fig_v_close_torch(result_bad, ref_bad, tol=FLOAT32_TOL)
        assert result_good["fig_v"].item() > result_bad["fig_v"].item()

    def test_calibration_results(self) -> None:
        """Calibration results are included iff calibration=True."""
        result_default = fractional_information_gain_validation_torch(**INPUTS_V)
        assert_fig_v_invariants_torch(result_default)

        result_true = fractional_information_gain_validation_torch(
            **INPUTS_V, calibration=True
        )
        assert_fig_v_invariants_torch(result_true, calibration=True)

        result_false = fractional_information_gain_validation_torch(
            **INPUTS_V, calibration=False
        )
        assert_fig_v_invariants_torch(result_false, calibration=False)

    def test_shrinkage_changes_result(self) -> None:
        """use_shrinkage=True and use_shrinkage=False produce different results."""
        # Training data gives items clearly asymmetric base rates (0.75 and 0.25).
        # Shrinkage pulls these toward the global mean (0.5), changing the baseline.
        inputs: FigVInputs = {
            **INPUTS_V,
            "y_train": np.array([1, 1, 1, 0, 1, 0, 0, 0], dtype=float),
            "item_id_train": np.array([1, 1, 1, 1, 2, 2, 2, 2]),
        }
        result_shrink = fractional_information_gain_validation_torch(
            **inputs, use_shrinkage=True
        )
        result_no_shrink = fractional_information_gain_validation_torch(
            **inputs, use_shrinkage=False
        )
        assert_fig_v_invariants_torch(result_shrink)
        assert_fig_v_invariants_torch(result_no_shrink)
        assert result_shrink["fig_v"].item() != pytest.approx(  # pyright: ignore[reportUnknownMemberType]
            result_no_shrink["fig_v"].item()
        )

    @pytest.mark.parametrize(
        "inputs_v", [INPUTS_V, INPUTS_V_STR], ids=["int_ids", "str_ids"]
    )
    def test_precomputed_baseline_matches_inline(self, inputs_v: FigVInputs) -> None:
        """Passing a precomputed baseline gives the same result as inline
        computation."""
        baseline = precompute_baseline_torch(
            y_train=inputs_v["y_train"],
            item_id_train=inputs_v["item_id_train"],
            use_shrinkage=False,
        )
        result_precomputed = fractional_information_gain_validation_torch(
            y_pred_eval=inputs_v["y_pred_eval"],
            y_eval=inputs_v["y_eval"],
            item_id_eval=inputs_v["item_id_eval"],
            student_id_eval=inputs_v["student_id_eval"],
            baseline=baseline,
        )
        result_inline = fractional_information_gain_validation_torch(
            **inputs_v, use_shrinkage=False
        )
        assert_fig_v_invariants_torch(result_precomputed)
        assert_fig_v_close_torch(result_precomputed, result_inline, tol=FLOAT32_TOL)

    def test_loss_and_grad(self) -> None:
        """fig_v_loss returns negated fig_v; gradients flow through y_pred_eval."""
        y_pred_eval = torch.tensor(
            INPUTS_V["y_pred_eval"], dtype=torch.float32, requires_grad=True
        )
        shared_kwargs = {
            "y_eval": INPUTS_V["y_eval"],
            "item_id_eval": INPUTS_V["item_id_eval"],
            "student_id_eval": INPUTS_V["student_id_eval"],
            "y_train": INPUTS_V["y_train"],
            "item_id_train": INPUTS_V["item_id_train"],
        }
        result_dict = fractional_information_gain_validation_torch(
            y_pred_eval=y_pred_eval,
            **shared_kwargs,  # pyright: ignore[reportArgumentType]
        )
        loss = fig_v_loss(
            y_pred_eval=y_pred_eval,
            **shared_kwargs,  # pyright: ignore[reportArgumentType]
        )
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        np.testing.assert_allclose(loss.item(), -result_dict["fig_v"].item())
        # Gradients flow through the fig_v (student-weighted) path.
        loss.backward()  # pyright: ignore[reportUnknownMemberType]
        assert y_pred_eval.grad is not None
        assert torch.all(torch.isfinite(y_pred_eval.grad))
        # Gradients also flow through the fig_v_pooled (observation-weighted) path.
        y_pred_eval.grad = None
        result_dict["fig_v_pooled"].backward()  # pyright: ignore[reportUnknownMemberType]
        assert y_pred_eval.grad is not None
        assert torch.all(torch.isfinite(y_pred_eval.grad))

    def test_gradient_direction_correct(self) -> None:
        """Loss gradient w.r.t. y_pred_eval is negative when y_eval=1.

        When all true labels are 1, increasing predictions reduces BCE, which
        increases FIG-V, and therefore reduces the loss (-fig_v). The gradient
        must be negative. Uses a single-student setup so pooled and
        student-weighted averages coincide, keeping the direction argument simple.
        """
        y_pred_eval = torch.tensor(
            [0.6, 0.7, 0.5, 0.65], dtype=torch.float32, requires_grad=True
        )
        loss = fig_v_loss(
            y_pred_eval=y_pred_eval,
            y_eval=np.array([1, 1, 1, 1], dtype=float),
            item_id_eval=INPUTS_V["item_id_eval"],
            student_id_eval=np.array([1, 1, 1, 1]),
            y_train=np.array([1, 0, 1, 0], dtype=float),
            item_id_train=INPUTS_V["item_id_train"],
            use_shrinkage=False,
        )
        loss.backward()  # pyright: ignore[reportUnknownMemberType]
        assert y_pred_eval.grad is not None
        assert torch.all(y_pred_eval.grad < 0), (
            f"Expected all gradients < 0 when y_eval=1, got {y_pred_eval.grad}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
