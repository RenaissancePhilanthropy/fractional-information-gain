"""
Cross-language tests to ensure Python (numpy), PyTorch, and R implementations
of FIG-V and FIG-C return the same results for identical inputs.
"""

# See comment in src/fig/fig_torch.py for why this is needed.
# pyright: reportPrivateImportUsage=false

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import FigCInputs, FigVInputs
from numpy.typing import ArrayLike

from fig import (
    fractional_information_gain_confidence,
    fractional_information_gain_validation,
)
from fig.fig_torch import (
    fractional_information_gain_confidence_torch,
    fractional_information_gain_validation_torch,
)

# Check if torch is available.
# The intermediate _has_torch avoids assigning to HAS_TORCH twice, which pyright
# flags as a constant redefinition (uppercase = constant).
try:
    import torch

    _has_torch = True
except ImportError:
    _has_torch = False

HAS_TORCH: bool = _has_torch


# Check if R is available
def check_r_available() -> bool:
    """Check if R is available on the system."""
    try:
        result = subprocess.run(
            ["R", "--version"], capture_output=True, text=True, timeout=10, check=False
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
    else:
        return result.returncode == 0


HAS_R = check_r_available()

# Tolerance for float64 vs float64 comparisons (numpy vs R).
NUMPY_R_TOL = 1e-12

# Tolerance for any comparison involving float32 (numpy vs PyTorch).
# Tighter values risk spurious failures across CPU architectures.
NUMPY_TORCH_TOL = 1e-6


def _assert_per_student_match(
    ids_ref: ArrayLike,
    vals_ref: ArrayLike,
    ids_other: ArrayLike,
    vals_other: ArrayLike,
    tol: float,
    label: str,
) -> None:
    """Assert two (student_ids, per-student value) pairs agree.

    Each side is aligned by sorting on its student_ids before comparison, so this
    does not assume the two implementations emit students in the same order. Both
    arrays are passed through ``np.atleast_1d`` so the single-student case (which
    JSON auto-unboxes to a scalar on the R side) is handled.
    """
    ids_ref = np.atleast_1d(np.asarray(ids_ref, dtype=float))
    ids_other = np.atleast_1d(np.asarray(ids_other, dtype=float))
    vals_ref = np.atleast_1d(np.asarray(vals_ref, dtype=float))
    vals_other = np.atleast_1d(np.asarray(vals_other, dtype=float))

    np.testing.assert_array_equal(
        np.sort(ids_ref),
        np.sort(ids_other),
        err_msg=f"{label}: student_ids differ ({ids_ref} vs {ids_other})",
    )
    np.testing.assert_allclose(
        vals_ref[np.argsort(ids_ref)],
        vals_other[np.argsort(ids_other)],
        atol=tol,
        err_msg=f"{label}: per-student values differ",
    )


def run_r_fig_v(
    y_pred_eval: np.ndarray,
    y_eval: np.ndarray,
    item_id_eval: np.ndarray,
    student_id_eval: np.ndarray,
    y_train: np.ndarray,
    item_id_train: np.ndarray,
    use_shrinkage: bool = True,
    alpha: float = 2.0,
    beta: float = 2.0,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """
    Run the R implementation of FIG-V and return results.

    Uses subprocess to call R with the same inputs.
    """
    # Create temporary files for data exchange
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save inputs to JSON
        input_data = {
            "y_pred_eval": y_pred_eval.tolist(),
            "y_eval": y_eval.tolist(),
            "item_id_eval": item_id_eval.tolist(),
            "student_id_eval": student_id_eval.tolist(),
            "y_train": y_train.tolist(),
            "item_id_train": item_id_train.tolist(),
            "use_shrinkage": use_shrinkage,
            "alpha": alpha,
            "beta": beta,
            "eps": eps,
        }

        input_file = Path(tmpdir) / "input.json"
        output_file = Path(tmpdir) / "output.json"

        with input_file.open("w") as f:
            json.dump(input_data, f)

        # R script to run
        r_script = f"""
library(jsonlite)

# Source the FIG implementation
source("r/R/fig.R")

# Read input data
input_data <- fromJSON("{input_file}")

# Run FIG-V
result <- fractional_information_gain_validation(
    y_pred_eval = input_data$y_pred_eval,
    y_eval = input_data$y_eval,
    item_id_eval = input_data$item_id_eval,
    student_id_eval = input_data$student_id_eval,
    y_train = input_data$y_train,
    item_id_train = input_data$item_id_train,
    use_shrinkage = input_data$use_shrinkage,
    alpha = input_data$alpha,
    beta = input_data$beta,
    eps = input_data$eps,
    calibration = FALSE
)

# Write output with full precision. as.numeric() strips the names from the
# per-student vector so it serialises as a plain array aligned with student_ids.
output_data <- list(
    fig_v = result$fig_v,
    fig_v_pooled = result$fig_v_pooled,
    fig_v_by_student = as.numeric(result$fig_v_by_student),
    student_ids = result$student_ids
)
write_json(output_data, "{output_file}", auto_unbox = TRUE, digits = 17)
"""

        # Get project root directory
        project_root = Path(__file__).resolve().parent.parent

        # Run R script
        result = subprocess.run(
            ["R", "--vanilla", "--slave", "-e", r_script],
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=60,
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"R script failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )

        # Read output
        with output_file.open() as f:
            return json.load(f)


def run_r_fig_c(
    y_pred_eval: np.ndarray,
    item_id_eval: np.ndarray,
    student_id_eval: np.ndarray,
    y_train: np.ndarray,
    item_id_train: np.ndarray,
    use_shrinkage: bool = True,
    alpha: float = 2.0,
    beta: float = 2.0,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """
    Run the R implementation of FIG-C and return results.

    Uses subprocess to call R with the same inputs.
    """
    # Create temporary files for data exchange
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save inputs to JSON (note: no y_eval for FIG-C)
        input_data = {
            "y_pred_eval": y_pred_eval.tolist(),
            "item_id_eval": item_id_eval.tolist(),
            "student_id_eval": student_id_eval.tolist(),
            "y_train": y_train.tolist(),
            "item_id_train": item_id_train.tolist(),
            "use_shrinkage": use_shrinkage,
            "alpha": alpha,
            "beta": beta,
            "eps": eps,
        }

        input_file = Path(tmpdir) / "input.json"
        output_file = Path(tmpdir) / "output.json"

        with input_file.open("w") as f:
            json.dump(input_data, f)

        # R script to run
        r_script = f"""
library(jsonlite)

# Source the FIG implementation
source("r/R/fig.R")

# Read input data
input_data <- fromJSON("{input_file}")

# Run FIG-C
result <- fractional_information_gain_confidence(
    y_pred_eval = input_data$y_pred_eval,
    item_id_eval = input_data$item_id_eval,
    student_id_eval = input_data$student_id_eval,
    y_train = input_data$y_train,
    item_id_train = input_data$item_id_train,
    use_shrinkage = input_data$use_shrinkage,
    alpha = input_data$alpha,
    beta = input_data$beta,
    eps = input_data$eps
)

# Write output with full precision. as.numeric() strips the names from the
# per-student vector so it serialises as a plain array aligned with student_ids.
output_data <- list(
    fig_c = result$fig_c,
    fig_c_pooled = result$fig_c_pooled,
    fig_c_by_student = as.numeric(result$fig_c_by_student),
    student_ids = result$student_ids
)
write_json(output_data, "{output_file}", auto_unbox = TRUE, digits = 17)
"""

        # Get project root directory
        project_root = Path(__file__).resolve().parent.parent

        # Run R script
        result = subprocess.run(
            ["R", "--vanilla", "--slave", "-e", r_script],
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=60,
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"R script failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )

        # Read output
        with output_file.open() as f:
            return json.load(f)


class TestCrossLanguageConsistency:
    """Tests that verify Python, PyTorch, and R implementations produce identical
    results."""

    @pytest.fixture
    def simple_test_data(self) -> FigVInputs:
        """Simple deterministic test data."""
        np.random.seed(42)
        return {
            "y_pred_eval": np.array([0.8, 0.6, 0.7, 0.9, 0.3, 0.5, 0.85, 0.4]),
            "y_eval": np.array([1, 0, 1, 1, 0, 1, 1, 0], dtype=float),
            "item_id_eval": np.array([1, 2, 1, 2, 1, 2, 1, 2]),
            "student_id_eval": np.array([1, 1, 2, 2, 3, 3, 4, 4]),
            "y_train": np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=float),
            "item_id_train": np.array([1, 1, 1, 1, 2, 2, 2, 2]),
        }

    @pytest.fixture
    def larger_test_data(self) -> FigVInputs:
        """Larger random test data."""
        np.random.seed(123)
        n_train = 500
        n_eval = 200
        n_items = 10
        n_students = 20

        return {
            "y_pred_eval": np.random.uniform(0.1, 0.9, n_eval),
            "y_eval": np.random.choice([0, 1], n_eval).astype(float),
            "item_id_eval": np.random.randint(1, n_items + 1, n_eval),
            "student_id_eval": np.random.randint(1, n_students + 1, n_eval),
            "y_train": np.random.choice([0, 1], n_train).astype(float),
            "item_id_train": np.random.randint(1, n_items + 1, n_train),
        }

    @pytest.fixture
    def edge_case_data(self) -> FigVInputs:
        """Edge case with extreme predictions and single student."""
        return {
            "y_pred_eval": np.array([0.01, 0.99, 0.5, 0.5, 0.75]),
            "y_eval": np.array([0, 1, 1, 0, 1], dtype=float),
            "item_id_eval": np.array([1, 1, 2, 2, 3]),
            "student_id_eval": np.array([1, 1, 1, 1, 1]),
            "y_train": np.array([1, 1, 0, 0, 1, 0], dtype=float),
            "item_id_train": np.array([1, 1, 2, 2, 3, 3]),
        }

    def test_numpy_vs_torch_simple(self, simple_test_data: FigVInputs) -> None:
        """Test that numpy and PyTorch implementations match on simple data."""
        if not HAS_TORCH:
            pytest.skip("PyTorch not available")

        # Run numpy version
        result_numpy = fractional_information_gain_validation(
            **simple_test_data, use_shrinkage=True, calibration=False
        )

        # Run PyTorch version
        torch_data: dict[str, Any] = {
            k: torch.tensor(v) if isinstance(v, np.ndarray) else v  # pyright: ignore[reportPossiblyUnboundVariable, reportUnnecessaryIsInstance]
            for k, v in simple_test_data.items()
        }
        result_torch = fractional_information_gain_validation_torch(
            **torch_data,  # pyright: ignore[reportArgumentType]
            use_shrinkage=True,
            calibration=False,
        )

        assert (
            abs(result_numpy["fig_v"] - result_torch["fig_v"].item()) < NUMPY_TORCH_TOL
        ), (
            f"fig_v mismatch: numpy={result_numpy['fig_v']}, "
            f"torch={result_torch['fig_v'].item()}"
        )
        assert (
            abs(result_numpy["fig_v_pooled"] - result_torch["fig_v_pooled"].item())
            < NUMPY_TORCH_TOL
        ), (
            f"fig_v_pooled mismatch: numpy={result_numpy['fig_v_pooled']}, "
            f"torch={result_torch['fig_v_pooled'].item()}"
        )

    def test_numpy_vs_torch_larger(self, larger_test_data: FigVInputs) -> None:
        """Test that numpy and PyTorch implementations match on larger data."""
        if not HAS_TORCH:
            pytest.skip("PyTorch not available")

        # Run numpy version
        result_numpy = fractional_information_gain_validation(
            **larger_test_data, use_shrinkage=True, calibration=False
        )

        # Run PyTorch version
        torch_data: dict[str, Any] = {
            k: torch.tensor(v) if isinstance(v, np.ndarray) else v  # pyright: ignore[reportPossiblyUnboundVariable, reportUnnecessaryIsInstance]
            for k, v in larger_test_data.items()
        }
        result_torch = fractional_information_gain_validation_torch(
            **torch_data,  # pyright: ignore[reportArgumentType]
            use_shrinkage=True,
            calibration=False,
        )

        assert (
            abs(result_numpy["fig_v"] - result_torch["fig_v"].item()) < NUMPY_TORCH_TOL
        ), (
            f"fig_v mismatch: numpy={result_numpy['fig_v']}, "
            f"torch={result_torch['fig_v'].item()}"
        )
        assert (
            abs(result_numpy["fig_v_pooled"] - result_torch["fig_v_pooled"].item())
            < NUMPY_TORCH_TOL
        ), (
            f"fig_v_pooled mismatch: numpy={result_numpy['fig_v_pooled']}, "
            f"torch={result_torch['fig_v_pooled'].item()}"
        )

    def test_numpy_vs_torch_no_shrinkage(self, simple_test_data: FigVInputs) -> None:
        """Test numpy vs PyTorch without shrinkage."""
        if not HAS_TORCH:
            pytest.skip("PyTorch not available")

        # Run numpy version
        result_numpy = fractional_information_gain_validation(
            **simple_test_data, use_shrinkage=False, calibration=False
        )

        # Run PyTorch version
        torch_data: dict[str, Any] = {
            k: torch.tensor(v) if isinstance(v, np.ndarray) else v  # pyright: ignore[reportPossiblyUnboundVariable, reportUnnecessaryIsInstance]
            for k, v in simple_test_data.items()
        }
        result_torch = fractional_information_gain_validation_torch(
            **torch_data,  # pyright: ignore[reportArgumentType]
            use_shrinkage=False,
            calibration=False,
        )

        assert (
            abs(result_numpy["fig_v"] - result_torch["fig_v"].item()) < NUMPY_TORCH_TOL
        )
        assert (
            abs(result_numpy["fig_v_pooled"] - result_torch["fig_v_pooled"].item())
            < NUMPY_TORCH_TOL
        )

    def test_numpy_vs_torch_edge_cases(self, edge_case_data: FigVInputs) -> None:
        """Test numpy vs PyTorch on edge cases."""
        if not HAS_TORCH:
            pytest.skip("PyTorch not available")

        # Run numpy version
        result_numpy = fractional_information_gain_validation(
            **edge_case_data, use_shrinkage=True, calibration=False
        )

        # Run PyTorch version
        torch_data: dict[str, Any] = {
            k: torch.tensor(v) if isinstance(v, np.ndarray) else v  # pyright: ignore[reportPossiblyUnboundVariable, reportUnnecessaryIsInstance]
            for k, v in edge_case_data.items()
        }
        result_torch = fractional_information_gain_validation_torch(
            **torch_data,  # pyright: ignore[reportArgumentType]
            use_shrinkage=True,
            calibration=False,
        )

        assert (
            abs(result_numpy["fig_v"] - result_torch["fig_v"].item()) < NUMPY_TORCH_TOL
        )
        assert (
            abs(result_numpy["fig_v_pooled"] - result_torch["fig_v_pooled"].item())
            < NUMPY_TORCH_TOL
        )

    @pytest.mark.skipif(not HAS_R, reason="R not available")
    def test_numpy_vs_r_simple(self, simple_test_data: FigVInputs) -> None:
        """Test that numpy and R implementations match on simple data."""
        # Run numpy version
        result_numpy = fractional_information_gain_validation(
            **simple_test_data, use_shrinkage=True, calibration=False
        )

        # Run R version
        result_r = run_r_fig_v(**simple_test_data, use_shrinkage=True)

        assert abs(result_numpy["fig_v"] - result_r["fig_v"]) < NUMPY_R_TOL, (
            f"fig_v mismatch: numpy={result_numpy['fig_v']}, R={result_r['fig_v']}"
        )
        assert (
            abs(result_numpy["fig_v_pooled"] - result_r["fig_v_pooled"]) < NUMPY_R_TOL
        ), (
            f"fig_v_pooled mismatch: numpy={result_numpy['fig_v_pooled']}, "
            f"R={result_r['fig_v_pooled']}"
        )

    @pytest.mark.skipif(not HAS_R, reason="R not available")
    def test_numpy_vs_r_larger(self, larger_test_data: FigVInputs) -> None:
        """Test that numpy and R implementations match on larger data."""
        # Run numpy version
        result_numpy = fractional_information_gain_validation(
            **larger_test_data, use_shrinkage=True, calibration=False
        )

        # Run R version
        result_r = run_r_fig_v(**larger_test_data, use_shrinkage=True)

        assert abs(result_numpy["fig_v"] - result_r["fig_v"]) < NUMPY_R_TOL, (
            f"fig_v mismatch: numpy={result_numpy['fig_v']}, R={result_r['fig_v']}"
        )
        assert (
            abs(result_numpy["fig_v_pooled"] - result_r["fig_v_pooled"]) < NUMPY_R_TOL
        ), (
            f"fig_v_pooled mismatch: numpy={result_numpy['fig_v_pooled']}, "
            f"R={result_r['fig_v_pooled']}"
        )

    @pytest.mark.skipif(not HAS_R, reason="R not available")
    def test_numpy_vs_r_no_shrinkage(self, simple_test_data: FigVInputs) -> None:
        """Test numpy vs R without shrinkage."""
        # Run numpy version
        result_numpy = fractional_information_gain_validation(
            **simple_test_data, use_shrinkage=False, calibration=False
        )

        # Run R version
        result_r = run_r_fig_v(**simple_test_data, use_shrinkage=False)

        assert abs(result_numpy["fig_v"] - result_r["fig_v"]) < NUMPY_R_TOL
        assert (
            abs(result_numpy["fig_v_pooled"] - result_r["fig_v_pooled"]) < NUMPY_R_TOL
        )

    @pytest.mark.skipif(not HAS_R, reason="R not available")
    def test_numpy_vs_r_edge_cases(self, edge_case_data: FigVInputs) -> None:
        """Test numpy vs R on edge cases."""
        # Run numpy version
        result_numpy = fractional_information_gain_validation(
            **edge_case_data, use_shrinkage=True, calibration=False
        )

        # Run R version
        result_r = run_r_fig_v(**edge_case_data, use_shrinkage=True)

        assert abs(result_numpy["fig_v"] - result_r["fig_v"]) < NUMPY_R_TOL
        assert (
            abs(result_numpy["fig_v_pooled"] - result_r["fig_v_pooled"]) < NUMPY_R_TOL
        )

    @pytest.mark.skipif(not HAS_TORCH or not HAS_R, reason="PyTorch or R not available")
    def test_all_three_implementations_simple(
        self, simple_test_data: FigVInputs
    ) -> None:
        """Test that all three implementations (numpy, torch, R) match."""
        # Run numpy version
        result_numpy = fractional_information_gain_validation(
            **simple_test_data, use_shrinkage=True, calibration=False
        )

        # Run PyTorch version
        torch_data: dict[str, Any] = {
            k: torch.tensor(v) if isinstance(v, np.ndarray) else v  # pyright: ignore[reportPossiblyUnboundVariable, reportUnnecessaryIsInstance]
            for k, v in simple_test_data.items()
        }
        result_torch = fractional_information_gain_validation_torch(
            **torch_data,  # pyright: ignore[reportArgumentType]
            use_shrinkage=True,
            calibration=False,
        )

        # Run R version
        result_r = run_r_fig_v(**simple_test_data, use_shrinkage=True)

        # Compare all three
        fig_v_numpy = result_numpy["fig_v"]
        fig_v_torch = result_torch["fig_v"].item()
        fig_v_r = result_r["fig_v"]

        fig_v_pooled_numpy = result_numpy["fig_v_pooled"]
        fig_v_pooled_torch = result_torch["fig_v_pooled"].item()
        fig_v_pooled_r = result_r["fig_v_pooled"]

        assert abs(fig_v_numpy - fig_v_torch) < NUMPY_TORCH_TOL, (
            f"numpy vs torch fig_v: {fig_v_numpy} vs {fig_v_torch}"
        )
        assert abs(fig_v_numpy - fig_v_r) < NUMPY_R_TOL, (
            f"numpy vs R fig_v: {fig_v_numpy} vs {fig_v_r}"
        )
        assert abs(fig_v_torch - fig_v_r) < NUMPY_TORCH_TOL, (
            f"torch vs R fig_v: {fig_v_torch} vs {fig_v_r}"
        )

        assert abs(fig_v_pooled_numpy - fig_v_pooled_torch) < NUMPY_TORCH_TOL
        assert abs(fig_v_pooled_numpy - fig_v_pooled_r) < NUMPY_R_TOL
        assert abs(fig_v_pooled_torch - fig_v_pooled_r) < NUMPY_TORCH_TOL

    @pytest.mark.skipif(not HAS_TORCH or not HAS_R, reason="PyTorch or R not available")
    def test_all_three_implementations_larger(
        self, larger_test_data: FigVInputs
    ) -> None:
        """Test all three implementations on larger data."""
        # Run numpy version
        result_numpy = fractional_information_gain_validation(
            **larger_test_data, use_shrinkage=True, calibration=False
        )

        # Run PyTorch version
        torch_data: dict[str, Any] = {
            k: torch.tensor(v) if isinstance(v, np.ndarray) else v  # pyright: ignore[reportPossiblyUnboundVariable, reportUnnecessaryIsInstance]
            for k, v in larger_test_data.items()
        }
        result_torch = fractional_information_gain_validation_torch(
            **torch_data,  # pyright: ignore[reportArgumentType]
            use_shrinkage=True,
            calibration=False,
        )

        # Run R version
        result_r = run_r_fig_v(**larger_test_data, use_shrinkage=True)

        assert (
            abs(result_numpy["fig_v"] - result_torch["fig_v"].item()) < NUMPY_TORCH_TOL
        )
        assert abs(result_numpy["fig_v"] - result_r["fig_v"]) < NUMPY_R_TOL
        assert (
            abs(result_numpy["fig_v_pooled"] - result_torch["fig_v_pooled"].item())
            < NUMPY_TORCH_TOL
        )
        assert (
            abs(result_numpy["fig_v_pooled"] - result_r["fig_v_pooled"]) < NUMPY_R_TOL
        )

    @pytest.mark.skipif(not HAS_R, reason="R not available")
    def test_r_with_different_shrinkage_params(
        self, simple_test_data: FigVInputs
    ) -> None:
        """Test R implementation with different alpha/beta shrinkage parameters."""
        for alpha, beta in [(1.0, 1.0), (2.0, 2.0), (0.5, 0.5), (3.0, 1.0)]:
            # Run numpy version
            result_numpy = fractional_information_gain_validation(
                **simple_test_data,
                use_shrinkage=True,
                alpha=alpha,
                beta=beta,
                calibration=False,
            )

            # Run R version
            result_r = run_r_fig_v(
                **simple_test_data, use_shrinkage=True, alpha=alpha, beta=beta
            )

            assert abs(result_numpy["fig_v"] - result_r["fig_v"]) < NUMPY_R_TOL, (
                f"Mismatch with alpha={alpha}, beta={beta}"
            )
            assert (
                abs(result_numpy["fig_v_pooled"] - result_r["fig_v_pooled"])
                < NUMPY_R_TOL
            )

    @pytest.mark.parametrize(
        "data_fixture",
        ["simple_test_data", "larger_test_data", "edge_case_data"],
    )
    @pytest.mark.parametrize(
        ("backend", "tol"),
        [
            pytest.param(
                "R",
                NUMPY_R_TOL,
                marks=pytest.mark.skipif(not HAS_R, reason="R not available"),
            ),
            pytest.param(
                "torch",
                NUMPY_TORCH_TOL,
                marks=pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available"),
            ),
        ],
    )
    def test_fig_v_per_student_parity(
        self,
        data_fixture: str,
        backend: str,
        tol: float,
        request: pytest.FixtureRequest,
    ) -> None:
        """numpy agrees with R and PyTorch on per-student FIG-V and student_ids.

        Covers the scenario x backend matrix. This is the regression guard for
        FIG-V's per-student outputs: before they were returned, the other side had
        nothing to compare against and this would fail on a shape mismatch.
        """
        data: FigVInputs = request.getfixturevalue(data_fixture)
        result_numpy = fractional_information_gain_validation(
            **data, use_shrinkage=True, calibration=False
        )
        if backend == "R":
            result_r = run_r_fig_v(**data, use_shrinkage=True)
            other_ids = result_r["student_ids"]
            other_vals = result_r["fig_v_by_student"]
        else:
            torch_data: dict[str, Any] = {
                k: torch.tensor(v) if isinstance(v, np.ndarray) else v  # pyright: ignore[reportPossiblyUnboundVariable, reportUnnecessaryIsInstance]
                for k, v in data.items()
            }
            result_torch = fractional_information_gain_validation_torch(
                **torch_data,  # pyright: ignore[reportArgumentType]
                use_shrinkage=True,
                calibration=False,
            )
            other_ids = result_torch["student_ids"]
            other_vals = result_torch["fig_v_by_student"].detach().cpu().numpy()
        _assert_per_student_match(
            result_numpy["student_ids"],
            result_numpy["fig_v_by_student"],
            other_ids,
            other_vals,
            tol,
            f"FIG-V numpy vs {backend} ({data_fixture})",
        )


class TestCrossLanguageMultipleSeeds:
    """Run cross-language tests with multiple random seeds."""

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    @pytest.mark.parametrize("seed", [1, 42, 123, 456, 789])
    def test_numpy_torch_multiple_seeds(self, seed: int) -> None:
        """Test numpy vs torch with multiple random seeds."""
        np.random.seed(seed)
        n = 100

        data: FigVInputs = {
            "y_pred_eval": np.random.uniform(0.1, 0.9, n),
            "y_eval": np.random.choice([0, 1], n).astype(float),
            "item_id_eval": np.random.randint(1, 6, n),
            "student_id_eval": np.random.randint(1, 11, n),
            "y_train": np.random.choice([0, 1], n * 2).astype(float),
            "item_id_train": np.random.randint(1, 6, n * 2),
        }

        result_numpy = fractional_information_gain_validation(**data, calibration=False)

        torch_data: dict[str, Any] = {
            k: torch.tensor(v) if isinstance(v, np.ndarray) else v  # pyright: ignore[reportPossiblyUnboundVariable, reportUnnecessaryIsInstance]
            for k, v in data.items()
        }
        result_torch = fractional_information_gain_validation_torch(
            **torch_data,  # pyright: ignore[reportArgumentType]
            calibration=False,
        )

        assert (
            abs(result_numpy["fig_v"] - result_torch["fig_v"].item()) < NUMPY_TORCH_TOL
        )
        assert (
            abs(result_numpy["fig_v_pooled"] - result_torch["fig_v_pooled"].item())
            < NUMPY_TORCH_TOL
        )

    @pytest.mark.skipif(not HAS_R, reason="R not available")
    @pytest.mark.parametrize("seed", [1, 42, 123])
    def test_numpy_r_multiple_seeds(self, seed: int) -> None:
        """Test numpy vs R with multiple random seeds."""
        np.random.seed(seed)
        n = 100

        data: FigVInputs = {
            "y_pred_eval": np.random.uniform(0.1, 0.9, n),
            "y_eval": np.random.choice([0, 1], n).astype(float),
            "item_id_eval": np.random.randint(1, 6, n),
            "student_id_eval": np.random.randint(1, 11, n),
            "y_train": np.random.choice([0, 1], n * 2).astype(float),
            "item_id_train": np.random.randint(1, 6, n * 2),
        }

        result_numpy = fractional_information_gain_validation(**data, calibration=False)
        result_r = run_r_fig_v(**data)

        assert abs(result_numpy["fig_v"] - result_r["fig_v"]) < NUMPY_R_TOL
        assert (
            abs(result_numpy["fig_v_pooled"] - result_r["fig_v_pooled"]) < NUMPY_R_TOL
        )


# ==============================================================================
# FIG-C Cross-Language Tests
# ==============================================================================


class TestCrossLanguageConsistencyFigC:
    """Tests that verify Python, PyTorch, and R FIG-C implementations produce identical
    results."""

    @pytest.fixture
    def simple_test_data_fig_c(self) -> FigCInputs:
        """Simple test dataset for FIG-C (no y_eval needed)."""
        np.random.seed(42)
        return {
            "y_pred_eval": np.array([0.8, 0.3, 0.9, 0.2, 0.7, 0.4, 0.85, 0.15]),
            "item_id_eval": np.array([1, 1, 2, 2, 3, 3, 1, 2]),
            "student_id_eval": np.array([1, 1, 1, 2, 2, 2, 3, 3]),
            "y_train": np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 0]),
            "item_id_train": np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 3]),
        }

    @pytest.fixture
    def larger_test_data_fig_c(self) -> FigCInputs:
        """Larger test dataset for FIG-C."""
        np.random.seed(123)
        n_eval = 200
        n_train = 500
        n_items = 10
        n_students = 20

        return {
            "y_pred_eval": np.random.uniform(0.1, 0.9, n_eval),
            "item_id_eval": np.random.randint(1, n_items + 1, n_eval),
            "student_id_eval": np.random.randint(1, n_students + 1, n_eval),
            "y_train": np.random.choice([0, 1], n_train).astype(float),
            "item_id_train": np.random.randint(1, n_items + 1, n_train),
        }

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_numpy_vs_torch_fig_c(self, simple_test_data_fig_c: FigCInputs) -> None:
        """Test that numpy and PyTorch FIG-C implementations produce same results."""
        data = simple_test_data_fig_c

        # NumPy version
        result_numpy = fractional_information_gain_confidence(**data)

        # PyTorch version
        torch_data: dict[str, Any] = {
            k: torch.tensor(v) if isinstance(v, np.ndarray) else v  # pyright: ignore[reportPossiblyUnboundVariable, reportUnnecessaryIsInstance]
            for k, v in data.items()
        }
        result_torch = fractional_information_gain_confidence_torch(**torch_data)  # pyright: ignore[reportArgumentType]

        assert (
            abs(result_numpy["fig_c"] - result_torch["fig_c"].item()) < NUMPY_TORCH_TOL
        ), (
            f"fig_c mismatch: numpy={result_numpy['fig_c']}, "
            f"torch={result_torch['fig_c'].item()}"
        )
        assert (
            abs(result_numpy["fig_c_pooled"] - result_torch["fig_c_pooled"].item())
            < NUMPY_TORCH_TOL
        ), (
            f"fig_c_pooled mismatch: numpy={result_numpy['fig_c_pooled']}, "
            f"torch={result_torch['fig_c_pooled'].item()}"
        )

    @pytest.mark.skipif(not HAS_R, reason="R not available")
    def test_numpy_vs_r_fig_c(self, simple_test_data_fig_c: FigCInputs) -> None:
        """Test that numpy and R FIG-C implementations produce same results."""
        data = simple_test_data_fig_c

        # NumPy version
        result_numpy = fractional_information_gain_confidence(**data)

        # R version
        result_r = run_r_fig_c(**data)

        assert abs(result_numpy["fig_c"] - result_r["fig_c"]) < NUMPY_R_TOL, (
            f"fig_c mismatch: numpy={result_numpy['fig_c']}, R={result_r['fig_c']}"
        )
        assert (
            abs(result_numpy["fig_c_pooled"] - result_r["fig_c_pooled"]) < NUMPY_R_TOL
        ), (
            f"fig_c_pooled mismatch: numpy={result_numpy['fig_c_pooled']}, "
            f"R={result_r['fig_c_pooled']}"
        )

    @pytest.mark.parametrize(
        "data_fixture",
        ["simple_test_data_fig_c", "larger_test_data_fig_c"],
    )
    @pytest.mark.parametrize(
        ("backend", "tol"),
        [
            pytest.param(
                "R",
                NUMPY_R_TOL,
                marks=pytest.mark.skipif(not HAS_R, reason="R not available"),
            ),
            pytest.param(
                "torch",
                NUMPY_TORCH_TOL,
                marks=pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available"),
            ),
        ],
    )
    def test_fig_c_per_student_parity(
        self,
        data_fixture: str,
        backend: str,
        tol: float,
        request: pytest.FixtureRequest,
    ) -> None:
        """numpy agrees with R and PyTorch on per-student FIG-C and student_ids.

        Covers the scenario x backend matrix (regression guard for FIG-C's
        per-student outputs).
        """
        data: FigCInputs = request.getfixturevalue(data_fixture)
        result_numpy = fractional_information_gain_confidence(**data)
        if backend == "R":
            result_r = run_r_fig_c(**data)
            other_ids = result_r["student_ids"]
            other_vals = result_r["fig_c_by_student"]
        else:
            torch_data: dict[str, Any] = {
                k: torch.tensor(v) if isinstance(v, np.ndarray) else v  # pyright: ignore[reportPossiblyUnboundVariable, reportUnnecessaryIsInstance]
                for k, v in data.items()
            }
            result_torch = fractional_information_gain_confidence_torch(**torch_data)  # pyright: ignore[reportArgumentType]
            other_ids = result_torch["student_ids"]
            other_vals = result_torch["fig_c_by_student"].detach().cpu().numpy()
        _assert_per_student_match(
            result_numpy["student_ids"],
            result_numpy["fig_c_by_student"],
            other_ids,
            other_vals,
            tol,
            f"FIG-C numpy vs {backend} ({data_fixture})",
        )

    @pytest.mark.skipif(not HAS_TORCH or not HAS_R, reason="PyTorch or R not available")
    def test_all_implementations_fig_c(
        self, larger_test_data_fig_c: FigCInputs
    ) -> None:
        """Test that all FIG-C implementations produce same results on larger
        dataset."""
        data = larger_test_data_fig_c

        # NumPy version
        result_numpy = fractional_information_gain_confidence(**data)

        # PyTorch version
        torch_data: dict[str, Any] = {
            k: torch.tensor(v) if isinstance(v, np.ndarray) else v  # pyright: ignore[reportPossiblyUnboundVariable, reportUnnecessaryIsInstance]
            for k, v in data.items()
        }
        result_torch = fractional_information_gain_confidence_torch(**torch_data)  # pyright: ignore[reportArgumentType]

        # R version
        result_r = run_r_fig_c(**data)

        # Compare NumPy vs PyTorch
        assert (
            abs(result_numpy["fig_c"] - result_torch["fig_c"].item()) < NUMPY_TORCH_TOL
        )
        assert (
            abs(result_numpy["fig_c_pooled"] - result_torch["fig_c_pooled"].item())
            < NUMPY_TORCH_TOL
        )

        # Compare NumPy vs R
        assert abs(result_numpy["fig_c"] - result_r["fig_c"]) < NUMPY_R_TOL
        assert (
            abs(result_numpy["fig_c_pooled"] - result_r["fig_c_pooled"]) < NUMPY_R_TOL
        )

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_fig_c_with_shrinkage_options(
        self, simple_test_data_fig_c: FigCInputs
    ) -> None:
        """Test FIG-C with and without shrinkage."""
        data = simple_test_data_fig_c

        for use_shrinkage in [True, False]:
            result_numpy = fractional_information_gain_confidence(
                **data, use_shrinkage=use_shrinkage
            )

            torch_data: dict[str, Any] = {
                k: torch.tensor(v) if isinstance(v, np.ndarray) else v  # pyright: ignore[reportPossiblyUnboundVariable, reportUnnecessaryIsInstance]
                for k, v in data.items()
            }
            result_torch = fractional_information_gain_confidence_torch(
                **torch_data,  # pyright: ignore[reportArgumentType]
                use_shrinkage=use_shrinkage,
            )

            assert (
                abs(result_numpy["fig_c"] - result_torch["fig_c"].item())
                < NUMPY_TORCH_TOL
            )
            assert (
                abs(result_numpy["fig_c_pooled"] - result_torch["fig_c_pooled"].item())
                < NUMPY_TORCH_TOL
            )


class TestCrossLanguageMultipleSeedsFigC:
    """Parametrized tests with multiple random seeds for FIG-C."""

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    @pytest.mark.parametrize("seed", [1, 42, 123])
    def test_numpy_torch_fig_c_multiple_seeds(self, seed: int) -> None:
        """Test numpy vs torch FIG-C with multiple random seeds."""
        np.random.seed(seed)
        n = 100

        data: FigCInputs = {
            "y_pred_eval": np.random.uniform(0.1, 0.9, n),
            "item_id_eval": np.random.randint(1, 6, n),
            "student_id_eval": np.random.randint(1, 11, n),
            "y_train": np.random.choice([0, 1], n * 2).astype(float),
            "item_id_train": np.random.randint(1, 6, n * 2),
        }

        result_numpy = fractional_information_gain_confidence(**data)

        torch_data: dict[str, Any] = {
            k: torch.tensor(v) if isinstance(v, np.ndarray) else v  # pyright: ignore[reportPossiblyUnboundVariable, reportUnnecessaryIsInstance]
            for k, v in data.items()
        }
        result_torch = fractional_information_gain_confidence_torch(**torch_data)  # pyright: ignore[reportArgumentType]

        assert (
            abs(result_numpy["fig_c"] - result_torch["fig_c"].item()) < NUMPY_TORCH_TOL
        )
        assert (
            abs(result_numpy["fig_c_pooled"] - result_torch["fig_c_pooled"].item())
            < NUMPY_TORCH_TOL
        )

    @pytest.mark.skipif(not HAS_R, reason="R not available")
    @pytest.mark.parametrize("seed", [1, 42, 123])
    def test_numpy_r_fig_c_multiple_seeds(self, seed: int) -> None:
        """Test numpy vs R FIG-C with multiple random seeds."""
        np.random.seed(seed)
        n = 100

        data: FigCInputs = {
            "y_pred_eval": np.random.uniform(0.1, 0.9, n),
            "item_id_eval": np.random.randint(1, 6, n),
            "student_id_eval": np.random.randint(1, 11, n),
            "y_train": np.random.choice([0, 1], n * 2).astype(float),
            "item_id_train": np.random.randint(1, 6, n * 2),
        }

        result_numpy = fractional_information_gain_confidence(**data)
        result_r = run_r_fig_c(**data)

        assert abs(result_numpy["fig_c"] - result_r["fig_c"]) < NUMPY_R_TOL
        assert (
            abs(result_numpy["fig_c_pooled"] - result_r["fig_c_pooled"]) < NUMPY_R_TOL
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
