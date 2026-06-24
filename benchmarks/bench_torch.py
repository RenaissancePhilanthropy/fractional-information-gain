"""PyTorch FIG-V benchmarks (forward+backward, with/without precomputed baseline)."""

import timeit

import torch

from benchmarks._harness import SCALES, make_data
from fig.fig_torch import (
    fractional_information_gain_validation_torch,
    precompute_baseline_torch,
)

_REPEAT = 10  # number of timed trials; min() across them is reported


def run(
    device: str = "cpu", scales: list[str] | None = None
) -> list[tuple[str, float]]:
    """Run Torch benchmarks and return timing rows.

    Parameters
    ----------
    device : str, default="cpu"
        Torch device string (e.g. "cpu", "cuda", "mps").
    scales : list of str, optional
        Scale labels to run (e.g. ["small", "medium"]). Default: all scales.

    Returns
    -------
    list of (label, min_ms)
    """
    rows: list[tuple[str, float]] = []
    active_scales = [s for s in SCALES if scales is None or s[0] in scales]

    for scale_label, n_eval, n_train, n_items, n_students in active_scales:
        np_data = make_data(n_eval, n_train, n_items, n_students)

        # Without precomputing of the baseline data.
        data_nopre = {**np_data}
        for k in ("y_pred_eval", "y_eval"):
            data_nopre[k] = torch.tensor(
                data_nopre[k], dtype=torch.float32, device=device
            )

        # The use of default arguments like these, rather than closures, circumvents
        # any possible issues and linter warnings regarding closing over a loop
        # variable. It serves the same purpose as the noqa: B203 in bench_numpy.py.
        def no_precompute_callable(data_nopre: dict = data_nopre) -> None:
            data_nopre["y_pred_eval"] = (
                data_nopre["y_pred_eval"].detach().requires_grad_(True)
            )
            fig = fractional_information_gain_validation_torch(
                **data_nopre, calibration=False
            )
            fig["fig_v"].mul(-1).backward()

        timing_result = timeit.repeat(no_precompute_callable, repeat=_REPEAT, number=1)
        ms = min(timing_result) * 1000
        label = f"torch  FIG-V  {scale_label:<6}  fwd+bwd              (n={n_eval:>6,})"
        rows.append((label, ms))

        # With precomputing of the baseline data.
        data_pre = {**data_nopre}
        data_pre["baseline"] = precompute_baseline_torch(
            data_pre["y_train"], data_pre["item_id_train"]
        )
        del data_pre["y_train"]
        del data_pre["item_id_train"]

        def precompute_callable(data_pre: dict = data_pre) -> None:
            data_pre["y_pred_eval"] = (
                data_pre["y_pred_eval"].detach().requires_grad_(True)
            )
            fig = fractional_information_gain_validation_torch(
                **data_pre, calibration=False
            )
            fig["fig_v"].mul(-1).backward()

        timing_result = timeit.repeat(precompute_callable, repeat=_REPEAT, number=1)
        ms = min(timing_result) * 1000
        label = f"torch  FIG-V  {scale_label:<6}  precomputed fwd+bwd  (n={n_eval:>6,})"
        rows.append((label, ms))

    return rows
