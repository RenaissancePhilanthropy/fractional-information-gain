"""NumPy FIG-V and FIG-C benchmarks."""

import timeit

from benchmarks._harness import SCALES, make_data
from fig.fig import (
    fractional_information_gain_confidence,
    fractional_information_gain_validation,
)

_REPEAT = 10  # number of timed trials; min() across them is reported


def run(scales: list[str] | None = None) -> list[tuple[str, float]]:
    """Run NumPy benchmarks and return timing rows.

    Parameters
    ----------
    scales : list of str, optional
        Scale labels to run (e.g. ["small", "medium"]). Default: all scales.

    Returns
    -------
    list of (label, min_ms)
    """
    rows: list[tuple[str, float]] = []
    active_scales = [s for s in SCALES if scales is None or s[0] in scales]

    for scale_label, n_eval, n_train, n_items, n_students in active_scales:
        data_v = make_data(n_eval, n_train, n_items, n_students)
        timing_result = timeit.repeat(
            # B023: avoid late binding of data in the lambda. The warning is a false
            # positive because data is only used within this loop iteration.
            lambda: fractional_information_gain_validation(**data_v, calibration=False),  # noqa: B023
            repeat=_REPEAT,
            number=1,
        )
        ms = min(timing_result) * 1000
        rows.append((f"numpy  FIG-V  {scale_label:<6}  (n={n_eval:>6,})", ms))

        data_c = {k: v for k, v in data_v.items() if k != "y_eval"}
        timing_result = timeit.repeat(
            # B023: avoid late binding of data in the lambda. The warning is a false
            # positive because data is only used within this loop iteration.
            lambda: fractional_information_gain_confidence(**data_c),  # noqa: B023
            repeat=_REPEAT,
            number=1,
        )
        ms = min(timing_result) * 1000
        rows.append((f"numpy  FIG-C  {scale_label:<6}  (n={n_eval:>6,})", ms))

    return rows
