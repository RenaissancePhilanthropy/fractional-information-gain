"""Profile FIG-V using cProfile.

Imports and data generation are excluded from the profiled region; only the
calls to fractional_information_gain_validation[_torch] are profiled.

Usage
-----
    uv run python -m benchmarks._profile                   # all backends, all scales
    uv run python -m benchmarks._profile --numpy-only
    uv run python -m benchmarks._profile --torch-no-precompute-only
    uv run python -m benchmarks._profile --torch-precompute-only
    uv run python -m benchmarks._profile --scale small
    uv run python -m benchmarks._profile --scale small --scale medium

Output
------
    Writes a cProfile stats file to /tmp/fig.prof.
    Visualise with: uvx snakeviz /tmp/fig.prof
"""

import argparse
import cProfile
import sys

from benchmarks._harness import SCALES, make_data
from fig.fig import fractional_information_gain_validation

_VALID_SCALES = [s[0] for s in SCALES]
# S108 complains about a fixed path. This one is intentional and fine in a benchmarking
# script.
_OUTPUT = "/tmp/fig.prof"  # noqa: S108


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile FIG-V.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--numpy-only", action="store_true")
    group.add_argument("--torch-no-precompute-only", action="store_true")
    group.add_argument("--torch-precompute-only", action="store_true")
    parser.add_argument(
        "--scale",
        action="append",
        choices=_VALID_SCALES,
        dest="scales",
        metavar="SCALE",
        help=(
            f"Scale(s) to profile ({', '.join(_VALID_SCALES)})."
            " Repeatable. Default: all."
        ),
    )
    return parser.parse_args()


def _active_scales(scales: list[str] | None) -> list[tuple[str, int, int, int, int]]:
    return [s for s in SCALES if scales is None or s[0] in scales]


def _profile_numpy(scales: list[str] | None, pr: cProfile.Profile) -> None:
    datasets = [
        make_data(n_eval, n_train, n_items, n_students)
        for _, n_eval, n_train, n_items, n_students in _active_scales(scales)
    ]
    for data in datasets:
        pr.enable()
        fractional_information_gain_validation(**data, calibration=False)
        pr.disable()


def _profile_torch(
    scales: list[str] | None,
    pr: cProfile.Profile,
    *,
    precompute_only: bool = False,
    no_precompute_only: bool = False,
) -> None:
    # torch is an optional dependency; deferred to avoid ImportError at module load.
    import torch  # noqa: PLC0415

    from fig.fig_torch import (  # noqa: PLC0415
        fractional_information_gain_validation_torch,
        precompute_baseline_torch,
    )

    datasets = []
    datasets_precomputed = []
    for _, n_eval, n_train, n_items, n_students in _active_scales(scales):
        data = make_data(n_eval, n_train, n_items, n_students)
        data["y_pred_eval"] = torch.tensor(data["y_pred_eval"], dtype=torch.float32)
        data["y_eval"] = torch.tensor(data["y_eval"], dtype=torch.float32)
        datasets.append(data)
        baseline = precompute_baseline_torch(data["y_train"], data["item_id_train"])
        data_precomputed = {**data, "baseline": baseline}
        del data_precomputed["y_train"]
        del data_precomputed["item_id_train"]
        datasets_precomputed.append(data_precomputed)

    if not precompute_only:
        for data in datasets:
            pr.enable()
            fractional_information_gain_validation_torch(**data, calibration=False)
            pr.disable()
    if not no_precompute_only:
        for data in datasets_precomputed:
            pr.enable()
            fractional_information_gain_validation_torch(**data, calibration=False)
            pr.disable()


def main() -> None:
    """Run the profiler CLI."""
    args = _parse_args()
    scales: list[str] | None = args.scales
    pr = cProfile.Profile()

    if not args.torch_no_precompute_only and not args.torch_precompute_only:
        _profile_numpy(scales, pr)

    if not args.numpy_only:
        try:
            # Probe for torch availability; imported only for the ImportError check.
            import torch  # noqa: F401, PLC0415
        except ImportError:
            print("torch not installed — skipping Torch profiling.", file=sys.stderr)
        else:
            _profile_torch(
                scales,
                pr,
                precompute_only=args.torch_precompute_only,
                no_precompute_only=args.torch_no_precompute_only,
            )

    pr.dump_stats(_OUTPUT)


if __name__ == "__main__":
    main()
