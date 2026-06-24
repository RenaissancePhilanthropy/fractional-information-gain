"""Benchmark runner for the FIG package.

Usage
-----
    uv run python -m benchmarks.run
    uv run python -m benchmarks.run --numpy-only
    uv run python -m benchmarks.run --torch-only
    uv run python -m benchmarks.run --scale small
    uv run python -m benchmarks.run --scale small --scale medium
"""

import argparse
import sys

from benchmarks._harness import SCALES, print_results
from benchmarks.bench_numpy import run as run_numpy

_VALID_SCALES = [s[0] for s in SCALES]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FIG benchmarks.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--numpy-only", action="store_true", help="Run only NumPy benchmarks."
    )
    group.add_argument(
        "--torch-only", action="store_true", help="Run only Torch benchmarks."
    )
    parser.add_argument(
        "--scale",
        action="append",
        choices=_VALID_SCALES,
        dest="scales",
        metavar="SCALE",
        help=f"Scale(s) to run ({', '.join(_VALID_SCALES)}). Repeatable. Default: all.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the benchmark CLI."""
    args = _parse_args()
    scales: list[str] | None = args.scales
    rows: list[tuple[str, float]] = []

    if not args.torch_only:
        print("Running NumPy benchmarks...")
        rows.extend(run_numpy(scales=scales))

    if not args.numpy_only:
        try:
            import torch  # noqa: F401, PLC0415
        except ImportError:
            print("torch not installed — skipping Torch benchmarks.", file=sys.stderr)
        else:
            from benchmarks.bench_torch import run as run_torch  # noqa: PLC0415

            print("Running Torch benchmarks...")
            rows.extend(run_torch(scales=scales))

    print()
    print_results(rows)


if __name__ == "__main__":
    main()
