"""Shared utilities for the FIG benchmarking suite."""

import numpy as np

# ---------------------------------------------------------------------------
# Benchmark scales
# ---------------------------------------------------------------------------

# (label, n_eval, n_train, n_items, n_students) triples used by all benchmarks.
SCALES: list[tuple[str, int, int, int, int]] = [
    ("small", 500, 1_500, 50, 20),
    ("medium", 50_000, 150_000, 1_000, 100),
    ("large", 5_000_000, 15_000_000, 1_000, 500),
]


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def make_data(
    n_eval: int,
    n_train: int,
    n_items: int,
    n_students: int,
    seed: int = 0,
) -> dict:
    """Generate synthetic benchmark data.

    Parameters
    ----------
    n_eval : int
        Number of eval observations.
    n_train : int
        Number of train observations.
    n_items : int
        Number of distinct items.
    n_students : int
        Number of distinct students.
    seed : int, default=0
        Random seed for reproducibility.

    Returns
    -------
    dict
        Keys: y_pred_eval, y_eval, item_id_eval, student_id_eval,
              y_train, item_id_train.
        All values are NumPy arrays (float64 for numeric, int64 for IDs).
        y_pred_eval values are in (0.01, 0.99).
    """
    rng = np.random.default_rng(seed)
    y_pred_eval = rng.uniform(0.01, 0.99, size=n_eval)
    y_eval = rng.integers(0, 2, size=n_eval).astype(np.float64)
    item_id_eval = rng.integers(1, n_items + 1, size=n_eval)
    student_id_eval = rng.integers(1, n_students + 1, size=n_eval)
    y_train = rng.integers(0, 2, size=n_train).astype(np.float64)
    item_id_train = rng.integers(1, n_items + 1, size=n_train)
    return {
        "y_pred_eval": y_pred_eval,
        "y_eval": y_eval,
        "item_id_eval": item_id_eval,
        "student_id_eval": student_id_eval,
        "y_train": y_train,
        "item_id_train": item_id_train,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def print_results(rows: list[tuple[str, float]]) -> None:
    """Print a formatted benchmark results table to stdout.

    Parameters
    ----------
    rows : list of (label, min_ms)
        Each row is a (benchmark label, min time in ms) pair.
    """
    if not rows:
        return
    col_width = max(len(label) for label, _ in rows) + 2
    header = f"{'benchmark':<{col_width}}  min (ms)"
    print(header)
    print("-" * len(header))
    for label, ms in rows:
        print(f"{label:<{col_width}}  {ms:>8.3f}")
