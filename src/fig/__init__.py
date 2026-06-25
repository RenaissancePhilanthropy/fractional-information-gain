"""Fractional Information Gain (FIG) metrics for knowledge-tracing models.

Public API
----------
fractional_information_gain_validation : FIG-V, requires ground truth labels.
fractional_information_gain_confidence : FIG-C, no ground truth needed.
"""

from importlib.metadata import version

from .fig import (
    CalibrationResult,
    FigCResult,
    FigVResult,
    fractional_information_gain_confidence,
    fractional_information_gain_validation,
)

__version__ = version("fractional-information-gain")

__all__ = [
    "CalibrationResult",
    "FigCResult",
    "FigVResult",
    "fractional_information_gain_confidence",
    "fractional_information_gain_validation",
]
