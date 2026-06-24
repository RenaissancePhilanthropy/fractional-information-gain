"""Fractional Information Gain (FIG) metrics for knowledge-tracing models.

Public API
----------
fractional_information_gain_validation : FIG-V, requires ground truth labels.
fractional_information_gain_confidence : FIG-C, no ground truth needed.
"""

from .fig import (
    CalibrationResult,
    FigCResult,
    FigVResult,
    fractional_information_gain_confidence,
    fractional_information_gain_validation,
)

__all__ = [
    "CalibrationResult",
    "FigCResult",
    "FigVResult",
    "fractional_information_gain_confidence",
    "fractional_information_gain_validation",
]
