"""
dynamic_confidence.py
=====================

This module provides helper functions to compute a dynamic confidence
adjustment for the reasoning brain.  It is a stub for future
implementations that adjust confidence based on recent successes and
failures.  The current version simply returns the average of a
sequence of metrics divided by ten.

Rationale: Having a rolling statistical measure of recent successes and
failures allows Maven to adapt its confidence scores over time rather
than relying on fixed thresholds.  Future versions may incorporate
weighted averages, variance measurements, or exponential smoothing.
"""

from __future__ import annotations
from typing import Sequence


def compute_dynamic_confidence(successes: Sequence[float]) -> float:
    """Compute a dynamic confidence adjustment based on recent success metrics.

    The provided sequence should contain values between 0 and 1 where
    1.0 denotes a perfect success and 0.0 a complete failure.  The
    returned value can be added to or subtracted from a base confidence
    score.  For now, this function returns the average divided by
    10 to produce a modest adjustment.

    Args:
        successes: A sequence of recent success ratios.

    Returns:
        A floating‑point adjustment factor.  If the input is empty or
        invalid, 0.0 is returned.
    """
    try:
        n = len(successes)
        if n:
            return float(sum(successes)) / n / 10.0
    except Exception:
        pass
    return 0.0