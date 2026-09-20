"""Solver-free first-Rabi-lobe threshold extraction for article artifacts.

A finite left-censored value is an upper bound, not a resolved threshold.
Only ``resolved`` and ``resolved_refined`` statuses may enter exact ratios
or threshold-intensity estimates.  The fluence grid must separately pass
its resolution audit; sampled crossings cannot exclude unsampled lobes.
"""

from __future__ import annotations

import numpy as np


def threshold_from_curve(
    fluence_j_cm2: np.ndarray,
    population: np.ndarray,
    target: float,
) -> tuple[float, str, tuple[int, int] | None]:
    """Interpolate in sqrt(fluence), stopping at the first observed descent.

    Flat points do not start a new lobe.  A scan already descending at its
    left edge cannot establish the first rising branch and is censored.
    """

    fluence = np.asarray(fluence_j_cm2, dtype=float)
    values = np.asarray(population, dtype=float)
    if fluence.ndim != 1 or values.shape != fluence.shape or fluence.size < 3:
        raise ValueError("Threshold extraction needs matching 1-D arrays with >=3 points.")
    if np.any(~np.isfinite(fluence)) or np.any(fluence <= 0.0) or np.any(np.diff(fluence) <= 0.0):
        raise ValueError("Fluence must be finite, positive and strictly increasing.")
    if not np.isfinite(target) or not 0.0 < target < 1.0:
        raise ValueError("target must lie in (0, 1).")
    if np.any(~np.isfinite(values)):
        return np.nan, "nonfinite", None
    if values[0] >= target:
        return float(fluence[0]), "left_censored", None

    differences = np.diff(values)
    descending = np.flatnonzero(differences < 0.0)
    last = int(descending[0]) if descending.size else values.size - 1
    if descending.size and not np.any(differences[:last] > 0.0):
        return np.nan, "left_lobe_censored", None
    for upper in range(1, last + 1):
        if values[upper - 1] < target <= values[upper]:
            x0, x1 = np.sqrt(fluence[[upper - 1, upper]])
            y0, y1 = values[[upper - 1, upper]]
            x = x0 + (target - y0) * (x1 - x0) / (y1 - y0)
            return float(x * x), "resolved", (upper - 1, upper)
    if descending.size:
        return np.nan, "not_reached_first_lobe", None
    return np.nan, "right_censored", None


def resolved_threshold_mask(status: np.ndarray | str) -> np.ndarray:
    """Reject bounds, missing values and unknown status labels."""

    return np.isin(np.asarray(status).astype(str), ("resolved", "resolved_refined"))


def resolved_threshold_ratio(
    numerator: np.ndarray | float,
    denominator: np.ndarray | float,
    numerator_status: np.ndarray | str,
    denominator_status: np.ndarray | str,
) -> np.ndarray:
    """Return ratios only when both thresholds are finite and resolved."""

    top, bottom, top_status, bottom_status = np.broadcast_arrays(
        np.asarray(numerator, dtype=float),
        np.asarray(denominator, dtype=float),
        np.asarray(numerator_status).astype(str),
        np.asarray(denominator_status).astype(str),
    )
    accepted = (
        resolved_threshold_mask(top_status)
        & resolved_threshold_mask(bottom_status)
        & np.isfinite(top)
        & np.isfinite(bottom)
        & (top > 0.0)
        & (bottom > 0.0)
    )
    return np.divide(top, bottom, out=np.full(top.shape, np.nan), where=accepted)
