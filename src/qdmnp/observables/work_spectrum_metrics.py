"""Pure-array convergence of the QD contrast on a metal background."""

from __future__ import annotations

import numpy as np


def delta_window_diagnostics(
    sigma: np.ndarray,
    sigma_half: np.ndarray,
    bare_sigma: np.ndarray,
    support: np.ndarray,
    *,
    relative_tolerance: float,
    absolute_tolerance_cm2: float = 0.0,
) -> dict[str, np.ndarray]:
    """Require |delta_full-delta_half| <= atol + rtol max|delta_full|.

    The bare MNP is evaluated analytically in frequency space, so the SAME
    converged reference is subtracted from both time windows.  An identically
    zero contrast is accepted only if the window error meets the absolute
    tolerance.  Unsupported Fourier samples never enter the certificate.
    """
    if not np.isfinite(relative_tolerance) or relative_tolerance <= 0.0:
        raise ValueError("relative_tolerance must be finite and positive.")
    if not np.isfinite(absolute_tolerance_cm2) or absolute_tolerance_cm2 < 0.0:
        raise ValueError("absolute_tolerance_cm2 must be finite and nonnegative.")
    full, half, bare = np.broadcast_arrays(
        np.asarray(sigma, dtype=float), np.asarray(sigma_half, dtype=float),
        np.asarray(bare_sigma, dtype=float),
    )
    if full.ndim < 1 or full.shape[-1] == 0:
        raise ValueError("Spectra need a nonempty final energy axis.")
    mask = np.broadcast_to(np.asarray(support, dtype=bool), full.shape)
    delta, delta_half = full - bare, half - bare
    finite = np.isfinite(delta) & np.isfinite(delta_half)
    valid = np.any(mask, axis=-1) & np.all(~mask | finite, axis=-1)
    error = np.max(np.where(mask & finite, np.abs(delta - delta_half), 0.0), axis=-1)
    scale = np.max(np.where(mask & finite, np.abs(delta), 0.0), axis=-1)
    error = np.where(valid, error, np.inf)
    normalized = np.divide(error, scale, out=np.full_like(error, np.inf), where=scale > 0)
    normalized = np.where(valid & (error == 0), 0.0, normalized)
    return {
        "delta_sigma_half_window_cm2": delta_half,
        "delta_window_max_absolute_change_cm2": error,
        "delta_window_scale_cm2": scale,
        "delta_window_max_normalized_change": normalized,
        "delta_window_converged": valid & (
            error <= absolute_tolerance_cm2 + relative_tolerance * scale
        ),
    }
