"""Shared, baseline-aware spectral features and sampling diagnostics.

The reported width is an operational half-prominence width in a fixed window.
It is not a fit to a Lorentzian or an estimate of a microscopic decay rate.
The reported peak energy is the vertex of the parabola through the sampled
maximum and its two neighbours, so shifts smaller than one grid step are not
quantized to grid nodes; the reported height remains the sampled maximum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks, peak_prominences, peak_widths


@dataclass(frozen=True)
class FeatureResult:
    energy_eV: float
    height: float
    prominence: float
    width_eV: float
    left_eV: float
    right_eV: float
    status: str
    competing_peak_count: int


def _vertex_energy(energy, values, index):
    """Parabolic vertex through an interior sampled maximum (non-uniform grid safe)."""
    if index <= 0 or index >= values.size - 1:
        return float(energy[index])
    x0, x1, x2 = energy[index - 1:index + 2]
    y0, y1, y2 = values[index - 1:index + 2]
    # Divided differences of the interpolating quadratic y = y1 + b (x-x1) + a (x-x1)^2.
    left, right = (y1 - y0) / (x1 - x0), (y2 - y1) / (x2 - x1)
    curvature = (right - left) / (x2 - x0)
    if not np.isfinite(curvature) or curvature >= 0.0:
        return float(x1)
    slope_at_x1 = left + curvature * (x1 - x0)
    vertex = x1 - slope_at_x1 / (2.0 * curvature)
    return float(np.clip(vertex, x0, x2))


def _components(energy, values, center, half_window):
    selected = np.flatnonzero(np.abs(energy - center) <= half_window)
    if selected.size < 5:
        return None
    local_energy, local = energy[selected], values[selected]
    peaks, _ = find_peaks(local)
    if not peaks.size:
        return local_energy, local, peaks, np.empty(0), np.empty(0), np.empty(0), np.empty(0, bool)
    prominence_data = peak_prominences(local, peaks)
    prominence, left_base, right_base = prominence_data
    _, _, left_ip, right_ip = peak_widths(
        local, peaks, rel_height=0.5, prominence_data=prominence_data
    )
    left = np.interp(left_ip, np.arange(local.size), local_energy)
    right = np.interp(right_ip, np.arange(local.size), local_energy)
    # scipy's prominence uses the HIGHER base.  If that base is a clipped
    # window edge, even a almost entirely clipped line gets a tiny "width".
    # Require the edge to be below half the peak excursion from the opposite
    # (lower) base before accepting a boundary-supported prominence.
    lower_base = np.minimum(local[left_base], local[right_base])
    lower_half_level = lower_base + 0.5 * (local[peaks] - lower_base)
    clipped = ((left_base == 0) & (local[0] >= lower_half_level)) | (
        (right_base == local.size - 1) & (local[-1] >= lower_half_level)
    )
    clipped |= (left_ip <= 0.0) | (right_ip >= local.size - 1)
    return local_energy, local, peaks, prominence, left, right, clipped


def extract_feature(
    energy_eV: np.ndarray,
    spectrum: np.ndarray,
    *,
    center_eV: float,
    half_window_eV: float,
    competing_prominence_fraction: float = 0.5,
) -> FeatureResult:
    """Select the nearest credible peak; reject clipped or ambiguous widths."""
    energy, values = np.asarray(energy_eV, float), np.asarray(spectrum, float)
    if energy.ndim != 1 or values.shape != energy.shape or energy.size < 5:
        raise ValueError("Feature extraction needs matching 1-D arrays with at least five points.")
    if np.any(~np.isfinite(energy)) or np.any(np.diff(energy) <= 0):
        raise ValueError("energy_eV must be finite and strictly increasing.")
    if not np.isfinite(center_eV) or not np.isfinite(half_window_eV) or half_window_eV <= 0:
        raise ValueError("Feature center must be finite and half-window finite and positive.")
    if not 0 < competing_prominence_fraction <= 1:
        raise ValueError("competing_prominence_fraction must lie in (0, 1].")
    empty = lambda status: FeatureResult(*(np.nan,) * 6, status, 0)
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        return empty("invalid_spectrum")
    components = _components(energy, values, center_eV, half_window_eV)
    if components is None:
        return empty("window_too_small")
    local_energy, local, peaks, prominence, left, right, clipped = components
    if not peaks.size:
        return empty("edge_or_no_peak")
    credible = np.flatnonzero(prominence >= 0.1 * np.max(prominence))
    distance = np.abs(local_energy[peaks[credible]] - center_eV)
    nearest = credible[np.isclose(distance, np.min(distance), rtol=0, atol=1e-15)]
    index = int(nearest[np.argmax(local[peaks[nearest]])])
    competing = int(np.count_nonzero(
        np.delete(prominence, index) >= competing_prominence_fraction * prominence[index]
    ))
    status = "edge_truncated" if clipped[index] else "split_or_ambiguous" if competing else "ok"
    return FeatureResult(
        _vertex_energy(local_energy, local, int(peaks[index])), float(local[peaks[index]]), float(prominence[index]),
        float(right[index] - left[index]) if status == "ok" else np.nan,
        float(left[index]) if status == "ok" else np.nan,
        float(right[index]) if status == "ok" else np.nan,
        status, competing,
    )


def sampling_diagnostics(
    energy_eV: np.ndarray,
    spectrum: np.ndarray,
    *,
    center_eV: float,
    half_window_eV: float,
    max_step_over_width: float = 0.05,
    max_coarsening_change: float = 0.05,
    max_window_change: float = 0.05,
) -> dict[str, float | bool]:
    """Check every credible component and stability on two interleaved subgrids.

    This is a sampling/coarsening certificate, not a proof excluding arbitrary
    unobserved subgrid poles.  Raw spectra remain available for grid refinement.
    Split curves can pass sampling without being assigned a single FWHM.
    """
    energy, values = np.asarray(energy_eV, float), np.asarray(spectrum, float)
    feature = extract_feature(energy, values, center_eV=center_eV, half_window_eV=half_window_eV)
    for name, tolerance in (("max_step_over_width", max_step_over_width),
                            ("max_coarsening_change", max_coarsening_change),
                            ("max_window_change", max_window_change)):
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    result = {"minimum_component_width_eV": np.nan, "step_over_component_width": np.inf,
              "coarsening_relative_change": np.inf, "window_relative_change": np.inf,
              "window_accepted": False, "accepted": False}
    if feature.status not in ("ok", "split_or_ambiguous"):
        return result
    local_energy, local, peaks, prominence, left, right, clipped = _components(
        energy, values, center_eV, half_window_eV
    )
    credible = prominence >= 0.1 * np.max(prominence)
    minimum_width = float(np.min((right - left)[credible]))
    step_ratio = float(np.max(np.diff(local_energy)) / minimum_width)
    # A symmetric, severely clipped line can have thousands of samples and
    # an apparently precise half-prominence width.  Vary the ACTUALLY
    # available window independently of grid spacing to detect that case.
    available_radius = min(center_eV - local_energy[0], local_energy[-1] - center_eV)
    inner = _components(energy, values, center_eV, 0.75 * available_radius)
    window_change = np.inf
    if inner is not None and inner[2].size:
        inner_energy, _, inner_peaks, inner_prominence, inner_left, inner_right, inner_clipped = inner
        inner_credible = inner_prominence >= 0.1 * np.max(inner_prominence)
        if np.count_nonzero(inner_credible) == np.count_nonzero(credible) and not np.any(inner_clipped[inner_credible]):
            window_change = max(
                float(np.max(np.abs((inner_right - inner_left)[inner_credible] / (right - left)[credible] - 1))),
                float(np.max(np.abs(inner_prominence[inner_credible] / prominence[credible] - 1))),
                float(np.max(np.abs(inner_energy[inner_peaks[inner_credible]] - local_energy[peaks[credible]]))) / minimum_width,
            )
    window_accepted = bool(not np.any(clipped[credible]) and window_change <= max_window_change)
    change = 0.0
    for offset in (0, 1):
        indices = np.unique(np.r_[0, np.arange(offset, energy.size, 2), energy.size - 1])
        if indices.size < 5:
            change = np.inf
            break
        coarse = extract_feature(energy[indices], values[indices], center_eV=center_eV,
                                 half_window_eV=half_window_eV)
        if coarse.status != feature.status:
            change = np.inf
            break
        mask = np.abs(energy[indices] - center_eV) <= half_window_eV
        change = max(change, abs(float(np.max(values[indices][mask])) / np.max(local) - 1))
        if feature.status == "ok":
            change = max(change, abs(coarse.energy_eV - feature.energy_eV) / minimum_width,
                         abs(coarse.width_eV / feature.width_eV - 1),
                         abs(coarse.height / feature.height - 1))
    result.update(minimum_component_width_eV=minimum_width, step_over_component_width=step_ratio,
                  coarsening_relative_change=float(change),
                  window_relative_change=float(window_change), window_accepted=window_accepted,
                  accepted=bool(not np.any(clipped[credible]) and step_ratio <= max_step_over_width
                                and change <= max_coarsening_change and window_accepted))
    return result
