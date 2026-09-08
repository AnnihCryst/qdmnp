"""Calculate direct/N=1/N-mode spheroid polarizability comparison.

The program performs the material fits once and writes a self-contained NPZ
artifact.  It deliberately creates no figure; use the companion
``qd_mnp_plot_material_dispersion_comparison.py`` program for styling.
"""

from __future__ import annotations

import argparse
import platform
from pathlib import Path
import sys
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import scipy

from article_observables.qd_mnp_material_modes_artifact import (
    atomic_write_npz,
    canonical_sha256,
    git_provenance,
    source_hashes,
)
from qd_mnp_rational_fit import (
    AU_ENERGY_EV,
    AU_LENGTH_M,
    MATERIAL_HIGH_FREQUENCY_EPSILON,
    MATERIAL_INTERPOLATION,
    HybridQDPlasmonModel,
    au_to_eV,
    make_params_with_overrides,
    params_to_physical_dict,
)


SCHEMA_NAME = "qd_mnp.material_dispersion_comparison"
SCHEMA_VERSION = 1
ORIENTATIONS = ("long", "trans")
BRANCHES = ("direct", "one", "multi")
FIT_BRANCHES = ("one", "multi")


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and non-negative")
    return parsed


def _validate_args(args: argparse.Namespace) -> None:
    if args.one_modes != 1:
        raise ValueError("--one-modes must equal 1 for an actual one-oscillator branch.")
    if args.multi_modes < 2:
        raise ValueError("--multi-modes must be at least 2.")
    if args.energy_points < 5:
        raise ValueError("--energy-points must be at least 5.")
    if args.energy_min_ev >= args.energy_max_ev:
        raise ValueError("Energy limits must satisfy energy_min_ev < energy_max_ev.")
    if args.fit_min_ev >= args.fit_max_ev:
        raise ValueError("Fit limits must satisfy fit_min_ev < fit_max_ev.")
    if (
        args.energy_min_ev < args.fit_min_ev
        or args.energy_max_ev > args.fit_max_ev
    ):
        raise ValueError("The plotted energy interval must lie inside the fit interval.")
    if args.c_nm <= args.a_nm:
        raise ValueError("This prolate-spheroid workflow requires c_nm > a_nm.")
    if (args.weight_center_ev is None) != (args.weight_sigma_ev is None):
        raise ValueError(
            "--weight-center-ev and --weight-sigma-ev must be supplied together."
        )
    if args.weight_sigma_ev is not None and args.weight_sigma_ev <= 0.0:
        raise ValueError("--weight-sigma-ev must be positive.")
    if args.alpha_objective_weight == args.inverse_objective_weight == 0.0:
        raise ValueError("At least one fit-objective weight must be positive.")


def _robust_residual_metrics(
    candidate: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Return residual, peak-normalized residual/error, NRMS and max error.

    Both normalizations use global reference scales and therefore remain
    finite when a point of the complex reference happens to be close to zero.
    """

    trial = np.asarray(candidate, dtype=complex)
    target = np.asarray(reference, dtype=complex)
    if trial.shape != target.shape or trial.ndim != 1:
        raise ValueError("candidate and reference must be matching one-dimensional arrays.")
    if np.any(~np.isfinite(trial)) or np.any(~np.isfinite(target)):
        raise ValueError("Residual inputs must be finite.")

    residual = trial - target
    target_peak = float(np.max(np.abs(target)))
    target_rms = float(np.sqrt(np.mean(np.abs(target) ** 2)))
    numerical_floor = max(
        np.finfo(float).tiny,
        np.finfo(float).eps * max(target_peak, 1.0),
    )
    peak_scale = max(target_peak, numerical_floor)
    rms_scale = max(target_rms, numerical_floor)
    normalized_residual = residual / peak_scale
    normalized_abs_error = np.abs(normalized_residual)
    nrms = float(np.sqrt(np.mean(np.abs(residual) ** 2)) / rms_scale)
    max_error = float(np.max(np.abs(residual)) / peak_scale)
    return residual, normalized_residual, normalized_abs_error, nrms, max_error


def _interpolate_crossing(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    level: float,
) -> float:
    if y1 == y0:
        return float(0.5 * (x0 + x1))
    fraction = float(np.clip((level - y0) / (y1 - y0), 0.0, 1.0))
    return float(x0 + fraction * (x1 - x0))


def _lspr_peak_metrics(
    energy_eV: np.ndarray,
    imag_alpha: np.ndarray,
) -> dict[str, float | str]:
    """Find the window peak and its operational baseline-corrected FWHM."""

    energy = np.asarray(energy_eV, dtype=float)
    signal = np.asarray(imag_alpha, dtype=float)
    if energy.ndim != 1 or signal.shape != energy.shape or energy.size < 3:
        raise ValueError("LSPR metrics require matching one-dimensional arrays of size >= 3.")
    if np.any(~np.isfinite(energy)) or np.any(~np.isfinite(signal)):
        raise ValueError("LSPR inputs must be finite.")
    if np.any(np.diff(energy) <= 0.0):
        raise ValueError("LSPR energies must be strictly increasing.")

    peak_index = int(np.argmax(signal))
    peak_energy = float(energy[peak_index])
    peak_value = float(signal[peak_index])
    baseline = float(np.min(signal))
    amplitude = peak_value - baseline
    scale = max(abs(peak_value), abs(baseline), 1.0)
    if peak_index in {0, energy.size - 1}:
        return {
            "peak_energy_eV": peak_energy,
            "peak_imag_alpha": peak_value,
            "baseline_imag_alpha": baseline,
            "half_level_imag_alpha": float(baseline + 0.5 * amplitude),
            "fwhm_eV": float("nan"),
            "status": "peak_at_window_boundary",
        }
    if amplitude <= 64.0 * np.finfo(float).eps * scale:
        return {
            "peak_energy_eV": peak_energy,
            "peak_imag_alpha": peak_value,
            "baseline_imag_alpha": baseline,
            "half_level_imag_alpha": baseline,
            "fwhm_eV": float("nan"),
            "status": "no_resolved_peak",
        }

    half_level = float(baseline + 0.5 * amplitude)
    left_candidates = np.flatnonzero(signal[:peak_index] <= half_level)
    right_candidates = np.flatnonzero(signal[peak_index + 1 :] <= half_level)
    if left_candidates.size == 0 and right_candidates.size == 0:
        status = "both_half_max_crossings_missing"
        fwhm = float("nan")
    elif left_candidates.size == 0:
        status = "left_half_max_crossing_missing"
        fwhm = float("nan")
    elif right_candidates.size == 0:
        status = "right_half_max_crossing_missing"
        fwhm = float("nan")
    else:
        left_lower = int(left_candidates[-1])
        left_upper = left_lower + 1
        right_upper = peak_index + 1 + int(right_candidates[0])
        right_lower = right_upper - 1
        left_energy = _interpolate_crossing(
            float(energy[left_lower]),
            float(signal[left_lower]),
            float(energy[left_upper]),
            float(signal[left_upper]),
            half_level,
        )
        right_energy = _interpolate_crossing(
            float(energy[right_lower]),
            float(signal[right_lower]),
            float(energy[right_upper]),
            float(signal[right_upper]),
            half_level,
        )
        fwhm = float(right_energy - left_energy)
        status = "ok" if fwhm > 0.0 else "nonpositive_fwhm"
        if fwhm <= 0.0:
            fwhm = float("nan")

    return {
        "peak_energy_eV": peak_energy,
        "peak_imag_alpha": peak_value,
        "baseline_imag_alpha": baseline,
        "half_level_imag_alpha": half_level,
        "fwhm_eV": fwhm,
        "status": status,
    }


def _fit_diagnostics(model: HybridQDPlasmonModel) -> dict[str, Any]:
    fit = model.fit
    stability = model.linear_stability
    return {
        "n_modes": int(model.n_modes),
        "alpha_inf": float(fit.alpha_inf),
        "normalized_rms_alpha_internal": float(fit.normalized_rms_alpha),
        "normalized_rms_inverse_alpha_internal": float(
            fit.normalized_rms_inv_alpha
        ),
        "max_normalized_alpha_error_internal": float(
            fit.max_normalized_alpha_error
        ),
        "rms_alpha_internal": float(fit.rms_alpha),
        "rms_inverse_alpha_internal": float(fit.rms_inv_alpha),
        "cost": float(fit.cost),
        "min_imag_alpha_fit_window": float(fit.min_imag_alpha_fit_window),
        "passivity_grid_points": int(fit.passivity_grid_points),
        "passive_on_fit_window": bool(fit.passive_on_fit_window),
        "nonnegative_imaginary_part_all_positive_frequencies": bool(
            fit.nonnegative_imaginary_part_all_positive_frequencies
        ),
        "linear_stable": bool(stability.stable),
        "spectral_abscissa_au": float(stability.spectral_abscissa_au),
        "stability_tolerance_au": float(stability.tolerance_au),
    }


def _assert_fit_physics(
    model: HybridQDPlasmonModel,
    *,
    branch: str,
    max_normalized_rms: float,
    max_pointwise_relative_error: float,
) -> dict[str, Any]:
    diagnostics = _fit_diagnostics(model)
    if not diagnostics["passive_on_fit_window"]:
        raise RuntimeError(f"The {branch} Lorentz fit is not passive on the fit window.")
    if not diagnostics["nonnegative_imaginary_part_all_positive_frequencies"]:
        raise RuntimeError(
            f"The {branch} Lorentz fit does not have non-negative Im(alpha) "
            "for all positive frequencies."
        )
    if not diagnostics["linear_stable"]:
        raise RuntimeError(f"The {branch} Lorentz realization is linearly unstable.")

    gate_pass = bool(
        diagnostics["normalized_rms_alpha_internal"] <= max_normalized_rms
        and diagnostics["normalized_rms_inverse_alpha_internal"]
        <= max_normalized_rms
        and diagnostics["max_normalized_alpha_error_internal"]
        <= max_pointwise_relative_error
    )
    diagnostics["accuracy_gate_pass"] = gate_pass
    if branch == "multi" and not gate_pass:
        raise RuntimeError(
            "The multi-oscillator fit failed the recorded accuracy gates: "
            f"NRMS(alpha)={diagnostics['normalized_rms_alpha_internal']:.6g}, "
            "NRMS(1/alpha)="
            f"{diagnostics['normalized_rms_inverse_alpha_internal']:.6g}, "
            "max normalized alpha error="
            f"{diagnostics['max_normalized_alpha_error_internal']:.6g}."
        )
    return diagnostics


def _make_model(
    *,
    orientation: str,
    n_modes: int,
    args: argparse.Namespace,
    enforce_accuracy_in_model: bool,
) -> HybridQDPlasmonModel:
    params = make_params_with_overrides(
        c_nm=args.c_nm,
        a_nm=args.a_nm,
        eps_m=args.eps_m,
        orientation=orientation,
    )
    return HybridQDPlasmonModel(
        params,
        orientation=orientation,
        n_modes=n_modes,
        fit_window_eV=(args.fit_min_ev, args.fit_max_ev),
        weight_center_eV=args.weight_center_ev,
        weight_sigma_eV=args.weight_sigma_ev,
        alpha_objective_weight=args.alpha_objective_weight,
        inv_alpha_objective_weight=args.inverse_objective_weight,
        max_fit_normalized_rms=(
            args.max_normalized_rms if enforce_accuracy_in_model else None
        ),
        max_fit_pointwise_relative_error=(
            args.max_pointwise_relative_error
            if enforce_accuracy_in_model
            else None
        ),
        radiative_consistency_policy="ignore",
        seed=args.seed,
        verbose=False,
    )


def calculate_material_dispersion_comparison(
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Perform all fits and return an NPZ-ready payload plus metadata."""

    _validate_args(args)
    energies = np.linspace(
        args.energy_min_ev,
        args.energy_max_ev,
        args.energy_points,
        dtype=float,
    )
    n_orientation = len(ORIENTATIONS)
    n_branch = len(BRANCHES)
    n_fit_branch = len(FIT_BRANCHES)
    n_energy = energies.size

    alpha_dimensionless = np.empty(
        (n_orientation, n_branch, n_energy), dtype=complex
    )
    alpha_au3 = np.empty_like(alpha_dimensionless)
    inverse_alpha_au_minus3 = np.empty_like(alpha_dimensionless)
    direct_target_consistency = np.empty(n_orientation, dtype=float)
    depolarization_factor = np.empty(n_orientation, dtype=float)
    polarizability_scale_au3 = np.empty(n_orientation, dtype=float)
    physical_parameters: dict[str, dict[str, Any]] = {}
    fit_metadata: dict[str, dict[str, Any]] = {}
    models: dict[tuple[str, str], HybridQDPlasmonModel] = {}

    mode_counts = {"one": int(args.one_modes), "multi": int(args.multi_modes)}
    for orientation_index, orientation in enumerate(ORIENTATIONS):
        one_model = _make_model(
            orientation=orientation,
            n_modes=args.one_modes,
            args=args,
            enforce_accuracy_in_model=False,
        )
        multi_model = _make_model(
            orientation=orientation,
            n_modes=args.multi_modes,
            args=args,
            enforce_accuracy_in_model=True,
        )
        models[(orientation, "one")] = one_model
        models[(orientation, "multi")] = multi_model

        one_direct = np.asarray(one_model.alpha_from_material(energies), dtype=complex)
        multi_direct = np.asarray(
            multi_model.alpha_from_material(energies), dtype=complex
        )
        direct_target_consistency[orientation_index] = float(
            np.max(np.abs(one_direct - multi_direct))
        )
        target_scale = max(float(np.max(np.abs(multi_direct))), 1.0)
        if direct_target_consistency[orientation_index] > 1.0e-12 * target_scale:
            raise RuntimeError(
                "The one- and multi-oscillator models did not expose the same "
                f"direct material target for orientation={orientation!r}."
            )

        alpha_dimensionless[orientation_index, 0] = multi_direct
        alpha_dimensionless[orientation_index, 1] = np.asarray(
            one_model.alpha_from_fit(energies), dtype=complex
        )
        alpha_dimensionless[orientation_index, 2] = np.asarray(
            multi_model.alpha_from_fit(energies), dtype=complex
        )
        depolarization_factor[orientation_index] = float(multi_model.L)
        polarizability_scale_au3[orientation_index] = float(multi_model.C)
        alpha_au3[orientation_index] = (
            polarizability_scale_au3[orientation_index]
            * alpha_dimensionless[orientation_index]
        )
        inverse_alpha_au_minus3[orientation_index] = (
            1.0 / alpha_au3[orientation_index]
        )
        physical_parameters[orientation] = params_to_physical_dict(
            multi_model.params, orientation
        )

        fit_metadata[orientation] = {}
        for branch, model in (("one", one_model), ("multi", multi_model)):
            fit_metadata[orientation][branch] = _assert_fit_physics(
                model,
                branch=branch,
                max_normalized_rms=args.max_normalized_rms,
                max_pointwise_relative_error=args.max_pointwise_relative_error,
            )

    if np.any(~np.isfinite(alpha_au3)) or np.any(
        ~np.isfinite(inverse_alpha_au_minus3)
    ):
        raise RuntimeError("The calculated alpha or inverse-alpha spectrum is non-finite.")

    alpha_residual = np.empty_like(alpha_au3)
    alpha_normalized_residual = np.empty_like(alpha_au3)
    alpha_normalized_abs_error = np.empty(alpha_au3.shape, dtype=float)
    inverse_residual = np.empty_like(inverse_alpha_au_minus3)
    inverse_normalized_residual = np.empty_like(inverse_alpha_au_minus3)
    inverse_normalized_abs_error = np.empty(alpha_au3.shape, dtype=float)
    nrms_alpha = np.empty((n_orientation, n_branch), dtype=float)
    nrms_inverse_alpha = np.empty_like(nrms_alpha)
    max_normalized_alpha_error = np.empty_like(nrms_alpha)
    max_normalized_inverse_alpha_error = np.empty_like(nrms_alpha)

    lspr_peak_energy_eV = np.empty((n_orientation, n_branch), dtype=float)
    lspr_peak_imag_alpha_au3 = np.empty_like(lspr_peak_energy_eV)
    lspr_baseline_imag_alpha_au3 = np.empty_like(lspr_peak_energy_eV)
    lspr_half_level_imag_alpha_au3 = np.empty_like(lspr_peak_energy_eV)
    lspr_fwhm_eV = np.empty_like(lspr_peak_energy_eV)
    lspr_status = np.empty((n_orientation, n_branch), dtype="<U40")

    for orientation_index in range(n_orientation):
        alpha_reference = alpha_au3[orientation_index, 0]
        inverse_reference = inverse_alpha_au_minus3[orientation_index, 0]
        for branch_index in range(n_branch):
            (
                alpha_residual[orientation_index, branch_index],
                alpha_normalized_residual[orientation_index, branch_index],
                alpha_normalized_abs_error[orientation_index, branch_index],
                nrms_alpha[orientation_index, branch_index],
                max_normalized_alpha_error[orientation_index, branch_index],
            ) = _robust_residual_metrics(
                alpha_au3[orientation_index, branch_index], alpha_reference
            )
            (
                inverse_residual[orientation_index, branch_index],
                inverse_normalized_residual[orientation_index, branch_index],
                inverse_normalized_abs_error[orientation_index, branch_index],
                nrms_inverse_alpha[orientation_index, branch_index],
                max_normalized_inverse_alpha_error[orientation_index, branch_index],
            ) = _robust_residual_metrics(
                inverse_alpha_au_minus3[orientation_index, branch_index],
                inverse_reference,
            )
            peak = _lspr_peak_metrics(
                energies, alpha_au3[orientation_index, branch_index].imag
            )
            lspr_peak_energy_eV[orientation_index, branch_index] = float(
                peak["peak_energy_eV"]
            )
            lspr_peak_imag_alpha_au3[orientation_index, branch_index] = float(
                peak["peak_imag_alpha"]
            )
            lspr_baseline_imag_alpha_au3[orientation_index, branch_index] = float(
                peak["baseline_imag_alpha"]
            )
            lspr_half_level_imag_alpha_au3[orientation_index, branch_index] = float(
                peak["half_level_imag_alpha"]
            )
            lspr_fwhm_eV[orientation_index, branch_index] = float(peak["fwhm_eV"])
            lspr_status[orientation_index, branch_index] = str(peak["status"])

    max_modes = args.multi_modes
    fit_alpha_inf = np.empty((n_orientation, n_fit_branch), dtype=float)
    fit_mode_count = np.empty((n_orientation, n_fit_branch), dtype=int)
    fit_coefficient_valid = np.zeros(
        (n_orientation, n_fit_branch, max_modes), dtype=bool
    )
    fit_strengths_au2 = np.full(
        (n_orientation, n_fit_branch, max_modes), np.nan, dtype=float
    )
    fit_omega_modes_au = np.full_like(fit_strengths_au2, np.nan)
    fit_omega_modes_eV = np.full_like(fit_strengths_au2, np.nan)
    fit_gamma_modes_au = np.full_like(fit_strengths_au2, np.nan)
    fit_gamma_modes_eV = np.full_like(fit_strengths_au2, np.nan)
    fit_passive_on_window = np.empty((n_orientation, n_fit_branch), dtype=bool)
    fit_nonnegative_imag_all_positive = np.empty_like(fit_passive_on_window)
    fit_linear_stable = np.empty_like(fit_passive_on_window)
    fit_accuracy_gate_pass = np.empty_like(fit_passive_on_window)
    fit_spectral_abscissa_au = np.empty(
        (n_orientation, n_fit_branch), dtype=float
    )
    fit_stability_tolerance_au = np.empty_like(fit_spectral_abscissa_au)
    fit_nrms_alpha_internal = np.empty_like(fit_spectral_abscissa_au)
    fit_nrms_inverse_alpha_internal = np.empty_like(fit_spectral_abscissa_au)
    fit_max_normalized_alpha_error_internal = np.empty_like(
        fit_spectral_abscissa_au
    )
    max_poles = 2 * max_modes + 3
    fit_linear_poles_au = np.full(
        (n_orientation, n_fit_branch, max_poles), np.nan + 1j * np.nan
    )
    fit_linear_pole_valid = np.zeros(
        (n_orientation, n_fit_branch, max_poles), dtype=bool
    )

    for orientation_index, orientation in enumerate(ORIENTATIONS):
        for fit_index, branch in enumerate(FIT_BRANCHES):
            model = models[(orientation, branch)]
            fit = model.fit
            diagnostics = fit_metadata[orientation][branch]
            count = int(model.n_modes)
            pole_count = int(model.linear_stability.poles_au.size)
            fit_alpha_inf[orientation_index, fit_index] = float(fit.alpha_inf)
            fit_mode_count[orientation_index, fit_index] = count
            fit_coefficient_valid[orientation_index, fit_index, :count] = True
            fit_strengths_au2[orientation_index, fit_index, :count] = (
                fit.strengths_au2
            )
            fit_omega_modes_au[orientation_index, fit_index, :count] = (
                fit.omega_modes_au
            )
            fit_omega_modes_eV[orientation_index, fit_index, :count] = au_to_eV(
                fit.omega_modes_au
            )
            fit_gamma_modes_au[orientation_index, fit_index, :count] = (
                fit.gamma_modes_au
            )
            fit_gamma_modes_eV[orientation_index, fit_index, :count] = au_to_eV(
                fit.gamma_modes_au
            )
            fit_passive_on_window[orientation_index, fit_index] = diagnostics[
                "passive_on_fit_window"
            ]
            fit_nonnegative_imag_all_positive[
                orientation_index, fit_index
            ] = diagnostics[
                "nonnegative_imaginary_part_all_positive_frequencies"
            ]
            fit_linear_stable[orientation_index, fit_index] = diagnostics[
                "linear_stable"
            ]
            fit_accuracy_gate_pass[orientation_index, fit_index] = diagnostics[
                "accuracy_gate_pass"
            ]
            fit_spectral_abscissa_au[orientation_index, fit_index] = diagnostics[
                "spectral_abscissa_au"
            ]
            fit_stability_tolerance_au[orientation_index, fit_index] = diagnostics[
                "stability_tolerance_au"
            ]
            fit_nrms_alpha_internal[orientation_index, fit_index] = diagnostics[
                "normalized_rms_alpha_internal"
            ]
            fit_nrms_inverse_alpha_internal[
                orientation_index, fit_index
            ] = diagnostics["normalized_rms_inverse_alpha_internal"]
            fit_max_normalized_alpha_error_internal[
                orientation_index, fit_index
            ] = diagnostics["max_normalized_alpha_error_internal"]
            fit_linear_poles_au[
                orientation_index, fit_index, :pole_count
            ] = model.linear_stability.poles_au
            fit_linear_pole_valid[
                orientation_index, fit_index, :pole_count
            ] = True

    material = models[("long", "multi")].params.material
    payload: dict[str, np.ndarray] = {
        "energy_eV": energies,
        "orientation_ids": np.asarray(ORIENTATIONS),
        "branch_ids": np.asarray(BRANCHES),
        "fit_branch_ids": np.asarray(FIT_BRANCHES),
        "mode_count_by_branch": np.asarray(
            [0, args.one_modes, args.multi_modes], dtype=int
        ),
        "depolarization_factor": depolarization_factor,
        "polarizability_scale_au3": polarizability_scale_au3,
        "direct_target_consistency_max_abs_dimensionless": (
            direct_target_consistency
        ),
        "alpha_dimensionless_complex": alpha_dimensionless,
        "alpha_dimensionless_real": alpha_dimensionless.real,
        "alpha_dimensionless_imag": alpha_dimensionless.imag,
        "alpha_complex_au3": alpha_au3,
        "alpha_real_au3": alpha_au3.real,
        "alpha_imag_au3": alpha_au3.imag,
        "inverse_alpha_complex_au_minus3": inverse_alpha_au_minus3,
        "inverse_alpha_real_au_minus3": inverse_alpha_au_minus3.real,
        "inverse_alpha_imag_au_minus3": inverse_alpha_au_minus3.imag,
        "alpha_residual_complex_au3": alpha_residual,
        "alpha_normalized_residual_complex": alpha_normalized_residual,
        "alpha_normalized_abs_error": alpha_normalized_abs_error,
        "inverse_alpha_residual_complex_au_minus3": inverse_residual,
        "inverse_alpha_normalized_residual_complex": inverse_normalized_residual,
        "inverse_alpha_normalized_abs_error": inverse_normalized_abs_error,
        "nrms_alpha": nrms_alpha,
        "nrms_inverse_alpha": nrms_inverse_alpha,
        "max_normalized_alpha_error": max_normalized_alpha_error,
        "max_normalized_inverse_alpha_error": (
            max_normalized_inverse_alpha_error
        ),
        "lspr_peak_energy_eV": lspr_peak_energy_eV,
        "lspr_peak_imag_alpha_au3": lspr_peak_imag_alpha_au3,
        "lspr_baseline_imag_alpha_au3": lspr_baseline_imag_alpha_au3,
        "lspr_half_level_imag_alpha_au3": lspr_half_level_imag_alpha_au3,
        "lspr_fwhm_eV": lspr_fwhm_eV,
        "lspr_status": lspr_status,
        "fit_alpha_inf_dimensionless": fit_alpha_inf,
        "fit_mode_count": fit_mode_count,
        "fit_coefficient_valid": fit_coefficient_valid,
        "fit_strengths_au2": fit_strengths_au2,
        "fit_omega_modes_au": fit_omega_modes_au,
        "fit_omega_modes_eV": fit_omega_modes_eV,
        "fit_gamma_modes_au": fit_gamma_modes_au,
        "fit_gamma_modes_eV": fit_gamma_modes_eV,
        "fit_passive_on_window": fit_passive_on_window,
        "fit_nonnegative_imag_all_positive": (
            fit_nonnegative_imag_all_positive
        ),
        "fit_linear_stable": fit_linear_stable,
        "fit_accuracy_gate_pass": fit_accuracy_gate_pass,
        "fit_spectral_abscissa_au": fit_spectral_abscissa_au,
        "fit_stability_tolerance_au": fit_stability_tolerance_au,
        "fit_nrms_alpha_internal": fit_nrms_alpha_internal,
        "fit_nrms_inverse_alpha_internal": fit_nrms_inverse_alpha_internal,
        "fit_max_normalized_alpha_error_internal": (
            fit_max_normalized_alpha_error_internal
        ),
        "fit_linear_poles_au": fit_linear_poles_au,
        "fit_linear_pole_valid": fit_linear_pole_valid,
        "material_energy_eV": np.asarray(material.energy_eV),
        "material_n": np.asarray(material.n),
        "material_k": np.asarray(material.k),
        "material_epsilon_complex": np.asarray(material.epsilon),
        "material_epsilon_real": np.asarray(material.epsilon.real),
        "material_epsilon_imag": np.asarray(material.epsilon.imag),
    }

    requested_inputs = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    helper_path = PROJECT_ROOT / "article_observables" / "qd_mnp_material_modes_artifact.py"
    model_path = PROJECT_ROOT / "qd_mnp_rational_fit.py"
    metadata: dict[str, Any] = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "workflow": "direct versus one- and multi-Lorentz material response",
        "calculation_script": Path(__file__).resolve().relative_to(
            PROJECT_ROOT
        ).as_posix(),
        "command_line": [str(value) for value in sys.argv],
        "requested_inputs": requested_inputs,
        "requested_inputs_sha256": canonical_sha256(requested_inputs),
        "resolved_inputs": {
            "energy_min_eV": float(energies[0]),
            "energy_max_eV": float(energies[-1]),
            "energy_points": int(energies.size),
            "fit_window_eV": [float(args.fit_min_ev), float(args.fit_max_ev)],
            "orientations": list(ORIENTATIONS),
            "branches": list(BRANCHES),
            "mode_counts": mode_counts,
            "c_nm": float(args.c_nm),
            "a_nm": float(args.a_nm),
            "aspect_ratio_c_over_a": float(args.c_nm / args.a_nm),
            "eps_m": float(args.eps_m),
            "weight_center_eV": args.weight_center_ev,
            "weight_sigma_eV": args.weight_sigma_ev,
            "seed": int(args.seed),
        },
        "quality_gates": {
            "one_branch_policy": (
                "accuracy is diagnostic only; passivity and linear stability "
                "are mandatory"
            ),
            "multi_branch_policy": (
                "accuracy, passivity, and linear stability are mandatory"
            ),
            "max_normalized_rms_alpha_and_inverse": float(
                args.max_normalized_rms
            ),
            "max_pointwise_normalized_alpha_error": float(
                args.max_pointwise_relative_error
            ),
            "multi_passed_for_all_orientations": bool(
                np.all(fit_accuracy_gate_pass[:, 1])
                and np.all(fit_passive_on_window[:, 1])
                and np.all(fit_nonnegative_imag_all_positive[:, 1])
                and np.all(fit_linear_stable[:, 1])
            ),
        },
        "physical_parameters_by_orientation": physical_parameters,
        "fit_diagnostics_by_orientation": fit_metadata,
        "definitions": {
            "direct": (
                "HybridQDPlasmonModel.alpha_from_material evaluated with the "
                "project's tabulated-material interpolation"
            ),
            "one": "HybridQDPlasmonModel.alpha_from_fit with one Lorentz mode",
            "multi": (
                "HybridQDPlasmonModel.alpha_from_fit with the requested "
                "multi-mode passive Lorentz realization"
            ),
            "alpha_au3": (
                "C times the dimensionless ellipsoid response returned by the "
                "public alpha API, C=eps_m*a^2*c/3 in atomic units"
            ),
            "normalized_residual": (
                "(candidate-direct)/max(max(abs(direct)), numerical floor)"
            ),
            "nrms": (
                "rms(abs(candidate-direct))/max(rms(abs(direct)), numerical floor)"
            ),
            "lspr_peak": "window maximum of Im(alpha)",
            "lspr_fwhm": (
                "nearest-crossing FWHM of Im(alpha), using half the peak "
                "height above the minimum value on the common energy window"
            ),
            "linear_stability": (
                "field-free coupled QD-MNP Jacobian exposed by "
                "HybridQDPlasmonModel"
            ),
        },
        "constants": {
            "atomic_unit_length_m": float(AU_LENGTH_M),
            "hartree_energy_eV": float(AU_ENERGY_EV),
            "material_interpolation": MATERIAL_INTERPOLATION,
            "material_high_frequency_epsilon": float(
                MATERIAL_HIGH_FREQUENCY_EPSILON
            ),
            "polarizability_formula": (
                "alpha=C*(epsilon-eps_m)/(eps_m+L*(epsilon-eps_m))"
            ),
            "time_harmonic_convention": (
                "the model convention in which passive fits have Im(alpha)>=0 "
                "at positive frequency"
            ),
        },
        "units": {
            "energy_eV": "eV",
            "alpha_complex_au3": "Bohr radius cubed",
            "inverse_alpha_complex_au_minus3": "inverse Bohr radius cubed",
            "fit_strengths_au2": "atomic angular-frequency squared",
            "fit_omega_modes_au": "atomic angular frequency",
            "fit_gamma_modes_au": "atomic angular frequency",
            "lspr_fwhm_eV": "eV",
        },
        "limitations": [
            "Local quasistatic ellipsoid material response; no retardation.",
            "No nonlocal, quantum-tunnelling, charge-transfer, or QCM correction.",
            "The operational FWHM is window- and baseline-definition dependent.",
            "A one-mode fit is retained as a diagnostic even when it fails the "
            "accuracy gates; it is never allowed to fail passivity or stability.",
        ],
        "provenance": {
            "git": git_provenance(PROJECT_ROOT),
            "source_sha256": source_hashes([__file__, helper_path, model_path]),
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
        },
    }
    return payload, metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/material_dispersion_comparison.npz"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--energy-min-ev", type=_positive_float, default=0.8)
    parser.add_argument("--energy-max-ev", type=_positive_float, default=3.0)
    parser.add_argument("--energy-points", type=int, default=801)
    parser.add_argument("--fit-min-ev", type=_positive_float, default=0.8)
    parser.add_argument("--fit-max-ev", type=_positive_float, default=3.0)
    parser.add_argument("--one-modes", type=int, default=1)
    parser.add_argument("--multi-modes", type=int, default=9)
    parser.add_argument("--c-nm", type=_positive_float, default=15.0)
    parser.add_argument("--a-nm", type=_positive_float, default=7.0)
    parser.add_argument("--eps-m", type=_positive_float, default=1.0)
    parser.add_argument("--weight-center-ev", type=float)
    parser.add_argument("--weight-sigma-ev", type=_positive_float)
    parser.add_argument(
        "--alpha-objective-weight", type=_nonnegative_float, default=1.0
    )
    parser.add_argument(
        "--inverse-objective-weight", type=_nonnegative_float, default=1.2
    )
    parser.add_argument(
        "--max-normalized-rms", type=_positive_float, default=0.025
    )
    parser.add_argument(
        "--max-pointwise-relative-error", type=_positive_float, default=0.05
    )
    parser.add_argument("--seed", type=int, default=12345)
    return parser


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    payload, metadata = calculate_material_dispersion_comparison(args)
    output = atomic_write_npz(
        args.output,
        payload,
        metadata,
        overwrite=args.overwrite,
    )
    print(f"Saved material-dispersion artifact to {output.resolve()}")
    return output


if __name__ == "__main__":
    main()
