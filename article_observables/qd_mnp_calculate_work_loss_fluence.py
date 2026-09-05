"""Calculate the full-QS pulse work-loss response versus incident fluence.

The numerical propagation and artifact generation live in this article script.
The companion ``qd_mnp_plot_work_loss_fluence.py`` reads the resulting NPZ only, so
figure styling never repeats the expensive time-domain calculations.

The primary plotted observable is the Shah-style carrier-frequency estimate

    sigma_QS,work(E_L; F) = k(E_L) Im[alpha_eff(E_L; F)] / epsilon_0,

where ``alpha_eff`` is obtained from the Fourier components of the nonlinear
pulse response.  The artifact also stores the broadband external-field work
``integral E_inc d(mu_total)/dt dt / F``.  Neither quantity is labelled as
metal-only absorption in this undressed local-quasistatic model.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import warnings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import scipy
from scipy.integrate import quad

from qd_mnp_full_qs_model import (
    FullQSSpheroidPulseModel,
    build_positive_dark_reduction,
)
from qd_mnp_rational_fit import (
    AU_DIPOLE_C_M,
    AU_ENERGY_EV,
    AU_ENERGY_J,
    AU_FIELD_V_M,
    AU_LENGTH_M,
    AU_TIME_S,
    C_SI,
    DEBYE_C_M,
    GaussianPulse,
    HybridQDPlasmonModel,
    MATERIAL_HIGH_FREQUENCY_EPSILON,
    MATERIAL_INTERPOLATION,
    NATIVE_MODEL_PROFILE,
    SCHEMA_VERSION,
    au_to_eV,
    au_to_fs,
    eV_to_au,
    field_au_to_si,
    fs_to_au,
    params_to_physical_dict,
    quasistatic_dipole_cross_section_estimates_cm2,
    epsilon_0,
)
from qd_mnp_params import make_params_with_overrides
from qd_mnp_spheroid_equatorial import EquatorialSpheroidGreenInteraction
from qd_mnp_spheroid_green import SpheroidGreenInteraction


WORK_LOSS_FLUENCE_SCHEMA = "qd_mnp.full_qs_work_loss_fluence"
WORK_LOSS_FLUENCE_SCHEMA_VERSION = 1
SIDE_DIRECT_REFERENCE_ORDER_MAX = 8
POLICIES = ("ignore", "raise", "warn")


@dataclass(frozen=True)
class ChannelSpec:
    key: str
    label: str
    orientation: str
    qd_placement: str
    side_transverse_alignment: str | None = None


ARTICLE_CHANNELS: dict[str, ChannelSpec] = {
    "axis_long": ChannelSpec(
        "axis_long", "tip, longitudinal", "long", "axis"
    ),
    "axis_trans": ChannelSpec(
        "axis_trans", "tip, transverse", "trans", "axis"
    ),
    "side_long": ChannelSpec(
        "side_long", "side, longitudinal", "long", "side"
    ),
    "side_trans_radial": ChannelSpec(
        "side_trans_radial",
        "side, transverse radial",
        "trans",
        "side",
        "radial",
    ),
    "side_trans_tangential": ChannelSpec(
        "side_trans_tangential",
        "side, transverse tangential",
        "trans",
        "side",
        "tangential",
    ),
}

# The minimum informative default isolates the effect of QD placement while
# keeping the laser/MNP orientation fixed.  Every article channel remains
# selectable from the CLI.
DEFAULT_CHANNELS = ("axis_long", "side_long")

PUBLICATION_GRID = {
    "fluence_min_j_cm2": 1.0e-8,
    "fluence_max_j_cm2": 1.0e-3,
    "points": 65,
    "scale": "sqrt",
    "max_midpoint_normalized_error": 0.01,
    "max_isolated_pulse_area_step_rad": 0.25,
    "convergence_policy": "raise",
}

QUICK_GRID = {
    "fluence_min_j_cm2": 1.0e-8,
    "fluence_max_j_cm2": 1.0e-3,
    "points": 3,
    "scale": "log",
    "max_midpoint_normalized_error": 0.25,
    "max_isolated_pulse_area_step_rad": 10.0,
    "convergence_policy": "warn",
}

GRID_AUDIT_OBSERVABLES = (
    "sigma_spectral_qs_work_loss_cm2",
    "sigma_energy_transfer_cm2",
    "delta_sigma_spectral_qs_work_loss_cm2",
    "delta_sigma_energy_transfer_cm2",
)


@dataclass(frozen=True)
class ChannelModel:
    spec: ChannelSpec
    params: object
    bright_model: HybridQDPlasmonModel
    full_model: FullQSSpheroidPulseModel
    kernel: object
    dark_reduction: object | None
    resolved_R_nm: float


@dataclass(frozen=True)
class WorkWindowAudit:
    alpha_eff_au3: complex
    sigma_spectral_cm2: float
    sigma_spectral_half_cm2: float
    sigma_energy_half_cm2: float
    sigma_spectral_relative_change: float
    sigma_energy_relative_change: float
    accepted: bool


@dataclass(frozen=True)
class BareMnpPulseWork:
    sigma_energy_transfer_cm2: float
    sigma_energy_cutoff_check_cm2: float
    cutoff_relative_change: float
    quadrature_absolute_error_cm2: float
    quadrature_relative_error: float
    integration_converged: bool
    work_from_incident_field_j: float
    work_cutoff_check_j: float
    work_passivity_tolerance_j: float
    work_nonnegative_within_tolerance: bool
    reference_fluence_j_cm2: float
    dimensionless_carrier_frequency: float
    dimensionless_cutoff_check: float
    dimensionless_cutoff_full: float


def resolved_center_distance_nm(
    spec: ChannelSpec,
    *,
    c_nm: float,
    a_nm: float,
    qd_radius_nm: float,
    gap_nm: float,
) -> float:
    """Return QD/MNP centre distance for a common surface-to-surface gap."""

    values = np.asarray([c_nm, a_nm, qd_radius_nm, gap_nm], dtype=float)
    if np.any(~np.isfinite(values)):
        raise ValueError("Geometry lengths must be finite.")
    if c_nm <= 0.0 or a_nm <= 0.0 or c_nm < a_nm:
        raise ValueError("Require c_nm >= a_nm > 0 for a prolate spheroid.")
    if qd_radius_nm < 0.0 or gap_nm <= 0.0:
        raise ValueError("qd_radius_nm must be non-negative and gap_nm positive.")
    directional_radius = c_nm if spec.qd_placement == "axis" else a_nm
    return float(directional_radius + qd_radius_nm + gap_nm)


def pulse_for_fluence(
    fluence_j_cm2: float,
    *,
    energy_eV: float,
    tau_fs: float,
    tau_kind: str,
    eps_m: float,
) -> GaussianPulse:
    """Construct a Gaussian pulse whose exact real-field fluence is requested."""

    values = np.asarray([fluence_j_cm2, energy_eV, tau_fs, eps_m], dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("Fluence, pulse energy, duration and eps_m must be positive.")
    unit = GaussianPulse(
        E0_au=1.0,
        omegaL_au=float(eV_to_au(energy_eV)),
        tau_au=float(fs_to_au(tau_fs)),
        tau_kind=tau_kind,
    )
    unit_fluence = unit.fluence_j_cm2(eps_m=eps_m)
    return GaussianPulse(
        E0_au=float(np.sqrt(fluence_j_cm2 / unit_fluence)),
        omegaL_au=unit.omegaL_au,
        tau_au=unit.tau_au,
        tau_kind=tau_kind,
    )


def spectral_effective_alpha_au(
    result: object,
    pulse: GaussianPulse,
    eps_m: float,
) -> complex:
    """Return alpha_eff=mu_total/(eps_m E_inc) at the pulse carrier."""

    phase = np.exp(1j * pulse.omegaL_au * np.asarray(result.t_au, dtype=float))
    mu_omega = np.trapezoid(
        np.asarray(result.mu_total_au, dtype=float) * phase,
        np.asarray(result.t_au, dtype=float),
    )
    e_omega = np.trapezoid(
        pulse.field(np.asarray(result.t_au, dtype=float)) * phase,
        np.asarray(result.t_au, dtype=float),
    )
    if abs(e_omega) < 1.0e-30:
        raise FloatingPointError("The carrier Fourier component of the pulse vanished.")
    alpha = complex(mu_omega / (eps_m * e_omega))
    if not np.isfinite(alpha):
        raise FloatingPointError("The carrier-frequency effective alpha is not finite.")
    return alpha


def _prefix_with_endpoint(
    time_au: np.ndarray,
    values: np.ndarray,
    cutoff_au: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a sampled prefix ending exactly at cutoff_au."""

    time = np.asarray(time_au, dtype=float)
    signal = np.asarray(values, dtype=float)
    if time.ndim != 1 or signal.shape != time.shape:
        raise ValueError("A convergence-audit signal must match its 1-D time grid.")
    if cutoff_au <= time[0] or cutoff_au > time[-1]:
        raise ValueError("The convergence-audit cutoff lies outside the trajectory.")
    right = int(np.searchsorted(time, cutoff_au, side="right"))
    prefix_time = time[:right]
    prefix_signal = signal[:right]
    if prefix_time[-1] < cutoff_au:
        prefix_time = np.concatenate((prefix_time, [cutoff_au]))
        prefix_signal = np.concatenate(
            (prefix_signal, [np.interp(cutoff_au, time, signal)])
        )
    return prefix_time, prefix_signal


def _half_window_observables(
    result: object,
    pulse: GaussianPulse,
    eps_m: float,
) -> tuple[complex, float]:
    """Recompute carrier alpha and W/F on the first half of the post window."""

    final_time = float(np.asarray(result.t_au)[-1])
    cutoff = 0.5 * final_time
    if cutoff <= 8.0 * pulse.sigma_t_au:
        return complex(np.nan, np.nan), float("nan")
    time, mu_total = _prefix_with_endpoint(result.t_au, result.mu_total_au, cutoff)
    _, mu_dot = _prefix_with_endpoint(result.t_au, result.mu_dot_total_au, cutoff)
    phase = np.exp(1j * pulse.omegaL_au * time)
    incident = pulse.field(time)
    e_omega = np.trapezoid(incident * phase, time)
    if abs(e_omega) < 1.0e-30:
        return complex(np.nan, np.nan), float("nan")
    alpha = complex(np.trapezoid(mu_total * phase, time) / (eps_m * e_omega))
    work_au = float(np.trapezoid(incident * mu_dot, time))
    sigma_energy = (
        work_au * AU_ENERGY_J / pulse.fluence_j_cm2(eps_m=eps_m)
    )
    return alpha, float(sigma_energy)


def _relative_change(full_value: complex | float, half_value: complex | float) -> float:
    full = complex(full_value)
    half = complex(half_value)
    if not (
        np.isfinite(full.real)
        and np.isfinite(full.imag)
        and np.isfinite(half.real)
        and np.isfinite(half.imag)
    ):
        return float("nan")
    return float(abs(full - half) / max(abs(full), np.finfo(float).tiny))


def _audit_work_observable_window(
    result: object,
    pulse: GaussianPulse,
    eps_m: float,
    max_relative_change: float,
) -> WorkWindowAudit:
    """Compare the published work observables on [t0,T] and [t0,T/2]."""

    alpha = spectral_effective_alpha_au(result, pulse, eps_m)
    sections = quasistatic_dipole_cross_section_estimates_cm2(
        alpha,
        pulse.omegaL_au,
        eps_m,
    )
    sigma_spectral = float(sections.quasistatic_work_loss_cm2)
    alpha_half, sigma_energy_half = _half_window_observables(result, pulse, eps_m)
    if np.isfinite(alpha_half.real) and np.isfinite(alpha_half.imag):
        half_sections = quasistatic_dipole_cross_section_estimates_cm2(
            alpha_half,
            pulse.omegaL_au,
            eps_m,
        )
        sigma_spectral_half = float(half_sections.quasistatic_work_loss_cm2)
    else:
        sigma_spectral_half = float("nan")
    spectral_change = _relative_change(sigma_spectral, sigma_spectral_half)
    energy_change = _relative_change(
        result.sigma_energy_transfer_cm2,
        sigma_energy_half,
    )
    accepted = bool(
        np.isfinite(spectral_change)
        and np.isfinite(energy_change)
        and spectral_change <= max_relative_change
        and energy_change <= max_relative_change
    )
    return WorkWindowAudit(
        alpha_eff_au3=alpha,
        sigma_spectral_cm2=sigma_spectral,
        sigma_spectral_half_cm2=sigma_spectral_half,
        sigma_energy_half_cm2=sigma_energy_half,
        sigma_spectral_relative_change=spectral_change,
        sigma_energy_relative_change=energy_change,
        accepted=accepted,
    )


def _bare_mnp_pulse_work_spectral_average(
    bright_model: HybridQDPlasmonModel,
    pulse: GaussianPulse,
    *,
    max_relative_change: float,
    convergence_policy: str,
    work_passivity_policy: str,
) -> BareMnpPulseWork:
    """Return exact linear bare-MNP pulse work for the active causal ADE fit.

    With the QD removed, each fitted material pole obeys

        q_k'' + gamma_k q_k' + omega_k^2 q_k = f_k E_inc,

    and the physical spheroid dipole is

        mu_MNP = C [alpha_inf E_inc + sum_k q_k].

    Parseval's identity therefore makes ``W_inc/F`` the pulse-spectrum-weighted
    average of ``k Im(alpha)/epsilon_0``.  For the real Gaussian carrier, use
    ``x=sigma_t*omega`` and integrate the exact positive-frequency weight.  No
    second time-domain propagation is needed, and the result is independent of
    fluence for fixed carrier and duration.
    """

    fit = bright_model.fit
    sigma = float(pulse.sigma_t_au)
    x_carrier = sigma * float(pulse.omegaL_au)
    cutoff_check = x_carrier + 10.0
    cutoff_full = x_carrier + 12.0
    normalization = float(
        np.sqrt(np.pi) * (1.0 + np.exp(-(x_carrier**2)))
    )
    if not np.isfinite(normalization) or normalization <= 0.0:
        raise FloatingPointError("The Gaussian spectral-weight normalization failed.")

    def gaussian_weight(x_value: float) -> float:
        return float(
            np.exp(-((x_value - x_carrier) ** 2))
            + np.exp(-((x_value + x_carrier) ** 2))
            + 2.0 * np.exp(-(x_value**2) - x_carrier**2)
        )

    def weighted_cross_section(x_value: float) -> float:
        omega_au = float(x_value / sigma)
        energy_eV = float(au_to_eV(omega_au))
        alpha_dimensionless = complex(
            bright_model.alpha_from_fit(
                np.asarray([energy_eV]),
                allow_extrapolation=True,
            )[0]
        )
        alpha_effective_au = (
            bright_model.C * alpha_dimensionless / bright_model.params.eps_m
        )
        cross_section = quasistatic_dipole_cross_section_estimates_cm2(
            alpha_effective_au,
            omega_au,
            bright_model.params.eps_m,
        )
        return float(cross_section.quasistatic_work_loss_cm2) * gaussian_weight(
            x_value
        )

    split_points = sorted(
        {
            float(value)
            for value in (
                x_carrier,
                *(sigma * np.asarray(fit.omega_modes_au, dtype=float)),
            )
            if 0.0 < float(value) < cutoff_full
        }
    )
    full_integral, full_error = quad(
        weighted_cross_section,
        0.0,
        cutoff_full,
        points=split_points,
        epsabs=1.0e-24,
        epsrel=1.0e-10,
        limit=500,
    )
    check_points = [value for value in split_points if value < cutoff_check]
    check_integral, _ = quad(
        weighted_cross_section,
        0.0,
        cutoff_check,
        points=check_points,
        epsabs=1.0e-24,
        epsrel=1.0e-10,
        limit=500,
    )
    sigma_energy = float(full_integral / normalization)
    sigma_energy_check = float(check_integral / normalization)
    absolute_error = float(abs(full_error) / normalization)
    cutoff_change = _relative_change(sigma_energy, sigma_energy_check)
    quadrature_relative_error = float(
        absolute_error / max(abs(sigma_energy), np.finfo(float).tiny)
    )
    integration_converged = bool(
        np.isfinite(cutoff_change)
        and np.isfinite(quadrature_relative_error)
        and cutoff_change <= max_relative_change
        and quadrature_relative_error <= max_relative_change
    )
    if not integration_converged:
        _apply_policy(
            convergence_policy,
            "The pulse-spectrum integral for bare-MNP work did not converge: "
            f"cutoff relative change={cutoff_change:.6g}, quadrature relative "
            f"error={quadrature_relative_error:.6g}, "
            f"limit={max_relative_change:.6g}.",
        )

    passivity_tolerance_cm2 = float(max(10.0 * absolute_error, 1.0e-15 * abs(sigma_energy)))
    work_nonnegative = bool(sigma_energy >= -passivity_tolerance_cm2)
    if not work_nonnegative:
        _apply_policy(
            work_passivity_policy,
            "The passive bare-MNP causal ADE produced negative pulse-averaged "
            f"work cross section={sigma_energy:.6g} cm^2, tolerance="
            f"{passivity_tolerance_cm2:.6g} cm^2.",
        )

    reference_fluence = float(
        pulse.fluence_j_cm2(eps_m=bright_model.params.eps_m)
    )
    return BareMnpPulseWork(
        sigma_energy_transfer_cm2=sigma_energy,
        sigma_energy_cutoff_check_cm2=sigma_energy_check,
        cutoff_relative_change=cutoff_change,
        quadrature_absolute_error_cm2=absolute_error,
        quadrature_relative_error=quadrature_relative_error,
        integration_converged=integration_converged,
        work_from_incident_field_j=float(sigma_energy * reference_fluence),
        work_cutoff_check_j=float(sigma_energy_check * reference_fluence),
        work_passivity_tolerance_j=float(
            passivity_tolerance_cm2 * reference_fluence
        ),
        work_nonnegative_within_tolerance=work_nonnegative,
        reference_fluence_j_cm2=reference_fluence,
        dimensionless_carrier_frequency=x_carrier,
        dimensionless_cutoff_check=cutoff_check,
        dimensionless_cutoff_full=cutoff_full,
    )


def _apply_policy(policy: str, message: str) -> None:
    if policy == "raise":
        raise RuntimeError(message)
    if policy == "warn":
        warnings.warn(message, RuntimeWarning, stacklevel=3)


def fluence_grid_resolution_diagnostics(
    fluence_j_cm2: np.ndarray,
    observable_values_cm2: np.ndarray,
    isolated_qd_pulse_area_rad: np.ndarray,
    *,
    max_midpoint_normalized_error: float,
    max_isolated_pulse_area_step_rad: float,
) -> dict[str, object]:
    """Audit a fine grid against the nested every-other-point grid.

    The interpolation coordinate is ``sqrt(fluence)``, which is proportional
    to incident field amplitude and to the isolated-QD resonant pulse area.
    Each midpoint residual is normalized by the maximum absolute value of the
    same channel/observable curve.  This gives a dimensionless, stable error
    even when a hybrid-minus-bare curve crosses zero.
    """

    fluence = np.asarray(fluence_j_cm2, dtype=float)
    values = np.asarray(observable_values_cm2, dtype=float)
    pulse_area = np.asarray(isolated_qd_pulse_area_rad, dtype=float)
    if fluence.ndim != 1 or fluence.size < 1:
        raise ValueError("The fluence grid must be a non-empty 1-D array.")
    if values.ndim != 3 or values.shape[2] != fluence.size:
        raise ValueError(
            "Work observables must have shape (n_channels, n_observables, n_fluences)."
        )
    if pulse_area.shape != fluence.shape:
        raise ValueError("Pulse area must match the fluence grid.")
    if np.any(~np.isfinite(values)) or np.any(~np.isfinite(pulse_area)):
        raise ValueError("Grid-audit observables and pulse areas must be finite.")
    for name, tolerance in (
        ("max_midpoint_normalized_error", max_midpoint_normalized_error),
        ("max_isolated_pulse_area_step_rad", max_isolated_pulse_area_step_rad),
    ):
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")

    curve_scale = np.max(np.abs(values), axis=2)
    if fluence.size < 3:
        midpoint_absolute_error = np.full(values.shape[:2], np.inf, dtype=float)
        midpoint_normalized_error = np.full(values.shape[:2], np.inf, dtype=float)
    else:
        amplitude_coordinate = np.sqrt(fluence)
        audit_indices = np.arange(1, fluence.size - 1, 2, dtype=int)
        if audit_indices.size == 0:
            midpoint_absolute_error = np.full(values.shape[:2], np.inf, dtype=float)
            midpoint_normalized_error = np.full(values.shape[:2], np.inf, dtype=float)
        else:
            left = audit_indices - 1
            right = audit_indices + 1
            fractions = (
                amplitude_coordinate[audit_indices] - amplitude_coordinate[left]
            ) / (amplitude_coordinate[right] - amplitude_coordinate[left])
            interpolated = values[:, :, left] + fractions[None, None, :] * (
                values[:, :, right] - values[:, :, left]
            )
            midpoint_absolute_error = np.max(
                np.abs(values[:, :, audit_indices] - interpolated), axis=2
            )
            midpoint_normalized_error = np.divide(
                midpoint_absolute_error,
                curve_scale,
                out=np.zeros_like(midpoint_absolute_error),
                where=curve_scale > np.finfo(float).tiny,
            )
            zero_scale_inconsistent = (
                (curve_scale <= np.finfo(float).tiny)
                & (midpoint_absolute_error > np.finfo(float).tiny)
            )
            midpoint_normalized_error[zero_scale_inconsistent] = np.inf

    maximum_area_step = (
        float(np.max(np.abs(np.diff(pulse_area))))
        if pulse_area.size >= 2
        else float("inf")
    )
    accepted_by_observable = np.asarray(
        midpoint_normalized_error <= max_midpoint_normalized_error,
        dtype=bool,
    )
    accepted_by_channel = np.asarray(
        np.all(accepted_by_observable, axis=1)
        & (maximum_area_step <= max_isolated_pulse_area_step_rad),
        dtype=bool,
    )
    return {
        "midpoint_absolute_error_cm2": midpoint_absolute_error,
        "curve_scale_cm2": curve_scale,
        "midpoint_normalized_error": midpoint_normalized_error,
        "maximum_isolated_pulse_area_step_rad": maximum_area_step,
        "accepted_by_observable": accepted_by_observable,
        "accepted_by_channel": accepted_by_channel,
        "accepted": bool(np.all(accepted_by_channel)),
        "audited_coordinate": (
            "sqrt(fluence), proportional to incident field amplitude and "
            "isolated-QD resonant pulse area"
        ),
    }


def _build_channel_model(
    spec: ChannelSpec,
    *,
    gap_nm: float,
    c_nm: float,
    a_nm: float,
    qd_radius_nm: float,
    eps_m: float,
    eps_qd: float,
    d_debye: float | None,
    omega0_eV: float,
    gamma_population_meV: float | None,
    gamma2_coherence_meV: float | None,
    qd_dipole_convention: str,
    spatial_order_max: int,
    material_fit_modes: int,
    fit_window_eV: tuple[float, float],
    fit_seed: int,
    alpha_objective_weight: float,
    inv_alpha_objective_weight: float,
    max_bright_fit_normalized_rms: float | None,
    max_bright_fit_pointwise_relative_error: float | None,
    bright_fit_quality_policy: str,
    radiative_consistency_policy: str,
    fit_quality_policy: str,
    max_modal_normalized_rms: float,
    max_modal_relative_error: float,
    modal_audit_points: int,
    spatial_convergence_policy: str,
    spatial_convergence_rtol: float,
    reduction_fit_grid_points: int,
    reduction_audit_grid_points: int,
    reduction_rms_tolerance: float,
    reduction_max_tolerance: float,
    reduction_max_nodes: int | None,
    reduction_policy: str,
    reduction_reaudit_points: int,
) -> ChannelModel:
    resolved_R_nm = resolved_center_distance_nm(
        spec,
        c_nm=c_nm,
        a_nm=a_nm,
        qd_radius_nm=qd_radius_nm,
        gap_nm=gap_nm,
    )
    params = make_params_with_overrides(
        c_nm=c_nm,
        a_nm=a_nm,
        r_nm=resolved_R_nm,
        qd_radius_nm=qd_radius_nm,
        eps_m=eps_m,
        eps_qd=eps_qd,
        d_debye=d_debye,
        omega0_ev=omega0_eV,
        gamma_population_mev=gamma_population_meV,
        gamma2_coherence_mev=gamma2_coherence_meV,
        qd_dipole_convention=qd_dipole_convention,
        orientation=spec.orientation,
        qd_placement=spec.qd_placement,
        side_transverse_alignment=spec.side_transverse_alignment,
    )
    bright_model = HybridQDPlasmonModel(
        params,
        orientation=spec.orientation,
        n_modes=material_fit_modes,
        fit_window_eV=fit_window_eV,
        alpha_objective_weight=alpha_objective_weight,
        inv_alpha_objective_weight=inv_alpha_objective_weight,
        max_fit_normalized_rms=None,
        max_fit_pointwise_relative_error=None,
        radiative_consistency_policy=radiative_consistency_policy,
        seed=fit_seed,
        verbose=False,
    )
    fit = bright_model.fit
    bright_fit_failed = bool(
        (
            max_bright_fit_normalized_rms is not None
            and (
                fit.normalized_rms_alpha > max_bright_fit_normalized_rms
                or fit.normalized_rms_inv_alpha > max_bright_fit_normalized_rms
            )
        )
        or (
            max_bright_fit_pointwise_relative_error is not None
            and fit.max_normalized_alpha_error
            > max_bright_fit_pointwise_relative_error
        )
    )
    if bright_fit_failed:
        _apply_policy(
            bright_fit_quality_policy,
            "The deliberately selected material Lorentz fit misses the "
            "configured accuracy gate: "
            f"NRMS(alpha)={fit.normalized_rms_alpha:.6g}, "
            f"NRMS(1/alpha)={fit.normalized_rms_inv_alpha:.6g}, "
            f"max normalized alpha error={fit.max_normalized_alpha_error:.6g}; "
            "this is expected for a one-oscillator control but must be "
            "reported as an approximation, not as an accurate Au fit.",
        )
    if spec.qd_placement == "side":
        kernel = EquatorialSpheroidGreenInteraction.from_params(
            params,
            orientation=spec.orientation,
            n_max=spatial_order_max,
        )
    else:
        kernel = SpheroidGreenInteraction.from_params(
            params,
            orientation=spec.orientation,
            n_max=spatial_order_max,
        )

    dark_reduction = None
    if spec.qd_placement == "side" and spatial_order_max > SIDE_DIRECT_REFERENCE_ORDER_MAX:
        dark_reduction = build_positive_dark_reduction(
            bright_model,
            kernel,
            fit_grid_points=reduction_fit_grid_points,
            audit_grid_points=reduction_audit_grid_points,
            rms_tolerance=reduction_rms_tolerance,
            max_tolerance=reduction_max_tolerance,
            max_nodes=reduction_max_nodes,
            policy=reduction_policy,
        )
        if not dark_reduction.diagnostics.accepted:
            raise RuntimeError(
                f"Channel {spec.key!r} has no accepted positive dark-mode reduction."
            )

    full_model = FullQSSpheroidPulseModel(
        bright_model,
        kernel,
        dark_reduction=dark_reduction,
        fit_quality_policy=fit_quality_policy,
        max_modal_normalized_rms=max_modal_normalized_rms,
        max_modal_relative_error=max_modal_relative_error,
        modal_audit_points=modal_audit_points,
        spatial_convergence_policy=spatial_convergence_policy,
        spatial_convergence_rtol=spatial_convergence_rtol,
        reduction_reaudit_points=reduction_reaudit_points,
        max_reduction_normalized_rms=reduction_rms_tolerance,
        max_reduction_normalized_error=reduction_max_tolerance,
    )
    return ChannelModel(
        spec=spec,
        params=params,
        bright_model=bright_model,
        full_model=full_model,
        kernel=kernel,
        dark_reduction=dark_reduction,
        resolved_R_nm=resolved_R_nm,
    )


def _solve_with_tail_extension(
    model: FullQSSpheroidPulseModel,
    pulse: GaussianPulse,
    *,
    pre_sigma: float,
    post_fs: float | None,
    max_auto_tail_extensions: int,
    tail_policy: str,
    tail_ratio_tolerance: float,
    tail_window_fraction: float,
    method: str,
    rtol: float,
    atol: float,
    points_per_fastest_cycle: float,
    spectral_window_policy: str,
    max_spectral_leakage: float,
    positivity_policy: str,
    positivity_tolerance: float,
    work_passivity_policy: str,
    eps_m: float,
    observable_convergence_policy: str,
    max_observable_window_relative_change: float,
) -> tuple[object, float, int, WorkWindowAudit]:
    start_au = -pre_sigma * pulse.sigma_t_au
    if post_fs is None:
        end_au = max(
            pre_sigma * pulse.sigma_t_au,
            model.recommended_post_pulse_time_au(),
        )
    else:
        end_au = float(fs_to_au(post_fs))
    extensions = 0
    while True:
        result = model.solve(
            pulse,
            t_span_au=(float(start_au), float(end_au)),
            method=method,
            rtol=rtol,
            atol=atol,
            points_per_fastest_cycle=points_per_fastest_cycle,
            spectral_window_policy=spectral_window_policy,
            max_spectral_leakage=max_spectral_leakage,
            positivity_policy=positivity_policy,
            positivity_tolerance=positivity_tolerance,
            work_passivity_policy=work_passivity_policy,
            response_tail_policy="ignore",
            response_tail_tolerance=tail_ratio_tolerance,
            response_tail_window_fraction=tail_window_fraction,
        )
        audit = _audit_work_observable_window(
            result,
            pulse,
            eps_m,
            max_observable_window_relative_change,
        )
        tail_accepted = bool(result.diagnostics.response_tail_converged)
        if tail_accepted and audit.accepted:
            break
        if post_fs is not None or extensions >= max_auto_tail_extensions:
            if not tail_accepted:
                _apply_policy(
                    tail_policy,
                    "The full-QS response tail did not converge: "
                    f"ratio={result.diagnostics.response_tail_ratio:.6g}, "
                    f"limit={tail_ratio_tolerance:.6g}.",
                )
            if not audit.accepted:
                _apply_policy(
                    observable_convergence_policy,
                    "Carrier-frequency and/or pulse-integrated work has not "
                    "converged with respect to truncating the saved trajectory "
                    "at half of its final positive time: relative changes="
                    f"({audit.sigma_spectral_relative_change:.6g}, "
                    f"{audit.sigma_energy_relative_change:.6g}), limit="
                    f"{max_observable_window_relative_change:.6g}.",
                )
            break
        end_au *= 2.0
        extensions += 1
    return result, float(au_to_fs(end_au)), extensions, audit


def _channel_metadata(bundle: ChannelModel, carrier_energy_eV: float) -> dict[str, object]:
    model = bundle.full_model
    bright = bundle.bright_model
    fit = bright.fit
    response = model.frequency_response_from_fit(np.asarray([carrier_energy_eV]))
    physical = params_to_physical_dict(bundle.params, bundle.spec.orientation)
    physical["coupling_model"] = "full_quasistatic_spheroid_field_point_qd"
    spatial = model.spatial_convergence_diagnostics
    modal = model.modal_fit_diagnostics
    stability = model.coupled_stability
    reduction = None
    if bundle.dark_reduction is not None:
        diagnostics = bundle.dark_reduction.diagnostics
        reduction = {
            "accepted": bool(diagnostics.accepted),
            "original_dark_mode_count": int(diagnostics.original_dark_mode_count),
            "reduced_node_count": int(diagnostics.reduced_node_count),
            "max_normalized_rms": float(diagnostics.max_normalized_rms),
            "max_normalized_error": float(diagnostics.max_normalized_error),
        }
    return {
        "key": bundle.spec.key,
        "label": bundle.spec.label,
        "orientation": bundle.spec.orientation,
        "qd_placement": bundle.spec.qd_placement,
        "side_transverse_alignment": bundle.spec.side_transverse_alignment,
        "resolved_R_nm": bundle.resolved_R_nm,
        "resolved_physical_parameters": physical,
        "material_fit": {
            "alpha_inf_au3": float(fit.alpha_inf),
            "strengths_au2": [float(value) for value in fit.strengths_au2],
            "omega_modes_au": [float(value) for value in fit.omega_modes_au],
            "gamma_modes_au": [float(value) for value in fit.gamma_modes_au],
            "nonnegative_imaginary_part_all_positive_frequencies": bool(
                fit.nonnegative_imaginary_part_all_positive_frequencies
            ),
            "normalized_rms_alpha": float(fit.normalized_rms_alpha),
            "normalized_rms_inverse_alpha": float(fit.normalized_rms_inv_alpha),
            "max_normalized_alpha_error": float(fit.max_normalized_alpha_error),
            "minimum_imaginary_alpha_fit_window": float(
                fit.min_imag_alpha_fit_window
            ),
            "passivity_grid_points": int(fit.passivity_grid_points),
            "passive_on_fit_window": bool(fit.passive_on_fit_window),
        },
        "full_qs": {
            "spatial_order_max": int(model.spatial_order_max),
            "exact_spatial_mode_count": int(model.exact_spatial_mode_count),
            "dynamic_spatial_mode_count": int(model.n_spatial_modes),
            "material_poles_per_spatial_mode": int(model.n_material_modes),
            "spatial_convergence_accepted": bool(spatial.accepted),
            "spatial_audit_grid_points": int(spatial.audit_grid_points),
            "spatial_convergence_tolerance": float(spatial.tolerance),
            "spatial_max_half_order_relative_change": float(
                spatial.max_half_order_relative_change
            ),
            "spatial_max_tail_block_relative_mass": float(
                spatial.max_tail_block_relative_mass
            ),
            "modal_fit_accepted": bool(modal.accepted),
            "modal_fit_passive_on_audit_grid": bool(
                modal.passive_on_audit_grid
            ),
            "modal_audit_grid_points": int(modal.audit_grid_points),
            "modal_fit_max_normalized_rms": float(modal.max_normalized_rms),
            "modal_fit_max_relative_error": float(modal.max_relative_error),
            "linearized_ground_state_stable": bool(stability.stable),
            "spectral_abscissa_au": (
                None
                if stability.spectral_abscissa_au is None
                else float(stability.spectral_abscissa_au)
            ),
            "dark_reduction": reduction,
            "carrier_response": {
                "A_au3": _complex_json(np.asarray(response.A_au3).reshape(-1)[0]),
                "B": _complex_json(np.asarray(response.B).reshape(-1)[0]),
                "K_au_minus3": _complex_json(
                    np.asarray(response.K_au_minus3).reshape(-1)[0]
                ),
            },
        },
    }


def _complex_json(value: complex) -> dict[str, float]:
    scalar = complex(value)
    return {"real": float(scalar.real), "imag": float(scalar.imag)}


def _source_file_hashes() -> dict[str, str]:
    """Fingerprint the exact local implementations used to create an artifact."""

    paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "qd_mnp_full_qs_model.py",
        PROJECT_ROOT / "qd_mnp_params.py",
        PROJECT_ROOT / "qd_mnp_rational_fit.py",
        PROJECT_ROOT / "qd_mnp_spheroid_equatorial.py",
        PROJECT_ROOT / "qd_mnp_spheroid_green.py",
    )
    return {
        path.relative_to(PROJECT_ROOT).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in paths
    }


def _git_provenance() -> dict[str, object]:
    provenance: dict[str, object] = {"commit": None, "working_tree_dirty": None}
    repository = PROJECT_ROOT
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=repository,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            cwd=repository,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return provenance
    provenance["commit"] = commit
    provenance["working_tree_dirty"] = bool(status.strip())
    return provenance


def _validate_calculation_inputs(
    *,
    channel_keys: tuple[str, ...],
    fluence_j_cm2: np.ndarray,
    spatial_order_max: int,
    material_fit_modes: int,
    fit_window_eV: tuple[float, float],
    pre_sigma: float,
    max_auto_tail_extensions: int,
    modal_audit_points: int,
    reduction_reaudit_points: int,
    alpha_objective_weight: float,
    inv_alpha_objective_weight: float,
    max_observable_window_relative_change: float,
    max_fluence_grid_midpoint_normalized_error: float,
    max_isolated_pulse_area_step_rad: float,
) -> None:
    if not channel_keys:
        raise ValueError("At least one article channel is required.")
    if len(set(channel_keys)) != len(channel_keys):
        raise ValueError("Channel names must be unique.")
    unknown = [key for key in channel_keys if key not in ARTICLE_CHANNELS]
    if unknown:
        raise ValueError(f"Unknown article channel(s): {', '.join(unknown)}")
    if fluence_j_cm2.ndim != 1 or fluence_j_cm2.size < 1:
        raise ValueError("fluence_j_cm2 must be a non-empty one-dimensional array.")
    if np.any(~np.isfinite(fluence_j_cm2)) or np.any(fluence_j_cm2 <= 0.0):
        raise ValueError("All fluences must be finite and positive.")
    if np.any(np.diff(fluence_j_cm2) <= 0.0):
        raise ValueError("Fluences must be strictly increasing.")
    if spatial_order_max < 1 or material_fit_modes < 1:
        raise ValueError("Spatial order and material fit-mode count must be positive.")
    if len(fit_window_eV) != 2 or not (
        np.isfinite(fit_window_eV).all()
        and 0.0 < fit_window_eV[0] < fit_window_eV[1]
    ):
        raise ValueError("fit_window_eV must satisfy 0 < min < max.")
    if not np.isfinite(pre_sigma) or pre_sigma < 6.0:
        raise ValueError("pre_sigma must be finite and at least 6.")
    if max_auto_tail_extensions < 0:
        raise ValueError("max_auto_tail_extensions must be non-negative.")
    if modal_audit_points < 101 or reduction_reaudit_points < 101:
        raise ValueError("Modal audit and reduction re-audit grids require >= 101 points.")
    if alpha_objective_weight < 0.0 or inv_alpha_objective_weight < 0.0:
        raise ValueError("Fit objective weights must be non-negative.")
    if alpha_objective_weight == 0.0 and inv_alpha_objective_weight == 0.0:
        raise ValueError("At least one fit objective weight must be positive.")
    if (
        not np.isfinite(max_observable_window_relative_change)
        or max_observable_window_relative_change <= 0.0
    ):
        raise ValueError(
            "max_observable_window_relative_change must be finite and positive."
        )
    for name, value in (
        (
            "max_fluence_grid_midpoint_normalized_error",
            max_fluence_grid_midpoint_normalized_error,
        ),
        ("max_isolated_pulse_area_step_rad", max_isolated_pulse_area_step_rad),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")


def calculate_work_loss_fluence(
    output_path: str | Path,
    *,
    channel_keys: tuple[str, ...] = DEFAULT_CHANNELS,
    fluence_j_cm2: np.ndarray | list[float] | None = None,
    fluence_grid_scale: str = "sqrt",
    calculation_preset: str = "publication",
    carrier_energy_eV: float = 2.042,
    pulse_tau_fs: float = 20.0,
    pulse_tau_kind: str = "fwhm_intensity",
    gap_nm: float = 1.0,
    c_nm: float = 15.0,
    a_nm: float = 7.0,
    qd_radius_nm: float = 2.0,
    eps_m: float = 1.0,
    eps_qd: float = 6.0,
    d_debye: float | None = None,
    omega0_eV: float = 2.042,
    gamma_population_meV: float | None = None,
    gamma2_coherence_meV: float | None = None,
    qd_dipole_convention: str = "effective_external",
    spatial_order_max: int = 80,
    material_fit_modes: int = 9,
    fit_window_eV: tuple[float, float] = (0.8, 3.0),
    fit_seed: int = 12345,
    alpha_objective_weight: float = 1.0,
    inv_alpha_objective_weight: float = 1.2,
    max_bright_fit_normalized_rms: float | None = 0.025,
    max_bright_fit_pointwise_relative_error: float | None = 0.05,
    bright_fit_quality_policy: str = "raise",
    radiative_consistency_policy: str = "warn",
    fit_quality_policy: str = "raise",
    max_modal_normalized_rms: float = 0.03,
    max_modal_relative_error: float = 0.06,
    modal_audit_points: int = 2001,
    spatial_convergence_policy: str = "raise",
    spatial_convergence_rtol: float = 2.0e-5,
    reduction_fit_grid_points: int = 1001,
    reduction_audit_grid_points: int = 1601,
    reduction_rms_tolerance: float = 1.0e-6,
    reduction_max_tolerance: float = 1.0e-4,
    reduction_max_nodes: int | None = None,
    reduction_policy: str = "raise",
    reduction_reaudit_points: int = 1709,
    method: str = "DOP853",
    rtol: float = 1.0e-8,
    atol: float = 1.0e-10,
    points_per_fastest_cycle: float = 20.0,
    pre_sigma: float = 10.0,
    post_fs: float | None = None,
    spectral_window_policy: str = "raise",
    max_spectral_leakage: float = 1.0e-3,
    positivity_policy: str = "raise",
    positivity_tolerance: float = 1.0e-7,
    work_passivity_policy: str = "raise",
    tail_policy: str = "raise",
    tail_ratio_tolerance: float = 1.0e-4,
    tail_window_fraction: float = 0.05,
    max_auto_tail_extensions: int = 3,
    observable_convergence_policy: str = "raise",
    max_observable_window_relative_change: float = 1.0e-3,
    fluence_grid_convergence_policy: str = "raise",
    max_fluence_grid_midpoint_normalized_error: float = 0.01,
    max_isolated_pulse_area_step_rad: float = 0.25,
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """Run full-QS dynamics and write one self-describing compressed NPZ."""

    output = Path(output_path)
    channel_keys = tuple(channel_keys)
    if fluence_j_cm2 is None:
        fluences = np.linspace(
            np.sqrt(PUBLICATION_GRID["fluence_min_j_cm2"]),
            np.sqrt(PUBLICATION_GRID["fluence_max_j_cm2"]),
            int(PUBLICATION_GRID["points"]),
        ) ** 2
    else:
        fluences = np.asarray(fluence_j_cm2, dtype=float)
    _validate_calculation_inputs(
        channel_keys=channel_keys,
        fluence_j_cm2=fluences,
        spatial_order_max=spatial_order_max,
        material_fit_modes=material_fit_modes,
        fit_window_eV=fit_window_eV,
        pre_sigma=pre_sigma,
        max_auto_tail_extensions=max_auto_tail_extensions,
        modal_audit_points=modal_audit_points,
        reduction_reaudit_points=reduction_reaudit_points,
        alpha_objective_weight=alpha_objective_weight,
        inv_alpha_objective_weight=inv_alpha_objective_weight,
        max_observable_window_relative_change=(
            max_observable_window_relative_change
        ),
        max_fluence_grid_midpoint_normalized_error=(
            max_fluence_grid_midpoint_normalized_error
        ),
        max_isolated_pulse_area_step_rad=max_isolated_pulse_area_step_rad,
    )
    for name, policy in (
        ("radiative_consistency_policy", radiative_consistency_policy),
        ("bright_fit_quality_policy", bright_fit_quality_policy),
        ("fit_quality_policy", fit_quality_policy),
        ("spatial_convergence_policy", spatial_convergence_policy),
        ("reduction_policy", reduction_policy),
        ("spectral_window_policy", spectral_window_policy),
        ("positivity_policy", positivity_policy),
        ("work_passivity_policy", work_passivity_policy),
        ("tail_policy", tail_policy),
        ("observable_convergence_policy", observable_convergence_policy),
        ("fluence_grid_convergence_policy", fluence_grid_convergence_policy),
    ):
        if policy not in POLICIES:
            raise ValueError(f"{name} must be one of {POLICIES}.")
    if fluence_grid_scale not in {"sqrt", "log", "linear", "custom"}:
        raise ValueError(
            "fluence_grid_scale must be one of ('sqrt', 'log', 'linear', 'custom')."
        )
    if calculation_preset not in {"publication", "quick", "custom"}:
        raise ValueError("calculation_preset must be publication, quick, or custom.")
    if output.suffix.lower() != ".npz":
        raise ValueError("output_path must have the .npz extension.")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {output}")

    n_channels = len(channel_keys)
    n_fluences = fluences.size
    shape = (n_channels, n_fluences)
    float_arrays = {
        name: np.full(shape, np.nan, dtype=float)
        for name in (
            "fluence_j_cm2",
            "pulse_E0_au",
            "pulse_E0_v_m",
            "peak_intensity_w_cm2",
            "alpha_eff_real_au3",
            "alpha_eff_imag_au3",
            "sigma_spectral_qs_work_loss_cm2",
            "sigma_bare_mnp_qs_work_loss_cm2",
            "delta_sigma_spectral_qs_work_loss_cm2",
            "sigma_energy_transfer_cm2",
            "sigma_bare_mnp_energy_transfer_cm2",
            "delta_sigma_energy_transfer_cm2",
            "sigma_spectral_half_window_cm2",
            "sigma_energy_half_window_cm2",
            "sigma_bare_mnp_energy_cutoff_check_cm2",
            "sigma_spectral_half_window_relative_change",
            "sigma_energy_half_window_relative_change",
            "sigma_bare_mnp_energy_cutoff_relative_change",
            "sigma_bare_mnp_energy_quadrature_relative_error",
            "work_from_incident_field_j",
            "excited_population_final",
            "excited_population_max",
            "response_tail_ratio",
            "pulse_spectral_leakage",
            "qd_source_spectral_leakage",
            "mnp_dipole_spectral_leakage",
            "mnp_drive_spectral_leakage",
            "mnp_field_spectral_leakage",
            "min_density_eigenvalue",
            "max_bloch_radius",
            "excited_population_min",
            "boundary_envelope_fraction",
            "solver_max_step_limit_au",
            "integration_frequency_ceiling_au",
            "incident_peak_rabi_frequency_au",
            "observed_peak_rabi_frequency_au",
            "work_passivity_tolerance_au",
            "post_fs_effective",
        )
    }
    bool_arrays = {
        name: np.zeros(shape, dtype=bool)
        for name in (
            "solver_success",
            "t_final_reached",
            "state_is_finite",
            "response_tail_converged",
            "observable_window_converged",
            "bare_mnp_energy_integration_converged",
            "work_nonnegative_within_tolerance",
            "bare_mnp_work_nonnegative_within_tolerance",
        )
    }
    int_arrays = {
        name: np.zeros(shape, dtype=np.int64)
        for name in (
            "solver_status",
            "solver_n_steps",
            "solver_nfev",
            "tail_extension_count",
            "rabi_step_refinement_count",
        )
    }

    bundles: list[ChannelModel] = []
    channel_metadata: list[dict[str, object]] = []
    bare_work_by_channel = np.empty(n_channels, dtype=float)
    carrier_A = np.empty(n_channels, dtype=complex)
    carrier_B = np.empty(n_channels, dtype=complex)
    carrier_K = np.empty(n_channels, dtype=complex)
    fit_alpha_inf = np.empty(n_channels, dtype=float)
    fit_strengths = np.empty((n_channels, material_fit_modes), dtype=float)
    fit_omega = np.empty_like(fit_strengths)
    fit_gamma = np.empty_like(fit_strengths)
    fit_energies_used: list[np.ndarray] = []
    fit_alpha_used: list[np.ndarray] = []
    bare_pulse_work_by_channel: list[BareMnpPulseWork] = []
    isolated_qd_pulse_area_rad = np.empty(n_fluences, dtype=float)

    for channel_index, key in enumerate(channel_keys):
        spec = ARTICLE_CHANNELS[key]
        if verbose:
            print(f"Building full-QS channel {key} ({channel_index + 1}/{n_channels})")
        bundle = _build_channel_model(
            spec,
            gap_nm=gap_nm,
            c_nm=c_nm,
            a_nm=a_nm,
            qd_radius_nm=qd_radius_nm,
            eps_m=eps_m,
            eps_qd=eps_qd,
            d_debye=d_debye,
            omega0_eV=omega0_eV,
            gamma_population_meV=gamma_population_meV,
            gamma2_coherence_meV=gamma2_coherence_meV,
            qd_dipole_convention=qd_dipole_convention,
            spatial_order_max=spatial_order_max,
            material_fit_modes=material_fit_modes,
            fit_window_eV=fit_window_eV,
            fit_seed=fit_seed,
            alpha_objective_weight=alpha_objective_weight,
            inv_alpha_objective_weight=inv_alpha_objective_weight,
            max_bright_fit_normalized_rms=max_bright_fit_normalized_rms,
            max_bright_fit_pointwise_relative_error=(
                max_bright_fit_pointwise_relative_error
            ),
            bright_fit_quality_policy=bright_fit_quality_policy,
            radiative_consistency_policy=radiative_consistency_policy,
            fit_quality_policy=fit_quality_policy,
            max_modal_normalized_rms=max_modal_normalized_rms,
            max_modal_relative_error=max_modal_relative_error,
            modal_audit_points=modal_audit_points,
            spatial_convergence_policy=spatial_convergence_policy,
            spatial_convergence_rtol=spatial_convergence_rtol,
            reduction_fit_grid_points=reduction_fit_grid_points,
            reduction_audit_grid_points=reduction_audit_grid_points,
            reduction_rms_tolerance=reduction_rms_tolerance,
            reduction_max_tolerance=reduction_max_tolerance,
            reduction_max_nodes=reduction_max_nodes,
            reduction_policy=reduction_policy,
            reduction_reaudit_points=reduction_reaudit_points,
        )
        bundles.append(bundle)
        channel_metadata.append(_channel_metadata(bundle, carrier_energy_eV))

        fit = bundle.bright_model.fit
        fit_alpha_inf[channel_index] = fit.alpha_inf
        fit_strengths[channel_index] = fit.strengths_au2
        fit_omega[channel_index] = fit.omega_modes_au
        fit_gamma[channel_index] = fit.gamma_modes_au
        fit_energies_used.append(np.asarray(fit.energies_used_eV, dtype=float))
        fit_alpha_used.append(np.asarray(fit.alpha_used, dtype=complex))
        response = bundle.full_model.frequency_response_from_fit(
            np.asarray([carrier_energy_eV])
        )
        carrier_A[channel_index] = np.asarray(response.A_au3).reshape(-1)[0]
        carrier_B[channel_index] = np.asarray(response.B).reshape(-1)[0]
        carrier_K[channel_index] = np.asarray(response.K_au_minus3).reshape(-1)[0]

        alpha_bare = (
            bundle.bright_model.C
            * bundle.bright_model.alpha_from_fit(np.asarray([carrier_energy_eV]))[0]
            / bundle.params.eps_m
        )
        bare_sections = quasistatic_dipole_cross_section_estimates_cm2(
            alpha_bare,
            float(eV_to_au(carrier_energy_eV)),
            bundle.params.eps_m,
        )
        bare_work = float(bare_sections.quasistatic_work_loss_cm2)
        bare_work_by_channel[channel_index] = bare_work
        bare_reference_pulse = pulse_for_fluence(
            float(fluences[0]),
            energy_eV=carrier_energy_eV,
            tau_fs=pulse_tau_fs,
            tau_kind=pulse_tau_kind,
            eps_m=bundle.params.eps_m,
        )
        bare_pulse_work = _bare_mnp_pulse_work_spectral_average(
            bundle.bright_model,
            bare_reference_pulse,
            max_relative_change=max_observable_window_relative_change,
            convergence_policy=observable_convergence_policy,
            work_passivity_policy=work_passivity_policy,
        )
        bare_pulse_work_by_channel.append(bare_pulse_work)
        channel_metadata[-1]["bare_mnp_pulse_work"] = {
            "definition": (
                "exact pulse-spectrum average of the linear causal-ADE "
                "work-loss spectrum for the isolated MNP; the reported cross "
                "section is fluence-independent for fixed pulse shape"
            ),
            "sigma_energy_transfer_cm2": (
                bare_pulse_work.sigma_energy_transfer_cm2
            ),
            "sigma_energy_cutoff_check_cm2": (
                bare_pulse_work.sigma_energy_cutoff_check_cm2
            ),
            "cutoff_relative_change": (
                bare_pulse_work.cutoff_relative_change
            ),
            "quadrature_absolute_error_cm2": (
                bare_pulse_work.quadrature_absolute_error_cm2
            ),
            "quadrature_relative_error": bare_pulse_work.quadrature_relative_error,
            "integration_converged": bare_pulse_work.integration_converged,
            "work_from_incident_field_j": (
                bare_pulse_work.work_from_incident_field_j
            ),
            "work_cutoff_check_j": bare_pulse_work.work_cutoff_check_j,
            "work_passivity_tolerance_j": (
                bare_pulse_work.work_passivity_tolerance_j
            ),
            "work_nonnegative_within_tolerance": (
                bare_pulse_work.work_nonnegative_within_tolerance
            ),
            "reference_fluence_j_cm2": bare_pulse_work.reference_fluence_j_cm2,
            "dimensionless_carrier_frequency": (
                bare_pulse_work.dimensionless_carrier_frequency
            ),
            "dimensionless_cutoff_check": (
                bare_pulse_work.dimensionless_cutoff_check
            ),
            "dimensionless_cutoff_full": bare_pulse_work.dimensionless_cutoff_full,
        }

        for fluence_index, requested_fluence in enumerate(fluences):
            if verbose:
                print(
                    f"  {key}: fluence {fluence_index + 1}/{n_fluences} "
                    f"= {requested_fluence:.6g} J/cm^2"
                )
            pulse = pulse_for_fluence(
                float(requested_fluence),
                energy_eV=carrier_energy_eV,
                tau_fs=pulse_tau_fs,
                tau_kind=pulse_tau_kind,
                eps_m=bundle.params.eps_m,
            )
            pulse_area = float(
                bundle.params.qd_local_field_factor
                * bundle.params.d_au
                * pulse.E0_au
                * np.sqrt(2.0 * np.pi)
                * pulse.sigma_t_au
            )
            if channel_index == 0:
                isolated_qd_pulse_area_rad[fluence_index] = pulse_area
            elif not np.isclose(
                pulse_area,
                isolated_qd_pulse_area_rad[fluence_index],
                rtol=2.0e-13,
                atol=0.0,
            ):
                raise RuntimeError(
                    "Article channels unexpectedly use different isolated-QD "
                    "pulse-area normalizations."
                )
            (
                result,
                effective_post_fs,
                extension_count,
                window_audit,
            ) = _solve_with_tail_extension(
                bundle.full_model,
                pulse,
                pre_sigma=pre_sigma,
                post_fs=post_fs,
                max_auto_tail_extensions=max_auto_tail_extensions,
                tail_policy=tail_policy,
                tail_ratio_tolerance=tail_ratio_tolerance,
                tail_window_fraction=tail_window_fraction,
                method=method,
                rtol=rtol,
                atol=atol,
                points_per_fastest_cycle=points_per_fastest_cycle,
                spectral_window_policy=spectral_window_policy,
                max_spectral_leakage=max_spectral_leakage,
                positivity_policy=positivity_policy,
                positivity_tolerance=positivity_tolerance,
                work_passivity_policy=work_passivity_policy,
                eps_m=bundle.params.eps_m,
                observable_convergence_policy=observable_convergence_policy,
                max_observable_window_relative_change=(
                    max_observable_window_relative_change
                ),
            )
            alpha = window_audit.alpha_eff_au3
            sigma_work = window_audit.sigma_spectral_cm2
            spectral_window_change = window_audit.sigma_spectral_relative_change
            energy_window_change = window_audit.sigma_energy_relative_change
            diagnostics = result.diagnostics
            values = {
                "fluence_j_cm2": result.fluence_j_cm2,
                "pulse_E0_au": pulse.E0_au,
                "pulse_E0_v_m": float(field_au_to_si(pulse.E0_au)),
                "peak_intensity_w_cm2": result.peak_intensity_w_cm2,
                "alpha_eff_real_au3": alpha.real,
                "alpha_eff_imag_au3": alpha.imag,
                "sigma_spectral_qs_work_loss_cm2": sigma_work,
                "sigma_bare_mnp_qs_work_loss_cm2": bare_work,
                "delta_sigma_spectral_qs_work_loss_cm2": sigma_work - bare_work,
                "sigma_energy_transfer_cm2": result.sigma_energy_transfer_cm2,
                "sigma_bare_mnp_energy_transfer_cm2": (
                    bare_pulse_work.sigma_energy_transfer_cm2
                ),
                "delta_sigma_energy_transfer_cm2": (
                    result.sigma_energy_transfer_cm2
                    - bare_pulse_work.sigma_energy_transfer_cm2
                ),
                "sigma_spectral_half_window_cm2": (
                    window_audit.sigma_spectral_half_cm2
                ),
                "sigma_energy_half_window_cm2": window_audit.sigma_energy_half_cm2,
                "sigma_bare_mnp_energy_cutoff_check_cm2": (
                    bare_pulse_work.sigma_energy_cutoff_check_cm2
                ),
                "sigma_spectral_half_window_relative_change": (
                    spectral_window_change
                ),
                "sigma_energy_half_window_relative_change": energy_window_change,
                "sigma_bare_mnp_energy_cutoff_relative_change": (
                    bare_pulse_work.cutoff_relative_change
                ),
                "sigma_bare_mnp_energy_quadrature_relative_error": (
                    bare_pulse_work.quadrature_relative_error
                ),
                "work_from_incident_field_j": result.work_from_incident_field_j,
                "excited_population_final": result.rho22[-1],
                "excited_population_max": np.max(result.rho22),
                "response_tail_ratio": diagnostics.response_tail_ratio,
                "pulse_spectral_leakage": diagnostics.pulse_spectral_leakage,
                "qd_source_spectral_leakage": diagnostics.qd_source_spectral_leakage,
                "mnp_dipole_spectral_leakage": diagnostics.mnp_dipole_spectral_leakage,
                "mnp_drive_spectral_leakage": diagnostics.mnp_drive_spectral_leakage,
                "mnp_field_spectral_leakage": diagnostics.mnp_field_spectral_leakage,
                "min_density_eigenvalue": diagnostics.min_density_eigenvalue,
                "max_bloch_radius": diagnostics.max_bloch_radius,
                "excited_population_min": diagnostics.excited_population_min,
                "boundary_envelope_fraction": diagnostics.boundary_envelope_fraction,
                "solver_max_step_limit_au": diagnostics.max_step_limit_au,
                "integration_frequency_ceiling_au": (
                    diagnostics.integration_frequency_ceiling_au
                ),
                "incident_peak_rabi_frequency_au": (
                    diagnostics.incident_peak_rabi_frequency_au
                ),
                "observed_peak_rabi_frequency_au": (
                    diagnostics.observed_peak_rabi_frequency_au
                ),
                "work_passivity_tolerance_au": (
                    diagnostics.work_passivity_tolerance_au
                ),
                "post_fs_effective": effective_post_fs,
            }
            for name, value in values.items():
                float_arrays[name][channel_index, fluence_index] = float(value)
            bool_arrays["solver_success"][channel_index, fluence_index] = bool(
                diagnostics.solver_success
            )
            bool_arrays["t_final_reached"][channel_index, fluence_index] = bool(
                diagnostics.t_final_reached
            )
            bool_arrays["state_is_finite"][channel_index, fluence_index] = bool(
                diagnostics.state_is_finite
            )
            bool_arrays["response_tail_converged"][channel_index, fluence_index] = bool(
                diagnostics.response_tail_converged
            )
            bool_arrays["observable_window_converged"][
                channel_index, fluence_index
            ] = window_audit.accepted
            bool_arrays["bare_mnp_energy_integration_converged"][
                channel_index, fluence_index
            ] = bare_pulse_work.integration_converged
            bool_arrays["work_nonnegative_within_tolerance"][
                channel_index, fluence_index
            ] = bool(diagnostics.work_nonnegative_within_tolerance)
            bool_arrays["bare_mnp_work_nonnegative_within_tolerance"][
                channel_index, fluence_index
            ] = bare_pulse_work.work_nonnegative_within_tolerance
            int_arrays["solver_status"][channel_index, fluence_index] = int(
                diagnostics.solver_status
            )
            int_arrays["solver_n_steps"][channel_index, fluence_index] = int(
                diagnostics.n_steps
            )
            int_arrays["solver_nfev"][channel_index, fluence_index] = int(
                diagnostics.nfev
            )
            int_arrays["tail_extension_count"][channel_index, fluence_index] = int(
                extension_count
            )
            int_arrays["rabi_step_refinement_count"][
                channel_index, fluence_index
            ] = int(diagnostics.rabi_step_refinement_count)

    grid_observable_values = np.stack(
        [float_arrays[name] for name in GRID_AUDIT_OBSERVABLES], axis=1
    )
    grid_diagnostics = fluence_grid_resolution_diagnostics(
        fluences,
        grid_observable_values,
        isolated_qd_pulse_area_rad,
        max_midpoint_normalized_error=(
            max_fluence_grid_midpoint_normalized_error
        ),
        max_isolated_pulse_area_step_rad=max_isolated_pulse_area_step_rad,
    )
    if not grid_diagnostics["accepted"]:
        errors = np.asarray(
            grid_diagnostics["midpoint_normalized_error"], dtype=float
        )
        accepted = np.asarray(
            grid_diagnostics["accepted_by_observable"], dtype=bool
        )
        failures = ", ".join(
            f"{channel_keys[channel_index]}/{GRID_AUDIT_OBSERVABLES[observable_index]}="
            f"{errors[channel_index, observable_index]:.4g}"
            for channel_index, observable_index in zip(*np.nonzero(~accepted))
        )
        _apply_policy(
            fluence_grid_convergence_policy,
            "The work-loss fluence grid is not publication-resolved in "
            "sqrt(fluence): midpoint normalized errors ["
            f"{failures}], allowed="
            f"{max_fluence_grid_midpoint_normalized_error:.4g}; maximum "
            "isolated-QD pulse-area step="
            f"{grid_diagnostics['maximum_isolated_pulse_area_step_rad']:.4g} rad, "
            f"allowed={max_isolated_pulse_area_step_rad:.4g} rad. Increase the "
            "number of fluence points or narrow the interval.",
        )

    first_params = bundles[0].params
    metadata = {
        "schema": WORK_LOSS_FLUENCE_SCHEMA,
        "schema_version": WORK_LOSS_FLUENCE_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": Path(__file__).resolve().relative_to(PROJECT_ROOT).as_posix(),
        "plotter": "article_observables/qd_mnp_plot_work_loss_fluence.py",
        "command_line": [str(value) for value in sys.argv],
        "git": _git_provenance(),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "core_schema_version": SCHEMA_VERSION,
            "source_file_sha256": _source_file_hashes(),
        },
        "model": {
            "profile": NATIVE_MODEL_PROFILE,
            "time_backend": "FullQSSpheroidPulseModel",
            "spatial_model": "local electrostatic prolate-spheroid field",
            "qd_model": "point electric dipole, semiclassical two-level system",
            "bare_mnp_pulse_backend": (
                "exact Gaussian spectral average of linear causal Lorentz/ADE fit"
            ),
            "material_interpolation": MATERIAL_INTERPOLATION,
            "material_high_frequency_epsilon": MATERIAL_HIGH_FREQUENCY_EPSILON,
        },
        "pulse_definition": {
            "real_field_formula": (
                "E_inc(t)=E0*exp[-t^2/(2*sigma_t^2)]*cos(omega_L*t)"
            ),
            "envelope_center_fs": 0.0,
            "carrier_phase_rad": 0.0,
            "time_origin": "Gaussian-envelope centre",
            "sigma_t_convention": (
                "sigma_t=pulse_tau for tau_kind=sigma; "
                "sigma_t=pulse_tau/(2*sqrt(ln(2))) for "
                "tau_kind=fwhm_intensity"
            ),
            "fluence_definition": (
                "integral n_m*epsilon_0*c*E_inc(t)^2 dt, "
                "n_m=sqrt(eps_m), including the exact finite-carrier correction"
            ),
        },
        "initial_conditions": {
            "initial_time": "t_start=-pre_sigma*sigma_t",
            "qd_bloch": {"W": -1.0, "Q": 0.0, "P": 0.0, "rho_ee": 0.0},
            "all_mnp_material_ADE_coordinates_q_k": 0.0,
            "all_mnp_material_ADE_velocities_dq_k_dt": 0.0,
            "external_work_accumulator_au": 0.0,
        },
        "inputs": {
            "channel_keys": list(channel_keys),
            "fluence_j_cm2": [float(value) for value in fluences],
            "calculation_preset": calculation_preset,
            "fluence_grid_scale": fluence_grid_scale,
            "carrier_energy_eV": float(carrier_energy_eV),
            "pulse_tau_fs": float(pulse_tau_fs),
            "pulse_tau_kind": pulse_tau_kind,
            "common_surface_gap_nm": float(gap_nm),
            "c_nm": float(c_nm),
            "a_nm": float(a_nm),
            "qd_radius_nm": float(qd_radius_nm),
            "eps_m": float(eps_m),
            "eps_qd": float(eps_qd),
            "d_debye_requested": None if d_debye is None else float(d_debye),
            "omega0_eV": float(omega0_eV),
            "gamma_population_meV_requested": (
                None if gamma_population_meV is None else float(gamma_population_meV)
            ),
            "gamma2_coherence_meV_requested": (
                None if gamma2_coherence_meV is None else float(gamma2_coherence_meV)
            ),
            "qd_dipole_convention": qd_dipole_convention,
            "spatial_order_max": int(spatial_order_max),
            "material_fit_modes": int(material_fit_modes),
            "fit_window_eV": [float(value) for value in fit_window_eV],
            "fit_seed": int(fit_seed),
            "alpha_objective_weight": float(alpha_objective_weight),
            "inv_alpha_objective_weight": float(inv_alpha_objective_weight),
            "max_bright_fit_normalized_rms": (
                None
                if max_bright_fit_normalized_rms is None
                else float(max_bright_fit_normalized_rms)
            ),
            "max_bright_fit_pointwise_relative_error": (
                None
                if max_bright_fit_pointwise_relative_error is None
                else float(max_bright_fit_pointwise_relative_error)
            ),
            "bright_fit_quality_policy": bright_fit_quality_policy,
        },
        "fluence_grid": {
            "observable_names": list(GRID_AUDIT_OBSERVABLES),
            "audited_coordinate": grid_diagnostics["audited_coordinate"],
            "coarsening_rule": (
                "remove odd-indexed fine-grid points and linearly interpolate "
                "them from adjacent retained points in sqrt(fluence)"
            ),
            "normalization": (
                "maximum absolute value of the same channel/observable curve"
            ),
            "max_midpoint_normalized_error": float(
                max_fluence_grid_midpoint_normalized_error
            ),
            "max_isolated_pulse_area_step_rad": float(
                max_isolated_pulse_area_step_rad
            ),
            "convergence_policy": fluence_grid_convergence_policy,
            "midpoint_normalized_error": [
                [None if not np.isfinite(value) else float(value) for value in row]
                for row in np.asarray(
                    grid_diagnostics["midpoint_normalized_error"], dtype=float
                )
            ],
            "maximum_isolated_pulse_area_step_rad": (
                None
                if not np.isfinite(
                    grid_diagnostics["maximum_isolated_pulse_area_step_rad"]
                )
                else float(
                    grid_diagnostics["maximum_isolated_pulse_area_step_rad"]
                )
            ),
            "accepted_by_channel": [
                bool(value)
                for value in np.asarray(
                    grid_diagnostics["accepted_by_channel"], dtype=bool
                )
            ],
            "accepted": bool(grid_diagnostics["accepted"]),
        },
        "solver": {
            "method": method,
            "rtol": float(rtol),
            "atol": float(atol),
            "points_per_fastest_cycle": float(points_per_fastest_cycle),
            "pre_sigma": float(pre_sigma),
            "post_fs_requested": None if post_fs is None else float(post_fs),
            "spectral_window_policy": spectral_window_policy,
            "max_spectral_leakage": float(max_spectral_leakage),
            "positivity_policy": positivity_policy,
            "positivity_tolerance": float(positivity_tolerance),
            "work_passivity_policy": work_passivity_policy,
            "tail_policy": tail_policy,
            "tail_ratio_tolerance": float(tail_ratio_tolerance),
            "tail_window_fraction": float(tail_window_fraction),
            "max_auto_tail_extensions": int(max_auto_tail_extensions),
            "observable_convergence_policy": observable_convergence_policy,
            "max_observable_window_relative_change": float(
                max_observable_window_relative_change
            ),
        },
        "quality_policies": {
            "radiative_consistency_policy": radiative_consistency_policy,
            "fit_quality_policy": fit_quality_policy,
            "max_modal_normalized_rms": float(max_modal_normalized_rms),
            "max_modal_relative_error": float(max_modal_relative_error),
            "modal_audit_points": int(modal_audit_points),
            "spatial_convergence_policy": spatial_convergence_policy,
            "spatial_convergence_rtol": float(spatial_convergence_rtol),
            "reduction_policy": reduction_policy,
            "reduction_fit_grid_points": int(reduction_fit_grid_points),
            "reduction_audit_grid_points": int(reduction_audit_grid_points),
            "reduction_rms_tolerance": float(reduction_rms_tolerance),
            "reduction_max_tolerance": float(reduction_max_tolerance),
            "reduction_max_nodes": (
                None if reduction_max_nodes is None else int(reduction_max_nodes)
            ),
            "reduction_reaudit_points": int(reduction_reaudit_points),
            "material_fit_alpha_objective_weight": float(
                bundles[0].bright_model.alpha_objective_weight
            ),
            "material_fit_inverse_alpha_objective_weight": float(
                bundles[0].bright_model.inv_alpha_objective_weight
            ),
            "material_fit_max_normalized_rms": (
                None
                if bundles[0].bright_model.max_fit_normalized_rms is None
                else float(bundles[0].bright_model.max_fit_normalized_rms)
            ),
            "material_fit_max_pointwise_relative_error": (
                None
                if bundles[0].bright_model.max_fit_pointwise_relative_error is None
                else float(bundles[0].bright_model.max_fit_pointwise_relative_error)
            ),
        },
        "physical_constants": {
            "atomic_unit_length_m": AU_LENGTH_M,
            "atomic_unit_time_s": AU_TIME_S,
            "atomic_unit_energy_j": AU_ENERGY_J,
            "atomic_unit_energy_eV": AU_ENERGY_EV,
            "atomic_unit_field_v_m": AU_FIELD_V_M,
            "atomic_unit_dipole_c_m": AU_DIPOLE_C_M,
            "debye_c_m": DEBYE_C_M,
            "vacuum_permittivity_f_m": epsilon_0,
            "speed_of_light_m_s": C_SI,
        },
        "channels": channel_metadata,
        "observables": {
            "sigma_spectral_qs_work_loss_cm2": (
                "k(E_L) Im(alpha_eff(E_L; F))/epsilon_0 from the undressed "
                "electrostatic carrier-frequency response"
            ),
            "delta_sigma_spectral_qs_work_loss_cm2": (
                "hybrid sigma_spectral_qs_work_loss_cm2 minus the bare-MNP "
                "reference of the same long/trans orientation"
            ),
            "sigma_energy_transfer_cm2": (
                "integral E_inc(t) d(mu_total)/dt dt divided by exact incident fluence"
            ),
            "sigma_bare_mnp_energy_transfer_cm2": (
                "the same pulse-integrated work definition for the isolated MNP, "
                "computed from the active causal Lorentz/ADE material fit"
            ),
            "delta_sigma_energy_transfer_cm2": (
                "hybrid sigma_energy_transfer_cm2 minus the pulse-integrated "
                "bare-MNP reference of the same long/trans orientation"
            ),
            "delta_interpretation_limit": (
                "hybrid-minus-bare-MNP work contains both direct work delivered "
                "to the QD and the coupling-induced change of MNP work, including "
                "back-action and interference; it is neither QD-only absorption "
                "nor metal-only heating"
            ),
            "fluence_grid_convergence": (
                "nested every-other-point coarsening in sqrt(fluence), combined "
                "with a maximum isolated-QD resonant pulse-area step"
            ),
            "window_convergence_audit": (
                "relative change of each published work observable when the same "
                "trajectory is truncated at half of its final positive time"
            ),
            "interpretation_limit": (
                "QS work loss is not a separately evaluated metal-heating cross "
                "section; the one-frequency estimate is not total nonlinear radiation"
            ),
        },
        "array_units": {
            "requested_fluence_j_cm2": "J cm^-2",
            "fluence_j_cm2": "J cm^-2",
            "common_surface_gap_nm": "nm",
            "resolved_R_nm": "nm",
            "directional_mnp_radius_nm": "nm",
            "carrier_energy_eV": "eV",
            "pulse_tau_fs": "fs",
            "pulse_E0_au": "atomic field",
            "pulse_E0_v_m": "V m^-1",
            "peak_intensity_w_cm2": "W cm^-2",
            "isolated_qd_pulse_area_rad": "rad",
            "fluence_grid_observable_name": "string identifier",
            "fluence_grid_midpoint_absolute_error_cm2": "cm^2",
            "fluence_grid_curve_scale_cm2": "cm^2",
            "fluence_grid_midpoint_normalized_error": "dimensionless",
            "fluence_grid_converged_by_observable": "boolean",
            "fluence_grid_converged_by_channel": "boolean",
            "maximum_isolated_pulse_area_step_rad": "rad",
            "material_energy_eV": "eV",
            "material_n": "dimensionless",
            "material_k": "dimensionless",
            "material_fit_alpha_inf_au3": "atomic polarizability",
            "material_fit_strengths_au2": "atomic frequency^2 times polarizability",
            "material_fit_omega_modes_au": "atomic angular frequency",
            "material_fit_omega_modes_eV": "eV",
            "material_fit_gamma_modes_au": "atomic angular frequency",
            "material_fit_gamma_modes_eV": "eV",
            "material_fit_energies_used_eV": "eV",
            "material_fit_alpha_used_real_au3": "atomic polarizability",
            "material_fit_alpha_used_imag_au3": "atomic polarizability",
            "alpha_eff_real_au3": "atomic polarizability",
            "alpha_eff_imag_au3": "atomic polarizability",
            "sigma_spectral_qs_work_loss_cm2": "cm^2",
            "sigma_bare_mnp_qs_work_loss_cm2": "cm^2",
            "delta_sigma_spectral_qs_work_loss_cm2": "cm^2",
            "sigma_energy_transfer_cm2": "cm^2",
            "sigma_bare_mnp_energy_transfer_cm2": "cm^2",
            "delta_sigma_energy_transfer_cm2": "cm^2",
            "sigma_spectral_half_window_cm2": "cm^2",
            "sigma_energy_half_window_cm2": "cm^2",
            "sigma_bare_mnp_energy_cutoff_check_cm2": "cm^2",
            "sigma_spectral_half_window_relative_change": "dimensionless",
            "sigma_energy_half_window_relative_change": "dimensionless",
            "sigma_bare_mnp_energy_cutoff_relative_change": "dimensionless",
            "sigma_bare_mnp_energy_quadrature_relative_error": "dimensionless",
            "bare_mnp_energy_transfer_by_channel_cm2": "cm^2",
            "bare_mnp_energy_cutoff_check_by_channel_cm2": "cm^2",
            "bare_mnp_energy_cutoff_relative_change_by_channel": "dimensionless",
            "bare_mnp_energy_quadrature_absolute_error_by_channel_cm2": "cm^2",
            "bare_mnp_energy_quadrature_relative_error_by_channel": "dimensionless",
            "bare_mnp_work_from_incident_field_by_channel_j": "J",
            "bare_mnp_work_cutoff_check_by_channel_j": "J",
            "bare_mnp_work_passivity_tolerance_by_channel_j": "J",
            "bare_mnp_reference_fluence_by_channel_j_cm2": "J cm^-2",
            "bare_mnp_dimensionless_carrier_frequency_by_channel": "dimensionless",
            "bare_mnp_dimensionless_cutoff_check_by_channel": "dimensionless",
            "bare_mnp_dimensionless_cutoff_full_by_channel": "dimensionless",
            "work_from_incident_field_j": "J",
            "carrier_A_real_au3": "atomic polarizability",
            "carrier_A_imag_au3": "atomic polarizability",
            "carrier_B_real": "dimensionless",
            "carrier_B_imag": "dimensionless",
            "carrier_K_real_au_minus3": "atomic length^-3",
            "carrier_K_imag_au_minus3": "atomic length^-3",
            "excited_population_final": "dimensionless",
            "excited_population_max": "dimensionless",
            "excited_population_min": "dimensionless",
            "response_tail_ratio": "dimensionless",
            "observable_window_converged": "boolean",
            "bare_mnp_energy_integration_converged": "boolean",
            "pulse_spectral_leakage": "dimensionless fraction",
            "qd_source_spectral_leakage": "dimensionless fraction",
            "mnp_drive_spectral_leakage": "dimensionless fraction",
            "mnp_dipole_spectral_leakage": "dimensionless fraction",
            "mnp_field_spectral_leakage": "dimensionless fraction",
            "min_density_eigenvalue": "dimensionless",
            "max_bloch_radius": "dimensionless",
            "boundary_envelope_fraction": "dimensionless",
            "solver_success": "boolean",
            "solver_status": "integer status code",
            "t_final_reached": "boolean",
            "state_is_finite": "boolean",
            "solver_max_step_limit_au": "atomic time",
            "integration_frequency_ceiling_au": "atomic angular frequency",
            "incident_peak_rabi_frequency_au": "atomic angular frequency",
            "observed_peak_rabi_frequency_au": "atomic angular frequency",
            "work_passivity_tolerance_au": "atomic energy",
            "post_fs_effective": "fs",
            "solver_n_steps": "count",
            "solver_nfev": "count",
            "tail_extension_count": "count",
            "rabi_step_refinement_count": "count",
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.stem}_", suffix=".npz", dir=output.parent
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(
            temporary,
            metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
            channel_key=np.asarray(channel_keys),
            channel_label=np.asarray(
                [ARTICLE_CHANNELS[key].label for key in channel_keys]
            ),
            orientation=np.asarray(
                [ARTICLE_CHANNELS[key].orientation for key in channel_keys]
            ),
            qd_placement=np.asarray(
                [ARTICLE_CHANNELS[key].qd_placement for key in channel_keys]
            ),
            side_transverse_alignment=np.asarray(
                [
                    ARTICLE_CHANNELS[key].side_transverse_alignment or ""
                    for key in channel_keys
                ]
            ),
            requested_fluence_j_cm2=fluences,
            common_surface_gap_nm=np.asarray(gap_nm),
            resolved_R_nm=np.asarray([bundle.resolved_R_nm for bundle in bundles]),
            directional_mnp_radius_nm=np.asarray(
                [
                    c_nm if bundle.spec.qd_placement == "axis" else a_nm
                    for bundle in bundles
                ]
            ),
            carrier_energy_eV=np.asarray(carrier_energy_eV),
            pulse_tau_fs=np.asarray(pulse_tau_fs),
            fit_window_eV=np.asarray(fit_window_eV),
            material_energy_eV=np.asarray(first_params.material.energy_eV),
            material_n=np.asarray(first_params.material.n),
            material_k=np.asarray(first_params.material.k),
            material_fit_alpha_inf_au3=fit_alpha_inf,
            material_fit_strengths_au2=fit_strengths,
            material_fit_omega_modes_au=fit_omega,
            material_fit_omega_modes_eV=au_to_eV(fit_omega),
            material_fit_gamma_modes_au=fit_gamma,
            material_fit_gamma_modes_eV=au_to_eV(fit_gamma),
            material_fit_energies_used_eV=np.stack(fit_energies_used),
            material_fit_alpha_used_real_au3=np.stack(fit_alpha_used).real,
            material_fit_alpha_used_imag_au3=np.stack(fit_alpha_used).imag,
            bare_mnp_qs_work_loss_by_channel_cm2=bare_work_by_channel,
            bare_mnp_energy_transfer_by_channel_cm2=np.asarray(
                [value.sigma_energy_transfer_cm2 for value in bare_pulse_work_by_channel]
            ),
            bare_mnp_energy_cutoff_check_by_channel_cm2=np.asarray(
                [
                    value.sigma_energy_cutoff_check_cm2
                    for value in bare_pulse_work_by_channel
                ]
            ),
            bare_mnp_energy_cutoff_relative_change_by_channel=np.asarray(
                [value.cutoff_relative_change for value in bare_pulse_work_by_channel]
            ),
            bare_mnp_energy_quadrature_absolute_error_by_channel_cm2=np.asarray(
                [
                    value.quadrature_absolute_error_cm2
                    for value in bare_pulse_work_by_channel
                ]
            ),
            bare_mnp_energy_quadrature_relative_error_by_channel=np.asarray(
                [value.quadrature_relative_error for value in bare_pulse_work_by_channel]
            ),
            bare_mnp_energy_integration_converged_by_channel=np.asarray(
                [value.integration_converged for value in bare_pulse_work_by_channel],
                dtype=bool,
            ),
            bare_mnp_work_from_incident_field_by_channel_j=np.asarray(
                [value.work_from_incident_field_j for value in bare_pulse_work_by_channel]
            ),
            bare_mnp_work_cutoff_check_by_channel_j=np.asarray(
                [value.work_cutoff_check_j for value in bare_pulse_work_by_channel]
            ),
            bare_mnp_work_passivity_tolerance_by_channel_j=np.asarray(
                [value.work_passivity_tolerance_j for value in bare_pulse_work_by_channel]
            ),
            bare_mnp_work_nonnegative_by_channel=np.asarray(
                [
                    value.work_nonnegative_within_tolerance
                    for value in bare_pulse_work_by_channel
                ],
                dtype=bool,
            ),
            bare_mnp_reference_fluence_by_channel_j_cm2=np.asarray(
                [value.reference_fluence_j_cm2 for value in bare_pulse_work_by_channel]
            ),
            bare_mnp_dimensionless_carrier_frequency_by_channel=np.asarray(
                [
                    value.dimensionless_carrier_frequency
                    for value in bare_pulse_work_by_channel
                ]
            ),
            bare_mnp_dimensionless_cutoff_check_by_channel=np.asarray(
                [value.dimensionless_cutoff_check for value in bare_pulse_work_by_channel]
            ),
            bare_mnp_dimensionless_cutoff_full_by_channel=np.asarray(
                [value.dimensionless_cutoff_full for value in bare_pulse_work_by_channel]
            ),
            carrier_A_real_au3=carrier_A.real,
            carrier_A_imag_au3=carrier_A.imag,
            carrier_B_real=carrier_B.real,
            carrier_B_imag=carrier_B.imag,
            carrier_K_real_au_minus3=carrier_K.real,
            carrier_K_imag_au_minus3=carrier_K.imag,
            isolated_qd_pulse_area_rad=isolated_qd_pulse_area_rad,
            fluence_grid_observable_name=np.asarray(GRID_AUDIT_OBSERVABLES),
            fluence_grid_midpoint_absolute_error_cm2=np.asarray(
                grid_diagnostics["midpoint_absolute_error_cm2"], dtype=float
            ),
            fluence_grid_curve_scale_cm2=np.asarray(
                grid_diagnostics["curve_scale_cm2"], dtype=float
            ),
            fluence_grid_midpoint_normalized_error=np.asarray(
                grid_diagnostics["midpoint_normalized_error"], dtype=float
            ),
            fluence_grid_converged_by_observable=np.asarray(
                grid_diagnostics["accepted_by_observable"], dtype=bool
            ),
            fluence_grid_converged_by_channel=np.asarray(
                grid_diagnostics["accepted_by_channel"], dtype=bool
            ),
            maximum_isolated_pulse_area_step_rad=np.asarray(
                grid_diagnostics["maximum_isolated_pulse_area_step_rad"],
                dtype=float,
            ),
            **float_arrays,
            **bool_arrays,
            **int_arrays,
        )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    if verbose:
        print(f"Saved full-QS work-loss artifact: {output.resolve()}")
    return output


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate full-QS sigma_QS,work(E*; fluence) and save a self-describing NPZ."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/work_loss_fluence/work_loss_fluence.npz"),
    )
    parser.add_argument(
        "--preset",
        choices=("publication", "quick"),
        default="publication",
        help=(
            "Publication uses 65 points uniform in sqrt(fluence) and strict "
            "grid certification; quick is a coarse diagnostic preview."
        ),
    )
    parser.add_argument(
        "--channel",
        dest="channels",
        action="append",
        choices=tuple(ARTICLE_CHANNELS),
        help=(
            "Article channel; repeat as needed. Default: axis_long and side_long."
        ),
    )
    parser.add_argument("--fluence-j-cm2", type=float, nargs="+")
    parser.add_argument("--fluence-min-j-cm2", type=float)
    parser.add_argument("--fluence-max-j-cm2", type=float)
    parser.add_argument("--fluence-points", type=int)
    parser.add_argument("--grid-scale", choices=("sqrt", "log", "linear"))
    parser.add_argument("--carrier-energy-ev", type=float, default=2.042)
    parser.add_argument("--pulse-tau-fs", type=float, default=20.0)
    parser.add_argument(
        "--pulse-tau-kind",
        choices=("fwhm_intensity", "sigma"),
        default="fwhm_intensity",
    )
    parser.add_argument("--gap-nm", type=float, default=1.0)
    parser.add_argument("--c-nm", type=float, default=15.0)
    parser.add_argument("--a-nm", type=float, default=7.0)
    parser.add_argument("--qd-radius-nm", type=float, default=2.0)
    parser.add_argument("--eps-m", type=float, default=1.0)
    parser.add_argument("--eps-qd", type=float, default=6.0)
    parser.add_argument("--d-debye", type=float)
    parser.add_argument("--omega0-ev", type=float, default=2.042)
    parser.add_argument("--gamma-population-mev", type=float)
    parser.add_argument("--gamma2-coherence-mev", type=float)
    parser.add_argument(
        "--qd-dipole-convention",
        choices=("effective_external", "bare_internal"),
        default="effective_external",
    )
    parser.add_argument("--spatial-order-max", type=int, default=80)
    parser.add_argument("--material-fit-modes", type=int, default=9)
    parser.add_argument("--fit-window-ev", nargs=2, type=float, default=(0.8, 3.0))
    parser.add_argument("--fit-seed", type=int, default=12345)
    parser.add_argument("--alpha-objective-weight", type=float, default=1.0)
    parser.add_argument("--inv-alpha-objective-weight", type=float, default=1.2)
    parser.add_argument("--max-bright-fit-normalized-rms", type=float, default=0.025)
    parser.add_argument(
        "--max-bright-fit-pointwise-relative-error", type=float, default=0.05
    )
    parser.add_argument(
        "--bright-fit-quality-policy", choices=POLICIES, default="raise"
    )
    parser.add_argument(
        "--radiative-consistency-policy", choices=POLICIES, default="warn"
    )
    parser.add_argument("--fit-quality-policy", choices=POLICIES, default="raise")
    parser.add_argument("--max-modal-normalized-rms", type=float, default=0.03)
    parser.add_argument("--max-modal-relative-error", type=float, default=0.06)
    parser.add_argument("--modal-audit-points", type=int, default=2001)
    parser.add_argument(
        "--spatial-convergence-policy", choices=POLICIES, default="raise"
    )
    parser.add_argument("--spatial-convergence-rtol", type=float, default=2.0e-5)
    parser.add_argument("--reduction-fit-grid-points", type=int, default=1001)
    parser.add_argument("--reduction-audit-grid-points", type=int, default=1601)
    parser.add_argument("--reduction-rms-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--reduction-max-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--reduction-max-nodes", type=int)
    parser.add_argument("--reduction-policy", choices=POLICIES, default="raise")
    parser.add_argument("--reduction-reaudit-points", type=int, default=1709)
    parser.add_argument(
        "--method",
        choices=("BDF", "DOP853", "LSODA", "RK45", "Radau"),
        default="DOP853",
    )
    parser.add_argument("--rtol", type=float, default=1.0e-8)
    parser.add_argument("--atol", type=float, default=1.0e-10)
    parser.add_argument("--points-per-fastest-cycle", type=float, default=20.0)
    parser.add_argument("--pre-sigma", type=float, default=10.0)
    parser.add_argument("--post-fs", type=float)
    parser.add_argument("--spectral-window-policy", choices=POLICIES, default="raise")
    parser.add_argument("--max-spectral-leakage", type=float, default=1.0e-3)
    parser.add_argument("--positivity-policy", choices=POLICIES, default="raise")
    parser.add_argument("--positivity-tolerance", type=float, default=1.0e-7)
    parser.add_argument("--work-passivity-policy", choices=POLICIES, default="raise")
    parser.add_argument("--tail-policy", choices=POLICIES, default="raise")
    parser.add_argument("--tail-ratio-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--tail-window-fraction", type=float, default=0.05)
    parser.add_argument("--max-auto-tail-extensions", type=int, default=3)
    parser.add_argument(
        "--observable-convergence-policy", choices=POLICIES, default="raise"
    )
    parser.add_argument(
        "--max-observable-window-relative-change", type=float, default=1.0e-3
    )
    parser.add_argument(
        "--fluence-grid-convergence-policy", choices=POLICIES
    )
    parser.add_argument(
        "--max-fluence-grid-midpoint-normalized-error", type=float
    )
    parser.add_argument("--max-isolated-pulse-area-step-rad", type=float)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = _parse_args(argv)
    grid_preset = PUBLICATION_GRID if args.preset == "publication" else QUICK_GRID
    grid_policy = (
        grid_preset["convergence_policy"]
        if args.fluence_grid_convergence_policy is None
        else args.fluence_grid_convergence_policy
    )
    midpoint_limit = (
        grid_preset["max_midpoint_normalized_error"]
        if args.max_fluence_grid_midpoint_normalized_error is None
        else args.max_fluence_grid_midpoint_normalized_error
    )
    pulse_area_step_limit = (
        grid_preset["max_isolated_pulse_area_step_rad"]
        if args.max_isolated_pulse_area_step_rad is None
        else args.max_isolated_pulse_area_step_rad
    )
    if args.fluence_j_cm2 is None:
        fluence_min = (
            grid_preset["fluence_min_j_cm2"]
            if args.fluence_min_j_cm2 is None
            else args.fluence_min_j_cm2
        )
        fluence_max = (
            grid_preset["fluence_max_j_cm2"]
            if args.fluence_max_j_cm2 is None
            else args.fluence_max_j_cm2
        )
        fluence_points = (
            int(grid_preset["points"])
            if args.fluence_points is None
            else args.fluence_points
        )
        grid_scale = grid_preset["scale"] if args.grid_scale is None else args.grid_scale
        if (
            not np.isfinite(fluence_min)
            or not np.isfinite(fluence_max)
            or fluence_min <= 0.0
            or fluence_max <= fluence_min
            or fluence_points < 2
        ):
            raise ValueError(
                "Fluence grid requires 0 < min < max and at least two points."
            )
        if grid_scale == "sqrt":
            fluences = np.linspace(
                np.sqrt(fluence_min), np.sqrt(fluence_max), fluence_points
            ) ** 2
        elif grid_scale == "log":
            fluences = np.geomspace(fluence_min, fluence_max, fluence_points)
        else:
            fluences = np.linspace(fluence_min, fluence_max, fluence_points)
    else:
        fluences = np.asarray(sorted(args.fluence_j_cm2), dtype=float)
        grid_scale = "custom" if args.grid_scale is None else args.grid_scale
    return calculate_work_loss_fluence(
        args.output,
        channel_keys=tuple(args.channels or DEFAULT_CHANNELS),
        fluence_j_cm2=fluences,
        fluence_grid_scale=grid_scale,
        calculation_preset=args.preset,
        carrier_energy_eV=args.carrier_energy_ev,
        pulse_tau_fs=args.pulse_tau_fs,
        pulse_tau_kind=args.pulse_tau_kind,
        gap_nm=args.gap_nm,
        c_nm=args.c_nm,
        a_nm=args.a_nm,
        qd_radius_nm=args.qd_radius_nm,
        eps_m=args.eps_m,
        eps_qd=args.eps_qd,
        d_debye=args.d_debye,
        omega0_eV=args.omega0_ev,
        gamma_population_meV=args.gamma_population_mev,
        gamma2_coherence_meV=args.gamma2_coherence_mev,
        qd_dipole_convention=args.qd_dipole_convention,
        spatial_order_max=args.spatial_order_max,
        material_fit_modes=args.material_fit_modes,
        fit_window_eV=tuple(args.fit_window_ev),
        fit_seed=args.fit_seed,
        alpha_objective_weight=args.alpha_objective_weight,
        inv_alpha_objective_weight=args.inv_alpha_objective_weight,
        max_bright_fit_normalized_rms=args.max_bright_fit_normalized_rms,
        max_bright_fit_pointwise_relative_error=(
            args.max_bright_fit_pointwise_relative_error
        ),
        bright_fit_quality_policy=args.bright_fit_quality_policy,
        radiative_consistency_policy=args.radiative_consistency_policy,
        fit_quality_policy=args.fit_quality_policy,
        max_modal_normalized_rms=args.max_modal_normalized_rms,
        max_modal_relative_error=args.max_modal_relative_error,
        modal_audit_points=args.modal_audit_points,
        spatial_convergence_policy=args.spatial_convergence_policy,
        spatial_convergence_rtol=args.spatial_convergence_rtol,
        reduction_fit_grid_points=args.reduction_fit_grid_points,
        reduction_audit_grid_points=args.reduction_audit_grid_points,
        reduction_rms_tolerance=args.reduction_rms_tolerance,
        reduction_max_tolerance=args.reduction_max_tolerance,
        reduction_max_nodes=args.reduction_max_nodes,
        reduction_policy=args.reduction_policy,
        reduction_reaudit_points=args.reduction_reaudit_points,
        method=args.method,
        rtol=args.rtol,
        atol=args.atol,
        points_per_fastest_cycle=args.points_per_fastest_cycle,
        pre_sigma=args.pre_sigma,
        post_fs=args.post_fs,
        spectral_window_policy=args.spectral_window_policy,
        max_spectral_leakage=args.max_spectral_leakage,
        positivity_policy=args.positivity_policy,
        positivity_tolerance=args.positivity_tolerance,
        work_passivity_policy=args.work_passivity_policy,
        tail_policy=args.tail_policy,
        tail_ratio_tolerance=args.tail_ratio_tolerance,
        tail_window_fraction=args.tail_window_fraction,
        max_auto_tail_extensions=args.max_auto_tail_extensions,
        observable_convergence_policy=args.observable_convergence_policy,
        max_observable_window_relative_change=(
            args.max_observable_window_relative_change
        ),
        fluence_grid_convergence_policy=grid_policy,
        max_fluence_grid_midpoint_normalized_error=midpoint_limit,
        max_isolated_pulse_area_step_rad=pulse_area_step_limit,
        overwrite=args.overwrite,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
