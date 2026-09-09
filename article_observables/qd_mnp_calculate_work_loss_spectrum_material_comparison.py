"""Calculate Shah-type pulse work-loss spectra for one and many Au poles.

One time-domain FQS propagation is performed for every combination of material
branch, QD/MNP channel and selected incident fluence.  The complete spectrum

    sigma_QS,work(E; F) = k(E) Im[alpha_eff(E; F)] / epsilon_0

is then recovered from the Fourier ratio of the saved total dipole and incident
field over the supported bandwidth of the *same* broadband pulse.  The script
never scans the carrier energy by repeatedly solving the ODE.  Its companion
plotter reads only the self-contained NPZ artifact produced here.

The reported quantity is an undressed local-quasistatic work-loss estimate,
analogous to the quantity called ``sigma_abs`` by Shah et al.; it is not a
separately evaluated metal-heating cross section.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import platform
import sys
from typing import Any
import warnings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import scipy

from article_observables.qd_mnp_calculate_work_loss_fluence import (
    ARTICLE_CHANNELS,
    _apply_policy,
    _bare_mnp_pulse_work_spectral_average,
    _build_channel_model,
    _channel_metadata,
    _prefix_with_endpoint,
    pulse_for_fluence,
)
from article_observables.qd_mnp_material_modes_artifact import (
    SCHEMA_VERSION,
    atomic_write_npz,
    canonical_sha256,
    git_provenance,
    source_hashes,
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
    MATERIAL_HIGH_FREQUENCY_EPSILON,
    MATERIAL_INTERPOLATION,
    NATIVE_MODEL_PROFILE,
    au_to_eV,
    au_to_fs,
    eV_to_au,
    epsilon_0,
    field_au_to_si,
    fs_to_au,
    quasistatic_dipole_cross_section_estimates_cm2,
)


SCHEMA_NAME = "qd_mnp.material_work_loss_spectrum_comparison"
BRANCH_IDS = np.asarray(["one", "multi"], dtype="U16")
POLICIES = ("raise", "warn", "ignore")
DEFAULT_CHANNELS = ("axis_long",)

PUBLICATION_PRESET = {
    "energy_points": 1001,
    "spatial_order_max": 80,
    "modal_audit_points": 2001,
    "spatial_convergence_policy": "raise",
    "energy_grid_convergence_policy": "raise",
    "spectral_support_policy": "raise",
    "incident_ft_policy": "raise",
}

QUICK_PRESET = {
    "energy_points": 121,
    "spatial_order_max": 4,
    "modal_audit_points": 301,
    "spatial_convergence_policy": "warn",
    "energy_grid_convergence_policy": "warn",
    "spectral_support_policy": "warn",
    "incident_ft_policy": "warn",
}

ENERGY_GRID_OBSERVABLES = (
    "sigma_qs_work_cm2",
    "delta_sigma_qs_work_cm2",
)


@dataclass(frozen=True)
class SpectrumWindowAudit:
    """Fourier spectrum and convergence checks for one propagated pulse."""

    alpha_eff_au3: np.ndarray
    total_dipole_ft_au: np.ndarray
    incident_field_ft_numeric_au: np.ndarray
    incident_field_ft_analytic_au: np.ndarray
    incident_field_relative_amplitude: np.ndarray
    support_mask: np.ndarray
    sigma_qs_work_cm2: np.ndarray
    sigma_half_window_cm2: np.ndarray
    spectrum_window_max_normalized_change: float
    sigma_energy_half_window_cm2: float
    energy_window_relative_change: float
    incident_ft_max_normalized_error: float
    incident_ft_max_pointwise_relative_error: float
    incident_ft_converged: bool
    spectrum_window_converged: bool


def analytic_incident_field_ft_au(
    pulse: Any,
    energies_eV: np.ndarray,
) -> np.ndarray:
    """Infinite-window +frequency transform of the real Gaussian carrier."""

    energy = np.asarray(energies_eV, dtype=float)
    if energy.ndim != 1 or energy.size == 0 or np.any(~np.isfinite(energy)):
        raise ValueError("energies_eV must be a finite non-empty one-dimensional grid.")
    omega = np.asarray(eV_to_au(energy), dtype=float)
    sigma = float(pulse.sigma_t_au)
    omega_l = float(pulse.omegaL_au)
    prefactor = 0.5 * float(pulse.E0_au) * sigma * np.sqrt(2.0 * np.pi)
    return prefactor * (
        np.exp(-0.5 * sigma**2 * (omega - omega_l) ** 2)
        + np.exp(-0.5 * sigma**2 * (omega + omega_l) ** 2)
    )


def _fourier_integral_grid(
    time_au: np.ndarray,
    signal: np.ndarray,
    energies_eV: np.ndarray,
    *,
    chunk_size: int = 32,
) -> np.ndarray:
    """Return integral signal(t) exp(+i omega t) dt on a bounded grid."""

    time = np.asarray(time_au, dtype=float)
    values = np.asarray(signal, dtype=float)
    energy = np.asarray(energies_eV, dtype=float)
    if (
        time.ndim != 1
        or values.shape != time.shape
        or time.size < 2
        or np.any(~np.isfinite(time))
        or np.any(~np.isfinite(values))
        or np.any(np.diff(time) <= 0.0)
    ):
        raise ValueError("Fourier input must be a finite signal on an increasing grid.")
    if energy.ndim != 1 or energy.size == 0 or np.any(~np.isfinite(energy)):
        raise ValueError("Fourier energies must be a finite one-dimensional grid.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")

    omega = np.asarray(eV_to_au(energy), dtype=float)
    transformed = np.empty(energy.shape, dtype=complex)
    for start in range(0, energy.size, chunk_size):
        stop = min(start + chunk_size, energy.size)
        phase = np.exp(1j * omega[start:stop, None] * time[None, :])
        transformed[start:stop] = np.trapezoid(
            phase * values[None, :],
            time,
            axis=1,
        )
    return transformed


def spectral_effective_alpha_grid(
    result: Any,
    pulse: Any,
    eps_m: float,
    energies_eV: np.ndarray,
    *,
    minimum_incident_relative_amplitude: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return alpha_eff and the field-support diagnostics on an energy grid."""

    if not np.isfinite(eps_m) or eps_m <= 0.0:
        raise ValueError("eps_m must be finite and positive.")
    if (
        not np.isfinite(minimum_incident_relative_amplitude)
        or not 0.0 < minimum_incident_relative_amplitude < 1.0
    ):
        raise ValueError("minimum_incident_relative_amplitude must lie in (0, 1).")
    energy = np.asarray(energies_eV, dtype=float)
    time = np.asarray(result.t_au, dtype=float)
    mu_total = np.asarray(result.mu_total_au, dtype=float)
    incident = np.asarray(pulse.field(time), dtype=float)
    mu_ft = _fourier_integral_grid(time, mu_total, energy)
    field_ft = _fourier_integral_grid(time, incident, energy)
    field_analytic = analytic_incident_field_ft_au(pulse, energy)
    carrier_analytic = analytic_incident_field_ft_au(
        pulse, np.asarray([au_to_eV(pulse.omegaL_au)], dtype=float)
    )[0]
    peak = max(abs(complex(carrier_analytic)), np.finfo(float).tiny)
    relative_amplitude = np.abs(field_analytic) / peak
    support = relative_amplitude >= minimum_incident_relative_amplitude
    if np.any(np.abs(field_ft) <= np.finfo(float).tiny * peak):
        raise FloatingPointError("A sampled incident-field Fourier component vanished.")
    alpha = mu_ft / (float(eps_m) * field_ft)
    if np.any(~np.isfinite(alpha)):
        raise FloatingPointError("The spectral effective polarizability is not finite.")
    return alpha, mu_ft, field_ft, field_analytic, relative_amplitude, support


def _relative_scalar_change(full_value: float, truncated_value: float) -> float:
    if not np.isfinite(full_value) or not np.isfinite(truncated_value):
        return float("nan")
    return float(
        abs(full_value - truncated_value)
        / max(abs(full_value), np.finfo(float).tiny)
    )


def _spectrum_window_audit(
    result: Any,
    pulse: Any,
    eps_m: float,
    energies_eV: np.ndarray,
    *,
    minimum_incident_relative_amplitude: float,
    max_incident_ft_pointwise_relative_error: float,
    max_spectrum_window_relative_change: float,
    max_energy_window_relative_change: float,
) -> SpectrumWindowAudit:
    """Compare the full and half-post-window Fourier observables."""

    (
        alpha,
        mu_ft,
        field_ft,
        field_analytic,
        relative_amplitude,
        support,
    ) = spectral_effective_alpha_grid(
        result,
        pulse,
        eps_m,
        energies_eV,
        minimum_incident_relative_amplitude=minimum_incident_relative_amplitude,
    )
    omega = np.asarray(eV_to_au(energies_eV), dtype=float)
    sigma = np.asarray(
        quasistatic_dipole_cross_section_estimates_cm2(alpha, omega, eps_m)
        .quasistatic_work_loss_cm2,
        dtype=float,
    )

    final_time = float(np.asarray(result.t_au, dtype=float)[-1])
    cutoff = 0.5 * final_time
    sigma_half = np.full_like(sigma, np.nan)
    sigma_energy_half = float("nan")
    if cutoff > 8.0 * pulse.sigma_t_au:
        time_half, mu_half = _prefix_with_endpoint(
            result.t_au, result.mu_total_au, cutoff
        )
        _, mu_dot_half = _prefix_with_endpoint(
            result.t_au, result.mu_dot_total_au, cutoff
        )
        half_result = argparse.Namespace(t_au=time_half, mu_total_au=mu_half)
        alpha_half, *_ = spectral_effective_alpha_grid(
            half_result,
            pulse,
            eps_m,
            energies_eV,
            minimum_incident_relative_amplitude=minimum_incident_relative_amplitude,
        )
        sigma_half = np.asarray(
            quasistatic_dipole_cross_section_estimates_cm2(
                alpha_half, omega, eps_m
            ).quasistatic_work_loss_cm2,
            dtype=float,
        )
        incident_half = np.asarray(pulse.field(time_half), dtype=float)
        work_half_au = float(np.trapezoid(incident_half * mu_dot_half, time_half))
        sigma_energy_half = float(
            work_half_au
            * AU_ENERGY_J
            / pulse.fluence_j_cm2(eps_m=eps_m)
        )

    supported = np.asarray(support, dtype=bool)
    if np.any(supported) and np.all(np.isfinite(sigma_half[supported])):
        curve_scale = max(
            float(np.max(np.abs(sigma[supported]))), np.finfo(float).tiny
        )
        spectrum_change = float(
            np.max(np.abs(sigma[supported] - sigma_half[supported])) / curve_scale
        )
    else:
        spectrum_change = float("nan")
    energy_change = _relative_scalar_change(
        float(result.sigma_energy_transfer_cm2), sigma_energy_half
    )

    carrier_analytic = analytic_incident_field_ft_au(
        pulse, np.asarray([au_to_eV(pulse.omegaL_au)], dtype=float)
    )[0]
    peak = max(abs(complex(carrier_analytic)), np.finfo(float).tiny)
    normalized_ft_error = float(np.max(np.abs(field_ft - field_analytic)) / peak)
    if np.any(supported):
        pointwise_ft_error = float(
            np.max(
                np.abs(field_ft[supported] - field_analytic[supported])
                / np.maximum(np.abs(field_analytic[supported]), np.finfo(float).tiny)
            )
        )
    else:
        pointwise_ft_error = float("inf")
    incident_converged = bool(
        np.any(supported)
        and np.isfinite(pointwise_ft_error)
        and pointwise_ft_error <= max_incident_ft_pointwise_relative_error
    )
    spectrum_converged = bool(
        np.isfinite(spectrum_change)
        and np.isfinite(energy_change)
        and spectrum_change <= max_spectrum_window_relative_change
        and energy_change <= max_energy_window_relative_change
    )
    return SpectrumWindowAudit(
        alpha_eff_au3=np.asarray(alpha, dtype=complex),
        total_dipole_ft_au=np.asarray(mu_ft, dtype=complex),
        incident_field_ft_numeric_au=np.asarray(field_ft, dtype=complex),
        incident_field_ft_analytic_au=np.asarray(field_analytic, dtype=float),
        incident_field_relative_amplitude=np.asarray(relative_amplitude, dtype=float),
        support_mask=np.asarray(support, dtype=bool),
        sigma_qs_work_cm2=sigma,
        sigma_half_window_cm2=sigma_half,
        spectrum_window_max_normalized_change=spectrum_change,
        sigma_energy_half_window_cm2=sigma_energy_half,
        energy_window_relative_change=energy_change,
        incident_ft_max_normalized_error=normalized_ft_error,
        incident_ft_max_pointwise_relative_error=pointwise_ft_error,
        incident_ft_converged=incident_converged,
        spectrum_window_converged=spectrum_converged,
    )


def _solve_with_spectrum_tail_extension(
    model: Any,
    pulse: Any,
    energies_eV: np.ndarray,
    *,
    eps_m: float,
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
    observable_convergence_policy: str,
    incident_ft_policy: str,
    minimum_incident_relative_amplitude: float,
    max_incident_ft_pointwise_relative_error: float,
    max_spectrum_window_relative_change: float,
    max_energy_window_relative_change: float,
) -> tuple[Any, float, int, SpectrumWindowAudit]:
    """Solve until both the response tail and full spectrum are converged."""

    start_au = -float(pre_sigma) * pulse.sigma_t_au
    if post_fs is None:
        end_au = max(
            float(pre_sigma) * pulse.sigma_t_au,
            float(model.recommended_post_pulse_time_au()),
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
        audit = _spectrum_window_audit(
            result,
            pulse,
            eps_m,
            energies_eV,
            minimum_incident_relative_amplitude=minimum_incident_relative_amplitude,
            max_incident_ft_pointwise_relative_error=(
                max_incident_ft_pointwise_relative_error
            ),
            max_spectrum_window_relative_change=max_spectrum_window_relative_change,
            max_energy_window_relative_change=max_energy_window_relative_change,
        )
        tail_converged = bool(result.diagnostics.response_tail_converged)
        if tail_converged and audit.spectrum_window_converged and audit.incident_ft_converged:
            break
        if post_fs is not None or extensions >= max_auto_tail_extensions:
            if not tail_converged:
                _apply_policy(
                    tail_policy,
                    "The FQS response tail did not converge for a published spectrum: "
                    f"ratio={result.diagnostics.response_tail_ratio:.6g}, "
                    f"limit={tail_ratio_tolerance:.6g}.",
                )
            if not audit.spectrum_window_converged:
                _apply_policy(
                    observable_convergence_policy,
                    "The Shah-type spectrum and/or W_inc/F changed when the saved "
                    "trajectory was truncated at half of its final positive time: "
                    f"changes=({audit.spectrum_window_max_normalized_change:.6g}, "
                    f"{audit.energy_window_relative_change:.6g}).",
                )
            if not audit.incident_ft_converged:
                _apply_policy(
                    incident_ft_policy,
                    "The sampled incident-pulse transform is inaccurate or the requested "
                    "energy grid leaves the configured pulse support: max pointwise "
                    f"relative error={audit.incident_ft_max_pointwise_relative_error:.6g}.",
                )
            break
        end_au *= 2.0
        extensions += 1
    return result, float(au_to_fs(end_au)), extensions, audit


def energy_grid_resolution_diagnostics(
    energy_eV: np.ndarray,
    observable_values: np.ndarray,
    *,
    max_midpoint_normalized_error: float,
) -> dict[str, np.ndarray | bool]:
    """Audit every other spectral point against adjacent-point interpolation."""

    energy = np.asarray(energy_eV, dtype=float)
    values = np.asarray(observable_values, dtype=float)
    if energy.ndim != 1 or energy.size < 5 or values.shape[-1] != energy.size:
        raise ValueError("Energy diagnostics require at least five matching points.")
    if np.any(~np.isfinite(energy)) or np.any(np.diff(energy) <= 0.0):
        raise ValueError("The energy grid must be finite and strictly increasing.")
    if np.any(~np.isfinite(values)):
        raise ValueError("Spectral observables must be finite for grid diagnostics.")
    if (
        not np.isfinite(max_midpoint_normalized_error)
        or max_midpoint_normalized_error <= 0.0
    ):
        raise ValueError("The midpoint-error limit must be finite and positive.")

    midpoint_indices = np.arange(1, energy.size - 1, 2, dtype=int)
    left = midpoint_indices - 1
    right = midpoint_indices + 1
    weights = (energy[midpoint_indices] - energy[left]) / (
        energy[right] - energy[left]
    )
    interpolated = values[..., left] + weights * (
        values[..., right] - values[..., left]
    )
    absolute_error = np.max(
        np.abs(values[..., midpoint_indices] - interpolated), axis=-1
    )
    curve_scale = np.max(np.abs(values), axis=-1)
    normalized_error = absolute_error / np.maximum(
        curve_scale, np.finfo(float).tiny
    )
    accepted = normalized_error <= max_midpoint_normalized_error
    return {
        "midpoint_absolute_error": absolute_error,
        "curve_scale": curve_scale,
        "midpoint_normalized_error": normalized_error,
        "accepted": accepted,
        "all_accepted": bool(np.all(accepted)),
    }


def _apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    defaults = PUBLICATION_PRESET if args.preset == "publication" else QUICK_PRESET
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    return args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/article/work_loss_spectrum_material_comparison.npz"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preset", choices=("publication", "quick"), default="publication")
    parser.add_argument(
        "--channel",
        dest="channels",
        action="append",
        choices=tuple(ARTICLE_CHANNELS),
        help="Repeat for selected QD-position/electric-field channels; default axis_long.",
    )
    parser.add_argument(
        "--fluence-j-cm2",
        type=float,
        nargs="+",
        required=True,
        help="Selected fluences from the preceding P_exc(F) calculation.",
    )
    parser.add_argument(
        "--fluence-label",
        nargs="+",
        help="Optional labels matching --fluence-j-cm2, e.g. linear threshold nonlinear.",
    )
    parser.add_argument("--selection-source-artifact", type=Path)
    parser.add_argument("--carrier-energy-ev", type=float, default=2.042)
    parser.add_argument("--pulse-tau-fs", type=float, default=20.0)
    parser.add_argument(
        "--pulse-tau-kind",
        choices=("fwhm_intensity", "sigma"),
        default="fwhm_intensity",
    )
    parser.add_argument("--energy-min-ev", type=float, default=1.90)
    parser.add_argument("--energy-max-ev", type=float, default=2.18)
    parser.add_argument("--energy-points", type=int)

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

    parser.add_argument("--one-fit-modes", type=int, default=1)
    parser.add_argument("--multi-fit-modes", type=int, default=9)
    parser.add_argument("--fit-window-ev", nargs=2, type=float, default=(0.8, 3.0))
    parser.add_argument("--fit-seed", type=int, default=12345)
    parser.add_argument("--alpha-objective-weight", type=float, default=1.0)
    parser.add_argument("--inv-alpha-objective-weight", type=float, default=1.2)
    parser.add_argument("--max-bright-fit-normalized-rms", type=float, default=0.025)
    parser.add_argument(
        "--max-bright-fit-pointwise-relative-error", type=float, default=0.05
    )
    parser.add_argument(
        "--one-fit-accuracy-policy", choices=("warn", "ignore"), default="warn"
    )
    parser.add_argument("--radiative-consistency-policy", choices=POLICIES, default="warn")
    parser.add_argument("--fit-quality-policy", choices=POLICIES, default="raise")
    parser.add_argument("--max-modal-normalized-rms", type=float, default=0.03)
    parser.add_argument("--max-modal-relative-error", type=float, default=0.08)
    parser.add_argument("--modal-audit-points", type=int)
    parser.add_argument("--spatial-order-max", type=int)
    parser.add_argument("--spatial-convergence-policy", choices=POLICIES)
    parser.add_argument("--spatial-convergence-rtol", type=float, default=2.0e-5)
    parser.add_argument("--reduction-fit-grid-points", type=int, default=1001)
    parser.add_argument("--reduction-audit-grid-points", type=int, default=1601)
    parser.add_argument("--reduction-rms-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--reduction-max-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--reduction-max-nodes", type=int)
    parser.add_argument("--reduction-policy", choices=POLICIES, default="raise")
    parser.add_argument("--reduction-reaudit-points", type=int, default=1709)

    parser.add_argument(
        "--method", choices=("BDF", "DOP853", "LSODA", "RK45", "Radau"), default="DOP853"
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
        "--max-spectrum-window-relative-change", type=float, default=1.0e-3
    )
    parser.add_argument(
        "--max-energy-window-relative-change", type=float, default=1.0e-3
    )
    parser.add_argument("--spectral-support-policy", choices=POLICIES)
    parser.add_argument(
        "--minimum-incident-relative-amplitude", type=float, default=1.0e-2
    )
    parser.add_argument("--incident-ft-policy", choices=POLICIES)
    parser.add_argument(
        "--max-incident-ft-pointwise-relative-error", type=float, default=5.0e-3
    )
    parser.add_argument("--energy-grid-convergence-policy", choices=POLICIES)
    parser.add_argument(
        "--max-energy-grid-midpoint-normalized-error", type=float, default=2.0e-3
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.channels is None:
        args.channels = list(DEFAULT_CHANNELS)
    return _apply_preset(args)


def _validate_args(args: argparse.Namespace) -> None:
    positive = (
        "carrier_energy_ev",
        "pulse_tau_fs",
        "energy_min_ev",
        "energy_max_ev",
        "gap_nm",
        "c_nm",
        "a_nm",
        "qd_radius_nm",
        "eps_m",
        "eps_qd",
        "omega0_ev",
        "alpha_objective_weight",
        "inv_alpha_objective_weight",
        "max_bright_fit_normalized_rms",
        "max_bright_fit_pointwise_relative_error",
        "max_modal_normalized_rms",
        "max_modal_relative_error",
        "spatial_convergence_rtol",
        "reduction_rms_tolerance",
        "reduction_max_tolerance",
        "rtol",
        "atol",
        "points_per_fastest_cycle",
        "pre_sigma",
        "tail_ratio_tolerance",
        "tail_window_fraction",
        "max_spectrum_window_relative_change",
        "max_energy_window_relative_change",
        "minimum_incident_relative_amplitude",
        "max_incident_ft_pointwise_relative_error",
        "max_energy_grid_midpoint_normalized_error",
    )
    for name in positive:
        value = float(getattr(args, name))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive.")
    if not args.energy_min_ev < args.energy_max_ev:
        raise ValueError("Require energy_min_ev < energy_max_ev.")
    if not args.energy_min_ev <= args.carrier_energy_ev <= args.energy_max_ev:
        raise ValueError("The fixed pulse carrier must lie inside the spectral grid.")
    if not args.fit_window_ev[0] < args.fit_window_ev[1]:
        raise ValueError("The material fit window must be increasing.")
    if (
        args.energy_min_ev < args.fit_window_ev[0]
        or args.energy_max_ev > args.fit_window_ev[1]
    ):
        raise ValueError("The spectral grid must lie inside the material fit window.")
    if args.energy_points < 5:
        raise ValueError("--energy-points must be at least five.")
    if args.one_fit_modes != 1 or args.multi_fit_modes < 2:
        raise ValueError("Require one_fit_modes=1 and multi_fit_modes>=2.")
    if args.spatial_order_max < 1 or args.modal_audit_points < 101:
        raise ValueError("Spatial order must be positive and modal audit needs >=101 points.")
    if args.c_nm < args.a_nm:
        raise ValueError("The prolate spheroid requires c_nm >= a_nm.")
    if not args.channels or len(set(args.channels)) != len(args.channels):
        raise ValueError("The channel list must be non-empty and unique.")
    fluence = np.asarray(args.fluence_j_cm2, dtype=float)
    if (
        fluence.ndim != 1
        or fluence.size < 2
        or np.any(~np.isfinite(fluence))
        or np.any(fluence <= 0.0)
        or len(set(fluence.tolist())) != fluence.size
    ):
        raise ValueError("Provide at least two distinct positive finite fluences.")
    if args.fluence_label is not None and len(args.fluence_label) != fluence.size:
        raise ValueError("--fluence-label must contain one label per fluence.")
    for name in ("d_debye", "gamma_population_mev", "gamma2_coherence_mev", "post_fs"):
        value = getattr(args, name)
        if value is not None and (not np.isfinite(value) or value < 0.0):
            raise ValueError(
                f"--{name.replace('_', '-')} must be finite and non-negative."
            )
    if args.post_fs == 0.0:
        raise ValueError("--post-fs must be positive when it is supplied.")
    if args.max_auto_tail_extensions < 0:
        raise ValueError("--max-auto-tail-extensions must be non-negative.")
    if not 0.0 < args.minimum_incident_relative_amplitude < 1.0:
        raise ValueError("--minimum-incident-relative-amplitude must lie in (0, 1).")
    if not 0.0 < args.tail_window_fraction < 1.0:
        raise ValueError("--tail-window-fraction must lie in (0, 1).")
    if not 0.0 <= args.max_spectral_leakage < 1.0:
        raise ValueError("--max-spectral-leakage must lie in [0, 1).")
    if not np.isfinite(args.positivity_tolerance) or args.positivity_tolerance < 0.0:
        raise ValueError("--positivity-tolerance must be finite and non-negative.")


def _fit_tables(
    bundles: list[list[Any]],
    max_modes: int,
) -> dict[str, np.ndarray]:
    n_branch = len(bundles)
    n_channel = len(bundles[0])
    mask = np.zeros((n_branch, n_channel, max_modes), dtype=bool)
    strengths = np.zeros_like(mask, dtype=float)
    omega = np.zeros_like(strengths)
    gamma = np.zeros_like(strengths)
    alpha_inf = np.empty((n_branch, n_channel), dtype=float)
    nrms_alpha = np.empty_like(alpha_inf)
    nrms_inv = np.empty_like(alpha_inf)
    max_error = np.empty_like(alpha_inf)
    passive = np.empty_like(alpha_inf, dtype=bool)
    modal_accepted = np.empty_like(alpha_inf, dtype=bool)
    spatial_accepted = np.empty_like(alpha_inf, dtype=bool)
    coupled_stable = np.empty_like(alpha_inf, dtype=bool)
    bright_stable = np.empty_like(alpha_inf, dtype=bool)
    for branch_index, row in enumerate(bundles):
        for channel_index, bundle in enumerate(row):
            fit = bundle.bright_model.fit
            count = fit.strengths_au2.size
            mask[branch_index, channel_index, :count] = True
            strengths[branch_index, channel_index, :count] = fit.strengths_au2
            omega[branch_index, channel_index, :count] = fit.omega_modes_au
            gamma[branch_index, channel_index, :count] = fit.gamma_modes_au
            alpha_inf[branch_index, channel_index] = fit.alpha_inf
            nrms_alpha[branch_index, channel_index] = fit.normalized_rms_alpha
            nrms_inv[branch_index, channel_index] = fit.normalized_rms_inv_alpha
            max_error[branch_index, channel_index] = fit.max_normalized_alpha_error
            modal = bundle.full_model.modal_fit_diagnostics
            spatial = bundle.full_model.spatial_convergence_diagnostics
            passive[branch_index, channel_index] = bool(
                fit.passive_on_fit_window
                and fit.passive_for_all_positive_frequencies
                and modal.passive_on_audit_grid
            )
            modal_accepted[branch_index, channel_index] = bool(modal.accepted)
            spatial_accepted[branch_index, channel_index] = bool(spatial.accepted)
            coupled_stable[branch_index, channel_index] = bool(
                bundle.full_model.coupled_stability.stable
            )
            bright_stable[branch_index, channel_index] = bool(
                bundle.bright_model.linear_stability.stable
            )
            if not (
                passive[branch_index, channel_index]
                and coupled_stable[branch_index, channel_index]
                and bright_stable[branch_index, channel_index]
            ):
                raise RuntimeError(
                    f"Material branch {branch_index}, channel {channel_index} is not passive/stable."
                )
    return {
        "material_fit_mode_mask": mask,
        "material_fit_strengths_au2": strengths,
        "material_fit_omega_modes_au": omega,
        "material_fit_omega_modes_eV": au_to_eV(omega),
        "material_fit_gamma_modes_au": gamma,
        "material_fit_gamma_modes_eV": au_to_eV(gamma),
        "material_fit_alpha_inf_au3": alpha_inf,
        "material_fit_normalized_rms_alpha": nrms_alpha,
        "material_fit_normalized_rms_inverse_alpha": nrms_inv,
        "material_fit_max_normalized_alpha_error": max_error,
        "fit_passive": passive,
        "modal_fit_accepted": modal_accepted,
        "spatial_convergence_accepted": spatial_accepted,
        "coupled_stable": coupled_stable,
        "bright_stable": bright_stable,
    }


def calculate_payload(
    args: argparse.Namespace,
    *,
    generator_path: Path | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Run the selected pulse spectra and return artifact arrays plus metadata."""

    _validate_args(args)
    generator = Path(__file__) if generator_path is None else Path(generator_path)
    energy = np.linspace(args.energy_min_ev, args.energy_max_ev, args.energy_points)
    fluence = np.asarray(sorted(args.fluence_j_cm2), dtype=float)
    if args.fluence_label is None:
        if fluence.size == 3:
            fluence_labels = np.asarray(["linear", "threshold", "nonlinear"], dtype="U32")
        else:
            fluence_labels = np.asarray(
                [f"F{index + 1}" for index in range(fluence.size)], dtype="U32"
            )
    else:
        label_by_value = dict(zip(args.fluence_j_cm2, args.fluence_label))
        fluence_labels = np.asarray([label_by_value[value] for value in fluence], dtype="U64")

    # Verify pulse support before building or solving a heavy FQS model.
    unit_pulse = pulse_for_fluence(
        float(fluence[0]),
        energy_eV=args.carrier_energy_ev,
        tau_fs=args.pulse_tau_fs,
        tau_kind=args.pulse_tau_kind,
        eps_m=args.eps_m,
    )
    analytic = analytic_incident_field_ft_au(unit_pulse, energy)
    analytic_peak = analytic_incident_field_ft_au(
        unit_pulse, np.asarray([args.carrier_energy_ev])
    )[0]
    relative_support = np.abs(analytic) / max(abs(analytic_peak), np.finfo(float).tiny)
    support_mask_1d = relative_support >= args.minimum_incident_relative_amplitude
    if not np.all(support_mask_1d):
        unsupported = energy[~support_mask_1d]
        _apply_policy(
            args.spectral_support_policy,
            "The requested energy interval leaves the reliable incident-pulse "
            f"bandwidth at {unsupported.size} points; first/last unsupported="
            f"{unsupported[0]:.6g}/{unsupported[-1]:.6g} eV.",
        )

    branch_mode_counts = np.asarray(
        [args.one_fit_modes, args.multi_fit_modes], dtype=np.int64
    )
    channel_keys = tuple(args.channels)
    n_branch = BRANCH_IDS.size
    n_channel = len(channel_keys)
    n_fluence = fluence.size
    n_energy = energy.size
    spectrum_shape = (n_branch, n_channel, n_fluence, n_energy)
    trace_shape = (n_branch, n_channel, n_fluence)

    complex_spectra = {
        name: np.empty(spectrum_shape, dtype=complex)
        for name in (
            "alpha_eff_au3",
            "total_dipole_ft_au",
            "incident_field_ft_numeric_au",
        )
    }
    float_spectra = {
        name: np.empty(spectrum_shape, dtype=float)
        for name in (
            "sigma_qs_work_cm2",
            "sigma_half_window_cm2",
        )
    }
    float_traces = {
        name: np.empty(trace_shape, dtype=float)
        for name in (
            "actual_fluence_j_cm2",
            "pulse_E0_au",
            "pulse_E0_v_m",
            "peak_intensity_w_cm2",
            "sigma_energy_transfer_cm2",
            "sigma_energy_half_window_cm2",
            "energy_window_relative_change",
            "work_from_incident_field_j",
            "excited_population_final",
            "excited_population_max",
            "excited_population_min",
            "response_tail_ratio",
            "spectrum_window_max_normalized_change",
            "incident_ft_max_normalized_error",
            "incident_ft_max_pointwise_relative_error",
            "pulse_spectral_leakage",
            "qd_source_spectral_leakage",
            "mnp_dipole_spectral_leakage",
            "mnp_drive_spectral_leakage",
            "mnp_field_spectral_leakage",
            "min_density_eigenvalue",
            "max_bloch_radius",
            "boundary_envelope_fraction",
            "post_fs_effective",
        )
    }
    bool_traces = {
        name: np.zeros(trace_shape, dtype=bool)
        for name in (
            "solver_success",
            "t_final_reached",
            "state_is_finite",
            "response_tail_converged",
            "spectrum_window_converged",
            "incident_ft_converged",
            "work_nonnegative_within_tolerance",
            "density_matrix_positive",
        )
    }
    int_traces = {
        name: np.zeros(trace_shape, dtype=np.int64)
        for name in (
            "solver_status",
            "solver_n_steps",
            "solver_nfev",
            "tail_extension_count",
            "rabi_step_refinement_count",
        )
    }

    bundles: list[list[Any]] = []
    channel_documents: dict[str, list[dict[str, Any]]] = {}
    for branch_index, branch_id in enumerate(BRANCH_IDS.astype(str)):
        row: list[Any] = []
        documents: list[dict[str, Any]] = []
        accuracy_policy = (
            args.one_fit_accuracy_policy if branch_id == "one" else args.fit_quality_policy
        )
        for channel_index, key in enumerate(channel_keys):
            if not args.quiet:
                print(
                    f"Building {branch_id} FQS branch, channel {key} "
                    f"({channel_index + 1}/{n_channel}) ...",
                    flush=True,
                )
            bundle = _build_channel_model(
                ARTICLE_CHANNELS[key],
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
                material_fit_modes=int(branch_mode_counts[branch_index]),
                fit_window_eV=tuple(args.fit_window_ev),
                fit_seed=args.fit_seed,
                alpha_objective_weight=args.alpha_objective_weight,
                inv_alpha_objective_weight=args.inv_alpha_objective_weight,
                max_bright_fit_normalized_rms=args.max_bright_fit_normalized_rms,
                max_bright_fit_pointwise_relative_error=(
                    args.max_bright_fit_pointwise_relative_error
                ),
                bright_fit_quality_policy=accuracy_policy,
                radiative_consistency_policy=args.radiative_consistency_policy,
                fit_quality_policy=accuracy_policy,
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
            )
            row.append(bundle)
            documents.append(_channel_metadata(bundle, args.carrier_energy_ev))
        bundles.append(row)
        channel_documents[branch_id] = documents

    fit_payload = _fit_tables(bundles, int(args.multi_fit_modes))
    bare_sigma = np.empty((n_branch, n_channel, n_energy), dtype=float)
    bare_energy_sigma = np.empty((n_branch, n_channel, n_fluence), dtype=float)
    interaction_A = np.empty((n_branch, n_channel, n_energy), dtype=complex)
    interaction_B = np.empty_like(interaction_A)
    interaction_K = np.empty_like(interaction_A)
    pulse_area = np.empty(n_fluence, dtype=float)
    analytic_ft_by_fluence = np.empty((n_fluence, n_energy), dtype=float)
    relative_amplitude_by_fluence = np.empty_like(analytic_ft_by_fluence)
    support_by_fluence = np.empty_like(analytic_ft_by_fluence, dtype=bool)

    for branch_index, branch_id in enumerate(BRANCH_IDS.astype(str)):
        for channel_index, bundle in enumerate(bundles[branch_index]):
            response = bundle.full_model.frequency_response_from_fit(energy)
            interaction_A[branch_index, channel_index] = response.A_au3
            interaction_B[branch_index, channel_index] = response.B
            interaction_K[branch_index, channel_index] = response.K_au_minus3
            alpha_bare = (
                bundle.bright_model.C
                * bundle.bright_model.alpha_from_fit(energy)
                / bundle.params.eps_m
            )
            bare_sigma[branch_index, channel_index] = np.asarray(
                quasistatic_dipole_cross_section_estimates_cm2(
                    alpha_bare,
                    np.asarray(eV_to_au(energy), dtype=float),
                    bundle.params.eps_m,
                ).quasistatic_work_loss_cm2,
                dtype=float,
            )

            for fluence_index, selected_fluence in enumerate(fluence):
                if not args.quiet:
                    print(
                        f"  {branch_id}/{channel_keys[channel_index]}: "
                        f"F={selected_fluence:.6g} J/cm^2 "
                        f"({fluence_index + 1}/{n_fluence})",
                        flush=True,
                    )
                pulse = pulse_for_fluence(
                    float(selected_fluence),
                    energy_eV=args.carrier_energy_ev,
                    tau_fs=args.pulse_tau_fs,
                    tau_kind=args.pulse_tau_kind,
                    eps_m=bundle.params.eps_m,
                )
                if branch_index == 0 and channel_index == 0:
                    analytic_ft_by_fluence[fluence_index] = (
                        analytic_incident_field_ft_au(pulse, energy)
                    )
                    carrier_ft = analytic_incident_field_ft_au(
                        pulse, np.asarray([args.carrier_energy_ev])
                    )[0]
                    relative_amplitude_by_fluence[fluence_index] = (
                        np.abs(analytic_ft_by_fluence[fluence_index])
                        / max(abs(carrier_ft), np.finfo(float).tiny)
                    )
                    support_by_fluence[fluence_index] = (
                        relative_amplitude_by_fluence[fluence_index]
                        >= args.minimum_incident_relative_amplitude
                    )
                    pulse_area[fluence_index] = float(
                        bundle.params.qd_local_field_factor
                        * bundle.params.d_au
                        * pulse.E0_au
                        * np.sqrt(2.0 * np.pi)
                        * pulse.sigma_t_au
                    )

                result, effective_post_fs, extensions, audit = (
                    _solve_with_spectrum_tail_extension(
                        bundle.full_model,
                        pulse,
                        energy,
                        eps_m=bundle.params.eps_m,
                        pre_sigma=args.pre_sigma,
                        post_fs=args.post_fs,
                        max_auto_tail_extensions=args.max_auto_tail_extensions,
                        tail_policy=args.tail_policy,
                        tail_ratio_tolerance=args.tail_ratio_tolerance,
                        tail_window_fraction=args.tail_window_fraction,
                        method=args.method,
                        rtol=args.rtol,
                        atol=args.atol,
                        points_per_fastest_cycle=args.points_per_fastest_cycle,
                        spectral_window_policy=args.spectral_window_policy,
                        max_spectral_leakage=args.max_spectral_leakage,
                        positivity_policy=args.positivity_policy,
                        positivity_tolerance=args.positivity_tolerance,
                        work_passivity_policy=args.work_passivity_policy,
                        observable_convergence_policy=(
                            args.observable_convergence_policy
                        ),
                        incident_ft_policy=args.incident_ft_policy,
                        minimum_incident_relative_amplitude=(
                            args.minimum_incident_relative_amplitude
                        ),
                        max_incident_ft_pointwise_relative_error=(
                            args.max_incident_ft_pointwise_relative_error
                        ),
                        max_spectrum_window_relative_change=(
                            args.max_spectrum_window_relative_change
                        ),
                        max_energy_window_relative_change=(
                            args.max_energy_window_relative_change
                        ),
                    )
                )
                complex_spectra["alpha_eff_au3"][branch_index, channel_index, fluence_index] = audit.alpha_eff_au3
                complex_spectra["total_dipole_ft_au"][branch_index, channel_index, fluence_index] = audit.total_dipole_ft_au
                complex_spectra["incident_field_ft_numeric_au"][branch_index, channel_index, fluence_index] = audit.incident_field_ft_numeric_au
                float_spectra["sigma_qs_work_cm2"][branch_index, channel_index, fluence_index] = audit.sigma_qs_work_cm2
                float_spectra["sigma_half_window_cm2"][branch_index, channel_index, fluence_index] = audit.sigma_half_window_cm2

                diagnostics = result.diagnostics
                trace_values = {
                    "actual_fluence_j_cm2": result.fluence_j_cm2,
                    "pulse_E0_au": pulse.E0_au,
                    "pulse_E0_v_m": field_au_to_si(pulse.E0_au),
                    "peak_intensity_w_cm2": result.peak_intensity_w_cm2,
                    "sigma_energy_transfer_cm2": result.sigma_energy_transfer_cm2,
                    "sigma_energy_half_window_cm2": audit.sigma_energy_half_window_cm2,
                    "energy_window_relative_change": audit.energy_window_relative_change,
                    "work_from_incident_field_j": result.work_from_incident_field_j,
                    "excited_population_final": result.rho22[-1],
                    "excited_population_max": np.max(result.rho22),
                    "excited_population_min": diagnostics.excited_population_min,
                    "response_tail_ratio": diagnostics.response_tail_ratio,
                    "spectrum_window_max_normalized_change": audit.spectrum_window_max_normalized_change,
                    "incident_ft_max_normalized_error": audit.incident_ft_max_normalized_error,
                    "incident_ft_max_pointwise_relative_error": audit.incident_ft_max_pointwise_relative_error,
                    "pulse_spectral_leakage": diagnostics.pulse_spectral_leakage,
                    "qd_source_spectral_leakage": diagnostics.qd_source_spectral_leakage,
                    "mnp_dipole_spectral_leakage": diagnostics.mnp_dipole_spectral_leakage,
                    "mnp_drive_spectral_leakage": diagnostics.mnp_drive_spectral_leakage,
                    "mnp_field_spectral_leakage": diagnostics.mnp_field_spectral_leakage,
                    "min_density_eigenvalue": diagnostics.min_density_eigenvalue,
                    "max_bloch_radius": diagnostics.max_bloch_radius,
                    "boundary_envelope_fraction": diagnostics.boundary_envelope_fraction,
                    "post_fs_effective": effective_post_fs,
                }
                for name, value in trace_values.items():
                    float_traces[name][branch_index, channel_index, fluence_index] = float(value)
                bool_values = {
                    "solver_success": diagnostics.solver_success,
                    "t_final_reached": diagnostics.t_final_reached,
                    "state_is_finite": diagnostics.state_is_finite,
                    "response_tail_converged": diagnostics.response_tail_converged,
                    "spectrum_window_converged": audit.spectrum_window_converged,
                    "incident_ft_converged": audit.incident_ft_converged,
                    "work_nonnegative_within_tolerance": diagnostics.work_nonnegative_within_tolerance,
                    "density_matrix_positive": diagnostics.min_density_eigenvalue >= -args.positivity_tolerance,
                }
                for name, value in bool_values.items():
                    bool_traces[name][branch_index, channel_index, fluence_index] = bool(value)
                int_values = {
                    "solver_status": diagnostics.solver_status,
                    "solver_n_steps": diagnostics.n_steps,
                    "solver_nfev": diagnostics.nfev,
                    "tail_extension_count": extensions,
                    "rabi_step_refinement_count": diagnostics.rabi_step_refinement_count,
                }
                for name, value in int_values.items():
                    int_traces[name][branch_index, channel_index, fluence_index] = int(value)

                bare_pulse = _bare_mnp_pulse_work_spectral_average(
                    bundle.bright_model,
                    pulse,
                    max_relative_change=args.max_energy_window_relative_change,
                    convergence_policy=args.observable_convergence_policy,
                    work_passivity_policy=args.work_passivity_policy,
                )
                bare_energy_sigma[branch_index, channel_index, fluence_index] = (
                    bare_pulse.sigma_energy_transfer_cm2
                )

    sigma = float_spectra["sigma_qs_work_cm2"]
    delta = sigma - bare_sigma[:, :, None, :]
    common_support = np.all(support_by_fluence, axis=0)
    if np.count_nonzero(common_support) < 5:
        raise RuntimeError(
            "Fewer than five energy points lie inside the common pulse-support band."
        )
    grid_values = np.stack((sigma, delta), axis=-2)[..., common_support]
    grid_diagnostics = energy_grid_resolution_diagnostics(
        energy[common_support],
        grid_values,
        max_midpoint_normalized_error=args.max_energy_grid_midpoint_normalized_error,
    )
    grid_accepted = np.all(np.asarray(grid_diagnostics["accepted"]), axis=-1)
    if not bool(grid_diagnostics["all_accepted"]):
        _apply_policy(
            args.energy_grid_convergence_policy,
            "The energy grid failed nested midpoint interpolation for one or more "
            "sigma/delta curves; increase --energy-points.",
        )

    payload: dict[str, np.ndarray] = {
        "branch_id": BRANCH_IDS,
        "branch_material_mode_count": branch_mode_counts,
        "channel_key": np.asarray(channel_keys, dtype="U32"),
        "channel_label": np.asarray(
            [ARTICLE_CHANNELS[key].label for key in channel_keys], dtype="U80"
        ),
        "selected_fluence_j_cm2": fluence,
        "selected_fluence_label": fluence_labels,
        "carrier_energy_eV": np.asarray(float(args.carrier_energy_ev)),
        "energy_eV": energy,
        "incident_field_ft_analytic_au": analytic_ft_by_fluence,
        "incident_field_relative_amplitude": relative_amplitude_by_fluence,
        "spectrum_support_mask": support_by_fluence,
        "isolated_qd_pulse_area_rad": pulse_area,
        "bare_mnp_sigma_qs_work_cm2": bare_sigma,
        "delta_sigma_qs_work_cm2": delta,
        "bare_mnp_sigma_energy_transfer_cm2": bare_energy_sigma,
        "interaction_A_real_au3": interaction_A.real,
        "interaction_A_imag_au3": interaction_A.imag,
        "interaction_B_real": interaction_B.real,
        "interaction_B_imag": interaction_B.imag,
        "interaction_K_real_au_minus3": interaction_K.real,
        "interaction_K_imag_au_minus3": interaction_K.imag,
        "energy_grid_observable_name": np.asarray(ENERGY_GRID_OBSERVABLES),
        "energy_grid_midpoint_absolute_error_cm2": np.asarray(
            grid_diagnostics["midpoint_absolute_error"], dtype=float
        ),
        "energy_grid_curve_scale_cm2": np.asarray(
            grid_diagnostics["curve_scale"], dtype=float
        ),
        "energy_grid_midpoint_normalized_error": np.asarray(
            grid_diagnostics["midpoint_normalized_error"], dtype=float
        ),
        "energy_grid_converged": np.asarray(grid_accepted, dtype=bool),
        "material_energy_eV": np.asarray(bundles[0][0].params.material.energy_eV),
        "material_n": np.asarray(bundles[0][0].params.material.n),
        "material_k": np.asarray(bundles[0][0].params.material.k),
        **fit_payload,
        **float_spectra,
        **float_traces,
        **bool_traces,
        **int_traces,
    }
    for name, values in complex_spectra.items():
        payload[f"{name}_real"] = values.real
        payload[f"{name}_imag"] = values.imag

    source_selection: dict[str, Any] | None = None
    if args.selection_source_artifact is not None:
        source = args.selection_source_artifact.resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Selection-source artifact not found: {source}")
        source_selection = {
            "path": str(source),
            "sha256": source_hashes((source,))[str(source)],
        }

    resolved_arguments = {
        key: value if not isinstance(value, Path) else str(value)
        for key, value in vars(args).items()
    }
    metadata: dict[str, Any] = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "multi_fit_mode_count": int(args.multi_fit_modes),
        "purpose": (
            "Shah-type pulse spectrum of the operational local-QS work-loss "
            "cross section for one- and multi-pole Au representations"
        ),
        "requested_and_resolved_arguments": resolved_arguments,
        "input_signature_sha256": canonical_sha256(resolved_arguments),
        "fluence_selection": {
            "source_artifact": source_selection,
            "rule": (
                "values are supplied from the preceding P_exc(F) calculation; "
                "recommended roles are weak-field, threshold and nonlinear"
            ),
            "labels": fluence_labels.tolist(),
        },
        "calculation_method": {
            "ode_solves": "one per branch x channel x selected fluence",
            "spectral_reconstruction": (
                "chunked integral of real time traces with exp(+i omega t); "
                "the carrier is fixed and is not scanned"
            ),
            "fourier_sign": "+i omega t, consistent with passive Im(alpha)>=0",
            "energy_grid_support": (
                "analytic real-Gaussian field amplitude relative to its carrier value"
            ),
        },
        "model_scope": {
            "time_backend": "FullQSSpheroidPulseModel",
            "spatial_model": "full analytic local-electrostatic prolate-spheroid response",
            "qd_model": "point electric dipole, semiclassical two-level system",
            "branches_differ_only_by": "number of Lorentz material poles and accuracy policy",
            "laser_propagation_direction_included": False,
            "electric_field_direction_included": True,
            "limitations": (
                "local quasistatics, homogeneous axisymmetric spheroid, point QD; "
                "no retardation, nonlocality, tunnelling or charge transfer"
            ),
        },
        "observable_definitions": {
            "alpha_eff": "mu_total(E)/(eps_m E_inc(E)) from the nonlinear pulse trace",
            "sigma_qs_work_cm2": "k(E) Im(alpha_eff(E;F))/epsilon_0",
            "delta_sigma_qs_work_cm2": "hybrid minus bare-MNP result for the same branch/channel",
            "terminology": (
                "operational quasistatic work-loss estimate analogous to Shah sigma_abs; "
                "not separately proven metal absorption or exact extinction; for a "
                "nonlinear pulse it is a spectral output/input ratio, not a "
                "frequency-local linear susceptibility"
            ),
            "delta_limit": (
                "contains QD work, coupling-induced MNP work, back-action and interference; "
                "it is not QD-only absorption"
            ),
        },
        "pulse_definition": {
            "real_field": "E0 exp[-t^2/(2 sigma_t^2)] cos(omega_L t)",
            "carrier_energy_eV": float(args.carrier_energy_ev),
            "duration_fs": float(args.pulse_tau_fs),
            "duration_kind": args.pulse_tau_kind,
            "fluence": "exact integral n_m epsilon_0 c E_inc(t)^2 dt",
        },
        "quality_gates": {
            "minimum_incident_relative_amplitude": float(
                args.minimum_incident_relative_amplitude
            ),
            "max_incident_ft_pointwise_relative_error": float(
                args.max_incident_ft_pointwise_relative_error
            ),
            "max_spectrum_window_relative_change": float(
                args.max_spectrum_window_relative_change
            ),
            "max_energy_window_relative_change": float(
                args.max_energy_window_relative_change
            ),
            "max_energy_grid_midpoint_normalized_error": float(
                args.max_energy_grid_midpoint_normalized_error
            ),
            "one_fit_accuracy_policy": args.one_fit_accuracy_policy,
            "multi_fit_accuracy_policy": args.fit_quality_policy,
            "note": "N=1 may miss accuracy; passivity and stability remain mandatory",
        },
        "branches": {
            branch_id: channel_documents[branch_id]
            for branch_id in BRANCH_IDS.astype(str)
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
        "material": {
            "interpolation": MATERIAL_INTERPOLATION,
            "high_frequency_epsilon": MATERIAL_HIGH_FREQUENCY_EPSILON,
            "source": "material_energy_eV/material_n/material_k arrays",
        },
        "model_profile": NATIVE_MODEL_PROFILE,
        "array_dimensions": {
            "spectra": "branch x channel x selected_fluence x energy",
            "trace_diagnostics": "branch x channel x selected_fluence",
            "bare_spectrum": "branch x channel x energy",
        },
        "array_units": {
            "selected_fluence_j_cm2": "J cm^-2",
            "carrier_energy_eV": "eV",
            "energy_eV": "eV",
            "pulse_E0_au": "atomic electric field",
            "pulse_E0_v_m": "V m^-1",
            "peak_intensity_w_cm2": "W cm^-2",
            "alpha_eff_au3_real/imag": "a0^3",
            "total_dipole_ft_au_real/imag": "atomic dipole times atomic time",
            "incident_field_ft_numeric_au_real/imag": (
                "atomic electric field times atomic time"
            ),
            "sigma_qs_work_cm2": "cm^2",
            "bare_mnp_sigma_qs_work_cm2": "cm^2",
            "delta_sigma_qs_work_cm2": "cm^2",
            "sigma_energy_transfer_cm2": "cm^2",
            "work_from_incident_field_j": "J",
            "post_fs_effective": "fs",
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
        "provenance": {
            "generator": str(generator.resolve()),
            "git": git_provenance(PROJECT_ROOT),
            "source_sha256": source_hashes(
                (
                    generator,
                    PROJECT_ROOT / "article_observables" / "qd_mnp_calculate_work_loss_fluence.py",
                    PROJECT_ROOT / "article_observables" / "qd_mnp_material_modes_artifact.py",
                    PROJECT_ROOT / "qd_mnp_rational_fit.py",
                    PROJECT_ROOT / "qd_mnp_full_qs_model.py",
                    PROJECT_ROOT / "qd_mnp_spheroid_green.py",
                    PROJECT_ROOT / "qd_mnp_spheroid_equatorial.py",
                )
            ),
        },
    }
    return payload, metadata


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    payload, metadata = calculate_payload(args, generator_path=Path(__file__))
    output = atomic_write_npz(
        args.output,
        payload,
        metadata,
        overwrite=args.overwrite,
    )
    if not args.quiet:
        print(f"Saved {SCHEMA_NAME} v{SCHEMA_VERSION}: {output.resolve()}")
    return output


if __name__ == "__main__":
    main()
