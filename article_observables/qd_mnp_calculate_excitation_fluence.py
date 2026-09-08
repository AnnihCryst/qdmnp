"""Calculate post-pulse QD excitation versus incident pulse fluence.

The five inequivalent symmetry channels of an axisymmetric spheroid are
calculated with :class:`qd_mnp_full_qs_model.FullQSSpheroidPulseModel`.  An
isolated-QD control uses the identical non-RWA Bloch equations with the MNP
field set to zero. Calculation and plotting are deliberately separated: this
article-observables script writes one self-contained NPZ artifact, including a
JSON metadata document, and does not create figures.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import tomllib
from typing import Any
import warnings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import scipy
from scipy.constants import epsilon_0 as EPSILON_0_SI, hbar as HBAR_SI
from scipy.integrate import solve_ivp

from qd_mnp_full_qs_model import (
    FullQSSolveDiagnostics,
    FullQSSpheroidPulseModel,
    build_positive_dark_reduction,
)
from qd_mnp_rational_fit import (
    AU_DIPOLE_C_M,
    AU_ENERGY_EV,
    AU_ENERGY_J,
    AU_FIELD_V_M,
    AU_LENGTH_M,
    AU_SPEED_OF_LIGHT,
    AU_TIME_S,
    C_SI,
    DEBYE_C_M,
    E_CHARGE,
    GaussianPulse,
    HybridQDPlasmonModel,
    MATERIAL_HIGH_FREQUENCY_EPSILON,
    MATERIAL_INTERPOLATION,
    NATIVE_MODEL_PROFILE,
    au_to_eV,
    au_to_fs,
    au_to_nm,
    eV_to_au,
    field_au_to_si,
    fs_to_au,
    make_params_with_overrides,
    params_to_physical_dict,
)
from qd_mnp_spheroid_equatorial import EquatorialSpheroidGreenInteraction
from qd_mnp_spheroid_green import SpheroidGreenInteraction


SCHEMA_NAME = "qd_mnp_excitation_fluence"
SCHEMA_VERSION = 1
SIDE_DIRECT_REFERENCE_ORDER_MAX = 8
POLICIES = ("raise", "warn", "ignore")


@dataclass(frozen=True)
class ChannelSpec:
    channel_id: str
    label: str
    qd_placement: str
    orientation: str
    side_transverse_alignment: str | None = None


HYBRID_CHANNELS = (
    ChannelSpec("axis_long", "tip, longitudinal", "axis", "long"),
    ChannelSpec("axis_trans", "tip, transverse", "axis", "trans"),
    ChannelSpec("side_long", "side, longitudinal", "side", "long"),
    ChannelSpec(
        "side_trans_radial",
        "side, transverse radial",
        "side",
        "trans",
        "radial",
    ),
    ChannelSpec(
        "side_trans_tangential",
        "side, transverse tangential",
        "side",
        "trans",
        "tangential",
    ),
)
ALL_CHANNEL_IDS = np.asarray(
    ["bare_qd", *(channel.channel_id for channel in HYBRID_CHANNELS)],
    dtype="U32",
)


PUBLICATION_PRESET = {
    "fluence_min_j_cm2": 1.0e-10,
    "fluence_max_j_cm2": 1.0e-3,
    "points": 65,
    "grid_scale": "sqrt",
    "max_isolated_pulse_area_step_rad": 0.25,
    "max_fluence_grid_midpoint_error": 0.01,
    "fluence_grid_convergence_policy": "raise",
    "spatial_order_max": 80,
    "material_fit_modes": 9,
    "modal_audit_points": 2001,
    "reduction_fit_grid_points": 1001,
    "reduction_audit_grid_points": 1601,
    "post_fs": None,
    "spatial_convergence_policy": "raise",
    "tail_policy": "raise",
    "spectral_window_policy": "warn",
    "fit_quality_policy": "raise",
    "bright_fit_quality_policy": "raise",
    "population_decay_policy": "raise",
}

QUICK_PRESET = {
    "fluence_min_j_cm2": 1.0e-9,
    "fluence_max_j_cm2": 1.0e-4,
    "points": 3,
    "grid_scale": "log",
    "max_isolated_pulse_area_step_rad": 10.0,
    "max_fluence_grid_midpoint_error": 0.25,
    "fluence_grid_convergence_policy": "warn",
    # Deliberately tiny preview: it keeps the stability audit deterministic
    # and fast.  Spatial-convergence warnings mark it as non-publication data.
    "spatial_order_max": 2,
    "material_fit_modes": 9,
    "modal_audit_points": 301,
    "reduction_fit_grid_points": 301,
    "reduction_audit_grid_points": 401,
    "post_fs": 250.0,
    "spatial_convergence_policy": "warn",
    "tail_policy": "warn",
    "spectral_window_policy": "ignore",
    "fit_quality_policy": "warn",
    "bright_fit_quality_policy": "warn",
    "population_decay_policy": "warn",
}


def _json_ready(value: Any) -> Any:
    """Convert NumPy/dataclass values to strict, portable JSON values."""

    if hasattr(value, "__dataclass_fields__"):
        return _json_ready(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, float):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _git_provenance() -> dict[str, Any]:
    result: dict[str, Any] = {"commit": None, "working_tree_dirty": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return result
    result["commit"] = commit
    result["working_tree_dirty"] = bool(status.strip())
    return result


def _source_file_hashes() -> dict[str, str]:
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


def _project_version() -> str | None:
    try:
        return importlib.metadata.version("qd-mnp")
    except importlib.metadata.PackageNotFoundError:
        pyproject = PROJECT_ROOT / "pyproject.toml"
        try:
            with pyproject.open("rb") as stream:
                return str(tomllib.load(stream)["project"]["version"])
        except (OSError, KeyError, tomllib.TOMLDecodeError):
            return None


def _apply_policy(policy: str, message: str) -> None:
    if policy == "raise":
        raise RuntimeError(message)
    if policy == "warn":
        warnings.warn(message, RuntimeWarning, stacklevel=3)


def fluence_grid_resolution_diagnostics(
    fluence_j_cm2: np.ndarray,
    population: np.ndarray,
    isolated_pulse_area_rad: np.ndarray,
    *,
    max_midpoint_error: float,
    max_pulse_area_step_rad: float,
) -> dict[str, Any]:
    """Audit a fine fluence grid against its nested every-other-point grid."""

    fluence = np.asarray(fluence_j_cm2, dtype=float)
    values = np.asarray(population, dtype=float)
    pulse_area = np.asarray(isolated_pulse_area_rad, dtype=float)
    if fluence.ndim != 1 or values.ndim != 2 or values.shape[1] != fluence.size:
        raise ValueError("Population must have shape (n_channels, n_fluences).")
    if pulse_area.shape != fluence.shape:
        raise ValueError("Pulse area must match the fluence grid.")
    if fluence.size < 3:
        midpoint_error = np.full(values.shape[0], np.inf, dtype=float)
    else:
        amplitude_coordinate = np.sqrt(fluence)
        audit_indices = np.arange(1, fluence.size - 1, 2, dtype=int)
        if audit_indices.size == 0:
            midpoint_error = np.full(values.shape[0], np.inf, dtype=float)
        else:
            left = audit_indices - 1
            right = audit_indices + 1
            fractions = (
                amplitude_coordinate[audit_indices] - amplitude_coordinate[left]
            ) / (amplitude_coordinate[right] - amplitude_coordinate[left])
            interpolated = values[:, left] + fractions[None, :] * (
                values[:, right] - values[:, left]
            )
            midpoint_error = np.max(
                np.abs(values[:, audit_indices] - interpolated), axis=1
            )
    maximum_area_step = (
        float(np.max(np.abs(np.diff(pulse_area))))
        if pulse_area.size >= 2
        else float("inf")
    )
    accepted_by_channel = np.asarray(
        (midpoint_error <= max_midpoint_error)
        & (maximum_area_step <= max_pulse_area_step_rad),
        dtype=bool,
    )
    return {
        "midpoint_interpolation_error_by_channel": midpoint_error,
        "maximum_isolated_pulse_area_step_rad": maximum_area_step,
        "accepted_by_channel": accepted_by_channel,
        "accepted": bool(np.all(accepted_by_channel)),
        "audited_coordinate": "sqrt(fluence), proportional to incident field amplitude",
    }


def _resolved_settings(args: argparse.Namespace) -> dict[str, Any]:
    preset = PUBLICATION_PRESET if args.preset == "publication" else QUICK_PRESET
    requested_cli_arguments = vars(args).copy()
    settings = vars(args).copy()
    for key, value in preset.items():
        if settings.get(key) is None:
            settings[key] = value

    if args.fluence_grid_j_cm2 is not None:
        grid = np.asarray(args.fluence_grid_j_cm2, dtype=float)
        if grid.ndim != 1 or grid.size < 3 or np.any(~np.isfinite(grid)) or np.any(grid <= 0.0):
            raise ValueError(
                "--fluence-grid-j-cm2 must contain at least three finite positive values "
                "so that fluence-grid convergence can be audited."
            )
        if np.any(np.diff(grid) <= 0.0):
            raise ValueError("--fluence-grid-j-cm2 values must be strictly increasing.")
        settings["fluence_grid_j_cm2"] = grid
    else:
        f_min = float(settings["fluence_min_j_cm2"])
        f_max = float(settings["fluence_max_j_cm2"])
        points = int(settings["points"])
        if not (np.isfinite(f_min) and np.isfinite(f_max) and 0.0 < f_min < f_max):
            raise ValueError("The fluence interval must satisfy 0 < minimum < maximum.")
        if points < 3:
            raise ValueError("--points must be at least 3 for a converged-dependence audit.")
        if settings["grid_scale"] == "log":
            grid = np.geomspace(f_min, f_max, points)
        elif settings["grid_scale"] == "sqrt":
            grid = np.linspace(np.sqrt(f_min), np.sqrt(f_max), points) ** 2
        else:
            grid = np.linspace(f_min, f_max, points)
        settings["fluence_grid_j_cm2"] = grid

    for key in (
        "c_nm",
        "a_nm",
        "qd_radius_nm",
        "gap_nm",
        "eps_m",
        "eps_qd",
        "pulse_energy_ev",
        "pulse_tau_fs",
        "start_sigma",
        "rtol",
        "atol",
        "tail_ratio_tolerance",
        "tail_window_fraction",
        "max_spectral_leakage",
        "positivity_tolerance",
        "max_modal_normalized_rms",
        "max_modal_relative_error",
        "max_bright_fit_normalized_rms",
        "max_bright_fit_pointwise_relative_error",
        "max_population_decay_fraction_at_read",
        "max_isolated_pulse_area_step_rad",
        "max_fluence_grid_midpoint_error",
        "spatial_convergence_rtol",
        "reduction_rms_tolerance",
        "reduction_max_tolerance",
    ):
        value = float(settings[key])
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{key} must be finite and positive.")
        settings[key] = value
    if settings["c_nm"] < settings["a_nm"]:
        raise ValueError("This script expects an axisymmetric prolate spheroid with c_nm >= a_nm.")
    if not (0.0 < settings["tail_window_fraction"] <= 1.0):
        raise ValueError("tail_window_fraction must lie in (0, 1].")
    if not (0.0 < settings["max_spectral_leakage"] < 1.0):
        raise ValueError("max_spectral_leakage must lie in (0, 1).")
    if settings["max_population_decay_fraction_at_read"] >= 1.0:
        raise ValueError(
            "max_population_decay_fraction_at_read must be smaller than 1."
        )
    for key in (
        "spatial_order_max",
        "material_fit_modes",
        "modal_audit_points",
        "reduction_fit_grid_points",
        "reduction_audit_grid_points",
        "reduction_reaudit_points",
        "points_per_fastest_cycle",
        "workers",
    ):
        settings[key] = int(settings[key])
        minimum = 101 if "grid_points" in key or key == "modal_audit_points" else 1
        if key == "points_per_fastest_cycle":
            minimum = 8
        if settings[key] < minimum:
            raise ValueError(f"{key} must be at least {minimum}.")
    settings["max_auto_tail_extensions"] = int(settings["max_auto_tail_extensions"])
    if settings["max_auto_tail_extensions"] < 0:
        raise ValueError("max_auto_tail_extensions must be non-negative.")
    if settings["reduction_fit_grid_points"] == settings["reduction_audit_grid_points"]:
        raise ValueError("Reduction fit and audit grids must have different sizes.")
    if settings["post_fs"] is not None:
        settings["post_fs"] = float(settings["post_fs"])
        if not np.isfinite(settings["post_fs"]) or settings["post_fs"] <= 0.0:
            raise ValueError("post_fs must be finite and positive or omitted.")
    settings["_requested_cli_arguments"] = requested_cli_arguments
    return settings


def _channel_center_distance_nm(channel: ChannelSpec, settings: dict[str, Any]) -> float:
    directional_radius = settings["c_nm"] if channel.qd_placement == "axis" else settings["a_nm"]
    return float(directional_radius + settings["qd_radius_nm"] + settings["gap_nm"])


def _channel_params(channel: ChannelSpec, settings: dict[str, Any]):
    return make_params_with_overrides(
        c_nm=settings["c_nm"],
        a_nm=settings["a_nm"],
        r_nm=_channel_center_distance_nm(channel, settings),
        qd_radius_nm=settings["qd_radius_nm"],
        eps_m=settings["eps_m"],
        eps_qd=settings["eps_qd"],
        d_debye=settings["d_debye"],
        omega0_ev=settings["omega0_ev"],
        gamma_population_mev=settings["gamma_population_mev"],
        gamma2_coherence_mev=settings["gamma2_coherence_mev"],
        qd_dipole_convention=settings["qd_dipole_convention"],
        orientation=channel.orientation,
        qd_placement=channel.qd_placement,
        side_transverse_alignment=channel.side_transverse_alignment,
    )


def _build_channel_model(channel: ChannelSpec, settings: dict[str, Any]):
    params = _channel_params(channel, settings)
    bright_model = HybridQDPlasmonModel(
        params,
        orientation=channel.orientation,
        n_modes=settings["material_fit_modes"],
        fit_window_eV=(settings["fit_min_ev"], settings["fit_max_ev"]),
        weight_center_eV=settings["weight_center_ev"],
        weight_sigma_eV=settings["weight_sigma_ev"],
        max_fit_normalized_rms=None,
        max_fit_pointwise_relative_error=None,
        radiative_consistency_policy=settings["radiative_consistency_policy"],
        verbose=settings["verbose_fit"],
    )
    fit = bright_model.fit
    if (
        fit.normalized_rms_alpha > settings["max_bright_fit_normalized_rms"]
        or fit.normalized_rms_inv_alpha
        > settings["max_bright_fit_normalized_rms"]
        or fit.max_normalized_alpha_error
        > settings["max_bright_fit_pointwise_relative_error"]
    ):
        _apply_policy(
            settings["bright_fit_quality_policy"],
            "The deliberately selected material Lorentz fit misses the "
            "configured accuracy gate: "
            f"NRMS(alpha)={fit.normalized_rms_alpha:.6g}, "
            f"NRMS(1/alpha)={fit.normalized_rms_inv_alpha:.6g}, "
            f"max normalized alpha error={fit.max_normalized_alpha_error:.6g}; "
            "this is expected for a one-oscillator control but must be "
            "reported as an approximation, not as an accurate Au fit.",
        )
    if channel.qd_placement == "side":
        kernel = EquatorialSpheroidGreenInteraction.from_params(
            params,
            orientation=channel.orientation,
            n_max=settings["spatial_order_max"],
        )
    else:
        kernel = SpheroidGreenInteraction.from_params(
            params,
            orientation=channel.orientation,
            n_max=settings["spatial_order_max"],
        )

    reduction = None
    if channel.qd_placement == "side" and settings["spatial_order_max"] > SIDE_DIRECT_REFERENCE_ORDER_MAX:
        reduction = build_positive_dark_reduction(
            bright_model,
            kernel,
            fit_grid_points=settings["reduction_fit_grid_points"],
            audit_grid_points=settings["reduction_audit_grid_points"],
            rms_tolerance=settings["reduction_rms_tolerance"],
            max_tolerance=settings["reduction_max_tolerance"],
            max_nodes=settings["reduction_max_nodes"],
            policy=settings["reduction_policy"],
        )
        if not reduction.diagnostics.accepted:
            raise RuntimeError(f"Unaccepted dark-kernel reduction for {channel.channel_id}.")

    model = FullQSSpheroidPulseModel(
        bright_model,
        kernel,
        fit_quality_policy=settings["fit_quality_policy"],
        max_modal_normalized_rms=settings["max_modal_normalized_rms"],
        max_modal_relative_error=settings["max_modal_relative_error"],
        modal_audit_points=settings["modal_audit_points"],
        spatial_convergence_policy=settings["spatial_convergence_policy"],
        spatial_convergence_rtol=settings["spatial_convergence_rtol"],
        dark_reduction=reduction,
        reduction_reaudit_points=settings["reduction_reaudit_points"],
        max_reduction_normalized_rms=settings["reduction_rms_tolerance"],
        max_reduction_normalized_error=settings["reduction_max_tolerance"],
    )
    return params, bright_model, kernel, reduction, model


def _pulse_for_fluence(fluence_j_cm2: float, settings: dict[str, Any]) -> GaussianPulse:
    omega = float(eV_to_au(settings["pulse_energy_ev"]))
    tau = float(fs_to_au(settings["pulse_tau_fs"]))
    reference = GaussianPulse(E0_au=1.0, omegaL_au=omega, tau_au=tau, tau_kind=settings["pulse_tau_kind"])
    reference_fluence = reference.fluence_j_cm2(eps_m=settings["eps_m"])
    amplitude = float(np.sqrt(float(fluence_j_cm2) / reference_fluence))
    pulse = GaussianPulse(E0_au=amplitude, omegaL_au=omega, tau_au=tau, tau_kind=settings["pulse_tau_kind"])
    relative_error = abs(pulse.fluence_j_cm2(eps_m=settings["eps_m"]) - fluence_j_cm2) / fluence_j_cm2
    if relative_error > 5.0e-13:
        raise RuntimeError("Internal fluence-to-field conversion failed its round-trip check.")
    return pulse


def _windowed_ratio(t_au: np.ndarray, signal: np.ndarray, fraction: float) -> float:
    peak = float(np.max(np.abs(signal)))
    if peak == 0.0:
        return 0.0
    start = float(t_au[-1] - fraction * (t_au[-1] - t_au[0]))
    index = max(0, int(np.searchsorted(t_au, start, side="left")) - 1)
    tail_t = np.asarray(t_au[index:], dtype=float)
    tail_y = np.asarray(signal[index:], dtype=float)
    duration = float(tail_t[-1] - tail_t[0])
    if duration <= 0.0:
        return float(abs(tail_y[-1]) / peak)
    rms = float(np.sqrt(max(np.trapezoid(tail_y**2, tail_t) / duration, 0.0)))
    return rms / peak


def _solve_bare_qd(pulse: GaussianPulse, params, t_span_au: tuple[float, float], settings: dict[str, Any]):
    local = float(params.qd_local_field_factor)

    def rhs(time: float, state: np.ndarray) -> np.ndarray:
        W, Q, P = state
        rabi = 2.0 * params.d_au * local * float(pulse.field(time))
        return np.asarray(
            [
                rabi * Q - params.gamma_au * (W + 1.0),
                -params.omega0_au * P - rabi * W - params.Gamma_au * Q,
                params.omega0_au * Q - params.Gamma_au * P,
            ],
            dtype=float,
        )

    frequency_ceiling = max(
        float(pulse.omegaL_au),
        float(params.omega0_au),
        float(2.0 * abs(params.d_au) * local * abs(pulse.E0_au)),
    )
    max_step = 2.0 * np.pi / (settings["points_per_fastest_cycle"] * frequency_ceiling)
    solution = solve_ivp(
        rhs,
        t_span=t_span_au,
        y0=np.asarray([-1.0, 0.0, 0.0]),
        method=settings["method"],
        rtol=settings["rtol"],
        atol=settings["atol"],
        max_step=max_step,
    )
    if not solution.success or solution.t.size < 2 or np.any(~np.isfinite(solution.y)):
        raise RuntimeError(f"Isolated-QD solve failed: {solution.message}")
    W, Q, P = solution.y
    rho22 = 0.5 * (W + 1.0)
    bloch_radius = np.sqrt(W**2 + Q**2 + P**2)
    minimum_eigenvalue = float(np.min(0.5 * (1.0 - bloch_radius)))
    if minimum_eigenvalue < -settings["positivity_tolerance"]:
        _apply_policy(
            settings["positivity_policy"],
            "The isolated-QD trajectory left the Bloch ball: "
            f"minimum eigenvalue={minimum_eigenvalue:.6e}.",
        )
    coherence_tail = _windowed_ratio(
        solution.t,
        np.hypot(Q, P),
        settings["tail_window_fraction"],
    )
    steps = np.diff(solution.t)
    diagnostics = {
        "solver_success": bool(solution.success),
        "solver_status": int(solution.status),
        "solver_message": str(solution.message),
        "n_steps": int(solution.t.size - 1),
        "nfev": int(solution.nfev),
        "min_step_au": float(np.min(steps)),
        "max_step_au": float(np.max(steps)),
        "max_step_limit_au": float(max_step),
        "integration_frequency_ceiling_au": float(frequency_ceiling),
        "t_final_reached": bool(np.isclose(solution.t[-1], t_span_au[1], rtol=0.0, atol=1.0e-10 * max(abs(t_span_au[1]), 1.0))),
        "state_is_finite": bool(np.all(np.isfinite(solution.y))),
        "boundary_envelope_fraction": float(np.max(pulse.envelope(np.asarray(t_span_au)))),
        "excited_population_min": float(np.min(rho22)),
        "excited_population_max": float(np.max(rho22)),
        "max_bloch_radius": float(np.max(bloch_radius)),
        "min_density_eigenvalue": minimum_eigenvalue,
        "pulse_spectral_fraction_in_fit_window": float(pulse.positive_frequency_spectral_fraction((settings["fit_min_ev"], settings["fit_max_ev"]))),
        "pulse_spectral_leakage": float(pulse.spectral_leakage_fraction((settings["fit_min_ev"], settings["fit_max_ev"]))),
        "response_tail_ratio": float(coherence_tail),
        "response_tail_tolerance": float(settings["tail_ratio_tolerance"]),
        "response_tail_converged": bool(coherence_tail <= settings["tail_ratio_tolerance"]),
        "response_tail_window_fraction": float(settings["tail_window_fraction"]),
    }
    return solution.t, rho22, diagnostics


def _solve_full_model(model: FullQSSpheroidPulseModel, pulse: GaussianPulse, t_span_au: tuple[float, float], settings: dict[str, Any]):
    return model.solve(
        pulse,
        t_span_au=t_span_au,
        method=settings["method"],
        rtol=settings["rtol"],
        atol=settings["atol"],
        points_per_fastest_cycle=settings["points_per_fastest_cycle"],
        spectral_window_policy=settings["spectral_window_policy"],
        max_spectral_leakage=settings["max_spectral_leakage"],
        positivity_policy=settings["positivity_policy"],
        positivity_tolerance=settings["positivity_tolerance"],
        work_passivity_policy=settings["work_passivity_policy"],
        response_tail_policy="ignore",
        response_tail_tolerance=settings["tail_ratio_tolerance"],
        response_tail_window_fraction=settings["tail_window_fraction"],
    )


def _diagnostic_array_dtype(field_name: str):
    if field_name in {"solver_message"}:
        return "U512", ""
    if field_name in {
        "solver_success",
        "t_final_reached",
        "state_is_finite",
        "response_tail_converged",
        "work_nonnegative_within_tolerance",
        "spectral_abscissa_available",
        "spectral_abscissa_is_bound",
        "decay_rate_estimate_is_exact",
    }:
        return bool, False
    if field_name in {
        "solver_status",
        "n_steps",
        "nfev",
        "spatial_order_max",
        "exact_spatial_mode_count",
        "dynamic_spatial_mode_count",
        "reduced_dark_node_count",
        "material_poles_per_spatial_order",
        "rabi_step_refinement_count",
    }:
        return np.int64, -1
    return float, np.nan


def _diagnostic_units(field_name: str) -> str:
    if field_name in {"min_step_au", "max_step_au", "max_step_limit_au"}:
        return "atomic unit of time"
    if field_name in {
        "integration_frequency_ceiling_au",
        "spectral_abscissa_au",
        "decay_rate_estimate_au",
        "spectral_radius_au",
        "incident_peak_rabi_frequency_au",
        "observed_peak_rabi_frequency_au",
    }:
        return "atomic angular frequency"
    if field_name == "work_passivity_tolerance_au":
        return "hartree"
    if field_name == "solver_message":
        return "text"
    return "1"


def _model_metadata(
    channel: ChannelSpec, params, bright, kernel, reduction, model, settings
) -> dict[str, Any]:
    physical = params_to_physical_dict(params, channel.orientation)
    physical["coupling_model"] = "analytic_full_qs_axisymmetric_spheroid_green"
    physical["directional_mnp_radius_nm"] = float(
        au_to_nm(params.directional_mnp_radius_au)
    )
    physical["directional_mnp_radius_to_separation"] = float(
        params.directional_mnp_radius_au / params.R_au
    )
    reduction_metadata = None
    if reduction is not None:
        reduction_metadata = {
            "source_measure_sha256": reduction.source_measure_sha256,
            "depolarization_nodes": reduction.depolarization_nodes,
            "weights_au_minus3": reduction.weights_au_minus3,
            "source_mode_indices": reduction.source_mode_indices,
            "diagnostics": reduction.diagnostics,
            "current_transfer_reaudit": model.dark_reduction_reaudit_diagnostics,
        }
    return _json_ready(
        {
            "channel_id": channel.channel_id,
            "label": channel.label,
            "qd_placement": channel.qd_placement,
            "orientation": channel.orientation,
            "side_transverse_alignment": channel.side_transverse_alignment,
            "physical_parameters": physical,
            "resolved_R_nm": float(au_to_nm(params.R_au)),
            "resolved_surface_gap_nm": float(au_to_nm(params.surface_gap_au)),
            "kernel_class": type(kernel).__name__,
            "kernel_is_spherical": bool(kernel.is_spherical),
            "spatial_order_max": int(model.spatial_order_max),
            "exact_spatial_mode_count": int(model.exact_spatial_mode_count),
            "dynamic_spatial_mode_count": int(model.n_spatial_modes),
            "bright_mode_index": int(model.bright_mode_index),
            "material_fit": {
                "n_modes": int(bright.n_modes),
                "fit_window_eV": bright.fit_window_eV,
                "weight_center_eV": bright.weight_center_eV,
                "weight_sigma_eV": bright.weight_sigma_eV,
                "alpha_inf": bright.fit.alpha_inf,
                "strengths_au2_array": "fit_strengths_au2",
                "omega_modes_au_array": "fit_omega_modes_au",
                "gamma_modes_au_array": "fit_gamma_modes_au",
                "alpha_objective_weight": bright.alpha_objective_weight,
                "inv_alpha_objective_weight": bright.inv_alpha_objective_weight,
                "scenario_quality_policy": settings["bright_fit_quality_policy"],
                "max_fit_normalized_rms_gate": settings[
                    "max_bright_fit_normalized_rms"
                ],
                "max_fit_pointwise_relative_error_gate": settings[
                    "max_bright_fit_pointwise_relative_error"
                ],
                "deterministic_fit_seed": bright.seed,
                "normalized_rms_alpha": bright.fit.normalized_rms_alpha,
                "normalized_rms_inv_alpha": bright.fit.normalized_rms_inv_alpha,
                "max_normalized_alpha_error": bright.fit.max_normalized_alpha_error,
                "min_imag_alpha_fit_window": bright.fit.min_imag_alpha_fit_window,
                "passive_for_all_positive_frequencies": bright.fit.passive_for_all_positive_frequencies,
                "passivity_grid_points": bright.fit.passivity_grid_points,
            },
            "modal_transform_diagnostics": model.modal_fit_diagnostics,
            "spatial_convergence_diagnostics": model.spatial_convergence_diagnostics,
            "coupled_stability_diagnostics": model.coupled_stability,
            "dark_reduction": reduction_metadata,
        }
    )


def calculate_excitation_fluence(settings: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Run all channels and return NPZ arrays plus JSON-compatible metadata."""

    built = []
    for channel in HYBRID_CHANNELS:
        print(f"Building full-QS channel {channel.channel_id} ...", flush=True)
        built.append((channel, *_build_channel_model(channel, settings)))

    reference_params = built[0][1]
    fluence = np.asarray(settings["fluence_grid_j_cm2"], dtype=float)
    n_fluence = fluence.size
    n_all = ALL_CHANNEL_IDS.size
    p_read = np.empty((n_all, n_fluence), dtype=float)
    p_max = np.empty_like(p_read)
    t_max_fs = np.empty_like(p_read)
    read_time_fs = np.empty(n_fluence, dtype=float)
    tail_extension_count = np.empty(n_fluence, dtype=np.int64)
    population_decay_fraction_at_read = np.empty(n_fluence, dtype=float)

    diagnostic_payload: dict[str, np.ndarray] = {}
    for diagnostic_field in fields(FullQSSolveDiagnostics):
        dtype, fill = _diagnostic_array_dtype(diagnostic_field.name)
        diagnostic_payload[f"full_diagnostic__{diagnostic_field.name}"] = np.full(
            (len(HYBRID_CHANNELS), n_fluence),
            fill,
            dtype=dtype,
        )

    bare_diagnostic_names = (
        "solver_success",
        "solver_status",
        "solver_message",
        "n_steps",
        "nfev",
        "min_step_au",
        "max_step_au",
        "max_step_limit_au",
        "integration_frequency_ceiling_au",
        "t_final_reached",
        "state_is_finite",
        "boundary_envelope_fraction",
        "excited_population_min",
        "excited_population_max",
        "max_bloch_radius",
        "min_density_eigenvalue",
        "pulse_spectral_fraction_in_fit_window",
        "pulse_spectral_leakage",
        "response_tail_ratio",
        "response_tail_tolerance",
        "response_tail_converged",
        "response_tail_window_fraction",
    )
    for name in bare_diagnostic_names:
        dtype, fill = _diagnostic_array_dtype(name)
        diagnostic_payload[f"bare_diagnostic__{name}"] = np.full(n_fluence, fill, dtype=dtype)

    e0_au = np.empty(n_fluence, dtype=float)
    e0_v_m = np.empty(n_fluence, dtype=float)
    peak_intensity = np.empty(n_fluence, dtype=float)
    pulse_area = np.empty(n_fluence, dtype=float)

    if settings["post_fs"] is None:
        reference_pulse = _pulse_for_fluence(float(fluence[0]), settings)
        initial_end_au = max(
            *(item[-1].recommended_post_pulse_time_au() for item in built),
            settings["start_sigma"] * reference_pulse.sigma_t_au,
        )
        automatic_post = True
    else:
        initial_end_au = float(fs_to_au(settings["post_fs"]))
        automatic_post = False

    for fluence_index, target_fluence in enumerate(fluence):
        pulse = _pulse_for_fluence(float(target_fluence), settings)
        e0_au[fluence_index] = pulse.E0_au
        e0_v_m[fluence_index] = float(field_au_to_si(pulse.E0_au))
        peak_intensity[fluence_index] = pulse.peak_intensity_w_cm2(eps_m=settings["eps_m"])
        pulse_area[fluence_index] = float(
            reference_params.qd_local_field_factor
            * reference_params.d_au
            * pulse.E0_au
            * np.sqrt(2.0 * np.pi)
            * pulse.sigma_t_au
        )
        start_au = -settings["start_sigma"] * pulse.sigma_t_au
        end_au = float(initial_end_au)
        full_results = None
        bare_result = None
        extensions = 0
        while True:
            t_span = (float(start_au), float(end_au))
            models = [item[-1] for item in built]
            if settings["workers"] > 1:
                with ThreadPoolExecutor(max_workers=settings["workers"], thread_name_prefix="excitation-fluence") as executor:
                    full_results = list(
                        executor.map(
                            lambda active_model: _solve_full_model(active_model, pulse, t_span, settings),
                            models,
                        )
                    )
            else:
                full_results = [
                    _solve_full_model(active_model, pulse, t_span, settings)
                    for active_model in models
                ]
            bare_result = _solve_bare_qd(pulse, reference_params, t_span, settings)
            all_tails_converged = all(
                result.diagnostics.response_tail_converged for result in full_results
            ) and bool(bare_result[2]["response_tail_converged"])
            if all_tails_converged or not automatic_post or extensions >= settings["max_auto_tail_extensions"]:
                break
            end_au *= 2.0
            extensions += 1

        if full_results is None or bare_result is None:
            raise RuntimeError("Internal error: no pulse solutions were produced.")
        if not all_tails_converged:
            ratios = [result.diagnostics.response_tail_ratio for result in full_results]
            ratios.append(float(bare_result[2]["response_tail_ratio"]))
            _apply_policy(
                settings["tail_policy"],
                "The common read-time window failed the response-tail gate at "
                f"fluence={target_fluence:.6g} J/cm^2; max ratio={max(ratios):.6g}.",
            )

        read_time_fs[fluence_index] = float(au_to_fs(end_au))
        tail_extension_count[fluence_index] = extensions
        population_decay_fraction_at_read[fluence_index] = float(
            -np.expm1(-reference_params.gamma_au * end_au)
        )
        if (
            population_decay_fraction_at_read[fluence_index]
            > settings["max_population_decay_fraction_at_read"]
        ):
            _apply_policy(
                settings["population_decay_policy"],
                "The read time is no longer safely before population "
                "relaxation at "
                f"fluence={target_fluence:.6g} J/cm^2: free-decay fraction="
                f"{population_decay_fraction_at_read[fluence_index]:.6g}, "
                "limit="
                f"{settings['max_population_decay_fraction_at_read']:.6g}.",
            )
        bare_t, bare_rho, bare_diag = bare_result
        bare_max_index = int(np.argmax(bare_rho))
        p_read[0, fluence_index] = float(bare_rho[-1])
        p_max[0, fluence_index] = float(bare_rho[bare_max_index])
        t_max_fs[0, fluence_index] = float(au_to_fs(bare_t[bare_max_index]))
        for name in bare_diagnostic_names:
            diagnostic_payload[f"bare_diagnostic__{name}"][fluence_index] = bare_diag[name]

        for channel_index, result in enumerate(full_results):
            maximum_index = int(np.argmax(result.rho22))
            p_read[channel_index + 1, fluence_index] = float(result.rho22[-1])
            p_max[channel_index + 1, fluence_index] = float(result.rho22[maximum_index])
            t_max_fs[channel_index + 1, fluence_index] = float(au_to_fs(result.t_au[maximum_index]))
            for diagnostic_field in fields(FullQSSolveDiagnostics):
                value = getattr(result.diagnostics, diagnostic_field.name)
                target = diagnostic_payload[f"full_diagnostic__{diagnostic_field.name}"]
                target[channel_index, fluence_index] = np.nan if value is None else value

        print(
            f"[{fluence_index + 1}/{n_fluence}] fluence={target_fluence:.6e} J/cm^2, "
            f"read={read_time_fs[fluence_index]:.3f} fs",
            flush=True,
        )

    # A dependence P_exc(F) is meaningful only when every point is sampled at
    # one physical read time.  Adaptive tail extension can occasionally select
    # a longer window for only part of the fluence grid.  In that uncommon
    # case, repeat once at the largest accepted window; normally this branch is
    # free because every point passes at the shared initial estimate.
    if automatic_post and not np.allclose(
        read_time_fs,
        read_time_fs[0],
        rtol=0.0,
        atol=1.0e-10 * max(abs(float(read_time_fs[0])), 1.0),
    ):
        common_settings = dict(settings)
        common_settings["post_fs"] = float(np.max(read_time_fs))
        common_settings["_automatic_common_read_rerun"] = True
        print(
            "Adaptive tails selected different read times; repeating the "
            f"fluence grid at common t_read={common_settings['post_fs']:.6g} fs.",
            flush=True,
        )
        return calculate_excitation_fluence(common_settings)

    grid_diagnostics = fluence_grid_resolution_diagnostics(
        fluence,
        p_read,
        pulse_area,
        max_midpoint_error=settings["max_fluence_grid_midpoint_error"],
        max_pulse_area_step_rad=settings["max_isolated_pulse_area_step_rad"],
    )
    if not grid_diagnostics["accepted"]:
        errors = np.asarray(
            grid_diagnostics["midpoint_interpolation_error_by_channel"],
            dtype=float,
        )
        failed = ", ".join(
            f"{channel_id}: {error:.4g}"
            for channel_id, error, accepted in zip(
                ALL_CHANNEL_IDS,
                errors,
                grid_diagnostics["accepted_by_channel"],
            )
            if not accepted
        )
        _apply_policy(
            settings["fluence_grid_convergence_policy"],
            "The P_exc(fluence) grid is not publication-resolved in sqrt(fluence): "
            f"midpoint errors [{failed}], allowed="
            f"{settings['max_fluence_grid_midpoint_error']:.4g}; maximum isolated-QD "
            f"pulse-area step={grid_diagnostics['maximum_isolated_pulse_area_step_rad']:.4g} "
            f"rad, allowed={settings['max_isolated_pulse_area_step_rad']:.4g} rad. "
            "Increase --points or narrow the fluence interval.",
        )

    fit_strengths = np.stack([item[2].fit.strengths_au2 for item in built])
    fit_omega = np.stack([item[2].fit.omega_modes_au for item in built])
    fit_gamma = np.stack([item[2].fit.gamma_modes_au for item in built])
    fit_alpha_inf = np.asarray([item[2].fit.alpha_inf for item in built], dtype=float)

    exact_offsets = [0]
    exact_L: list[np.ndarray] = []
    exact_w: list[np.ndarray] = []
    dynamic_offsets = [0]
    dynamic_L: list[np.ndarray] = []
    dynamic_w: list[np.ndarray] = []
    for _, _, _, kernel, _, model in built:
        kernel_L = np.asarray(
            kernel.depolarization_by_mode if hasattr(kernel, "depolarization_by_mode") else kernel.depolarization_by_degree,
            dtype=float,
        )
        kernel_w = np.asarray(
            kernel.reaction_weight_by_mode_au_minus3 if hasattr(kernel, "reaction_weight_by_mode_au_minus3") else kernel.reaction_weight_by_degree_au_minus3,
            dtype=float,
        )
        exact_L.append(kernel_L)
        exact_w.append(kernel_w)
        exact_offsets.append(exact_offsets[-1] + kernel_L.size)
        dynamic_L.append(np.asarray(model.modal_depolarization, dtype=float))
        dynamic_w.append(np.asarray(model.modal_reaction_weights_au_minus3, dtype=float))
        dynamic_offsets.append(dynamic_offsets[-1] + model.n_spatial_modes)

    payload: dict[str, np.ndarray] = {
        "channel_id": ALL_CHANNEL_IDS,
        "hybrid_channel_id": np.asarray([channel.channel_id for channel in HYBRID_CHANNELS], dtype="U32"),
        "fluence_j_cm2": fluence,
        "pulse_e0_au": e0_au,
        "pulse_e0_v_m": e0_v_m,
        "peak_intensity_w_cm2": peak_intensity,
        "isolated_qd_pulse_area_rad": pulse_area,
        "p_exc_read": p_read,
        "p_exc_max": p_max,
        "time_of_p_exc_max_fs": t_max_fs,
        "read_time_fs": read_time_fs,
        "tail_extension_count": tail_extension_count,
        "population_decay_fraction_at_read": population_decay_fraction_at_read,
        "fluence_grid_midpoint_interpolation_error": np.asarray(
            grid_diagnostics["midpoint_interpolation_error_by_channel"], dtype=float
        ),
        "fluence_grid_converged_by_channel": np.asarray(
            grid_diagnostics["accepted_by_channel"], dtype=bool
        ),
        "maximum_isolated_pulse_area_step_rad": np.asarray(
            grid_diagnostics["maximum_isolated_pulse_area_step_rad"], dtype=float
        ),
        "material_energy_eV": np.asarray(reference_params.material.energy_eV),
        "material_n": np.asarray(reference_params.material.n),
        "material_k": np.asarray(reference_params.material.k),
        "fit_alpha_inf": fit_alpha_inf,
        "fit_strengths_au2": fit_strengths,
        "fit_omega_modes_au": fit_omega,
        "fit_gamma_modes_au": fit_gamma,
        "fit_omega_modes_eV": np.asarray(au_to_eV(fit_omega)),
        "fit_gamma_modes_eV": np.asarray(au_to_eV(fit_gamma)),
        "exact_spatial_mode_offsets": np.asarray(exact_offsets, dtype=np.int64),
        "exact_spatial_depolarization": np.concatenate(exact_L),
        "exact_spatial_reaction_weight_au_minus3": np.concatenate(exact_w),
        "dynamic_spatial_mode_offsets": np.asarray(dynamic_offsets, dtype=np.int64),
        "dynamic_spatial_depolarization": np.concatenate(dynamic_L),
        "dynamic_spatial_reaction_weight_au_minus3": np.concatenate(dynamic_w),
        **diagnostic_payload,
    }

    array_units = {
        "channel_id": "text",
        "hybrid_channel_id": "text",
        "fluence_j_cm2": "J cm^-2",
        "pulse_e0_au": "atomic unit of electric field",
        "pulse_e0_v_m": "V m^-1",
        "peak_intensity_w_cm2": "W cm^-2",
        "isolated_qd_pulse_area_rad": "rad",
        "p_exc_read": "1",
        "p_exc_max": "1",
        "time_of_p_exc_max_fs": "fs",
        "read_time_fs": "fs",
        "tail_extension_count": "1",
        "population_decay_fraction_at_read": "1",
        "fluence_grid_midpoint_interpolation_error": "absolute population",
        "fluence_grid_converged_by_channel": "boolean",
        "maximum_isolated_pulse_area_step_rad": "rad",
        "material_energy_eV": "eV",
        "material_n": "1",
        "material_k": "1",
        "fit_alpha_inf": "1",
        "fit_strengths_au2": "atomic frequency squared",
        "fit_omega_modes_au": "atomic angular frequency",
        "fit_gamma_modes_au": "atomic angular frequency",
        "fit_omega_modes_eV": "eV (hbar omega)",
        "fit_gamma_modes_eV": "eV (hbar gamma)",
        "exact_spatial_mode_offsets": "index",
        "exact_spatial_depolarization": "1",
        "exact_spatial_reaction_weight_au_minus3": "bohr^-3",
        "dynamic_spatial_mode_offsets": "index",
        "dynamic_spatial_depolarization": "1",
        "dynamic_spatial_reaction_weight_au_minus3": "bohr^-3",
    }
    for diagnostic_field in fields(FullQSSolveDiagnostics):
        array_units[f"full_diagnostic__{diagnostic_field.name}"] = _diagnostic_units(diagnostic_field.name)
    for name in bare_diagnostic_names:
        array_units[f"bare_diagnostic__{name}"] = _diagnostic_units(name)

    metadata = _json_ready(
        {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now().astimezone().isoformat(),
            "purpose": "post-pulse exciton population versus incident pulse fluence",
            "model_scope": {
                "hybrid_model": "FullQSSpheroidPulseModel only",
                "isolated_control": "same non-RWA TLS Bloch equations with MNP field exactly zero",
                "electromagnetic_regime": "local quasistatic axisymmetric spheroid; point-dipole QD",
                "legacy_model_used": False,
                "native_model_profile": NATIVE_MODEL_PROFILE,
                "material_interpolation": MATERIAL_INTERPOLATION,
                "material_high_frequency_epsilon": (
                    MATERIAL_HIGH_FREQUENCY_EPSILON
                ),
                "initial_bloch_state_W_Q_P": [-1.0, 0.0, 0.0],
                "pulse_envelope_center_fs": 0.0,
                "pulse_carrier_phase_rad": 0.0,
                "side_direct_reference_order_max": (
                    SIDE_DIRECT_REFERENCE_ORDER_MAX
                ),
            },
            "observable": {
                "p_exc_read": "rho_ee at the common final read time after the pulse/response tail",
                "p_exc_max": "maximum rho_ee over the propagated trajectory; diagnostic, not the primary article observable",
                "fluence": "integral of n_m epsilon_0 c E(t)^2 over time for the real Gaussian carrier",
                "fluence_grid_convergence": (
                    "each odd-index population is compared with interpolation "
                    "between the neighboring even-index points in sqrt(fluence); "
                    "the isolated-QD pulse-area step is gated separately"
                ),
            },
            "fluence_grid_diagnostics": grid_diagnostics,
            "time_window": {
                "requested_post_fs": settings["_requested_cli_arguments"].get(
                    "post_fs"
                ),
                "post_was_automatic": bool(
                    settings["_requested_cli_arguments"].get("post_fs") is None
                ),
                "common_read_time_fs": float(read_time_fs[0]),
                "all_fluences_share_read_time": bool(
                    np.allclose(
                        read_time_fs,
                        read_time_fs[0],
                        rtol=0.0,
                        atol=1.0e-10 * max(abs(float(read_time_fs[0])), 1.0),
                    )
                ),
                "automatic_common_read_rerun": bool(
                    settings.get("_automatic_common_read_rerun", False)
                ),
            },
            "requested_cli_arguments": settings["_requested_cli_arguments"],
            "resolved_settings": {
                key: value
                for key, value in settings.items()
                if not key.startswith("_")
            },
            "channels": [
                {
                    "channel_id": "bare_qd",
                    "label": "isolated QD",
                    "qd_placement": None,
                    "orientation": None,
                    "side_transverse_alignment": None,
                },
                *[
                    _model_metadata(
                        channel, params, bright, kernel, reduction, model, settings
                    )
                    for channel, params, bright, kernel, reduction, model in built
                ],
            ],
            "array_units": array_units,
            "fundamental_and_conversion_constants": {
                "AU_LENGTH_M": AU_LENGTH_M,
                "AU_TIME_S": AU_TIME_S,
                "AU_ENERGY_J": AU_ENERGY_J,
                "AU_ENERGY_EV": AU_ENERGY_EV,
                "AU_FIELD_V_M": AU_FIELD_V_M,
                "AU_DIPOLE_C_M": AU_DIPOLE_C_M,
                "DEBYE_C_M": DEBYE_C_M,
                "elementary_charge_C": E_CHARGE,
                "vacuum_permittivity_F_m": EPSILON_0_SI,
                "reduced_Planck_constant_J_s": HBAR_SI,
                "speed_of_light_m_s": C_SI,
                "atomic_unit_speed": AU_SPEED_OF_LIGHT,
                "pi": np.pi,
            },
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "project_version": _project_version(),
                "calculation_script": Path(__file__).resolve().relative_to(
                    PROJECT_ROOT
                ).as_posix(),
                "git": _git_provenance(),
                "source_file_sha256": _source_file_hashes(),
            },
        }
    )
    return payload, metadata


def write_excitation_artifact(
    path: str | Path,
    payload: dict[str, np.ndarray],
    metadata: dict[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write one pickle-free NPZ containing arrays and JSON metadata."""

    output = Path(path)
    if output.suffix.lower() != ".npz":
        raise ValueError("The output artifact must have a .npz suffix.")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {output}")
    document = dict(metadata)
    document["artifact_file"] = output.name
    document["array_keys"] = sorted([*payload, "metadata_json"])
    metadata_json = json.dumps(_json_ready(document), ensure_ascii=False, sort_keys=True, allow_nan=False)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{output.stem}_", suffix=".npz", dir=output.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, metadata_json=np.asarray(metadata_json), **payload)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser so comparison orchestrators can reuse it exactly."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/excitation_fluence.npz"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preset", choices=("publication", "quick"), default="publication")
    parser.add_argument("--fluence-grid-j-cm2", nargs="+", type=float)
    parser.add_argument("--fluence-min-j-cm2", type=float)
    parser.add_argument("--fluence-max-j-cm2", type=float)
    parser.add_argument("--points", type=int)
    parser.add_argument("--grid-scale", choices=("log", "linear", "sqrt"))

    parser.add_argument("--pulse-energy-ev", type=float, default=2.042)
    parser.add_argument("--pulse-tau-fs", type=float, default=20.0)
    parser.add_argument("--pulse-tau-kind", choices=("sigma", "fwhm_intensity"), default="fwhm_intensity")
    parser.add_argument("--c-nm", type=float, default=15.0)
    parser.add_argument("--a-nm", type=float, default=7.0)
    parser.add_argument("--qd-radius-nm", type=float, default=2.0)
    parser.add_argument("--gap-nm", type=float, default=1.0)
    parser.add_argument("--eps-m", type=float, default=1.0)
    parser.add_argument("--eps-qd", type=float, default=6.0)
    parser.add_argument("--d-debye", type=float)
    parser.add_argument("--omega0-ev", type=float, default=2.042)
    parser.add_argument("--gamma-population-mev", type=float)
    parser.add_argument("--gamma2-coherence-mev", type=float)
    parser.add_argument("--qd-dipole-convention", choices=("effective_external", "bare_internal"), default="effective_external")

    parser.add_argument("--spatial-order-max", type=int)
    parser.add_argument("--material-fit-modes", type=int)
    parser.add_argument("--fit-min-ev", type=float, default=0.8)
    parser.add_argument("--fit-max-ev", type=float, default=3.0)
    parser.add_argument("--weight-center-ev", type=float)
    parser.add_argument("--weight-sigma-ev", type=float)
    parser.add_argument("--max-bright-fit-normalized-rms", type=float, default=0.025)
    parser.add_argument(
        "--max-bright-fit-pointwise-relative-error", type=float, default=0.05
    )
    parser.add_argument("--modal-audit-points", type=int)
    parser.add_argument("--reduction-fit-grid-points", type=int)
    parser.add_argument("--reduction-audit-grid-points", type=int)
    parser.add_argument("--reduction-reaudit-points", type=int, default=1709)
    parser.add_argument("--reduction-rms-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--reduction-max-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--reduction-max-nodes", type=int)

    parser.add_argument("--method", choices=("DOP853", "RK45", "Radau", "BDF", "LSODA"), default="DOP853")
    parser.add_argument("--rtol", type=float, default=1.0e-8)
    parser.add_argument("--atol", type=float, default=1.0e-10)
    parser.add_argument("--points-per-fastest-cycle", type=int, default=20)
    parser.add_argument("--start-sigma", type=float, default=10.0)
    parser.add_argument("--post-fs", type=float)
    parser.add_argument("--max-auto-tail-extensions", type=int, default=2)
    parser.add_argument("--tail-ratio-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--tail-window-fraction", type=float, default=0.05)
    parser.add_argument(
        "--max-population-decay-fraction-at-read", type=float, default=1.0e-3
    )
    parser.add_argument("--max-isolated-pulse-area-step-rad", type=float)
    parser.add_argument("--max-fluence-grid-midpoint-error", type=float)
    parser.add_argument("--max-spectral-leakage", type=float, default=1.0e-3)
    parser.add_argument("--positivity-tolerance", type=float, default=1.0e-7)
    parser.add_argument("--max-modal-normalized-rms", type=float, default=0.03)
    parser.add_argument("--max-modal-relative-error", type=float, default=0.06)
    parser.add_argument("--spatial-convergence-rtol", type=float, default=2.0e-5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--verbose-fit", action="store_true")

    parser.add_argument("--radiative-consistency-policy", choices=POLICIES, default="warn")
    parser.add_argument("--spectral-window-policy", choices=POLICIES)
    parser.add_argument("--positivity-policy", choices=POLICIES, default="raise")
    parser.add_argument("--tail-policy", choices=POLICIES)
    parser.add_argument("--work-passivity-policy", choices=POLICIES, default="raise")
    parser.add_argument("--fit-quality-policy", choices=POLICIES)
    parser.add_argument("--bright-fit-quality-policy", choices=POLICIES)
    parser.add_argument("--population-decay-policy", choices=POLICIES)
    parser.add_argument("--fluence-grid-convergence-policy", choices=POLICIES)
    parser.add_argument("--spatial-convergence-policy", choices=POLICIES)
    parser.add_argument("--reduction-policy", choices=POLICIES, default="raise")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_argument_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    settings = _resolved_settings(args)
    payload, metadata = calculate_excitation_fluence(settings)
    output = write_excitation_artifact(args.output, payload, metadata, overwrite=args.overwrite)
    print(f"Saved self-contained excitation-fluence artifact: {output}")
    return output


if __name__ == "__main__":
    main()
