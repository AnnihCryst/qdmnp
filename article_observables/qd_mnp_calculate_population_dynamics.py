"""Calculate article-ready QD population dynamics and save one NPZ artifact.

This article executable performs the expensive full-quasistatic pulse
propagation. It does not create figures. The companion
``qd_mnp_plot_population_dynamics.py``
loads only the resulting NPZ file, so figure styling never repeats the physical
calculation.
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
from typing import Iterable
import warnings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import scipy
from scipy.constants import c as C_SI, elementary_charge, epsilon_0, hbar
from scipy.integrate import solve_ivp

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
    AU_SPEED_OF_LIGHT,
    AU_TIME_S,
    DEBYE_C_M,
    DEFAULT_AU_MATERIAL,
    MATERIAL_HIGH_FREQUENCY_EPSILON,
    MATERIAL_INTERPOLATION,
    GaussianPulse,
    HybridQDPlasmonModel,
    au_to_fs,
    eV_to_au,
    fs_to_au,
    make_params_with_overrides,
    params_to_physical_dict,
    response_tail_ratio,
)
from qd_mnp_spheroid_equatorial import EquatorialSpheroidGreenInteraction
from qd_mnp_spheroid_green import SpheroidGreenInteraction


SCHEMA_NAME = "qd_mnp_population_dynamics"
SCHEMA_VERSION = 1
SIDE_DIRECT_REFERENCE_ORDER_MAX = 8


@dataclass(frozen=True)
class ChannelSpec:
    placement: str
    orientation: str
    side_alignment: str | None
    label: str


CHANNELS: dict[str, ChannelSpec] = {
    "axis_long": ChannelSpec(
        "axis", "long", None, r"tip, $E\parallel z$"
    ),
    "axis_trans": ChannelSpec(
        "axis", "trans", None, r"tip, $E\perp z$"
    ),
    "side_long": ChannelSpec(
        "side", "long", None, r"side, $E\parallel z$"
    ),
    "side_trans_radial": ChannelSpec(
        "side", "trans", "radial", r"side, transverse radial"
    ),
    "side_trans_tangential": ChannelSpec(
        "side", "trans", "tangential", r"side, transverse tangential"
    ),
}


def _surface_gap_separation_nm(
    spec: ChannelSpec,
    *,
    c_nm: float,
    a_nm: float,
    qd_radius_nm: float,
    gap_nm: float,
) -> float:
    directional_radius = c_nm if spec.placement == "axis" else a_nm
    return float(directional_radius + qd_radius_nm + gap_nm)


def _pulse_for_fluence(
    *,
    fluence_j_cm2: float,
    pulse_energy_eV: float,
    pulse_tau_fs: float,
    pulse_tau_kind: str,
    eps_m: float,
) -> GaussianPulse:
    if not np.isfinite(fluence_j_cm2) or fluence_j_cm2 <= 0.0:
        raise ValueError("fluence_j_cm2 must be finite and positive.")
    unit = GaussianPulse(
        E0_au=1.0,
        omegaL_au=float(eV_to_au(pulse_energy_eV)),
        tau_au=float(fs_to_au(pulse_tau_fs)),
        tau_kind=pulse_tau_kind,
    )
    unit_fluence = unit.fluence_j_cm2(eps_m=eps_m)
    return GaussianPulse(
        E0_au=float(np.sqrt(fluence_j_cm2 / unit_fluence)),
        omegaL_au=unit.omegaL_au,
        tau_au=unit.tau_au,
        tau_kind=unit.tau_kind,
    )


def _git_provenance() -> dict[str, object]:
    result: dict[str, object] = {"commit": None, "dirty": None}
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
    result["dirty"] = bool(status.strip())
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


def _fit_metadata(model: HybridQDPlasmonModel) -> dict[str, object]:
    fit = model.fit
    return {
        "fit_window_eV": [float(value) for value in model.fit_window_eV],
        "n_material_modes": int(model.n_modes),
        "alpha_inf": float(fit.alpha_inf),
        "strengths_au2": [float(value) for value in fit.strengths_au2],
        "omega_modes_au": [float(value) for value in fit.omega_modes_au],
        "gamma_modes_au": [float(value) for value in fit.gamma_modes_au],
        "normalized_rms_alpha": float(fit.normalized_rms_alpha),
        "normalized_rms_inv_alpha": float(fit.normalized_rms_inv_alpha),
        "max_normalized_alpha_error": float(fit.max_normalized_alpha_error),
        "minimum_imaginary_alpha_fit_window": float(
            fit.min_imag_alpha_fit_window
        ),
        "passivity_grid_points": int(fit.passivity_grid_points),
        "passive_on_fit_window": bool(fit.passive_on_fit_window),
        "globally_passive": bool(
            fit.nonnegative_imaginary_part_all_positive_frequencies
        ),
    }


def _reduction_metadata(reduction, model) -> dict[str, object]:
    if reduction is None:
        return {"applied": False, "current_transfer_reaudit": None}
    diagnostics = reduction.diagnostics
    reaudit = model.dark_reduction_reaudit_diagnostics
    if reaudit is None:
        raise RuntimeError(
            "An applied dark-mode reduction lacks its active-transfer re-audit."
        )
    return {
        "applied": True,
        "source_measure_sha256": reduction.source_measure_sha256,
        "depolarization_nodes": [
            float(value) for value in reduction.depolarization_nodes
        ],
        "weights_au_minus3": [
            float(value) for value in reduction.weights_au_minus3
        ],
        "source_mode_indices": [
            [int(index) for index in group]
            for group in reduction.source_mode_indices
        ],
        "accepted": bool(diagnostics.accepted),
        "passive_on_audit_grid": bool(diagnostics.passive_on_audit_grid),
        "fit_normalized_rms": float(diagnostics.fit_normalized_rms),
        "fit_max_normalized_error": float(
            diagnostics.fit_max_normalized_error
        ),
        "audit_normalized_rms": float(diagnostics.audit_normalized_rms),
        "audit_max_normalized_error": float(
            diagnostics.audit_max_normalized_error
        ),
        "max_normalized_rms": float(diagnostics.max_normalized_rms),
        "max_normalized_error": float(diagnostics.max_normalized_error),
        "total_weight_relative_error": float(
            diagnostics.total_weight_relative_error
        ),
        "first_moment_relative_error": float(
            diagnostics.first_moment_relative_error
        ),
        "fit_grid_points": int(diagnostics.fit_grid_points),
        "audit_grid_points": int(diagnostics.audit_grid_points),
        "original_dark_mode_count": int(diagnostics.original_dark_mode_count),
        "positive_dark_mode_count": int(diagnostics.positive_dark_mode_count),
        "reduced_node_count": int(diagnostics.reduced_node_count),
        "rms_tolerance": float(diagnostics.rms_tolerance),
        "max_tolerance": float(diagnostics.max_tolerance),
        "current_transfer_reaudit": {
            "accepted": bool(reaudit.accepted),
            "passive_on_audit_grid": bool(reaudit.passive_on_audit_grid),
            "normalized_rms": float(reaudit.normalized_rms),
            "max_normalized_error": float(reaudit.max_normalized_error),
            "minimum_imaginary_part": float(reaudit.minimum_imaginary_part),
            "audit_grid_points": int(reaudit.audit_grid_points),
            "rms_tolerance": float(reaudit.rms_tolerance),
            "max_tolerance": float(reaudit.max_tolerance),
            "energy_window_eV": [
                float(value) for value in reaudit.energy_window_eV
            ],
        },
    }


def _full_qs_certificate_metadata(model) -> dict[str, object]:
    """Serialize construction-time full-QS certificates used by the plot gate."""

    spatial = model.spatial_convergence_diagnostics
    modal = model.modal_fit_diagnostics
    stability = model.coupled_stability
    return {
        "spatial_convergence": {
            "accepted": bool(spatial.accepted),
            "tolerance": float(spatial.tolerance),
            "max_half_order_relative_change": float(
                spatial.max_half_order_relative_change
            ),
            "max_tail_block_relative_mass": float(
                spatial.max_tail_block_relative_mass
            ),
            "audit_grid_points": int(spatial.audit_grid_points),
            "energy_window_eV": [
                float(value) for value in spatial.energy_window_eV
            ],
        },
        "modal_transform": {
            "accepted": bool(modal.accepted),
            "passive_on_audit_grid": bool(modal.passive_on_audit_grid),
            "max_normalized_rms": float(modal.max_normalized_rms),
            "max_relative_error": float(modal.max_relative_error),
            "K_normalized_rms": float(modal.K_normalized_rms),
            "K_max_relative_error": float(modal.K_max_relative_error),
            "audit_grid_points": int(modal.audit_grid_points),
        },
        "coupled_stability": {
            "stable": bool(stability.stable),
            "spectral_abscissa_au": (
                None
                if stability.spectral_abscissa_au is None
                else float(stability.spectral_abscissa_au)
            ),
            "spectral_abscissa_available": bool(
                stability.spectral_abscissa_available
            ),
            "spectral_abscissa_is_bound": bool(
                stability.spectral_abscissa_is_bound
            ),
            "decay_rate_estimate_au": float(stability.decay_rate_estimate_au),
            "decay_rate_estimate_is_exact": bool(
                stability.decay_rate_estimate_is_exact
            ),
            "spectral_radius_au": float(stability.spectral_radius_au),
            "tolerance_au": float(stability.tolerance_au),
            "eigensolver": str(stability.eigensolver),
            "coherent_state_dimension": int(
                stability.coherent_state_dimension
            ),
        },
    }


def _build_full_qs_model(spec: ChannelSpec, args: argparse.Namespace):
    separation_nm = _surface_gap_separation_nm(
        spec,
        c_nm=args.c_nm,
        a_nm=args.a_nm,
        qd_radius_nm=args.qd_radius_nm,
        gap_nm=args.gap_nm,
    )
    params = make_params_with_overrides(
        c_nm=args.c_nm,
        a_nm=args.a_nm,
        r_nm=separation_nm,
        qd_radius_nm=args.qd_radius_nm,
        eps_m=args.eps_m,
        eps_qd=args.eps_qd,
        d_debye=args.d_debye,
        omega0_ev=args.omega0_ev,
        gamma_population_mev=args.gamma_population_mev,
        gamma2_coherence_mev=args.gamma2_coherence_mev,
        qd_dipole_convention=args.qd_dipole_convention,
        orientation=spec.orientation,
        qd_placement=spec.placement,
        side_transverse_alignment=spec.side_alignment,
    )
    bright_model = HybridQDPlasmonModel(
        params,
        orientation=spec.orientation,
        n_modes=args.material_fit_modes,
        fit_window_eV=(args.fit_min_ev, args.fit_max_ev),
        alpha_objective_weight=args.alpha_objective_weight,
        inv_alpha_objective_weight=args.inv_alpha_objective_weight,
        max_fit_normalized_rms=None,
        max_fit_pointwise_relative_error=None,
        radiative_consistency_policy=args.radiative_consistency_policy,
        seed=args.fit_seed,
        verbose=False,
    )
    fit = bright_model.fit
    if (
        fit.normalized_rms_alpha > args.max_bright_fit_normalized_rms
        or fit.normalized_rms_inv_alpha > args.max_bright_fit_normalized_rms
        or fit.max_normalized_alpha_error
        > args.max_bright_fit_pointwise_relative_error
    ):
        _apply_policy(
            args.bright_fit_quality_policy,
            "The deliberately selected material Lorentz fit misses the "
            "configured accuracy gate: "
            f"NRMS(alpha)={fit.normalized_rms_alpha:.6g}, "
            f"NRMS(1/alpha)={fit.normalized_rms_inv_alpha:.6g}, "
            f"max normalized alpha error={fit.max_normalized_alpha_error:.6g}; "
            "this is expected for a one-oscillator control but must be "
            "reported as an approximation, not as an accurate Au fit.",
        )
    if spec.placement == "side":
        kernel = EquatorialSpheroidGreenInteraction.from_params(
            params,
            orientation=spec.orientation,
            n_max=args.spatial_order_max,
        )
    else:
        kernel = SpheroidGreenInteraction.from_params(
            params,
            orientation=spec.orientation,
            n_max=args.spatial_order_max,
        )

    reduction = None
    if (
        spec.placement == "side"
        and args.spatial_order_max > SIDE_DIRECT_REFERENCE_ORDER_MAX
    ):
        reduction = build_positive_dark_reduction(
            bright_model,
            kernel,
            fit_grid_points=args.reduction_fit_grid_points,
            audit_grid_points=args.reduction_audit_grid_points,
            rms_tolerance=args.reduction_rms_tolerance,
            max_tolerance=args.reduction_max_tolerance,
            max_nodes=args.reduction_max_nodes,
            policy=args.reduction_policy,
        )
        if not reduction.diagnostics.accepted:
            raise RuntimeError(
                f"The dark-kernel reduction for {spec.label} was not accepted."
            )

    full_model = FullQSSpheroidPulseModel(
        bright_model,
        kernel,
        dark_reduction=reduction,
        fit_quality_policy=args.fit_quality_policy,
        max_modal_normalized_rms=args.max_modal_normalized_rms,
        max_modal_relative_error=args.max_modal_relative_error,
        modal_audit_points=args.modal_audit_points,
        spatial_convergence_policy=args.spatial_convergence_policy,
        spatial_convergence_rtol=args.spatial_convergence_rtol,
        reduction_reaudit_points=args.reduction_reaudit_points,
        max_reduction_normalized_rms=args.reduction_rms_tolerance,
        max_reduction_normalized_error=args.reduction_max_tolerance,
    )
    return params, bright_model, kernel, full_model, reduction


def _solve_isolated_qd(
    *,
    pulse: GaussianPulse,
    params,
    t_span_au: tuple[float, float],
    method: str,
    rtol: float,
    atol: float,
    points_per_fastest_cycle: float,
    response_tail_window_fraction: float,
    positivity_policy: str,
    positivity_tolerance: float,
) -> dict[str, object]:
    local = params.qd_local_field_factor

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
        pulse.omegaL_au,
        params.omega0_au,
        abs(2.0 * params.d_au * local * pulse.E0_au),
    )
    max_step = 2.0 * np.pi / (
        points_per_fastest_cycle * frequency_ceiling
    )
    solution = solve_ivp(
        rhs,
        t_span_au,
        np.asarray([-1.0, 0.0, 0.0]),
        method=method,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        dense_output=False,
    )
    if not solution.success:
        raise RuntimeError(f"The isolated-QD integration failed: {solution.message}")
    W, Q, P = solution.y
    rho22 = 0.5 * (W + 1.0)
    bloch_radius = np.sqrt(W**2 + Q**2 + P**2)
    min_density_eigenvalue = float(0.5 * (1.0 - np.max(bloch_radius)))
    if min_density_eigenvalue < -positivity_tolerance:
        _apply_policy(
            positivity_policy,
            "The isolated-QD trajectory left the Bloch ball: minimum density "
            f"eigenvalue={min_density_eigenvalue:.6g}, tolerance="
            f"{positivity_tolerance:.6g}.",
        )
    time_au = solution.t
    mu_qd = local * params.d_au * P
    coherence_tail = response_tail_ratio(
        np.hypot(Q, P),
        time_au,
        tail_fraction=response_tail_window_fraction,
    )
    dipole_tail = response_tail_ratio(
        mu_qd,
        time_au,
        tail_fraction=response_tail_window_fraction,
    )
    return {
        "t_au": time_au,
        "W": W,
        "Q": Q,
        "P": P,
        "rho22": rho22,
        "mu_qd_au": mu_qd,
        "mu_mnp_au": np.zeros_like(time_au),
        "mu_total_au": local * params.d_au * P,
        "mnp_field_at_qd_au": np.zeros_like(time_au),
        "effective_qd_field_au": local * pulse.field(time_au),
        "diagnostics": {
            "solver_success": bool(solution.success),
            "solver_status": int(solution.status),
            "solver_message": str(solution.message),
            "solver_n_steps": int(solution.t.size),
            "solver_nfev": int(solution.nfev),
            "max_step_limit_au": float(max_step),
            "t_final_reached": bool(
                solution.t.size > 0
                and np.isclose(
                    solution.t[-1],
                    t_span_au[1],
                    rtol=1.0e-12,
                    atol=1.0e-12,
                )
            ),
            "state_is_finite": bool(
                np.all(np.isfinite(solution.t))
                and np.all(np.isfinite(solution.y))
            ),
            "excited_population_min": float(np.min(rho22)),
            "excited_population_max": float(np.max(rho22)),
            "max_bloch_radius": float(np.max(bloch_radius)),
            "min_density_eigenvalue": min_density_eigenvalue,
            "response_tail_ratio": float(max(coherence_tail, dipole_tail)),
            "response_tail_tolerance": None,
            "response_tail_converged": None,
        },
    }


def _apply_policy(policy: str, message: str) -> None:
    if policy == "raise":
        raise RuntimeError(message)
    if policy == "warn":
        warnings.warn(message, RuntimeWarning, stacklevel=3)


def calculate_population_dynamics(args: argparse.Namespace) -> Path:
    output = args.output.resolve()
    if output.suffix.lower() != ".npz":
        raise ValueError("--output must have the .npz extension.")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {output}")
    if not args.channels:
        raise ValueError("At least one hybrid channel is required.")
    if len(set(args.channels)) != len(args.channels):
        raise ValueError("--channels must not contain duplicates.")
    if args.common_time_points < 101:
        raise ValueError("common_time_points must be at least 101.")
    if args.start_sigma < 6.0:
        raise ValueError("start_sigma must be at least 6.")
    if args.post_fs is not None and args.post_fs <= 0.0:
        raise ValueError("post_fs must be positive.")
    if args.gap_nm <= 0.0:
        raise ValueError("gap_nm must be positive.")
    if args.fit_min_ev >= args.fit_max_ev:
        raise ValueError("fit_min_ev must be smaller than fit_max_ev.")
    if args.max_auto_tail_extensions < 0:
        raise ValueError("max_auto_tail_extensions must be non-negative.")
    if args.points_per_fastest_cycle <= 0.0:
        raise ValueError("points_per_fastest_cycle must be positive.")
    for name in (
        "max_bright_fit_normalized_rms",
        "max_bright_fit_pointwise_relative_error",
        "max_population_decay_fraction_at_read",
    ):
        value = float(getattr(args, name))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    if args.max_population_decay_fraction_at_read >= 1.0:
        raise ValueError(
            "max_population_decay_fraction_at_read must be smaller than 1."
        )
    if not 0.0 < args.response_tail_window_fraction <= 1.0:
        raise ValueError("response_tail_window_fraction must lie in (0, 1].")

    pulse = _pulse_for_fluence(
        fluence_j_cm2=args.fluence_j_cm2,
        pulse_energy_eV=args.pulse_energy_ev,
        pulse_tau_fs=args.pulse_tau_fs,
        pulse_tau_kind=args.pulse_tau_kind,
        eps_m=args.eps_m,
    )
    built_models: dict[str, tuple] = {}
    for channel_id in args.channels:
        built_models[channel_id] = _build_full_qs_model(
            CHANNELS[channel_id], args
        )

    reference_params = built_models[args.channels[0]][0]
    start_au = -args.start_sigma * pulse.sigma_t_au
    automatic_post = args.post_fs is None
    if automatic_post:
        end_au = max(
            args.start_sigma * pulse.sigma_t_au,
            *(
                built_models[channel_id][3].recommended_post_pulse_time_au()
                for channel_id in args.channels
            ),
        )
    else:
        end_au = float(fs_to_au(args.post_fs))
    if end_au <= args.start_sigma * pulse.sigma_t_au:
        raise ValueError(
            "post_fs must leave both pulse boundaries at least start_sigma "
            "Gaussian widths from the pulse centre."
        )

    tail_extension_count = 0
    while True:
        t_span_au = (float(start_au), float(end_au))
        bare = _solve_isolated_qd(
            pulse=pulse,
            params=reference_params,
            t_span_au=t_span_au,
            method=args.method,
            rtol=args.rtol,
            atol=args.atol,
            points_per_fastest_cycle=args.points_per_fastest_cycle,
            response_tail_window_fraction=args.response_tail_window_fraction,
            positivity_policy=args.positivity_policy,
            positivity_tolerance=args.positivity_tolerance,
        )
        full_results = {}
        for channel_id in args.channels:
            model = built_models[channel_id][3]
            full_results[channel_id] = model.solve(
                pulse,
                t_span_au=t_span_au,
                method=args.method,
                rtol=args.rtol,
                atol=args.atol,
                points_per_fastest_cycle=args.points_per_fastest_cycle,
                spectral_window_policy=args.spectral_window_policy,
                max_spectral_leakage=args.max_spectral_leakage,
                positivity_policy=args.positivity_policy,
                positivity_tolerance=args.positivity_tolerance,
                work_passivity_policy=args.work_passivity_policy,
                response_tail_policy="ignore",
                response_tail_tolerance=args.response_tail_tolerance,
                response_tail_window_fraction=args.response_tail_window_fraction,
            )
        tail_ratios = {
            "bare_qd": float(bare["diagnostics"]["response_tail_ratio"]),
            **{
                channel_id: float(result.diagnostics.response_tail_ratio)
                for channel_id, result in full_results.items()
            },
        }
        all_tails_converged = all(
            np.isfinite(value) and value <= args.response_tail_tolerance
            for value in tail_ratios.values()
        )
        if (
            not automatic_post
            or all_tails_converged
            or tail_extension_count >= args.max_auto_tail_extensions
        ):
            break
        end_au *= 2.0
        tail_extension_count += 1

    if not all_tails_converged:
        details = ", ".join(
            f"{name}={value:.6g}" for name, value in tail_ratios.items()
        )
        _apply_policy(
            args.response_tail_policy,
            "The common readout window did not satisfy the response-tail "
            f"tolerance {args.response_tail_tolerance:g}: {details}.",
        )

    decay_fraction_at_read = float(
        -np.expm1(-reference_params.gamma_au * max(float(end_au), 0.0))
    )
    if decay_fraction_at_read > args.max_population_decay_fraction_at_read:
        _apply_policy(
            args.population_decay_policy,
            "The common read time is no longer safely before population "
            "relaxation: the free-decay fraction from the pulse centre is "
            f"{decay_fraction_at_read:.6g}, above the configured limit "
            f"{args.max_population_decay_fraction_at_read:.6g}.",
        )

    step_limits = [
        float(bare["diagnostics"]["max_step_limit_au"]),
        *(
            float(result.diagnostics.max_step_limit_au)
            for result in full_results.values()
        ),
    ]
    required_common_time_points = int(
        np.ceil((end_au - start_au) / min(step_limits)) + 1
    )
    resolved_common_time_points = max(
        int(args.common_time_points), required_common_time_points
    )
    time_au = np.linspace(start_au, end_au, resolved_common_time_points)

    channel_ids = ["bare_qd", *args.channels]
    labels = ["isolated QD", *(CHANNELS[name].label for name in args.channels)]
    trace_names = (
        "W",
        "Q",
        "P",
        "rho22",
        "mu_qd_au",
        "mu_mnp_au",
        "mu_total_au",
        "mnp_field_at_qd_au",
        "effective_qd_field_au",
    )
    collected: dict[str, list[np.ndarray]] = {
        name: [
            np.interp(time_au, bare["t_au"], np.asarray(bare[name], dtype=float))
        ]
        for name in trace_names
    }
    diagnostics_by_channel: dict[str, dict[str, object]] = {
        "bare_qd": bare["diagnostics"]
    }
    diagnostics_by_channel["bare_qd"]["response_tail_tolerance"] = float(
        args.response_tail_tolerance
    )
    diagnostics_by_channel["bare_qd"]["response_tail_converged"] = bool(
        tail_ratios["bare_qd"] <= args.response_tail_tolerance
    )
    physical_parameters: dict[str, dict[str, object]] = {
        "bare_qd": {
            **params_to_physical_dict(
                reference_params, built_models[args.channels[0]][1].orientation
            ),
            "coupling_model": "isolated_qd_no_mnp_coupling",
        }
    }
    model_metadata: dict[str, dict[str, object]] = {}
    native_population_max = [float(np.max(bare["rho22"]))]
    native_population_max_time_fs = [
        float(au_to_fs(bare["t_au"][int(np.argmax(bare["rho22"]))]))
    ]

    for channel_id in args.channels:
        spec = CHANNELS[channel_id]
        params, bright, kernel, model, reduction = built_models[channel_id]
        result = full_results[channel_id]
        result_maximum_index = int(np.argmax(result.rho22))
        native_population_max.append(float(result.rho22[result_maximum_index]))
        native_population_max_time_fs.append(
            float(au_to_fs(result.t_au[result_maximum_index]))
        )
        source = {
            "W": result.W,
            "Q": result.Q,
            "P": result.P,
            "rho22": result.rho22,
            "mu_qd_au": result.mu_d_au,
            "mu_mnp_au": result.mu_p_au,
            "mu_total_au": result.mu_total_au,
            "mnp_field_at_qd_au": result.mnp_field_at_qd_au,
            "effective_qd_field_au": result.effective_qd_field_au,
        }
        for name in trace_names:
            collected[name].append(
                np.interp(time_au, result.t_au, np.asarray(source[name]))
            )
        diagnostics = result.diagnostics
        diagnostics_by_channel[channel_id] = {
            "solver_success": bool(diagnostics.solver_success),
            "solver_status": int(diagnostics.solver_status),
            "solver_message": str(diagnostics.solver_message),
            "solver_n_steps": int(diagnostics.n_steps),
            "solver_nfev": int(diagnostics.nfev),
            "max_step_limit_au": float(diagnostics.max_step_limit_au),
            "t_final_reached": bool(diagnostics.t_final_reached),
            "state_is_finite": bool(diagnostics.state_is_finite),
            "excited_population_min": float(
                diagnostics.excited_population_min
            ),
            "excited_population_max": float(
                diagnostics.excited_population_max
            ),
            "max_bloch_radius": float(diagnostics.max_bloch_radius),
            "min_density_eigenvalue": float(
                diagnostics.min_density_eigenvalue
            ),
            "pulse_spectral_fraction_in_fit_window": float(
                diagnostics.pulse_spectral_fraction_in_fit_window
            ),
            "pulse_spectral_leakage": float(
                diagnostics.pulse_spectral_leakage
            ),
            "qd_source_spectral_fraction_in_fit_window": float(
                diagnostics.qd_source_spectral_fraction_in_fit_window
            ),
            "qd_source_spectral_leakage": float(
                diagnostics.qd_source_spectral_leakage
            ),
            "mnp_drive_spectral_fraction_in_fit_window": float(
                diagnostics.mnp_drive_spectral_fraction_in_fit_window
            ),
            "mnp_drive_spectral_leakage": float(
                diagnostics.mnp_drive_spectral_leakage
            ),
            "mnp_dipole_spectral_fraction_in_fit_window": float(
                diagnostics.mnp_dipole_spectral_fraction_in_fit_window
            ),
            "mnp_dipole_spectral_leakage": float(
                diagnostics.mnp_dipole_spectral_leakage
            ),
            "mnp_field_spectral_fraction_in_fit_window": float(
                diagnostics.mnp_field_spectral_fraction_in_fit_window
            ),
            "mnp_field_spectral_leakage": float(
                diagnostics.mnp_field_spectral_leakage
            ),
            "response_tail_ratio": float(diagnostics.response_tail_ratio),
            "response_tail_tolerance": float(
                diagnostics.response_tail_tolerance
            ),
            "response_tail_converged": bool(
                diagnostics.response_tail_converged
                and tail_ratios[channel_id] <= args.response_tail_tolerance
            ),
            "work_nonnegative_within_tolerance": bool(
                diagnostics.work_nonnegative_within_tolerance
            ),
            "work_passivity_tolerance_au": float(
                diagnostics.work_passivity_tolerance_au
            ),
            "work_from_incident_field_j": float(
                result.work_from_incident_field_j
            ),
            "sigma_energy_transfer_cm2": float(
                result.sigma_energy_transfer_cm2
            ),
            "exact_spatial_mode_count": int(
                diagnostics.exact_spatial_mode_count
            ),
            "dynamic_spatial_mode_count": int(
                diagnostics.dynamic_spatial_mode_count
            ),
        }
        physical_parameters[channel_id] = params_to_physical_dict(
            params, spec.orientation
        )
        model_metadata[channel_id] = {
            "implementation": "FullQSSpheroidPulseModel",
            "spatial_kernel": type(kernel).__name__,
            "spatial_order_max": int(args.spatial_order_max),
            "exact_spatial_mode_count": int(model.exact_spatial_mode_count),
            "dynamic_spatial_mode_count": int(model.n_spatial_modes),
            "material_fit": _fit_metadata(bright),
            "material_fit_quality_gate": {
                "policy": args.bright_fit_quality_policy,
                "max_normalized_rms": float(
                    args.max_bright_fit_normalized_rms
                ),
                "max_pointwise_relative_error": float(
                    args.max_bright_fit_pointwise_relative_error
                ),
            },
            **_full_qs_certificate_metadata(model),
            "dark_reduction": _reduction_metadata(reduction, model),
        }

    stacked = {
        name: np.stack(values, axis=0) for name, values in collected.items()
    }
    rho22 = stacked["rho22"]
    final_population = rho22[:, -1]
    maximum_population = np.asarray(native_population_max, dtype=float)
    maximum_time_fs = np.asarray(native_population_max_time_fs, dtype=float)

    metadata = {
        "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "calculation_script": Path(__file__).resolve().relative_to(
            PROJECT_ROOT
        ).as_posix(),
        "command_line": [str(value) for value in sys.argv],
        "git": _git_provenance(),
        "source_file_sha256": _source_file_hashes(),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "requested_inputs": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "resolved_pulse": {
            "energy_eV": float(args.pulse_energy_ev),
            "tau_fs": float(args.pulse_tau_fs),
            "tau_kind": args.pulse_tau_kind,
            "E0_au": float(pulse.E0_au),
            "fluence_j_cm2": float(pulse.fluence_j_cm2(eps_m=args.eps_m)),
            "peak_intensity_w_cm2": float(
                pulse.peak_intensity_w_cm2(eps_m=args.eps_m)
            ),
            "envelope_centre_fs": 0.0,
            "carrier_phase_rad": 0.0,
            "real_field_convention": (
                "E0 exp[-t^2/(2 sigma_t^2)] cos(omega_L t)"
            ),
        },
        "initial_conditions": {
            "bloch_state_W_Q_P": [-1.0, 0.0, 0.0],
            "excited_population": 0.0,
            "all_material_ADE_coordinates_and_velocities": 0.0,
            "external_work_accumulator": 0.0,
        },
        "time_window": {
            "start_fs": float(au_to_fs(start_au)),
            "end_fs": float(au_to_fs(end_au)),
            "requested_post_fs": (
                None if args.post_fs is None else float(args.post_fs)
            ),
            "post_was_automatic": bool(automatic_post),
            "automatic_extension_count": int(tail_extension_count),
            "maximum_automatic_extensions": int(args.max_auto_tail_extensions),
            "tail_ratio_tolerance": float(args.response_tail_tolerance),
            "tail_ratio_by_channel": tail_ratios,
            "all_channel_tails_converged": bool(all_tails_converged),
            "population_decay_fraction_from_pulse_centre": decay_fraction_at_read,
            "maximum_population_decay_fraction_at_read": float(
                args.max_population_decay_fraction_at_read
            ),
            "population_decay_policy": args.population_decay_policy,
            "common_time_points_requested": int(args.common_time_points),
            "common_time_points_required_by_frequency_ceiling": int(
                required_common_time_points
            ),
            "common_time_points_resolved": int(time_au.size),
        },
        "channels": [
            {"id": channel_id, "label": label}
            for channel_id, label in zip(channel_ids, labels)
        ],
        "physical_parameters_by_channel": physical_parameters,
        "model_by_channel": model_metadata,
        "diagnostics_by_channel": diagnostics_by_channel,
        "definitions": {
            "rho22": "excited-state population (W + 1) / 2",
            "population_final": "rho22 at the final saved time",
            "population_max": "maximum rho22 on each solver's native adaptive grid",
            "incident_field_au": "real Gaussian-carrier incident field",
            "time_zero": "centre of the Gaussian field envelope",
        },
        "units": {
            "time_fs": "fs",
            "field": "atomic unit of electric field",
            "dipole": "atomic unit of electric dipole",
            "population": "dimensionless",
        },
        "fundamental_and_conversion_constants": {
            "speed_of_light_m_s": float(C_SI),
            "vacuum_permittivity_f_m": float(epsilon_0),
            "elementary_charge_c": float(elementary_charge),
            "reduced_planck_constant_j_s": float(hbar),
            "atomic_unit_length_m": float(AU_LENGTH_M),
            "atomic_unit_time_s": float(AU_TIME_S),
            "atomic_unit_energy_j": float(AU_ENERGY_J),
            "atomic_unit_energy_eV": float(AU_ENERGY_EV),
            "atomic_unit_field_v_m": float(AU_FIELD_V_M),
            "atomic_unit_dipole_c_m": float(AU_DIPOLE_C_M),
            "atomic_unit_speed": float(AU_SPEED_OF_LIGHT),
            "debye_c_m": float(DEBYE_C_M),
            "material_high_frequency_epsilon": float(
                MATERIAL_HIGH_FREQUENCY_EPSILON
            ),
            "material_interpolation": MATERIAL_INTERPOLATION,
        },
        "scope": [
            "local quasistatics",
            "point-dipole QD",
            "homogeneous axisymmetric metal spheroid",
            "semiclassical two-level QD",
            "fixed phenomenological QD relaxation rates",
        ],
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
            channel_ids=np.asarray(channel_ids),
            channel_labels=np.asarray(labels),
            time_au=time_au,
            time_fs=np.asarray(au_to_fs(time_au), dtype=float),
            incident_field_au=np.asarray(pulse.field(time_au), dtype=float),
            population_final=np.asarray(final_population, dtype=float),
            population_max=np.asarray(maximum_population, dtype=float),
            population_max_time_fs=np.asarray(maximum_time_fs, dtype=float),
            material_energy_eV=np.asarray(DEFAULT_AU_MATERIAL.energy_eV),
            material_n=np.asarray(DEFAULT_AU_MATERIAL.n),
            material_k=np.asarray(DEFAULT_AU_MATERIAL.k),
            **stacked,
        )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/article_population_dynamics.npz"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--channels",
        nargs="+",
        choices=tuple(CHANNELS),
        default=["axis_long", "side_long"],
        help="Hybrid channels; the isolated-QD reference is always included.",
    )
    parser.add_argument("--fluence-j-cm2", type=float, default=1.0e-7)
    parser.add_argument("--pulse-energy-ev", type=float, default=2.042)
    parser.add_argument("--pulse-tau-fs", type=float, default=20.0)
    parser.add_argument(
        "--pulse-tau-kind",
        choices=("fwhm_intensity", "sigma"),
        default="fwhm_intensity",
    )
    parser.add_argument("--start-sigma", type=float, default=10.0)
    parser.add_argument(
        "--post-fs",
        type=float,
        default=None,
        help="Explicit common final time; omit for automatic tail convergence.",
    )
    parser.add_argument("--max-auto-tail-extensions", type=int, default=3)
    parser.add_argument("--common-time-points", type=int, default=4001)

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
    parser.add_argument(
        "--qd-dipole-convention",
        choices=("effective_external", "bare_internal"),
        default="effective_external",
    )

    parser.add_argument("--spatial-order-max", type=int, default=80)
    parser.add_argument("--material-fit-modes", type=int, default=9)
    parser.add_argument("--fit-min-ev", type=float, default=0.8)
    parser.add_argument("--fit-max-ev", type=float, default=3.0)
    parser.add_argument("--fit-seed", type=int, default=12345)
    parser.add_argument("--alpha-objective-weight", type=float, default=1.0)
    parser.add_argument("--inv-alpha-objective-weight", type=float, default=1.2)
    parser.add_argument("--max-bright-fit-normalized-rms", type=float, default=0.025)
    parser.add_argument(
        "--max-bright-fit-pointwise-relative-error", type=float, default=0.05
    )
    parser.add_argument(
        "--bright-fit-quality-policy",
        choices=("raise", "warn", "ignore"),
        default="raise",
    )
    parser.add_argument("--reduction-fit-grid-points", type=int, default=1001)
    parser.add_argument("--reduction-audit-grid-points", type=int, default=1601)
    parser.add_argument("--reduction-rms-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--reduction-max-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--reduction-max-nodes", type=int)
    parser.add_argument(
        "--reduction-policy", choices=("raise", "warn", "ignore"), default="raise"
    )
    parser.add_argument(
        "--fit-quality-policy", choices=("raise", "warn", "ignore"), default="raise"
    )
    parser.add_argument("--max-modal-normalized-rms", type=float, default=0.03)
    parser.add_argument("--max-modal-relative-error", type=float, default=0.06)
    parser.add_argument("--modal-audit-points", type=int, default=2001)
    parser.add_argument("--reduction-reaudit-points", type=int, default=1709)
    parser.add_argument(
        "--spatial-convergence-policy",
        choices=("raise", "warn", "ignore"),
        default="raise",
    )
    parser.add_argument("--spatial-convergence-rtol", type=float, default=2.0e-5)
    parser.add_argument(
        "--radiative-consistency-policy",
        choices=("raise", "warn", "ignore"),
        default="warn",
    )

    parser.add_argument(
        "--method",
        choices=("DOP853", "RK45", "Radau", "BDF", "LSODA"),
        default="DOP853",
    )
    parser.add_argument("--rtol", type=float, default=1.0e-8)
    parser.add_argument("--atol", type=float, default=1.0e-10)
    parser.add_argument("--points-per-fastest-cycle", type=float, default=20.0)
    parser.add_argument(
        "--spectral-window-policy",
        choices=("raise", "warn", "ignore"),
        default="raise",
    )
    parser.add_argument("--max-spectral-leakage", type=float, default=1.0e-3)
    parser.add_argument(
        "--positivity-policy", choices=("raise", "warn", "ignore"), default="raise"
    )
    parser.add_argument("--positivity-tolerance", type=float, default=1.0e-7)
    parser.add_argument(
        "--work-passivity-policy",
        choices=("raise", "warn", "ignore"),
        default="raise",
    )
    parser.add_argument(
        "--response-tail-policy",
        choices=("raise", "warn", "ignore"),
        default="raise",
        help="A short dynamics window may retain free QD coherence.",
    )
    parser.add_argument("--response-tail-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--response-tail-window-fraction", type=float, default=0.05)
    parser.add_argument(
        "--max-population-decay-fraction-at-read", type=float, default=1.0e-3
    )
    parser.add_argument(
        "--population-decay-policy",
        choices=("raise", "warn", "ignore"),
        default="raise",
    )
    return parser.parse_args(argv)


def main() -> None:
    output = calculate_population_dynamics(parse_args())
    print(f"Saved population-dynamics artifact to {output}")


if __name__ == "__main__":
    main()
