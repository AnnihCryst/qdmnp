"""Calculate FQS QD-excitation spectra for direct, one- and multi-mode gold response.

The spatial model is held fixed.  Only the representation of the spheroid's
frequency-dependent material response changes:

``direct``
    tabulated complex optical constants evaluated by the analytic FQS kernel;
``one``
    one passive Lorentz oscillator fitted to the same material response;
``multi``
    a configurable passive multi-oscillator fit (nine modes by default).

The program writes a self-contained, pickle-free NPZ artifact and never draws
a figure.  Its companion plotting program reads only that artifact.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import platform
import sys
from typing import Any
import warnings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from article_observables.qd_mnp_calculate_excitation_fluence import (
    HYBRID_CHANNELS,
    ChannelSpec,
)
from article_observables.qd_mnp_material_modes_artifact import (
    atomic_write_npz,
    canonical_sha256,
    git_provenance,
    source_hashes,
)
from qd_mnp_full_qs_model import FullQSSpheroidPulseModel
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
    MATERIAL_HIGH_FREQUENCY_EPSILON,
    MATERIAL_INTERPOLATION,
    NATIVE_MODEL_PROFILE,
    HybridQDPlasmonModel,
    au_to_eV,
    make_params_with_overrides,
    params_to_physical_dict,
)
from qd_mnp_spheroid_equatorial import EquatorialSpheroidGreenInteraction
from qd_mnp_spheroid_green import (
    SpheroidGreenInteraction,
    qd_linear_polarizability_from_params,
    solve_linear_hybrid_response,
)


SCHEMA_NAME = "qd_mnp.material_excitation_spectrum_comparison"
SCHEMA_VERSION = 1
MATERIAL_MODEL_IDS = np.asarray(["direct", "one", "multi"], dtype="U16")
FIT_MODEL_IDS = np.asarray(["one", "multi"], dtype="U16")
CHANNEL_BY_ID = {channel.channel_id: channel for channel in HYBRID_CHANNELS}
POLICIES = ("raise", "warn", "ignore")


PUBLICATION_PRESET = {
    "energy_points": 2001,
    "spatial_order_max": 80,
    "multi_fit_modes": 9,
    "modal_audit_points": 2001,
    "spatial_convergence_policy": "raise",
    "energy_resolution_policy": "raise",
}

QUICK_PRESET = {
    "energy_points": 121,
    "spatial_order_max": 4,
    # Keep the production material approximation even in a quick spatial/grid
    # preview: a smaller fit would not satisfy the production accuracy gates.
    "multi_fit_modes": 9,
    "modal_audit_points": 301,
    "spatial_convergence_policy": "warn",
    "energy_resolution_policy": "warn",
}


def _apply_policy(policy: str, message: str) -> None:
    if policy == "raise":
        raise RuntimeError(message)
    if policy == "warn":
        warnings.warn(message, RuntimeWarning, stacklevel=2)


def _apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    defaults = PUBLICATION_PRESET if args.preset == "publication" else QUICK_PRESET
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.feature_center_ev is None:
        args.feature_center_ev = args.omega0_ev
    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preset", choices=("publication", "quick"), default="publication")
    parser.add_argument(
        "--channels",
        nargs="+",
        choices=tuple(CHANNEL_BY_ID),
        default=["axis_long", "axis_trans"],
        help="QD position/electric-field channels; no propagation direction is used.",
    )
    parser.add_argument("--gap-nm", type=float, default=2.0)
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

    parser.add_argument("--energy-min-ev", type=float, default=1.85)
    parser.add_argument("--energy-max-ev", type=float, default=2.25)
    parser.add_argument("--energy-points", type=int)
    parser.add_argument("--feature-center-ev", type=float)
    parser.add_argument("--feature-half-window-ev", type=float, default=0.12)
    parser.add_argument("--max-energy-step-over-isolated-fwhm", type=float, default=0.05)
    parser.add_argument("--energy-resolution-policy", choices=POLICIES)

    parser.add_argument("--spatial-order-max", type=int)
    parser.add_argument("--multi-fit-modes", type=int)
    parser.add_argument("--fit-min-ev", type=float, default=0.8)
    parser.add_argument("--fit-max-ev", type=float, default=3.0)
    parser.add_argument("--weight-center-ev", type=float)
    parser.add_argument("--weight-sigma-ev", type=float)
    parser.add_argument("--modal-audit-points", type=int)
    parser.add_argument("--max-bright-fit-normalized-rms", type=float, default=0.025)
    parser.add_argument(
        "--max-bright-fit-pointwise-relative-error",
        type=float,
        default=0.05,
    )
    parser.add_argument("--max-modal-normalized-rms", type=float, default=0.03)
    parser.add_argument("--max-modal-relative-error", type=float, default=0.08)
    parser.add_argument("--spatial-convergence-rtol", type=float, default=2.0e-5)
    parser.add_argument("--spatial-convergence-policy", choices=POLICIES)
    parser.add_argument(
        "--one-fit-quality-policy",
        choices=("warn", "ignore"),
        default="warn",
        help="A one-mode accuracy miss is diagnostic only; stability/passivity remain mandatory.",
    )
    parser.add_argument("--radiative-consistency-policy", choices=POLICIES, default="warn")
    parser.add_argument("--verbose-fit", action="store_true")
    return _apply_preset(parser.parse_args(argv))


def _validate_args(args: argparse.Namespace) -> None:
    positive_names = (
        "gap_nm",
        "c_nm",
        "a_nm",
        "qd_radius_nm",
        "eps_m",
        "eps_qd",
        "omega0_ev",
        "feature_half_window_ev",
        "max_energy_step_over_isolated_fwhm",
        "max_bright_fit_normalized_rms",
        "max_bright_fit_pointwise_relative_error",
        "max_modal_normalized_rms",
        "max_modal_relative_error",
        "spatial_convergence_rtol",
    )
    for name in positive_names:
        value = float(getattr(args, name))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive.")
    if args.c_nm < args.a_nm:
        raise ValueError("The analytic prolate-spheroid kernels require c_nm >= a_nm.")
    if not (0.0 < args.energy_min_ev < args.energy_max_ev):
        raise ValueError("Energy bounds must satisfy 0 < energy_min_ev < energy_max_ev.")
    if args.energy_points < 5:
        raise ValueError("--energy-points must be at least five.")
    if not (0.0 < args.fit_min_ev < args.fit_max_ev):
        raise ValueError("Fit bounds must satisfy 0 < fit_min_ev < fit_max_ev.")
    if args.energy_min_ev < args.fit_min_ev or args.energy_max_ev > args.fit_max_ev:
        raise ValueError("The plotted energy grid must lie inside the common fit window.")
    if args.multi_fit_modes < 2:
        raise ValueError("--multi-fit-modes must be at least two.")
    if args.spatial_order_max < 1:
        raise ValueError("--spatial-order-max must be positive.")
    if args.modal_audit_points < 101:
        raise ValueError("--modal-audit-points must be at least 101.")
    if not args.channels:
        raise ValueError("--channels must contain at least one channel.")
    if len(set(args.channels)) != len(args.channels):
        raise ValueError("--channels must not contain duplicates.")
    if (args.weight_center_ev is None) != (args.weight_sigma_ev is None):
        raise ValueError("--weight-center-ev and --weight-sigma-ev must be set together.")
    if args.weight_sigma_ev is not None and (
        not np.isfinite(args.weight_sigma_ev) or args.weight_sigma_ev <= 0.0
    ):
        raise ValueError("--weight-sigma-ev must be finite and positive.")
    for name in ("d_debye", "gamma_population_mev", "gamma2_coherence_mev"):
        value = getattr(args, name)
        if value is not None and (not np.isfinite(value) or value < 0.0):
            raise ValueError(f"--{name.replace('_', '-')} must be finite and non-negative.")


def _center_distance_nm(spec: ChannelSpec, args: argparse.Namespace) -> float:
    particle_radius = args.c_nm if spec.qd_placement == "axis" else args.a_nm
    return float(particle_radius + args.qd_radius_nm + args.gap_nm)


def _make_params(spec: ChannelSpec, args: argparse.Namespace):
    return make_params_with_overrides(
        c_nm=args.c_nm,
        a_nm=args.a_nm,
        r_nm=_center_distance_nm(spec, args),
        qd_radius_nm=args.qd_radius_nm,
        eps_m=args.eps_m,
        eps_qd=args.eps_qd,
        d_debye=args.d_debye,
        omega0_ev=args.omega0_ev,
        gamma_population_mev=args.gamma_population_mev,
        gamma2_coherence_mev=args.gamma2_coherence_mev,
        qd_dipole_convention=args.qd_dipole_convention,
        orientation=spec.orientation,
        qd_placement=spec.qd_placement,
        side_transverse_alignment=spec.side_transverse_alignment,
    )


def _make_kernel(spec: ChannelSpec, params: Any, n_max: int):
    if spec.qd_placement == "axis":
        return SpheroidGreenInteraction.from_params(
            params,
            orientation=spec.orientation,
            n_max=n_max,
        )
    return EquatorialSpheroidGreenInteraction.from_params(
        params,
        orientation=spec.orientation,
        n_max=n_max,
    )


def _make_bright_model(
    params: Any,
    spec: ChannelSpec,
    args: argparse.Namespace,
    *,
    n_modes: int,
    enforce_accuracy: bool,
) -> HybridQDPlasmonModel:
    return HybridQDPlasmonModel(
        params,
        orientation=spec.orientation,
        n_modes=n_modes,
        fit_window_eV=(args.fit_min_ev, args.fit_max_ev),
        weight_center_eV=args.weight_center_ev,
        weight_sigma_eV=args.weight_sigma_ev,
        max_fit_normalized_rms=(
            args.max_bright_fit_normalized_rms if enforce_accuracy else None
        ),
        max_fit_pointwise_relative_error=(
            args.max_bright_fit_pointwise_relative_error if enforce_accuracy else None
        ),
        radiative_consistency_policy=args.radiative_consistency_policy,
        verbose=args.verbose_fit,
    )


def _make_full_model(
    bright: HybridQDPlasmonModel,
    kernel: Any,
    args: argparse.Namespace,
    *,
    enforce_accuracy: bool,
) -> FullQSSpheroidPulseModel:
    # FullQSSpheroidPulseModel always enforces passive causal transformed modes
    # and coupled stability.  Only an accuracy miss is downgraded for N=1.
    return FullQSSpheroidPulseModel(
        bright,
        kernel,
        fit_quality_policy="raise" if enforce_accuracy else args.one_fit_quality_policy,
        max_modal_normalized_rms=args.max_modal_normalized_rms,
        max_modal_relative_error=args.max_modal_relative_error,
        modal_audit_points=args.modal_audit_points,
        spatial_convergence_policy=args.spatial_convergence_policy,
        spatial_convergence_rtol=args.spatial_convergence_rtol,
    )


def _warn_if_one_fit_is_inaccurate(
    model: HybridQDPlasmonModel,
    spec: ChannelSpec,
    args: argparse.Namespace,
) -> None:
    fit = model.fit
    misses = (
        fit.normalized_rms_alpha > args.max_bright_fit_normalized_rms
        or fit.normalized_rms_inv_alpha > args.max_bright_fit_normalized_rms
        or fit.max_normalized_alpha_error
        > args.max_bright_fit_pointwise_relative_error
    )
    if misses and args.one_fit_quality_policy == "warn":
        warnings.warn(
            "The one-oscillator material fit is retained as the deliberately "
            f"low-accuracy comparison for {spec.channel_id}: NRMS(alpha)="
            f"{fit.normalized_rms_alpha:.6g}, NRMS(1/alpha)="
            f"{fit.normalized_rms_inv_alpha:.6g}, max normalized alpha error="
            f"{fit.max_normalized_alpha_error:.6g}. Passivity and stability "
            "were still enforced.",
            RuntimeWarning,
            stacklevel=2,
        )


def _normalized_error(candidate: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    candidate = np.asarray(candidate)
    reference = np.asarray(reference)
    if candidate.shape != reference.shape:
        raise ValueError("Candidate and reference arrays must have identical shapes.")
    difference = candidate - reference
    rms_scale = float(np.sqrt(np.mean(np.abs(reference) ** 2)))
    max_scale = float(np.max(np.abs(reference)))
    tiny = np.finfo(float).tiny
    nrms = float(np.sqrt(np.mean(np.abs(difference) ** 2)) / max(rms_scale, tiny))
    maximum = float(np.max(np.abs(difference)) / max(max_scale, tiny))
    return nrms, maximum


def _crossing(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    level: float,
) -> float:
    if y1 == y0:
        return float(0.5 * (x0 + x1))
    return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))


def extract_fwhm_feature(
    energy_eV: np.ndarray,
    spectrum: np.ndarray,
    *,
    center_eV: float,
    half_window_eV: float,
) -> dict[str, float | int | str]:
    """Extract the strongest local QD feature and its operational FWHM."""

    energy = np.asarray(energy_eV, dtype=float)
    values = np.asarray(spectrum, dtype=float)
    if energy.ndim != 1 or values.shape != energy.shape or energy.size < 5:
        raise ValueError("Feature extraction needs matching 1-D arrays of length >= 5.")
    if np.any(~np.isfinite(energy)) or np.any(np.diff(energy) <= 0.0):
        raise ValueError("energy_eV must be finite and strictly increasing.")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        return {
            "energy_eV": np.nan,
            "height": np.nan,
            "fwhm_eV": np.nan,
            "left_eV": np.nan,
            "right_eV": np.nan,
            "status": "invalid_spectrum",
            "competing_peak_count": 0,
        }
    selected = np.flatnonzero(np.abs(energy - float(center_eV)) <= half_window_eV)
    if selected.size < 5:
        return {
            "energy_eV": np.nan,
            "height": np.nan,
            "fwhm_eV": np.nan,
            "left_eV": np.nan,
            "right_eV": np.nan,
            "status": "window_too_small",
            "competing_peak_count": 0,
        }
    lo, hi = int(selected[0]), int(selected[-1])
    local = values[lo : hi + 1]
    local_peak = int(np.argmax(local))
    peak_index = lo + local_peak
    peak = float(values[peak_index])
    if local_peak == 0 or local_peak == local.size - 1 or peak <= 0.0:
        return {
            "energy_eV": float(energy[peak_index]),
            "height": peak,
            "fwhm_eV": np.nan,
            "left_eV": np.nan,
            "right_eV": np.nan,
            "status": "edge_or_zero_peak",
            "competing_peak_count": 0,
        }

    local_maxima = np.flatnonzero(
        (local[1:-1] > local[:-2]) & (local[1:-1] >= local[2:])
    ) + 1
    competing = int(
        np.count_nonzero(local[local_maxima] >= 0.5 * peak)
        - int(local_peak in local_maxima)
    )
    half = 0.5 * peak
    left_candidates = np.flatnonzero(values[lo:peak_index] <= half)
    right_candidates = np.flatnonzero(values[peak_index + 1 : hi + 1] <= half)
    if left_candidates.size == 0 or right_candidates.size == 0:
        status = "both_half_crossings_missing"
        if left_candidates.size > 0:
            status = "right_half_crossing_missing"
        elif right_candidates.size > 0:
            status = "left_half_crossing_missing"
        return {
            "energy_eV": float(energy[peak_index]),
            "height": peak,
            "fwhm_eV": np.nan,
            "left_eV": np.nan,
            "right_eV": np.nan,
            "status": status,
            "competing_peak_count": max(competing, 0),
        }
    left_low = lo + int(left_candidates[-1])
    right_high = peak_index + 1 + int(right_candidates[0])
    left = _crossing(
        energy[left_low], values[left_low], energy[left_low + 1], values[left_low + 1], half
    )
    right = _crossing(
        energy[right_high - 1],
        values[right_high - 1],
        energy[right_high],
        values[right_high],
        half,
    )
    return {
        "energy_eV": float(energy[peak_index]),
        "height": peak,
        "fwhm_eV": float(right - left),
        "left_eV": left,
        "right_eV": right,
        "status": "split_or_ambiguous" if competing > 0 else "ok",
        "competing_peak_count": max(competing, 0),
    }


def _kernel_mode_arrays(kernel: Any, spec: ChannelSpec) -> dict[str, np.ndarray]:
    if hasattr(kernel, "mode_degrees"):
        degree = np.asarray(kernel.mode_degrees, dtype=np.int64)
        order = np.asarray(kernel.mode_orders, dtype=np.int64)
        sector = np.asarray(kernel.mode_sectors, dtype="U8")
        depolarization = np.asarray(kernel.depolarization_by_mode, dtype=float)
        weight = np.asarray(kernel.reaction_weight_by_mode_au_minus3, dtype=float)
        geometric = np.asarray(kernel.geometric_factor_by_mode, dtype=float)
        log_geometric = np.asarray(kernel.log_abs_geometric_factor_by_mode, dtype=float)
    else:
        degree = np.asarray(kernel.degrees, dtype=np.int64)
        order = np.full(degree.shape, 0 if spec.orientation == "long" else 1, dtype=np.int64)
        sector = np.full(degree.shape, "axis", dtype="U8")
        depolarization = np.asarray(kernel.depolarization_by_degree, dtype=float)
        weight = np.asarray(kernel.reaction_weight_by_degree_au_minus3, dtype=float)
        geometric = np.asarray(kernel.geometric_factor_by_degree, dtype=float)
        log_geometric = np.asarray(kernel.log_abs_geometric_factor_by_degree, dtype=float)
    return {
        "degree": degree,
        "order": order,
        "sector": sector,
        "depolarization": depolarization,
        "reaction_weight": weight,
        "geometric": geometric,
        "log_geometric": log_geometric,
    }


def _physical_constants() -> dict[str, float]:
    return {
        "epsilon_0_SI": float(8.8541878128e-12),
        "c_SI_m_s": float(C_SI),
        "elementary_charge_C": float(E_CHARGE),
        "debye_C_m": float(DEBYE_C_M),
        "atomic_energy_eV": float(AU_ENERGY_EV),
        "atomic_energy_J": float(AU_ENERGY_J),
        "atomic_time_s": float(AU_TIME_S),
        "atomic_length_m": float(AU_LENGTH_M),
        "atomic_dipole_C_m": float(AU_DIPOLE_C_M),
        "atomic_field_V_m": float(AU_FIELD_V_M),
        "atomic_speed_of_light": float(AU_SPEED_OF_LIGHT),
    }


def calculate_payload(
    args: argparse.Namespace,
    *,
    generator_path: Path | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    _validate_args(args)
    generator = Path(__file__) if generator_path is None else Path(generator_path)
    energy = np.linspace(args.energy_min_ev, args.energy_max_ev, args.energy_points)
    specs = [CHANNEL_BY_ID[channel_id] for channel_id in args.channels]
    n_model = MATERIAL_MODEL_IDS.size
    n_fit = FIT_MODEL_IDS.size
    n_channel = len(specs)
    n_energy = energy.size

    response_shape = (n_model, n_channel, n_energy)
    material_alpha = np.empty(response_shape, dtype=complex)
    interaction_A = np.empty(response_shape, dtype=complex)
    interaction_B = np.empty(response_shape, dtype=complex)
    interaction_K = np.empty(response_shape, dtype=complex)
    qd_transfer = np.empty(response_shape, dtype=complex)
    alpha_effective = np.empty(response_shape, dtype=complex)
    spectrum = np.empty(response_shape, dtype=float)
    beta_by_channel = np.empty((n_channel, n_energy), dtype=complex)
    center_distance_nm = np.empty(n_channel, dtype=float)

    fits: list[list[Any]] = [[None for _ in specs] for _ in range(n_fit)]
    full_models: list[list[Any]] = [[None for _ in specs] for _ in range(n_fit)]
    kernels: list[Any] = []
    params_by_channel: list[Any] = []
    direct_half_order = np.empty((n_channel, n_energy), dtype=float)
    direct_tail_block = np.empty((n_channel, n_energy), dtype=float)

    for channel_index, spec in enumerate(specs):
        params = _make_params(spec, args)
        kernel = _make_kernel(spec, params, args.spatial_order_max)
        one_bright = _make_bright_model(
            params, spec, args, n_modes=1, enforce_accuracy=False
        )
        _warn_if_one_fit_is_inaccurate(one_bright, spec, args)
        multi_bright = _make_bright_model(
            params,
            spec,
            args,
            n_modes=args.multi_fit_modes,
            enforce_accuracy=True,
        )
        one_full = _make_full_model(
            one_bright, kernel, args, enforce_accuracy=False
        )
        multi_full = _make_full_model(
            multi_bright, kernel, args, enforce_accuracy=True
        )
        for label, bright, full in (
            ("one", one_bright, one_full),
            ("multi", multi_bright, multi_full),
        ):
            if not (
                bright.fit.passive_on_fit_window
                and bright.fit.passive_for_all_positive_frequencies
                and bright.linear_stability.stable
                and full.coupled_stability.stable
            ):
                raise RuntimeError(
                    f"The {label} realization for {spec.channel_id} is not passive/stable."
                )

        direct_response = kernel.response_from_material(params.material, energy)
        fitted_responses = (
            one_full.frequency_response_from_fit(energy),
            multi_full.frequency_response_from_fit(energy),
        )
        beta = qd_linear_polarizability_from_params(params, energy)
        beta_by_channel[channel_index] = beta
        direct_half_order[channel_index] = np.asarray(
            direct_response.relative_half_order_change(), dtype=float
        )
        direct_tail_block[channel_index] = np.asarray(
            direct_response.relative_tail_block(), dtype=float
        )

        bright_models = (one_bright, multi_bright)
        material_alpha[:, channel_index] = np.stack(
            (
                one_bright.alpha_from_material(energy),
                one_bright.alpha_from_fit(energy),
                multi_bright.alpha_from_fit(energy),
            ),
            axis=0,
        )
        for model_index, response in enumerate((direct_response, *fitted_responses)):
            coupled = solve_linear_hybrid_response(response, beta, eps_m=params.eps_m)
            interaction_A[model_index, channel_index] = response.A_au3
            interaction_B[model_index, channel_index] = response.B
            interaction_K[model_index, channel_index] = response.K_au_minus3
            qd_transfer[model_index, channel_index] = coupled.qd_dipole_over_field_au3
            alpha_effective[model_index, channel_index] = coupled.alpha_effective_au3
            spectrum[model_index, channel_index] = np.abs(
                coupled.qd_dipole_over_field_au3
            ) ** 2

        fits[0][channel_index] = bright_models[0].fit
        fits[1][channel_index] = bright_models[1].fit
        full_models[0][channel_index] = one_full
        full_models[1][channel_index] = multi_full
        kernels.append(kernel)
        params_by_channel.append(params)
        center_distance_nm[channel_index] = _center_distance_nm(spec, args)

    isolated_transfer = beta_by_channel[0].copy()
    if not np.allclose(
        beta_by_channel,
        isolated_transfer[None, :],
        rtol=2.0e-14,
        atol=0.0,
    ):
        raise RuntimeError("The isolated-QD transfer unexpectedly differs between channels.")
    isolated_spectrum = np.abs(isolated_transfer) ** 2

    residual_sources = {
        "material_alpha_dimensionless": material_alpha,
        "interaction_A_au3": interaction_A,
        "interaction_B": interaction_B,
        "interaction_K_au_minus3": interaction_K,
        "qd_transfer_au3": qd_transfer,
        "alpha_effective_au3": alpha_effective,
        "excitation_spectrum_au6": spectrum,
    }
    residual_payload: dict[str, np.ndarray] = {}
    metric_payload: dict[str, np.ndarray] = {}
    for name, values in residual_sources.items():
        residual = values - values[0:1]
        residual_payload[f"{name}_residual_vs_direct"] = residual
        nrms = np.zeros((n_model, n_channel), dtype=float)
        maximum = np.zeros((n_model, n_channel), dtype=float)
        for model_index in range(1, n_model):
            for channel_index in range(n_channel):
                nrms[model_index, channel_index], maximum[model_index, channel_index] = (
                    _normalized_error(
                        values[model_index, channel_index],
                        values[0, channel_index],
                    )
                )
        metric_payload[f"{name}_nrms_error_vs_direct"] = nrms
        metric_payload[f"{name}_max_normalized_error_vs_direct"] = maximum

    direct_peak_scale = np.max(spectrum[0], axis=-1)
    normalized_spectrum_residual = np.divide(
        spectrum - spectrum[0:1],
        direct_peak_scale[None, :, None],
        out=np.full_like(spectrum, np.nan),
        where=direct_peak_scale[None, :, None] > np.finfo(float).tiny,
    )

    feature_shape = (n_model, n_channel)
    peak_energy = np.full(feature_shape, np.nan)
    peak_height = np.full(feature_shape, np.nan)
    fwhm = np.full(feature_shape, np.nan)
    half_left = np.full(feature_shape, np.nan)
    half_right = np.full(feature_shape, np.nan)
    feature_status = np.full(feature_shape, "not_evaluated", dtype="U40")
    competing_count = np.zeros(feature_shape, dtype=np.int64)
    for index in np.ndindex(feature_shape):
        feature = extract_fwhm_feature(
            energy,
            spectrum[index],
            center_eV=args.feature_center_ev,
            half_window_eV=args.feature_half_window_ev,
        )
        peak_energy[index] = feature["energy_eV"]
        peak_height[index] = feature["height"]
        fwhm[index] = feature["fwhm_eV"]
        half_left[index] = feature["left_eV"]
        half_right[index] = feature["right_eV"]
        feature_status[index] = feature["status"]
        competing_count[index] = feature["competing_peak_count"]

    isolated_feature = extract_fwhm_feature(
        energy,
        isolated_spectrum,
        center_eV=args.feature_center_ev,
        half_window_eV=args.feature_half_window_ev,
    )
    feature_mask = np.abs(energy - args.feature_center_ev) <= args.feature_half_window_ev
    isolated_window_peak = float(np.max(isolated_spectrum[feature_mask]))
    optimized_window_peak = np.max(spectrum[..., feature_mask], axis=-1)
    excitation_gain = optimized_window_peak / isolated_window_peak
    gain_at_own_peak = np.empty(feature_shape, dtype=float)
    for index in np.ndindex(feature_shape):
        gain_at_own_peak[index] = peak_height[index] / float(
            np.interp(peak_energy[index], energy, isolated_spectrum)
        )

    energy_step = float(energy[1] - energy[0])
    isolated_fwhm = float(isolated_feature["fwhm_eV"])
    energy_step_over_fwhm = (
        energy_step / isolated_fwhm
        if np.isfinite(isolated_fwhm) and isolated_fwhm > 0.0
        else np.inf
    )
    if energy_step_over_fwhm > args.max_energy_step_over_isolated_fwhm:
        _apply_policy(
            args.energy_resolution_policy,
            "The spectral grid is too coarse relative to the isolated-QD FWHM: "
            f"dE/Gamma0={energy_step_over_fwhm:.6g}, limit="
            f"{args.max_energy_step_over_isolated_fwhm:.6g}.",
        )

    # Rectangular fit-coefficient tables use an explicit mask; zero padding is
    # never interpreted as an oscillator.
    max_modes = args.multi_fit_modes
    fit_mode_mask = np.zeros((n_fit, n_channel, max_modes), dtype=bool)
    fit_strengths = np.zeros((n_fit, n_channel, max_modes), dtype=float)
    fit_omega = np.zeros((n_fit, n_channel, max_modes), dtype=float)
    fit_gamma = np.zeros((n_fit, n_channel, max_modes), dtype=float)
    fit_alpha_inf = np.empty((n_fit, n_channel), dtype=float)
    fit_rms_alpha = np.empty((n_fit, n_channel), dtype=float)
    fit_rms_inv_alpha = np.empty((n_fit, n_channel), dtype=float)
    fit_nrms_alpha = np.empty((n_fit, n_channel), dtype=float)
    fit_nrms_inv_alpha = np.empty((n_fit, n_channel), dtype=float)
    fit_max_alpha_error = np.empty((n_fit, n_channel), dtype=float)
    fit_min_imag = np.empty((n_fit, n_channel), dtype=float)
    fit_cost = np.empty((n_fit, n_channel), dtype=float)
    fit_passivity_points = np.empty((n_fit, n_channel), dtype=np.int64)
    fit_passive_window = np.empty((n_fit, n_channel), dtype=bool)
    fit_passive_positive = np.empty((n_fit, n_channel), dtype=bool)

    fit_sample_offsets = [0]
    fit_energy_used: list[np.ndarray] = []
    fit_alpha_used: list[np.ndarray] = []
    bright_pole_offsets = [0]
    bright_poles: list[np.ndarray] = []

    modal_offsets = [0]
    modal_nrms: list[np.ndarray] = []
    modal_max: list[np.ndarray] = []
    modal_min_imag: list[np.ndarray] = []
    coupled_right_offsets = [0]
    coupled_right: list[np.ndarray] = []
    coupled_large_offsets = [0]
    coupled_large: list[np.ndarray] = []

    modal_scalar_shape = (n_fit, n_channel)
    modal_K_nrms = np.empty(modal_scalar_shape, dtype=float)
    modal_K_max = np.empty(modal_scalar_shape, dtype=float)
    modal_max_nrms = np.empty(modal_scalar_shape, dtype=float)
    modal_max_relative = np.empty(modal_scalar_shape, dtype=float)
    modal_passive = np.empty(modal_scalar_shape, dtype=bool)
    modal_accepted = np.empty(modal_scalar_shape, dtype=bool)
    spatial_max_half = np.empty(modal_scalar_shape, dtype=float)
    spatial_max_tail = np.empty(modal_scalar_shape, dtype=float)
    spatial_accepted = np.empty(modal_scalar_shape, dtype=bool)
    coupled_abscissa = np.full(modal_scalar_shape, np.nan)
    coupled_abscissa_available = np.empty(modal_scalar_shape, dtype=bool)
    coupled_abscissa_is_bound = np.empty(modal_scalar_shape, dtype=bool)
    coupled_decay = np.empty(modal_scalar_shape, dtype=float)
    coupled_decay_exact = np.empty(modal_scalar_shape, dtype=bool)
    coupled_radius = np.empty(modal_scalar_shape, dtype=float)
    coupled_tolerance = np.empty(modal_scalar_shape, dtype=float)
    coupled_stable = np.empty(modal_scalar_shape, dtype=bool)
    coupled_dimension = np.empty(modal_scalar_shape, dtype=np.int64)
    coupled_eigensolver = np.empty(modal_scalar_shape, dtype="U64")
    bright_abscissa = np.empty(modal_scalar_shape, dtype=float)
    bright_tolerance = np.empty(modal_scalar_shape, dtype=float)
    bright_stable = np.empty(modal_scalar_shape, dtype=bool)

    spatial_audit_half = np.empty(
        (n_fit, n_channel, args.modal_audit_points), dtype=float
    )
    spatial_audit_tail = np.empty_like(spatial_audit_half)
    for fit_index in range(n_fit):
        for channel_index in range(n_channel):
            fit = fits[fit_index][channel_index]
            full = full_models[fit_index][channel_index]
            count = fit.strengths_au2.size
            fit_mode_mask[fit_index, channel_index, :count] = True
            fit_strengths[fit_index, channel_index, :count] = fit.strengths_au2
            fit_omega[fit_index, channel_index, :count] = fit.omega_modes_au
            fit_gamma[fit_index, channel_index, :count] = fit.gamma_modes_au
            fit_alpha_inf[fit_index, channel_index] = fit.alpha_inf
            fit_rms_alpha[fit_index, channel_index] = fit.rms_alpha
            fit_rms_inv_alpha[fit_index, channel_index] = fit.rms_inv_alpha
            fit_nrms_alpha[fit_index, channel_index] = fit.normalized_rms_alpha
            fit_nrms_inv_alpha[fit_index, channel_index] = fit.normalized_rms_inv_alpha
            fit_max_alpha_error[fit_index, channel_index] = fit.max_normalized_alpha_error
            fit_min_imag[fit_index, channel_index] = fit.min_imag_alpha_fit_window
            fit_cost[fit_index, channel_index] = fit.cost
            fit_passivity_points[fit_index, channel_index] = fit.passivity_grid_points
            fit_passive_window[fit_index, channel_index] = fit.passive_on_fit_window
            fit_passive_positive[fit_index, channel_index] = (
                fit.passive_for_all_positive_frequencies
            )
            fit_energy_values = np.asarray(fit.energies_used_eV, dtype=float)
            fit_alpha_values = np.asarray(fit.alpha_used, dtype=complex)
            fit_energy_used.append(fit_energy_values)
            fit_alpha_used.append(fit_alpha_values)
            fit_sample_offsets.append(
                fit_sample_offsets[-1] + fit_energy_values.size
            )

            bright_diag = full.bright_model.linear_stability
            poles = np.asarray(bright_diag.poles_au, dtype=complex)
            bright_poles.append(poles)
            bright_pole_offsets.append(bright_pole_offsets[-1] + poles.size)
            bright_abscissa[fit_index, channel_index] = bright_diag.spectral_abscissa_au
            bright_tolerance[fit_index, channel_index] = bright_diag.tolerance_au
            bright_stable[fit_index, channel_index] = bright_diag.stable

            modal = full.modal_fit_diagnostics
            for target, values in (
                (modal_nrms, modal.normalized_rms_by_degree),
                (modal_max, modal.max_relative_error_by_degree),
                (modal_min_imag, modal.minimum_imaginary_part_by_degree),
            ):
                target.append(np.asarray(values, dtype=float))
            modal_offsets.append(
                modal_offsets[-1] + np.asarray(modal.normalized_rms_by_degree).size
            )
            modal_K_nrms[fit_index, channel_index] = modal.K_normalized_rms
            modal_K_max[fit_index, channel_index] = modal.K_max_relative_error
            modal_max_nrms[fit_index, channel_index] = modal.max_normalized_rms
            modal_max_relative[fit_index, channel_index] = modal.max_relative_error
            modal_passive[fit_index, channel_index] = modal.passive_on_audit_grid
            modal_accepted[fit_index, channel_index] = modal.accepted

            spatial = full.spatial_convergence_diagnostics
            spatial_max_half[fit_index, channel_index] = spatial.max_half_order_relative_change
            spatial_max_tail[fit_index, channel_index] = spatial.max_tail_block_relative_mass
            spatial_accepted[fit_index, channel_index] = spatial.accepted
            spatial_audit_half[fit_index, channel_index] = (
                spatial.half_order_relative_change_by_energy
            )
            spatial_audit_tail[fit_index, channel_index] = (
                spatial.tail_block_relative_mass_by_energy
            )

            coupled = full.coupled_stability
            right = np.asarray(coupled.rightmost_poles_au, dtype=complex)
            large = np.asarray(coupled.largest_magnitude_poles_au, dtype=complex)
            coupled_right.append(right)
            coupled_large.append(large)
            coupled_right_offsets.append(coupled_right_offsets[-1] + right.size)
            coupled_large_offsets.append(coupled_large_offsets[-1] + large.size)
            if coupled.spectral_abscissa_au is not None:
                coupled_abscissa[fit_index, channel_index] = coupled.spectral_abscissa_au
            coupled_abscissa_available[fit_index, channel_index] = (
                coupled.spectral_abscissa_available
            )
            coupled_abscissa_is_bound[fit_index, channel_index] = (
                coupled.spectral_abscissa_is_bound
            )
            coupled_decay[fit_index, channel_index] = coupled.decay_rate_estimate_au
            coupled_decay_exact[fit_index, channel_index] = coupled.decay_rate_estimate_is_exact
            coupled_radius[fit_index, channel_index] = coupled.spectral_radius_au
            coupled_tolerance[fit_index, channel_index] = coupled.tolerance_au
            coupled_stable[fit_index, channel_index] = coupled.stable
            coupled_dimension[fit_index, channel_index] = coupled.coherent_state_dimension
            coupled_eigensolver[fit_index, channel_index] = coupled.eigensolver

    spatial_offsets = [0]
    spatial_degree: list[np.ndarray] = []
    spatial_order: list[np.ndarray] = []
    spatial_sector: list[np.ndarray] = []
    spatial_depolarization: list[np.ndarray] = []
    spatial_weight: list[np.ndarray] = []
    spatial_geometric: list[np.ndarray] = []
    spatial_log_geometric: list[np.ndarray] = []
    for kernel, spec in zip(kernels, specs):
        arrays = _kernel_mode_arrays(kernel, spec)
        spatial_degree.append(arrays["degree"])
        spatial_order.append(arrays["order"])
        spatial_sector.append(arrays["sector"])
        spatial_depolarization.append(arrays["depolarization"])
        spatial_weight.append(arrays["reaction_weight"])
        spatial_geometric.append(arrays["geometric"])
        spatial_log_geometric.append(arrays["log_geometric"])
        spatial_offsets.append(spatial_offsets[-1] + arrays["degree"].size)

    payload: dict[str, np.ndarray] = {
        "material_model_id": MATERIAL_MODEL_IDS,
        "model_id": MATERIAL_MODEL_IDS,
        "fit_model_id": FIT_MODEL_IDS,
        "channel_id": np.asarray(args.channels, dtype="U32"),
        "energy_eV": energy,
        "surface_gap_nm": np.asarray(args.gap_nm),
        "center_distance_nm": center_distance_nm,
        "isolated_qd_transfer_au3": isolated_transfer,
        "isolated_excitation_spectrum_au6": isolated_spectrum,
        "isolated_qd_spectrum": isolated_spectrum,
        "qd_beta_by_channel_au3": beta_by_channel,
        "material_alpha_dimensionless": material_alpha,
        "interaction_A_au3": interaction_A,
        "interaction_B": interaction_B,
        "interaction_K_au_minus3": interaction_K,
        "qd_transfer_au3": qd_transfer,
        "alpha_effective_au3": alpha_effective,
        "excitation_spectrum_au6": spectrum,
        "qd_excitation_spectrum": spectrum,
        "excitation_spectrum_residual_normalized_to_direct_peak": (
            normalized_spectrum_residual
        ),
        **residual_payload,
        **metric_payload,
        "isolated_peak_energy_eV": np.asarray(isolated_feature["energy_eV"]),
        "isolated_peak_height_au6": np.asarray(isolated_feature["height"]),
        "isolated_fwhm_eV": np.asarray(isolated_feature["fwhm_eV"]),
        "isolated_half_max_left_eV": np.asarray(isolated_feature["left_eV"]),
        "isolated_half_max_right_eV": np.asarray(isolated_feature["right_eV"]),
        "isolated_feature_status": np.asarray(isolated_feature["status"], dtype="U40"),
        "isolated_competing_peak_count": np.asarray(
            isolated_feature["competing_peak_count"], dtype=np.int64
        ),
        "peak_energy_eV": peak_energy,
        "peak_height_au6": peak_height,
        "fwhm_eV": fwhm,
        "half_max_left_eV": half_left,
        "half_max_right_eV": half_right,
        "feature_status": feature_status,
        "competing_peak_count": competing_count,
        "excitation_gain_optimized": excitation_gain,
        "excitation_gain_at_own_peak": gain_at_own_peak,
        "resonance_shift_vs_direct_eV": peak_energy - peak_energy[0:1],
        "fwhm_difference_vs_direct_eV": fwhm - fwhm[0:1],
        "gain_difference_vs_direct": excitation_gain - excitation_gain[0:1],
        "energy_step_eV": np.asarray(energy_step),
        "energy_step_over_isolated_fwhm": np.asarray(energy_step_over_fwhm),
        "fit_mode_count": np.asarray(
            [[1] * n_channel, [args.multi_fit_modes] * n_channel], dtype=np.int64
        ),
        "fit_entry_fit_model_index": np.repeat(
            np.arange(n_fit, dtype=np.int64), n_channel
        ),
        "fit_entry_channel_index": np.tile(
            np.arange(n_channel, dtype=np.int64), n_fit
        ),
        "fit_mode_mask": fit_mode_mask,
        "fit_alpha_inf": fit_alpha_inf,
        "fit_strengths_au2": fit_strengths,
        "fit_omega_modes_au": fit_omega,
        "fit_gamma_modes_au": fit_gamma,
        "fit_omega_modes_eV": fit_omega * AU_ENERGY_EV,
        "fit_gamma_modes_eV": fit_gamma * AU_ENERGY_EV,
        "fit_strengths_eV2": fit_strengths * AU_ENERGY_EV**2,
        "fit_sample_entry_offsets": np.asarray(fit_sample_offsets, dtype=np.int64),
        "fit_energies_used_eV": np.concatenate(fit_energy_used),
        "fit_alpha_used_dimensionless": np.concatenate(fit_alpha_used),
        "fit_rms_alpha": fit_rms_alpha,
        "fit_rms_inv_alpha": fit_rms_inv_alpha,
        "fit_normalized_rms_alpha": fit_nrms_alpha,
        "fit_normalized_rms_inv_alpha": fit_nrms_inv_alpha,
        "fit_max_normalized_alpha_error": fit_max_alpha_error,
        "fit_min_imag_alpha_fit_window": fit_min_imag,
        "fit_cost": fit_cost,
        "fit_passivity_grid_points": fit_passivity_points,
        "fit_passive_on_fit_window": fit_passive_window,
        "fit_passive_for_all_positive_frequencies": fit_passive_positive,
        "bright_stability_entry_offsets": np.asarray(bright_pole_offsets, dtype=np.int64),
        "bright_stability_poles_au": np.concatenate(bright_poles),
        "bright_stability_spectral_abscissa_au": bright_abscissa,
        "bright_stability_tolerance_au": bright_tolerance,
        "bright_stability_stable": bright_stable,
        "modal_fit_entry_offsets": np.asarray(modal_offsets, dtype=np.int64),
        "modal_fit_normalized_rms": np.concatenate(modal_nrms),
        "modal_fit_max_relative_error": np.concatenate(modal_max),
        "modal_fit_minimum_imaginary_part": np.concatenate(modal_min_imag),
        "modal_fit_K_normalized_rms": modal_K_nrms,
        "modal_fit_K_max_relative_error": modal_K_max,
        "modal_fit_max_normalized_rms": modal_max_nrms,
        "modal_fit_max_relative_error_over_modes": modal_max_relative,
        "modal_fit_passive_on_audit_grid": modal_passive,
        "modal_fit_accepted": modal_accepted,
        "spatial_convergence_max_half_order_relative_change": spatial_max_half,
        "spatial_convergence_max_tail_block_relative_mass": spatial_max_tail,
        "spatial_convergence_accepted": spatial_accepted,
        "spatial_convergence_audit_energy_eV": np.linspace(
            args.fit_min_ev, args.fit_max_ev, args.modal_audit_points
        ),
        "spatial_convergence_half_order_relative_change": spatial_audit_half,
        "spatial_convergence_tail_block_relative_mass": spatial_audit_tail,
        "coupled_stability_rightmost_entry_offsets": np.asarray(
            coupled_right_offsets, dtype=np.int64
        ),
        "coupled_stability_rightmost_poles_au": np.concatenate(coupled_right),
        "coupled_stability_largest_entry_offsets": np.asarray(
            coupled_large_offsets, dtype=np.int64
        ),
        "coupled_stability_largest_magnitude_poles_au": np.concatenate(coupled_large),
        "coupled_stability_spectral_abscissa_au": coupled_abscissa,
        "coupled_stability_spectral_abscissa_available": coupled_abscissa_available,
        "coupled_stability_spectral_abscissa_is_bound": coupled_abscissa_is_bound,
        "coupled_stability_decay_rate_estimate_au": coupled_decay,
        "coupled_stability_decay_rate_estimate_is_exact": coupled_decay_exact,
        "coupled_stability_spectral_radius_au": coupled_radius,
        "coupled_stability_tolerance_au": coupled_tolerance,
        "coupled_stability_stable": coupled_stable,
        "coupled_stability_coherent_state_dimension": coupled_dimension,
        "coupled_stability_eigensolver": coupled_eigensolver,
        "spatial_channel_offsets": np.asarray(spatial_offsets, dtype=np.int64),
        "spatial_mode_degree": np.concatenate(spatial_degree),
        "spatial_mode_order": np.concatenate(spatial_order),
        "spatial_mode_sector": np.concatenate(spatial_sector),
        "spatial_mode_depolarization": np.concatenate(spatial_depolarization),
        "spatial_mode_reaction_weight_au_minus3": np.concatenate(spatial_weight),
        "spatial_mode_geometric_factor": np.concatenate(spatial_geometric),
        "spatial_mode_log_abs_geometric_factor": np.concatenate(spatial_log_geometric),
        "direct_spatial_half_order_relative_change": direct_half_order,
        "direct_spatial_tail_block_relative_mass": direct_tail_block,
        "material_energy_eV": np.asarray(params_by_channel[0].material.energy_eV),
        "material_n": np.asarray(params_by_channel[0].material.n),
        "material_k": np.asarray(params_by_channel[0].material.k),
    }

    channel_documents = []
    surface_relation = {
        "axis_long": "normal",
        "axis_trans": "tangential",
        "side_long": "tangential",
        "side_trans_radial": "normal",
        "side_trans_tangential": "tangential",
    }
    for spec, params in zip(specs, params_by_channel):
        document = asdict(spec)
        document["field_relative_to_particle_axis"] = spec.orientation
        document["field_relative_to_local_surface"] = surface_relation[spec.channel_id]
        document["center_distance_nm"] = _center_distance_nm(spec, args)
        document["physical_parameters"] = params_to_physical_dict(
            params, orientation=spec.orientation
        )
        channel_documents.append(document)

    resolved_arguments = {
        key: value if not isinstance(value, Path) else str(value)
        for key, value in vars(args).items()
    }
    metadata: dict[str, Any] = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "observable": "S(E)=abs(p_QD(E)/E_inc(E))^2",
        "calculation_family": "frequency_domain_weak_field_FQS",
        "material_model_ids": MATERIAL_MODEL_IDS.tolist(),
        "fit_model_ids": FIT_MODEL_IDS.tolist(),
        "multi_fit_mode_count": int(args.multi_fit_modes),
        "requested_and_resolved_arguments": resolved_arguments,
        "requested_arguments": resolved_arguments,
        "resolved_arguments": resolved_arguments,
        "input_signature_sha256": canonical_sha256(resolved_arguments),
        "channels": channel_documents,
        "geometry_convention": {
            "gap": "surface-to-surface distance",
            "axis_center_distance_nm": "c_nm + qd_radius_nm + gap_nm",
            "side_center_distance_nm": "a_nm + qd_radius_nm + gap_nm",
            "spatial_model": "identical analytic full local-QS kernel for direct/one/multi",
            "field_only": True,
            "laser_propagation_direction_included": False,
        },
        "model_api": {
            "direct_FQS": "Spheroid/EquatorialSpheroidGreenInteraction.response_from_material",
            "fitted_FQS": "FullQSSpheroidPulseModel.frequency_response_from_fit",
            "linear_hybrid": "qd_mnp_spheroid_green.solve_linear_hybrid_response",
            "isolated_QD": "qd_mnp_spheroid_green.qd_linear_polarizability_from_params",
        },
        "fit_quality_semantics": {
            "one": "accuracy is diagnostic warn/ignore; passivity and stability are mandatory",
            "multi": "bright and transformed-modal accuracy, passivity and stability gates must pass",
            "bright_nrms_limit": float(args.max_bright_fit_normalized_rms),
            "bright_pointwise_limit": float(
                args.max_bright_fit_pointwise_relative_error
            ),
            "modal_nrms_limit": float(args.max_modal_normalized_rms),
            "modal_relative_limit": float(args.max_modal_relative_error),
        },
        "feature_definition": {
            "window_center_eV": float(args.feature_center_ev),
            "window_half_width_eV": float(args.feature_half_window_ev),
            "resonance": "strongest sampled point inside the fixed feature window",
            "fwhm": "linear-interpolated crossings at half the absolute peak height",
            "gain_optimized": "window maximum divided by isolated-QD window maximum",
            "gain_at_own_peak": "hybrid peak divided by isolated spectrum at the same energy",
            "ambiguous_status": "a second local maximum reaches at least half the selected peak",
        },
        "units": {
            "energy": "eV",
            "surface_gap": "nm",
            "qd_transfer": "atomic polarizability volume (a0^3)",
            "alpha_effective": "atomic polarizability volume (a0^3)",
            "A": "a0^3",
            "B": "dimensionless",
            "K": "a0^-3",
            "excitation_spectrum": "a0^6",
            "fit_strength": "Hartree^2",
            "fit_frequency_and_damping": "Hartree",
        },
        "material": {
            "interpolation": MATERIAL_INTERPOLATION,
            "high_frequency_epsilon": float(MATERIAL_HIGH_FREQUENCY_EPSILON),
            "source": "material_energy_eV/material_n/material_k arrays in this artifact",
        },
        "model_profile": NATIVE_MODEL_PROFILE,
        "physical_constants": _physical_constants(),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "provenance": {
            "git": git_provenance(PROJECT_ROOT),
            "generator": str(generator.resolve()),
            "source_sha256": source_hashes(
                (
                    generator,
                    Path(__file__),
                    PROJECT_ROOT / "article_observables" / "qd_mnp_material_modes_artifact.py",
                    PROJECT_ROOT / "qd_mnp_rational_fit.py",
                    PROJECT_ROOT / "qd_mnp_full_qs_model.py",
                    PROJECT_ROOT / "qd_mnp_spheroid_green.py",
                    PROJECT_ROOT / "qd_mnp_spheroid_equatorial.py",
                )
            ),
        },
        "limitations": [
            "local quasistatic homogeneous spheroid and point-like QD",
            "fixed phenomenological QD gamma1 and Gamma2; no LDOS-derived decay correction",
            "no laser propagation direction, retardation, radiation pattern, nonlocality, tunnelling or charge transfer",
            "the reported FWHM is an operational excitation-spectrum width, not a Purcell lifetime linewidth",
            "one and multi curves are approximations of the same tabulated material and not separately fitted experimental systems",
        ],
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
    print(f"Saved {SCHEMA_NAME} v{SCHEMA_VERSION}: {output}")
    return output


if __name__ == "__main__":
    main()
