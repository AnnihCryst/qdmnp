"""Shared DD/FQS engines for article gap metrics.

This module is intentionally not a command-line entry point.  The ten small
``qd_mnp_{calculate,plot}_*_gap.py`` programs below use it so that every
reported dependence has a calculation/plot pair without copying either the
physics adapters or the artifact validation code.

The spatial comparison is strictly between the current public APIs:

* :class:`qd_mnp_rational_fit.HybridQDPlasmonModel` (central point-dipole/DD),
* :class:`qd_mnp_full_qs_model.FullQSSpheroidPulseModel` (full local-QS/FQS),
* :class:`qd_mnp_spheroid_green.LegacyDipoleInteraction` and the analytic
  spheroidal Green kernels for frequency-domain spectra.

No inverse-polarizability compatibility program is imported or evaluated.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any, Callable, Iterable, Literal
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.constants import epsilon_0 as EPSILON_0_SI
from scipy.optimize import brentq
from scipy.signal import find_peaks, peak_prominences, peak_widths

from article_observables.qd_mnp_calculate_excitation_fluence import (
    HYBRID_CHANNELS,
    ChannelSpec,
    _solve_bare_qd,
    fluence_grid_resolution_diagnostics,
)
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
    eV_to_au,
    field_au_to_si,
    fs_to_au,
    make_params_with_overrides,
    params_to_physical_dict,
)
from qd_mnp_spheroid_equatorial import EquatorialSpheroidGreenInteraction
from qd_mnp_spheroid_green import (
    LegacyDipoleInteraction,
    SpheroidGreenInteraction,
    qd_linear_polarizability_from_params,
    solve_linear_hybrid_response,
)


SCHEMA_NAME = "qd_mnp.dd_fqs_gap_metric"
SCHEMA_VERSION = 1
MODEL_IDS = np.asarray(["dd", "fqs"], dtype="U8")
CHANNEL_BY_ID = {item.channel_id: item for item in HYBRID_CHANNELS}
SPECTRAL_METRICS = {
    "excitation_gain",
    "resonance_shift",
    "spectral_width",
    "model_discrepancy",
}
ALL_METRICS = SPECTRAL_METRICS | {"threshold_fluence"}
POLICIES = ("raise", "warn", "ignore")
SIDE_DIRECT_REFERENCE_ORDER_MAX = 8


@dataclass(frozen=True)
class BuiltChannel:
    spec: ChannelSpec
    gap_nm: float
    params: Any
    dd_model: HybridQDPlasmonModel
    kernel: Any
    fqs_model: FullQSSpheroidPulseModel | None
    reduction: Any | None


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


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _apply_policy(policy: str, message: str) -> None:
    if policy == "raise":
        raise RuntimeError(message)
    if policy == "warn":
        warnings.warn(message, RuntimeWarning, stacklevel=2)


def _git_provenance() -> dict[str, Any]:
    def command(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip() or None

    return {
        "commit": command("rev-parse", "HEAD"),
        "branch": command("branch", "--show-current"),
        "working_tree_porcelain": command("status", "--short"),
    }


def _source_hashes(extra_paths: Iterable[Path] = ()) -> dict[str, str]:
    paths = [
        Path(__file__).resolve(),
        PROJECT_ROOT / "qd_mnp_rational_fit.py",
        PROJECT_ROOT / "qd_mnp_full_qs_model.py",
        PROJECT_ROOT / "qd_mnp_spheroid_green.py",
        PROJECT_ROOT / "qd_mnp_spheroid_equatorial.py",
        *extra_paths,
    ]
    hashes: dict[str, str] = {}
    for path in paths:
        resolved = path.resolve()
        if resolved.is_file():
            try:
                key = str(resolved.relative_to(PROJECT_ROOT)).replace("\\", "/")
            except ValueError:
                key = str(resolved)
            hashes[key] = hashlib.sha256(resolved.read_bytes()).hexdigest()
    return hashes


def _physical_constants() -> dict[str, float]:
    return {
        "epsilon_0_SI": float(EPSILON_0_SI),
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


def _atomic_write_npz(
    output: Path,
    payload: dict[str, np.ndarray],
    metadata: dict[str, Any],
    *,
    overwrite: bool,
) -> Path:
    output = output.resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing artifact {output}; pass --overwrite."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    document = _json_ready(metadata)
    document["array_keys"] = sorted([*payload, "metadata_json"])
    encoded = json.dumps(document, ensure_ascii=False, sort_keys=True, allow_nan=False)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output.stem}.",
            suffix=".npz",
            dir=output.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        np.savez_compressed(temporary, metadata_json=np.asarray(encoded), **payload)
        os.replace(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return output


def load_gap_metric_artifact(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        if "metadata_json" not in archive.files:
            raise ValueError("Artifact has no metadata_json.")
        try:
            metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("metadata_json is not valid scalar JSON.") from exc
        payload = {
            name: np.asarray(archive[name]).copy()
            for name in archive.files
            if name != "metadata_json"
        }
    if metadata.get("schema_name") != SCHEMA_NAME:
        raise ValueError(
            f"Expected schema_name={SCHEMA_NAME!r}, got {metadata.get('schema_name')!r}."
        )
    if int(metadata.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version={metadata.get('schema_version')!r}."
        )
    expected = set(metadata.get("array_keys", ()))
    actual = {*payload, "metadata_json"}
    if expected and expected != actual:
        raise ValueError("Artifact array_keys do not match the stored arrays.")
    for required in ("channel_id", "model_id", "gap_nm"):
        if required not in payload:
            raise ValueError(f"Artifact misses required array {required!r}.")
    return payload, metadata


def _validate_common_inputs(args: argparse.Namespace) -> None:
    positive = (
        "c_nm",
        "a_nm",
        "qd_radius_nm",
        "eps_m",
        "eps_qd",
        "omega0_ev",
        "fit_min_ev",
        "fit_max_ev",
        "rtol",
        "atol",
        "tail_ratio_tolerance",
        "tail_window_fraction",
        "spatial_convergence_rtol",
        "dd_tolerance",
    )
    for name in positive:
        value = float(getattr(args, name))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive.")
    if args.c_nm < args.a_nm:
        raise ValueError("The current analytic kernels require a prolate spheroid c_nm >= a_nm.")
    if args.fit_min_ev >= args.fit_max_ev:
        raise ValueError("--fit-min-ev must be smaller than --fit-max-ev.")
    gaps = np.asarray(args.gaps_nm, dtype=float)
    if gaps.ndim != 1 or gaps.size < 1 or np.any(~np.isfinite(gaps)) or np.any(gaps <= 0.0):
        raise ValueError("--gaps-nm must contain finite positive surface gaps.")
    if np.any(np.diff(gaps) <= 0.0):
        raise ValueError("--gaps-nm must be strictly increasing.")
    if len(set(args.channels)) != len(args.channels):
        raise ValueError("--channels must not contain duplicates.")
    for name in args.channels:
        if name not in CHANNEL_BY_ID:
            raise ValueError(f"Unknown channel {name!r}.")
    if args.spatial_order_max < 1 or args.material_fit_modes < 1:
        raise ValueError("Spatial order and material fit mode count must be positive.")
    for optional_name in ("pulse_tau_fs", "start_sigma", "points_per_fastest_cycle"):
        if hasattr(args, optional_name):
            value = float(getattr(args, optional_name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"--{optional_name.replace('_', '-')} must be finite and positive."
                )


def _resolved_center_distance_nm(spec: ChannelSpec, gap_nm: float, args: argparse.Namespace) -> float:
    directional_radius = args.c_nm if spec.qd_placement == "axis" else args.a_nm
    return float(directional_radius + args.qd_radius_nm + gap_nm)


def _make_params(spec: ChannelSpec, gap_nm: float, args: argparse.Namespace):
    return make_params_with_overrides(
        c_nm=args.c_nm,
        a_nm=args.a_nm,
        r_nm=_resolved_center_distance_nm(spec, gap_nm, args),
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


def _build_channel(
    spec: ChannelSpec,
    gap_nm: float,
    args: argparse.Namespace,
    *,
    require_time_model: bool,
) -> BuiltChannel:
    params = _make_params(spec, gap_nm, args)
    dd_model = HybridQDPlasmonModel(
        params,
        orientation=spec.orientation,
        n_modes=args.material_fit_modes,
        fit_window_eV=(args.fit_min_ev, args.fit_max_ev),
        weight_center_eV=args.weight_center_ev,
        weight_sigma_eV=args.weight_sigma_ev,
        max_fit_normalized_rms=None,
        max_fit_pointwise_relative_error=None,
        radiative_consistency_policy=args.radiative_consistency_policy,
        verbose=args.verbose_fit,
    )
    if require_time_model:
        fit = dd_model.fit
        if (
            fit.normalized_rms_alpha > args.max_bright_fit_normalized_rms
            or fit.normalized_rms_inv_alpha > args.max_bright_fit_normalized_rms
            or fit.max_normalized_alpha_error
            > args.max_bright_fit_pointwise_relative_error
        ):
            _apply_policy(
                args.bright_fit_quality_policy,
                "The common material fit used by DD and FQS misses the configured "
                f"accuracy gate for {spec.channel_id}: NRMS(alpha)="
                f"{fit.normalized_rms_alpha:.6g}, NRMS(1/alpha)="
                f"{fit.normalized_rms_inv_alpha:.6g}, max normalized alpha error="
                f"{fit.max_normalized_alpha_error:.6g}.",
            )
    if spec.qd_placement == "axis":
        kernel = SpheroidGreenInteraction.from_params(
            params,
            orientation=spec.orientation,
            n_max=args.spatial_order_max,
        )
    else:
        kernel = EquatorialSpheroidGreenInteraction.from_params(
            params,
            orientation=spec.orientation,
            n_max=args.spatial_order_max,
        )

    reduction = None
    fqs_model = None
    if require_time_model:
        if spec.qd_placement == "side" and args.spatial_order_max > SIDE_DIRECT_REFERENCE_ORDER_MAX:
            reduction = build_positive_dark_reduction(
                dd_model,
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
                    f"Dark-kernel reduction was rejected for {spec.channel_id}, gap={gap_nm:g} nm."
                )
        fqs_model = FullQSSpheroidPulseModel(
            dd_model,
            kernel,
            fit_quality_policy=args.fit_quality_policy,
            max_modal_normalized_rms=args.max_modal_normalized_rms,
            max_modal_relative_error=args.max_modal_relative_error,
            modal_audit_points=args.modal_audit_points,
            spatial_convergence_policy=args.spatial_convergence_policy,
            spatial_convergence_rtol=args.spatial_convergence_rtol,
            dark_reduction=reduction,
            reduction_reaudit_points=args.reduction_reaudit_points,
            max_reduction_normalized_rms=args.reduction_rms_tolerance,
            max_reduction_normalized_error=args.reduction_max_tolerance,
        )
    return BuiltChannel(spec, float(gap_nm), params, dd_model, kernel, fqs_model, reduction)


def extract_feature(
    energy_eV: np.ndarray,
    spectrum: np.ndarray,
    *,
    center_eV: float,
    half_window_eV: float,
    competing_prominence_fraction: float = 0.5,
) -> FeatureResult:
    """Track the peak nearest the QD energy and measure half-prominence width."""

    energy = np.asarray(energy_eV, dtype=float)
    values = np.asarray(spectrum, dtype=float)
    if energy.ndim != 1 or values.shape != energy.shape or energy.size < 5:
        raise ValueError("Feature extraction needs matching 1-D arrays with at least five points.")
    if np.any(~np.isfinite(energy)) or np.any(np.diff(energy) <= 0.0):
        raise ValueError("energy_eV must be finite and strictly increasing.")
    if np.any(~np.isfinite(values)):
        return FeatureResult(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, "nonfinite", 0)
    mask = np.abs(energy - float(center_eV)) <= float(half_window_eV)
    indices = np.flatnonzero(mask)
    if indices.size < 5:
        return FeatureResult(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, "window_too_small", 0)
    lo, hi = int(indices[0]), int(indices[-1])
    local = values[lo : hi + 1]
    peaks, _ = find_peaks(local)
    if peaks.size == 0:
        return FeatureResult(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, "no_peak", 0)
    global_peaks = peaks + lo
    prominences = peak_prominences(local, peaks)[0]
    # A tiny numerical ripple closest to E_X must not replace the actual QD
    # feature.  Track the nearest peak among features whose prominence is at
    # least 10% of the strongest resolved prominence in the fixed window.
    credible = np.flatnonzero(prominences >= 0.1 * float(np.max(prominences)))
    distances = np.abs(energy[global_peaks[credible]] - float(center_eV))
    nearest = credible[
        np.flatnonzero(
            np.isclose(distances, np.min(distances), rtol=0.0, atol=1.0e-15)
        )
    ]
    if nearest.size > 1:
        selected_local_index = int(
            nearest[np.argmax(values[global_peaks[nearest]])]
        )
    else:
        selected_local_index = int(nearest[0])
    selected_local_peak = int(peaks[selected_local_index])
    selected_global_peak = int(global_peaks[selected_local_index])

    selected_prominence = float(prominences[selected_local_index])
    if not np.isfinite(selected_prominence) or selected_prominence <= 0.0:
        return FeatureResult(
            float(energy[selected_global_peak]),
            float(values[selected_global_peak]),
            selected_prominence,
            np.nan,
            np.nan,
            np.nan,
            "zero_prominence",
            0,
        )
    competing = int(
        np.count_nonzero(
            np.delete(prominences, selected_local_index)
            >= competing_prominence_fraction * selected_prominence
        )
    )
    selected_prominence_data = peak_prominences(
        local, np.asarray([selected_local_peak])
    )
    width_samples, _, left_ips, right_ips = peak_widths(
        local,
        np.asarray([selected_local_peak]),
        rel_height=0.5,
        prominence_data=(
            np.asarray([selected_prominence]),
            np.asarray([selected_prominence_data[1][0]]),
            np.asarray([selected_prominence_data[2][0]]),
        ),
    )
    del width_samples
    sample_axis = np.arange(local.size, dtype=float)
    left = float(np.interp(float(left_ips[0]), sample_axis, energy[lo : hi + 1]))
    right = float(np.interp(float(right_ips[0]), sample_axis, energy[lo : hi + 1]))
    status = "split_or_ambiguous" if competing else "ok"
    width = np.nan if competing else right - left
    return FeatureResult(
        float(energy[selected_global_peak]),
        float(values[selected_global_peak]),
        selected_prominence,
        float(width),
        left,
        right,
        status,
        competing,
    )


def threshold_from_curve(
    fluence_j_cm2: np.ndarray,
    population: np.ndarray,
    target: float,
) -> tuple[float, str, tuple[int, int] | None]:
    """Return the first upward crossing before the first resolved Rabi maximum."""

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
    peak_candidates = np.flatnonzero((differences[:-1] > 0.0) & (differences[1:] <= 0.0)) + 1
    last = int(peak_candidates[0]) if peak_candidates.size else values.size - 1
    for upper in range(1, last + 1):
        if values[upper - 1] < target <= values[upper]:
            x0, x1 = np.sqrt(fluence[[upper - 1, upper]])
            y0, y1 = values[[upper - 1, upper]]
            x = x0 + (target - y0) * (x1 - x0) / (y1 - y0)
            return float(x * x), "resolved", (upper - 1, upper)
    if peak_candidates.size:
        return np.nan, "not_reached_first_lobe", None
    return np.nan, "right_censored", None


def dd_validity_distance(
    gap_nm: np.ndarray,
    discrepancy: np.ndarray,
    tolerance: float,
) -> float:
    """Smallest gap after which every larger sampled gap satisfies tolerance."""

    gap = np.asarray(gap_nm, dtype=float)
    delta = np.asarray(discrepancy, dtype=float)
    if gap.ndim != 1 or delta.shape != gap.shape or np.any(np.diff(gap) <= 0.0):
        raise ValueError("gap/discrepancy arrays must match and gaps must increase.")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive.")
    accepted = np.isfinite(delta) & (delta <= float(tolerance))
    suffix = np.logical_and.accumulate(accepted[::-1])[::-1]
    indices = np.flatnonzero(suffix)
    return float(gap[indices[0]]) if indices.size else np.nan


def _pulse_for_fluence(
    fluence_j_cm2: float,
    carrier_energy_eV: float,
    args: argparse.Namespace,
) -> GaussianPulse:
    reference = GaussianPulse(
        E0_au=1.0,
        omegaL_au=float(eV_to_au(carrier_energy_eV)),
        tau_au=float(fs_to_au(args.pulse_tau_fs)),
        tau_kind=args.pulse_tau_kind,
    )
    amplitude = np.sqrt(float(fluence_j_cm2) / reference.fluence_j_cm2(eps_m=args.eps_m))
    pulse = GaussianPulse(
        E0_au=float(amplitude),
        omegaL_au=float(eV_to_au(carrier_energy_eV)),
        tau_au=float(fs_to_au(args.pulse_tau_fs)),
        tau_kind=args.pulse_tau_kind,
    )
    achieved = pulse.fluence_j_cm2(eps_m=args.eps_m)
    if not np.isclose(achieved, fluence_j_cm2, rtol=5.0e-13, atol=0.0):
        raise RuntimeError("Fluence-to-field round-trip failed.")
    return pulse


def _bare_settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "method": args.method,
        "rtol": args.rtol,
        "atol": args.atol,
        "points_per_fastest_cycle": args.points_per_fastest_cycle,
        "positivity_tolerance": args.positivity_tolerance,
        "positivity_policy": args.positivity_policy,
        "tail_window_fraction": args.tail_window_fraction,
        "tail_ratio_tolerance": args.tail_ratio_tolerance,
        "fit_min_ev": args.fit_min_ev,
        "fit_max_ev": args.fit_max_ev,
    }


def _solve_population(
    kind: Literal["dd", "fqs"],
    bundle: BuiltChannel,
    pulse: GaussianPulse,
    t_span_au: tuple[float, float],
    args: argparse.Namespace,
) -> tuple[float, dict[str, Any]]:
    if kind == "dd":
        result = bundle.dd_model.solve(
            pulse,
            method=args.method,
            rtol=args.rtol,
            atol=args.atol,
            t_span_au=t_span_au,
            positivity_tol=args.positivity_tolerance,
            positivity_policy=args.positivity_policy,
            spectral_window_policy=args.spectral_window_policy,
            max_spectral_leakage=args.max_spectral_leakage,
        )
        rho = 0.5 * (result.y[2 * bundle.dd_model.n_modes] + 1.0)
        diagnostics = result.diagnostics
    else:
        if bundle.fqs_model is None:
            raise RuntimeError("FQS time model was not built.")
        result = bundle.fqs_model.solve(
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
            response_tail_tolerance=args.tail_ratio_tolerance,
            response_tail_window_fraction=args.tail_window_fraction,
        )
        rho = result.rho22
        diagnostics = result.diagnostics
    tail_converged = bool(getattr(diagnostics, "response_tail_converged", True))
    tail_ratio = float(getattr(diagnostics, "response_tail_ratio", np.nan))
    if not tail_converged:
        _apply_policy(
            args.tail_policy,
            f"{kind.upper()} response tail did not converge for {bundle.spec.channel_id}, "
            f"gap={bundle.gap_nm:g} nm: ratio={tail_ratio:.6g}.",
        )
    return float(rho[-1]), {
        "tail_converged": tail_converged,
        "tail_ratio": tail_ratio,
        "nfev": int(getattr(diagnostics, "nfev", -1)),
        "population_min": float(np.min(rho)),
        "population_max": float(np.max(rho)),
    }


def _solve_bare_population(
    pulse: GaussianPulse,
    params: Any,
    t_span_au: tuple[float, float],
    args: argparse.Namespace,
) -> tuple[float, dict[str, Any]]:
    _, rho, diagnostics = _solve_bare_qd(
        pulse, params, t_span_au, _bare_settings(args)
    )
    tail_converged = bool(diagnostics["response_tail_converged"])
    tail_ratio = float(diagnostics["response_tail_ratio"])
    if not tail_converged:
        _apply_policy(
            args.tail_policy,
            "Isolated-QD coherence tail did not converge at the common read "
            f"time: ratio={tail_ratio:.6g}.",
        )
    return float(rho[-1]), {
        "tail_converged": tail_converged,
        "tail_ratio": tail_ratio,
        "nfev": int(diagnostics["nfev"]),
        "population_min": float(np.min(rho)),
        "population_max": float(np.max(rho)),
    }


def _common_time_span(bundles: list[BuiltChannel], pulse: GaussianPulse, args: argparse.Namespace) -> tuple[float, float]:
    start = -float(args.start_sigma) * pulse.sigma_t_au
    if args.post_fs is not None:
        end = float(fs_to_au(args.post_fs))
    else:
        candidates = [float(args.start_sigma) * pulse.sigma_t_au]
        for bundle in bundles:
            if bundle.fqs_model is None:
                raise RuntimeError("Automatic common read time requires FQS time models.")
            candidates.append(bundle.dd_model.recommended_post_pulse_time_au())
            candidates.append(bundle.fqs_model.recommended_post_pulse_time_au())
        end = max(candidates)
    if max(pulse.envelope(np.asarray([start, end]))) > 1.0e-6:
        raise ValueError("The common time span truncates the incident Gaussian pulse.")
    reference_params = bundles[0].params
    decay_fraction = float(-np.expm1(-reference_params.gamma_au * end))
    if decay_fraction > args.max_population_decay_fraction_at_read:
        _apply_policy(
            args.population_decay_policy,
            "The common read time is not safely before population relaxation: "
            f"free-decay fraction={decay_fraction:.6g}, limit="
            f"{args.max_population_decay_fraction_at_read:.6g}.",
        )
    return float(start), float(end)


def _base_metadata(
    args: argparse.Namespace,
    *,
    metric: str,
    computation_family: str,
    generator_path: Path,
) -> dict[str, Any]:
    channels = [CHANNEL_BY_ID[name] for name in args.channels]
    surface_relation = {
        "axis_long": "normal",
        "axis_trans": "tangential",
        "side_long": "tangential",
        "side_trans_radial": "normal",
        "side_trans_tangential": "tangential",
    }
    channel_documents = []
    for item in channels:
        document = asdict(item)
        document["field_relative_to_particle_axis"] = item.orientation
        document["field_relative_to_local_surface"] = surface_relation[
            item.channel_id
        ]
        channel_documents.append(document)
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now().astimezone().isoformat(),
        "primary_metric": metric,
        "computation_family": computation_family,
        "model_ids": MODEL_IDS.tolist(),
        "model_api": {
            "dd_time": "qd_mnp_rational_fit.HybridQDPlasmonModel.solve",
            "dd_frequency": "qd_mnp_spheroid_green.LegacyDipoleInteraction.frequency_response",
            "fqs_time": "qd_mnp_full_qs_model.FullQSSpheroidPulseModel.solve",
            "fqs_frequency": "analytic spheroidal Green kernel response_from_material",
            "linear_coupling": "qd_mnp_spheroid_green.solve_linear_hybrid_response",
        },
        "channels": channel_documents,
        "geometry_convention": {
            "gap": "surface-to-surface distance",
            "axis_center_distance_nm": "c_nm + qd_radius_nm + gap_nm",
            "side_center_distance_nm": "a_nm + qd_radius_nm + gap_nm",
            "field_only": True,
            "laser_propagation_direction_included": False,
        },
        "requested_arguments": _json_ready(vars(args)),
        "physical_constants": _physical_constants(),
        "material": {
            "interpolation": MATERIAL_INTERPOLATION,
            "high_frequency_epsilon": float(MATERIAL_HIGH_FREQUENCY_EPSILON),
        },
        "model_profile": NATIVE_MODEL_PROFILE,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
        "provenance": {
            "git": _git_provenance(),
            "source_sha256": _source_hashes((generator_path,)),
            "generator": str(generator_path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
        "limitations": [
            "local quasistatic homogeneous spheroid and point-like QD",
            "fixed phenomenological QD gamma1 and Gamma2; no LDOS-derived Purcell correction",
            "no propagation direction, retardation, radiation pattern, nonlocality or charge transfer",
            "field polarization labels describe E only",
        ],
    }


def _flatten_spatial_data(bundles: list[BuiltChannel]) -> dict[str, np.ndarray]:
    exact_offsets = [0]
    exact_degree: list[np.ndarray] = []
    exact_order: list[np.ndarray] = []
    exact_sector: list[np.ndarray] = []
    exact_depolarization: list[np.ndarray] = []
    exact_weight: list[np.ndarray] = []
    dynamic_offsets = [0]
    dynamic_depolarization: list[np.ndarray] = []
    dynamic_weight: list[np.ndarray] = []
    for bundle in bundles:
        kernel = bundle.kernel
        depolarization = np.asarray(
            kernel.depolarization_by_mode
            if hasattr(kernel, "depolarization_by_mode")
            else kernel.depolarization_by_degree,
            dtype=float,
        )
        weight = np.asarray(
            kernel.reaction_weight_by_mode_au_minus3
            if hasattr(kernel, "reaction_weight_by_mode_au_minus3")
            else kernel.reaction_weight_by_degree_au_minus3,
            dtype=float,
        )
        degree = np.asarray(
            kernel.mode_degrees if hasattr(kernel, "mode_degrees") else kernel.degrees,
            dtype=np.int64,
        )
        if hasattr(kernel, "mode_orders"):
            order = np.asarray(kernel.mode_orders, dtype=np.int64)
            sector = np.asarray(kernel.mode_sectors, dtype="U8")
        else:
            order = np.full(
                degree.shape,
                0 if bundle.spec.orientation == "long" else 1,
                dtype=np.int64,
            )
            sector = np.full(degree.shape, "axis", dtype="U8")
        exact_degree.append(degree)
        exact_order.append(order)
        exact_sector.append(sector)
        exact_depolarization.append(depolarization)
        exact_weight.append(weight)
        exact_offsets.append(exact_offsets[-1] + degree.size)

        if bundle.fqs_model is not None:
            dynamic_dep = np.asarray(
                bundle.fqs_model.modal_depolarization, dtype=float
            )
            dynamic_w = np.asarray(
                bundle.fqs_model.modal_reaction_weights_au_minus3, dtype=float
            )
        else:
            dynamic_dep = np.empty(0, dtype=float)
            dynamic_w = np.empty(0, dtype=float)
        dynamic_depolarization.append(dynamic_dep)
        dynamic_weight.append(dynamic_w)
        dynamic_offsets.append(dynamic_offsets[-1] + dynamic_dep.size)
    return {
        "spatial_bundle_offsets": np.asarray(exact_offsets, dtype=np.int64),
        "spatial_mode_degree": np.concatenate(exact_degree),
        "spatial_mode_order": np.concatenate(exact_order),
        "spatial_mode_sector": np.concatenate(exact_sector),
        "spatial_mode_depolarization": np.concatenate(exact_depolarization),
        "spatial_mode_reaction_weight_au_minus3": np.concatenate(exact_weight),
        "dynamic_spatial_bundle_offsets": np.asarray(
            dynamic_offsets, dtype=np.int64
        ),
        "dynamic_spatial_depolarization": np.concatenate(dynamic_depolarization),
        "dynamic_spatial_reaction_weight_au_minus3": np.concatenate(
            dynamic_weight
        ),
    }


def _spectral_metrics(
    energy_eV: np.ndarray,
    isolated: np.ndarray,
    spectra: np.ndarray,
    *,
    center_eV: float,
    half_window_eV: float,
) -> dict[str, np.ndarray]:
    # spectra dimensions: model, channel, gap, energy
    isolated_feature = extract_feature(
        energy_eV,
        isolated,
        center_eV=center_eV,
        half_window_eV=half_window_eV,
    )
    shape = spectra.shape[:-1]
    peak_energy = np.full(shape, np.nan)
    peak_height = np.full(shape, np.nan)
    prominence = np.full(shape, np.nan)
    width = np.full(shape, np.nan)
    left = np.full(shape, np.nan)
    right = np.full(shape, np.nan)
    status = np.full(shape, "not_evaluated", dtype="U32")
    competing = np.zeros(shape, dtype=np.int64)
    for index in np.ndindex(shape):
        feature = extract_feature(
            energy_eV,
            spectra[index],
            center_eV=center_eV,
            half_window_eV=half_window_eV,
        )
        peak_energy[index] = feature.energy_eV
        peak_height[index] = feature.height
        prominence[index] = feature.prominence
        width[index] = feature.width_eV
        left[index] = feature.left_eV
        right[index] = feature.right_eV
        status[index] = feature.status
        competing[index] = feature.competing_peak_count

    window_mask = np.abs(energy_eV - float(center_eV)) <= float(half_window_eV)
    denominator_peak = float(np.max(isolated[window_mask]))
    optimized_height = np.max(spectra[..., window_mask], axis=-1)
    gain_optimized = (
        optimized_height / denominator_peak
        if denominator_peak > 0.0
        else np.full(shape, np.nan)
    )
    isolated_at_center = float(np.interp(center_eV, energy_eV, isolated))
    fixed = np.empty(shape, dtype=float)
    for index in np.ndindex(shape):
        fixed[index] = float(np.interp(center_eV, energy_eV, spectra[index]))
    gain_fixed = fixed / isolated_at_center if isolated_at_center > 0.0 else np.full(shape, np.nan)
    gamma0 = isolated_feature.width_eV
    feature_is_unique = status == "ok"
    if (
        isolated_feature.status == "ok"
        and np.isfinite(gamma0)
        and gamma0 > 0.0
    ):
        shift_ratio = np.where(
            feature_is_unique,
            (peak_energy - isolated_feature.energy_eV) / gamma0,
            np.nan,
        )
    else:
        shift_ratio = np.full(shape, np.nan)
    width_ratio = width / gamma0 if np.isfinite(gamma0) and gamma0 > 0.0 else np.full(shape, np.nan)

    dd = spectra[0][..., window_mask]
    fqs = spectra[1][..., window_mask]
    discrepancy_energy = energy_eV[window_mask]
    numerator = np.trapezoid(
        (dd - fqs) ** 2, discrepancy_energy, axis=-1
    )
    denominator = np.trapezoid(fqs**2, discrepancy_energy, axis=-1)
    delta_l2 = np.sqrt(
        np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan),
            where=denominator > np.finfo(float).tiny,
        )
    )
    max_fqs = np.max(np.abs(fqs), axis=-1)
    delta_linf = np.divide(
        np.max(np.abs(dd - fqs), axis=-1),
        max_fqs,
        out=np.full_like(max_fqs, np.nan),
        where=max_fqs > np.finfo(float).tiny,
    )
    return {
        "isolated_peak_energy_eV": np.asarray(isolated_feature.energy_eV),
        "isolated_peak_height": np.asarray(isolated_feature.height),
        "isolated_peak_prominence": np.asarray(isolated_feature.prominence),
        "isolated_fwhm_eV": np.asarray(isolated_feature.width_eV),
        "isolated_feature_status": np.asarray(isolated_feature.status),
        "peak_energy_eV": peak_energy,
        "peak_height": peak_height,
        "peak_prominence": prominence,
        "fwhm_eV": width,
        "half_prominence_left_eV": left,
        "half_prominence_right_eV": right,
        "feature_status": status,
        "competing_peak_count": competing,
        "excitation_gain_optimized": gain_optimized,
        "optimized_window_peak_height": optimized_height,
        "excitation_peak_difference_from_isolated": (
            optimized_height - denominator_peak
        ),
        "excitation_gain_at_reference_energy": gain_fixed,
        "excitation_difference_at_reference_energy": (
            fixed - isolated_at_center
        ),
        "resonance_shift_over_gamma0": shift_ratio,
        "spectral_width_over_gamma0": width_ratio,
        "spectral_l2_relative_dd_vs_fqs": delta_l2,
        "spectral_linf_relative_dd_vs_fqs": delta_linf,
        "dd_minus_fqs_peak_shift_over_gamma0": shift_ratio[0] - shift_ratio[1],
        "dd_minus_fqs_width_over_gamma0": width_ratio[0] - width_ratio[1],
    }


def compute_spectral_payload(
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    _validate_common_inputs(args)
    if not (
        np.isfinite(args.energy_min_ev)
        and np.isfinite(args.energy_max_ev)
        and 0.0 < args.energy_min_ev < args.energy_max_ev
    ):
        raise ValueError("Energy bounds must satisfy 0 < minimum < maximum.")
    if args.energy_points < 5:
        raise ValueError("--energy-points must be at least five.")
    if not np.isfinite(args.feature_half_window_ev) or args.feature_half_window_ev <= 0.0:
        raise ValueError("--feature-half-window-ev must be finite and positive.")
    if not np.isfinite(args.weak_fluence_j_cm2) or args.weak_fluence_j_cm2 <= 0.0:
        raise ValueError("--weak-fluence-j-cm2 must be finite and positive.")
    if not np.isfinite(args.weak_linearity_check_factor) or not 0.0 < args.weak_linearity_check_factor < 1.0:
        raise ValueError("--weak-linearity-check-factor must lie in (0, 1).")
    if not np.isfinite(args.max_weak_linearity_relative_error) or args.max_weak_linearity_relative_error <= 0.0:
        raise ValueError("--max-weak-linearity-relative-error must be finite and positive.")
    if not np.isfinite(args.max_weak_population) or not 0.0 < args.max_weak_population < 1.0:
        raise ValueError("--max-weak-population must lie in (0, 1).")
    if not np.isfinite(args.max_energy_step_over_gamma0) or args.max_energy_step_over_gamma0 <= 0.0:
        raise ValueError("--max-energy-step-over-gamma0 must be finite and positive.")
    energy = np.linspace(args.energy_min_ev, args.energy_max_ev, args.energy_points)
    if energy.size < 5 or energy[0] >= energy[-1]:
        raise ValueError("The energy grid must contain at least five increasing points.")
    if not energy[0] <= args.feature_center_ev <= energy[-1]:
        raise ValueError("--feature-center-ev must lie inside the energy grid.")
    channels = [CHANNEL_BY_ID[name] for name in args.channels]
    gaps = np.asarray(args.gaps_nm, dtype=float)
    bundles: list[BuiltChannel] = []
    need_time = args.spectral_observable == "weak_pulse_excitation"
    for spec in channels:
        for gap in gaps:
            print(f"Building DD/FQS {spec.channel_id}, gap={gap:g} nm ...", flush=True)
            bundles.append(_build_channel(spec, float(gap), args, require_time_model=need_time))

    n_model, n_channel, n_gap, n_energy = 2, len(channels), gaps.size, energy.size
    spectra = np.empty((n_model, n_channel, n_gap, n_energy), dtype=float)
    qd_transfer = np.full((n_model, n_channel, n_gap, n_energy), np.nan + 0j)
    alpha_effective = np.full_like(qd_transfer, np.nan + 0j)
    interaction_A = np.full_like(qd_transfer, np.nan + 0j)
    interaction_B = np.full_like(qd_transfer, np.nan + 0j)
    interaction_K = np.full_like(qd_transfer, np.nan + 0j)
    half_order_error = np.empty((n_channel, n_gap), dtype=float)
    tail_error = np.empty((n_channel, n_gap), dtype=float)
    center_distance = np.empty((n_channel, n_gap), dtype=float)
    tail_converged = np.zeros((n_model, n_channel, n_gap, n_energy), dtype=bool)
    tail_ratio = np.full((n_model, n_channel, n_gap, n_energy), np.nan)
    isolated_tail_converged = np.zeros(n_energy, dtype=bool)
    isolated_tail_ratio = np.full(n_energy, np.nan)
    weak_check_spectra = np.full_like(spectra, np.nan)
    weak_check_isolated = np.full(n_energy, np.nan)
    weak_main_population_max = np.nan
    weak_time_span_au = np.full(2, np.nan)
    weak_main_pulse_e0_au = np.full(n_energy, np.nan)
    weak_check_pulse_e0_au = np.full(n_energy, np.nan)
    weak_main_peak_intensity_w_cm2 = np.full(n_energy, np.nan)
    weak_check_peak_intensity_w_cm2 = np.full(n_energy, np.nan)

    reference_params = bundles[0].params
    beta_isolated = qd_linear_polarizability_from_params(reference_params, energy)
    if args.spectral_observable == "linear_qd_response":
        isolated = np.abs(beta_isolated) ** 2
        for flat_index, bundle in enumerate(bundles):
            ci, gi = divmod(flat_index, n_gap)
            center_distance[ci, gi] = _resolved_center_distance_nm(
                bundle.spec, bundle.gap_nm, args
            )
            dd_response = LegacyDipoleInteraction(bundle.dd_model).frequency_response(
                energy, mnp_response="material"
            )
            fqs_response = bundle.kernel.response_from_material(bundle.params.material, energy)
            for mi, response in enumerate((dd_response, fqs_response)):
                coupled = solve_linear_hybrid_response(
                    response,
                    qd_linear_polarizability_from_params(bundle.params, energy),
                    eps_m=bundle.params.eps_m,
                )
                qd_transfer[mi, ci, gi] = coupled.qd_dipole_over_field_au3
                alpha_effective[mi, ci, gi] = coupled.alpha_effective_au3
                interaction_A[mi, ci, gi] = response.A_au3
                interaction_B[mi, ci, gi] = response.B
                interaction_K[mi, ci, gi] = response.K_au_minus3
                spectra[mi, ci, gi] = np.abs(coupled.qd_dipole_over_field_au3) ** 2
            half_order_error[ci, gi] = float(
                np.max(fqs_response.relative_half_order_change())
            )
            tail_error[ci, gi] = float(np.max(fqs_response.relative_tail_block()))
            if half_order_error[ci, gi] > args.spatial_convergence_rtol:
                _apply_policy(
                    args.spatial_convergence_policy,
                    f"FQS spatial half-order change={half_order_error[ci, gi]:.6g} "
                    f"for {bundle.spec.channel_id}, gap={bundle.gap_nm:g} nm exceeds "
                    f"{args.spatial_convergence_rtol:.6g}.",
                )
    else:
        reference_pulse = _pulse_for_fluence(
            args.weak_fluence_j_cm2,
            float(energy[0]),
            args,
        )
        t_span = _common_time_span(bundles, reference_pulse, args)
        weak_time_span_au[:] = t_span
        isolated = np.empty(n_energy, dtype=float)
        main_population_maximum = 0.0
        check_fluence = args.weak_fluence_j_cm2 * args.weak_linearity_check_factor
        for ei, carrier in enumerate(energy):
            pulse = _pulse_for_fluence(args.weak_fluence_j_cm2, float(carrier), args)
            weak_main_pulse_e0_au[ei] = pulse.E0_au
            weak_main_peak_intensity_w_cm2[ei] = pulse.peak_intensity_w_cm2(
                eps_m=args.eps_m
            )
            bare_population, bare_diagnostics = _solve_bare_population(
                pulse, reference_params, t_span, args
            )
            isolated[ei] = bare_population / args.weak_fluence_j_cm2
            isolated_tail_converged[ei] = bare_diagnostics["tail_converged"]
            isolated_tail_ratio[ei] = bare_diagnostics["tail_ratio"]
            main_population_maximum = max(
                main_population_maximum,
                float(bare_diagnostics["population_max"]),
            )
            check_pulse = _pulse_for_fluence(check_fluence, float(carrier), args)
            weak_check_pulse_e0_au[ei] = check_pulse.E0_au
            weak_check_peak_intensity_w_cm2[ei] = check_pulse.peak_intensity_w_cm2(
                eps_m=args.eps_m
            )
            bare_check_population, _ = _solve_bare_population(
                check_pulse, reference_params, t_span, args
            )
            weak_check_isolated[ei] = bare_check_population / check_fluence
            for flat_index, bundle in enumerate(bundles):
                ci, gi = divmod(flat_index, n_gap)
                center_distance[ci, gi] = _resolved_center_distance_nm(
                    bundle.spec, bundle.gap_nm, args
                )
                for mi, kind in enumerate(("dd", "fqs")):
                    population, diagnostics = _solve_population(
                        kind,
                        bundle,
                        pulse,
                        t_span,
                        args,
                    )
                    spectra[mi, ci, gi, ei] = population / args.weak_fluence_j_cm2
                    tail_converged[mi, ci, gi, ei] = diagnostics["tail_converged"]
                    tail_ratio[mi, ci, gi, ei] = diagnostics["tail_ratio"]
                    main_population_maximum = max(
                        main_population_maximum,
                        float(diagnostics["population_max"]),
                    )
                    check_population, _ = _solve_population(
                        kind,
                        bundle,
                        check_pulse,
                        t_span,
                        args,
                    )
                    weak_check_spectra[mi, ci, gi, ei] = (
                        check_population / check_fluence
                    )
            print(f"[{ei + 1}/{n_energy}] carrier={carrier:.6g} eV", flush=True)
        weak_main_population_max = float(main_population_maximum)
        if weak_main_population_max > args.max_weak_population:
            _apply_policy(
                args.weak_linearity_policy,
                "The requested weak-pulse spectrum is not safely weak-field: "
                f"maximum population={weak_main_population_max:.6g}, allowed="
                f"{args.max_weak_population:.6g}.",
            )
        # Save the direct-material frequency responses as diagnostics too.  The
        # pulse population remains the primary observable and uses the common
        # causal fit; these arrays let the artifact expose the spatial A/B/K
        # difference without another expensive pulse solve.
        for flat_index, bundle in enumerate(bundles):
            ci, gi = divmod(flat_index, n_gap)
            dd_response = LegacyDipoleInteraction(bundle.dd_model).frequency_response(
                energy, mnp_response="material"
            )
            fqs_response = bundle.kernel.response_from_material(
                bundle.params.material, energy
            )
            beta = qd_linear_polarizability_from_params(bundle.params, energy)
            for mi, response in enumerate((dd_response, fqs_response)):
                coupled = solve_linear_hybrid_response(
                    response, beta, eps_m=bundle.params.eps_m
                )
                qd_transfer[mi, ci, gi] = coupled.qd_dipole_over_field_au3
                alpha_effective[mi, ci, gi] = coupled.alpha_effective_au3
                interaction_A[mi, ci, gi] = response.A_au3
                interaction_B[mi, ci, gi] = response.B
                interaction_K[mi, ci, gi] = response.K_au_minus3
            half_order_error[ci, gi] = float(
                np.max(fqs_response.relative_half_order_change())
            )
            tail_error[ci, gi] = float(
                np.max(fqs_response.relative_tail_block())
            )
            if half_order_error[ci, gi] > args.spatial_convergence_rtol:
                _apply_policy(
                    args.spatial_convergence_policy,
                    f"FQS spatial half-order change={half_order_error[ci, gi]:.6g} "
                    f"for {bundle.spec.channel_id}, gap={bundle.gap_nm:g} nm exceeds "
                    f"{args.spatial_convergence_rtol:.6g}.",
                )

    if args.spectral_observable == "weak_pulse_excitation":
        isolated_scale = max(
            float(np.sqrt(np.mean(weak_check_isolated**2))),
            np.finfo(float).tiny,
        )
        weak_linearity_isolated_error = float(
            np.sqrt(np.mean((isolated - weak_check_isolated) ** 2))
            / isolated_scale
        )
        weak_linearity_error = np.sqrt(
            np.mean((spectra - weak_check_spectra) ** 2, axis=-1)
        ) / np.maximum(
            np.sqrt(np.mean(weak_check_spectra**2, axis=-1)),
            np.finfo(float).tiny,
        )
        maximum_linearity_error = max(
            weak_linearity_isolated_error,
            float(np.max(weak_linearity_error)),
        )
        if maximum_linearity_error > args.max_weak_linearity_relative_error:
            _apply_policy(
                args.weak_linearity_policy,
                "P_exc/fluence did not converge under the weak-fluence check: "
                f"max normalized RMS change={maximum_linearity_error:.6g}, allowed="
                f"{args.max_weak_linearity_relative_error:.6g}.",
            )
    else:
        weak_linearity_isolated_error = np.nan
        weak_linearity_error = np.full((n_model, n_channel, n_gap), np.nan)

    derived = _spectral_metrics(
        energy,
        isolated,
        spectra,
        center_eV=args.feature_center_ev,
        half_window_eV=args.feature_half_window_ev,
    )
    a_scale = np.maximum(
        np.abs(interaction_A[1]),
        np.finfo(float).tiny,
    )
    max_relative_a_disagreement = float(
        np.max(np.abs(interaction_A[0] - interaction_A[1]) / a_scale)
    )
    if max_relative_a_disagreement > 1.0e-10:
        raise RuntimeError(
            "DD and FQS did not receive the same bright MNP polarizability A: "
            f"maximum relative disagreement={max_relative_a_disagreement:.6g}."
        )
    gamma0 = float(np.asarray(derived["isolated_fwhm_eV"]))
    max_step = float(np.max(np.diff(energy)))
    resolution_ratio = max_step / gamma0 if np.isfinite(gamma0) and gamma0 > 0.0 else np.inf
    resolution_accepted = bool(resolution_ratio <= args.max_energy_step_over_gamma0)
    if not resolution_accepted:
        _apply_policy(
            args.energy_resolution_policy,
            "The extracted isolated-QD feature is not energy-grid resolved: "
            f"max dE/Gamma0={resolution_ratio:.6g}, allowed="
            f"{args.max_energy_step_over_gamma0:.6g}.",
        )
    validity_gap = np.asarray(
        [
            dd_validity_distance(
                gaps,
                derived["spectral_l2_relative_dd_vs_fqs"][ci],
                args.dd_tolerance,
            )
            for ci in range(n_channel)
        ],
        dtype=float,
    )
    first = bundles[0]
    material = first.params.material
    time_model_payload: dict[str, np.ndarray] = {}
    if need_time:
        fqs_models = [bundle.fqs_model for bundle in bundles]
        if any(model is None for model in fqs_models):
            raise RuntimeError("Weak-pulse spectra require every FQS time model.")
        fit_alpha_inf = np.asarray(
            [bundle.dd_model.fit.alpha_inf for bundle in bundles], dtype=float
        ).reshape(n_channel, n_gap)
        fit_strengths = np.stack(
            [bundle.dd_model.fit.strengths_au2 for bundle in bundles]
        ).reshape(n_channel, n_gap, args.material_fit_modes)
        fit_omega = np.stack(
            [bundle.dd_model.fit.omega_modes_au for bundle in bundles]
        ).reshape(n_channel, n_gap, args.material_fit_modes)
        fit_gamma = np.stack(
            [bundle.dd_model.fit.gamma_modes_au for bundle in bundles]
        ).reshape(n_channel, n_gap, args.material_fit_modes)
        time_model_payload = {
            "fit_alpha_inf": fit_alpha_inf,
            "fit_strengths_au2": fit_strengths,
            "fit_omega_modes_au": fit_omega,
            "fit_gamma_modes_au": fit_gamma,
            "fit_omega_modes_eV": np.asarray(au_to_eV(fit_omega)),
            "fit_gamma_modes_eV": np.asarray(au_to_eV(fit_gamma)),
            "fit_normalized_rms_alpha": np.asarray(
                [bundle.dd_model.fit.normalized_rms_alpha for bundle in bundles]
            ).reshape(n_channel, n_gap),
            "fit_normalized_rms_inverse_alpha": np.asarray(
                [bundle.dd_model.fit.normalized_rms_inv_alpha for bundle in bundles]
            ).reshape(n_channel, n_gap),
            "fit_max_normalized_alpha_error": np.asarray(
                [bundle.dd_model.fit.max_normalized_alpha_error for bundle in bundles]
            ).reshape(n_channel, n_gap),
            "fqs_spatial_convergence_accepted": np.asarray(
                [model.spatial_convergence_diagnostics.accepted for model in fqs_models],
                dtype=bool,
            ).reshape(n_channel, n_gap),
            "fqs_modal_fit_normalized_rms": np.asarray(
                [model.modal_fit_diagnostics.max_normalized_rms for model in fqs_models]
            ).reshape(n_channel, n_gap),
            "fqs_modal_fit_max_relative_error": np.asarray(
                [model.modal_fit_diagnostics.max_relative_error for model in fqs_models]
            ).reshape(n_channel, n_gap),
            "fqs_modal_fit_accepted": np.asarray(
                [model.modal_fit_diagnostics.accepted for model in fqs_models],
                dtype=bool,
            ).reshape(n_channel, n_gap),
            "fqs_coupled_stability_accepted": np.asarray(
                [model.coupled_stability.stable for model in fqs_models],
                dtype=bool,
            ).reshape(n_channel, n_gap),
        }
    payload = {
        "channel_id": np.asarray(args.channels, dtype="U32"),
        "model_id": MODEL_IDS,
        "gap_nm": gaps,
        "center_distance_nm": center_distance,
        "energy_eV": energy,
        "isolated_qd_spectrum": isolated,
        "qd_excitation_spectrum": spectra,
        "weak_check_isolated_qd_spectrum": weak_check_isolated,
        "weak_check_qd_excitation_spectrum": weak_check_spectra,
        "weak_linearity_isolated_normalized_rms": np.asarray(
            weak_linearity_isolated_error
        ),
        "weak_linearity_normalized_rms": weak_linearity_error,
        "weak_main_population_max": np.asarray(weak_main_population_max),
        "weak_main_fluence_j_cm2": np.asarray(args.weak_fluence_j_cm2),
        "weak_check_fluence_j_cm2": np.asarray(
            args.weak_fluence_j_cm2 * args.weak_linearity_check_factor
        ),
        "weak_time_span_au": weak_time_span_au,
        "weak_read_time_fs": np.asarray(au_to_fs(weak_time_span_au[1])),
        "weak_main_pulse_e0_au": weak_main_pulse_e0_au,
        "weak_check_pulse_e0_au": weak_check_pulse_e0_au,
        "weak_main_peak_intensity_w_cm2": weak_main_peak_intensity_w_cm2,
        "weak_check_peak_intensity_w_cm2": weak_check_peak_intensity_w_cm2,
        "qd_dipole_over_field_au3": qd_transfer,
        "alpha_effective_au3": alpha_effective,
        "interaction_A_au3": interaction_A,
        "interaction_B": interaction_B,
        "interaction_K_au_minus3": interaction_K,
        "fqs_max_relative_half_order_change": half_order_error,
        "fqs_max_relative_tail_block": tail_error,
        "pulse_tail_converged": tail_converged,
        "pulse_tail_ratio": tail_ratio,
        "isolated_pulse_tail_converged": isolated_tail_converged,
        "isolated_pulse_tail_ratio": isolated_tail_ratio,
        "energy_resolution_ratio_dE_over_gamma0": np.asarray(resolution_ratio),
        "energy_resolution_accepted": np.asarray(resolution_accepted),
        "max_relative_A_disagreement_dd_vs_fqs": np.asarray(
            max_relative_a_disagreement
        ),
        "dd_validity_gap_nm": validity_gap,
        "material_energy_eV": np.asarray(material.energy_eV),
        "material_n": np.asarray(material.n),
        "material_k": np.asarray(material.k),
        **_flatten_spatial_data(bundles),
        **time_model_payload,
        **derived,
    }
    units = {
        "gap_nm": "nm surface-to-surface",
        "center_distance_nm": "nm centre-to-centre",
        "energy_eV": "eV",
        "isolated_qd_spectrum": "bohr^6" if args.spectral_observable == "linear_qd_response" else "(J cm^-2)^-1",
        "qd_excitation_spectrum": "bohr^6" if args.spectral_observable == "linear_qd_response" else "(J cm^-2)^-1",
        "qd_dipole_over_field_au3": "bohr^3",
        "alpha_effective_au3": "bohr^3",
        "interaction_A_au3": "bohr^3",
        "interaction_B": "1",
        "interaction_K_au_minus3": "bohr^-3",
        "weak_time_span_au": "atomic time",
        "weak_read_time_fs": "fs",
        "weak_main_pulse_e0_au": "atomic electric field",
        "weak_check_pulse_e0_au": "atomic electric field",
        "weak_main_peak_intensity_w_cm2": "W cm^-2",
        "weak_check_peak_intensity_w_cm2": "W cm^-2",
        "peak_energy_eV": "eV",
        "fwhm_eV": "eV",
        "excitation_gain_optimized": "1",
        "excitation_gain_at_reference_energy": "1",
        "resonance_shift_over_gamma0": "1",
        "spectral_width_over_gamma0": "1",
        "spectral_l2_relative_dd_vs_fqs": "1",
        "spectral_linf_relative_dd_vs_fqs": "1",
    }
    if need_time:
        units.update(
            {
                "fit_alpha_inf": "1",
                "fit_strengths_au2": "atomic frequency squared",
                "fit_omega_modes_au": "atomic frequency",
                "fit_gamma_modes_au": "atomic frequency",
                "fit_omega_modes_eV": "eV",
                "fit_gamma_modes_eV": "eV",
            }
        )
    metadata = {
        "spectrum_observable": args.spectral_observable,
        "spectrum_definition": (
            "abs(p_QD/E_inc)^2 from reciprocal linear response"
            if args.spectral_observable == "linear_qd_response"
            else "rho_ee(t_read)/fluence from weak finite Gaussian pulses"
        ),
        "weak_linearity_check": {
            "enabled": args.spectral_observable == "weak_pulse_excitation",
            "check_fluence_factor": float(args.weak_linearity_check_factor),
            "max_normalized_rms_change": float(
                args.max_weak_linearity_relative_error
            ),
            "max_population": float(args.max_weak_population),
        },
        "weak_pulse_read_time": {
            "enabled": args.spectral_observable == "weak_pulse_excitation",
            "time_span_au": weak_time_span_au.tolist(),
            "t_read_fs": float(au_to_fs(weak_time_span_au[1])),
            "common_to_every_model_gap_channel_and_carrier_energy": True,
        },
        "feature_definition": (
            "peak nearest feature_center_eV inside the fixed window; width is "
            "the connected half-prominence interval; competing comparable peaks make width undefined"
        ),
        "gamma0_definition": "isolated-QD FWHM extracted by the identical feature algorithm",
        "discrepancy_definition": (
            "sqrt(integral_W (S_DD-S_FQS)^2 dE / integral_W S_FQS^2 dE) "
            "on the fixed feature window W, without separately normalizing either spectrum"
        ),
        "dd_validity_definition": (
            "smallest sampled gap for which discrepancy stays below tolerance at every larger sampled gap"
        ),
        "array_dimensions": {
            "qd_excitation_spectrum": ["model", "channel", "gap", "energy"],
            "peak_energy_eV": ["model", "channel", "gap"],
            "spectral_l2_relative_dd_vs_fqs": ["channel", "gap"],
        },
        "array_units": units,
        "reference_physical_parameters": params_to_physical_dict(first.params, first.spec.orientation),
        "energy_resolution_gate": {
            "max_step_over_gamma0": float(args.max_energy_step_over_gamma0),
            "accepted": resolution_accepted,
        },
        "dd_tolerance": float(args.dd_tolerance),
        "same_material_A_certificate": {
            "max_relative_disagreement": max_relative_a_disagreement,
            "accepted_tolerance": 1.0e-10,
        },
    }
    return payload, metadata


def _first_crossing_bracket(
    fluence: np.ndarray,
    population: np.ndarray,
    target: float,
) -> tuple[int, int] | None:
    _, status, bracket = threshold_from_curve(fluence, population, target)
    return bracket if status == "resolved" else None


def compute_threshold_payload(
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    _validate_common_inputs(args)
    if not 0.0 < args.target_population < 1.0:
        raise ValueError("--target-population must lie in (0, 1).")
    if not 0.0 < args.fluence_min_j_cm2 < args.fluence_max_j_cm2:
        raise ValueError("Fluence bounds must satisfy 0 < min < max.")
    if args.fluence_points < 3:
        raise ValueError("--fluence-points must be at least three.")
    if not np.isfinite(args.carrier_energy_ev) or args.carrier_energy_ev <= 0.0:
        raise ValueError("--carrier-energy-ev must be finite and positive.")
    if not np.isfinite(args.threshold_root_xtol_sqrt_fluence) or args.threshold_root_xtol_sqrt_fluence <= 0.0:
        raise ValueError("--threshold-root-xtol-sqrt-fluence must be finite and positive.")
    if not np.isfinite(args.threshold_root_rtol) or args.threshold_root_rtol <= 0.0:
        raise ValueError("--threshold-root-rtol must be finite and positive.")
    if not np.isfinite(args.max_isolated_pulse_area_step_rad) or args.max_isolated_pulse_area_step_rad <= 0.0:
        raise ValueError("--max-isolated-pulse-area-step-rad must be finite and positive.")
    if not np.isfinite(args.max_fluence_grid_midpoint_error) or args.max_fluence_grid_midpoint_error <= 0.0:
        raise ValueError("--max-fluence-grid-midpoint-error must be finite and positive.")
    channels = [CHANNEL_BY_ID[name] for name in args.channels]
    gaps = np.asarray(args.gaps_nm, dtype=float)
    fluence = np.linspace(
        np.sqrt(args.fluence_min_j_cm2),
        np.sqrt(args.fluence_max_j_cm2),
        args.fluence_points,
    ) ** 2
    bundles: list[BuiltChannel] = []
    for spec in channels:
        for gap in gaps:
            print(f"Building pulse DD/FQS {spec.channel_id}, gap={gap:g} nm ...", flush=True)
            bundles.append(_build_channel(spec, float(gap), args, require_time_model=True))
    reference_pulse = _pulse_for_fluence(float(fluence[0]), args.carrier_energy_ev, args)
    t_span = _common_time_span(bundles, reference_pulse, args)
    reference_params = bundles[0].params

    isolated = np.empty(fluence.size, dtype=float)
    isolated_tail_converged = np.ones(fluence.size, dtype=bool)
    isolated_tail_ratio = np.zeros(fluence.size, dtype=float)
    isolated_nfev = np.zeros(fluence.size, dtype=np.int64)
    n_model, n_channel, n_gap = 2, len(channels), gaps.size
    population = np.empty((n_model, n_channel, n_gap, fluence.size), dtype=float)
    tail_converged = np.ones_like(population, dtype=bool)
    tail_ratio = np.zeros_like(population)
    nfev = np.zeros_like(population, dtype=np.int64)
    center_distance = np.empty((n_channel, n_gap), dtype=float)

    caches: dict[tuple[str, int, float], float] = {}

    def evaluate(kind: str, flat_index: int, target_fluence: float) -> float:
        key = (kind, flat_index, float(target_fluence))
        if key in caches:
            return caches[key]
        pulse = _pulse_for_fluence(target_fluence, args.carrier_energy_ev, args)
        value, _ = _solve_population(kind, bundles[flat_index], pulse, t_span, args)
        caches[key] = value
        return value

    bare_cache: dict[float, float] = {}

    def evaluate_bare(target_fluence: float) -> float:
        key = float(target_fluence)
        if key not in bare_cache:
            pulse = _pulse_for_fluence(key, args.carrier_energy_ev, args)
            value, _ = _solve_bare_population(
                pulse, reference_params, t_span, args
            )
            bare_cache[key] = value
        return bare_cache[key]

    for fi, target_fluence in enumerate(fluence):
        pulse = _pulse_for_fluence(float(target_fluence), args.carrier_energy_ev, args)
        bare_value, bare_diagnostics = _solve_bare_population(
            pulse, reference_params, t_span, args
        )
        bare_cache[float(target_fluence)] = bare_value
        isolated[fi] = bare_value
        isolated_tail_converged[fi] = bare_diagnostics["tail_converged"]
        isolated_tail_ratio[fi] = bare_diagnostics["tail_ratio"]
        isolated_nfev[fi] = bare_diagnostics["nfev"]
        for flat_index, bundle in enumerate(bundles):
            ci, gi = divmod(flat_index, n_gap)
            center_distance[ci, gi] = _resolved_center_distance_nm(bundle.spec, bundle.gap_nm, args)
            for mi, kind in enumerate(("dd", "fqs")):
                value, diagnostics = _solve_population(kind, bundle, pulse, t_span, args)
                caches[(kind, flat_index, float(target_fluence))] = value
                population[mi, ci, gi, fi] = value
                tail_converged[mi, ci, gi, fi] = diagnostics["tail_converged"]
                tail_ratio[mi, ci, gi, fi] = diagnostics["tail_ratio"]
                nfev[mi, ci, gi, fi] = diagnostics["nfev"]
        print(f"[{fi + 1}/{fluence.size}] fluence={target_fluence:.6e} J/cm^2", flush=True)

    threshold_iso, status_iso, bracket_iso = threshold_from_curve(
        fluence, isolated, args.target_population
    )
    thresholds = np.full((n_model, n_channel, n_gap), np.nan)
    statuses = np.full((n_model, n_channel, n_gap), "not_evaluated", dtype="U32")
    for mi, kind in enumerate(("dd", "fqs")):
        for ci in range(n_channel):
            for gi in range(n_gap):
                flat_index = ci * n_gap + gi
                estimate, status, bracket = threshold_from_curve(
                    fluence, population[mi, ci, gi], args.target_population
                )
                if args.refine_threshold and status == "resolved" and bracket is not None:
                    lo, hi = bracket
                    x_lo, x_hi = np.sqrt(fluence[[lo, hi]])
                    root = brentq(
                        lambda x: evaluate(kind, flat_index, float(x * x))
                        - args.target_population,
                        float(x_lo),
                        float(x_hi),
                        xtol=args.threshold_root_xtol_sqrt_fluence,
                        rtol=args.threshold_root_rtol,
                    )
                    estimate = float(root * root)
                    status = "resolved_refined"
                thresholds[mi, ci, gi] = estimate
                statuses[mi, ci, gi] = status
    if args.refine_threshold and status_iso == "resolved" and bracket_iso is not None:
        lo, hi = bracket_iso
        x_lo, x_hi = np.sqrt(fluence[[lo, hi]])
        root = brentq(
            lambda x: evaluate_bare(float(x * x)) - args.target_population,
            float(x_lo),
            float(x_hi),
            xtol=args.threshold_root_xtol_sqrt_fluence,
            rtol=args.threshold_root_rtol,
        )
        threshold_iso = float(root * root)
        status_iso = "resolved_refined"

    resolved_status = np.char.startswith(statuses, "resolved")
    isolated_resolved = str(status_iso).startswith("resolved")
    threshold_ratio = np.full_like(thresholds, np.nan)
    efficiency = np.full_like(thresholds, np.nan)
    if isolated_resolved and np.isfinite(threshold_iso):
        np.divide(
            thresholds,
            threshold_iso,
            out=threshold_ratio,
            where=resolved_status,
        )
        np.divide(
            threshold_iso,
            thresholds,
            out=efficiency,
            where=resolved_status,
        )
    paired_resolved = resolved_status[0] & resolved_status[1]
    signed_bias = np.divide(
        thresholds[0] - thresholds[1],
        thresholds[1],
        out=np.full_like(thresholds[0], np.nan),
        where=paired_resolved,
    )
    absolute_bias = np.abs(signed_bias)
    validity_gap = np.asarray(
        [dd_validity_distance(gaps, absolute_bias[ci], args.dd_tolerance) for ci in range(n_channel)]
    )

    reference_fluence = threshold_iso if np.isfinite(threshold_iso) else args.reference_fluence_j_cm2
    reference_population = np.empty((n_model, n_channel, n_gap), dtype=float)
    for mi in range(n_model):
        for ci in range(n_channel):
            for gi in range(n_gap):
                reference_population[mi, ci, gi] = np.interp(
                    np.sqrt(reference_fluence),
                    np.sqrt(fluence),
                    population[mi, ci, gi],
                )
    isolated_reference_population = float(
        np.interp(np.sqrt(reference_fluence), np.sqrt(fluence), isolated)
    )
    population_gain = (
        reference_population / isolated_reference_population
        if isolated_reference_population >= args.minimum_population_for_relative_gain
        else np.full_like(reference_population, np.nan)
    )
    population_difference = reference_population - isolated_reference_population

    pulses = [
        _pulse_for_fluence(float(value), args.carrier_energy_ev, args)
        for value in fluence
    ]
    e0_au = np.asarray([pulse.E0_au for pulse in pulses])
    intensity = np.asarray([pulse.peak_intensity_w_cm2(eps_m=args.eps_m) for pulse in pulses])
    pulse_area = np.asarray(
        [
            reference_params.qd_local_field_factor
            * reference_params.d_au
            * pulse.E0_au
            * np.sqrt(2.0 * np.pi)
            * pulse.sigma_t_au
            for pulse in pulses
        ],
        dtype=float,
    )
    all_population_curves = np.concatenate(
        [isolated[None, :], population.reshape(-1, fluence.size)], axis=0
    )
    grid_diagnostics = fluence_grid_resolution_diagnostics(
        fluence,
        all_population_curves,
        pulse_area,
        max_midpoint_error=args.max_fluence_grid_midpoint_error,
        max_pulse_area_step_rad=args.max_isolated_pulse_area_step_rad,
    )
    if not grid_diagnostics["accepted"]:
        _apply_policy(
            args.fluence_grid_convergence_policy,
            "The nonlinear fluence grid is not resolved in sqrt(fluence): "
            f"max midpoint population error="
            f"{float(np.max(grid_diagnostics['midpoint_interpolation_error_by_channel'])):.6g}, "
            f"allowed={args.max_fluence_grid_midpoint_error:.6g}; max isolated "
            f"pulse-area step={grid_diagnostics['maximum_isolated_pulse_area_step_rad']:.6g} rad, "
            f"allowed={args.max_isolated_pulse_area_step_rad:.6g} rad.",
        )
    threshold_intensity = np.full_like(thresholds, np.nan)
    for index in np.ndindex(thresholds.shape):
        if np.isfinite(thresholds[index]):
            threshold_intensity[index] = _pulse_for_fluence(
                float(thresholds[index]), args.carrier_energy_ev, args
            ).peak_intensity_w_cm2(eps_m=args.eps_m)
    isolated_threshold_intensity = (
        _pulse_for_fluence(threshold_iso, args.carrier_energy_ev, args).peak_intensity_w_cm2(
            eps_m=args.eps_m
        )
        if np.isfinite(threshold_iso)
        else np.nan
    )
    first = bundles[0]
    material = first.params.material
    fit_alpha_inf = np.asarray(
        [bundle.dd_model.fit.alpha_inf for bundle in bundles], dtype=float
    ).reshape(n_channel, n_gap)
    fit_strengths = np.stack([bundle.dd_model.fit.strengths_au2 for bundle in bundles]).reshape(
        n_channel, n_gap, args.material_fit_modes
    )
    fit_omega = np.stack([bundle.dd_model.fit.omega_modes_au for bundle in bundles]).reshape(
        n_channel, n_gap, args.material_fit_modes
    )
    fit_gamma = np.stack([bundle.dd_model.fit.gamma_modes_au for bundle in bundles]).reshape(
        n_channel, n_gap, args.material_fit_modes
    )
    fit_nrms_alpha = np.asarray(
        [bundle.dd_model.fit.normalized_rms_alpha for bundle in bundles]
    ).reshape(n_channel, n_gap)
    fit_nrms_inverse = np.asarray(
        [bundle.dd_model.fit.normalized_rms_inv_alpha for bundle in bundles]
    ).reshape(n_channel, n_gap)
    fit_max_error = np.asarray(
        [bundle.dd_model.fit.max_normalized_alpha_error for bundle in bundles]
    ).reshape(n_channel, n_gap)
    spatial_half_error = np.asarray(
        [
            bundle.fqs_model.spatial_convergence_diagnostics.max_half_order_relative_change
            for bundle in bundles
        ]
    ).reshape(n_channel, n_gap)
    spatial_tail_error = np.asarray(
        [
            bundle.fqs_model.spatial_convergence_diagnostics.max_tail_block_relative_mass
            for bundle in bundles
        ]
    ).reshape(n_channel, n_gap)
    spatial_accepted = np.asarray(
        [bundle.fqs_model.spatial_convergence_diagnostics.accepted for bundle in bundles],
        dtype=bool,
    ).reshape(n_channel, n_gap)
    modal_nrms = np.asarray(
        [bundle.fqs_model.modal_fit_diagnostics.max_normalized_rms for bundle in bundles]
    ).reshape(n_channel, n_gap)
    modal_max_error = np.asarray(
        [bundle.fqs_model.modal_fit_diagnostics.max_relative_error for bundle in bundles]
    ).reshape(n_channel, n_gap)
    modal_accepted = np.asarray(
        [bundle.fqs_model.modal_fit_diagnostics.accepted for bundle in bundles],
        dtype=bool,
    ).reshape(n_channel, n_gap)
    stability_accepted = np.asarray(
        [bundle.fqs_model.coupled_stability.stable for bundle in bundles],
        dtype=bool,
    ).reshape(n_channel, n_gap)
    payload = {
        "channel_id": np.asarray(args.channels, dtype="U32"),
        "model_id": MODEL_IDS,
        "gap_nm": gaps,
        "center_distance_nm": center_distance,
        "fluence_j_cm2": fluence,
        "pulse_e0_au": e0_au,
        "pulse_e0_v_m": np.asarray(field_au_to_si(e0_au)),
        "peak_intensity_w_cm2": intensity,
        "isolated_qd_pulse_area_rad": pulse_area,
        "isolated_qd_population": isolated,
        "isolated_pulse_tail_converged": isolated_tail_converged,
        "isolated_pulse_tail_ratio": isolated_tail_ratio,
        "isolated_solver_nfev": isolated_nfev,
        "hybrid_population": population,
        "pulse_tail_converged": tail_converged,
        "pulse_tail_ratio": tail_ratio,
        "solver_nfev": nfev,
        "fluence_grid_midpoint_error_isolated": np.asarray(
            grid_diagnostics["midpoint_interpolation_error_by_channel"][0]
        ),
        "fluence_grid_midpoint_error_hybrid": np.asarray(
            grid_diagnostics["midpoint_interpolation_error_by_channel"][1:]
        ).reshape(n_model, n_channel, n_gap),
        "fluence_grid_converged": np.asarray(grid_diagnostics["accepted"]),
        "maximum_isolated_pulse_area_step_rad": np.asarray(
            grid_diagnostics["maximum_isolated_pulse_area_step_rad"]
        ),
        "threshold_population": np.asarray(args.target_population),
        "isolated_threshold_fluence_j_cm2": np.asarray(threshold_iso),
        "isolated_threshold_status": np.asarray(status_iso),
        "threshold_fluence_j_cm2": thresholds,
        "threshold_status": statuses,
        "threshold_fluence_ratio_to_isolated": threshold_ratio,
        "fluence_efficiency_isolated_over_hybrid": efficiency,
        "threshold_peak_intensity_w_cm2": threshold_intensity,
        "isolated_threshold_peak_intensity_w_cm2": np.asarray(isolated_threshold_intensity),
        "signed_threshold_bias_dd_vs_fqs": signed_bias,
        "absolute_threshold_discrepancy_dd_vs_fqs": absolute_bias,
        "dd_validity_gap_nm": validity_gap,
        "reference_fluence_j_cm2": np.asarray(reference_fluence),
        "isolated_population_at_reference": np.asarray(isolated_reference_population),
        "hybrid_population_at_reference": reference_population,
        "population_gain_at_reference": population_gain,
        "population_difference_at_reference": population_difference,
        "read_time_fs": np.asarray(au_to_fs(t_span[1])),
        "material_energy_eV": np.asarray(material.energy_eV),
        "material_n": np.asarray(material.n),
        "material_k": np.asarray(material.k),
        "fit_alpha_inf": fit_alpha_inf,
        "fit_strengths_au2": fit_strengths,
        "fit_omega_modes_au": fit_omega,
        "fit_gamma_modes_au": fit_gamma,
        "fit_omega_modes_eV": np.asarray(au_to_eV(fit_omega)),
        "fit_gamma_modes_eV": np.asarray(au_to_eV(fit_gamma)),
        "fit_normalized_rms_alpha": fit_nrms_alpha,
        "fit_normalized_rms_inverse_alpha": fit_nrms_inverse,
        "fit_max_normalized_alpha_error": fit_max_error,
        "fqs_spatial_half_order_relative_error": spatial_half_error,
        "fqs_spatial_tail_block_relative_error": spatial_tail_error,
        "fqs_spatial_convergence_accepted": spatial_accepted,
        "fqs_modal_fit_normalized_rms": modal_nrms,
        "fqs_modal_fit_max_relative_error": modal_max_error,
        "fqs_modal_fit_accepted": modal_accepted,
        "fqs_coupled_stability_accepted": stability_accepted,
        **_flatten_spatial_data(bundles),
    }
    metadata = {
        "threshold_definition": (
            "first upward crossing of target population before the first resolved Rabi maximum; "
            "interpolation and optional Brent refinement are performed in sqrt(fluence)"
        ),
        "primary_ratio_definition": "F_eta(hybrid) / F_eta(isolated QD); values below one are beneficial",
        "efficiency_definition": "F_eta(isolated QD) / F_eta(hybrid); values above one are beneficial",
        "signed_model_bias_definition": "(F_eta_DD-F_eta_FQS)/F_eta_FQS",
        "intensity_interpretation": (
            "fluence and peak-intensity ratios are identical only because pulse duration, shape and host are fixed"
        ),
        "array_dimensions": {
            "hybrid_population": ["model", "channel", "gap", "fluence"],
            "threshold_fluence_j_cm2": ["model", "channel", "gap"],
            "absolute_threshold_discrepancy_dd_vs_fqs": ["channel", "gap"],
        },
        "array_units": {
            "gap_nm": "nm surface-to-surface",
            "fluence_j_cm2": "J cm^-2",
            "peak_intensity_w_cm2": "W cm^-2",
            "hybrid_population": "1",
            "threshold_fluence_j_cm2": "J cm^-2",
            "threshold_peak_intensity_w_cm2": "W cm^-2",
            "threshold_fluence_ratio_to_isolated": "1",
            "fit_alpha_inf": "1",
            "fit_strengths_au2": "atomic frequency squared",
            "fit_omega_modes_au": "atomic frequency",
            "fit_gamma_modes_au": "atomic frequency",
            "fit_omega_modes_eV": "eV",
            "fit_gamma_modes_eV": "eV",
        },
        "reference_physical_parameters": params_to_physical_dict(first.params, first.spec.orientation),
        "read_time": {
            "t_read_fs": float(au_to_fs(t_span[1])),
            "common_to_every_model_gap_channel_and_fluence": True,
        },
        "fluence_grid_gate": {
            "accepted": bool(grid_diagnostics["accepted"]),
            "max_midpoint_population_error": float(
                args.max_fluence_grid_midpoint_error
            ),
            "max_isolated_pulse_area_step_rad": float(
                args.max_isolated_pulse_area_step_rad
            ),
        },
        "dd_tolerance": float(args.dd_tolerance),
    }
    return payload, metadata


def _apply_preset(args: argparse.Namespace, *, threshold: bool) -> argparse.Namespace:
    publication = args.preset == "publication"
    if args.gaps_nm is None:
        args.gaps_nm = [1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0] if publication else [1.0, 10.0]
    if args.spatial_order_max is None:
        args.spatial_order_max = 80 if publication else 4
    if args.material_fit_modes is None:
        args.material_fit_modes = 9 if publication else 3
    if args.modal_audit_points is None:
        args.modal_audit_points = 2001 if publication else 301
    for policy_name in (
        "fit_quality_policy",
        "bright_fit_quality_policy",
        "spatial_convergence_policy",
        "tail_policy",
        "population_decay_policy",
    ):
        if hasattr(args, policy_name) and getattr(args, policy_name) is None:
            setattr(args, policy_name, "raise" if publication else "warn")
    if threshold:
        if args.carrier_energy_ev is None:
            args.carrier_energy_ev = args.omega0_ev
        if args.fluence_points is None:
            args.fluence_points = 65 if publication else 7
        if args.fluence_grid_convergence_policy is None:
            args.fluence_grid_convergence_policy = (
                "raise" if publication else "warn"
            )
    else:
        if args.feature_center_ev is None:
            args.feature_center_ev = args.omega0_ev
        if args.energy_points is None:
            args.energy_points = 2001 if publication else 121
        if args.energy_resolution_policy is None:
            args.energy_resolution_policy = "raise" if publication else "warn"
        if args.weak_linearity_policy is None:
            args.weak_linearity_policy = "raise" if publication else "warn"
    return args


def _add_common_calculation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preset", choices=("publication", "quick"), default="publication")
    parser.add_argument("--gaps-nm", nargs="+", type=float)
    parser.add_argument("--channels", nargs="+", choices=tuple(CHANNEL_BY_ID), default=list(CHANNEL_BY_ID))
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
    parser.add_argument("--spatial-order-max", type=int)
    parser.add_argument("--material-fit-modes", type=int)
    parser.add_argument("--fit-min-ev", type=float, default=0.8)
    parser.add_argument("--fit-max-ev", type=float, default=3.0)
    parser.add_argument("--weight-center-ev", type=float)
    parser.add_argument("--weight-sigma-ev", type=float)
    parser.add_argument("--modal-audit-points", type=int)
    parser.add_argument("--max-modal-normalized-rms", type=float, default=0.03)
    parser.add_argument("--max-modal-relative-error", type=float, default=0.08)
    parser.add_argument("--max-bright-fit-normalized-rms", type=float, default=0.025)
    parser.add_argument(
        "--max-bright-fit-pointwise-relative-error", type=float, default=0.05
    )
    parser.add_argument("--spatial-convergence-rtol", type=float, default=2.0e-5)
    parser.add_argument("--reduction-fit-grid-points", type=int, default=1001)
    parser.add_argument("--reduction-audit-grid-points", type=int, default=1601)
    parser.add_argument("--reduction-reaudit-points", type=int, default=1709)
    parser.add_argument("--reduction-rms-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--reduction-max-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--reduction-max-nodes", type=int)
    parser.add_argument("--dd-tolerance", type=float, default=0.1)
    parser.add_argument("--verbose-fit", action="store_true")
    parser.add_argument("--radiative-consistency-policy", choices=POLICIES, default="warn")
    parser.add_argument("--fit-quality-policy", choices=POLICIES)
    parser.add_argument("--bright-fit-quality-policy", choices=POLICIES)
    parser.add_argument("--spatial-convergence-policy", choices=POLICIES)
    parser.add_argument("--reduction-policy", choices=POLICIES, default="raise")


def _add_pulse_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pulse-tau-fs", type=float, default=20.0)
    parser.add_argument("--pulse-tau-kind", choices=("sigma", "fwhm_intensity"), default="fwhm_intensity")
    parser.add_argument("--post-fs", type=float)
    parser.add_argument("--start-sigma", type=float, default=10.0)
    parser.add_argument("--method", choices=("DOP853", "RK45", "Radau", "BDF", "LSODA"), default="DOP853")
    parser.add_argument("--rtol", type=float, default=1.0e-8)
    parser.add_argument("--atol", type=float, default=1.0e-10)
    parser.add_argument("--points-per-fastest-cycle", type=int, default=20)
    parser.add_argument("--max-spectral-leakage", type=float, default=1.0e-3)
    parser.add_argument("--positivity-tolerance", type=float, default=1.0e-7)
    parser.add_argument("--tail-ratio-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--tail-window-fraction", type=float, default=0.05)
    parser.add_argument("--max-population-decay-fraction-at-read", type=float, default=1.0e-3)
    parser.add_argument("--spectral-window-policy", choices=POLICIES, default="warn")
    parser.add_argument("--positivity-policy", choices=POLICIES, default="raise")
    parser.add_argument("--work-passivity-policy", choices=POLICIES, default="raise")
    parser.add_argument("--tail-policy", choices=POLICIES)
    parser.add_argument("--population-decay-policy", choices=POLICIES)


def parse_spectral_calculation_args(metric: str, argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Calculate DD/FQS {metric.replace('_', ' ')} versus QD-MNP gap."
    )
    _add_common_calculation_arguments(parser)
    parser.add_argument("--source-artifact", type=Path)
    parser.add_argument("--energy-min-ev", type=float, default=1.85)
    parser.add_argument("--energy-max-ev", type=float, default=2.25)
    parser.add_argument("--energy-points", type=int)
    parser.add_argument("--feature-center-ev", type=float)
    parser.add_argument("--feature-half-window-ev", type=float, default=0.12)
    parser.add_argument(
        "--spectral-observable",
        choices=("linear_qd_response", "weak_pulse_excitation"),
        default="linear_qd_response",
    )
    parser.add_argument("--weak-fluence-j-cm2", type=float, default=1.0e-10)
    parser.add_argument("--weak-linearity-check-factor", type=float, default=0.25)
    parser.add_argument(
        "--max-weak-linearity-relative-error", type=float, default=0.01
    )
    parser.add_argument("--max-weak-population", type=float, default=0.02)
    parser.add_argument("--weak-linearity-policy", choices=POLICIES)
    parser.add_argument("--max-energy-step-over-gamma0", type=float, default=0.05)
    parser.add_argument("--energy-resolution-policy", choices=POLICIES)
    _add_pulse_arguments(parser)
    return _apply_preset(parser.parse_args(argv), threshold=False)


def parse_threshold_calculation_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate DD/FQS threshold fluence versus QD-MNP gap.")
    _add_common_calculation_arguments(parser)
    _add_pulse_arguments(parser)
    parser.add_argument("--carrier-energy-ev", type=float)
    parser.add_argument("--fluence-min-j-cm2", type=float, default=1.0e-10)
    parser.add_argument("--fluence-max-j-cm2", type=float, default=1.0e-3)
    parser.add_argument("--fluence-points", type=int)
    parser.add_argument("--target-population", type=float, default=0.5)
    parser.add_argument("--reference-fluence-j-cm2", type=float, default=1.0e-7)
    parser.add_argument("--minimum-population-for-relative-gain", type=float, default=0.05)
    parser.add_argument("--max-isolated-pulse-area-step-rad", type=float, default=0.25)
    parser.add_argument("--max-fluence-grid-midpoint-error", type=float, default=0.01)
    parser.add_argument(
        "--fluence-grid-convergence-policy", choices=POLICIES
    )
    parser.add_argument(
        "--refine-threshold",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--threshold-root-xtol-sqrt-fluence", type=float, default=1.0e-14)
    parser.add_argument("--threshold-root-rtol", type=float, default=1.0e-8)
    return _apply_preset(parser.parse_args(argv), threshold=True)


def run_spectral_calculation(
    metric: str,
    generator_path: Path,
    argv: list[str] | None = None,
) -> Path:
    if metric not in SPECTRAL_METRICS:
        raise ValueError(f"Unknown spectral metric {metric!r}.")
    args = parse_spectral_calculation_args(metric, argv)
    if args.source_artifact is not None:
        payload, inherited = load_gap_metric_artifact(args.source_artifact)
        allowed_families = {"spectral_gap_scan"}
        if metric == "model_discrepancy":
            allowed_families.add("threshold_gap_scan")
        if inherited.get("computation_family") not in allowed_families:
            raise ValueError(
                "--source-artifact has an incompatible computation family."
            )
        metadata = dict(inherited)
        metadata.update(
            {
                "created_at": datetime.now().astimezone().isoformat(),
                "primary_metric": metric,
                "derived_from_artifact": str(args.source_artifact.resolve()),
                "derivation_arguments": _json_ready(vars(args)),
                "provenance": {
                    **inherited.get("provenance", {}),
                    "derivation_generator": str(generator_path.resolve()),
                    "derivation_source_sha256": _source_hashes((generator_path,)),
                    "derivation_git": _git_provenance(),
                },
            }
        )
    else:
        payload, details = compute_spectral_payload(args)
        metadata = _base_metadata(
            args,
            metric=metric,
            computation_family="spectral_gap_scan",
            generator_path=generator_path,
        )
        metadata.update(details)
    output = _atomic_write_npz(args.output, payload, metadata, overwrite=args.overwrite)
    print(f"Saved DD/FQS {metric} artifact: {output}")
    return output


def run_threshold_calculation(generator_path: Path, argv: list[str] | None = None) -> Path:
    args = parse_threshold_calculation_args(argv)
    payload, details = compute_threshold_payload(args)
    metadata = _base_metadata(
        args,
        metric="threshold_fluence",
        computation_family="threshold_gap_scan",
        generator_path=generator_path,
    )
    metadata.update(details)
    output = _atomic_write_npz(args.output, payload, metadata, overwrite=args.overwrite)
    print(f"Saved DD/FQS threshold-fluence artifact: {output}")
    return output


def _channel_labels(metadata: dict[str, Any], channel_ids: np.ndarray) -> list[str]:
    mapping = {
        item["channel_id"]: item.get("label", item["channel_id"])
        for item in metadata.get("channels", ())
    }
    return [mapping.get(str(item), str(item)) for item in channel_ids]


def _require_spectral(payload: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
    if metadata.get("computation_family") != "spectral_gap_scan":
        raise ValueError("This plotter requires a spectral_gap_scan artifact.")
    required = {
        "energy_eV",
        "isolated_qd_spectrum",
        "qd_excitation_spectrum",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"Spectral artifact misses arrays: {sorted(missing)}")


def _recompute_spectral_derived(
    payload: dict[str, np.ndarray],
    *,
    center_eV: float,
    half_window_eV: float,
) -> dict[str, np.ndarray]:
    return _spectral_metrics(
        payload["energy_eV"],
        payload["isolated_qd_spectrum"],
        payload["qd_excitation_spectrum"],
        center_eV=center_eV,
        half_window_eV=half_window_eV,
    )


def _save_figure(fig: Any, output: Path, *, dpi: int, show: bool) -> Path:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return output


def plot_gap_metric(
    metric: str,
    artifact: Path,
    output: Path,
    *,
    feature_center_ev: float | None,
    feature_half_window_ev: float | None,
    target_population: float | None,
    dd_tolerance: float,
    gain_kind: str,
    x_scale: str,
    dpi: int,
    show: bool,
) -> Path:
    if metric not in ALL_METRICS:
        raise ValueError(f"Unknown metric {metric!r}.")
    if not np.isfinite(dd_tolerance) or dd_tolerance <= 0.0:
        raise ValueError("dd_tolerance must be finite and positive.")
    if gain_kind not in {"optimized", "fixed"}:
        raise ValueError("gain_kind must be 'optimized' or 'fixed'.")
    if x_scale not in {"linear", "log"}:
        raise ValueError("x_scale must be 'linear' or 'log'.")
    if isinstance(dpi, bool) or int(dpi) != dpi or dpi < 1:
        raise ValueError("dpi must be a positive integer.")
    payload, metadata = load_gap_metric_artifact(artifact)
    gaps = np.asarray(payload["gap_nm"], dtype=float)
    channels = np.asarray(payload["channel_id"]).astype(str)
    labels = _channel_labels(metadata, channels)
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.8, len(channels)))
    fig, ax = plt.subplots(figsize=(8.2, 5.2))

    if (
        metric == "model_discrepancy"
        and metadata.get("computation_family") == "threshold_gap_scan"
    ):
        if "absolute_threshold_discrepancy_dd_vs_fqs" not in payload:
            raise ValueError(
                "Threshold artifact has no DD/FQS threshold discrepancy array."
            )
        values = np.asarray(
            payload["absolute_threshold_discrepancy_dd_vs_fqs"], dtype=float
        )
        ylabel = r"$\delta_{\mathcal{F}}$"
        for ci, (label, color) in enumerate(zip(labels, colors)):
            ax.plot(gaps, values[ci], marker="o", color=color, label=label)
            validity = dd_validity_distance(gaps, values[ci], dd_tolerance)
            if np.isfinite(validity):
                ax.scatter(
                    [validity],
                    [dd_tolerance],
                    marker="v",
                    s=45,
                    color=color,
                    zorder=5,
                )
        ax.axhline(
            dd_tolerance,
            color="black",
            ls="--",
            lw=1.0,
            label=f"tolerance {dd_tolerance:g}",
        )
        ax.set_yscale("log")
        ax.set_xlabel(r"surface gap $g$ (nm)")
        ax.set_ylabel(ylabel)
        ax.set_xscale(x_scale)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
        fig.suptitle("DD–FQS threshold-fluence discrepancy")
        return _save_figure(fig, output, dpi=dpi, show=show)
    if metric in SPECTRAL_METRICS:
        _require_spectral(payload, metadata)
        center = (
            float(feature_center_ev)
            if feature_center_ev is not None
            else float(metadata["requested_arguments"]["feature_center_ev"])
        )
        half_window = (
            float(feature_half_window_ev)
            if feature_half_window_ev is not None
            else float(metadata["requested_arguments"]["feature_half_window_ev"])
        )
        derived = _recompute_spectral_derived(
            payload,
            center_eV=center,
            half_window_eV=half_window,
        )
        if metric in {"resonance_shift", "spectral_width"}:
            statuses = np.asarray(derived["feature_status"]).astype(str)
            unresolved = int(np.count_nonzero(statuses != "ok"))
            if unresolved:
                warnings.warn(
                    f"{unresolved} DD/FQS feature points are unresolved or split; "
                    "their scalar shift/width is plotted as NaN. Inspect "
                    "feature_status and the saved raw spectra.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        if metric == "excitation_gain":
            key = (
                "excitation_gain_optimized"
                if gain_kind == "optimized"
                else "excitation_gain_at_reference_energy"
            )
            values = derived[key]
            ylabel = (
                r"$G_{\rm exc}=\max_E S/\max_E S_0$"
                if gain_kind == "optimized"
                else r"$G_{\rm exc}(E_*)=S(E_*)/S_0(E_*)$"
            )
            ax.axhline(1.0, color="black", lw=0.9, alpha=0.5)
        elif metric == "resonance_shift":
            values = derived["resonance_shift_over_gamma0"]
            ylabel = r"$(E_{\rm res}-E_{\rm res,0})/\Gamma_0$"
            ax.axhline(0.0, color="black", lw=0.9, alpha=0.5)
        elif metric == "spectral_width":
            values = derived["spectral_width_over_gamma0"]
            ylabel = r"$\Gamma_{\rm eff}/\Gamma_0$"
            ax.axhline(1.0, color="black", lw=0.9, alpha=0.5)
        else:
            values = derived["spectral_l2_relative_dd_vs_fqs"]
            ylabel = r"$\delta_{\rm spec}$"
            for ci, (label, color) in enumerate(zip(labels, colors)):
                ax.plot(gaps, values[ci], marker="o", color=color, label=label)
                validity = dd_validity_distance(gaps, values[ci], dd_tolerance)
                if np.isfinite(validity):
                    ax.scatter(
                        [validity],
                        [dd_tolerance],
                        marker="v",
                        s=45,
                        color=color,
                        zorder=5,
                    )
            ax.axhline(dd_tolerance, color="black", ls="--", lw=1.0, label=f"tolerance {dd_tolerance:g}")
            ax.set_yscale("log")
            ax.set_xlabel(r"surface gap $g$ (nm)")
            ax.set_ylabel(ylabel)
            ax.set_xscale(x_scale)
            ax.grid(True, which="both", alpha=0.25)
            ax.legend(fontsize=8, ncol=2)
            fig.suptitle("DD–FQS spectral discrepancy")
            return _save_figure(fig, output, dpi=dpi, show=show)

        for ci, (label, color) in enumerate(zip(labels, colors)):
            ax.plot(gaps, values[1, ci], marker="o", color=color, label=f"{label}, FQS")
            ax.plot(gaps, values[0, ci], marker="s", ls="--", color=color, alpha=0.72, label=f"{label}, DD")
        ax.set_xlabel(r"surface gap $g$ (nm)")
        ax.set_ylabel(ylabel)
        ax.set_xscale(x_scale)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7.5, ncol=2)
        fig.suptitle(metric.replace("_", " ").title())
    else:
        if metadata.get("computation_family") != "threshold_gap_scan":
            raise ValueError("Threshold plotter requires a threshold_gap_scan artifact.")
        target = (
            float(target_population)
            if target_population is not None
            else float(np.asarray(payload["threshold_population"]))
        )
        fluence = payload["fluence_j_cm2"]
        isolated = payload["isolated_qd_population"]
        hybrid = payload["hybrid_population"]
        stored_target = float(np.asarray(payload["threshold_population"]))
        if np.isclose(target, stored_target, rtol=0.0, atol=1.0e-15):
            values = np.asarray(
                payload["threshold_fluence_ratio_to_isolated"], dtype=float
            )
        else:
            warnings.warn(
                "The requested target population differs from the calculation "
                "target; thresholds are recomputed by sqrt(fluence) interpolation "
                "of the stored grid and are not Brent-refined.",
                RuntimeWarning,
                stacklevel=2,
            )
            iso_threshold, iso_status, _ = threshold_from_curve(
                fluence, isolated, target
            )
            values = np.full(hybrid.shape[:-1], np.nan)
            if str(iso_status).startswith("resolved") and np.isfinite(iso_threshold):
                for index in np.ndindex(hybrid.shape[:-1]):
                    threshold, status, _ = threshold_from_curve(
                        fluence, hybrid[index], target
                    )
                    if str(status).startswith("resolved"):
                        values[index] = threshold / iso_threshold
        for ci, (label, color) in enumerate(zip(labels, colors)):
            ax.plot(gaps, values[1, ci], marker="o", color=color, label=f"{label}, FQS")
            ax.plot(gaps, values[0, ci], marker="s", ls="--", color=color, alpha=0.72, label=f"{label}, DD")
        ax.axhline(1.0, color="black", lw=0.9, alpha=0.5)
        ax.set_xlabel(r"surface gap $g$ (nm)")
        ax.set_ylabel(r"$\mathcal{F}_\eta/\mathcal{F}_{\eta,0}$")
        ax.set_xscale(x_scale)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7.5, ncol=2)
        fig.suptitle(f"Threshold fluence ratio, target population={target:g}")
    return _save_figure(fig, output, dpi=dpi, show=show)


def parse_plot_args(metric: str, argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Plot {metric.replace('_', ' ')} from a saved NPZ artifact.")
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-center-ev", type=float)
    parser.add_argument("--feature-half-window-ev", type=float)
    parser.add_argument("--target-population", type=float)
    parser.add_argument("--dd-tolerance", type=float, default=0.1)
    parser.add_argument("--gain-kind", choices=("optimized", "fixed"), default="optimized")
    parser.add_argument("--x-scale", choices=("linear", "log"), default="log")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args(argv)


def run_gap_plot(metric: str, argv: list[str] | None = None) -> Path:
    if metric not in ALL_METRICS:
        raise ValueError(f"Unknown metric {metric!r}.")
    args = parse_plot_args(metric, argv)
    output = plot_gap_metric(
        metric,
        args.artifact,
        args.output,
        feature_center_ev=args.feature_center_ev,
        feature_half_window_ev=args.feature_half_window_ev,
        target_population=args.target_population,
        dd_tolerance=args.dd_tolerance,
        gain_kind=args.gain_kind,
        x_scale=args.x_scale,
        dpi=args.dpi,
        show=args.show,
    )
    print(f"Saved {metric} figure: {output}")
    return output
