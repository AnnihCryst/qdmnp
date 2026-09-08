"""Calculate one-vs-multi Lorentz FQS excitation curves in one NPZ artifact.

The expensive propagation is delegated to the existing, validated
``qd_mnp_calculate_excitation_fluence`` article API.  This orchestrator changes
only the number of material Lorentz poles (and the explicitly diagnostic fit
accuracy policy for the one-pole branch), verifies scenario identity, and
stores both branches together for plot-only post-processing.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import platform
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import scipy

from article_observables.qd_mnp_calculate_excitation_fluence import (
    HYBRID_CHANNELS,
    _build_channel_model,
    _pulse_for_fluence,
    _resolved_settings,
    build_argument_parser,
    calculate_excitation_fluence,
)
from article_observables.qd_mnp_material_modes_artifact import (
    atomic_write_npz,
    canonical_sha256,
    git_provenance,
    source_hashes,
)
from article_observables.qd_mnp_plot_excitation_fluence import (
    validate_excitation_comparison,
)
from qd_mnp_rational_fit import (
    AU_DIPOLE_C_M,
    AU_ENERGY_EV,
    AU_ENERGY_J,
    AU_FIELD_V_M,
    AU_LENGTH_M,
    AU_TIME_S,
    au_to_fs,
)


SCHEMA_NAME = "qd_mnp.material_excitation_fluence_comparison"
BRANCH_IDS = np.asarray(["one", "multi"], dtype="U16")


def _uniform_read_time(payload: dict[str, np.ndarray]) -> float:
    read_time = np.asarray(payload["read_time_fs"], dtype=float)
    if read_time.ndim != 1 or read_time.size == 0 or np.any(~np.isfinite(read_time)):
        raise ValueError("Underlying fluence calculation returned invalid read times.")
    tolerance = 1.0e-10 * max(abs(float(read_time[0])), 1.0)
    if not np.allclose(read_time, read_time[0], rtol=0.0, atol=tolerance):
        raise ValueError("Underlying fluence calculation did not use one read time.")
    return float(read_time[0])


def _first_threshold(
    fluence: np.ndarray,
    population: np.ndarray,
    target: float,
) -> tuple[float, str, int]:
    """First rising-branch threshold, interpolated in sqrt(fluence)."""

    x = np.asarray(fluence, dtype=float)
    y = np.asarray(population, dtype=float)
    if y[0] >= target:
        return float(x[0]), "left_censored", 0
    for index in range(x.size - 1):
        y0 = float(y[index])
        y1 = float(y[index + 1])
        if y0 < target <= y1 and y1 > y0:
            root_x = np.sqrt(x[index]) + (
                (target - y0)
                / (y1 - y0)
                * (np.sqrt(x[index + 1]) - np.sqrt(x[index]))
            )
            return float(root_x**2), "interpolated", index
    return float("nan"), "not_reached", -1


def _threshold_arrays(
    fluence: np.ndarray,
    population: np.ndarray,
    target: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    thresholds = np.full(population.shape[:-1], np.nan, dtype=float)
    status = np.empty(population.shape[:-1], dtype="U24")
    bracket_index = np.full(population.shape[:-1], -1, dtype=np.int64)
    for index in np.ndindex(population.shape[:-1]):
        value, label, bracket = _first_threshold(fluence, population[index], target)
        thresholds[index] = value
        status[index] = label
        bracket_index[index] = bracket
    return thresholds, status, bracket_index


def _scenario_document(settings: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "material_fit_modes",
        "bright_fit_quality_policy",
        "fit_quality_policy",
        "output",
        "overwrite",
        "one_material_fit_modes",
        "target_population",
        "_requested_cli_arguments",
    }
    return {
        key: value
        for key, value in settings.items()
        if key not in excluded
    }


def _auto_common_post_fs(
    one_settings: dict[str, Any],
    multi_settings: dict[str, Any],
) -> float:
    """Resolve a common model-derived read time without propagating the sweep."""

    end_times_au: list[float] = []
    for label, settings in (("one", one_settings), ("multi", multi_settings)):
        print(f"Prebuilding {label} material branch for common read time ...", flush=True)
        for channel in HYBRID_CHANNELS:
            *_, model = _build_channel_model(channel, settings)
            end_times_au.append(float(model.recommended_post_pulse_time_au()))
    reference_pulse = _pulse_for_fluence(
        float(np.asarray(multi_settings["fluence_grid_j_cm2"])[0]),
        multi_settings,
    )
    end_times_au.append(
        float(multi_settings["start_sigma"] * reference_pulse.sigma_t_au)
    )
    return float(au_to_fs(max(end_times_au)))


def calculate_material_fluence_comparison(
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Run and combine the one-pole and production material branches."""

    if not 0.0 < args.target_population < 1.0:
        raise ValueError("--target-population must lie strictly between 0 and 1.")

    resolved = _resolved_settings(args)
    multi_settings = deepcopy(resolved)
    one_settings = deepcopy(resolved)
    one_settings["material_fit_modes"] = int(args.one_material_fit_modes)
    one_settings["bright_fit_quality_policy"] = args.one_fit_accuracy_policy
    one_settings["fit_quality_policy"] = args.one_fit_accuracy_policy

    if multi_settings["material_fit_modes"] < 2:
        raise ValueError(
            "--material-fit-modes must be at least 2 for the multi-oscillator branch."
        )

    if one_settings["material_fit_modes"] == multi_settings["material_fit_modes"]:
        raise ValueError("The one and multi branches must use different mode counts.")
    if one_settings["material_fit_modes"] != 1:
        raise ValueError(
            "--one-material-fit-modes must equal 1 for the one-oscillator branch."
        )
    if one_settings["material_fit_modes"] > multi_settings["material_fit_modes"]:
        raise ValueError("The diagnostic one branch must use fewer modes than multi.")

    auto_read_time = resolved["post_fs"] is None
    if auto_read_time:
        common_post_fs = _auto_common_post_fs(one_settings, multi_settings)
        one_settings["post_fs"] = common_post_fs
        multi_settings["post_fs"] = common_post_fs
        print(f"Using common model-derived t_read={common_post_fs:.6g} fs.", flush=True)

    branch_results: list[tuple[dict[str, np.ndarray], dict[str, Any]]] = []
    for branch_id, settings in (("one", one_settings), ("multi", multi_settings)):
        print(
            f"Calculating {branch_id} branch with N={settings['material_fit_modes']} ...",
            flush=True,
        )
        branch_results.append(calculate_excitation_fluence(settings))

    validate_excitation_comparison(branch_results)
    one_payload, one_metadata = branch_results[0]
    multi_payload, multi_metadata = branch_results[1]
    one_read = _uniform_read_time(one_payload)
    multi_read = _uniform_read_time(multi_payload)
    if not np.isclose(
        one_read,
        multi_read,
        rtol=0.0,
        atol=1.0e-10 * max(abs(one_read), abs(multi_read), 1.0),
    ):
        raise RuntimeError("Material branches do not share one physical read time.")

    channel_id = np.asarray(one_payload["channel_id"]).astype("U32")
    hybrid_channel_id = np.asarray(one_payload["hybrid_channel_id"]).astype("U32")
    fluence = np.asarray(one_payload["fluence_j_cm2"], dtype=float)
    p_read = np.stack(
        [
            np.asarray(one_payload["p_exc_read"], dtype=float),
            np.asarray(multi_payload["p_exc_read"], dtype=float),
        ]
    )
    p_max = np.stack(
        [
            np.asarray(one_payload["p_exc_max"], dtype=float),
            np.asarray(multi_payload["p_exc_max"], dtype=float),
        ]
    )
    t_max = np.stack(
        [
            np.asarray(one_payload["time_of_p_exc_max_fs"], dtype=float),
            np.asarray(multi_payload["time_of_p_exc_max_fs"], dtype=float),
        ]
    )
    thresholds, threshold_status, threshold_bracket = _threshold_arrays(
        fluence,
        p_read,
        float(args.target_population),
    )
    peak_intensity = np.asarray(one_payload["peak_intensity_w_cm2"], dtype=float)
    threshold_intensity = np.full_like(thresholds, np.nan)
    for index in np.ndindex(thresholds.shape):
        if np.isfinite(thresholds[index]):
            threshold_intensity[index] = float(
                np.interp(thresholds[index], fluence, peak_intensity)
            )

    hybrid_slice = slice(1, None)
    threshold_ratio = np.divide(
        thresholds[0, hybrid_slice],
        thresholds[1, hybrid_slice],
        out=np.full(thresholds.shape[1] - 1, np.nan),
        where=(
            np.isfinite(thresholds[0, hybrid_slice])
            & np.isfinite(thresholds[1, hybrid_slice])
            & (thresholds[1, hybrid_slice] > 0.0)
        ),
    )

    payload: dict[str, np.ndarray] = {
        "branch_id": BRANCH_IDS,
        "branch_material_mode_count": np.asarray(
            [one_settings["material_fit_modes"], multi_settings["material_fit_modes"]],
            dtype=np.int64,
        ),
        "channel_id": channel_id,
        "hybrid_channel_id": hybrid_channel_id,
        "fluence_j_cm2": fluence,
        "pulse_e0_au": np.asarray(one_payload["pulse_e0_au"]),
        "pulse_e0_v_m": np.asarray(one_payload["pulse_e0_v_m"]),
        "peak_intensity_w_cm2": peak_intensity,
        "isolated_qd_pulse_area_rad": np.asarray(
            one_payload["isolated_qd_pulse_area_rad"]
        ),
        "read_time_fs": np.asarray(one_payload["read_time_fs"]),
        "p_exc_read": p_read,
        "p_exc_max": p_max,
        "time_of_p_exc_max_fs": t_max,
        "population_one_minus_multi": p_read[0] - p_read[1],
        "target_population": np.asarray(float(args.target_population)),
        "threshold_fluence_j_cm2": thresholds,
        "threshold_peak_intensity_w_cm2": threshold_intensity,
        "threshold_status": threshold_status,
        "threshold_bracket_index": threshold_bracket,
        "hybrid_threshold_ratio_one_to_multi": threshold_ratio,
    }
    # Preserve every branch-specific raw result and diagnostic.  Prefixing
    # handles the intentionally different Lorentz-array lengths losslessly.
    for branch_id, branch_payload in zip(BRANCH_IDS.astype(str), (one_payload, multi_payload)):
        for key, value in branch_payload.items():
            payload[f"{branch_id}__{key}"] = np.asarray(value)

    scenario = _scenario_document(multi_settings)
    source_paths = (
        Path(__file__),
        PROJECT_ROOT / "article_observables" / "qd_mnp_calculate_excitation_fluence.py",
        PROJECT_ROOT / "article_observables" / "qd_mnp_material_modes_artifact.py",
        PROJECT_ROOT / "qd_mnp_rational_fit.py",
        PROJECT_ROOT / "qd_mnp_full_qs_model.py",
        PROJECT_ROOT / "qd_mnp_spheroid_green.py",
        PROJECT_ROOT / "qd_mnp_spheroid_equatorial.py",
    )
    metadata: dict[str, Any] = {
        "schema_name": SCHEMA_NAME,
        "schema_version": 1,
        "purpose": (
            "one-pole versus production multi-pole material-dispersion effect "
            "on nonlinear post-pulse exciton population"
        ),
        "computation_family": "material_mode_excitation_fluence_comparison",
        "branch_roles": [
            {
                "branch_id": "one",
                "n_modes": int(one_settings["material_fit_modes"]),
                "role": "diagnostic_one_pole_approximation",
                "fit_accuracy_policy": args.one_fit_accuracy_policy,
            },
            {
                "branch_id": "multi",
                "n_modes": int(multi_settings["material_fit_modes"]),
                "role": "production_multi_pole_approximation",
                "fit_accuracy_policy": multi_settings["fit_quality_policy"],
            },
        ],
        "scenario_sha256": canonical_sha256(scenario),
        "common_scenario": scenario,
        "requested_arguments": vars(args),
        "resolved_common_read_time_fs": one_read,
        "common_read_time_was_model_derived": bool(auto_read_time),
        "observable_definitions": {
            "p_exc_read": "rho_ee at the common post-pulse read time",
            "p_exc_max": "maximum rho_ee on the saved solver trajectory",
            "threshold": (
                "first rising crossing of target population, interpolated "
                "linearly in sqrt(fluence); left-censored if already reached"
            ),
            "threshold_ratio": "F_eta(one) / F_eta(multi)",
        },
        "model_scope": {
            "spatial_model": "FullQSSpheroidPulseModel for both branches",
            "only_changed_physics_input": "number of Lorentz material poles",
            "laser_propagation_direction_included": False,
            "electric_field_direction_included": True,
            "reference_material_time_backend": "not available; nonlinear comparison is one versus multi",
            "limitations": (
                "local quasistatics, homogeneous axisymmetric spheroid, point-dipole QD; "
                "no retardation, nonlocality, tunnelling or charge transfer"
            ),
        },
        "branch_metadata": {
            "one": one_metadata,
            "multi": multi_metadata,
        },
        "array_units": {
            "fluence_j_cm2": "J cm^-2",
            "pulse_e0_au": "atomic electric field",
            "pulse_e0_v_m": "V m^-1",
            "peak_intensity_w_cm2": "W cm^-2",
            "read_time_fs": "fs",
            "p_exc_read": "1",
            "p_exc_max": "1",
            "time_of_p_exc_max_fs": "fs",
            "threshold_fluence_j_cm2": "J cm^-2",
            "threshold_peak_intensity_w_cm2": "W cm^-2",
        },
        "constants": {
            "atomic_energy_eV": AU_ENERGY_EV,
            "atomic_energy_J": AU_ENERGY_J,
            "atomic_time_s": AU_TIME_S,
            "atomic_length_m": AU_LENGTH_M,
            "atomic_field_V_m": AU_FIELD_V_M,
            "atomic_dipole_C_m": AU_DIPOLE_C_M,
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "provenance": {
            "generator": str(Path(__file__).resolve()),
            "source_sha256": source_hashes(source_paths),
            "git": git_provenance(PROJECT_ROOT),
        },
    }
    return payload, metadata


def build_parser() -> argparse.ArgumentParser:
    parser = build_argument_parser()
    parser.description = __doc__
    parser.set_defaults(
        output=Path("results/article/excitation_fluence_material_comparison.npz")
    )
    parser.add_argument(
        "--one-material-fit-modes",
        type=int,
        default=1,
        help="Mode count for the deliberately approximate diagnostic branch.",
    )
    parser.add_argument(
        "--one-fit-accuracy-policy",
        choices=("raise", "warn"),
        default="warn",
        help="Only the material/modal accuracy policy may be relaxed for N=1.",
    )
    parser.add_argument("--target-population", type=float, default=0.5)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    payload, metadata = calculate_material_fluence_comparison(args)
    output = atomic_write_npz(
        args.output,
        payload,
        metadata,
        overwrite=args.overwrite,
    )
    print(f"Saved material-mode excitation-fluence comparison: {output}")
    return output


if __name__ == "__main__":
    main()
