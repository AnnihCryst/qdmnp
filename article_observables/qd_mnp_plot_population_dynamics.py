"""Plot the saved article QD population dynamics without rerunning the model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np


EXPECTED_SCHEMA_NAME = "qd_mnp_population_dynamics"
SUPPORTED_SCHEMA_VERSIONS = {1}


def load_population_artifact(
    path: str | Path,
    *,
    allow_unconverged: bool = False,
) -> dict[str, object]:
    artifact_path = Path(path)
    with np.load(artifact_path, allow_pickle=False) as artifact:
        required = {
            "metadata_json",
            "channel_ids",
            "channel_labels",
            "time_fs",
            "incident_field_au",
            "rho22",
            "population_final",
            "population_max",
        }
        missing = required.difference(artifact.files)
        if missing:
            raise ValueError(
                f"{artifact_path} is missing required arrays: "
                + ", ".join(sorted(missing))
            )
        metadata = json.loads(str(np.asarray(artifact["metadata_json"]).item()))
        schema = metadata.get("schema", {})
        if schema.get("name") != EXPECTED_SCHEMA_NAME:
            raise ValueError(
                f"Expected schema {EXPECTED_SCHEMA_NAME!r}, got "
                f"{schema.get('name')!r}."
            )
        if schema.get("version") not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"Unsupported population-dynamics schema version "
                f"{schema.get('version')!r}."
            )
        data = {
            name: np.array(artifact[name], copy=True) for name in required - {"metadata_json"}
        }
    data["metadata"] = metadata

    channel_ids = np.asarray(data["channel_ids"])
    channel_labels = np.asarray(data["channel_labels"])
    time_fs = np.asarray(data["time_fs"], dtype=float)
    rho22 = np.asarray(data["rho22"], dtype=float)
    incident = np.asarray(data["incident_field_au"], dtype=float)
    if channel_ids.ndim != 1 or channel_labels.shape != channel_ids.shape:
        raise ValueError("Channel identifiers and labels must be matching 1-D arrays.")
    channel_id_strings = channel_ids.astype(str)
    if channel_ids.size == 0 or channel_id_strings[0] != "bare_qd":
        raise ValueError("The first saved channel must be the isolated-QD reference.")
    if np.unique(channel_id_strings).size != channel_ids.size:
        raise ValueError("Channel identifiers must be unique.")
    if time_fs.ndim != 1 or incident.shape != time_fs.shape:
        raise ValueError("time_fs and incident_field_au must be matching 1-D arrays.")
    if rho22.shape != (channel_ids.size, time_fs.size):
        raise ValueError("rho22 must have shape (n_channels, n_times).")
    population_final = np.asarray(data["population_final"], dtype=float)
    population_max = np.asarray(data["population_max"], dtype=float)
    if population_final.shape != (channel_ids.size,) or population_max.shape != (
        channel_ids.size,
    ):
        raise ValueError("Population summaries must have shape (n_channels,).")
    if not (
        np.all(np.isfinite(time_fs))
        and np.all(np.isfinite(incident))
        and np.all(np.isfinite(rho22))
        and np.all(np.isfinite(population_final))
        and np.all(np.isfinite(population_max))
    ):
        raise ValueError("The saved plotting arrays must be finite.")
    if time_fs.size < 2 or np.any(np.diff(time_fs) <= 0.0):
        raise ValueError("time_fs must be strictly increasing and contain at least 2 points.")
    if not np.allclose(population_final, rho22[:, -1], rtol=1.0e-12, atol=1.0e-14):
        raise ValueError("population_final is inconsistent with the saved trajectories.")

    time_window = metadata.get("time_window")
    if not isinstance(time_window, dict):
        raise ValueError("metadata.time_window is required for convergence validation.")
    ratios = time_window.get("tail_ratio_by_channel")
    tolerance = time_window.get("tail_ratio_tolerance")
    try:
        tolerance_value = float(tolerance)
        ratio_values = {
            str(key): float(value) for key, value in dict(ratios).items()
        }
    except (TypeError, ValueError):
        tolerance_value = float("nan")
        ratio_values = {}
    tail_certificate = bool(
        time_window.get("all_channel_tails_converged") is True
        and np.isfinite(tolerance_value)
        and tolerance_value > 0.0
        and set(ratio_values) == set(channel_id_strings)
        and all(
            np.isfinite(value) and value <= tolerance_value
            for value in ratio_values.values()
        )
    )
    try:
        decay_fraction = float(
            time_window.get("population_decay_fraction_from_pulse_centre")
        )
        maximum_decay = float(
            time_window.get("maximum_population_decay_fraction_at_read")
        )
    except (TypeError, ValueError):
        decay_fraction = float("nan")
        maximum_decay = float("nan")
    decay_certificate = bool(
        np.isfinite(decay_fraction)
        and np.isfinite(maximum_decay)
        and 0.0 <= decay_fraction <= maximum_decay < 1.0
    )
    channels_metadata = metadata.get("channels")
    if not isinstance(channels_metadata, list) or len(channels_metadata) != len(
        channel_id_strings
    ):
        raise ValueError("metadata.channels does not match the saved channel arrays.")
    for index, channel in enumerate(channels_metadata):
        if not isinstance(channel, dict) or channel.get("id") != str(
            channel_id_strings[index]
        ):
            raise ValueError("metadata.channels order/identity is inconsistent.")
        if channel.get("label") != str(channel_labels[index]):
            raise ValueError("metadata.channels labels are inconsistent.")

    requested_inputs = metadata.get("requested_inputs", {})
    diagnostics_by_channel = metadata.get("diagnostics_by_channel")
    diagnostic_channels_well_formed = bool(
        isinstance(diagnostics_by_channel, dict)
        and set(diagnostics_by_channel) == set(channel_id_strings)
        and all(
            isinstance(diagnostics_by_channel[channel_id], dict)
            for channel_id in channel_id_strings
        )
    )
    try:
        positivity_tolerance = float(requested_inputs["positivity_tolerance"])
        spectral_tolerance = float(requested_inputs["max_spectral_leakage"])
        modal_rms_tolerance = float(
            requested_inputs["max_modal_normalized_rms"]
        )
        modal_relative_tolerance = float(
            requested_inputs["max_modal_relative_error"]
        )
    except (KeyError, TypeError, ValueError):
        positivity_tolerance = spectral_tolerance = float("nan")
        modal_rms_tolerance = modal_relative_tolerance = float("nan")

    def diagnostic_float(channel_id: str, name: str) -> float:
        if not diagnostic_channels_well_formed:
            return float("nan")
        try:
            return float(diagnostics_by_channel[channel_id][name])
        except (KeyError, TypeError, ValueError):
            return float("nan")

    solver_certificate = bool(
        diagnostic_channels_well_formed
        and all(
            diagnostics_by_channel[channel_id].get("solver_success") is True
            and diagnostics_by_channel[channel_id].get("solver_status") == 0
            and diagnostics_by_channel[channel_id].get("t_final_reached") is True
            and diagnostics_by_channel[channel_id].get("state_is_finite") is True
            and int(
                diagnostics_by_channel[channel_id].get("solver_n_steps", 0)
            )
            > 0
            and int(diagnostics_by_channel[channel_id].get("solver_nfev", 0))
            > 0
            and np.isfinite(
                diagnostic_float(channel_id, "max_step_limit_au")
            )
            and diagnostic_float(channel_id, "max_step_limit_au") > 0.0
            for channel_id in channel_id_strings
        )
    )
    native_minimum = np.asarray(
        [
            diagnostic_float(channel_id, "excited_population_min")
            for channel_id in channel_id_strings
        ],
        dtype=float,
    )
    native_maximum = np.asarray(
        [
            diagnostic_float(channel_id, "excited_population_max")
            for channel_id in channel_id_strings
        ],
        dtype=float,
    )
    minimum_density_eigenvalue = np.asarray(
        [
            diagnostic_float(channel_id, "min_density_eigenvalue")
            for channel_id in channel_id_strings
        ],
        dtype=float,
    )
    maximum_bloch_radius = np.asarray(
        [
            diagnostic_float(channel_id, "max_bloch_radius")
            for channel_id in channel_id_strings
        ],
        dtype=float,
    )
    positivity_certificate = bool(
        np.isfinite(positivity_tolerance)
        and positivity_tolerance >= 0.0
        and np.min(rho22) >= -positivity_tolerance
        and np.max(rho22) <= 1.0 + positivity_tolerance
        and np.min(population_final) >= -positivity_tolerance
        and np.max(population_final) <= 1.0 + positivity_tolerance
        and np.min(population_max) >= -positivity_tolerance
        and np.max(population_max) <= 1.0 + positivity_tolerance
        and np.all(np.isfinite(native_minimum))
        and np.all(np.isfinite(native_maximum))
        and np.all(np.isfinite(minimum_density_eigenvalue))
        and np.all(np.isfinite(maximum_bloch_radius))
        and np.all(native_minimum >= -positivity_tolerance)
        and np.all(native_maximum <= 1.0 + positivity_tolerance)
        and np.all(minimum_density_eigenvalue >= -positivity_tolerance)
        and np.all(maximum_bloch_radius <= 1.0 + 2.0 * positivity_tolerance)
        and np.allclose(
            population_max,
            native_maximum,
            rtol=1.0e-12,
            atol=max(1.0e-14, positivity_tolerance),
        )
        and np.all(
            population_max
            >= np.max(rho22, axis=1) - max(1.0e-14, positivity_tolerance)
        )
    )
    diagnostic_tail_certificate = bool(
        diagnostic_channels_well_formed
        and all(
            diagnostics_by_channel[channel_id].get("response_tail_converged")
            is True
            and np.isfinite(
                diagnostic_float(channel_id, "response_tail_tolerance")
            )
            and np.isclose(
                diagnostic_float(channel_id, "response_tail_tolerance"),
                tolerance_value,
                rtol=1.0e-12,
                atol=0.0,
            )
            and np.isclose(
                diagnostic_float(channel_id, "response_tail_ratio"),
                ratio_values.get(channel_id, np.nan),
                rtol=1.0e-12,
                atol=0.0,
            )
            for channel_id in channel_id_strings
        )
    )

    hybrid_channel_ids = [str(value) for value in channel_id_strings[1:]]
    spectral_names = (
        "pulse_spectral",
        "qd_source_spectral",
        "mnp_drive_spectral",
        "mnp_dipole_spectral",
        "mnp_field_spectral",
    )
    spectral_certificate = bool(
        diagnostic_channels_well_formed
        and np.isfinite(spectral_tolerance)
        and 0.0 <= spectral_tolerance < 1.0
        and all(
            all(
                np.isfinite(diagnostic_float(channel_id, f"{name}_leakage"))
                and np.isfinite(
                    diagnostic_float(
                        channel_id, f"{name}_fraction_in_fit_window"
                    )
                )
                and -1.0e-12
                <= diagnostic_float(
                    channel_id, f"{name}_fraction_in_fit_window"
                )
                <= 1.0 + 1.0e-12
                and -1.0e-12
                <= diagnostic_float(channel_id, f"{name}_leakage")
                <= spectral_tolerance
                and np.isclose(
                    diagnostic_float(channel_id, f"{name}_leakage")
                    + diagnostic_float(
                        channel_id, f"{name}_fraction_in_fit_window"
                    ),
                    1.0,
                    rtol=1.0e-11,
                    atol=1.0e-12,
                )
                for name in spectral_names
            )
            for channel_id in hybrid_channel_ids
        )
    )
    work_passivity_certificate = bool(
        diagnostic_channels_well_formed
        and all(
            diagnostics_by_channel[channel_id].get(
                "work_nonnegative_within_tolerance"
            )
            is True
            and np.isfinite(
                diagnostic_float(channel_id, "work_passivity_tolerance_au")
            )
            and diagnostic_float(channel_id, "work_passivity_tolerance_au")
            >= 0.0
            and np.isfinite(
                diagnostic_float(channel_id, "work_from_incident_field_j")
            )
            and np.isfinite(
                diagnostic_float(channel_id, "sigma_energy_transfer_cm2")
            )
            for channel_id in hybrid_channel_ids
        )
    )

    model_by_channel = metadata.get("model_by_channel")
    model_metadata_well_formed = bool(
        isinstance(model_by_channel, dict)
        and set(model_by_channel) == set(hybrid_channel_ids)
        and all(
            isinstance(model_by_channel[channel_id], dict)
            for channel_id in hybrid_channel_ids
        )
    )
    spatial_certificate = model_metadata_well_formed
    modal_certificate = model_metadata_well_formed
    stability_certificate = model_metadata_well_formed
    dark_reduction_certificate = model_metadata_well_formed
    material_passivity_certificate = model_metadata_well_formed
    for channel_id in hybrid_channel_ids:
        if not model_metadata_well_formed:
            break
        model = model_by_channel[channel_id]
        spatial = model.get("spatial_convergence", {})
        modal = model.get("modal_transform", {})
        stability = model.get("coupled_stability", {})
        reduction = model.get("dark_reduction", {})
        material = model.get("material_fit", {})
        try:
            spatial_tolerance = float(spatial["tolerance"])
            spatial_half_order = float(
                spatial["max_half_order_relative_change"]
            )
            spatial_tail_mass = float(spatial["max_tail_block_relative_mass"])
            spatial_ok = bool(
                spatial.get("accepted") is True
                and np.isfinite(spatial_tolerance)
                and spatial_tolerance > 0.0
                and np.isfinite(spatial_half_order)
                and np.isfinite(spatial_tail_mass)
                and spatial_half_order <= spatial_tolerance
                and spatial_tail_mass <= spatial_tolerance
                and int(spatial["audit_grid_points"]) >= 2
            )
        except (KeyError, TypeError, ValueError):
            spatial_ok = False
        spatial_certificate = bool(spatial_certificate and spatial_ok)

        try:
            modal_values = (
                float(modal["max_normalized_rms"]),
                float(modal["K_normalized_rms"]),
                float(modal["max_relative_error"]),
                float(modal["K_max_relative_error"]),
            )
            modal_ok = bool(
                modal.get("accepted") is True
                and modal.get("passive_on_audit_grid") is True
                and np.isfinite(modal_rms_tolerance)
                and modal_rms_tolerance > 0.0
                and np.isfinite(modal_relative_tolerance)
                and modal_relative_tolerance > 0.0
                and all(np.isfinite(value) for value in modal_values)
                and max(modal_values[:2]) <= modal_rms_tolerance
                and max(modal_values[2:]) <= modal_relative_tolerance
                and int(modal["audit_grid_points"]) >= 2
            )
        except (KeyError, TypeError, ValueError):
            modal_ok = False
        modal_certificate = bool(modal_certificate and modal_ok)

        try:
            spectral_radius = float(stability["spectral_radius_au"])
            stability_tolerance = float(stability["tolerance_au"])
            decay_rate = float(stability["decay_rate_estimate_au"])
            abscissa_available = stability["spectral_abscissa_available"] is True
            abscissa_raw = stability.get("spectral_abscissa_au")
            abscissa_ok = bool(
                (not abscissa_available and abscissa_raw is None)
                or (
                    abscissa_available
                    and np.isfinite(float(abscissa_raw))
                    and float(abscissa_raw) <= stability_tolerance
                )
            )
            stability_ok = bool(
                stability.get("stable") is True
                and np.isfinite(spectral_radius)
                and spectral_radius > 0.0
                and np.isfinite(stability_tolerance)
                and stability_tolerance >= 0.0
                and np.isfinite(decay_rate)
                and decay_rate > 0.0
                and isinstance(stability.get("eigensolver"), str)
                and bool(stability.get("eigensolver"))
                and int(stability["coherent_state_dimension"]) > 0
                and abscissa_ok
            )
        except (KeyError, TypeError, ValueError):
            stability_ok = False
        stability_certificate = bool(stability_certificate and stability_ok)

        try:
            material_ok = bool(
                material.get("passive_on_fit_window") is True
                and material.get("globally_passive") is True
                and np.isfinite(
                    float(material["minimum_imaginary_alpha_fit_window"])
                )
                and int(material["passivity_grid_points"]) >= 2
            )
        except (KeyError, TypeError, ValueError):
            material_ok = False
        material_passivity_certificate = bool(
            material_passivity_certificate and material_ok
        )

        reduction_ok = False
        if isinstance(reduction, dict) and reduction.get("applied") is False:
            reduction_ok = reduction.get("current_transfer_reaudit") is None
        elif isinstance(reduction, dict) and reduction.get("applied") is True:
            reaudit = reduction.get("current_transfer_reaudit", {})
            try:
                reduction_rms_tolerance = float(reduction["rms_tolerance"])
                reduction_max_tolerance = float(reduction["max_tolerance"])
                reaudit_rms_tolerance = float(reaudit["rms_tolerance"])
                reaudit_max_tolerance = float(reaudit["max_tolerance"])
                reduction_ok = bool(
                    reduction.get("accepted") is True
                    and reduction.get("passive_on_audit_grid") is True
                    and np.isfinite(reduction_rms_tolerance)
                    and reduction_rms_tolerance > 0.0
                    and np.isfinite(reduction_max_tolerance)
                    and reduction_max_tolerance > 0.0
                    and float(reduction["max_normalized_rms"])
                    <= reduction_rms_tolerance
                    and float(reduction["max_normalized_error"])
                    <= reduction_max_tolerance
                    and int(reduction["original_dark_mode_count"]) >= 1
                    and int(reduction["positive_dark_mode_count"]) >= 1
                    and int(reduction["reduced_node_count"]) >= 1
                    and reaudit.get("accepted") is True
                    and reaudit.get("passive_on_audit_grid") is True
                    and np.isfinite(reaudit_rms_tolerance)
                    and reaudit_rms_tolerance > 0.0
                    and np.isfinite(reaudit_max_tolerance)
                    and reaudit_max_tolerance > 0.0
                    and float(reaudit["normalized_rms"])
                    <= reaudit_rms_tolerance
                    and float(reaudit["max_normalized_error"])
                    <= reaudit_max_tolerance
                )
            except (KeyError, TypeError, ValueError):
                reduction_ok = False
        dark_reduction_certificate = bool(
            dark_reduction_certificate and reduction_ok
        )

    if not allow_unconverged:
        failures = []
        if not tail_certificate or not diagnostic_tail_certificate:
            failures.append("response-tail certificate")
        if not decay_certificate:
            failures.append("population-decay-at-read certificate")
        if not solver_certificate:
            failures.append("solver-completion certificate")
        if not positivity_certificate:
            failures.append("population-positivity certificate")
        if not spectral_certificate:
            failures.append("material-fit spectral-coverage certificate")
        if not work_passivity_certificate:
            failures.append("external-work passivity certificate")
        if not spatial_certificate:
            failures.append("full-QS spatial-convergence certificate")
        if not modal_certificate:
            failures.append("full-QS modal-transform certificate")
        if not stability_certificate:
            failures.append("full-QS coupled-stability certificate")
        if not dark_reduction_certificate:
            failures.append("full-QS dark-reduction certificate")
        if not material_passivity_certificate:
            failures.append("material-passivity certificate")
        if failures:
            raise ValueError(
                "Refusing to plot an uncertified population-dynamics artifact; failed: "
                + ", ".join(failures)
                + ". Recalculate with certified model/solver settings, or use "
                "allow_unconverged=True only for diagnostic inspection."
            )
    return data


def plot_population_dynamics(
    data: dict[str, object],
    *,
    output_path: str | Path,
    show_field: bool = True,
    title: str | None = None,
    dpi: int = 220,
    show: bool = False,
    x_min_fs: float | None = None,
    x_max_fs: float | None = None,
) -> Figure:
    time_fs = np.asarray(data["time_fs"], dtype=float)
    rho22 = np.asarray(data["rho22"], dtype=float)
    labels = [str(value) for value in np.asarray(data["channel_labels"])]
    incident = np.asarray(data["incident_field_au"], dtype=float)

    figure, axis = plt.subplots(figsize=(8.0, 5.2), constrained_layout=True)
    for index, label in enumerate(labels):
        line_style = "--" if index == 0 else "-"
        line_width = 1.8 if index == 0 else 2.0
        axis.plot(
            time_fs,
            rho22[index],
            linestyle=line_style,
            linewidth=line_width,
            label=label,
        )
    axis.set_xlabel("Time, fs")
    axis.set_ylabel(r"Excited-state population $\rho_{ee}(t)$")
    population_min = float(np.min(rho22))
    population_maximum = float(np.max(rho22))
    axis.set_ylim(
        min(-0.02, 1.05 * population_min),
        max(1.02, 1.05 * population_maximum),
    )
    axis.grid(alpha=0.22)
    if x_min_fs is not None or x_max_fs is not None:
        left = float(time_fs[0]) if x_min_fs is None else float(x_min_fs)
        right = float(time_fs[-1]) if x_max_fs is None else float(x_max_fs)
        if not np.isfinite(left) or not np.isfinite(right) or left >= right:
            raise ValueError("The displayed time window must satisfy x_min_fs < x_max_fs.")
        axis.set_xlim(left, right)

    handles, legend_labels = axis.get_legend_handles_labels()
    if show_field and np.max(np.abs(incident)) > np.finfo(float).tiny:
        field_axis = axis.twinx()
        normalized_field = incident / np.max(np.abs(incident))
        field_line = field_axis.plot(
            time_fs,
            normalized_field,
            color="0.45",
            linewidth=0.9,
            alpha=0.55,
            label=r"$E_{\rm inc}(t)/E_0$",
        )[0]
        field_axis.set_ylabel("Normalized incident field", color="0.4")
        field_axis.set_ylim(-1.15, 1.15)
        field_axis.tick_params(axis="y", colors="0.4")
        handles.append(field_line)
        legend_labels.append(field_line.get_label())
    axis.legend(handles, legend_labels, loc="best", frameon=False)

    metadata = data["metadata"]
    resolved_pulse = metadata.get("resolved_pulse", {})
    if title is None:
        energy = resolved_pulse.get("energy_eV")
        fluence = resolved_pulse.get("fluence_j_cm2")
        if energy is not None and fluence is not None:
            title = rf"$E_L={float(energy):.4g}$ eV, $\mathcal{{F}}={float(fluence):.3g}$ J/cm$^2$"
    if title:
        axis.set_title(title)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi)
    if show:
        plt.show()
    return figure


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="NPZ produced by the calculation script.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--title")
    parser.add_argument("--no-field", action="store_true")
    parser.add_argument("--x-min-fs", type=float)
    parser.add_argument("--x-max-fs", type=float)
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--allow-unconverged",
        action="store_true",
        help=(
            "Plot despite failed numerical/full-QS certificates for diagnostic "
            "inspection only."
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    data = load_population_artifact(
        args.input,
        allow_unconverged=args.allow_unconverged,
    )
    output = (
        args.output
        if args.output is not None
        else args.input.with_name(f"{args.input.stem}.png")
    )
    figure = plot_population_dynamics(
        data,
        output_path=output,
        show_field=not args.no_field,
        title=args.title,
        dpi=args.dpi,
        show=args.show,
        x_min_fs=args.x_min_fs,
        x_max_fs=args.x_max_fs,
    )
    if not args.show:
        plt.close(figure)
    print(f"Saved population-dynamics figure to {output.resolve()}")


if __name__ == "__main__":
    main()
