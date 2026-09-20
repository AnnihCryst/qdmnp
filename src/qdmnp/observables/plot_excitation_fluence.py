"""Plot the article QD-excitation dependence from a saved artifact only.

This module intentionally imports neither the full-QS solver nor any other
QD--MNP calculation module.  Styling and light derived quantities (for example
threshold fluences) can therefore be changed without repeating propagation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


SCHEMA_NAME = "qd_mnp_excitation_fluence"
SUPPORTED_SCHEMA_VERSIONS = {1}
REQUIRED_ARRAYS = {
    "metadata_json",
    "channel_id",
    "fluence_j_cm2",
    "p_exc_read",
    "p_exc_max",
    "read_time_fs",
    "bare_diagnostic__response_tail_converged",
    "full_diagnostic__response_tail_converged",
    "population_decay_fraction_at_read",
    "fluence_grid_midpoint_interpolation_error",
    "fluence_grid_converged_by_channel",
    "maximum_isolated_pulse_area_step_rad",
    "isolated_qd_pulse_area_rad",
    "bare_diagnostic__solver_success",
    "bare_diagnostic__t_final_reached",
    "bare_diagnostic__state_is_finite",
    "bare_diagnostic__min_density_eigenvalue",
    "bare_diagnostic__boundary_envelope_fraction",
    "bare_diagnostic__pulse_spectral_leakage",
    "bare_diagnostic__response_tail_ratio",
    "bare_diagnostic__response_tail_tolerance",
    "full_diagnostic__solver_success",
    "full_diagnostic__t_final_reached",
    "full_diagnostic__state_is_finite",
    "full_diagnostic__min_density_eigenvalue",
    "full_diagnostic__boundary_envelope_fraction",
    "full_diagnostic__work_nonnegative_within_tolerance",
    "full_diagnostic__pulse_spectral_leakage",
    "full_diagnostic__qd_source_spectral_leakage",
    "full_diagnostic__mnp_drive_spectral_leakage",
    "full_diagnostic__mnp_dipole_spectral_leakage",
    "full_diagnostic__mnp_field_spectral_leakage",
    "full_diagnostic__response_tail_ratio",
    "full_diagnostic__response_tail_tolerance",
}

COMPATIBLE_SETTING_KEYS = (
    "pulse_energy_ev",
    "pulse_tau_fs",
    "pulse_tau_kind",
    "c_nm",
    "a_nm",
    "qd_radius_nm",
    "gap_nm",
    "eps_m",
    "eps_qd",
    "d_debye",
    "omega0_ev",
    "gamma_population_mev",
    "gamma2_coherence_mev",
    "qd_dipole_convention",
    "fit_min_ev",
    "fit_max_ev",
    "weight_center_ev",
    "weight_sigma_ev",
    "spatial_order_max",
    "post_fs",
    "start_sigma",
    "method",
    "rtol",
    "atol",
    "points_per_fastest_cycle",
    "max_auto_tail_extensions",
    "tail_ratio_tolerance",
    "tail_window_fraction",
    "max_population_decay_fraction_at_read",
    "max_spectral_leakage",
    "positivity_tolerance",
    "max_modal_normalized_rms",
    "max_modal_relative_error",
    "spatial_convergence_rtol",
    "max_bright_fit_normalized_rms",
    "max_bright_fit_pointwise_relative_error",
    "modal_audit_points",
    "reduction_fit_grid_points",
    "reduction_audit_grid_points",
    "reduction_reaudit_points",
    "reduction_rms_tolerance",
    "reduction_max_tolerance",
    "reduction_max_nodes",
    "max_isolated_pulse_area_step_rad",
    "max_fluence_grid_midpoint_error",
)


def _validated_bool_array(
    data: dict[str, np.ndarray], name: str, shape: tuple[int, ...]
) -> np.ndarray:
    values = np.asarray(data[name])
    if values.dtype.kind != "b":
        raise ValueError(f"{name} must have boolean dtype, got {values.dtype}.")
    if values.shape != shape:
        raise ValueError(f"{name} has shape {values.shape}; expected {shape}.")
    return values


def load_excitation_artifact(
    path: str | Path,
    *,
    allow_unconverged: bool = False,
    allow_approximate_material_fit: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load and validate a pickle-free excitation-fluence NPZ artifact."""

    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(f"Excitation-fluence artifact does not exist: {artifact}")
    with np.load(artifact, allow_pickle=False) as archive:
        missing = REQUIRED_ARRAYS.difference(archive.files)
        if missing:
            raise ValueError(f"Artifact is missing required arrays: {sorted(missing)}")
        try:
            metadata = json.loads(str(archive["metadata_json"].item()))
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("metadata_json is not a scalar valid JSON document.") from exc
        data = {key: np.asarray(archive[key]).copy() for key in archive.files if key != "metadata_json"}

    if metadata.get("schema_name") != SCHEMA_NAME:
        raise ValueError(
            f"Expected schema_name={SCHEMA_NAME!r}, got {metadata.get('schema_name')!r}."
        )
    version = metadata.get("schema_version")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(f"Unsupported excitation-fluence schema version: {version!r}.")

    channel_ids = np.asarray(data["channel_id"]).astype(str)
    fluence = np.asarray(data["fluence_j_cm2"], dtype=float)
    population = np.asarray(data["p_exc_read"], dtype=float)
    population_max = np.asarray(data["p_exc_max"], dtype=float)
    if channel_ids.ndim != 1 or fluence.ndim != 1:
        raise ValueError("channel_id and fluence_j_cm2 must be one-dimensional.")
    if channel_ids.size < 2 or channel_ids[0] != "bare_qd":
        raise ValueError("channel_id must begin with the isolated-QD reference 'bare_qd'.")
    if np.unique(channel_ids).size != channel_ids.size:
        raise ValueError("channel_id values must be unique.")
    expected_shape = (channel_ids.size, fluence.size)
    if population.shape != expected_shape or population_max.shape != expected_shape:
        raise ValueError(
            "Population arrays must have shape (number of channels, number of fluences)."
        )
    if (
        fluence.size < 1
        or np.any(~np.isfinite(fluence))
        or np.any(fluence <= 0.0)
        or np.any(np.diff(fluence) <= 0.0)
    ):
        raise ValueError("fluence_j_cm2 must be finite, positive, and strictly increasing.")
    if np.any(~np.isfinite(population)) or np.any(~np.isfinite(population_max)):
        raise ValueError("Population arrays contain non-finite values.")
    tolerance = 1.0e-6
    if np.any(population < -tolerance) or np.any(population > 1.0 + tolerance):
        raise ValueError("p_exc_read lies outside the physical [0, 1] interval.")
    if np.any(population_max < -tolerance) or np.any(population_max > 1.0 + tolerance):
        raise ValueError("p_exc_max lies outside the physical [0, 1] interval.")
    if np.any(population_max + tolerance < population):
        raise ValueError("p_exc_max cannot be smaller than the read-time population.")
    read_time = np.asarray(data["read_time_fs"], dtype=float)
    if (
        read_time.shape != fluence.shape
        or np.any(~np.isfinite(read_time))
        or np.any(read_time <= 0.0)
    ):
        raise ValueError("read_time_fs must be finite and positive for every fluence.")
    if not np.allclose(
        read_time,
        read_time[0],
        rtol=0.0,
        atol=1.0e-10 * max(abs(float(read_time[0])), 1.0),
    ):
        raise ValueError("All P_exc(fluence) points must share one physical read time.")
    bare_tail = _validated_bool_array(
        data, "bare_diagnostic__response_tail_converged", fluence.shape
    )
    full_tail = _validated_bool_array(
        data,
        "full_diagnostic__response_tail_converged",
        (channel_ids.size - 1, fluence.size),
    )
    decay_fraction = np.asarray(
        data["population_decay_fraction_at_read"], dtype=float
    )
    if decay_fraction.shape != fluence.shape or np.any(~np.isfinite(decay_fraction)):
        raise ValueError(
            "population_decay_fraction_at_read must be finite for every fluence."
        )
    settings = metadata.get("resolved_settings", {})
    numerical_failures: list[str] = []
    try:
        decay_limit = float(settings["max_population_decay_fraction_at_read"])
        positivity_tolerance = float(settings["positivity_tolerance"])
        spectral_tolerance = float(settings["max_spectral_leakage"])
        grid_error_limit = float(settings["max_fluence_grid_midpoint_error"])
        pulse_area_step_limit = float(settings["max_isolated_pulse_area_step_rad"])
        configured_tail_tolerance = float(settings["tail_ratio_tolerance"])
    except (KeyError, TypeError, ValueError):
        decay_limit = positivity_tolerance = spectral_tolerance = float("nan")
        grid_error_limit = pulse_area_step_limit = float("nan")
        configured_tail_tolerance = float("nan")

    if not (
        np.isfinite(decay_limit)
        and 0.0 < decay_limit < 1.0
        and np.all(decay_fraction <= decay_limit)
    ):
        numerical_failures.append("post-pulse population-decay window")

    grid_error = np.asarray(
        data["fluence_grid_midpoint_interpolation_error"], dtype=float
    )
    grid_accepted = _validated_bool_array(
        data, "fluence_grid_converged_by_channel", (channel_ids.size,)
    )
    pulse_area_step = np.asarray(
        data["maximum_isolated_pulse_area_step_rad"], dtype=float
    )
    pulse_area = np.asarray(data["isolated_qd_pulse_area_rad"], dtype=float)
    if grid_error.shape != (channel_ids.size,):
        raise ValueError("Saved fluence-grid diagnostics do not match channel_id.")
    if pulse_area_step.shape != ():
        raise ValueError("maximum_isolated_pulse_area_step_rad must be a scalar.")
    if (
        pulse_area.shape != fluence.shape
        or np.any(~np.isfinite(pulse_area))
        or np.any(pulse_area <= 0.0)
        or np.any(np.diff(pulse_area) <= 0.0)
    ):
        raise ValueError(
            "isolated_qd_pulse_area_rad must be finite, positive, and increasing."
        )
    proportionality = pulse_area / np.sqrt(fluence)
    if not np.allclose(
        proportionality,
        proportionality[0],
        rtol=2.0e-12,
        atol=0.0,
    ):
        raise ValueError("The saved isolated-QD pulse area is inconsistent with fluence.")
    recomputed_area_step = float(np.max(np.abs(np.diff(pulse_area))))
    if not np.isclose(
        float(pulse_area_step), recomputed_area_step, rtol=2.0e-12, atol=0.0
    ):
        raise ValueError("The saved maximum pulse-area step is inconsistent.")
    recomputed_grid_accepted = (
        np.isfinite(grid_error)
        & (grid_error <= grid_error_limit)
        & np.isfinite(float(pulse_area_step))
        & (float(pulse_area_step) <= pulse_area_step_limit)
    )
    if not (
        np.isfinite(grid_error_limit)
        and grid_error_limit > 0.0
        and np.isfinite(pulse_area_step_limit)
        and pulse_area_step_limit > 0.0
        and np.array_equal(grid_accepted, recomputed_grid_accepted)
        and np.all(grid_accepted)
    ):
        numerical_failures.append("fluence-grid resolution")

    bare_shape = fluence.shape
    full_shape = (channel_ids.size - 1, fluence.size)
    for prefix, shape in (("bare_diagnostic__", bare_shape), ("full_diagnostic__", full_shape)):
        for name in ("solver_success", "t_final_reached", "state_is_finite"):
            values = _validated_bool_array(data, prefix + name, shape)
            if not np.all(values):
                numerical_failures.append(prefix + name)

        minimum_density = np.asarray(data[prefix + "min_density_eigenvalue"], dtype=float)
        if minimum_density.shape != shape:
            raise ValueError(
                f"{prefix + 'min_density_eigenvalue'} has shape {minimum_density.shape}; "
                f"expected {shape}."
            )
        if not (
            np.isfinite(positivity_tolerance)
            and positivity_tolerance >= 0.0
            and np.all(np.isfinite(minimum_density))
            and np.all(minimum_density >= -positivity_tolerance)
        ):
            numerical_failures.append(prefix + "density-matrix positivity")

        boundary_envelope = np.asarray(
            data[prefix + "boundary_envelope_fraction"], dtype=float
        )
        if boundary_envelope.shape != shape:
            raise ValueError(
                f"{prefix + 'boundary_envelope_fraction'} has shape "
                f"{boundary_envelope.shape}; expected {shape}."
            )
        if not (
            np.all(np.isfinite(boundary_envelope))
            and np.all(boundary_envelope >= 0.0)
            and np.all(boundary_envelope <= 1.0e-6)
        ):
            numerical_failures.append(prefix + "incident-pulse boundary")

        tail_ratio = np.asarray(data[prefix + "response_tail_ratio"], dtype=float)
        tail_tolerance = np.asarray(data[prefix + "response_tail_tolerance"], dtype=float)
        if tail_ratio.shape != shape or tail_tolerance.shape != shape:
            raise ValueError(f"{prefix}response-tail diagnostics have inconsistent shapes.")
        if not (
            np.all(np.isfinite(tail_ratio))
            and np.all(np.isfinite(tail_tolerance))
            and np.all(tail_tolerance > 0.0)
            and np.isfinite(configured_tail_tolerance)
            and configured_tail_tolerance > 0.0
            and np.allclose(
                tail_tolerance,
                configured_tail_tolerance,
                rtol=2.0e-12,
                atol=0.0,
            )
            and np.all(tail_ratio <= tail_tolerance)
        ):
            numerical_failures.append(prefix + "response-tail ratio")

    full_work = _validated_bool_array(
        data, "full_diagnostic__work_nonnegative_within_tolerance", full_shape
    )
    if not np.all(full_work):
        numerical_failures.append("full-QS work passivity")

    spectral_arrays = [
        np.asarray(data["bare_diagnostic__pulse_spectral_leakage"], dtype=float),
        *[
            np.asarray(data[f"full_diagnostic__{name}"], dtype=float)
            for name in (
                "pulse_spectral_leakage",
                "qd_source_spectral_leakage",
                "mnp_drive_spectral_leakage",
                "mnp_dipole_spectral_leakage",
                "mnp_field_spectral_leakage",
            )
        ],
    ]
    if spectral_arrays[0].shape != bare_shape or any(
        values.shape != full_shape for values in spectral_arrays[1:]
    ):
        raise ValueError("Saved spectral-leakage diagnostics have inconsistent shapes.")
    if not (
        np.isfinite(spectral_tolerance)
        and 0.0 <= spectral_tolerance < 1.0
        and all(np.all(np.isfinite(values)) for values in spectral_arrays)
        and all(np.all(values <= spectral_tolerance) for values in spectral_arrays)
    ):
        numerical_failures.append("material-fit spectral coverage")

    channels = metadata.get("channels")
    if not isinstance(channels, list) or len(channels) != channel_ids.size:
        raise ValueError("metadata.channels does not match channel_id.")
    material_fit_failures: list[str] = []
    for index, (channel_id, channel) in enumerate(zip(channel_ids, channels)):
        if not isinstance(channel, dict) or channel.get("channel_id") != channel_id:
            raise ValueError("metadata.channels order/identity is inconsistent.")
        if index == 0:
            continue
        material_fit = channel.get("material_fit", {})
        modal = channel.get("modal_transform_diagnostics", {})
        spatial = channel.get("spatial_convergence_diagnostics", {})
        stability = channel.get("coupled_stability_diagnostics", {})
        reduction = channel.get("dark_reduction")
        reduction_accepted = reduction is None or (
            isinstance(reduction, dict)
            and reduction.get("diagnostics", {}).get("accepted") is True
            and reduction.get("current_transfer_reaudit", {}).get("accepted") is True
        )
        if not (
            modal.get("passive_on_audit_grid") is True
            and spatial.get("accepted") is True
            and stability.get("stable") is True
            and reduction_accepted
        ):
            numerical_failures.append(f"full-QS model certificate ({channel_id})")
        if material_fit.get("passive_for_all_positive_frequencies") is not True:
            numerical_failures.append(f"material passivity ({channel_id})")
        try:
            rms_limit = float(material_fit["max_fit_normalized_rms_gate"])
            pointwise_limit = float(
                material_fit["max_fit_pointwise_relative_error_gate"]
            )
            fit_accepted = (
                max(
                    float(material_fit["normalized_rms_alpha"]),
                    float(material_fit["normalized_rms_inv_alpha"]),
                )
                <= rms_limit
                and float(material_fit["max_normalized_alpha_error"])
                <= pointwise_limit
            )
        except (KeyError, TypeError, ValueError):
            fit_accepted = False
        if not fit_accepted:
            material_fit_failures.append(channel_id)
        if modal.get("accepted") is not True:
            if fit_accepted:
                numerical_failures.append(f"modal-transform accuracy ({channel_id})")
            elif channel_id not in material_fit_failures:
                material_fit_failures.append(channel_id)

    if not allow_unconverged:
        if not (np.all(bare_tail) and np.all(full_tail)):
            numerical_failures.append("response-tail flags")
        if numerical_failures:
            raise ValueError(
                "Refusing to plot an uncertified excitation-fluence artifact; failed: "
                + ", ".join(dict.fromkeys(numerical_failures))
                + ". Recalculate with a resolved grid/tail/model, or use "
                "allow_unconverged=True only for diagnostic inspection."
            )
    if material_fit_failures and not allow_approximate_material_fit:
        raise ValueError(
            "The material polarizability fit misses its configured accuracy gate "
            "for channel(s): "
            + ", ".join(material_fit_failures)
            + ". Use allow_approximate_material_fit=True only for an explicitly "
            "labelled one-pole/approximation control."
        )
    return data, metadata


def _channel_labels(metadata: dict[str, Any], channel_ids: np.ndarray) -> list[str]:
    definitions = {
        str(item.get("channel_id")): str(item.get("label", item.get("channel_id")))
        for item in metadata.get("channels", [])
        if isinstance(item, dict) and item.get("channel_id") is not None
    }
    return [definitions.get(str(channel_id), str(channel_id)) for channel_id in channel_ids]


def threshold_fluence(
    fluence_j_cm2: np.ndarray,
    population: np.ndarray,
    target_population: float,
) -> float:
    """Return the first field-amplitude interpolation crossing of a target.

    The nonlinear curve need not be monotone.  The first upward crossing is
    reported because it is the minimum sampled fluence interval that reaches
    the requested excitation.  NaN means that the saved grid does not bracket
    an upward crossing: the target is either not reached or already exceeded
    at the lower boundary.
    """

    x = np.asarray(fluence_j_cm2, dtype=float)
    y = np.asarray(population, dtype=float)
    if not np.isfinite(target_population) or not 0.0 < target_population < 1.0:
        raise ValueError("target_population must lie strictly between 0 and 1.")
    at_or_above = np.flatnonzero(y >= target_population)
    if at_or_above.size == 0:
        return float("nan")
    right = int(at_or_above[0])
    if right == 0:
        return float("nan")
    left = right - 1
    y0, y1 = float(y[left]), float(y[right])
    if y1 == y0:
        return float(x[right])
    weight = (target_population - y0) / (y1 - y0)
    amplitude = np.sqrt(x[left]) + weight * (
        np.sqrt(x[right]) - np.sqrt(x[left])
    )
    return float(amplitude * amplitude)


def _threshold_with_status(
    fluence_j_cm2: np.ndarray,
    population: np.ndarray,
    target_population: float,
) -> tuple[float, str]:
    x = np.asarray(fluence_j_cm2, dtype=float)
    y = np.asarray(population, dtype=float)
    if not np.isfinite(target_population) or not 0.0 < target_population < 1.0:
        raise ValueError("target_population must lie strictly between 0 and 1.")
    indices = np.flatnonzero(y >= target_population)
    if indices.size == 0:
        return float("nan"), "not_reached"
    if int(indices[0]) == 0:
        return float(x[0]), "left_censored"
    return threshold_fluence(x, y, target_population), "interpolated"


def plot_excitation_fluence(
    data: dict[str, np.ndarray],
    metadata: dict[str, Any],
    output_path: str | Path,
    *,
    target_population: float | None = None,
    include_maximum: bool = False,
    ratio_panel: bool = False,
    linear_x: bool = False,
    title: str | None = None,
    dpi: int = 250,
    show: bool = False,
) -> Path:
    """Create the article-style plot using only arrays already in the NPZ."""

    output = Path(output_path)
    channel_ids = np.asarray(data["channel_id"]).astype(str)
    labels = _channel_labels(metadata, channel_ids)
    fluence = np.asarray(data["fluence_j_cm2"], dtype=float)
    population = np.asarray(data["p_exc_read"], dtype=float)
    population_max = np.asarray(data["p_exc_max"], dtype=float)
    fit_label = _material_fit_label(metadata)
    thresholds = (
        None
        if target_population is None
        else [
            _threshold_with_status(fluence, row, target_population)
            for row in population
        ]
    )

    if ratio_panel:
        figure, (axis, ratio_axis) = plt.subplots(
            2,
            1,
            figsize=(8.0, 7.2),
            sharex=True,
            gridspec_kw={"height_ratios": (2.2, 1.0)},
        )
    else:
        figure, axis = plt.subplots(figsize=(8.0, 5.3))
        ratio_axis = None

    colors = ["0.15", "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    markers = ["o", "s", "^", "D", "v", "P"]
    for index, (channel_id, label) in enumerate(zip(channel_ids, labels)):
        plotted_label = label
        if channel_id != "bare_qd":
            plotted_label += f"; {fit_label}"
        if thresholds is not None:
            threshold, status = thresholds[index]
            if status == "interpolated":
                suffix = rf", $\mathcal{{F}}_{{req}}\approx{threshold:.2g}$ J cm$^{{-2}}$"
            elif status == "left_censored":
                suffix = rf", $\mathcal{{F}}_{{req}}\leq{threshold:.2g}$ J cm$^{{-2}}$"
            else:
                suffix = r", target not reached"
            plotted_label += suffix
        axis.plot(
            fluence,
            population[index],
            color=colors[index % len(colors)],
            marker=markers[index % len(markers)],
            ms=4.5,
            lw=1.8,
            label=plotted_label,
        )
        if include_maximum:
            axis.plot(
                fluence,
                population_max[index],
                color=colors[index % len(colors)],
                lw=1.0,
                ls="--",
                alpha=0.65,
            )

    if target_population is not None:
        if not 0.0 < target_population < 1.0:
            raise ValueError("target_population must lie strictly between 0 and 1.")
        axis.axhline(target_population, color="0.45", lw=1.1, ls=":")
        assert thresholds is not None
        finite = [value for value, status in thresholds if status == "interpolated"]
        if finite:
            axis.text(
                0.02,
                0.03,
                rf"first crossing of $P_{{\rm exc}}={target_population:g}$ is interpolated in $\sqrt{{\mathcal{{F}}}}$",
                transform=axis.transAxes,
                fontsize=8.5,
                color="0.35",
            )

    axis.set_ylabel(r"Post-pulse exciton population $P_{\rm exc}$")
    axis.set_ylim(-0.025, 1.025)
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=8.4, ncol=2)
    if include_maximum:
        axis.text(
            0.98,
            0.03,
            "solid: read-time; dashed: trajectory maximum",
            transform=axis.transAxes,
            ha="right",
            fontsize=8.2,
            color="0.35",
        )

    if ratio_axis is not None:
        bare = population[0]
        for index in range(1, channel_ids.size):
            ratio = np.divide(
                population[index],
                bare,
                out=np.full_like(bare, np.nan),
                where=np.abs(bare) > 1.0e-14,
            )
            ratio_axis.plot(
                fluence,
                ratio,
                color=colors[index % len(colors)],
                marker=markers[index % len(markers)],
                ms=3.8,
                lw=1.5,
            )
        ratio_axis.axhline(1.0, color="0.3", lw=1.0, ls=":")
        ratio_axis.set_ylabel(r"$P_{\rm exc}/P_{\rm exc}^{bare}$")
        ratio_axis.grid(True, which="both", alpha=0.25)

    x_axis = ratio_axis if ratio_axis is not None else axis
    x_axis.set_xlabel(r"Incident pulse fluence $\mathcal{F}$, J cm$^{-2}$")
    if not linear_x:
        x_axis.set_xscale("log")
    if title:
        axis.set_title(title)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(figure)
    return output


def _material_fit_label(metadata: dict[str, Any]) -> str:
    hybrid_channels = [
        channel
        for channel in metadata.get("channels", [])[1:]
        if isinstance(channel, dict)
        and isinstance(channel.get("material_fit"), dict)
        and channel["material_fit"].get("n_modes") is not None
    ]
    modes = {int(channel["material_fit"]["n_modes"]) for channel in hybrid_channels}
    if len(modes) == 1:
        approximate = False
        modal_uncertified = False
        for channel in hybrid_channels:
            fit = channel["material_fit"]
            try:
                fit_misses_gate = (
                    max(
                        float(fit["normalized_rms_alpha"]),
                        float(fit["normalized_rms_inv_alpha"]),
                    )
                    > float(fit["max_fit_normalized_rms_gate"])
                    or float(fit["max_normalized_alpha_error"])
                    > float(fit["max_fit_pointwise_relative_error_gate"])
                )
                approximate = approximate or fit_misses_gate
                modal_uncertified = modal_uncertified or (
                    channel.get("modal_transform_diagnostics", {}).get("accepted")
                    is not True
                )
            except (KeyError, TypeError, ValueError):
                approximate = True
        label = f"N={modes.pop()}"
        if approximate:
            return label + " (approx.)"
        if modal_uncertified:
            return label + " (modal uncertified)"
        return label
    return str(metadata.get("artifact_file", "material fit"))


def _compatible_value(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return bool(
            np.isclose(float(left), float(right), rtol=2.0e-12, atol=0.0)
        )
    return left == right


def validate_excitation_comparison(
    artifacts: list[tuple[dict[str, np.ndarray], dict[str, Any]]],
) -> None:
    """Reject an N-pole comparison if anything except material fit differs."""

    if not artifacts:
        raise ValueError("At least one excitation artifact is required.")
    reference_data, reference_metadata = artifacts[0]
    reference_channels = np.asarray(reference_data["channel_id"]).astype(str)
    reference_fluence = np.asarray(reference_data["fluence_j_cm2"], dtype=float)
    reference_read_time = np.asarray(reference_data["read_time_fs"], dtype=float)
    reference_settings = reference_metadata.get("resolved_settings", {})
    for index, (data, metadata) in enumerate(artifacts[1:], start=2):
        channels = np.asarray(data["channel_id"]).astype(str)
        if not np.array_equal(channels, reference_channels):
            raise ValueError(f"Artifact {index} has a different channel set/order.")
        if not np.allclose(
            np.asarray(data["fluence_j_cm2"], dtype=float),
            reference_fluence,
            rtol=2.0e-12,
            atol=0.0,
        ):
            raise ValueError(f"Artifact {index} has a different fluence grid.")
        if not np.allclose(
            np.asarray(data["read_time_fs"], dtype=float),
            reference_read_time,
            rtol=0.0,
            atol=1.0e-10 * max(abs(float(reference_read_time[0])), 1.0),
        ):
            raise ValueError(f"Artifact {index} has a different population read time.")
        settings = metadata.get("resolved_settings", {})
        differing = [
            key
            for key in COMPATIBLE_SETTING_KEYS
            if not _compatible_value(reference_settings.get(key), settings.get(key))
        ]
        if differing:
            raise ValueError(
                f"Artifact {index} differs in non-material scenario setting(s): "
                + ", ".join(differing)
            )
        for name in ("p_exc_read", "p_exc_max"):
            if not np.allclose(
                np.asarray(data[name], dtype=float)[0],
                np.asarray(reference_data[name], dtype=float)[0],
                rtol=2.0e-9,
                atol=2.0e-11,
            ):
                raise ValueError(
                    f"Artifact {index} has a different isolated-QD reference in {name}."
                )


def plot_excitation_comparison(
    artifacts: list[tuple[dict[str, np.ndarray], dict[str, Any]]],
    output_path: str | Path,
    *,
    channel_filter: list[str] | None = None,
    target_population: float | None = None,
    include_maximum: bool = False,
    ratio_panel: bool = False,
    linear_x: bool = False,
    title: str | None = None,
    dpi: int = 250,
    show: bool = False,
) -> Path:
    """Overlay compatible material fits using only data saved in NPZ files."""

    validate_excitation_comparison(artifacts)
    reference_data, reference_metadata = artifacts[0]
    channel_ids = np.asarray(reference_data["channel_id"]).astype(str)
    labels = _channel_labels(reference_metadata, channel_ids)
    label_by_id = dict(zip(channel_ids, labels))
    if channel_filter is None:
        selected = list(channel_ids)
    else:
        selected = list(dict.fromkeys(channel_filter))
        unknown = [channel for channel in selected if channel not in channel_ids]
        if unknown:
            raise ValueError("Unknown --channel value(s): " + ", ".join(unknown))
        if not selected:
            raise ValueError("At least one channel must be selected.")
    selected_indices = [int(np.flatnonzero(channel_ids == value)[0]) for value in selected]
    fluence = np.asarray(reference_data["fluence_j_cm2"], dtype=float)

    if ratio_panel:
        figure, (axis, ratio_axis) = plt.subplots(
            2,
            1,
            figsize=(8.4, 7.4),
            sharex=True,
            gridspec_kw={"height_ratios": (2.2, 1.0)},
        )
    else:
        figure, axis = plt.subplots(figsize=(8.4, 5.5))
        ratio_axis = None

    colors = ["0.15", "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    line_styles = ["-", "--", ":", "-."]
    markers = ["o", "s", "^", "D"]
    for artifact_index, (data, metadata) in enumerate(artifacts):
        population = np.asarray(data["p_exc_read"], dtype=float)
        population_max = np.asarray(data["p_exc_max"], dtype=float)
        fit_label = _material_fit_label(metadata)
        for channel_index in selected_indices:
            channel_id = channel_ids[channel_index]
            if channel_id == "bare_qd" and artifact_index > 0:
                continue
            curve_label = label_by_id[channel_id]
            if channel_id != "bare_qd" or len(artifacts) == 1:
                curve_label += f"; {fit_label}"
            if target_population is not None:
                crossing, crossing_status = _threshold_with_status(
                    fluence, population[channel_index], target_population
                )
                if crossing_status == "interpolated":
                    curve_label += rf", $\mathcal{{F}}_{{req}}\approx{crossing:.2g}$ J cm$^{{-2}}$"
                elif crossing_status == "left_censored":
                    curve_label += rf", $\mathcal{{F}}_{{req}}\leq{crossing:.2g}$ J cm$^{{-2}}$"
                else:
                    curve_label += ", target not reached"
            axis.plot(
                fluence,
                population[channel_index],
                color=colors[channel_index % len(colors)],
                ls=line_styles[artifact_index % len(line_styles)],
                marker=markers[artifact_index % len(markers)],
                ms=4.0,
                lw=1.8,
                label=curve_label,
            )
            if include_maximum:
                axis.plot(
                    fluence,
                    population_max[channel_index],
                    color=colors[channel_index % len(colors)],
                    ls=line_styles[artifact_index % len(line_styles)],
                    lw=0.9,
                    alpha=0.4,
                )
            if ratio_axis is not None and channel_id != "bare_qd":
                bare = population[0]
                ratio_axis.plot(
                    fluence,
                    np.divide(
                        population[channel_index],
                        bare,
                        out=np.full_like(bare, np.nan),
                        where=np.abs(bare) > 1.0e-14,
                    ),
                    color=colors[channel_index % len(colors)],
                    ls=line_styles[artifact_index % len(line_styles)],
                    marker=markers[artifact_index % len(markers)],
                    ms=3.5,
                    lw=1.5,
                )

    if target_population is not None:
        if not 0.0 < target_population < 1.0:
            raise ValueError("target_population must lie strictly between 0 and 1.")
        axis.axhline(target_population, color="0.45", lw=1.1, ls=":")
        axis.text(
            0.02,
            0.03,
            rf"first crossing of $P_{{\rm exc}}={target_population:g}$ is interpolated in $\sqrt{{\mathcal{{F}}}}$",
            transform=axis.transAxes,
            fontsize=8.5,
            color="0.35",
        )
    axis.set_ylabel(r"Post-pulse exciton population $P_{\rm exc}$")
    axis.set_ylim(-0.025, 1.025)
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=7.8, ncol=2)
    if ratio_axis is not None:
        ratio_axis.axhline(1.0, color="0.3", lw=1.0, ls=":")
        ratio_axis.set_ylabel(r"$P_{\rm exc}/P_{\rm exc}^{bare}$")
        ratio_axis.grid(True, which="both", alpha=0.25)
    x_axis = ratio_axis if ratio_axis is not None else axis
    x_axis.set_xlabel(r"Incident pulse fluence $\mathcal{F}$, J cm$^{-2}$")
    if not linear_x:
        x_axis.set_xscale("log")
    if title:
        axis.set_title(title)
    figure.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(figure)
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "artifact",
        nargs="+",
        type=Path,
        help="One or more compatible NPZ files written by the calculation script.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--target-population", type=float)
    parser.add_argument("--include-maximum", action="store_true")
    parser.add_argument("--ratio-panel", action="store_true")
    parser.add_argument("--linear-x", action="store_true")
    parser.add_argument("--title")
    parser.add_argument("--dpi", type=int, default=250)
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--channel",
        action="append",
        help="Channel to plot; repeat to show a compact subset in a comparison.",
    )
    parser.add_argument(
        "--allow-unconverged",
        action="store_true",
        help="Permit a diagnostic plot even when saved response-tail gates failed.",
    )
    parser.add_argument(
        "--allow-approximate-material-fit",
        action="store_true",
        help="Permit an explicitly labelled inaccurate one-pole material-fit control.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")
    artifacts = [
        load_excitation_artifact(
            artifact,
            allow_unconverged=args.allow_unconverged,
            allow_approximate_material_fit=args.allow_approximate_material_fit,
        )
        for artifact in args.artifact
    ]
    first_path = args.artifact[0]
    default_suffix = ".comparison.png" if len(artifacts) > 1 else ".png"
    output = (
        args.output
        if args.output is not None
        else first_path.with_name(f"{first_path.stem}{default_suffix}")
    )
    if len(artifacts) == 1 and args.channel is None:
        data, metadata = artifacts[0]
        result = plot_excitation_fluence(
            data,
            metadata,
            output,
            target_population=args.target_population,
            include_maximum=args.include_maximum,
            ratio_panel=args.ratio_panel,
            linear_x=args.linear_x,
            title=args.title,
            dpi=args.dpi,
            show=args.show,
        )
    else:
        result = plot_excitation_comparison(
            artifacts,
            output,
            channel_filter=args.channel,
            target_population=args.target_population,
            include_maximum=args.include_maximum,
            ratio_panel=args.ratio_panel,
            linear_x=args.linear_x,
            title=args.title,
            dpi=args.dpi,
            show=args.show,
        )
    print(f"Saved excitation-fluence plot: {result}")
    if args.target_population is not None:
        for artifact_index, (data, metadata) in enumerate(artifacts):
            channel_ids = np.asarray(data["channel_id"]).astype(str)
            labels = _channel_labels(metadata, channel_ids)
            selected = set(args.channel) if args.channel is not None else set(channel_ids)
            for channel_id, label, row in zip(
                channel_ids, labels, np.asarray(data["p_exc_read"], dtype=float)
            ):
                if channel_id not in selected:
                    continue
                if channel_id == "bare_qd" and artifact_index > 0:
                    continue
                value, status = _threshold_with_status(
                    np.asarray(data["fluence_j_cm2"], dtype=float),
                    row,
                    args.target_population,
                )
                if status == "interpolated":
                    rendered = f"approximately {value:.8g} J/cm^2"
                elif status == "left_censored":
                    rendered = f"at most {value:.8g} J/cm^2 (left-censored)"
                else:
                    rendered = "target not reached on saved grid"
                fit_suffix = (
                    "" if channel_id == "bare_qd" else f"; {_material_fit_label(metadata)}"
                )
                print(
                    f"  {label}{fit_suffix}: "
                    f"F_req(P={args.target_population:g}) is {rendered}"
                )
    return result


if __name__ == "__main__":
    main()
