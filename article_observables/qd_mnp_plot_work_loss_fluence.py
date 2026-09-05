"""Plot a saved full-QS work-loss-versus-fluence NPZ artifact.

This module deliberately imports no QD--MNP solver.  It only validates and
plots arrays already produced by the companion calculation module in this directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np


WORK_LOSS_FLUENCE_SCHEMA = "qd_mnp.full_qs_work_loss_fluence"
SUPPORTED_SCHEMA_VERSIONS = {1}
REQUIRED_ARRAYS = {
    "metadata_json",
    "channel_key",
    "channel_label",
    "orientation",
    "requested_fluence_j_cm2",
    "fluence_j_cm2",
    "sigma_spectral_qs_work_loss_cm2",
    "sigma_bare_mnp_qs_work_loss_cm2",
    "delta_sigma_spectral_qs_work_loss_cm2",
    "sigma_energy_transfer_cm2",
    "sigma_bare_mnp_energy_transfer_cm2",
    "delta_sigma_energy_transfer_cm2",
    "response_tail_converged",
    "observable_window_converged",
    "bare_mnp_energy_integration_converged",
    "sigma_spectral_half_window_relative_change",
    "sigma_energy_half_window_relative_change",
    "sigma_bare_mnp_energy_cutoff_check_cm2",
    "sigma_bare_mnp_energy_cutoff_relative_change",
    "sigma_bare_mnp_energy_quadrature_relative_error",
    "solver_success",
    "solver_status",
    "t_final_reached",
    "state_is_finite",
    "work_nonnegative_within_tolerance",
    "bare_mnp_work_nonnegative_within_tolerance",
    "response_tail_ratio",
    "min_density_eigenvalue",
    "boundary_envelope_fraction",
    "pulse_spectral_leakage",
    "qd_source_spectral_leakage",
    "mnp_dipole_spectral_leakage",
    "mnp_drive_spectral_leakage",
    "mnp_field_spectral_leakage",
    "isolated_qd_pulse_area_rad",
    "fluence_grid_observable_name",
    "fluence_grid_midpoint_absolute_error_cm2",
    "fluence_grid_curve_scale_cm2",
    "fluence_grid_midpoint_normalized_error",
    "fluence_grid_converged_by_observable",
    "fluence_grid_converged_by_channel",
    "maximum_isolated_pulse_area_step_rad",
    "bare_mnp_qs_work_loss_by_channel_cm2",
    "bare_mnp_energy_transfer_by_channel_cm2",
    "bare_mnp_energy_cutoff_check_by_channel_cm2",
    "bare_mnp_energy_cutoff_relative_change_by_channel",
    "bare_mnp_energy_quadrature_absolute_error_by_channel_cm2",
    "bare_mnp_energy_quadrature_relative_error_by_channel",
    "bare_mnp_energy_integration_converged_by_channel",
    "bare_mnp_work_from_incident_field_by_channel_j",
    "bare_mnp_work_cutoff_check_by_channel_j",
    "bare_mnp_work_passivity_tolerance_by_channel_j",
    "bare_mnp_work_nonnegative_by_channel",
    "bare_mnp_reference_fluence_by_channel_j_cm2",
    "bare_mnp_dimensionless_carrier_frequency_by_channel",
    "bare_mnp_dimensionless_cutoff_check_by_channel",
    "bare_mnp_dimensionless_cutoff_full_by_channel",
}

BOOLEAN_ARRAYS = (
    "response_tail_converged",
    "observable_window_converged",
    "bare_mnp_energy_integration_converged",
    "solver_success",
    "t_final_reached",
    "state_is_finite",
    "work_nonnegative_within_tolerance",
    "bare_mnp_work_nonnegative_within_tolerance",
    "fluence_grid_converged_by_observable",
    "fluence_grid_converged_by_channel",
    "bare_mnp_energy_integration_converged_by_channel",
    "bare_mnp_work_nonnegative_by_channel",
)

GRID_AUDIT_OBSERVABLES = (
    "sigma_spectral_qs_work_loss_cm2",
    "sigma_energy_transfer_cm2",
    "delta_sigma_spectral_qs_work_loss_cm2",
    "delta_sigma_energy_transfer_cm2",
)


def _recompute_grid_diagnostics(
    fluence_j_cm2: np.ndarray,
    observable_values_cm2: np.ndarray,
    pulse_area_rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Recompute the calculator's nested-grid certificate from saved data."""

    fluence = np.asarray(fluence_j_cm2, dtype=float)
    values = np.asarray(observable_values_cm2, dtype=float)
    pulse_area = np.asarray(pulse_area_rad, dtype=float)
    curve_scale = np.max(np.abs(values), axis=2)
    if fluence.size < 3:
        absolute_error = np.full(values.shape[:2], np.inf, dtype=float)
        normalized_error = np.full(values.shape[:2], np.inf, dtype=float)
    else:
        coordinate = np.sqrt(fluence)
        audit_indices = np.arange(1, fluence.size - 1, 2, dtype=int)
        if audit_indices.size == 0:
            absolute_error = np.full(values.shape[:2], np.inf, dtype=float)
            normalized_error = np.full(values.shape[:2], np.inf, dtype=float)
        else:
            left = audit_indices - 1
            right = audit_indices + 1
            fractions = (coordinate[audit_indices] - coordinate[left]) / (
                coordinate[right] - coordinate[left]
            )
            interpolated = values[:, :, left] + fractions[None, None, :] * (
                values[:, :, right] - values[:, :, left]
            )
            absolute_error = np.max(
                np.abs(values[:, :, audit_indices] - interpolated), axis=2
            )
            normalized_error = np.divide(
                absolute_error,
                curve_scale,
                out=np.zeros_like(absolute_error),
                where=curve_scale > np.finfo(float).tiny,
            )
            normalized_error[
                (curve_scale <= np.finfo(float).tiny)
                & (absolute_error > np.finfo(float).tiny)
            ] = np.inf
    maximum_area_step = (
        float(np.max(np.abs(np.diff(pulse_area))))
        if pulse_area.size >= 2
        else float("inf")
    )
    return absolute_error, curve_scale, normalized_error, maximum_area_step


def load_work_loss_artifact(
    path: str | Path,
    *,
    allow_unconverged: bool = False,
    allow_approximate_material_fit: bool = False,
) -> dict[str, object]:
    """Load and validate the plotting subset without enabling pickle."""

    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(f"Work-loss artifact does not exist: {artifact}")
    with np.load(artifact, allow_pickle=False) as stored:
        missing = sorted(REQUIRED_ARRAYS.difference(stored.files))
        if missing:
            raise ValueError(
                "Work-loss artifact is missing required array(s): " + ", ".join(missing)
            )
        try:
            metadata = json.loads(str(np.asarray(stored["metadata_json"]).item()))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("metadata_json is not a scalar valid JSON document.") from exc
        data = {
            name: np.array(stored[name], copy=True)
            for name in stored.files
            if name != "metadata_json"
        }

    if metadata.get("schema") != WORK_LOSS_FLUENCE_SCHEMA:
        raise ValueError(
            f"Unsupported artifact schema {metadata.get('schema')!r}; "
            f"expected {WORK_LOSS_FLUENCE_SCHEMA!r}."
        )
    version = metadata.get("schema_version")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(f"Unsupported work-loss artifact schema version: {version!r}.")

    for name in BOOLEAN_ARRAYS:
        if np.asarray(data[name]).dtype != np.dtype(bool):
            raise ValueError(
                f"{name} must be stored with the exact NumPy boolean dtype."
            )

    channel_keys = np.asarray(data["channel_key"])
    channel_labels = np.asarray(data["channel_label"])
    orientations = np.asarray(data["orientation"])
    requested_fluence = np.asarray(data["requested_fluence_j_cm2"], dtype=float)
    if channel_keys.ndim != 1 or channel_keys.size == 0:
        raise ValueError("channel_key must be a non-empty one-dimensional array.")
    if np.unique(channel_keys.astype(str)).size != channel_keys.size:
        raise ValueError("channel_key values must be unique.")
    if channel_labels.shape != channel_keys.shape or orientations.shape != channel_keys.shape:
        raise ValueError("Channel labels/orientations do not match channel_key.")
    if requested_fluence.ndim != 1 or requested_fluence.size == 0:
        raise ValueError("requested_fluence_j_cm2 must be a non-empty 1D array.")
    if np.any(~np.isfinite(requested_fluence)) or np.any(requested_fluence <= 0.0):
        raise ValueError("Requested fluences must be finite and positive.")
    if np.any(np.diff(requested_fluence) <= 0.0):
        raise ValueError("Requested fluences must be strictly increasing.")

    expected_shape = (channel_keys.size, requested_fluence.size)
    for name in (
        "fluence_j_cm2",
        "sigma_spectral_qs_work_loss_cm2",
        "sigma_bare_mnp_qs_work_loss_cm2",
        "delta_sigma_spectral_qs_work_loss_cm2",
        "sigma_energy_transfer_cm2",
        "sigma_bare_mnp_energy_transfer_cm2",
        "delta_sigma_energy_transfer_cm2",
    ):
        values = np.asarray(data[name], dtype=float)
        if values.shape != expected_shape:
            raise ValueError(
                f"{name} has shape {values.shape}; expected {expected_shape}."
            )
        if np.any(~np.isfinite(values)):
            raise ValueError(f"{name} contains a non-finite value.")
    if np.any(np.asarray(data["fluence_j_cm2"], dtype=float) <= 0.0):
        raise ValueError("Resolved fluences must be positive.")
    resolved_fluence = np.asarray(data["fluence_j_cm2"], dtype=float)
    if not np.allclose(
        resolved_fluence,
        requested_fluence[None, :],
        rtol=2.0e-12,
        atol=0.0,
    ):
        raise ValueError("Resolved fluences are inconsistent with the requested grid.")
    if not np.allclose(
        np.asarray(data["delta_sigma_spectral_qs_work_loss_cm2"], dtype=float),
        np.asarray(data["sigma_spectral_qs_work_loss_cm2"], dtype=float)
        - np.asarray(data["sigma_bare_mnp_qs_work_loss_cm2"], dtype=float),
        rtol=2.0e-13,
        atol=0.0,
    ):
        raise ValueError("The saved carrier-frequency work-loss difference is inconsistent.")
    if not np.allclose(
        np.asarray(data["delta_sigma_energy_transfer_cm2"], dtype=float),
        np.asarray(data["sigma_energy_transfer_cm2"], dtype=float)
        - np.asarray(data["sigma_bare_mnp_energy_transfer_cm2"], dtype=float),
        rtol=2.0e-13,
        atol=0.0,
    ):
        raise ValueError("The saved pulse-integrated work difference is inconsistent.")

    grid_names = np.asarray(data["fluence_grid_observable_name"]).astype(str)
    if (
        grid_names.shape != (len(GRID_AUDIT_OBSERVABLES),)
        or tuple(grid_names.tolist()) != GRID_AUDIT_OBSERVABLES
    ):
        raise ValueError("fluence_grid_observable_name has an unsupported order.")
    pulse_area = np.asarray(data["isolated_qd_pulse_area_rad"], dtype=float)
    if (
        pulse_area.shape != requested_fluence.shape
        or np.any(~np.isfinite(pulse_area))
        or np.any(pulse_area <= 0.0)
        or np.any(np.diff(pulse_area) <= 0.0)
    ):
        raise ValueError(
            "isolated_qd_pulse_area_rad must be finite, positive, strictly "
            "increasing, and match the fluence grid."
        )
    grid_shape = (channel_keys.size, len(GRID_AUDIT_OBSERVABLES))
    saved_grid_absolute = np.asarray(
        data["fluence_grid_midpoint_absolute_error_cm2"], dtype=float
    )
    saved_grid_scale = np.asarray(data["fluence_grid_curve_scale_cm2"], dtype=float)
    saved_grid_normalized = np.asarray(
        data["fluence_grid_midpoint_normalized_error"], dtype=float
    )
    saved_grid_by_observable = np.asarray(
        data["fluence_grid_converged_by_observable"], dtype=bool
    )
    saved_grid_by_channel = np.asarray(
        data["fluence_grid_converged_by_channel"], dtype=bool
    )
    saved_area_step = np.asarray(
        data["maximum_isolated_pulse_area_step_rad"], dtype=float
    )
    for name, values in (
        ("fluence_grid_midpoint_absolute_error_cm2", saved_grid_absolute),
        ("fluence_grid_curve_scale_cm2", saved_grid_scale),
        ("fluence_grid_midpoint_normalized_error", saved_grid_normalized),
        ("fluence_grid_converged_by_observable", saved_grid_by_observable),
    ):
        if values.shape != grid_shape:
            raise ValueError(f"{name} has shape {values.shape}; expected {grid_shape}.")
        if np.any(np.isnan(values)) or np.any(values < 0.0):
            raise ValueError(f"{name} must contain non-negative values, or +inf.")
    if saved_grid_by_channel.shape != (channel_keys.size,):
        raise ValueError("fluence_grid_converged_by_channel has an invalid shape.")
    if saved_area_step.shape != ():
        raise ValueError("maximum_isolated_pulse_area_step_rad must be a scalar.")
    grid_values = np.stack(
        [np.asarray(data[name], dtype=float) for name in GRID_AUDIT_OBSERVABLES],
        axis=1,
    )
    (
        recomputed_grid_absolute,
        recomputed_grid_scale,
        recomputed_grid_normalized,
        recomputed_area_step,
    ) = _recompute_grid_diagnostics(requested_fluence, grid_values, pulse_area)
    for name, saved, recomputed in (
        (
            "fluence_grid_midpoint_absolute_error_cm2",
            saved_grid_absolute,
            recomputed_grid_absolute,
        ),
        ("fluence_grid_curve_scale_cm2", saved_grid_scale, recomputed_grid_scale),
        (
            "fluence_grid_midpoint_normalized_error",
            saved_grid_normalized,
            recomputed_grid_normalized,
        ),
    ):
        if not np.allclose(saved, recomputed, rtol=2.0e-13, atol=0.0):
            raise ValueError(f"{name} is inconsistent with the saved work curves.")
    if not np.isclose(
        float(saved_area_step), recomputed_area_step, rtol=2.0e-13, atol=0.0
    ):
        raise ValueError(
            "maximum_isolated_pulse_area_step_rad is inconsistent with pulse area."
        )
    grid_metadata = metadata.get("fluence_grid", {})
    try:
        grid_error_tolerance = float(
            grid_metadata["max_midpoint_normalized_error"]
        )
        grid_area_step_tolerance = float(
            grid_metadata["max_isolated_pulse_area_step_rad"]
        )
    except (KeyError, TypeError, ValueError):
        grid_error_tolerance = grid_area_step_tolerance = float("nan")
    recomputed_grid_by_observable = np.asarray(
        recomputed_grid_normalized <= grid_error_tolerance, dtype=bool
    )
    recomputed_grid_by_channel = np.asarray(
        np.all(recomputed_grid_by_observable, axis=1)
        & (recomputed_area_step <= grid_area_step_tolerance),
        dtype=bool,
    )
    if not np.array_equal(saved_grid_by_observable, recomputed_grid_by_observable):
        raise ValueError(
            "fluence_grid_converged_by_observable is inconsistent with its tolerance."
        )
    if not np.array_equal(saved_grid_by_channel, recomputed_grid_by_channel):
        raise ValueError(
            "fluence_grid_converged_by_channel is inconsistent with its tolerance."
        )
    if grid_metadata.get("observable_names") != list(GRID_AUDIT_OBSERVABLES):
        raise ValueError("metadata.fluence_grid observable names are inconsistent.")
    if grid_metadata.get("accepted") is not bool(np.all(saved_grid_by_channel)):
        raise ValueError("metadata.fluence_grid accepted flag is inconsistent.")
    if grid_metadata.get("accepted_by_channel") != [
        bool(value) for value in saved_grid_by_channel
    ]:
        raise ValueError(
            "metadata.fluence_grid accepted_by_channel is inconsistent."
        )
    metadata_grid_errors = grid_metadata.get("midpoint_normalized_error")
    if not isinstance(metadata_grid_errors, list) or len(metadata_grid_errors) != (
        channel_keys.size
    ):
        raise ValueError("metadata.fluence_grid midpoint errors are inconsistent.")
    for channel_index, row in enumerate(metadata_grid_errors):
        if not isinstance(row, list) or len(row) != len(GRID_AUDIT_OBSERVABLES):
            raise ValueError("metadata.fluence_grid midpoint errors are inconsistent.")
        for observable_index, metadata_value in enumerate(row):
            saved_value = saved_grid_normalized[channel_index, observable_index]
            if metadata_value is None:
                if np.isfinite(saved_value):
                    raise ValueError(
                        "metadata.fluence_grid midpoint errors are inconsistent."
                    )
            elif not np.isclose(
                float(metadata_value), saved_value, rtol=2.0e-13, atol=0.0
            ):
                raise ValueError(
                    "metadata.fluence_grid midpoint errors are inconsistent."
                )
    metadata_area_step = grid_metadata.get(
        "maximum_isolated_pulse_area_step_rad"
    )
    if metadata_area_step is None:
        area_step_metadata_matches = not np.isfinite(float(saved_area_step))
    else:
        try:
            area_step_metadata_matches = bool(
                np.isclose(
                    float(metadata_area_step),
                    float(saved_area_step),
                    rtol=2.0e-13,
                    atol=0.0,
                )
            )
        except (TypeError, ValueError):
            area_step_metadata_matches = False
    if not area_step_metadata_matches:
        raise ValueError("metadata.fluence_grid pulse-area step is inconsistent.")

    bare_channel_float_names = (
        "bare_mnp_qs_work_loss_by_channel_cm2",
        "bare_mnp_energy_transfer_by_channel_cm2",
        "bare_mnp_energy_cutoff_check_by_channel_cm2",
        "bare_mnp_energy_cutoff_relative_change_by_channel",
        "bare_mnp_energy_quadrature_absolute_error_by_channel_cm2",
        "bare_mnp_energy_quadrature_relative_error_by_channel",
        "bare_mnp_work_from_incident_field_by_channel_j",
        "bare_mnp_work_cutoff_check_by_channel_j",
        "bare_mnp_work_passivity_tolerance_by_channel_j",
        "bare_mnp_reference_fluence_by_channel_j_cm2",
        "bare_mnp_dimensionless_carrier_frequency_by_channel",
        "bare_mnp_dimensionless_cutoff_check_by_channel",
        "bare_mnp_dimensionless_cutoff_full_by_channel",
    )
    bare_by_channel: dict[str, np.ndarray] = {}
    for name in bare_channel_float_names:
        values = np.asarray(data[name], dtype=float)
        if values.shape != (channel_keys.size,):
            raise ValueError(
                f"{name} has shape {values.shape}; expected {(channel_keys.size,)}."
            )
        if np.any(~np.isfinite(values)):
            raise ValueError(f"{name} contains a non-finite value.")
        bare_by_channel[name] = values
    for name in (
        "bare_mnp_energy_cutoff_relative_change_by_channel",
        "bare_mnp_energy_quadrature_absolute_error_by_channel_cm2",
        "bare_mnp_energy_quadrature_relative_error_by_channel",
        "bare_mnp_work_passivity_tolerance_by_channel_j",
    ):
        if np.any(bare_by_channel[name] < 0.0):
            raise ValueError(f"{name} must be non-negative.")
    bare_integration_by_channel = np.asarray(
        data["bare_mnp_energy_integration_converged_by_channel"], dtype=bool
    )
    bare_nonnegative_by_channel = np.asarray(
        data["bare_mnp_work_nonnegative_by_channel"], dtype=bool
    )
    for name, values in (
        (
            "bare_mnp_energy_integration_converged_by_channel",
            bare_integration_by_channel,
        ),
        ("bare_mnp_work_nonnegative_by_channel", bare_nonnegative_by_channel),
    ):
        if values.shape != (channel_keys.size,):
            raise ValueError(f"{name} has an invalid shape.")

    def require_broadcast_match(matrix_name: str, channel_name: str) -> None:
        if not np.allclose(
            np.asarray(data[matrix_name], dtype=float),
            bare_by_channel[channel_name][:, None],
            rtol=2.0e-13,
            atol=0.0,
        ):
            raise ValueError(
                f"{matrix_name} is not a fluence-independent copy of {channel_name}."
            )

    require_broadcast_match(
        "sigma_bare_mnp_qs_work_loss_cm2",
        "bare_mnp_qs_work_loss_by_channel_cm2",
    )
    require_broadcast_match(
        "sigma_bare_mnp_energy_transfer_cm2",
        "bare_mnp_energy_transfer_by_channel_cm2",
    )
    require_broadcast_match(
        "sigma_bare_mnp_energy_cutoff_check_cm2",
        "bare_mnp_energy_cutoff_check_by_channel_cm2",
    )
    require_broadcast_match(
        "sigma_bare_mnp_energy_cutoff_relative_change",
        "bare_mnp_energy_cutoff_relative_change_by_channel",
    )
    require_broadcast_match(
        "sigma_bare_mnp_energy_quadrature_relative_error",
        "bare_mnp_energy_quadrature_relative_error_by_channel",
    )
    if not np.array_equal(
        np.asarray(data["bare_mnp_energy_integration_converged"], dtype=bool),
        np.broadcast_to(bare_integration_by_channel[:, None], expected_shape),
    ):
        raise ValueError("The bare-MNP integration certificate varies with fluence.")
    if not np.array_equal(
        np.asarray(data["bare_mnp_work_nonnegative_within_tolerance"], dtype=bool),
        np.broadcast_to(bare_nonnegative_by_channel[:, None], expected_shape),
    ):
        raise ValueError("The bare-MNP passivity certificate varies with fluence.")
    reference_fluence_by_channel = bare_by_channel[
        "bare_mnp_reference_fluence_by_channel_j_cm2"
    ]
    if not np.allclose(
        reference_fluence_by_channel,
        requested_fluence[0],
        rtol=2.0e-12,
        atol=0.0,
    ):
        raise ValueError("The bare-MNP reference fluence is inconsistent.")
    if not np.allclose(
        bare_by_channel["bare_mnp_work_from_incident_field_by_channel_j"],
        bare_by_channel["bare_mnp_energy_transfer_by_channel_cm2"]
        * reference_fluence_by_channel,
        rtol=2.0e-13,
        atol=0.0,
    ):
        raise ValueError("Bare-MNP work and W/F cross section are inconsistent.")
    if not np.allclose(
        bare_by_channel["bare_mnp_work_cutoff_check_by_channel_j"],
        bare_by_channel["bare_mnp_energy_cutoff_check_by_channel_cm2"]
        * reference_fluence_by_channel,
        rtol=2.0e-13,
        atol=0.0,
    ):
        raise ValueError("Bare-MNP cutoff work and cutoff cross section disagree.")
    if np.any(
        bare_by_channel["bare_mnp_work_passivity_tolerance_by_channel_j"] < 0.0
    ):
        raise ValueError("Bare-MNP passivity tolerance must be non-negative.")
    carrier_dimensionless = bare_by_channel[
        "bare_mnp_dimensionless_carrier_frequency_by_channel"
    ]
    if np.any(carrier_dimensionless <= 0.0) or not np.allclose(
        bare_by_channel["bare_mnp_dimensionless_cutoff_check_by_channel"],
        carrier_dimensionless + 10.0,
        rtol=2.0e-13,
        atol=0.0,
    ) or not np.allclose(
        bare_by_channel["bare_mnp_dimensionless_cutoff_full_by_channel"],
        carrier_dimensionless + 12.0,
        rtol=2.0e-13,
        atol=0.0,
    ):
        raise ValueError("Bare-MNP dimensionless spectral cutoffs are inconsistent.")
    for orientation in np.unique(orientations.astype(str)):
        indices = np.flatnonzero(orientations.astype(str) == orientation)
        for name in (
            "bare_mnp_qs_work_loss_by_channel_cm2",
            "bare_mnp_energy_transfer_by_channel_cm2",
        ):
            values = bare_by_channel[name][indices]
            if not np.allclose(values, values[0], rtol=2.0e-13, atol=0.0):
                raise ValueError(
                    f"Bare-MNP reference differs within orientation {orientation!r}."
                )

    tail_converged = np.asarray(data["response_tail_converged"], dtype=bool)
    window_converged = np.asarray(data["observable_window_converged"], dtype=bool)
    bare_integration_converged = np.asarray(
        data["bare_mnp_energy_integration_converged"], dtype=bool
    )
    if tail_converged.shape != expected_shape:
        raise ValueError(
            "response_tail_converged has shape "
            f"{tail_converged.shape}; expected {expected_shape}."
        )
    if window_converged.shape != expected_shape:
        raise ValueError(
            "observable_window_converged has shape "
            f"{window_converged.shape}; expected {expected_shape}."
        )
    if bare_integration_converged.shape != expected_shape:
        raise ValueError(
            "bare_mnp_energy_integration_converged has shape "
            f"{bare_integration_converged.shape}; expected {expected_shape}."
        )
    window_changes = []
    for name in (
        "sigma_spectral_half_window_relative_change",
        "sigma_energy_half_window_relative_change",
    ):
        values = np.asarray(data[name], dtype=float)
        if values.shape != expected_shape:
            raise ValueError(
                f"{name} has shape {values.shape}; expected {expected_shape}."
            )
        if np.any(values < 0.0):
            raise ValueError(f"{name} must be non-negative.")
        window_changes.append(values)

    bare_cutoff_change = np.asarray(
        data["sigma_bare_mnp_energy_cutoff_relative_change"], dtype=float
    )
    bare_quadrature_error = np.asarray(
        data["sigma_bare_mnp_energy_quadrature_relative_error"], dtype=float
    )
    for name, values in (
        ("sigma_bare_mnp_energy_cutoff_relative_change", bare_cutoff_change),
        ("sigma_bare_mnp_energy_quadrature_relative_error", bare_quadrature_error),
    ):
        if values.shape != expected_shape:
            raise ValueError(f"{name} has shape {values.shape}; expected {expected_shape}.")
        if np.any(values < 0.0):
            raise ValueError(f"{name} must be non-negative.")

    boolean_diagnostics = {
        name: np.asarray(data[name], dtype=bool)
        for name in (
            "solver_success",
            "t_final_reached",
            "state_is_finite",
            "work_nonnegative_within_tolerance",
            "bare_mnp_work_nonnegative_within_tolerance",
        )
    }
    for name, values in boolean_diagnostics.items():
        if values.shape != expected_shape:
            raise ValueError(f"{name} has shape {values.shape}; expected {expected_shape}.")

    solver_status = np.asarray(data["solver_status"])
    if solver_status.shape != expected_shape or solver_status.dtype.kind not in "iu":
        raise ValueError("solver_status must be an integer array on the result grid.")
    if np.any(solver_status != 0):
        raise ValueError("solver_status must be zero for every successful propagation.")

    scalar_diagnostics = {
        name: np.asarray(data[name], dtype=float)
        for name in (
            "response_tail_ratio",
            "min_density_eigenvalue",
            "boundary_envelope_fraction",
            "pulse_spectral_leakage",
            "qd_source_spectral_leakage",
            "mnp_dipole_spectral_leakage",
            "mnp_drive_spectral_leakage",
            "mnp_field_spectral_leakage",
        )
    }
    for name, values in scalar_diagnostics.items():
        if values.shape != expected_shape:
            raise ValueError(f"{name} has shape {values.shape}; expected {expected_shape}.")
    if np.any(scalar_diagnostics["response_tail_ratio"] < 0.0):
        raise ValueError("response_tail_ratio must be non-negative.")
    boundary_envelope = scalar_diagnostics["boundary_envelope_fraction"]
    if np.any(~np.isfinite(boundary_envelope)) or np.any(
        (boundary_envelope < 0.0) | (boundary_envelope > 1.0e-6)
    ):
        raise ValueError(
            "boundary_envelope_fraction must lie in [0, 1e-6] for an untruncated pulse."
        )
    for name, values in scalar_diagnostics.items():
        if name.endswith("spectral_leakage") and np.any(
            (values < 0.0) | (values > 1.0)
        ):
            raise ValueError(f"{name} must lie in [0, 1].")

    solver_metadata = metadata.get("solver", {})
    quality_metadata = metadata.get("quality_policies", {})
    inputs_metadata = metadata.get("inputs", {})
    pulse_metadata = metadata.get("pulse_definition", {})
    initial_metadata = metadata.get("initial_conditions", {})
    qd_initial = initial_metadata.get("qd_bloch", {})
    if not (
        isinstance(pulse_metadata.get("real_field_formula"), str)
        and pulse_metadata.get("envelope_center_fs") == 0.0
        and pulse_metadata.get("carrier_phase_rad") == 0.0
        and qd_initial == {"W": -1.0, "Q": 0.0, "P": 0.0, "rho_ee": 0.0}
        and initial_metadata.get("all_mnp_material_ADE_coordinates_q_k") == 0.0
        and initial_metadata.get("all_mnp_material_ADE_velocities_dq_k_dt")
        == 0.0
        and initial_metadata.get("external_work_accumulator_au") == 0.0
    ):
        raise ValueError("Pulse definition or initial-condition metadata is incomplete.")
    try:
        tail_tolerance = float(solver_metadata["tail_ratio_tolerance"])
        window_tolerance = float(
            solver_metadata["max_observable_window_relative_change"]
        )
        positivity_tolerance = float(solver_metadata["positivity_tolerance"])
        spectral_tolerance = float(solver_metadata["max_spectral_leakage"])
    except (KeyError, TypeError, ValueError):
        tail_tolerance = window_tolerance = float("nan")
        positivity_tolerance = spectral_tolerance = float("nan")

    numerical_failures: list[str] = []
    if not (
        np.isfinite(grid_error_tolerance)
        and grid_error_tolerance > 0.0
        and np.isfinite(grid_area_step_tolerance)
        and grid_area_step_tolerance > 0.0
        and np.all(saved_grid_by_observable)
        and np.all(saved_grid_by_channel)
        and bool(grid_metadata.get("accepted"))
    ):
        numerical_failures.append("fluence-grid resolution")
    for name, values in boolean_diagnostics.items():
        if not np.all(values):
            numerical_failures.append(name)
    if not (
        np.isfinite(tail_tolerance)
        and tail_tolerance > 0.0
        and np.all(np.isfinite(scalar_diagnostics["response_tail_ratio"]))
        and np.all(scalar_diagnostics["response_tail_ratio"] <= tail_tolerance)
    ):
        numerical_failures.append("response-tail ratio")
    if not (
        np.isfinite(window_tolerance)
        and window_tolerance > 0.0
        and all(np.all(np.isfinite(values)) for values in window_changes)
        and all(np.all(values <= window_tolerance) for values in window_changes)
        and np.all(np.isfinite(bare_cutoff_change))
        and np.all(np.isfinite(bare_quadrature_error))
        and np.all(bare_cutoff_change <= window_tolerance)
        and np.all(bare_quadrature_error <= window_tolerance)
    ):
        numerical_failures.append("work-observable integration diagnostics")
    if not (
        np.isfinite(positivity_tolerance)
        and positivity_tolerance >= 0.0
        and np.all(np.isfinite(scalar_diagnostics["min_density_eigenvalue"]))
        and np.all(
            scalar_diagnostics["min_density_eigenvalue"] >= -positivity_tolerance
        )
    ):
        numerical_failures.append("density-matrix positivity")
    spectral_values = [
        values
        for name, values in scalar_diagnostics.items()
        if name.endswith("spectral_leakage")
    ]
    if not (
        np.isfinite(spectral_tolerance)
        and 0.0 <= spectral_tolerance < 1.0
        and all(np.all(np.isfinite(values)) for values in spectral_values)
        and all(np.all(values <= spectral_tolerance) for values in spectral_values)
    ):
        numerical_failures.append("material-fit spectral coverage")

    channels_metadata = metadata.get("channels")
    if not isinstance(channels_metadata, list) or len(channels_metadata) != channel_keys.size:
        raise ValueError("metadata.channels does not match the saved channel array.")
    material_fit_failures: list[str] = []
    try:
        max_bright_rms = inputs_metadata["max_bright_fit_normalized_rms"]
        max_bright_pointwise = inputs_metadata[
            "max_bright_fit_pointwise_relative_error"
        ]
        max_bright_rms = None if max_bright_rms is None else float(max_bright_rms)
        max_bright_pointwise = (
            None if max_bright_pointwise is None else float(max_bright_pointwise)
        )
    except (KeyError, TypeError, ValueError):
        max_bright_rms = max_bright_pointwise = float("nan")
    for index, channel in enumerate(channels_metadata):
        if not isinstance(channel, dict) or channel.get("key") != str(channel_keys[index]):
            raise ValueError("metadata.channels order/identity is inconsistent.")
        full_qs = channel.get("full_qs", {})
        material_fit = channel.get("material_fit", {})
        bare_metadata = channel.get("bare_mnp_pulse_work", {})
        bare_metadata_float_mapping = {
            "sigma_energy_transfer_cm2": "bare_mnp_energy_transfer_by_channel_cm2",
            "sigma_energy_cutoff_check_cm2": (
                "bare_mnp_energy_cutoff_check_by_channel_cm2"
            ),
            "cutoff_relative_change": (
                "bare_mnp_energy_cutoff_relative_change_by_channel"
            ),
            "quadrature_absolute_error_cm2": (
                "bare_mnp_energy_quadrature_absolute_error_by_channel_cm2"
            ),
            "quadrature_relative_error": (
                "bare_mnp_energy_quadrature_relative_error_by_channel"
            ),
            "work_from_incident_field_j": (
                "bare_mnp_work_from_incident_field_by_channel_j"
            ),
            "work_cutoff_check_j": "bare_mnp_work_cutoff_check_by_channel_j",
            "work_passivity_tolerance_j": (
                "bare_mnp_work_passivity_tolerance_by_channel_j"
            ),
            "reference_fluence_j_cm2": (
                "bare_mnp_reference_fluence_by_channel_j_cm2"
            ),
            "dimensionless_carrier_frequency": (
                "bare_mnp_dimensionless_carrier_frequency_by_channel"
            ),
            "dimensionless_cutoff_check": (
                "bare_mnp_dimensionless_cutoff_check_by_channel"
            ),
            "dimensionless_cutoff_full": (
                "bare_mnp_dimensionless_cutoff_full_by_channel"
            ),
        }
        for metadata_name, array_name in bare_metadata_float_mapping.items():
            try:
                metadata_value = float(bare_metadata[metadata_name])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Missing bare-MNP metadata field {metadata_name!r}."
                ) from exc
            if not np.isclose(
                metadata_value,
                bare_by_channel[array_name][index],
                rtol=2.0e-13,
                atol=0.0,
            ):
                raise ValueError(
                    f"Bare-MNP metadata {metadata_name!r} disagrees with {array_name}."
                )
        if bare_metadata.get("integration_converged") is not bool(
            bare_integration_by_channel[index]
        ) or bare_metadata.get("work_nonnegative_within_tolerance") is not bool(
            bare_nonnegative_by_channel[index]
        ):
            raise ValueError("Bare-MNP boolean metadata disagrees with saved arrays.")
        if not (
            full_qs.get("spatial_convergence_accepted") is True
            and full_qs.get("linearized_ground_state_stable") is True
            and (
                full_qs.get("dark_reduction") is None
                or full_qs.get("dark_reduction", {}).get("accepted") is True
            )
        ):
            numerical_failures.append(f"full-QS model certificate ({channel_keys[index]})")
        if full_qs.get("modal_fit_passive_on_audit_grid") is not True:
            numerical_failures.append(
                f"full-QS modal passivity ({channel_keys[index]})"
            )
        if full_qs.get("modal_fit_accepted") is not True:
            material_fit_failures.append(str(channel_keys[index]))
        if not (
            material_fit.get("passive_on_fit_window") is True
            and material_fit.get(
                "nonnegative_imaginary_part_all_positive_frequencies"
            )
            is True
        ):
            numerical_failures.append(f"material passivity ({channel_keys[index]})")
        try:
            rms_alpha = float(material_fit["normalized_rms_alpha"])
            rms_inverse = float(material_fit["normalized_rms_inverse_alpha"])
            pointwise = float(material_fit["max_normalized_alpha_error"])
            fit_accepted = bool(
                (max_bright_rms is None or max(rms_alpha, rms_inverse) <= max_bright_rms)
                and (
                    max_bright_pointwise is None
                    or pointwise <= max_bright_pointwise
                )
            )
        except (KeyError, TypeError, ValueError):
            fit_accepted = False
        if not fit_accepted:
            material_fit_failures.append(str(channel_keys[index]))

    if not allow_unconverged:
        failures: list[str] = list(numerical_failures)
        if not np.all(tail_converged):
            failures.append("response tail")
        if not np.all(window_converged):
            failures.append("work-observable integration window")
        if not np.all(bare_integration_converged):
            failures.append("bare-MNP pulse-work spectral integration")
        if any(np.any(~np.isfinite(values)) for values in window_changes):
            failures.append("finite window-convergence diagnostics")
        if failures:
            raise ValueError(
                "Refusing to plot an uncertified work-loss artifact; failed: "
                + ", ".join(failures)
                + ". Recalculate with automatic/longer tails, or use "
                "allow_unconverged=True only for diagnostic inspection."
            )
    if material_fit_failures and not allow_approximate_material_fit:
        material_fit_failures = list(dict.fromkeys(material_fit_failures))
        raise ValueError(
            "The material polarizability and/or derived modal response misses "
            "its configured accuracy gate "
            "for channel(s): "
            + ", ".join(material_fit_failures)
            + ". Use allow_approximate_material_fit=True only for an explicitly "
            "labelled one-pole/approximation control."
        )

    return {"path": artifact, "metadata": metadata, "arrays": data}


def create_work_loss_figure(
    artifact_data: dict[str, object],
    *,
    title: str | None = None,
    logarithmic_fluence: bool = True,
) -> Figure:
    """Create raw and bare-MNP-subtracted panels from loaded arrays."""

    arrays = artifact_data["arrays"]
    metadata = artifact_data["metadata"]
    channel_keys = np.asarray(arrays["channel_key"]).astype(str)
    labels = np.asarray(arrays["channel_label"]).astype(str)
    orientations = np.asarray(arrays["orientation"]).astype(str)
    fluence = np.asarray(arrays["fluence_j_cm2"], dtype=float)
    sigma = np.asarray(arrays["sigma_spectral_qs_work_loss_cm2"], dtype=float)
    bare = np.asarray(arrays["sigma_bare_mnp_qs_work_loss_cm2"], dtype=float)
    energy = np.asarray(arrays["sigma_energy_transfer_cm2"], dtype=float)
    bare_energy = np.asarray(
        arrays["sigma_bare_mnp_energy_transfer_cm2"], dtype=float
    )
    delta = np.asarray(
        arrays["delta_sigma_spectral_qs_work_loss_cm2"], dtype=float
    )
    delta_energy = np.asarray(arrays["delta_sigma_energy_transfer_cm2"], dtype=float)

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 7.0),
        sharex=True,
        constrained_layout=True,
    )
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    bare_orientation_drawn: set[str] = set()
    for index, (key, label, orientation) in enumerate(
        zip(channel_keys, labels, orientations)
    ):
        color = colors[index % len(colors)]
        axes[0].plot(
            fluence[index],
            sigma[index],
            marker="o",
            ms=4.0,
            lw=1.7,
            color=color,
            label=f"{label or key}: carrier",
        )
        axes[0].plot(
            fluence[index],
            energy[index],
            ls=":",
            marker="s",
            ms=3.5,
            lw=1.5,
            color=color,
            label=rf"{label or key}: pulse $W_{{\rm inc}}/\mathcal{{F}}$",
        )
        if orientation not in bare_orientation_drawn:
            axes[0].plot(
                fluence[index],
                bare[index],
                ls="--",
                lw=1.25,
                color=color,
                alpha=0.72,
                label=f"bare MNP ({orientation})",
            )
            bare_orientation_drawn.add(orientation)
            axes[0].plot(
                fluence[index],
                bare_energy[index],
                ls="-.",
                lw=1.25,
                color=color,
                alpha=0.72,
                label=rf"bare MNP pulse $W_{{\rm inc}}/\mathcal{{F}}$ ({orientation})",
            )
        axes[1].plot(
            fluence[index],
            delta[index],
            marker="o",
            ms=4.0,
            lw=1.7,
            color=color,
            label=label or key,
        )
        axes[1].plot(
            fluence[index],
            delta_energy[index],
            ls=":",
            marker="s",
            ms=3.5,
            lw=1.5,
            color=color,
            label=rf"{label or key}: $\Delta(W_{{\rm inc}}/\mathcal{{F}})$",
        )

    axes[0].set_ylabel(
        r"$\sigma_{\rm QS,work}$ or $W_{\rm inc}/\mathcal{F}$ (cm$^2$)"
    )
    axes[1].set_ylabel(r"Hybrid minus bare-MNP work (cm$^2$)")
    axes[1].set_xlabel(r"Incident fluence $\mathcal{F}$ (J cm$^{-2}$)")
    axes[1].axhline(0.0, color="0.3", lw=0.9, ls=":")
    if logarithmic_fluence:
        axes[0].set_xscale("log")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.24)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        axis.legend(frameon=False, fontsize=9)

    if title is None:
        inputs = metadata.get("inputs", {})
        energy = inputs.get("carrier_energy_eV")
        tau = inputs.get("pulse_tau_fs")
        gap = inputs.get("common_surface_gap_nm")
        if all(value is not None for value in (energy, tau, gap)):
            title = (
                rf"Full-QS pulse response: $E_*={float(energy):g}$ eV, "
                rf"$\tau={float(tau):g}$ fs, $g={float(gap):g}$ nm"
            )
    if title:
        figure.suptitle(title)
    return figure


def plot_work_loss_fluence(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    title: str | None = None,
    logarithmic_fluence: bool = True,
    dpi: int = 240,
    show: bool = False,
    allow_unconverged: bool = False,
    allow_approximate_material_fit: bool = False,
) -> Path:
    """Read one NPZ artifact and save its work-loss figure."""

    if isinstance(dpi, bool) or not isinstance(dpi, (int, np.integer)) or dpi < 50:
        raise ValueError("dpi must be an integer of at least 50.")
    loaded = load_work_loss_artifact(
        input_path,
        allow_unconverged=allow_unconverged,
        allow_approximate_material_fit=allow_approximate_material_fit,
    )
    source = Path(input_path)
    output = source.with_suffix(".png") if output_path is None else Path(output_path)
    if not output.suffix:
        raise ValueError("The output figure path must have an extension.")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure = create_work_loss_figure(
        loaded,
        title=title,
        logarithmic_fluence=logarithmic_fluence,
    )
    figure.savefig(output, dpi=int(dpi), bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)
    return output


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot sigma_QS,work(E*; fluence) from an existing calculation NPZ only."
        )
    )
    parser.add_argument("input", type=Path, help="NPZ made by the calculation script.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--title")
    parser.add_argument("--linear-fluence", action="store_true")
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--allow-unconverged",
        action="store_true",
        help="Plot a failed tail/window audit for diagnostic inspection only.",
    )
    parser.add_argument(
        "--allow-approximate-material-fit",
        action="store_true",
        help="Permit a deliberately inaccurate one-pole material-fit control.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = _parse_args(argv)
    output = plot_work_loss_fluence(
        args.input,
        args.output,
        title=args.title,
        logarithmic_fluence=not args.linear_fluence,
        dpi=args.dpi,
        show=args.show,
        allow_unconverged=args.allow_unconverged,
        allow_approximate_material_fit=args.allow_approximate_material_fit,
    )
    print(f"Saved figure: {output.resolve()}")
    return output


if __name__ == "__main__":
    main(sys.argv[1:])
