"""Plot saved one- versus multi-mode transient work-loss spectra.

The companion calculator performs every time-domain propagation and stores a
pickle-free NPZ artifact.  This module deliberately imports no QD--MNP solver:
it validates the saved arrays and changes figure presentation only.

The plotted quantity is the operational local-QS work-loss estimate
``k Im(alpha_eff) / epsilon_0`` used as a Shah-style optical-response proxy.  It
must not be relabelled as a separately calculated metal-heating cross section.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from article_observables.qd_mnp_material_modes_artifact import (
    load_npz_artifact,
    save_figure,
)


SCHEMA_NAME = "qd_mnp.material_work_loss_spectrum_comparison"
SCHEMA_VERSION = 1
BRANCH_IDS = ("one", "multi")

REQUIRED_ARRAYS = {
    "branch_id",
    "branch_material_mode_count",
    "channel_key",
    "channel_label",
    "selected_fluence_j_cm2",
    "selected_fluence_label",
    "carrier_energy_eV",
    "energy_eV",
    "sigma_qs_work_cm2",
    "bare_mnp_sigma_qs_work_cm2",
    "delta_sigma_qs_work_cm2",
    "spectrum_support_mask",
    "solver_success",
    "t_final_reached",
    "state_is_finite",
    "response_tail_converged",
    "spectrum_window_converged",
    "work_nonnegative_within_tolerance",
    "density_matrix_positive",
    "incident_ft_converged",
    "spatial_convergence_accepted",
    "modal_fit_accepted",
    "fit_passive",
    "bright_stable",
    "coupled_stable",
    "energy_grid_converged",
}

_SOLVE_CERTIFICATES = (
    "solver_success",
    "t_final_reached",
    "state_is_finite",
    "response_tail_converged",
    "spectrum_window_converged",
    "work_nonnegative_within_tolerance",
    "density_matrix_positive",
    "incident_ft_converged",
    "energy_grid_converged",
)

_MODEL_CERTIFICATES = (
    "spatial_convergence_accepted",
    "fit_passive",
    "bright_stable",
    "coupled_stable",
)


def _require_exact_boolean(
    payload: dict[str, np.ndarray],
    name: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    values = np.asarray(payload[name])
    if values.shape != shape:
        raise ValueError(f"{name} has shape {values.shape}; expected {shape}.")
    if values.dtype != np.dtype(bool):
        raise ValueError(f"{name} must use the exact NumPy boolean dtype.")
    return values


def _validate_payload(
    payload: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> tuple[int, int, int, int]:
    """Validate array identities, dimensions and algebraic invariants."""

    branch_id = np.asarray(payload["branch_id"])
    if branch_id.ndim != 1 or tuple(branch_id.astype(str)) != BRANCH_IDS:
        raise ValueError("branch_id must be exactly ['one', 'multi'] in that order.")
    branch_count = branch_id.size

    mode_count = np.asarray(payload["branch_material_mode_count"])
    if mode_count.shape != (branch_count,) or mode_count.dtype.kind not in "iu":
        raise ValueError(
            "branch_material_mode_count must be an integer vector matching branch_id."
        )
    if int(mode_count[0]) != 1 or int(mode_count[1]) < 2:
        raise ValueError(
            "The one branch must have one mode and the multi branch at least two."
        )

    channel_key = np.asarray(payload["channel_key"])
    channel_label = np.asarray(payload["channel_label"])
    if channel_key.ndim != 1 or channel_key.size == 0:
        raise ValueError("channel_key must be a non-empty one-dimensional array.")
    if channel_label.shape != channel_key.shape:
        raise ValueError("channel_label must match channel_key.")
    if np.unique(channel_key.astype(str)).size != channel_key.size:
        raise ValueError("channel_key values must be unique.")
    channel_count = channel_key.size

    fluence = np.asarray(payload["selected_fluence_j_cm2"], dtype=float)
    if (
        fluence.ndim != 1
        or fluence.size == 0
        or np.any(~np.isfinite(fluence))
        or np.any(fluence <= 0.0)
        or np.any(np.diff(fluence) <= 0.0)
    ):
        raise ValueError(
            "selected_fluence_j_cm2 must be finite, positive and strictly increasing."
        )
    fluence_count = fluence.size
    fluence_label = np.asarray(payload["selected_fluence_label"])
    if fluence_label.shape != fluence.shape or fluence_label.dtype.kind not in "US":
        raise ValueError(
            "selected_fluence_label must be a string vector matching the fluence grid."
        )

    energy = np.asarray(payload["energy_eV"], dtype=float)
    if (
        energy.ndim != 1
        or energy.size < 2
        or np.any(~np.isfinite(energy))
        or np.any(energy <= 0.0)
        or np.any(np.diff(energy) <= 0.0)
    ):
        raise ValueError("energy_eV must be finite, positive and strictly increasing.")
    energy_count = energy.size

    carrier = np.asarray(payload["carrier_energy_eV"], dtype=float)
    if carrier.shape != () or not np.isfinite(float(carrier)) or float(carrier) <= 0.0:
        raise ValueError("carrier_energy_eV must be one finite positive scalar.")
    if not energy[0] <= float(carrier) <= energy[-1]:
        raise ValueError("The pulse carrier must lie inside the saved energy interval.")

    spectrum_shape = (branch_count, channel_count, fluence_count, energy_count)
    bare_shape = (branch_count, channel_count, energy_count)
    sigma = np.asarray(payload["sigma_qs_work_cm2"], dtype=float)
    bare = np.asarray(payload["bare_mnp_sigma_qs_work_cm2"], dtype=float)
    delta = np.asarray(payload["delta_sigma_qs_work_cm2"], dtype=float)
    if sigma.shape != spectrum_shape:
        raise ValueError(
            f"sigma_qs_work_cm2 has shape {sigma.shape}; expected {spectrum_shape}."
        )
    if bare.shape != bare_shape:
        raise ValueError(
            "bare_mnp_sigma_qs_work_cm2 has shape "
            f"{bare.shape}; expected {bare_shape}."
        )
    if delta.shape != spectrum_shape:
        raise ValueError(
            f"delta_sigma_qs_work_cm2 has shape {delta.shape}; "
            f"expected {spectrum_shape}."
        )
    if np.any(np.isinf(sigma)) or np.any(np.isinf(bare)) or np.any(np.isinf(delta)):
        raise ValueError("Saved work-loss spectra must not contain infinities.")
    if np.any(~np.isfinite(bare)):
        raise ValueError("The analytic bare-MNP spectra must be finite everywhere.")

    support = _require_exact_boolean(
        payload,
        "spectrum_support_mask",
        (fluence_count, energy_count),
    )
    nearest_carrier_index = int(np.argmin(np.abs(energy - float(carrier))))
    for fluence_index, supported in enumerate(support):
        indices = np.flatnonzero(supported)
        if indices.size < 2:
            raise ValueError(
                "Every selected fluence must have at least two supported spectral points."
            )
        if not np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
            raise ValueError("Each spectrum_support_mask row must be one contiguous band.")
        if not bool(supported[nearest_carrier_index]):
            raise ValueError("The spectral support must contain the pulse carrier.")
        selected_sigma = sigma[:, :, fluence_index, supported]
        selected_delta = delta[:, :, fluence_index, supported]
        if np.any(~np.isfinite(selected_sigma)) or np.any(~np.isfinite(selected_delta)):
            raise ValueError("Supported work-loss samples must all be finite.")

    expected_delta = sigma - bare[:, :, None, :]
    if not np.allclose(
        delta,
        expected_delta,
        rtol=2.0e-12,
        atol=1.0e-30,
        equal_nan=True,
    ):
        raise ValueError(
            "delta_sigma_qs_work_cm2 is inconsistent with hybrid minus bare MNP."
        )

    solve_shape = (branch_count, channel_count, fluence_count)
    model_shape = (branch_count, channel_count)
    for name in _SOLVE_CERTIFICATES:
        _require_exact_boolean(payload, name, solve_shape)
    for name in (*_MODEL_CERTIFICATES, "modal_fit_accepted"):
        _require_exact_boolean(payload, name, model_shape)

    metadata_mode_count = metadata.get("multi_fit_mode_count")
    if metadata_mode_count is not None:
        try:
            matches = int(metadata_mode_count) == int(mode_count[1])
        except (TypeError, ValueError):
            matches = False
        if not matches:
            raise ValueError(
                "metadata.multi_fit_mode_count disagrees with the saved branch count."
            )
    return branch_count, channel_count, fluence_count, energy_count


def _certificate_failures(payload: dict[str, np.ndarray]) -> list[str]:
    """Return failed numerical/physical certificates.

    A failed *accuracy* flag is allowed for the deliberately approximate N=1
    material branch.  Its passivity/stability flags remain mandatory, while
    the production multi-mode branch must pass modal accuracy as well.
    """

    failures: list[str] = []
    for name in (*_SOLVE_CERTIFICATES, *_MODEL_CERTIFICATES):
        values = np.asarray(payload[name], dtype=bool)
        if not np.all(values):
            failures.append(f"{name} ({int(np.count_nonzero(~values))} failed)")
    modal = np.asarray(payload["modal_fit_accepted"], dtype=bool)
    if not np.all(modal[1]):
        failures.append(
            "modal_fit_accepted for multi "
            f"({int(np.count_nonzero(~modal[1]))} failed)"
        )
    return failures


def load_work_loss_spectrum_artifact(
    path: str | Path,
    *,
    allow_unconverged: bool = False,
) -> dict[str, Any]:
    """Load and validate one transient-spectrum comparison artifact."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Work-loss spectrum artifact does not exist: {source}")
    payload, metadata = load_npz_artifact(
        source,
        schema_name=SCHEMA_NAME,
        required_arrays=REQUIRED_ARRAYS,
    )
    if int(metadata.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema version {metadata.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}."
        )
    _validate_payload(payload, metadata)
    failures = _certificate_failures(payload)
    if failures and not allow_unconverged:
        raise ValueError(
            "Refusing to plot an uncertified work-loss spectrum artifact; failed: "
            + ", ".join(failures)
            + ". Use --allow-unconverged only for a watermarked diagnostic plot."
        )
    return {
        "path": source,
        "metadata": metadata,
        "arrays": payload,
        "quality_failures": failures,
    }


def _selected_energy_mask(
    energy_eV: np.ndarray,
    lower_eV: float | None,
    upper_eV: float | None,
) -> np.ndarray:
    lower = float(energy_eV[0]) if lower_eV is None else float(lower_eV)
    upper = float(energy_eV[-1]) if upper_eV is None else float(upper_eV)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("The plotting energy bounds must be finite and increasing.")
    selected = (energy_eV >= lower) & (energy_eV <= upper)
    if np.count_nonzero(selected) < 2:
        raise ValueError("The requested energy interval contains fewer than two points.")
    return selected


def create_work_loss_spectrum_figure(
    artifact_data: dict[str, Any],
    *,
    channel: str | None = None,
    energy_min_eV: float | None = None,
    energy_max_eV: float | None = None,
    title: str | None = None,
) -> Figure:
    """Create a 2-by-fluence Shah-style work-loss spectrum figure."""

    payload = artifact_data["arrays"]
    metadata = artifact_data["metadata"]
    failures = list(artifact_data.get("quality_failures", []))
    branch_id = np.asarray(payload["branch_id"]).astype(str)
    mode_count = np.asarray(payload["branch_material_mode_count"], dtype=int)
    channel_keys = np.asarray(payload["channel_key"]).astype(str)
    channel_labels = np.asarray(payload["channel_label"]).astype(str)
    fluence = np.asarray(payload["selected_fluence_j_cm2"], dtype=float)
    fluence_labels = np.asarray(payload["selected_fluence_label"]).astype(str)
    carrier = float(np.asarray(payload["carrier_energy_eV"]))
    energy = np.asarray(payload["energy_eV"], dtype=float)
    sigma = np.asarray(payload["sigma_qs_work_cm2"], dtype=float)
    bare = np.asarray(payload["bare_mnp_sigma_qs_work_cm2"], dtype=float)
    delta = np.asarray(payload["delta_sigma_qs_work_cm2"], dtype=float)
    support = np.asarray(payload["spectrum_support_mask"], dtype=bool)

    selected_channel = channel_keys[0] if channel is None else str(channel)
    matches = np.flatnonzero(channel_keys == selected_channel)
    if matches.size != 1:
        raise ValueError(
            f"Channel {selected_channel!r} is absent from the artifact; available: "
            + ", ".join(channel_keys)
        )
    channel_index = int(matches[0])
    range_mask = _selected_energy_mask(energy, energy_min_eV, energy_max_eV)
    for fluence_index in range(fluence.size):
        if np.count_nonzero(range_mask & support[fluence_index]) < 2:
            raise ValueError(
                "The requested energy interval has fewer than two supported points "
                f"for fluence index {fluence_index}."
            )

    figure, axes = plt.subplots(
        2,
        fluence.size,
        figsize=(4.25 * fluence.size, 6.8),
        sharex="col",
        sharey="row",
        squeeze=False,
        constrained_layout=True,
    )
    branch_styles = (
        ("#d95f02", "--"),
        ("#1b9e77", "-"),
    )
    for fluence_index, fluence_value in enumerate(fluence):
        mask = range_mask & support[fluence_index]
        upper_axis = axes[0, fluence_index]
        lower_axis = axes[1, fluence_index]
        for branch_index, (color, linestyle) in enumerate(branch_styles):
            branch_label = (
                "one oscillator"
                if branch_id[branch_index] == "one"
                else f"{mode_count[branch_index]} oscillators"
            )
            upper_axis.plot(
                energy[mask],
                sigma[branch_index, channel_index, fluence_index, mask],
                color=color,
                linestyle=linestyle,
                linewidth=2.1,
                label=f"hybrid, {branch_label}",
            )
            upper_axis.plot(
                energy[mask],
                bare[branch_index, channel_index, mask],
                color="0.25" if branch_index else "0.55",
                linestyle=linestyle,
                linewidth=1.25,
                alpha=0.85,
                label=f"bare MNP, {branch_label}",
            )
            lower_axis.plot(
                energy[mask],
                delta[branch_index, channel_index, fluence_index, mask],
                color=color,
                linestyle=linestyle,
                linewidth=2.1,
                label=branch_label,
            )
        upper_axis.axvline(carrier, color="0.35", linestyle=":", linewidth=1.0)
        lower_axis.axvline(carrier, color="0.35", linestyle=":", linewidth=1.0)
        lower_axis.axhline(0.0, color="0.35", linestyle=":", linewidth=0.9)
        role = fluence_labels[fluence_index].strip()
        title_prefix = f"{role}: " if role else ""
        upper_axis.set_title(
            title_prefix + rf"$\mathcal{{F}}={fluence_value:.3g}$ J cm$^{{-2}}$"
        )
        lower_axis.set_xlabel("Photon energy (eV)")
        for axis in (upper_axis, lower_axis):
            axis.grid(True, alpha=0.24)
            axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))

    axes[0, 0].set_ylabel(r"$\sigma_{\rm QS,work}$ (cm$^2$)")
    axes[1, 0].set_ylabel(
        r"$\Delta\sigma_{\rm QS,work}$ (cm$^2$)"
    )
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[1, 0].legend(frameon=False, fontsize=8)

    if title is None:
        label = channel_labels[channel_index] or selected_channel
        title = (
            f"Transient full-QS work-loss spectra: {label}; "
            rf"$E_L={carrier:g}$ eV"
        )
    if title:
        figure.suptitle(title)

    if failures:
        figure.text(
            0.5,
            0.5,
            "UNCERTIFIED DIAGNOSTIC",
            ha="center",
            va="center",
            rotation=28.0,
            fontsize=24,
            color="#b2182b",
            alpha=0.22,
            weight="bold",
        )
        figure.text(
            0.5,
            0.012,
            "Failed certificates: " + "; ".join(failures),
            ha="center",
            va="bottom",
            fontsize=7,
            color="#8c1d2c",
        )
    return figure


def plot_work_loss_spectrum_material_comparison(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    channel: str | None = None,
    energy_min_eV: float | None = None,
    energy_max_eV: float | None = None,
    title: str | None = None,
    dpi: int = 240,
    show: bool = False,
    allow_unconverged: bool = False,
) -> Path:
    """Validate one NPZ artifact and save its comparison figure."""

    if isinstance(dpi, bool) or not isinstance(dpi, (int, np.integer)) or dpi < 50:
        raise ValueError("dpi must be an integer of at least 50.")
    loaded = load_work_loss_spectrum_artifact(
        input_path,
        allow_unconverged=allow_unconverged,
    )
    source = Path(input_path)
    destination = source.with_suffix(".png") if output_path is None else Path(output_path)
    if not destination.suffix:
        raise ValueError("The output figure path must have a file extension.")
    figure = create_work_loss_spectrum_figure(
        loaded,
        channel=channel,
        energy_min_eV=energy_min_eV,
        energy_max_eV=energy_max_eV,
        title=title,
    )
    return save_figure(figure, destination, dpi=int(dpi), show=show)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="NPZ made by the companion calculator.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--channel", help="One channel_key saved in the artifact.")
    parser.add_argument("--energy-min-ev", type=float)
    parser.add_argument("--energy-max-ev", type=float)
    parser.add_argument("--title")
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--allow-unconverged",
        action="store_true",
        help="Permit a failed certificate and add an explicit diagnostic watermark.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = _parse_args(argv)
    output = plot_work_loss_spectrum_material_comparison(
        args.input,
        args.output,
        channel=args.channel,
        energy_min_eV=args.energy_min_ev,
        energy_max_eV=args.energy_max_ev,
        title=args.title,
        dpi=args.dpi,
        show=args.show,
        allow_unconverged=args.allow_unconverged,
    )
    print(f"Saved figure: {output.resolve()}")
    return output


if __name__ == "__main__":
    main(sys.argv[1:])
