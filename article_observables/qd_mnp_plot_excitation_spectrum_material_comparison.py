"""Plot saved direct/one-/multi-oscillator FQS QD-excitation spectra.

This program performs no material fitting and imports no QD--MNP solver.  It
loads one self-contained NPZ artifact produced by the companion calculation
script, then draws absolute spectra and residuals relative to the direct
tabulated-material calculation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from article_observables.qd_mnp_material_modes_artifact import (
    load_npz_artifact,
    save_figure,
)


SCHEMA_NAME = "qd_mnp.material_excitation_spectrum_comparison"
REQUIRED_ARRAYS = (
    "material_model_id",
    "channel_id",
    "energy_eV",
    "surface_gap_nm",
    "isolated_excitation_spectrum_au6",
    "excitation_spectrum_au6",
    "excitation_spectrum_residual_normalized_to_direct_peak",
    "peak_energy_eV",
    "feature_status",
    "energy_step_over_isolated_fwhm",
    "fit_passive_on_fit_window",
    "fit_passive_for_all_positive_frequencies",
    "bright_stability_stable",
    "modal_fit_passive_on_audit_grid",
    "modal_fit_accepted",
    "spatial_convergence_accepted",
    "coupled_stability_stable",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--channels",
        nargs="+",
        help="Channel IDs to draw; by default every channel stored in the artifact.",
    )
    parser.add_argument("--energy-min-ev", type=float)
    parser.add_argument("--energy-max-ev", type=float)
    parser.add_argument(
        "--residual-scale",
        choices=("fraction", "percent"),
        default="percent",
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--title")
    parser.add_argument(
        "--allow-unconverged",
        action="store_true",
        help="Draw a watermarked diagnostic figure when saved certificates fail.",
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args(argv)


def _validate_payload(payload: dict[str, np.ndarray]) -> None:
    energy = np.asarray(payload["energy_eV"], dtype=float)
    models = np.asarray(payload["material_model_id"]).astype(str)
    channels = np.asarray(payload["channel_id"]).astype(str)
    spectra = np.asarray(payload["excitation_spectrum_au6"], dtype=float)
    residual = np.asarray(
        payload["excitation_spectrum_residual_normalized_to_direct_peak"],
        dtype=float,
    )
    isolated = np.asarray(payload["isolated_excitation_spectrum_au6"], dtype=float)
    expected = (models.size, channels.size, energy.size)
    if models.tolist() != ["direct", "one", "multi"]:
        raise ValueError("material_model_id must be ['direct', 'one', 'multi'].")
    if energy.ndim != 1 or energy.size < 5 or np.any(~np.isfinite(energy)):
        raise ValueError("energy_eV must be a finite one-dimensional grid.")
    if np.any(np.diff(energy) <= 0.0):
        raise ValueError("energy_eV must be strictly increasing.")
    if spectra.shape != expected or residual.shape != expected:
        raise ValueError(
            "Spectrum and residual arrays must have dimensions model x channel x energy."
        )
    if isolated.shape != energy.shape:
        raise ValueError("isolated_excitation_spectrum_au6 must match energy_eV.")
    if np.any(~np.isfinite(spectra)) or np.any(spectra < 0.0):
        raise ValueError("The saved excitation spectra must be finite and non-negative.")
    if np.any(~np.isfinite(isolated)) or np.any(isolated < 0.0):
        raise ValueError("The saved isolated-QD spectrum must be finite and non-negative.")
    if np.any(~np.isfinite(residual)):
        raise ValueError("The saved normalized residuals must be finite.")


def _diagnostic_failures(
    payload: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> list[str]:
    """Return failed saved certificates, allowing only N=1 accuracy misses."""

    channel_count = np.asarray(payload["channel_id"]).size
    expected = (2, channel_count)
    failures: list[str] = []
    for key in (
        "fit_passive_on_fit_window",
        "fit_passive_for_all_positive_frequencies",
        "bright_stability_stable",
        "modal_fit_passive_on_audit_grid",
        "spatial_convergence_accepted",
        "coupled_stability_stable",
    ):
        values = np.asarray(payload[key])
        if values.shape != expected or values.dtype.kind != "b" or not np.all(values):
            failures.append(key)

    # A deliberately coarse N=1 fit may fail only its accuracy certificate.
    # The production multi-mode realization must pass the complete transformed
    # modal gate.
    modal_accepted = np.asarray(payload["modal_fit_accepted"])
    if (
        modal_accepted.shape != expected
        or modal_accepted.dtype.kind != "b"
        or not np.all(modal_accepted[1])
    ):
        failures.append("multi-mode modal accuracy")

    ratio = np.asarray(payload["energy_step_over_isolated_fwhm"], dtype=float)
    resolved = metadata.get("resolved_arguments", {})
    try:
        resolution_limit = float(resolved["max_energy_step_over_isolated_fwhm"])
    except (KeyError, TypeError, ValueError):
        resolution_limit = float("nan")
    if not (
        ratio.shape == ()
        and np.isfinite(float(ratio))
        and np.isfinite(resolution_limit)
        and resolution_limit > 0.0
        and float(ratio) <= resolution_limit
    ):
        failures.append("spectral-grid resolution")
    return list(dict.fromkeys(failures))


def _channel_labels(metadata: dict[str, Any], ids: np.ndarray) -> dict[str, str]:
    labels = {str(channel): str(channel) for channel in ids}
    for document in metadata.get("channels", []):
        channel_id = str(document.get("channel_id", ""))
        if channel_id in labels:
            labels[channel_id] = str(document.get("label", channel_id))
    return labels


def plot_artifact(
    input_path: str | Path,
    output_path: str | Path,
    *,
    channels: list[str] | tuple[str, ...] | None = None,
    energy_min_ev: float | None = None,
    energy_max_ev: float | None = None,
    residual_scale: str = "percent",
    dpi: int = 220,
    title: str | None = None,
    allow_unconverged: bool = False,
    show: bool = False,
) -> Path:
    payload, metadata = load_npz_artifact(
        input_path,
        schema_name=SCHEMA_NAME,
        required_arrays=REQUIRED_ARRAYS,
    )
    _validate_payload(payload)
    failures = _diagnostic_failures(payload, metadata)
    if failures and not allow_unconverged:
        raise ValueError(
            "Refusing to plot an uncertified material-spectrum comparison; failed: "
            + ", ".join(failures)
            + ". Use --allow-unconverged only for a watermarked diagnostic plot."
        )
    energy = np.asarray(payload["energy_eV"], dtype=float)
    channel_ids = np.asarray(payload["channel_id"]).astype(str)
    spectra = np.asarray(payload["excitation_spectrum_au6"], dtype=float)
    isolated = np.asarray(payload["isolated_excitation_spectrum_au6"], dtype=float)
    residual = np.asarray(
        payload["excitation_spectrum_residual_normalized_to_direct_peak"], dtype=float
    )
    peak_energy = np.asarray(payload["peak_energy_eV"], dtype=float)
    status = np.asarray(payload["feature_status"]).astype(str)

    requested = channel_ids.tolist() if channels is None else list(channels)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("The selected channel list must be non-empty and unique.")
    unknown = sorted(set(requested) - set(channel_ids.tolist()))
    if unknown:
        raise ValueError(f"Channels are absent from the artifact: {unknown}.")
    indices = [int(np.flatnonzero(channel_ids == channel)[0]) for channel in requested]

    lower = float(energy[0]) if energy_min_ev is None else float(energy_min_ev)
    upper = float(energy[-1]) if energy_max_ev is None else float(energy_max_ev)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("The plotting energy interval must have finite increasing bounds.")
    mask = (energy >= lower) & (energy <= upper)
    if np.count_nonzero(mask) < 2:
        raise ValueError("The requested plotting interval contains fewer than two points.")
    if residual_scale not in {"fraction", "percent"}:
        raise ValueError("residual_scale must be 'fraction' or 'percent'.")
    if dpi <= 0:
        raise ValueError("dpi must be positive.")

    labels = _channel_labels(metadata, channel_ids)
    styles = (
        ("direct", "#111111", "-", 2.4),
        ("one oscillator", "#d95f02", "--", 2.1),
        (
            f"{int(metadata.get('multi_fit_mode_count', 0)) or 'multi'} oscillators",
            "#1b9e77",
            "-.",
            2.1,
        ),
    )
    columns = len(indices)
    figure, axes = plt.subplots(
        2,
        columns,
        figsize=(5.4 * columns, 7.2),
        sharex="col",
        squeeze=False,
        gridspec_kw={"height_ratios": (1.6, 1.0)},
    )
    gap = float(np.asarray(payload["surface_gap_nm"]).item())
    residual_factor = 100.0 if residual_scale == "percent" else 1.0

    for column, channel_index in enumerate(indices):
        channel_id = str(channel_ids[channel_index])
        spectrum_axis = axes[0, column]
        residual_axis = axes[1, column]
        spectrum_axis.plot(
            energy[mask],
            isolated[mask],
            color="#777777",
            linestyle=":",
            linewidth=1.8,
            label="isolated QD",
        )
        for model_index, (label, color, linestyle, linewidth) in enumerate(styles):
            spectrum_axis.plot(
                energy[mask],
                spectra[model_index, channel_index, mask],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                label=label,
            )
            if status[model_index, channel_index] == "ok":
                peak = peak_energy[model_index, channel_index]
                spectrum_axis.axvline(
                    peak,
                    color=color,
                    linewidth=0.7,
                    alpha=0.35,
                )
        for model_index, (label, color, linestyle, linewidth) in enumerate(styles[1:], 1):
            residual_axis.plot(
                energy[mask],
                residual_factor * residual[model_index, channel_index, mask],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                label=f"{label} - direct",
            )
        residual_axis.axhline(0.0, color="#555555", linewidth=0.8)
        spectrum_axis.set_title(f"{labels[channel_id]}\n$g={gap:g}$ nm")
        spectrum_axis.grid(alpha=0.22)
        residual_axis.grid(alpha=0.22)
        residual_axis.set_xlabel("Photon energy $E$ (eV)")
        if column == 0:
            spectrum_axis.set_ylabel(
                r"$S(E)=|p_{\mathrm{QD}}/E_{\mathrm{inc}}|^2$ ($a_0^6$)"
            )
            residual_axis.set_ylabel(
                r"$(S_{\mathrm{fit}}-S_{\mathrm{direct}})/\max S_{\mathrm{direct}}$"
                + (" (%)" if residual_scale == "percent" else "")
            )
        spectrum_axis.legend(frameon=False, fontsize=9)
        residual_axis.legend(frameon=False, fontsize=9)

    figure.suptitle(
        title
        or "QD excitation: tabulated material vs Lorentz representations",
        y=1.01,
    )
    if failures:
        figure.text(
            0.5,
            0.5,
            "DIAGNOSTIC - UNCONVERGED",
            ha="center",
            va="center",
            rotation=25,
            fontsize=28,
            color="crimson",
            alpha=0.18,
        )
    figure.tight_layout()
    return save_figure(figure, output_path, dpi=dpi, show=show)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    output = plot_artifact(
        args.input,
        args.output,
        channels=args.channels,
        energy_min_ev=args.energy_min_ev,
        energy_max_ev=args.energy_max_ev,
        residual_scale=args.residual_scale,
        dpi=args.dpi,
        title=args.title,
        allow_unconverged=args.allow_unconverged,
        show=args.show,
    )
    print(f"Saved figure: {output}")
    return output


if __name__ == "__main__":
    main()
