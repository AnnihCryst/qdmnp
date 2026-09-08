"""Plot a saved direct/N=1/N-mode material-dispersion NPZ artifact.

This module never imports the QD--MNP solver or SciPy.  Changing labels,
colours, limits, or resolution therefore does not repeat any material fit.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from article_observables.qd_mnp_material_modes_artifact import (
    load_npz_artifact,
    save_figure,
)


SCHEMA_NAME = "qd_mnp.material_dispersion_comparison"
REQUIRED_ARRAYS = {
    "energy_eV",
    "orientation_ids",
    "branch_ids",
    "fit_branch_ids",
    "mode_count_by_branch",
    "alpha_complex_au3",
    "alpha_real_au3",
    "alpha_imag_au3",
    "inverse_alpha_complex_au_minus3",
    "inverse_alpha_real_au_minus3",
    "inverse_alpha_imag_au_minus3",
    "alpha_normalized_abs_error",
    "inverse_alpha_normalized_abs_error",
    "nrms_alpha",
    "nrms_inverse_alpha",
    "max_normalized_alpha_error",
    "max_normalized_inverse_alpha_error",
    "lspr_peak_energy_eV",
    "lspr_fwhm_eV",
    "lspr_status",
    "fit_mode_count",
    "fit_alpha_inf_dimensionless",
    "fit_coefficient_valid",
    "fit_strengths_au2",
    "fit_omega_modes_au",
    "fit_omega_modes_eV",
    "fit_gamma_modes_au",
    "fit_gamma_modes_eV",
    "fit_passive_on_window",
    "fit_nonnegative_imag_all_positive",
    "fit_linear_stable",
    "fit_accuracy_gate_pass",
    "material_energy_eV",
    "material_n",
    "material_k",
}


def _as_strings(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{name} must be a one-dimensional string array.")
    if array.dtype.kind == "S":
        return np.char.decode(array, "utf-8")
    return array.astype(str)


def _all_finite(values: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(np.asarray(values))))


def load_material_dispersion_artifact(
    path: str | Path,
) -> dict[str, Any]:
    """Load and validate one plotting artifact without importing a solver."""

    payload, metadata = load_npz_artifact(
        path,
        schema_name=SCHEMA_NAME,
        required_arrays=REQUIRED_ARRAYS,
    )
    energy = np.asarray(payload["energy_eV"], dtype=float)
    orientations = _as_strings(payload["orientation_ids"], name="orientation_ids")
    branches = _as_strings(payload["branch_ids"], name="branch_ids")
    fit_branches = _as_strings(payload["fit_branch_ids"], name="fit_branch_ids")
    if tuple(orientations) != ("long", "trans"):
        raise ValueError("orientation_ids must be exactly ['long', 'trans'].")
    if tuple(branches) != ("direct", "one", "multi"):
        raise ValueError("branch_ids must be exactly ['direct', 'one', 'multi'].")
    if tuple(fit_branches) != ("one", "multi"):
        raise ValueError("fit_branch_ids must be exactly ['one', 'multi'].")
    if energy.ndim != 1 or energy.size < 5 or not _all_finite(energy):
        raise ValueError("energy_eV must be a finite one-dimensional grid of size >= 5.")
    if np.any(np.diff(energy) <= 0.0):
        raise ValueError("energy_eV must be strictly increasing.")

    spectrum_shape = (orientations.size, branches.size, energy.size)
    summary_shape = (orientations.size, branches.size)
    fit_shape = (orientations.size, fit_branches.size)
    spectrum_names = (
        "alpha_complex_au3",
        "alpha_real_au3",
        "alpha_imag_au3",
        "inverse_alpha_complex_au_minus3",
        "inverse_alpha_real_au_minus3",
        "inverse_alpha_imag_au_minus3",
        "alpha_normalized_abs_error",
        "inverse_alpha_normalized_abs_error",
    )
    for name in spectrum_names:
        if np.asarray(payload[name]).shape != spectrum_shape:
            raise ValueError(f"{name} must have shape {spectrum_shape}.")
        if not _all_finite(payload[name]):
            raise ValueError(f"{name} must contain only finite values.")
    for name in (
        "nrms_alpha",
        "nrms_inverse_alpha",
        "max_normalized_alpha_error",
        "max_normalized_inverse_alpha_error",
        "lspr_peak_energy_eV",
    ):
        if np.asarray(payload[name]).shape != summary_shape:
            raise ValueError(f"{name} must have shape {summary_shape}.")
        if not _all_finite(payload[name]):
            raise ValueError(f"{name} must contain only finite values.")
    if np.asarray(payload["lspr_fwhm_eV"]).shape != summary_shape:
        raise ValueError(f"lspr_fwhm_eV must have shape {summary_shape}.")
    fwhm = np.asarray(payload["lspr_fwhm_eV"], dtype=float)
    if np.any(np.isinf(fwhm)):
        raise ValueError("lspr_fwhm_eV may contain finite values or NaN, not infinity.")
    statuses = np.asarray(payload["lspr_status"])
    if statuses.shape != summary_shape or statuses.dtype.kind not in {"U", "S"}:
        raise ValueError("lspr_status must be a matching two-dimensional string array.")
    for name in (
        "fit_alpha_inf_dimensionless",
        "fit_mode_count",
        "fit_passive_on_window",
        "fit_nonnegative_imag_all_positive",
        "fit_linear_stable",
        "fit_accuracy_gate_pass",
    ):
        if np.asarray(payload[name]).shape != fit_shape:
            raise ValueError(f"{name} must have shape {fit_shape}.")

    alpha = np.asarray(payload["alpha_complex_au3"], dtype=complex)
    inverse = np.asarray(
        payload["inverse_alpha_complex_au_minus3"], dtype=complex
    )
    if not np.allclose(
        alpha.real,
        payload["alpha_real_au3"],
        rtol=2.0e-13,
        atol=1.0e-13,
    ) or not np.allclose(
        alpha.imag,
        payload["alpha_imag_au3"],
        rtol=2.0e-13,
        atol=1.0e-13,
    ):
        raise ValueError("Stored real/imaginary alpha arrays disagree with complex alpha.")
    if not np.allclose(
        inverse,
        1.0 / alpha,
        rtol=2.0e-12,
        atol=1.0e-15,
    ):
        raise ValueError("Stored inverse alpha is inconsistent with alpha.")
    if not np.allclose(
        inverse.real,
        payload["inverse_alpha_real_au_minus3"],
        rtol=2.0e-13,
        atol=1.0e-18,
    ) or not np.allclose(
        inverse.imag,
        payload["inverse_alpha_imag_au_minus3"],
        rtol=2.0e-13,
        atol=1.0e-18,
    ):
        raise ValueError(
            "Stored real/imaginary inverse-alpha arrays disagree with complex data."
        )

    for name in (
        "alpha_normalized_abs_error",
        "inverse_alpha_normalized_abs_error",
        "nrms_alpha",
        "nrms_inverse_alpha",
        "max_normalized_alpha_error",
        "max_normalized_inverse_alpha_error",
    ):
        if np.any(np.asarray(payload[name], dtype=float) < 0.0):
            raise ValueError(f"{name} must be non-negative.")
    if not np.allclose(payload["alpha_normalized_abs_error"][:, 0], 0.0):
        raise ValueError("The direct branch must have zero alpha error.")
    if not np.allclose(
        payload["inverse_alpha_normalized_abs_error"][:, 0], 0.0
    ):
        raise ValueError("The direct branch must have zero inverse-alpha error.")

    mode_count = np.asarray(payload["fit_mode_count"], dtype=int)
    coefficient_valid = np.asarray(payload["fit_coefficient_valid"], dtype=bool)
    if coefficient_valid.ndim != 3 or coefficient_valid.shape[:2] != fit_shape:
        raise ValueError("fit_coefficient_valid has an invalid shape.")
    coefficient_shape = coefficient_valid.shape
    for name in (
        "fit_strengths_au2",
        "fit_omega_modes_au",
        "fit_omega_modes_eV",
        "fit_gamma_modes_au",
        "fit_gamma_modes_eV",
    ):
        if np.asarray(payload[name]).shape != coefficient_shape:
            raise ValueError(f"{name} must have shape {coefficient_shape}.")
    if np.any(mode_count < 1) or np.any(mode_count > coefficient_shape[2]):
        raise ValueError("fit_mode_count is inconsistent with coefficient storage.")
    for orientation_index in range(orientations.size):
        for fit_index in range(fit_branches.size):
            expected = np.arange(coefficient_shape[2]) < mode_count[
                orientation_index, fit_index
            ]
            if not np.array_equal(
                coefficient_valid[orientation_index, fit_index], expected
            ):
                raise ValueError("fit_coefficient_valid is inconsistent with mode counts.")
            for name in (
                "fit_strengths_au2",
                "fit_omega_modes_au",
                "fit_omega_modes_eV",
                "fit_gamma_modes_au",
                "fit_gamma_modes_eV",
            ):
                values = np.asarray(payload[name])[orientation_index, fit_index]
                if not np.all(np.isfinite(values[expected])):
                    raise ValueError(f"Valid entries of {name} must be finite.")
            strengths = np.asarray(payload["fit_strengths_au2"])[
                orientation_index, fit_index, expected
            ]
            omega_au = np.asarray(payload["fit_omega_modes_au"])[
                orientation_index, fit_index, expected
            ]
            gamma_au = np.asarray(payload["fit_gamma_modes_au"])[
                orientation_index, fit_index, expected
            ]
            if np.any(strengths < 0.0) or np.any(omega_au <= 0.0) or np.any(
                gamma_au <= 0.0
            ):
                raise ValueError(
                    "Lorentz strengths must be non-negative and modal "
                    "frequencies/dampings positive."
                )

    if not np.all(np.asarray(payload["fit_passive_on_window"], dtype=bool)):
        raise ValueError("Artifact contains a fit that is not passive on its fit window.")
    if not np.all(
        np.asarray(payload["fit_nonnegative_imag_all_positive"], dtype=bool)
    ):
        raise ValueError("Artifact contains a fit that failed its global Im(alpha) check.")
    if not np.all(np.asarray(payload["fit_linear_stable"], dtype=bool)):
        raise ValueError("Artifact contains a linearly unstable Lorentz realization.")
    if not np.all(np.asarray(payload["fit_accuracy_gate_pass"], dtype=bool)[:, 1]):
        raise ValueError("The multi-mode branch did not pass its accuracy gate.")
    quality = metadata.get("quality_gates")
    if not isinstance(quality, dict) or quality.get(
        "multi_passed_for_all_orientations"
    ) is not True:
        raise ValueError("Metadata does not certify the multi-mode branch.")

    material_energy = np.asarray(payload["material_energy_eV"], dtype=float)
    material_n = np.asarray(payload["material_n"], dtype=float)
    material_k = np.asarray(payload["material_k"], dtype=float)
    if (
        material_energy.ndim != 1
        or material_n.shape != material_energy.shape
        or material_k.shape != material_energy.shape
        or material_energy.size < 2
        or not _all_finite(material_energy)
        or not _all_finite(material_n)
        or not _all_finite(material_k)
        or np.any(np.diff(material_energy) <= 0.0)
    ):
        raise ValueError("Stored material table is malformed.")

    result: dict[str, Any] = dict(payload)
    result["metadata"] = metadata
    result["orientation_ids"] = orientations
    result["branch_ids"] = branches
    result["fit_branch_ids"] = fit_branches
    return result


def _branch_labels(data: dict[str, Any]) -> list[str]:
    counts = np.asarray(data["mode_count_by_branch"], dtype=int)
    if counts.shape != (3,) or counts[0] != 0 or counts[1] != 1 or counts[2] < 2:
        raise ValueError("mode_count_by_branch must encode direct, N=1, and N>=2.")
    return ["direct material", r"Lorentz $N=1$", rf"Lorentz $N={counts[2]}$"]


def plot_material_dispersion_comparison(
    data: dict[str, Any],
    *,
    title: str | None = None,
    logarithmic_errors: bool = True,
) -> Figure:
    """Build a four-row direct/one/multi comparison on absolute scales."""

    energy = np.asarray(data["energy_eV"], dtype=float)
    alpha = np.asarray(data["alpha_complex_au3"], dtype=complex)
    alpha_error = np.asarray(data["alpha_normalized_abs_error"], dtype=float)
    inverse_error = np.asarray(
        data["inverse_alpha_normalized_abs_error"], dtype=float
    )
    peak_energy = np.asarray(data["lspr_peak_energy_eV"], dtype=float)
    fwhm = np.asarray(data["lspr_fwhm_eV"], dtype=float)
    nrms_alpha = np.asarray(data["nrms_alpha"], dtype=float)
    nrms_inverse = np.asarray(data["nrms_inverse_alpha"], dtype=float)
    labels = _branch_labels(data)

    colors = ("black", "#d95f02", "#1b9e77")
    styles = ("-", "--", "-.")
    widths = (2.2, 1.8, 1.9)
    figure, axes = plt.subplots(
        4,
        2,
        figsize=(12.5, 13.0),
        sharex=True,
        sharey="row",
        constrained_layout=True,
    )
    orientation_titles = (
        r"Longitudinal response, $E\parallel z$",
        r"Transverse response, $E\perp z$",
    )
    for orientation_index, orientation_title in enumerate(orientation_titles):
        axes[0, orientation_index].set_title(orientation_title)
        for branch_index, label in enumerate(labels):
            plot_options = {
                "color": colors[branch_index],
                "linestyle": styles[branch_index],
                "linewidth": widths[branch_index],
                "label": label,
            }
            axes[0, orientation_index].plot(
                energy, alpha[orientation_index, branch_index].real, **plot_options
            )
            axes[1, orientation_index].plot(
                energy, alpha[orientation_index, branch_index].imag, **plot_options
            )
            axes[1, orientation_index].scatter(
                [peak_energy[orientation_index, branch_index]],
                [
                    np.interp(
                        peak_energy[orientation_index, branch_index],
                        energy,
                        alpha[orientation_index, branch_index].imag,
                    )
                ],
                color=colors[branch_index],
                marker="o",
                s=20,
                zorder=4,
            )

        positive_errors = np.concatenate(
            (
                alpha_error[orientation_index, 1:].ravel(),
                inverse_error[orientation_index, 1:].ravel(),
            )
        )
        positive_errors = positive_errors[positive_errors > 0.0]
        floor = (
            max(float(np.min(positive_errors)) * 0.25, 1.0e-16)
            if positive_errors.size
            else 1.0e-16
        )
        for branch_index in (1, 2):
            axes[2, orientation_index].plot(
                energy,
                np.maximum(alpha_error[orientation_index, branch_index], floor),
                color=colors[branch_index],
                linestyle=styles[branch_index],
                linewidth=widths[branch_index],
                label=labels[branch_index],
            )
            axes[3, orientation_index].plot(
                energy,
                np.maximum(inverse_error[orientation_index, branch_index], floor),
                color=colors[branch_index],
                linestyle=styles[branch_index],
                linewidth=widths[branch_index],
                label=labels[branch_index],
            )
        if logarithmic_errors:
            axes[2, orientation_index].set_yscale("log")
            axes[3, orientation_index].set_yscale("log")

        annotation_lines = []
        for branch_index in (1, 2):
            width_text = (
                f"{fwhm[orientation_index, branch_index]:.3g} eV"
                if np.isfinite(fwhm[orientation_index, branch_index])
                else "not resolved"
            )
            annotation_lines.append(
                f"{labels[branch_index]}: "
                f"NRMS={nrms_alpha[orientation_index, branch_index]:.3g}, "
                f"FWHM={width_text}"
            )
        axes[2, orientation_index].text(
            0.02,
            0.04,
            "\n".join(annotation_lines),
            transform=axes[2, orientation_index].transAxes,
            fontsize=8.5,
            va="bottom",
            bbox={"facecolor": "white", "alpha": 0.76, "edgecolor": "0.75"},
        )
        inverse_lines = [
            f"{labels[index]}: NRMS={nrms_inverse[orientation_index, index]:.3g}"
            for index in (1, 2)
        ]
        axes[3, orientation_index].text(
            0.02,
            0.04,
            "\n".join(inverse_lines),
            transform=axes[3, orientation_index].transAxes,
            fontsize=8.5,
            va="bottom",
            bbox={"facecolor": "white", "alpha": 0.76, "edgecolor": "0.75"},
        )

    axes[0, 0].set_ylabel(r"$\mathrm{Re}\,\alpha$ ($a_0^3$)")
    axes[1, 0].set_ylabel(r"$\mathrm{Im}\,\alpha$ ($a_0^3$)")
    axes[2, 0].set_ylabel(r"$|\alpha-\alpha_{\rm dir}|/\max|\alpha_{\rm dir}|$")
    axes[3, 0].set_ylabel(
        r"$|\alpha^{-1}-\alpha_{\rm dir}^{-1}|/"
        r"\max|\alpha_{\rm dir}^{-1}|$"
    )
    for axis in axes[-1]:
        axis.set_xlabel(r"Photon energy $E=\hbar\omega$ (eV)")
    for axis in axes.flat:
        axis.grid(alpha=0.24, which="both")
        axis.set_xlim(float(energy[0]), float(energy[-1]))
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="outside upper center",
        ncol=3,
        frameon=False,
    )
    if title:
        figure.suptitle(title, y=1.035)
    return figure


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="NPZ produced by the calculation script.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--title")
    parser.add_argument("--linear-errors", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    data = load_material_dispersion_artifact(args.input)
    output = (
        args.output
        if args.output is not None
        else args.input.with_name(f"{args.input.stem}.png")
    )
    figure = plot_material_dispersion_comparison(
        data,
        title=args.title,
        logarithmic_errors=not args.linear_errors,
    )
    destination = save_figure(
        figure,
        output,
        dpi=args.dpi,
        show=args.show,
    )
    print(f"Saved material-dispersion figure to {destination.resolve()}")
    return destination


if __name__ == "__main__":
    main()
