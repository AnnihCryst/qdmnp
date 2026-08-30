"""Reconstruct Yan et al. (2008), Fig. 2(a), with the full-QS backend.

The target is the complex exciton self-interaction energy ``G_N`` for a
point-like quantum dot next to a spherical gold nanoparticle, with spherical
multipoles retained through degree ``N``.  The calculation deliberately uses
``SpheroidGreenInteraction`` in its exact spherical limit.  Yan's Eq. (15) is
evaluated independently and overlaid as a normalization/mapping check.

Paper
-----
J.-Y. Yan, W. Zhang, S. Duan, X.-G. Zhao, and A. O. Govorov,
"Optical properties of coupled metal-semiconductor and metal-molecule
nanocrystal complexes: Role of multipole effects", Phys. Rev. B 77, 165301
(2008), DOI: 10.1103/PhysRevB.77.165301.

Parameter provenance
--------------------
Yan explicitly gives the Au radius (15 nm), centre distance (20 nm), QD
background permittivity (6), and Johnson--Christy Au data for Fig. 2(a).  The
host permittivity and absolute transition dipole are not restated there.  The
defaults below use eps_e=1 and mu=e*0.65 nm from Yan's cited dipole-model
reference, Zhang et al., Phys. Rev. Lett. 97, 146804 (2006).  Outputs mark
these two values as cross-paper completions, not as explicit Yan parameters.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from qd_mnp_rational_fit import (
    AU_ENERGY_EV,
    dipole_au_to_debye,
    make_params_with_overrides,
    nm_to_au,
    timestamped_run_dir,
    write_json,
)
from qd_mnp_spheroid_green import SpheroidGreenInteraction


ARTICLE_DOI = "10.1103/PhysRevB.77.165301"
ARTICLE_FIGURE = "2(a)"
SCHEMA_VERSION = 1
PLOT_FILENAME = "yan2008_fig2a_full_qs.png"
CSV_FILENAME = "yan2008_fig2a_curves.csv"
METADATA_FILENAME = "metadata.json"
DEFAULT_ARTICLE_PDF = (
    Path(__file__).resolve().parents[1] / "articles" / "yan2008.pdf"
)
PAPER_FIG2A_RASTER_ANCHORS = {
    # Read from the vector-like coloured curves after rendering the publisher
    # PDF page at 220 dpi and calibrating against the printed axes.  These are
    # deliberately low-precision visual anchors, not author-supplied raw data.
    "N10_real_peak": {
        "energy_eV": 2.335,
        "G_meV": 0.2075,
        "energy_uncertainty_eV": 0.015,
        "G_uncertainty_meV": 0.002,
    },
    "N10_imag_peak": {
        "energy_eV": 2.496,
        "G_meV": 0.0873,
        "energy_uncertainty_eV": 0.020,
        "G_uncertainty_meV": 0.0025,
    },
}
PAPER_PROFILE = {
    "sphere_radius_nm": 15.0,
    "centre_distance_nm": 20.0,
    "eps_host": 1.0,
    "eps_qd": 6.0,
    "transition_dipole_e_nm": 0.65,
}
PAPER_PEAK_SEARCH_WINDOW_EV = (2.2, 2.65)
PAPER_MAX_GRID_STEP_EV = 0.015
PAPER_PARAMETER_SOURCES = {
    "sphere_radius_nm": "Yan 2008, Fig. 2(a) caption/text",
    "centre_distance_nm": "Yan 2008, Fig. 2(a) caption/text",
    "eps_qd": "Yan 2008, Fig. 2(a) text",
    "eps_host": (
        "not restated by Yan for Fig. 2(a); completed with eps_e=1 "
        "from cited Zhang et al. (2006)"
    ),
    "transition_dipole_e_nm": (
        "not restated by Yan for Fig. 2(a); completed with mu=e*0.65 nm "
        "from cited Zhang et al. (2006)"
    ),
}


@dataclass(frozen=True)
class Yan2008Fig2AResult:
    """Calculated full-QS and independently evaluated Yan Eq. (15) curves."""

    energy_eV: np.ndarray
    degrees: np.ndarray
    g_full_qs_meV: np.ndarray
    g_yan_eq15_meV: np.ndarray
    epsilon_gold: np.ndarray
    local_field_factor: float
    bare_dipole_au: float

    def __post_init__(self) -> None:
        energy = np.asarray(self.energy_eV, dtype=float)
        degrees = np.asarray(self.degrees, dtype=int)
        core = np.asarray(self.g_full_qs_meV, dtype=complex)
        reference = np.asarray(self.g_yan_eq15_meV, dtype=complex)
        epsilon = np.asarray(self.epsilon_gold, dtype=complex)
        expected = (degrees.size, energy.size)
        if energy.ndim != 1 or energy.size < 2:
            raise ValueError("energy_eV must be a one-dimensional grid.")
        if degrees.ndim != 1 or degrees.size < 1:
            raise ValueError("degrees must be a non-empty one-dimensional array.")
        if core.shape != expected or reference.shape != expected:
            raise ValueError("G arrays must have shape (n_max, energy_points).")
        if epsilon.shape != energy.shape:
            raise ValueError("epsilon_gold must have the same shape as energy_eV.")
        if not (
            np.all(np.isfinite(energy))
            and np.all(np.isfinite(core))
            and np.all(np.isfinite(reference))
            and np.all(np.isfinite(epsilon))
        ):
            raise ValueError("The Fig. 2(a) result contains non-finite values.")
        for value in (energy, degrees, core, reference, epsilon):
            value.setflags(write=False)
        object.__setattr__(self, "energy_eV", energy)
        object.__setattr__(self, "degrees", degrees)
        object.__setattr__(self, "g_full_qs_meV", core)
        object.__setattr__(self, "g_yan_eq15_meV", reference)
        object.__setattr__(self, "epsilon_gold", epsilon)

    @property
    def absolute_difference_meV(self) -> np.ndarray:
        return np.abs(self.g_full_qs_meV - self.g_yan_eq15_meV)

    @property
    def max_absolute_difference_meV(self) -> float:
        return float(np.max(self.absolute_difference_meV))

    @property
    def max_relative_difference(self) -> float:
        scale = float(np.max(np.abs(self.g_yan_eq15_meV)))
        return self.max_absolute_difference_meV / max(scale, np.finfo(float).tiny)


def _validate_calculation_inputs(
    *,
    energy_window_eV: tuple[float, float],
    energy_points: int,
    n_max: int,
    sphere_radius_nm: float,
    centre_distance_nm: float,
    eps_host: float,
    eps_qd: float,
    transition_dipole_e_nm: float,
) -> None:
    if len(energy_window_eV) != 2 or not (
        np.isfinite(energy_window_eV[0])
        and np.isfinite(energy_window_eV[1])
        and 0.0 < energy_window_eV[0] < energy_window_eV[1]
    ):
        raise ValueError("energy_window_eV must satisfy 0 < min < max.")
    if (
        isinstance(energy_points, (bool, np.bool_))
        or not isinstance(energy_points, (int, np.integer))
        or energy_points < 2
    ):
        raise ValueError("energy_points must be an integer at least 2.")
    if (
        isinstance(n_max, (bool, np.bool_))
        or not isinstance(n_max, (int, np.integer))
        or n_max < 1
    ):
        raise ValueError("n_max must be an integer at least 1.")
    scalars = np.asarray(
        [
            sphere_radius_nm,
            centre_distance_nm,
            eps_host,
            eps_qd,
            transition_dipole_e_nm,
        ],
        dtype=float,
    )
    if np.any(~np.isfinite(scalars)) or np.any(scalars <= 0.0):
        raise ValueError("Geometry, permittivities and dipole must be positive.")
    if centre_distance_nm <= sphere_radius_nm:
        raise ValueError("The point QD centre must lie outside the Au sphere.")


def _yan_eq15_cumulative_g_au(
    *,
    epsilon_particle: np.ndarray,
    degrees: np.ndarray,
    radius_au: float,
    distance_au: float,
    eps_host: float,
    eps_qd: float,
    bare_dipole_au: float,
) -> np.ndarray:
    """Evaluate Yan Eq. (15) for longitudinal polarization in atomic units.

    For the longitudinal branch used in Fig. 2(a), ``s_n=(n+1)^2``.  The
    stable radius form below is algebraically identical to
    ``R0**(2*n+1)/Rd**(2*n+4)`` but avoids large intermediate powers.
    """

    n = np.asarray(degrees, dtype=float)[:, None]
    epsilon = np.asarray(epsilon_particle, dtype=complex)[None, :]
    gamma_n = (epsilon - eps_host) / (
        epsilon + ((n + 1.0) / n) * eps_host
    )
    s_n = (n + 1.0) ** 2
    eps_eff1 = (eps_qd + 2.0 * eps_host) / 3.0
    radial_factor = (radius_au / distance_au) ** (2.0 * n + 1.0) / distance_au**3
    g_by_degree = (
        s_n
        * eps_host
        * gamma_n
        * radial_factor
        * bare_dipole_au**2
        / eps_eff1**2
    )
    return np.cumsum(g_by_degree, axis=0)


def calculate_fig2a(
    *,
    energy_window_eV: tuple[float, float] = (1.5, 3.5),
    energy_points: int = 801,
    n_max: int = 10,
    sphere_radius_nm: float = 15.0,
    centre_distance_nm: float = 20.0,
    eps_host: float = 1.0,
    eps_qd: float = 6.0,
    transition_dipole_e_nm: float = 0.65,
) -> Yan2008Fig2AResult:
    """Calculate the Fig. 2(a) curves through the public full-QS API."""

    _validate_calculation_inputs(
        energy_window_eV=energy_window_eV,
        energy_points=energy_points,
        n_max=n_max,
        sphere_radius_nm=sphere_radius_nm,
        centre_distance_nm=centre_distance_nm,
        eps_host=eps_host,
        eps_qd=eps_qd,
        transition_dipole_e_nm=transition_dipole_e_nm,
    )
    energy_points = int(energy_points)
    n_max = int(n_max)
    energy = np.linspace(*energy_window_eV, energy_points)

    # In atomic units the elementary charge equals one, so a dipole expressed
    # in e*nm has the same numerical conversion as a length in nm.
    bare_dipole_au = float(nm_to_au(transition_dipole_e_nm))
    params = make_params_with_overrides(
        c_nm=sphere_radius_nm,
        a_nm=sphere_radius_nm,
        r_nm=centre_distance_nm,
        qd_radius_nm=0.0,
        eps_m=eps_host,
        eps_qd=eps_qd,
        d_debye=float(dipole_au_to_debye(bare_dipole_au)),
        omega0_ev=2.5,
        qd_dipole_convention="bare_internal",
        orientation="long",
    )
    kernel = SpheroidGreenInteraction.from_params(
        params,
        orientation="long",
        n_max=n_max,
    )
    response = kernel.response_from_material(params.material, energy)
    g_full_qs_au = (
        params.qd_external_dipole_au**2 * response.cumulative_K_au_minus3
    )
    epsilon = params.material.epsilon_at(energy)
    g_yan_au = _yan_eq15_cumulative_g_au(
        epsilon_particle=epsilon,
        degrees=response.degrees,
        radius_au=float(params.a_au),
        distance_au=float(params.R_au),
        eps_host=eps_host,
        eps_qd=eps_qd,
        bare_dipole_au=bare_dipole_au,
    )
    au_to_meV = AU_ENERGY_EV * 1000.0
    return Yan2008Fig2AResult(
        energy_eV=energy,
        degrees=response.degrees,
        g_full_qs_meV=g_full_qs_au * au_to_meV,
        g_yan_eq15_meV=g_yan_au * au_to_meV,
        epsilon_gold=epsilon,
        local_field_factor=float(params.qd_local_field_factor),
        bare_dipole_au=bare_dipole_au,
    )


def _create_unique_run_dir(output_dir: str | Path) -> Path:
    first = timestamped_run_dir(output_dir)
    first.parent.mkdir(parents=True, exist_ok=True)
    suffix = 0
    while True:
        candidate = first if suffix == 0 else first.with_name(
            f"{first.name}_{suffix:03d}"
        )
        try:
            candidate.mkdir(exist_ok=False)
        except FileExistsError:
            suffix += 1
            continue
        return candidate


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_curves_csv(path: Path, result: Yan2008Fig2AResult) -> None:
    fieldnames = ["energy_eV", "epsilon_Au_real", "epsilon_Au_imag"]
    for degree in result.degrees:
        for source in ("full_qs", "yan_eq15"):
            fieldnames.extend(
                [
                    f"G_N{degree}_{source}_real_meV",
                    f"G_N{degree}_{source}_imag_meV",
                ]
            )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index, energy in enumerate(result.energy_eV):
            row: dict[str, float] = {
                "energy_eV": float(energy),
                "epsilon_Au_real": float(result.epsilon_gold[index].real),
                "epsilon_Au_imag": float(result.epsilon_gold[index].imag),
            }
            for mode_index, degree in enumerate(result.degrees):
                for source, values in (
                    ("full_qs", result.g_full_qs_meV),
                    ("yan_eq15", result.g_yan_eq15_meV),
                ):
                    value = values[mode_index, index]
                    row[f"G_N{degree}_{source}_real_meV"] = float(value.real)
                    row[f"G_N{degree}_{source}_imag_meV"] = float(value.imag)
            writer.writerow(row)


def _curve_summary(result: Yan2008Fig2AResult, mode_index: int) -> dict[str, float]:
    values = result.g_full_qs_meV[mode_index]
    real_peak = int(np.argmax(values.real))
    imag_peak = int(np.argmax(values.imag))
    return {
        "N": int(result.degrees[mode_index]),
        "max_real_G_meV": float(values.real[real_peak]),
        "energy_at_max_real_G_eV": float(result.energy_eV[real_peak]),
        "max_imag_G_meV": float(values.imag[imag_peak]),
        "energy_at_max_imag_G_eV": float(result.energy_eV[imag_peak]),
    }


def _paper_raster_comparison_mode(
    result: Yan2008Fig2AResult,
    *,
    sphere_radius_nm: float,
    centre_distance_nm: float,
    eps_host: float,
    eps_qd: float,
    transition_dipole_e_nm: float,
) -> tuple[int | None, str | None]:
    """Return the N=10 row only when a raster peak check is meaningful."""

    supplied_profile = {
        "sphere_radius_nm": sphere_radius_nm,
        "centre_distance_nm": centre_distance_nm,
        "eps_host": eps_host,
        "eps_qd": eps_qd,
        "transition_dipole_e_nm": transition_dipole_e_nm,
    }
    changed = [
        name
        for name, expected in PAPER_PROFILE.items()
        if not np.isclose(
            supplied_profile[name],
            expected,
            rtol=0.0,
            atol=1.0e-12,
        )
    ]
    if changed:
        return None, (
            "paper-profile parameters were overridden: " + ", ".join(changed)
        )

    matches = np.flatnonzero(result.degrees == 10)
    if matches.size != 1:
        return None, "multipole degree N=10 was not calculated"

    search_min, search_max = PAPER_PEAK_SEARCH_WINDOW_EV
    if result.energy_eV[0] > search_min or result.energy_eV[-1] < search_max:
        return None, (
            "energy grid does not cover the complete paper-peak search window "
            f"[{search_min}, {search_max}] eV"
        )
    max_grid_step = float(np.max(np.diff(result.energy_eV)))
    if max_grid_step > PAPER_MAX_GRID_STEP_EV:
        return None, (
            f"energy-grid step {max_grid_step:.6g} eV exceeds the "
            f"{PAPER_MAX_GRID_STEP_EV:.6g} eV raster-check limit"
        )
    return int(matches[0]), None


def _parameter_provenance(name: str, value: float) -> str:
    source = PAPER_PARAMETER_SOURCES[name]
    expected = PAPER_PROFILE[name]
    if np.isclose(value, expected, rtol=0.0, atol=1.0e-12):
        return source
    return (
        f"runtime override: {name}={value:.16g}; canonical paper-profile "
        f"value {expected:.16g} and its source were not used"
    )


def _paper_raster_peak_comparison(
    result: Yan2008Fig2AResult,
    mode_index: int,
) -> dict[str, dict[str, float | bool]]:
    comparison: dict[str, dict[str, float | bool]] = {}
    values = result.g_full_qs_meV[mode_index]
    search_min, search_max = PAPER_PEAK_SEARCH_WINDOW_EV
    search_mask = (result.energy_eV >= search_min) & (
        result.energy_eV <= search_max
    )
    search_indices = np.flatnonzero(search_mask)
    for name, component in (
        ("N10_real_peak", "real"),
        ("N10_imag_peak", "imag"),
    ):
        component_values = getattr(values, component)
        calculated_index = int(
            search_indices[np.argmax(component_values[search_indices])]
        )
        calculated_value = float(component_values[calculated_index])
        calculated_energy = float(result.energy_eV[calculated_index])
        anchor = PAPER_FIG2A_RASTER_ANCHORS[name]
        energy_difference = calculated_energy - anchor["energy_eV"]
        g_difference = calculated_value - anchor["G_meV"]
        comparison[name] = {
            "paper_energy_eV": anchor["energy_eV"],
            "paper_energy_uncertainty_eV": anchor["energy_uncertainty_eV"],
            "paper_G_meV": anchor["G_meV"],
            "paper_G_uncertainty_meV": anchor["G_uncertainty_meV"],
            "full_qs_energy_eV": calculated_energy,
            "full_qs_G_meV": calculated_value,
            "energy_difference_eV": energy_difference,
            "G_difference_meV": g_difference,
            "relative_G_difference": g_difference / anchor["G_meV"],
            "energy_difference_in_readout_uncertainties": (
                energy_difference / anchor["energy_uncertainty_eV"]
            ),
            "G_difference_in_readout_uncertainties": (
                g_difference / anchor["G_uncertainty_meV"]
            ),
            "inside_readout_intervals": bool(
                abs(energy_difference) <= anchor["energy_uncertainty_eV"]
                and abs(g_difference) <= anchor["G_uncertainty_meV"]
            ),
        }
    return comparison


def _plot_fig2a(
    run_dir: Path,
    result: Yan2008Fig2AResult,
    *,
    show_paper_anchors: bool,
) -> Figure:
    figure, axis = plt.subplots(figsize=(7.2, 5.4))
    count = max(result.degrees.size - 1, 1)
    for index, degree in enumerate(result.degrees):
        alpha = 0.48 + 0.52 * index / count
        axis.plot(
            result.energy_eV,
            result.g_full_qs_meV[index].real,
            color="red",
            linewidth=1.25,
            alpha=alpha,
        )
        axis.plot(
            result.energy_eV,
            result.g_full_qs_meV[index].imag,
            color="blue",
            linestyle=(0, (2, 2)),
            linewidth=1.2,
            alpha=alpha,
        )

    # Independent Eq. (15) markers are shown only for the limiting curves, so
    # their exact overlap with the API result remains visible without clutter.
    marker_stride = max(1, result.energy_eV.size // 24)
    marker_indices = np.arange(0, result.energy_eV.size, marker_stride)
    for index in sorted({0, result.degrees.size - 1}):
        axis.plot(
            result.energy_eV[marker_indices],
            result.g_yan_eq15_meV[index, marker_indices].real,
            linestyle="none",
            marker="o",
            markersize=3.2,
            markerfacecolor="none",
            markeredgecolor="darkred",
            markeredgewidth=0.7,
        )
        axis.plot(
            result.energy_eV[marker_indices],
            result.g_yan_eq15_meV[index, marker_indices].imag,
            linestyle="none",
            marker="s",
            markersize=3.0,
            markerfacecolor="none",
            markeredgecolor="navy",
            markeredgewidth=0.7,
        )

    if show_paper_anchors:
        for anchor in PAPER_FIG2A_RASTER_ANCHORS.values():
            axis.errorbar(
                anchor["energy_eV"],
                anchor["G_meV"],
                xerr=anchor["energy_uncertainty_eV"],
                yerr=anchor["G_uncertainty_meV"],
                linestyle="none",
                marker="x",
                markersize=6.0,
                color="black",
                capsize=2.0,
                zorder=5,
            )

    axis.plot([], [], color="red", label=r"full-QS: $\operatorname{Re}G_N$")
    axis.plot(
        [],
        [],
        color="blue",
        linestyle=(0, (2, 2)),
        label=r"full-QS: $\operatorname{Im}G_N$",
    )
    axis.plot(
        [],
        [],
        linestyle="none",
        marker="o",
        markerfacecolor="none",
        markeredgecolor="0.2",
        label="Yan Eq. (15), N=1 and N=max",
    )
    if show_paper_anchors:
        axis.plot(
            [],
            [],
            linestyle="none",
            marker="x",
            color="black",
            label="Fig. 2(a) graphical peak readout, N=10",
        )
    axis.set(
        xlabel=r"Photon energy $\hbar\omega$ (eV)",
        ylabel=r"Self-interaction energy $\hbar G_N$ (meV)",
        xlim=(float(result.energy_eV[0]), float(result.energy_eV[-1])),
    )
    plotted_min = float(
        min(np.min(result.g_full_qs_meV.real), np.min(result.g_full_qs_meV.imag))
    )
    plotted_max = float(
        max(np.max(result.g_full_qs_meV.real), np.max(result.g_full_qs_meV.imag))
    )
    anchor_top = (
        max(
            anchor["G_meV"] + anchor["G_uncertainty_meV"]
            for anchor in PAPER_FIG2A_RASTER_ANCHORS.values()
        )
        if show_paper_anchors
        else 0.0
    )
    axis.set_ylim(
        min(0.0, 1.05 * plotted_min),
        max(0.22, 1.05 * plotted_max, 1.02 * anchor_top),
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=9)
    axis.text(
        0.02,
        0.97,
        f"Yan et al. Fig. 2(a): N=1...{result.degrees[-1]}",
        transform=axis.transAxes,
        va="top",
    )
    figure.tight_layout()
    figure.savefig(run_dir / PLOT_FILENAME, dpi=220)
    return figure


def run_reproduction(
    *,
    output_dir: str | Path = "results/literature/yan2008_fig2a",
    article_pdf: str | Path = DEFAULT_ARTICLE_PDF,
    energy_window_eV: tuple[float, float] = (1.5, 3.5),
    energy_points: int = 801,
    n_max: int = 10,
    sphere_radius_nm: float = 15.0,
    centre_distance_nm: float = 20.0,
    eps_host: float = 1.0,
    eps_qd: float = 6.0,
    transition_dipole_e_nm: float = 0.65,
    make_plot: bool = True,
    show: bool = False,
) -> Path:
    """Calculate, validate and export the Yan Fig. 2(a) reconstruction."""

    result = calculate_fig2a(
        energy_window_eV=energy_window_eV,
        energy_points=energy_points,
        n_max=n_max,
        sphere_radius_nm=sphere_radius_nm,
        centre_distance_nm=centre_distance_nm,
        eps_host=eps_host,
        eps_qd=eps_qd,
        transition_dipole_e_nm=transition_dipole_e_nm,
    )
    run_dir = _create_unique_run_dir(output_dir)
    _write_curves_csv(run_dir / CSV_FILENAME, result)
    article_path = Path(article_pdf).resolve()
    nmax_real_peak_index = int(np.argmax(result.g_full_qs_meV[-1].real))
    n1_at_peak = result.g_full_qs_meV[0, nmax_real_peak_index]
    nmax_at_peak = result.g_full_qs_meV[-1, nmax_real_peak_index]
    n1_real_max = float(np.max(result.g_full_qs_meV[0].real))
    nmax_real_max = float(np.max(result.g_full_qs_meV[-1].real))
    raster_mode_index, raster_unavailable_reason = _paper_raster_comparison_mode(
        result,
        sphere_radius_nm=sphere_radius_nm,
        centre_distance_nm=centre_distance_nm,
        eps_host=eps_host,
        eps_qd=eps_qd,
        transition_dipole_e_nm=transition_dipole_e_nm,
    )
    raster_comparison = (
        _paper_raster_peak_comparison(result, raster_mode_index)
        if raster_mode_index is not None
        else None
    )
    paper_text_comparison = None
    if raster_mode_index is not None:
        search_min, search_max = PAPER_PEAK_SEARCH_WINDOW_EV
        search_indices = np.flatnonzero(
            (result.energy_eV >= search_min)
            & (result.energy_eV <= search_max)
        )
        n10_real_peak_index = int(
            search_indices[
                np.argmax(
                    result.g_full_qs_meV[
                        raster_mode_index,
                        search_indices,
                    ].real
                )
            ]
        )
        n1_at_n10_peak = result.g_full_qs_meV[0, n10_real_peak_index].real
        n10_at_peak = result.g_full_qs_meV[
            raster_mode_index,
            n10_real_peak_index,
        ].real
        paper_text_comparison = {
            "energy_at_N10_real_peak_eV": float(
                result.energy_eV[n10_real_peak_index]
            ),
            "real_G_N1_at_N10_peak_meV": float(n1_at_n10_peak),
            "real_G_N10_at_its_peak_meV": float(n10_at_peak),
            "real_G_N10_over_N1_at_N10_peak": float(
                n10_at_peak / n1_at_n10_peak
            ),
            "ratio_of_real_peak_heights_N10_over_N1": float(
                np.max(result.g_full_qs_meV[raster_mode_index].real)
                / n1_real_max
            ),
        }
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "article": {
            "citation": (
                "J.-Y. Yan et al., Phys. Rev. B 77, 165301 (2008)"
            ),
            "doi": ARTICLE_DOI,
            "figure": ARTICLE_FIGURE,
            "pdf_path": str(article_path),
            "pdf_sha256": _sha256(article_path),
        },
        "model": {
            "backend": "SpheroidGreenInteraction exact spherical limit",
            "orientation": "long",
            "quantity": "hbar*G_N = d_bare^2*l_QD^2*sum_(n<=N) K_n",
            "independent_reference": "Yan et al. Eq. (15)",
        },
        "physical_parameters": {
            "sphere_radius_nm": sphere_radius_nm,
            "centre_distance_nm": centre_distance_nm,
            "surface_distance_nm_point_qd": centre_distance_nm - sphere_radius_nm,
            "eps_qd": eps_qd,
            "eps_host": eps_host,
            "transition_dipole_e_nm": transition_dipole_e_nm,
            "transition_dipole_au": result.bare_dipole_au,
            "qd_local_field_factor": result.local_field_factor,
            "gold_data": "Johnson-Christy through the project material API",
        },
        "parameter_provenance": {
            "sphere_radius_nm": _parameter_provenance(
                "sphere_radius_nm",
                sphere_radius_nm,
            ),
            "centre_distance_nm": _parameter_provenance(
                "centre_distance_nm",
                centre_distance_nm,
            ),
            "eps_qd": _parameter_provenance("eps_qd", eps_qd),
            "gold_data": "Yan 2008 reference 28: Johnson and Christy (1972)",
            "eps_host": _parameter_provenance("eps_host", eps_host),
            "transition_dipole_e_nm": _parameter_provenance(
                "transition_dipole_e_nm",
                transition_dipole_e_nm,
            ),
            "orientation": (
                "longitudinal Eq. (15) branch s_n=(n+1)^2, matching Fig. 2(a)"
            ),
        },
        "numerical_settings": {
            "energy_min_eV": float(result.energy_eV[0]),
            "energy_max_eV": float(result.energy_eV[-1]),
            "energy_points": int(result.energy_eV.size),
            "n_max": int(result.degrees[-1]),
        },
        "comparison": {
            "max_absolute_full_qs_vs_eq15_meV": result.max_absolute_difference_meV,
            "max_relative_full_qs_vs_eq15": result.max_relative_difference,
            "paper_text_target": "N=10 shift is almost seven times N=1",
            "paper_text_target_evaluated": raster_mode_index is not None,
            "paper_text_target_unavailable_reason": raster_unavailable_reason,
            "paper_text_target_comparison": paper_text_comparison,
            "energy_at_Nmax_real_peak_eV": float(
                result.energy_eV[nmax_real_peak_index]
            ),
            "real_G_N1_at_Nmax_peak_meV": float(n1_at_peak.real),
            "real_G_Nmax_at_its_peak_meV": float(nmax_at_peak.real),
            "real_G_Nmax_over_N1_at_Nmax_peak": float(
                nmax_at_peak.real / n1_at_peak.real
            ),
            "ratio_of_real_peak_heights_Nmax_over_N1": float(
                nmax_real_max / n1_real_max
            ),
            "curve_summaries": [
                _curve_summary(result, 0),
                _curve_summary(result, result.degrees.size - 1),
            ],
            "paper_raster_peak_comparison_available": raster_mode_index is not None,
            "paper_raster_peak_comparison_unavailable_reason": (
                raster_unavailable_reason
            ),
            "paper_raster_peak_comparison": raster_comparison,
            "paper_raster_peak_search_window_eV": list(
                PAPER_PEAK_SEARCH_WINDOW_EV
            ),
            "digitized_full_curve_metrics_available": False,
            "digitization_note": (
                "The paper supplies no raw numerical array. Two low-precision "
                "N=10 peak anchors were read from a 220-dpi rendering of the "
                "publisher PDF and carry explicit graphical uncertainties. "
                "The uncertainty-normalized differences are readout-scale "
                "diagnostics, not statistical z scores. The anchors are not "
                "presented as author-supplied data or proof of quantitative "
                "agreement; the full-curve normalization check is instead made "
                "against Yan Eq. (15)."
            ),
        },
    }
    write_json(run_dir / METADATA_FILENAME, metadata)

    figure: Figure | None = None
    if make_plot:
        figure = _plot_fig2a(
            run_dir,
            result,
            show_paper_anchors=raster_mode_index is not None,
        )
    if show and figure is not None:
        plt.show()
    elif figure is not None:
        plt.close(figure)
    return run_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="results/literature/yan2008_fig2a",
    )
    parser.add_argument("--article-pdf", default=str(DEFAULT_ARTICLE_PDF))
    parser.add_argument("--energy-min-ev", type=float, default=1.5)
    parser.add_argument("--energy-max-ev", type=float, default=3.5)
    parser.add_argument("--energy-points", type=int, default=801)
    parser.add_argument("--n-max", type=int, default=10)
    parser.add_argument("--sphere-radius-nm", type=float, default=15.0)
    parser.add_argument("--centre-distance-nm", type=float, default=20.0)
    parser.add_argument("--eps-host", type=float, default=1.0)
    parser.add_argument("--eps-qd", type=float, default=6.0)
    parser.add_argument("--transition-dipole-e-nm", type=float, default=0.65)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    run_dir = run_reproduction(
        output_dir=args.output_dir,
        article_pdf=args.article_pdf,
        energy_window_eV=(args.energy_min_ev, args.energy_max_ev),
        energy_points=args.energy_points,
        n_max=args.n_max,
        sphere_radius_nm=args.sphere_radius_nm,
        centre_distance_nm=args.centre_distance_nm,
        eps_host=args.eps_host,
        eps_qd=args.eps_qd,
        transition_dipole_e_nm=args.transition_dipole_e_nm,
        make_plot=not args.no_plot,
        show=args.show,
    )
    print(f"Saved Yan 2008 Fig. 2(a) reconstruction to {run_dir}")


if __name__ == "__main__":
    main()
