#!/usr/bin/env python3
"""Reproduce McMillan et al. (2016), Fig. 4(c), with the new time core.

The article-matched curves use one *spatial* order (the dipole approximation
of the paper) and eighty passive *material* poles.  Thus this script directly
tests the causal Lorentz/ADE realization coupled to the non-RWA Bloch system.
It does not claim that Fig. 4(c) independently validates higher spatial
multipoles; that is a separate benchmark.

If ``--article-pdf`` is supplied, panel 4(c) is digitized directly from page 6
of the arXiv/accepted-manuscript layout and pointwise comparison metrics are
written to ``metadata.json``.  Without a local PDF, the calculation is still
run and compared with the numerical landmarks explicitly stated in the paper.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

# Allow both documented invocation forms:
# ``python -m literature_reproductions.mcmillan2016_fig4c_full_qs`` and
# ``python literature_reproductions/mcmillan2016_fig4c_full_qs.py``.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from literature_reproductions.mcmillan2016_common import (
    EXCITON_ENERGY_EV,
    FIT_WINDOW_EV,
    PAPER_ARXIV_URL,
    PAPER_DOI,
    PULSE_AREA_PI,
    PULSE_CYCLES,
    build_paper_matched_model,
    find_article_pdf,
    fit_passive_lorentz_sphere,
    make_etchegoin_material,
    make_paper_params,
    make_paper_pulse,
    paper_time_fs,
    sphere_bright_susceptibility,
)
from qd_mnp_rational_fit import (
    RationalLorentzFit,
    au_to_fs,
    eV_to_au,
    field_au_to_si,
    timestamped_run_dir,
    write_json,
)


SEPARATIONS_NM = (80.0, 20.0, 13.0)
COLORS = {80.0: "#d62728", 20.0: "#2ca02c", 13.0: "#1f77b4"}


@dataclass(frozen=True)
class DigitizedCurve:
    separation_nm: float
    time_fs: np.ndarray
    rho22: np.ndarray
    observed: np.ndarray


def _snap_dark_line(
    darkness: np.ndarray,
    expected: int,
    *,
    axis: int,
    search_radius: int,
    slice_start: int,
    slice_stop: int,
) -> int:
    candidates = range(max(0, expected - search_radius), expected + search_radius + 1)
    if axis == 0:
        scores = [int(np.count_nonzero(darkness[value, slice_start:slice_stop])) for value in candidates]
    else:
        scores = [int(np.count_nonzero(darkness[slice_start:slice_stop, value])) for value in candidates]
    return int(list(candidates)[int(np.argmax(scores))])


def _render_article_page(pdf_path: Path, output_png: Path) -> None:
    executable = shutil.which("pdftoppm")
    if executable is None:
        raise RuntimeError(
            "pdftoppm is required only for --article-pdf digitization and was not found."
        )
    prefix = output_png.with_suffix("")
    subprocess.run(
        [
            executable,
            "-f",
            "6",
            "-l",
            "6",
            "-singlefile",
            "-png",
            "-r",
            "250",
            str(pdf_path),
            str(prefix),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    rendered = prefix.with_suffix(".png")
    if rendered != output_png:
        rendered.replace(output_png)


def digitize_article_fig4c(
    pdf_path: Path,
    *,
    duration_fs: float,
) -> dict[float, DigitizedCurve]:
    """Digitize the red/green/blue curves in the published panel 4(c).

    Calibration uses the actual plot frame, snapped near its known relative
    location after a deterministic 250-dpi render.  The legend and the
    polarizability inset are explicitly excluded.  ``observed`` distinguishes
    real colored pixels from interpolation across dotted-line gaps/occlusion.
    """

    with tempfile.TemporaryDirectory(prefix="mcmillan2016_fig4c_") as temporary:
        png_path = Path(temporary) / "page6.png"
        _render_article_page(pdf_path, png_path)
        raw = np.asarray(mpimg.imread(png_path))
    if raw.ndim != 3 or raw.shape[2] < 3:
        raise RuntimeError("Rendered article page is not an RGB image.")
    rgb = raw[:, :, :3]
    if np.issubdtype(rgb.dtype, np.floating):
        rgb = np.asarray(np.rint(np.clip(rgb, 0.0, 1.0) * 255.0), dtype=np.uint8)
    else:
        rgb = np.asarray(rgb, dtype=np.uint8)

    height, width = rgb.shape[:2]
    if not (0.74 < width / height < 0.80):
        raise RuntimeError(
            f"Unexpected PDF page aspect ratio for Fig. 4 digitization: {width}x{height}."
        )
    darkness = np.all(rgb < 100, axis=2)
    expected_x0 = round(373.0 / 2125.0 * width)
    expected_x1 = round(1044.0 / 2125.0 * width)
    expected_y0 = round(1285.0 / 2750.0 * height)
    expected_y1 = round(1715.0 / 2750.0 * height)
    x0 = _snap_dark_line(
        darkness,
        expected_x0,
        axis=1,
        search_radius=max(6, round(width / 250.0)),
        slice_start=round(0.44 * height),
        slice_stop=round(0.64 * height),
    )
    x1 = _snap_dark_line(
        darkness,
        expected_x1,
        axis=1,
        search_radius=max(6, round(width / 250.0)),
        slice_start=round(0.44 * height),
        slice_stop=round(0.64 * height),
    )
    y0 = _snap_dark_line(
        darkness,
        expected_y0,
        axis=0,
        search_radius=max(6, round(height / 300.0)),
        slice_start=x0,
        slice_stop=x1 + 1,
    )
    y1 = _snap_dark_line(
        darkness,
        expected_y1,
        axis=0,
        search_radius=max(6, round(height / 300.0)),
        slice_start=x0,
        slice_stop=x1 + 1,
    )
    if not (x1 - x0 > 0.25 * width and y1 - y0 > 0.12 * height):
        raise RuntimeError("Could not locate the panel 4(c) plot frame.")

    panel = rgb[y0 : y1 + 1, x0 : x1 + 1]
    red = (
        (panel[:, :, 0] > 150)
        & (panel[:, :, 0] > 1.5 * panel[:, :, 1].astype(float))
        & (panel[:, :, 0] > 1.5 * panel[:, :, 2].astype(float))
    )
    green = (
        (panel[:, :, 1] > 100)
        & (panel[:, :, 1] > 1.3 * panel[:, :, 0].astype(float))
        & (panel[:, :, 1] > 1.3 * panel[:, :, 2].astype(float))
    )
    blue = (
        (panel[:, :, 2] > 150)
        & (panel[:, :, 2] > 1.5 * panel[:, :, 0].astype(float))
        & (panel[:, :, 2] > 1.5 * panel[:, :, 1].astype(float))
    )
    masks = {80.0: red, 20.0: green, 13.0: blue}

    pixel_x = np.arange(x1 - x0 + 1)
    time_fs = pixel_x * duration_fs / float(x1 - x0)
    curves: dict[float, DigitizedCurve] = {}
    for separation, mask in masks.items():
        pixel_y = np.full(pixel_x.size, np.nan)
        observed = np.zeros(pixel_x.size, dtype=bool)
        for local_x in pixel_x:
            candidates = np.flatnonzero(mask[:, local_x])
            absolute_x = local_x + x0
            absolute_y = candidates + y0
            # Top-left legend and lower-right polarizability inset.
            excluded = (
                (
                    (absolute_x >= round(380.0 / 2125.0 * width))
                    & (absolute_x <= round(610.0 / 2125.0 * width))
                    & (absolute_y <= round(1380.0 / 2750.0 * height))
                )
                | (
                    (absolute_x >= round(790.0 / 2125.0 * width))
                    & (absolute_x <= round(1025.0 / 2125.0 * width))
                    & (absolute_y >= round(1460.0 / 2750.0 * height))
                )
            )
            candidates = candidates[~excluded]
            if candidates.size:
                pixel_y[local_x] = float(np.median(candidates))
                observed[local_x] = True
        if np.count_nonzero(observed) < 0.45 * pixel_x.size:
            raise RuntimeError(
                f"Too few colored pixels were found for R={separation:g} nm."
            )
        rho_observed = (float(y1 - y0) - pixel_y) / float(y1 - y0)
        rho22 = np.interp(
            time_fs,
            time_fs[observed],
            rho_observed[observed],
        )
        curves[separation] = DigitizedCurve(
            separation_nm=separation,
            time_fs=time_fs.copy(),
            rho22=np.clip(rho22, 0.0, 1.0),
            observed=observed,
        )
    return curves


def evaluate_fit(fit: RationalLorentzFit, energies_eV: np.ndarray) -> np.ndarray:
    omega = np.asarray(eV_to_au(energies_eV), dtype=float)
    denominator = (
        fit.omega_modes_au[:, None] ** 2
        - omega[None, :] ** 2
        - 1j * fit.gamma_modes_au[:, None] * omega[None, :]
    )
    return np.asarray(
        fit.alpha_inf + np.sum(fit.strengths_au2[:, None] / denominator, axis=0),
        dtype=complex,
    )


def curve_metrics(
    time_fs: np.ndarray,
    rho22: np.ndarray,
    reference: DigitizedCurve | None,
) -> dict[str, object]:
    post = time_fs >= 18.0
    post_indices = np.flatnonzero(post)
    peak_index = int(post_indices[np.argmax(rho22[post])])
    post_peak = float(rho22[peak_index])
    first_near_peak = int(
        post_indices[np.flatnonzero(rho22[post] >= post_peak - 0.01)[0]]
    )
    payload: dict[str, object] = {
        "post_18fs_peak_rho22": post_peak,
        "post_18fs_peak_time_fs": float(time_fs[peak_index]),
        "first_time_within_0p01_of_post_peak_fs": float(time_fs[first_near_peak]),
        "final_rho22": float(rho22[-1]),
        "global_max_rho22": float(np.max(rho22)),
        "global_min_rho22": float(np.min(rho22)),
    }
    if reference is not None:
        simulated = np.interp(reference.time_fs, time_fs, rho22)
        observed = reference.observed
        residual = simulated[observed] - reference.rho22[observed]
        observed_post = observed & (reference.time_fs >= 18.0)
        post_residual = simulated[observed_post] - reference.rho22[observed_post]
        plateau = observed & (reference.time_fs >= 30.0)
        article_plateau = float(np.mean(reference.rho22[plateau]))
        simulated_plateau = float(np.mean(simulated[plateau]))
        payload["digitized_comparison"] = {
            "observed_raster_points": int(np.count_nonzero(observed)),
            "rmse_rho22_all_observed": float(np.sqrt(np.mean(residual**2))),
            "max_abs_rho22_all_observed": float(np.max(np.abs(residual))),
            "rmse_rho22_post_18fs": float(np.sqrt(np.mean(post_residual**2))),
            "max_abs_rho22_post_18fs": float(np.max(np.abs(post_residual))),
            "article_mean_rho22_30fs_to_end": article_plateau,
            "simulated_mean_rho22_30fs_to_end": simulated_plateau,
            "mean_rho22_difference_30fs_to_end": float(
                simulated_plateau - article_plateau
            ),
            "digitization_population_half_pixel": float(0.5 / 430.0),
            "digitization_time_half_pixel_fs": float(
                0.5 * reference.time_fs[-1] / (reference.time_fs.size - 1)
            ),
        }
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce McMillan et al. 2016 Fig. 4(c): rho22(t) for a 10-cycle "
            "5-pi sech pulse using the causal full-QS time core in its N=1 limit."
        )
    )
    parser.add_argument(
        "--article-pdf",
        type=Path,
        default=None,
        help=(
            "Optional local McMillan 2016 PDF. If supplied, page 6 panel 4(c) "
            "is digitized and pointwise metrics are calculated."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/literature_mcmillan2016_fig4c"),
    )
    parser.add_argument("--selected-material-poles", type=int, default=80)
    parser.add_argument("--fit-points", type=int, default=1001)
    parser.add_argument("--points-per-fastest-cycle", type=float, default=20.0)
    parser.add_argument("--rtol", type=float, default=3.0e-8)
    parser.add_argument("--atol", type=float, default=1.0e-10)
    parser.add_argument(
        "--skip-step-convergence",
        action="store_true",
        help="Skip the independent 1.5x max-step refinement for R=13 nm.",
    )
    parser.add_argument(
        "--material-pole-convergence-counts",
        type=int,
        nargs=2,
        default=(60, 100),
        metavar=("LOW", "HIGH"),
        help=(
            "Two additional material-pole counts used to bracket the default "
            "80-pole R=13 nm trajectory (default: 60 100)."
        ),
    )
    parser.add_argument(
        "--skip-material-pole-convergence",
        action="store_true",
        help="Skip the additional R=13 nm material-pole convergence runs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.points_per_fastest_cycle < 8.0:
        raise ValueError("--points-per-fastest-cycle must be at least 8.")
    if any(value < 8 for value in args.material_pole_convergence_counts):
        raise ValueError("Material-pole convergence counts must be at least 8.")
    article_pdf = find_article_pdf(args.article_pdf)
    run_dir = timestamped_run_dir(args.output_root.resolve())
    run_dir.mkdir(parents=True, exist_ok=False)

    started = time.perf_counter()
    material = make_etchegoin_material()
    fit, fit_audit = fit_passive_lorentz_sphere(
        selected_poles=args.selected_material_poles,
        fit_points=args.fit_points,
    )

    curves: dict[float, dict[str, np.ndarray]] = {}
    diagnostics: dict[str, dict[str, object]] = {}
    pulse_metadata: dict[str, dict[str, float]] = {}
    duration_fs: float | None = None
    for separation in SEPARATIONS_NM:
        params = make_paper_params(separation, material=material)
        model = build_paper_matched_model(params, fit, spatial_orders=1)
        profile = make_paper_pulse(params)
        current_duration_fs = float(au_to_fs(profile.duration_au))
        if duration_fs is None:
            duration_fs = current_duration_fs
        elif not np.isclose(duration_fs, current_duration_fs, rtol=0.0, atol=1.0e-12):
            raise RuntimeError("Paper pulse durations unexpectedly differ between separations.")
        result = model.solve(
            profile.pulse,
            t_span_au=(-0.5 * profile.duration_au, 0.5 * profile.duration_au),
            method="DOP853",
            rtol=args.rtol,
            atol=args.atol,
            points_per_fastest_cycle=args.points_per_fastest_cycle,
            spectral_window_policy="raise",
            # The incident and all MNP channels are below 1e-3.  The nonlinear
            # QD source has about 0.0033 outside the paper's 0-10 eV fit window;
            # retain the article window and record this bounded leakage.
            max_spectral_leakage=0.01,
            positivity_policy="raise",
            positivity_tolerance=1.0e-7,
            work_passivity_policy="warn",
            # Fig. 4 stops at T while T1,T2 are hundreds of ps/ns.  Therefore
            # a post-pulse component-tail criterion is inapplicable here.
            response_tail_policy="ignore",
        )
        paper_t = paper_time_fs(result.t_au, profile)
        curves[separation] = {
            "time_fs": paper_t,
            "rho22": result.rho22.copy(),
            "W": result.W.copy(),
            "mu_p_au": result.mu_p_au.copy(),
            "incident_field_au": result.incident_field_au.copy(),
        }
        diagnostics[f"R_{separation:g}_nm"] = asdict(result.diagnostics)
        pulse_metadata[f"R_{separation:g}_nm"] = {
            "E0_au": float(profile.pulse.E0_au),
            "E0_V_per_m": float(field_au_to_si(profile.pulse.E0_au)),
            "duration_fs": current_duration_fs,
            "tau_p_fs": float(au_to_fs(profile.tau_p_au)),
            "local_field_amplitude_factor_abs": float(
                profile.local_field_amplitude_factor
            ),
            "pulse_positive_frequency_leakage_outside_0p01_10eV": float(
                profile.pulse.spectral_leakage_fraction(FIT_WINDOW_EV)
            ),
        }

    assert duration_fs is not None
    digitized = (
        None
        if article_pdf is None
        else digitize_article_fig4c(article_pdf, duration_fs=duration_fs)
    )
    metrics = {
        f"R_{separation:g}_nm": curve_metrics(
            curves[separation]["time_fs"],
            curves[separation]["rho22"],
            None if digitized is None else digitized[separation],
        )
        for separation in SEPARATIONS_NM
    }

    convergence: dict[str, object] | None = None
    if not args.skip_step_convergence:
        separation = 13.0
        params = make_paper_params(separation, material=material)
        model = build_paper_matched_model(params, fit, spatial_orders=1)
        profile = make_paper_pulse(params)
        refined = model.solve(
            profile.pulse,
            t_span_au=(-0.5 * profile.duration_au, 0.5 * profile.duration_au),
            method="DOP853",
            rtol=args.rtol,
            atol=args.atol,
            points_per_fastest_cycle=1.5 * args.points_per_fastest_cycle,
            spectral_window_policy="raise",
            max_spectral_leakage=0.01,
            positivity_policy="raise",
            work_passivity_policy="warn",
            response_tail_policy="ignore",
        )
        refined_t = paper_time_fs(refined.t_au, profile)
        base_t = curves[separation]["time_fs"]
        base_rho = curves[separation]["rho22"]
        difference = np.interp(base_t, refined_t, refined.rho22) - base_rho
        convergence = {
            "separation_nm": separation,
            "base_points_per_fastest_cycle": float(args.points_per_fastest_cycle),
            "refined_points_per_fastest_cycle": float(
                1.5 * args.points_per_fastest_cycle
            ),
            "rho22_rms_difference": float(np.sqrt(np.mean(difference**2))),
            "rho22_max_abs_difference": float(np.max(np.abs(difference))),
            "refined_n_steps": int(refined.diagnostics.n_steps),
        }

    material_pole_convergence: dict[str, object] | None = None
    if not args.skip_material_pole_convergence:
        separation = 13.0
        base_t = curves[separation]["time_fs"]
        base_rho = curves[separation]["rho22"]
        variants: dict[str, object] = {}
        for pole_count in args.material_pole_convergence_counts:
            variant_fit, variant_audit = fit_passive_lorentz_sphere(
                selected_poles=pole_count,
                fit_points=args.fit_points,
            )
            params = make_paper_params(separation, material=material)
            model = build_paper_matched_model(params, variant_fit, spatial_orders=1)
            profile = make_paper_pulse(params)
            variant_result = model.solve(
                profile.pulse,
                t_span_au=(-0.5 * profile.duration_au, 0.5 * profile.duration_au),
                method="DOP853",
                rtol=args.rtol,
                atol=args.atol,
                points_per_fastest_cycle=args.points_per_fastest_cycle,
                spectral_window_policy="raise",
                max_spectral_leakage=0.01,
                positivity_policy="raise",
                positivity_tolerance=1.0e-7,
                work_passivity_policy="warn",
                response_tail_policy="ignore",
            )
            variant_t = paper_time_fs(variant_result.t_au, profile)
            variant_rho = np.interp(base_t, variant_t, variant_result.rho22)
            difference = variant_rho - base_rho
            variants[str(pole_count)] = {
                "fit_audit": asdict(variant_audit),
                "rho22_rms_difference_from_base": float(
                    np.sqrt(np.mean(difference**2))
                ),
                "rho22_max_abs_difference_from_base": float(
                    np.max(np.abs(difference))
                ),
                "final_rho22": float(variant_result.rho22[-1]),
                "n_steps": int(variant_result.diagnostics.n_steps),
            }
        material_pole_convergence = {
            "separation_nm": separation,
            "base_material_poles": int(fit.strengths_au2.size),
            "variants": variants,
        }

    figure, axis = plt.subplots(figsize=(8.4, 5.8), constrained_layout=True)
    for separation in SEPARATIONS_NM:
        values = curves[separation]
        axis.plot(
            values["time_fs"],
            values["rho22"],
            color=COLORS[separation],
            linewidth=2.0,
            label=f"full-QS core (n=1), R={separation:g} nm",
        )
        if digitized is not None:
            reference = digitized[separation]
            sampled = np.flatnonzero(reference.observed)[::5]
            axis.plot(
                reference.time_fs[sampled],
                reference.rho22[sampled],
                linestyle="none",
                marker=".",
                markersize=3.0,
                alpha=0.70,
                color=COLORS[separation],
                label=f"article raster, R={separation:g} nm",
            )
    if digitized is None:
        axis.scatter(
            [23.0, 21.0, 20.0],
            [0.95, 0.95, 0.95],
            marker="x",
            s=45,
            color=[COLORS[value] for value in SEPARATIONS_NM],
            label="peak landmarks stated in article text",
            zorder=5,
        )
        axis.scatter(
            [duration_fs],
            [0.60],
            marker="x",
            s=45,
            color=COLORS[13.0],
            label="R=13 nm endpoint stated as about 0.6",
            zorder=5,
        )
    axis.set_xlim(0.0, duration_fs)
    axis.set_ylim(-0.01, 1.01)
    axis.set_xlabel("Time, t (fs)")
    axis.set_ylabel(r"Excited-state population, $\rho_{22}(t)$")
    axis.set_title("McMillan et al. (2016), Fig. 4(c): causal material-memory test")
    axis.grid(alpha=0.20)
    axis.legend(loc="upper left", fontsize=8, ncol=2 if digitized is not None else 1)

    inset = axis.inset_axes([0.58, 0.10, 0.38, 0.30])
    fit_energies = np.linspace(FIT_WINDOW_EV[0], FIT_WINDOW_EV[1], 1201)
    exact_alpha_over_a3 = sphere_bright_susceptibility(fit_energies) / 3.0
    fitted_alpha_over_a3 = evaluate_fit(fit, fit_energies) / 3.0
    inset.plot(fit_energies, exact_alpha_over_a3.real, color="0.45", linewidth=1.3)
    inset.plot(fit_energies, exact_alpha_over_a3.imag, color="0.45", linewidth=1.3)
    inset.plot(
        fit_energies,
        fitted_alpha_over_a3.real,
        color="#1f77b4",
        linestyle="--",
        linewidth=1.0,
        label="Re fit",
    )
    inset.plot(
        fit_energies,
        fitted_alpha_over_a3.imag,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="Im fit",
    )
    inset.set_xlim(0.0, 10.0)
    inset.set_ylim(0.0, 2.0)
    inset.set_xlabel(r"$\hbar\omega$ (eV)", fontsize=7)
    inset.set_ylabel(r"$\alpha_{\mathrm{MNP}}/a^3$", fontsize=7)
    inset.tick_params(labelsize=7)
    inset.grid(alpha=0.15)
    inset.legend(fontsize=6, frameon=False, ncol=2, loc="upper right")

    figure.savefig(run_dir / "mcmillan2016_fig4c_full_qs.png", dpi=220)
    figure.savefig(run_dir / "mcmillan2016_fig4c_full_qs.pdf")
    plt.close(figure)

    common_time = np.linspace(0.0, duration_fs, 2001)
    csv_columns = [common_time]
    csv_header = ["time_fs"]
    archive: dict[str, np.ndarray] = {}
    for separation in SEPARATIONS_NM:
        values = curves[separation]
        rho_uniform = np.interp(common_time, values["time_fs"], values["rho22"])
        csv_columns.append(rho_uniform)
        csv_header.append(f"rho22_full_qs_R{separation:g}nm")
        archive[f"time_fs_R{separation:g}nm"] = values["time_fs"]
        archive[f"rho22_R{separation:g}nm"] = values["rho22"]
        archive[f"W_R{separation:g}nm"] = values["W"]
        archive[f"mu_p_au_R{separation:g}nm"] = values["mu_p_au"]
        archive[f"incident_field_au_R{separation:g}nm"] = values[
            "incident_field_au"
        ]
        if digitized is not None:
            reference = digitized[separation]
            csv_columns.append(np.interp(common_time, reference.time_fs, reference.rho22))
            csv_header.append(f"rho22_article_digitized_R{separation:g}nm")
            archive[f"article_time_fs_R{separation:g}nm"] = reference.time_fs
            archive[f"article_rho22_R{separation:g}nm"] = reference.rho22
            archive[f"article_observed_R{separation:g}nm"] = reference.observed
    np.savetxt(
        run_dir / "curves.csv",
        np.column_stack(csv_columns),
        delimiter=",",
        header=",".join(csv_header),
        comments="",
    )
    np.savez_compressed(run_dir / "raw_curves.npz", **archive)

    elapsed = time.perf_counter() - started
    metadata = {
        "benchmark": {
            "article": "McMillan, Stella, Gruning, Phys. Rev. B 94, 125312 (2016)",
            "doi": PAPER_DOI,
            "arxiv": PAPER_ARXIV_URL,
            "target": "Fig. 4(c)",
            "article_pdf_used": None if article_pdf is None else str(article_pdf),
            "article_curve_comparison": (
                "textual_landmarks_only"
                if digitized is None
                else "automatic_250dpi_raster_digitization"
            ),
        },
        "model_scope": {
            "spatial_orders": 1,
            "spatial_interpretation": (
                "paper-matched dipole limit; this run validates material memory and "
                "Bloch coupling, not higher spatial multipoles"
            ),
            "material_poles": int(fit.strengths_au2.size),
            "material_interpretation": (
                "independent passive second-order Lorentz fit to the same Etchegoin "
                "sphere polarizability; not the paper's signed first-order PEOM table"
            ),
            "bloch_frame": "original carrier-resolved non-RWA W/Q/P equations",
            "rho22_definition": "rho22=(W+1)/2",
            "retardation": False,
        },
        "paper_parameters": {
            "mnp_radius_nm": 7.5,
            "mnp_size_provenance_note": (
                "The article imports a from Ref. 17; McMillan's thesis calls "
                "a=7.5 nm a diameter in prose, while its polarizability formula "
                "and Ref. 17 define a as the sphere radius. This run uses the "
                "formula parameter a=7.5 nm."
            ),
            "separations_nm": list(SEPARATIONS_NM),
            "bare_qd_dipole_e_nm": 0.65,
            "screened_qd_dipole_factor": 3.0 / 8.0,
            "exciton_energy_eV": EXCITON_ENERGY_EV,
            "population_lifetime_ns": 0.8,
            "coherence_lifetime_ns": 0.3,
            "pulse_cycles": PULSE_CYCLES,
            "pulse_area_pi": PULSE_AREA_PI,
            "pulse_duration_fs": duration_fs,
            "pulse_tau_p_fs": duration_fs / 30.0,
        },
        "etchegoin_material": {
            "drude_gamma_eV": 0.0729,
            "fit_window_eV": list(FIT_WINDOW_EV),
        },
        "passive_fit_audit": asdict(fit_audit),
        "solver_controls": {
            "method": "DOP853",
            "rtol": float(args.rtol),
            "atol": float(args.atol),
            "points_per_fastest_cycle": float(args.points_per_fastest_cycle),
            "max_allowed_channel_spectral_leakage": 0.01,
            "positivity_policy": "raise",
            "response_tail_policy": "ignore_by_design_at_article_endpoint_T",
        },
        "pulses": pulse_metadata,
        "solver_diagnostics": diagnostics,
        "comparison_metrics": metrics,
        "step_convergence": convergence,
        "material_pole_convergence": material_pole_convergence,
        "paper_text_landmarks": {
            "common_post_pulse_peak_rho22_approximately": 0.95,
            "post_pulse_peak_times_fs": {"R80": 23.0, "R20": 21.0, "R13": 20.0},
            "R13_low_or_final_rho22_approximately": 0.6,
        },
        "elapsed_seconds": float(elapsed),
        "claim_boundary": (
            "Agreement supports the causal material-pole plus non-RWA Bloch dynamics "
            "in the common N_spatial=1 quasistatic sphere limit. It is not an "
            "independent validation of the N_spatial>1 full-QS extension."
        ),
    }
    write_json(run_dir / "metadata.json", metadata)

    print(f"Output: {run_dir}")
    print(
        "Passive fit: "
        f"NRMS(alpha)={fit_audit.normalized_rms_alpha:.4%}, "
        f"max pointwise={fit_audit.max_pointwise_relative_alpha_error:.4%}"
    )
    for separation in SEPARATIONS_NM:
        values = metrics[f"R_{separation:g}_nm"]
        message = (
            f"R={separation:g} nm: post-18 fs peak="
            f"{values['post_18fs_peak_rho22']:.6f} at "
            f"{values['post_18fs_peak_time_fs']:.3f} fs, "
            f"rho22(T)={values['final_rho22']:.6f}"
        )
        comparison = values.get("digitized_comparison")
        if isinstance(comparison, dict):
            message += (
                f", raster RMSE(post-18 fs)="
                f"{comparison['rmse_rho22_post_18fs']:.6f}"
            )
        print(message)
    if convergence is not None:
        print(
            "R=13 nm step refinement: max |delta rho22|="
            f"{convergence['rho22_max_abs_difference']:.3e}"
        )
    if material_pole_convergence is not None:
        for pole_count, values in material_pole_convergence["variants"].items():
            print(
                f"R=13 nm, {pole_count} vs "
                f"{material_pole_convergence['base_material_poles']} material poles: "
                f"max |delta rho22|="
                f"{values['rho22_max_abs_difference_from_base']:.3e}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
