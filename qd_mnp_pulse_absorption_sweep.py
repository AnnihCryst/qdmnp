"""Нелинейный импульсный расчет поглощения системы КТ-МНЧ.
Скрипт решает временную задачу из ``qd_mnp_rational_fit.py`` для сетки
амплитуд и длительностей лазерного импульса. На выходе он сохраняет CSV и
график с двумя сечениями: интегральным ``W_abs/fluence`` и спектральным
``sigma_abs(omega_L)``, полученным из Фурье-компоненты временного отклика.
Работать с ним лучше после проверки слабополевого спектра в
``qd_mnp_linear_spectrum.py``. Физические параметры настраиваются аргументами
``--omega0-ev``, ``--gamma-dephasing-mev``, ``--d-debye``, ``--G``,
``--r-nm`` и ``--eps-m``. Для надежного спектрального сечения задавай
``--post-fs``: после импульса когерентность КТ должна успеть затухнуть.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qd_mnp_rational_fit import (
    AU_DIPOLE_C_M,
    AU_FIELD_V_M,
    AU_TIME_S,
    C_SI,
    GaussianPulse,
    HybridQDPlasmonModel,
    SCHEMA_VERSION,
    au_to_fs,
    eV_to_au,
    field_si_to_au,
    fs_to_au,
    params_to_physical_dict,
    timestamped_run_dir,
    write_json,
    epsilon_0,
)
from qd_mnp_params import make_params_with_overrides


def spectral_absorption_cross_section_cm2(
    result,
    pulse: GaussianPulse,
    eps_m: float,
    omega_eval_au: float | None = None,
) -> float:
    """Fourier-domain absorption cross section at one angular frequency.

    This follows the pulsed-response convention used by Shah et al.:

        alpha(omega) = int exp(-i omega t) mu(t) dt
                       / [eps_m int exp(-i omega t) E(t) dt]
        sigma_abs(omega) = k / eps0 * Im alpha(omega)

    The solver stores real fields written as cos(omega t). With this real-time
    convention the positive-frequency response compatible with p = alpha E is
    selected by exp(+i omega t); using exp(-i omega t) would return the complex
    conjugate and reverse the sign of Im alpha. The time integrals are evaluated
    in atomic units. The ratio is an atomic polarizability and is converted to
    SI before applying the cross-section formula.
    """
    omega_au = pulse.omegaL_au if omega_eval_au is None else float(omega_eval_au)
    phase = np.exp(1j * omega_au * result.t_au)

    mu_omega_au = np.trapezoid(result.mu_total_au * phase, result.t_au)
    e_omega_au = np.trapezoid(pulse.field(result.t_au) * phase, result.t_au)
    if abs(e_omega_au) < 1e-30:
        return np.nan

    alpha_au = mu_omega_au / (eps_m * e_omega_au)
    alpha_si = alpha_au * (AU_DIPOLE_C_M / AU_FIELD_V_M)

    omega_si = omega_au / AU_TIME_S
    n_m = np.sqrt(eps_m)
    k_si = n_m * omega_si / C_SI
    sigma_m2 = (k_si / epsilon_0) * alpha_si.imag
    return float(sigma_m2 * 1e4)


def isolated_qd_pulse_area(pulse: GaussianPulse, d_au: float) -> float:
    """Envelope pulse area for an isolated resonant two-level QD."""
    return float(d_au * pulse.E0_au * np.sqrt(2.0 * np.pi) * pulse.sigma_t_au)


def bare_mnp_spectral_cross_section_cm2(
    model: HybridQDPlasmonModel,
    omega_ev: float,
    eps_m: float,
) -> float:
    """Bare-MNP spectral cross section from the fitted polarizability."""
    omega_au = float(eV_to_au(omega_ev))
    alpha_mnp_au = model.C * model.alpha_from_fit(np.array([omega_ev]))[0] / eps_m
    alpha_mnp_si = alpha_mnp_au * (AU_DIPOLE_C_M / AU_FIELD_V_M)
    omega_si = omega_au / AU_TIME_S
    n_m = np.sqrt(eps_m)
    k_si = n_m * omega_si / C_SI
    sigma_m2 = (k_si / epsilon_0) * alpha_mnp_si.imag
    return float(sigma_m2 * 1e4)


def compute_sweep(
    *,
    tau_values_fs: list[float],
    e0_values_v_m: np.ndarray,
    omega_l_ev: float,
    n_modes: int,
    fit_window_ev: tuple[float, float],
    weight_center_ev: float | None,
    weight_sigma_ev: float | None,
    method: str,
    rtol: float,
    atol: float,
    pre_sigma: float | None,
    post_fs: float | None,
    c_nm: float | None,
    a_nm: float | None,
    r_nm: float | None,
    g_factor: float | None,
    eps_m: float | None,
    d_debye: float | None,
    omega0_ev: float | None,
    gamma_population_mev: float | None,
    gamma_dephasing_mev: float | None,
) -> tuple[list[dict[str, float]], list[dict[str, object]], object]:
    params = make_params_with_overrides(
        c_nm=c_nm,
        a_nm=a_nm,
        r_nm=r_nm,
        g_factor=g_factor,
        eps_m=eps_m,
        d_debye=d_debye,
        omega0_ev=omega0_ev,
        gamma_population_mev=gamma_population_mev,
        gamma_dephasing_mev=gamma_dephasing_mev,
    )
    model = HybridQDPlasmonModel(
        params,
        orientation="long",
        n_modes=n_modes,
        fit_window_eV=fit_window_ev,
        weight_center_eV=weight_center_ev,
        weight_sigma_eV=weight_sigma_ev,
        alpha_objective_weight=1.0,
        inv_alpha_objective_weight=1.2,
        verbose=True,
    )

    rows: list[dict[str, float]] = []
    traces: list[dict[str, object]] = []
    omega_l_au = float(eV_to_au(omega_l_ev))
    sigma_bare_mnp_cm2 = bare_mnp_spectral_cross_section_cm2(model, omega_l_ev, params.eps_m)

    for tau_index, tau_fs in enumerate(tau_values_fs):
        for e0_index, e0_v_m in enumerate(e0_values_v_m):
            pulse = GaussianPulse(
                E0_au=float(field_si_to_au(e0_v_m)),
                omegaL_au=omega_l_au,
                tau_au=float(fs_to_au(tau_fs)),
                tau_kind="fwhm_intensity",
            )
            t_span_au = None
            if pre_sigma is not None and post_fs is not None and post_fs > 0.0:
                t_span_au = (-float(pre_sigma) * pulse.sigma_t_au, float(fs_to_au(post_fs)))

            result = model.solve(pulse, method=method, rtol=rtol, atol=atol, t_span_au=t_span_au)
            sigma_spectral_cm2 = spectral_absorption_cross_section_cm2(
                result,
                pulse,
                eps_m=params.eps_m,
                omega_eval_au=omega_l_au,
            )
            W, Q, P = result.y[2 * n_modes : 2 * n_modes + 3]
            traces.append(
                {
                    "tau_index": int(tau_index),
                    "e0_index": int(e0_index),
                    "t_au": result.t_au,
                    "t_fs": au_to_fs(result.t_au),
                    "e_field_au": pulse.field(result.t_au),
                    "mu_p_au": result.mu_p_au,
                    "mu_d_au": result.mu_d_au,
                    "mu_total_au": result.mu_total_au,
                    "mu_dot_total_au": result.mu_dot_total_au,
                    "W": W,
                    "Q": Q,
                    "P": P,
                }
            )

            rows.append(
                {
                    "tau_fwhm_intensity_fs": float(tau_fs),
                    "e0_v_m": float(e0_v_m),
                    "omega_l_ev": float(omega_l_ev),
                    "peak_intensity_w_cm2": float(result.peak_intensity_w_cm2),
                    "fluence_j_cm2": float(result.fluence_j_cm2),
                    "pulse_area_isolated_qd": isolated_qd_pulse_area(pulse, params.d_au),
                    "sigma_energy_cm2": float(result.sigma_abs_cm2),
                    "sigma_spectral_cm2": float(sigma_spectral_cm2),
                    "sigma_bare_mnp_cm2": float(sigma_bare_mnp_cm2),
                    "sigma_spectral_minus_bare_cm2": float(sigma_spectral_cm2 - sigma_bare_mnp_cm2),
                    "absorbed_energy_j": float(result.absorbed_energy_j),
                    "solver_n_steps": float(len(result.t_au)),
                    "solver_success": float(bool(result.solve_ivp_result.success)),
                }
            )

    return rows, traces, params


def write_csv(rows: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_sweeps(rows: list[dict[str, float]], x_axis: str, output_path: Path | None, show: bool) -> None:
    tau_values = sorted({row["tau_fwhm_intensity_fs"] for row in rows})
    x_key = {
        "fluence": "fluence_j_cm2",
        "intensity": "peak_intensity_w_cm2",
        "pulse_area": "pulse_area_isolated_qd",
    }[x_axis]
    x_label = {
        "fluence": r"Fluence, J/cm$^2$",
        "intensity": r"Peak intensity, W/cm$^2$",
        "pulse_area": r"Isolated-QD pulse area",
    }[x_axis]

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    sigma_bare = rows[0].get("sigma_bare_mnp_cm2")
    for tau_fs in tau_values:
        group = [row for row in rows if row["tau_fwhm_intensity_fs"] == tau_fs]
        group.sort(key=lambda row: row[x_key])
        x = np.array([row[x_key] for row in group])
        sigma_energy = np.array([row["sigma_energy_cm2"] for row in group])
        sigma_spectral = np.array([row["sigma_spectral_cm2"] for row in group])

        label = f"{tau_fs:g} fs"
        axes[0].plot(x, sigma_energy, marker="o", ms=4, lw=1.8, label=label)
        axes[1].plot(x, sigma_spectral, marker="s", ms=4, lw=1.8, label=label)

    axes[0].set_ylabel(r"$\sigma_E = W_{abs}/\mathcal{F}$, cm$^2$")
    axes[1].set_ylabel(r"$\sigma_{abs}(\omega_L)$, cm$^2$")
    axes[1].set_xlabel(x_label)
    if sigma_bare is not None:
        axes[1].axhline(
            sigma_bare,
            color="0.25",
            lw=1.4,
            ls=":",
            label="bare MNP",
        )

    if x_axis in {"fluence", "intensity"}:
        axes[1].set_xscale("log")

    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(title="Pulse FWHM")

    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)


def _summary_grid(rows: list[dict[str, float]], key: str, n_tau: int, n_e0: int) -> np.ndarray:
    return np.asarray([row[key] for row in rows]).reshape(n_tau, n_e0)


def _concatenate_traces(traces: list[dict[str, object]], key: str) -> np.ndarray:
    if not traces:
        return np.asarray([], dtype=float)
    return np.concatenate([np.asarray(trace[key]) for trace in traces])


def save_artifact_run(
    *,
    rows: list[dict[str, float]],
    traces: list[dict[str, object]],
    params,
    args: argparse.Namespace,
    e0_values_v_m: np.ndarray,
    run_dir: Path,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    n_tau = len(args.tau_fs)
    n_e0 = len(e0_values_v_m)
    tau_values = np.asarray(args.tau_fs, dtype=float)

    trace_lengths = np.asarray([len(np.asarray(trace["t_au"])) for trace in traces], dtype=np.int64)
    trace_offsets = np.zeros_like(trace_lengths)
    if len(trace_lengths) > 1:
        trace_offsets[1:] = np.cumsum(trace_lengths[:-1])

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "script": "qd_mnp_pulse_absorption_sweep.py",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "physical": params_to_physical_dict(params, orientation="long"),
        "fit": {
            "n_modes": int(args.n_modes),
            "fit_window_ev": [float(args.fit_min_ev), float(args.fit_max_ev)],
            "weight_center_ev": None if args.weight_center_ev is None else float(args.weight_center_ev),
            "weight_sigma_ev": None if args.weight_sigma_ev is None else float(args.weight_sigma_ev),
            "alpha_objective_weight": 1.0,
            "inv_alpha_objective_weight": 1.2,
        },
        "sweep": {
            "tau_fs": [float(x) for x in tau_values],
            "e0_values_v_m": [float(x) for x in e0_values_v_m],
            "omega_l_ev": float(args.omega_l_ev),
            "pre_sigma": None if args.pre_sigma is None else float(args.pre_sigma),
            "post_fs": None if args.post_fs is None else float(args.post_fs),
            "x_axis": args.x_axis,
        },
        "solver": {
            "method": args.method,
            "rtol": float(args.rtol),
            "atol": float(args.atol),
        },
    }

    np.savez_compressed(
        run_dir / "data.npz",
        tau_fs_grid=np.repeat(tau_values[:, None], n_e0, axis=1),
        e0_v_m_grid=np.repeat(e0_values_v_m[None, :], n_tau, axis=0),
        peak_intensity_w_cm2=_summary_grid(rows, "peak_intensity_w_cm2", n_tau, n_e0),
        fluence_j_cm2=_summary_grid(rows, "fluence_j_cm2", n_tau, n_e0),
        pulse_area_isolated_qd=_summary_grid(rows, "pulse_area_isolated_qd", n_tau, n_e0),
        sigma_energy_cm2=_summary_grid(rows, "sigma_energy_cm2", n_tau, n_e0),
        sigma_spectral_cm2=_summary_grid(rows, "sigma_spectral_cm2", n_tau, n_e0),
        sigma_bare_mnp_cm2=_summary_grid(rows, "sigma_bare_mnp_cm2", n_tau, n_e0),
        sigma_spectral_minus_bare_cm2=_summary_grid(rows, "sigma_spectral_minus_bare_cm2", n_tau, n_e0),
        absorbed_energy_j=_summary_grid(rows, "absorbed_energy_j", n_tau, n_e0),
        solver_n_steps=_summary_grid(rows, "solver_n_steps", n_tau, n_e0),
        solver_success=_summary_grid(rows, "solver_success", n_tau, n_e0).astype(bool),
        trace_offsets=trace_offsets,
        trace_lengths=trace_lengths,
        trace_tau_index=np.asarray([trace["tau_index"] for trace in traces], dtype=np.int64),
        trace_e0_index=np.asarray([trace["e0_index"] for trace in traces], dtype=np.int64),
        trace_t_au=_concatenate_traces(traces, "t_au"),
        trace_t_fs=_concatenate_traces(traces, "t_fs"),
        trace_e_field_au=_concatenate_traces(traces, "e_field_au"),
        trace_mu_p_au=_concatenate_traces(traces, "mu_p_au"),
        trace_mu_d_au=_concatenate_traces(traces, "mu_d_au"),
        trace_mu_total_au=_concatenate_traces(traces, "mu_total_au"),
        trace_mu_dot_total_au=_concatenate_traces(traces, "mu_dot_total_au"),
        trace_W=_concatenate_traces(traces, "W"),
        trace_Q=_concatenate_traces(traces, "Q"),
        trace_P=_concatenate_traces(traces, "P"),
    )
    write_json(run_dir / "params.json", metadata)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare integral pulse and Fourier spectral absorption cross sections."
    )
    parser.add_argument("--tau-fs", nargs="+", type=float, default=[5.0, 10.0, 20.0])
    parser.add_argument("--e0-min", type=float, default=1e5, help="Minimum field amplitude, V/m.")
    parser.add_argument("--e0-max", type=float, default=1e9, help="Maximum field amplitude, V/m.")
    parser.add_argument("--points", type=int, default=18)
    parser.add_argument("--omega-l-ev", type=float, default=2.042)
    parser.add_argument("--n-modes", type=int, default=4)
    parser.add_argument("--fit-min-ev", type=float, default=0.8)
    parser.add_argument("--fit-max-ev", type=float, default=3.0)
    parser.add_argument("--weight-center-ev", type=float, default=2.35)
    parser.add_argument("--weight-sigma-ev", type=float, default=0.30)
    parser.add_argument("--c-nm", type=float, default=None)
    parser.add_argument("--a-nm", type=float, default=None)
    parser.add_argument("--r-nm", type=float, default=None)
    parser.add_argument("--G", dest="g_factor", type=float, default=None)
    parser.add_argument("--eps-m", type=float, default=None)
    parser.add_argument("--d-debye", type=float, default=None)
    parser.add_argument("--omega0-ev", type=float, default=None)
    parser.add_argument("--gamma-population-mev", type=float, default=None)
    parser.add_argument("--gamma-dephasing-mev", type=float, default=None)
    parser.add_argument("--method", choices=["Radau", "BDF", "LSODA", "RK45", "DOP853"], default="Radau")
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument(
        "--pre-sigma",
        type=float,
        default=8.0,
        help="Start time in pulse sigmas when --post-fs is positive.",
    )
    parser.add_argument(
        "--post-fs",
        type=float,
        default=0.0,
        help="If positive, integrate until this absolute post-pulse time in fs. "
        "Use e.g. 1500-2000 fs for publication-quality Fourier cross sections.",
    )
    parser.add_argument("--x-axis", choices=["fluence", "intensity", "pulse_area"], default="fluence")
    parser.add_argument("--csv", type=Path, default=Path("results/absorption_sections_sweep.csv"))
    parser.add_argument("--figure", type=Path, default=Path("results/absorption_sections_sweep.png"))
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--no-save-figure", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    e0_values = np.logspace(np.log10(args.e0_min), np.log10(args.e0_max), args.points)
    rows, traces, params = compute_sweep(
        tau_values_fs=args.tau_fs,
        e0_values_v_m=e0_values,
        omega_l_ev=args.omega_l_ev,
        n_modes=args.n_modes,
        fit_window_ev=(args.fit_min_ev, args.fit_max_ev),
        weight_center_ev=args.weight_center_ev,
        weight_sigma_ev=args.weight_sigma_ev,
        method=args.method,
        rtol=args.rtol,
        atol=args.atol,
        pre_sigma=args.pre_sigma,
        post_fs=args.post_fs,
        c_nm=args.c_nm,
        a_nm=args.a_nm,
        r_nm=args.r_nm,
        g_factor=args.g_factor,
        eps_m=args.eps_m,
        d_debye=args.d_debye,
        omega0_ev=args.omega0_ev,
        gamma_population_mev=args.gamma_population_mev,
        gamma_dephasing_mev=args.gamma_dephasing_mev,
    )
    write_csv(rows, args.csv)
    print(f"Wrote {len(rows)} rows to {args.csv}")

    run_dir = args.run_dir if args.run_dir is not None else timestamped_run_dir("results/pulse_absorption_sweeps")
    run_dir = Path(run_dir)
    save_artifact_run(rows=rows, traces=traces, params=params, args=args, e0_values_v_m=e0_values, run_dir=run_dir)
    plot_sweeps(rows, args.x_axis, run_dir / "absorption_sweep.png", show=False)
    print(f"Wrote pulse-sweep artifact run to {run_dir}")

    figure_path = None if args.no_save_figure else args.figure
    plot_sweeps(rows, args.x_axis, figure_path, show=not args.no_show)


if __name__ == "__main__":
    main()
