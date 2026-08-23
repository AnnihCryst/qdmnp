"""Слабополевой линейный спектр поглощения системы КТ-МНЧ.

Скрипт не интегрирует импульсную динамику. Он берет подогнанную поляризуемость
МНЧ из ``qd_mnp_rational_fit.py`` и линейный отклик двухуровневой КТ,
после чего считает спектральное сечение связанной системы, голой МНЧ и их
отношение.

Запускай его перед нелинейными расчетами, чтобы понять, есть ли около рабочей
энергии Фано-провал. Главные настройки: ``--omega0-ev`` задает расстройку КТ,
``--gamma-dephasing-mev`` - ширину линии, ``--d-debye`` - силу осциллятора,
``--G`` и ``--r-nm`` - диполь-дипольную связь, ``--eps-m`` - среду.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qd_mnp_rational_fit import (
    AU_DIPOLE_C_M,
    AU_FIELD_V_M,
    AU_TIME_S,
    C_SI,
    HybridQDPlasmonModel,
    eV_to_au,
    epsilon_0,
)
from qd_mnp_params import make_params_with_overrides


def qd_linear_polarizability_au(omega_au: np.ndarray, d_au: float, omega0_au: float, gamma2_au: float) -> np.ndarray:
    """Linear two-level QD dipole response beta = mu_d / E_eff.

    The expression follows from the weak-field Bloch equations with W=-1 and
    the exp(-i omega t) frequency convention.
    """
    return 2.0 * d_au**2 * omega0_au / (omega0_au**2 + (gamma2_au - 1j * omega_au) ** 2)


def cross_section_cm2(alpha_au: np.ndarray, omega_au: np.ndarray, eps_m: float) -> np.ndarray:
    alpha_si = alpha_au * (AU_DIPOLE_C_M / AU_FIELD_V_M)
    omega_si = omega_au / AU_TIME_S
    k_si = np.sqrt(eps_m) * omega_si / C_SI
    return (k_si / epsilon_0) * alpha_si.imag * 1e4


def linear_coupled_alpha_au(
    model: HybridQDPlasmonModel,
    energies_ev: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return effective coupled, bare-MNP, isolated-QD, and coupled-minus-bare alphas.

    All returned quantities are in the same effective convention used for
    spectral cross sections: alpha_eff = mu_total / (eps_m E_inc).
    """
    p = model.params
    omega = eV_to_au(energies_ev)

    alpha_mnp_dimless = model.alpha_from_fit(energies_ev)
    alpha_p = model.C * alpha_mnp_dimless
    beta_qd = qd_linear_polarizability_au(omega, p.d_au, p.omega0_au, p.Gamma_au)

    denominator = 1.0 - (model.J**2) * alpha_p * beta_qd
    mu_p_over_e = alpha_p * (1.0 + model.J * beta_qd) / denominator
    mu_d_over_e = beta_qd * (1.0 + model.J * alpha_p) / denominator

    alpha_coupled_eff = (mu_p_over_e + mu_d_over_e) / p.eps_m
    alpha_bare_mnp_eff = alpha_p / p.eps_m
    alpha_isolated_qd_eff = beta_qd / p.eps_m
    return (
        alpha_coupled_eff,
        alpha_bare_mnp_eff,
        alpha_isolated_qd_eff,
        alpha_coupled_eff - alpha_bare_mnp_eff,
    )


def compute_spectrum(
    *,
    energy_min_ev: float,
    energy_max_ev: float,
    points: int,
    n_modes: int,
    fit_window_ev: tuple[float, float],
    weight_center_ev: float | None,
    weight_sigma_ev: float | None,
    c_nm: float | None,
    a_nm: float | None,
    r_nm: float | None,
    g_factor: float | None,
    eps_m: float | None,
    d_debye: float | None,
    omega0_ev: float | None,
    gamma_population_mev: float | None,
    gamma_dephasing_mev: float | None,
) -> list[dict[str, float]]:
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

    energies = np.linspace(energy_min_ev, energy_max_ev, points)
    omega = eV_to_au(energies)
    alpha_coupled, alpha_bare, alpha_qd, alpha_delta = linear_coupled_alpha_au(model, energies)

    sigma_coupled = cross_section_cm2(alpha_coupled, omega, params.eps_m)
    sigma_bare = cross_section_cm2(alpha_bare, omega, params.eps_m)
    sigma_qd = cross_section_cm2(alpha_qd, omega, params.eps_m)
    sigma_delta = cross_section_cm2(alpha_delta, omega, params.eps_m)

    rows: list[dict[str, float]] = []
    for e, sc, sb, sq, sd in zip(energies, sigma_coupled, sigma_bare, sigma_qd, sigma_delta):
        rows.append(
            {
                "energy_ev": float(e),
                "sigma_coupled_cm2": float(sc),
                "sigma_bare_mnp_cm2": float(sb),
                "sigma_isolated_qd_cm2": float(sq),
                "sigma_coupled_minus_bare_cm2": float(sd),
                "ratio_coupled_to_bare": float(sc / sb) if sb != 0.0 else np.nan,
            }
        )
    return rows


def write_csv(rows: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_spectrum(rows: list[dict[str, float]], output_path: Path | None, show: bool) -> None:
    energy = np.array([row["energy_ev"] for row in rows])
    coupled = np.array([row["sigma_coupled_cm2"] for row in rows])
    bare = np.array([row["sigma_bare_mnp_cm2"] for row in rows])
    delta = np.array([row["sigma_coupled_minus_bare_cm2"] for row in rows])
    ratio = np.array([row["ratio_coupled_to_bare"] for row in rows])

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(energy, bare, lw=2.0, label="bare MNP")
    axes[0].plot(energy, coupled, lw=2.0, label="coupled QD+MNP")
    axes[0].set_ylabel(r"$\sigma_{abs}(\omega)$, cm$^2$")
    axes[0].legend()

    axes[1].axhline(0.0, color="0.25", lw=1.2)
    axes[1].plot(energy, delta, lw=2.0, label="coupled - bare")
    axes[1].set_ylabel(r"$\Delta\sigma$, cm$^2$")
    axes[1].set_xlabel("Photon energy, eV")

    ax_ratio = axes[1].twinx()
    ax_ratio.plot(energy, ratio, color="tab:green", lw=1.6, ls="--", label="coupled / bare")
    ax_ratio.axhline(1.0, color="tab:green", lw=1.0, ls=":")
    ax_ratio.set_ylabel("coupled / bare")

    for ax in axes:
        ax.grid(True, alpha=0.3)
    lines, labels = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax_ratio.get_legend_handles_labels()
    axes[1].legend(lines + lines2, labels + labels2, loc="best")

    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)


def print_diagnostics(rows: list[dict[str, float]], target_ev: float) -> None:
    closest = min(rows, key=lambda row: abs(row["energy_ev"] - target_ev))
    minimum = min(rows, key=lambda row: row["ratio_coupled_to_bare"])
    maximum = max(rows, key=lambda row: row["ratio_coupled_to_bare"])

    print("\n=== Linear-spectrum diagnostics ===")
    print(
        f"At {closest['energy_ev']:.6f} eV: "
        f"coupled={closest['sigma_coupled_cm2']:.6e} cm^2, "
        f"bare={closest['sigma_bare_mnp_cm2']:.6e} cm^2, "
        f"ratio={closest['ratio_coupled_to_bare']:.6g}, "
        f"delta={closest['sigma_coupled_minus_bare_cm2']:.6e} cm^2"
    )
    print(
        f"Deepest relative dip: {minimum['energy_ev']:.6f} eV, "
        f"ratio={minimum['ratio_coupled_to_bare']:.6g}, "
        f"delta={minimum['sigma_coupled_minus_bare_cm2']:.6e} cm^2"
    )
    print(
        f"Largest relative enhancement: {maximum['energy_ev']:.6f} eV, "
        f"ratio={maximum['ratio_coupled_to_bare']:.6g}, "
        f"delta={maximum['sigma_coupled_minus_bare_cm2']:.6e} cm^2"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weak-field linear QD-MNP absorption spectrum.")
    parser.add_argument("--energy-min-ev", type=float, default=1.6)
    parser.add_argument("--energy-max-ev", type=float, default=2.8)
    parser.add_argument("--points", type=int, default=1000)
    parser.add_argument("--target-ev", type=float, default=2.042)
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
    parser.add_argument("--csv", type=Path, default=Path("results/linear_spectrum.csv"))
    parser.add_argument("--figure", type=Path, default=Path("results/linear_spectrum.png"))
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--no-save-figure", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = compute_spectrum(
        energy_min_ev=args.energy_min_ev,
        energy_max_ev=args.energy_max_ev,
        points=args.points,
        n_modes=args.n_modes,
        fit_window_ev=(args.fit_min_ev, args.fit_max_ev),
        weight_center_ev=args.weight_center_ev,
        weight_sigma_ev=args.weight_sigma_ev,
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
    print_diagnostics(rows, args.target_ev)
    figure_path = None if args.no_save_figure else args.figure
    plot_spectrum(rows, figure_path, show=not args.no_show)


if __name__ == "__main__":
    main()
