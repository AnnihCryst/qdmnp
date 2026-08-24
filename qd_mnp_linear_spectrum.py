"""Слабополевой линейный спектр экстинкции системы КТ-МНЧ.

Скрипт не интегрирует импульсную динамику. Он берет подогнанную поляризуемость
МНЧ из ``qd_mnp_rational_fit.py`` и линейный отклик двухуровневой КТ,
после чего разделяет дипольные сечения экстинкции, рассеяния и материального
поглощения связанной системы, голой МНЧ и изолированной КТ.

Запускай его перед нелинейными расчетами, чтобы понять, есть ли около рабочей
энергии Фано-провал. Главные настройки: ``--omega0-ev`` задает расстройку КТ,
``--gamma2-coherence-mev`` - полную ширину когерентности, ``--d-debye`` - силу осциллятора,
``--G`` и ``--r-nm`` - диполь-дипольную связь, ``--eps-m`` - среду.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np

from qd_mnp_rational_fit import (
    HybridQDPlasmonModel,
    SCHEMA_VERSION,
    dipole_cross_sections_cm2,
    eV_to_au,
)
from qd_mnp_params import make_params_with_overrides


def qd_linear_polarizability_au(omega_au: np.ndarray, d_au: float, omega0_au: float, gamma2_au: float) -> np.ndarray:
    """Linear two-level QD dipole response beta = mu_d / E_eff.

    The expression follows from the weak-field Bloch equations with W=-1 and
    the exp(-i omega t) frequency convention.
    """
    return 2.0 * d_au**2 * omega0_au / (omega0_au**2 + (gamma2_au - 1j * omega_au) ** 2)


def extinction_cross_section_cm2(alpha_au: np.ndarray, omega_au: np.ndarray, eps_m: float) -> np.ndarray:
    """Dipole extinction k*Im(alpha_eff)/eps0 in cm^2."""
    return dipole_cross_sections_cm2(alpha_au, omega_au, eps_m).extinction_cm2


def scattering_cross_section_cm2(alpha_au: np.ndarray, omega_au: np.ndarray, eps_m: float) -> np.ndarray:
    """Dipole scattering in cm^2 for alpha_eff=mu/(eps_m E_inc)."""
    return dipole_cross_sections_cm2(alpha_au, omega_au, eps_m).scattering_cm2


def absorption_cross_section_cm2(alpha_au: np.ndarray, omega_au: np.ndarray, eps_m: float) -> np.ndarray:
    """Material absorption sigma_ext-sigma_sca; negative values are not clipped."""
    return dipole_cross_sections_cm2(alpha_au, omega_au, eps_m).absorption_cm2


def cross_section_cm2(alpha_au: np.ndarray, omega_au: np.ndarray, eps_m: float) -> np.ndarray:
    """Schema-1 compatibility alias; the old function always returned extinction."""
    warnings.warn(
        "cross_section_cm2() was ambiguous and returned extinction; use "
        "extinction_cross_section_cm2().",
        DeprecationWarning,
        stacklevel=2,
    )
    return extinction_cross_section_cm2(alpha_au, omega_au, eps_m)


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
    qd_radius_nm: float | None = None,
    g_factor: float | None,
    eps_m: float | None,
    d_debye: float | None,
    omega0_ev: float | None,
    gamma_population_mev: float | None,
    gamma_dephasing_mev: float | None = None,
    gamma2_coherence_mev: float | None = None,
) -> list[dict[str, float]]:
    if points < 2:
        raise ValueError("points must be at least 2.")
    if not (np.isfinite(energy_min_ev) and np.isfinite(energy_max_ev) and 0.0 < energy_min_ev < energy_max_ev):
        raise ValueError("Energy bounds must be finite, positive and strictly increasing.")
    params = make_params_with_overrides(
        c_nm=c_nm,
        a_nm=a_nm,
        r_nm=r_nm,
        qd_radius_nm=qd_radius_nm,
        g_factor=g_factor,
        eps_m=eps_m,
        d_debye=d_debye,
        omega0_ev=omega0_ev,
        gamma_population_mev=gamma_population_mev,
        gamma_dephasing_mev=gamma_dephasing_mev,
        gamma2_coherence_mev=gamma2_coherence_mev,
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
    alpha_coupled, alpha_bare, alpha_qd, _ = linear_coupled_alpha_au(model, energies)

    coupled = dipole_cross_sections_cm2(alpha_coupled, omega, params.eps_m)
    bare = dipole_cross_sections_cm2(alpha_bare, omega, params.eps_m)
    isolated_qd = dipole_cross_sections_cm2(alpha_qd, omega, params.eps_m)

    rows: list[dict[str, float]] = []
    for idx, e in enumerate(energies):
        ext_c = float(coupled.extinction_cm2[idx])
        sca_c = float(coupled.scattering_cm2[idx])
        abs_c = float(coupled.absorption_cm2[idx])
        ext_b = float(bare.extinction_cm2[idx])
        sca_b = float(bare.scattering_cm2[idx])
        abs_b = float(bare.absorption_cm2[idx])
        ext_q = float(isolated_qd.extinction_cm2[idx])
        sca_q = float(isolated_qd.scattering_cm2[idx])
        abs_q = float(isolated_qd.absorption_cm2[idx])
        ratio_ext = float(ext_c / ext_b) if ext_b != 0.0 else np.nan
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "energy_ev": float(e),
                "linearized_ground_state_stable": bool(model.linear_stability.stable),
                "linearized_ground_state_spectral_abscissa_au": float(
                    model.linear_stability.spectral_abscissa_au
                ),
                "sigma_ext_coupled_cm2": ext_c,
                "sigma_sca_coupled_cm2": sca_c,
                "sigma_abs_coupled_cm2": abs_c,
                "sigma_ext_bare_mnp_cm2": ext_b,
                "sigma_sca_bare_mnp_cm2": sca_b,
                "sigma_abs_bare_mnp_cm2": abs_b,
                "sigma_ext_isolated_qd_cm2": ext_q,
                "sigma_sca_isolated_qd_cm2": sca_q,
                "sigma_abs_isolated_qd_cm2": abs_q,
                "delta_sigma_ext_cm2": ext_c - ext_b,
                "delta_sigma_sca_cm2": sca_c - sca_b,
                "delta_sigma_abs_cm2": abs_c - abs_b,
                "ratio_ext_coupled_to_bare": ratio_ext,
                # Schema-1 compatibility aliases: these all meant extinction.
                "sigma_coupled_cm2": ext_c,
                "sigma_bare_mnp_cm2": ext_b,
                "sigma_isolated_qd_cm2": ext_q,
                "sigma_coupled_minus_bare_cm2": ext_c - ext_b,
                "ratio_coupled_to_bare": ratio_ext,
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
    coupled = np.array([row["sigma_ext_coupled_cm2"] for row in rows])
    bare = np.array([row["sigma_ext_bare_mnp_cm2"] for row in rows])
    delta = np.array([row["delta_sigma_ext_cm2"] for row in rows])
    ratio = np.array([row["ratio_ext_coupled_to_bare"] for row in rows])

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(energy, bare, lw=2.0, label="bare MNP")
    axes[0].plot(energy, coupled, lw=2.0, label="coupled QD+MNP")
    axes[0].set_ylabel(r"$\sigma_{ext}(\omega)$, cm$^2$")
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
    minimum = min(rows, key=lambda row: row["ratio_ext_coupled_to_bare"])
    maximum = max(rows, key=lambda row: row["ratio_ext_coupled_to_bare"])

    print("\n=== Linear-spectrum diagnostics ===")
    print(
        f"At {closest['energy_ev']:.6f} eV: "
        f"ext_coupled={closest['sigma_ext_coupled_cm2']:.6e} cm^2, "
        f"ext_bare={closest['sigma_ext_bare_mnp_cm2']:.6e} cm^2, "
        f"ratio={closest['ratio_ext_coupled_to_bare']:.6g}, "
        f"delta_ext={closest['delta_sigma_ext_cm2']:.6e} cm^2"
    )
    print(
        f"Deepest relative dip: {minimum['energy_ev']:.6f} eV, "
        f"ratio={minimum['ratio_ext_coupled_to_bare']:.6g}, "
        f"delta_ext={minimum['delta_sigma_ext_cm2']:.6e} cm^2"
    )
    print(
        f"Largest relative enhancement: {maximum['energy_ev']:.6f} eV, "
        f"ratio={maximum['ratio_ext_coupled_to_bare']:.6g}, "
        f"delta_ext={maximum['delta_sigma_ext_cm2']:.6e} cm^2"
    )

    min_abs = min(row["sigma_abs_coupled_cm2"] for row in rows)
    if min_abs < 0.0:
        print(
            "WARNING: sigma_abs=sigma_ext-sigma_sca becomes negative "
            f"(minimum {min_abs:.6e} cm^2). The effective dipole response fails the "
            "passive optical-theorem partition under these assumptions; a missing "
            "radiation-reaction correction or fit error must be excluded before "
            "material absorption is interpreted quantitatively."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weak-field linear QD-MNP dipole extinction spectrum.")
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
    parser.add_argument("--qd-radius-nm", type=float, default=None)
    parser.add_argument("--G", dest="g_factor", type=float, default=None)
    parser.add_argument("--eps-m", type=float, default=None)
    parser.add_argument("--d-debye", type=float, default=None)
    parser.add_argument("--omega0-ev", type=float, default=None)
    parser.add_argument("--gamma-population-mev", type=float, default=None)
    parser.add_argument(
        "--gamma2-coherence-mev",
        dest="gamma2_coherence_mev",
        metavar="GAMMA2_MEV",
        type=float,
        default=None,
        help="Total coherence HWHM hbar*Gamma2 in meV; requires Gamma2 >= gamma1/2.",
    )
    parser.add_argument(
        "--gamma-dephasing-mev",
        dest="gamma_dephasing_mev",
        metavar="GAMMA2_MEV",
        type=float,
        default=None,
        help="Deprecated alias for --gamma2-coherence-mev.",
    )
    parser.add_argument("--csv", type=Path, default=Path("results/linear_spectrum.csv"))
    parser.add_argument("--figure", type=Path, default=Path("results/linear_spectrum.png"))
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--no-save-figure", action="store_true")
    args = parser.parse_args()
    if (
        args.gamma2_coherence_mev is not None
        and args.gamma_dephasing_mev is not None
        and args.gamma2_coherence_mev != args.gamma_dephasing_mev
    ):
        parser.error("--gamma2-coherence-mev and --gamma-dephasing-mev must agree when both are supplied.")
    if args.gamma2_coherence_mev is None:
        args.gamma2_coherence_mev = args.gamma_dephasing_mev
    return args


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
        qd_radius_nm=args.qd_radius_nm,
        g_factor=args.g_factor,
        eps_m=args.eps_m,
        d_debye=args.d_debye,
        omega0_ev=args.omega0_ev,
        gamma_population_mev=args.gamma_population_mev,
        gamma2_coherence_mev=args.gamma2_coherence_mev,
    )
    write_csv(rows, args.csv)
    print(f"Wrote {len(rows)} rows to {args.csv}")
    print_diagnostics(rows, args.target_ev)
    figure_path = None if args.no_save_figure else args.figure
    plot_spectrum(rows, figure_path, show=not args.no_show)


if __name__ == "__main__":
    main()
