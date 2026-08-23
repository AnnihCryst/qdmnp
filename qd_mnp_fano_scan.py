"""Поиск параметров, дающих слабополевой Фано-провал в спектре КТ-МНЧ.

Скрипт перебирает энергию перехода КТ, ширину дефазировки, переходный дипольный
момент и эффективный фактор связи ``G``. Он ранжирует наборы параметров по
тому, насколько сильно связанная система поглощает меньше голой МНЧ около
``--target-ev``.

Это диагностический инструмент, а не самостоятельное физическое доказательство.
Большие значения ``G`` надо понимать как эффективное усиление связи
(ближнее поле, мультиполи, антенная геометрия), а не как обычный угловой фактор
точечных диполей, где типичны значения 2 и -1.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from qd_mnp_rational_fit import (
    AU_DIPOLE_C_M,
    HybridQDPlasmonModel,
    eV_to_au,
)
from qd_mnp_params import DEBYE_C_M, make_params_with_overrides
from qd_mnp_linear_spectrum import (
    cross_section_cm2,
    qd_linear_polarizability_au,
)


def scan_candidates(
    *,
    target_ev: float,
    window_ev: float,
    grid_points: int,
    omega0_min_ev: float,
    omega0_max_ev: float,
    omega0_points: int,
    gamma_dephasing_mev_values: list[float],
    d_debye_values: list[float],
    g_min: float,
    g_max: float,
    g_points: int,
    g_spacing: str,
    fit_window_ev: tuple[float, float],
    weight_center_ev: float | None,
    weight_sigma_ev: float | None,
    eps_m: float | None,
    r_nm: float | None,
    c_nm: float | None,
    a_nm: float | None,
) -> list[dict[str, float]]:
    params = make_params_with_overrides(
        eps_m=eps_m,
        r_nm=r_nm,
        c_nm=c_nm,
        a_nm=a_nm,
    )
    model = HybridQDPlasmonModel(
        params,
        orientation="long",
        n_modes=4,
        fit_window_eV=fit_window_ev,
        weight_center_eV=weight_center_ev,
        weight_sigma_eV=weight_sigma_ev,
        alpha_objective_weight=1.0,
        inv_alpha_objective_weight=1.2,
        verbose=True,
    )

    energies = np.linspace(target_ev - window_ev, target_ev + window_ev, grid_points)
    omega = eV_to_au(energies)
    target_index = int(np.argmin(np.abs(energies - target_ev)))

    alpha_p = model.C * model.alpha_from_fit(energies)
    sigma_bare = cross_section_cm2(alpha_p / params.eps_m, omega, params.eps_m)

    omega0_values = np.linspace(omega0_min_ev, omega0_max_ev, omega0_points)
    if g_spacing == "log":
        g_values = np.logspace(np.log10(g_min), np.log10(g_max), g_points)
    else:
        g_values = np.linspace(g_min, g_max, g_points)

    rows: list[dict[str, float]] = []
    for d_debye in d_debye_values:
        d_au = float(d_debye * DEBYE_C_M / AU_DIPOLE_C_M)
        for omega0_ev in omega0_values:
            omega0_au = float(eV_to_au(omega0_ev))
            for gamma_mev in gamma_dephasing_mev_values:
                gamma_au = float(eV_to_au(gamma_mev / 1000.0))
                beta = qd_linear_polarizability_au(omega, d_au, omega0_au, gamma_au)
                for g_factor in g_values:
                    j_coupling = float(g_factor) / (params.eps_m * params.R_au**3)
                    denominator = 1.0 - (j_coupling**2) * alpha_p * beta
                    mu_p_over_e = alpha_p * (1.0 + j_coupling * beta) / denominator
                    mu_d_over_e = beta * (1.0 + j_coupling * alpha_p) / denominator
                    alpha_eff = (mu_p_over_e + mu_d_over_e) / params.eps_m
                    sigma_coupled = cross_section_cm2(alpha_eff, omega, params.eps_m)
                    ratio = sigma_coupled / sigma_bare

                    finite = np.isfinite(ratio) & np.isfinite(sigma_coupled) & (sigma_coupled > 0.0)
                    if not np.any(finite):
                        continue

                    finite_indices = np.flatnonzero(finite)
                    min_local_index = finite_indices[np.argmin(ratio[finite])]
                    row = {
                        "target_ev": float(target_ev),
                        "ratio_at_target": float(ratio[target_index]),
                        "sigma_coupled_at_target_cm2": float(sigma_coupled[target_index]),
                        "sigma_bare_at_target_cm2": float(sigma_bare[target_index]),
                        "delta_at_target_cm2": float(sigma_coupled[target_index] - sigma_bare[target_index]),
                        "min_ratio_in_window": float(ratio[min_local_index]),
                        "min_ratio_energy_ev": float(energies[min_local_index]),
                        "min_delta_cm2": float(sigma_coupled[min_local_index] - sigma_bare[min_local_index]),
                        "omega0_ev": float(omega0_ev),
                        "gamma_dephasing_mev": float(gamma_mev),
                        "d_debye": float(d_debye),
                        "G": float(g_factor),
                        "R_nm": float(params.R_au * 0.0529177210903),
                        "eps_m": float(params.eps_m),
                    }
                    rows.append(row)

    rows.sort(key=lambda row: row["ratio_at_target"])
    return rows


def write_csv(rows: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_top(rows: list[dict[str, float]], n: int) -> None:
    print("\n=== Best Fano-dip candidates at target ===")
    for i, row in enumerate(rows[:n], start=1):
        print(
            f"{i:2d}. ratio@target={row['ratio_at_target']:.4g}, "
            f"min={row['min_ratio_in_window']:.4g} at {row['min_ratio_energy_ev']:.5f} eV, "
            f"omega0={row['omega0_ev']:.5f} eV, "
            f"Gamma2={row['gamma_dephasing_mev']:.3g} meV, "
            f"d={row['d_debye']:.3g} D, G={row['G']:.4g}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search weak-field parameters with a Fano dip near target energy.")
    parser.add_argument("--target-ev", type=float, default=2.042)
    parser.add_argument("--window-ev", type=float, default=0.08)
    parser.add_argument("--grid-points", type=int, default=401)
    parser.add_argument("--omega0-min-ev", type=float, default=2.00)
    parser.add_argument("--omega0-max-ev", type=float, default=2.07)
    parser.add_argument("--omega0-points", type=int, default=141)
    parser.add_argument("--gamma-dephasing-mev-values", nargs="+", type=float, default=[0.5, 1.0, 1.27, 2.0, 3.02])
    parser.add_argument("--d-debye-values", nargs="+", type=float, default=[13.9, 22.5, 30.0])
    parser.add_argument("--G-min", type=float, default=2.0)
    parser.add_argument("--G-max", type=float, default=20.0)
    parser.add_argument("--G-points", type=int, default=61)
    parser.add_argument("--G-spacing", choices=["linear", "log"], default="linear")
    parser.add_argument("--fit-min-ev", type=float, default=0.8)
    parser.add_argument("--fit-max-ev", type=float, default=3.0)
    parser.add_argument("--weight-center-ev", type=float, default=2.35)
    parser.add_argument("--weight-sigma-ev", type=float, default=0.30)
    parser.add_argument("--eps-m", type=float, default=None)
    parser.add_argument("--r-nm", type=float, default=None)
    parser.add_argument("--c-nm", type=float, default=None)
    parser.add_argument("--a-nm", type=float, default=None)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--csv", type=Path, default=Path("results/fano_parameter_scan.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = scan_candidates(
        target_ev=args.target_ev,
        window_ev=args.window_ev,
        grid_points=args.grid_points,
        omega0_min_ev=args.omega0_min_ev,
        omega0_max_ev=args.omega0_max_ev,
        omega0_points=args.omega0_points,
        gamma_dephasing_mev_values=args.gamma_dephasing_mev_values,
        d_debye_values=args.d_debye_values,
        g_min=args.G_min,
        g_max=args.G_max,
        g_points=args.G_points,
        g_spacing=args.G_spacing,
        fit_window_ev=(args.fit_min_ev, args.fit_max_ev),
        weight_center_ev=args.weight_center_ev,
        weight_sigma_ev=args.weight_sigma_ev,
        eps_m=args.eps_m,
        r_nm=args.r_nm,
        c_nm=args.c_nm,
        a_nm=args.a_nm,
    )
    write_csv(rows, args.csv)
    print(f"Wrote {len(rows)} rows to {args.csv}")
    print_top(rows, args.top)


if __name__ == "__main__":
    main()
