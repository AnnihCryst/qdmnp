"""Поиск параметров, дающих слабополевой Фано-провал в спектре КТ-МНЧ.

Скрипт перебирает энергию перехода КТ, полную ширину когерентности, переходный дипольный
момент и эффективный фактор связи ``G``. Он ранжирует наборы параметров по
тому, насколько сильно связанная система имеет меньшую экстинкцию, чем голая МНЧ, около
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
import warnings

import numpy as np

from qd_mnp_rational_fit import (
    AU_DIPOLE_C_M,
    HybridQDPlasmonModel,
    SCHEMA_VERSION,
    au_to_eV,
    au_to_nm,
    eV_to_au,
)
from qd_mnp_params import DEBYE_C_M, make_params_with_overrides
from qd_mnp_linear_spectrum import extinction_cross_section_cm2, qd_linear_polarizability_au


def scan_candidates(
    *,
    target_ev: float,
    window_ev: float,
    grid_points: int,
    omega0_min_ev: float,
    omega0_max_ev: float,
    omega0_points: int,
    gamma2_coherence_mev_values: list[float] | None = None,
    gamma_dephasing_mev_values: list[float] | None = None,
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
    qd_radius_nm: float | None = None,
    c_nm: float | None,
    a_nm: float | None,
    gamma_population_mev: float | None = None,
) -> list[dict[str, float]]:
    """Return Fano candidates for physically consistent ``gamma1``/``Gamma2``.

    ``gamma2_coherence_mev_values`` is the canonical Python keyword. The old
    ``gamma_dephasing_mev_values`` name is retained as an alias because it has
    always represented the total coherence width rather than pure dephasing.
    """
    if gamma2_coherence_mev_values is None:
        if gamma_dephasing_mev_values is None:
            raise TypeError(
                "gamma2_coherence_mev_values is required "
                "(gamma_dephasing_mev_values is the legacy alias)."
            )
        gamma2_values = list(gamma_dephasing_mev_values)
    else:
        gamma2_values = list(gamma2_coherence_mev_values)
        if gamma_dephasing_mev_values is not None:
            legacy_values = list(gamma_dephasing_mev_values)
            if gamma2_values != legacy_values:
                raise ValueError(
                    "Conflicting gamma2_coherence_mev_values and legacy "
                    "gamma_dephasing_mev_values were specified."
                )
    if not gamma2_values:
        raise ValueError("At least one Gamma2 coherence width must be specified.")
    if grid_points < 2 or omega0_points < 1 or g_points < 1:
        raise ValueError("grid_points must be >=2 and omega0_points/G_points must be >=1.")
    if not np.isfinite(target_ev) or target_ev <= 0.0 or not np.isfinite(window_ev) or window_ev <= 0.0:
        raise ValueError("target_ev and window_ev must be finite and positive.")
    if not (
        np.isfinite(omega0_min_ev)
        and np.isfinite(omega0_max_ev)
        and 0.0 < omega0_min_ev <= omega0_max_ev
    ):
        raise ValueError("QD transition-energy bounds must be finite, positive and ordered.")
    if not d_debye_values or np.any(~np.isfinite(d_debye_values)) or np.any(np.asarray(d_debye_values) <= 0.0):
        raise ValueError("d_debye_values must contain positive finite dipole magnitudes.")
    if not np.isfinite(g_min) or not np.isfinite(g_max) or g_min > g_max:
        raise ValueError("G bounds must be finite and ordered.")
    if g_spacing not in {"linear", "log"}:
        raise ValueError("g_spacing must be 'linear' or 'log'.")
    if g_spacing == "log" and (g_min <= 0.0 or g_max <= 0.0):
        raise ValueError("Logarithmic G spacing requires positive bounds.")

    params = make_params_with_overrides(
        eps_m=eps_m,
        r_nm=r_nm,
        qd_radius_nm=qd_radius_nm,
        c_nm=c_nm,
        a_nm=a_nm,
        gamma_population_mev=gamma_population_mev,
        # The MNP fit itself is independent of Gamma2, but the shared model
        # constructor validates the complete parameter object. Use a scanned
        # value instead of an unrelated default coherence width.
        gamma2_coherence_mev=float(gamma2_values[0]),
    )
    gamma1_population_mev = float(au_to_eV(params.gamma_au) * 1000.0)
    for gamma2_mev in gamma2_values:
        gamma2_au = float(eV_to_au(gamma2_mev / 1000.0))
        if not np.isfinite(gamma2_au) or gamma2_au < 0.5 * params.gamma_au:
            raise ValueError(
                "Each Gamma2 coherence width must satisfy Gamma2 >= gamma1/2; "
                f"got Gamma2={gamma2_mev:g} meV with gamma1="
                f"{gamma1_population_mev:g} meV."
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
    sigma_ext_bare = extinction_cross_section_cm2(alpha_p / params.eps_m, omega, params.eps_m)

    omega0_values = np.linspace(omega0_min_ev, omega0_max_ev, omega0_points)
    if g_spacing == "log":
        g_values = np.logspace(np.log10(g_min), np.log10(g_max), g_points)
    else:
        g_values = np.linspace(g_min, g_max, g_points)

    rows: list[dict[str, float]] = []
    unstable_candidate_count = 0
    for d_debye in d_debye_values:
        d_au = float(d_debye * DEBYE_C_M / AU_DIPOLE_C_M)
        for omega0_ev in omega0_values:
            omega0_au = float(eV_to_au(omega0_ev))
            for gamma_mev in gamma2_values:
                gamma_au = float(eV_to_au(gamma_mev / 1000.0))
                beta = qd_linear_polarizability_au(omega, d_au, omega0_au, gamma_au)
                for g_factor in g_values:
                    stability = model.linearized_ground_state_stability(
                        d_au=d_au,
                        omega0_au=omega0_au,
                        gamma1_au=params.gamma_au,
                        gamma2_au=gamma_au,
                        g_factor=float(g_factor),
                    )
                    if not stability.stable:
                        unstable_candidate_count += 1
                        continue
                    j_coupling = float(g_factor) / (params.eps_m * params.R_au**3)
                    denominator = 1.0 - (j_coupling**2) * alpha_p * beta
                    mu_p_over_e = alpha_p * (1.0 + j_coupling * beta) / denominator
                    mu_d_over_e = beta * (1.0 + j_coupling * alpha_p) / denominator
                    alpha_eff = (mu_p_over_e + mu_d_over_e) / params.eps_m
                    sigma_ext_coupled = extinction_cross_section_cm2(alpha_eff, omega, params.eps_m)
                    ratio = sigma_ext_coupled / sigma_ext_bare

                    finite = np.isfinite(ratio) & np.isfinite(sigma_ext_coupled) & (sigma_ext_coupled > 0.0)
                    if not np.any(finite):
                        continue
                    if not finite[target_index]:
                        continue

                    finite_indices = np.flatnonzero(finite)
                    min_local_index = finite_indices[np.argmin(ratio[finite])]
                    row = {
                        "schema_version": SCHEMA_VERSION,
                        "target_ev": float(target_ev),
                        "linearized_ground_state_stable": True,
                        "linearized_ground_state_spectral_abscissa_au": float(
                            stability.spectral_abscissa_au
                        ),
                        "ratio_at_target": float(ratio[target_index]),
                        "sigma_ext_coupled_at_target_cm2": float(sigma_ext_coupled[target_index]),
                        "sigma_ext_bare_at_target_cm2": float(sigma_ext_bare[target_index]),
                        "delta_sigma_ext_at_target_cm2": float(sigma_ext_coupled[target_index] - sigma_ext_bare[target_index]),
                        "min_ratio_in_window": float(ratio[min_local_index]),
                        "min_ratio_energy_ev": float(energies[min_local_index]),
                        "min_delta_sigma_ext_cm2": float(sigma_ext_coupled[min_local_index] - sigma_ext_bare[min_local_index]),
                        "omega0_ev": float(omega0_ev),
                        "gamma_population_mev": gamma1_population_mev,
                        "gamma2_coherence_mev": float(gamma_mev),
                        "gamma_pure_dephasing_mev": float(gamma_mev - 0.5 * gamma1_population_mev),
                        "d_debye": float(d_debye),
                        "G": float(g_factor),
                        "R_nm": float(au_to_nm(params.R_au)),
                        "qd_radius_nm": float(au_to_nm(params.qd_radius_au)),
                        "surface_gap_nm": float(au_to_nm(params.axial_surface_gap_au)),
                        "eps_m": float(params.eps_m),
                        # Schema-1 compatibility aliases: these meant extinction.
                        "sigma_coupled_at_target_cm2": float(sigma_ext_coupled[target_index]),
                        "sigma_bare_at_target_cm2": float(sigma_ext_bare[target_index]),
                        "delta_at_target_cm2": float(sigma_ext_coupled[target_index] - sigma_ext_bare[target_index]),
                        "min_delta_cm2": float(sigma_ext_coupled[min_local_index] - sigma_ext_bare[min_local_index]),
                        "gamma_dephasing_mev": float(gamma_mev),
                    }
                    rows.append(row)

    if unstable_candidate_count:
        warnings.warn(
            f'Excluded {unstable_candidate_count} Fano candidate(s) whose full '
            'field-free Jacobian has a pole in the unstable half-plane.',
            RuntimeWarning,
            stacklevel=2,
        )
    rows.sort(key=lambda row: row["ratio_at_target"])
    return rows


def write_csv(rows: list[dict[str, float]], path: Path) -> None:
    if not rows:
        raise ValueError("The Fano scan produced no finite candidates; no CSV was written.")
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
            f"gamma1={row['gamma_population_mev']:.3g} meV, "
            f"Gamma2={row['gamma2_coherence_mev']:.3g} meV, "
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
    parser.add_argument(
        "--gamma2-coherence-mev-values",
        dest="gamma2_coherence_mev_values",
        metavar="GAMMA2_MEV",
        nargs="+",
        type=float,
        default=None,
        help="Total coherence HWHM values hbar*Gamma2 in meV.",
    )
    parser.add_argument(
        "--gamma-dephasing-mev-values",
        dest="gamma_dephasing_mev_values",
        metavar="GAMMA2_MEV",
        nargs="+",
        type=float,
        default=None,
        help="Deprecated alias for --gamma2-coherence-mev-values.",
    )
    parser.add_argument(
        "--gamma-population-mev",
        type=float,
        default=None,
        help="Population-decay width hbar*gamma1 in meV; Gamma2 must be at least gamma1/2.",
    )
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
    parser.add_argument("--qd-radius-nm", type=float, default=None)
    parser.add_argument("--c-nm", type=float, default=None)
    parser.add_argument("--a-nm", type=float, default=None)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--csv", type=Path, default=Path("results/fano_parameter_scan.csv"))
    args = parser.parse_args()
    if (
        args.gamma2_coherence_mev_values is not None
        and args.gamma_dephasing_mev_values is not None
        and args.gamma2_coherence_mev_values != args.gamma_dephasing_mev_values
    ):
        parser.error(
            "--gamma2-coherence-mev-values and --gamma-dephasing-mev-values "
            "must agree when both are supplied."
        )
    if args.gamma2_coherence_mev_values is None:
        args.gamma2_coherence_mev_values = (
            args.gamma_dephasing_mev_values
            if args.gamma_dephasing_mev_values is not None
            else [1.51, 2.0, 2.78, 3.02]
        )
    return args


def main() -> None:
    args = parse_args()
    rows = scan_candidates(
        target_ev=args.target_ev,
        window_ev=args.window_ev,
        grid_points=args.grid_points,
        omega0_min_ev=args.omega0_min_ev,
        omega0_max_ev=args.omega0_max_ev,
        omega0_points=args.omega0_points,
        gamma2_coherence_mev_values=args.gamma2_coherence_mev_values,
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
        qd_radius_nm=args.qd_radius_nm,
        c_nm=args.c_nm,
        a_nm=args.a_nm,
        gamma_population_mev=args.gamma_population_mev,
    )
    write_csv(rows, args.csv)
    print(f"Wrote {len(rows)} rows to {args.csv}")
    print_top(rows, args.top)


if __name__ == "__main__":
    main()
