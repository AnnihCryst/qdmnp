"""Compare legacy dipole coupling with the analytic full-QS spheroid kernel.

The runner deliberately exports three branches:

``legacy``
    The unchanged central point-dipole interaction, B=A*J and K=A*J**2.
``spheroid_n1``
    Exact excitation and observation of the bright spheroidal order n=1.
``spheroid_full``
    The same exact bright cross channel B plus all retained reaction orders K_n.

The legacy-to-n1 difference measures the central-field/projection error even
within the bright mode.  The n1-to-full difference isolates higher spatial
plasmon modes.  All branches use the same material data, QD polarizability and
weak-field Bloch linearization.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
from typing import Iterable
import warnings

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from qd_mnp_linear_spectrum import quasistatic_work_loss_cross_section_cm2
from qd_mnp_rational_fit import (
    HybridQDPlasmonModel,
    au_to_nm,
    eV_to_au,
    field_polarization_from_orientation,
    make_params_with_overrides,
    nm_to_au,
    params_to_physical_dict,
    timestamped_run_dir,
    validate_qd_position,
)
from qd_mnp_spheroid_green import (
    LegacyDipoleInteraction,
    ProlateSpheroidGeometry,
    SpheroidGreenInteraction,
    legacy_dipole_response_from_A,
    qd_linear_polarizability_from_params,
    solve_linear_hybrid_response,
)


COMPARISON_SCHEMA_VERSION = 2
POLICIES = {"raise", "warn", "ignore"}
OWNED_PLOTS = (
    "linear_spectrum.png",
    "response_coefficients.png",
    "multipole_convergence.png",
    "gap_sweep.png",
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty comparison table: {path.name}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _complex_columns(prefix: str, value: complex) -> dict[str, float]:
    scalar = complex(value)
    return {
        f"{prefix}_real": float(scalar.real),
        f"{prefix}_imag": float(scalar.imag),
        f"{prefix}_abs": float(abs(scalar)),
    }


def _complex_metadata(value: complex) -> dict[str, float]:
    scalar = complex(value)
    return {
        "real": float(scalar.real),
        "imag": float(scalar.imag),
        "abs": float(abs(scalar)),
    }


def _automatic_orders(n_max: int) -> tuple[int, ...]:
    """Choose a compact cumulative-order audit that always ends at ``n_max``."""

    values = {1, n_max, max(1, n_max // 2)}
    order = 1
    while order < n_max:
        values.add(order)
        order *= 2
    return tuple(sorted(value for value in values if value <= n_max))


def _validate_orders(
    orders: Iterable[int] | None,
    n_max: int,
) -> tuple[int, ...]:
    if isinstance(n_max, bool) or not isinstance(n_max, (int, np.integer)):
        raise ValueError("n_max must be an integer.")
    n_max = int(n_max)
    if n_max < 1:
        raise ValueError("n_max must be at least 1.")
    if orders is None:
        return _automatic_orders(n_max)
    raw_values = list(orders)
    if any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer))
        for value in raw_values
    ):
        raise ValueError("multipole_orders must contain only integers.")
    values = sorted({int(value) for value in raw_values})
    if not values or values[0] < 1 or values[-1] > n_max:
        raise ValueError("multipole_orders must lie inside [1, n_max].")
    if n_max not in values:
        values.append(n_max)
    return tuple(values)


def _create_unique_run_dir(output_dir: str | Path) -> Path:
    """Atomically reserve a timestamped directory, adding a collision suffix."""

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


def _apply_convergence_policy(policy: str, message: str) -> None:
    if policy == "raise":
        raise RuntimeError(message)
    if policy == "warn":
        warnings.warn(message, RuntimeWarning, stacklevel=3)


def run_comparison(
    *,
    output_dir: str | Path = "results/spheroid_green_comparison",
    orientations: tuple[str, ...] = ("long", "trans"),
    qd_position: str = "tip",
    energy_window_eV: tuple[float, float] = (1.8, 2.3),
    energy_points: int = 1001,
    target_energy_eV: float = 2.042,
    n_max: int = 80,
    multipole_orders: tuple[int, ...] | None = None,
    gaps_nm: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0),
    c_nm: float = 15.0,
    a_nm: float = 7.0,
    r_nm: float = 18.0,
    qd_radius_nm: float = 2.0,
    eps_m: float = 1.0,
    eps_qd: float = 6.0,
    d_debye: float | None = None,
    omega0_eV: float = 2.042,
    gamma_population_meV: float | None = None,
    gamma2_coherence_meV: float | None = None,
    qd_dipole_convention: str = "effective_external",
    n_legacy_fit_modes: int = 9,
    convergence_rtol: float = 1.0e-8,
    convergence_policy: str = "raise",
    make_plots: bool = True,
    show: bool = False,
) -> Path:
    """Run the three-branch frequency, order and gap comparisons."""

    if len(energy_window_eV) != 2 or not (
        np.isfinite(energy_window_eV[0])
        and np.isfinite(energy_window_eV[1])
        and 0.0 < energy_window_eV[0] < energy_window_eV[1]
    ):
        raise ValueError("energy_window_eV must satisfy 0 < min < max.")
    if energy_points < 2:
        raise ValueError("energy_points must be at least 2.")
    if not energy_window_eV[0] <= target_energy_eV <= energy_window_eV[1]:
        raise ValueError("target_energy_eV must lie inside energy_window_eV.")
    if not orientations or any(value not in {"long", "trans"} for value in orientations):
        raise ValueError("orientations must contain only 'long' and/or 'trans'.")
    validate_qd_position(qd_position)
    if not np.isfinite(convergence_rtol) or convergence_rtol <= 0.0:
        raise ValueError("convergence_rtol must be finite and positive.")
    if convergence_policy not in POLICIES:
        raise ValueError(
            "convergence_policy must be 'raise', 'warn' or 'ignore'."
        )
    if any(not np.isfinite(gap) or gap <= 0.0 for gap in gaps_nm):
        raise ValueError("Every surface gap must be finite and positive.")
    orders = _validate_orders(multipole_orders, n_max)
    n_max = int(n_max)

    energies = np.linspace(energy_window_eV[0], energy_window_eV[1], energy_points)
    omega = np.asarray(eV_to_au(energies), dtype=float)
    spectrum_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    convergence_rows: list[dict[str, object]] = []
    gap_rows: list[dict[str, object]] = []
    orientation_metadata: dict[str, object] = {}
    plot_data: dict[str, dict[str, object]] = {}

    for orientation in orientations:
        params = make_params_with_overrides(
            c_nm=c_nm,
            a_nm=a_nm,
            r_nm=r_nm,
            qd_radius_nm=qd_radius_nm,
            eps_m=eps_m,
            eps_qd=eps_qd,
            d_debye=d_debye,
            omega0_ev=omega0_eV,
            gamma_population_mev=gamma_population_meV,
            gamma2_coherence_mev=gamma2_coherence_meV,
            qd_dipole_convention=qd_dipole_convention,
            qd_position=qd_position,
            field_polarization=field_polarization_from_orientation(orientation),
        )
        legacy_model = HybridQDPlasmonModel(
            params,
            orientation=orientation,
            n_modes=n_legacy_fit_modes,
            radiative_consistency_policy="ignore",
            verbose=False,
        )
        legacy = LegacyDipoleInteraction(legacy_model).frequency_response(
            energies,
            mnp_response="material",
        )
        kernel = SpheroidGreenInteraction.from_params(params, n_max=n_max)
        full = kernel.response_from_material(params.material, energies)
        bright = full.truncate(1)
        beta = qd_linear_polarizability_from_params(params, energies)
        branches = {
            "legacy": legacy,
            "spheroid_n1": bright,
            "spheroid_full": full,
        }
        linear = {
            name: solve_linear_hybrid_response(
                response,
                beta,
                eps_m=params.eps_m,
            )
            for name, response in branches.items()
        }
        work_loss = {
            name: quasistatic_work_loss_cross_section_cm2(
                values.alpha_effective_au3,
                omega,
                params.eps_m,
            )
            for name, values in linear.items()
        }
        bare_alpha = full.A_au3 / params.eps_m
        bare_work = quasistatic_work_loss_cross_section_cm2(
            bare_alpha,
            omega,
            params.eps_m,
        )
        half_order_change_grid = full.relative_half_order_change()
        tail_block_grid = full.relative_tail_block()
        max_half_order_change = float(np.max(half_order_change_grid))
        max_tail_block_mass = float(np.max(tail_block_grid))
        energy_grid_converged = bool(
            max_half_order_change <= convergence_rtol
            and max_tail_block_mass <= convergence_rtol
        )
        if not energy_grid_converged:
            _apply_convergence_policy(
                convergence_policy,
                f"The spheroidal Green series is not converged for orientation="
                f"{orientation!r} on the energy grid: max relative N/2-to-N "
                f"change={max_half_order_change:.6g}, tail-block mass="
                f"{max_tail_block_mass:.6g}, tolerance={convergence_rtol:.6g}, "
                f"n_max={n_max}.",
            )

        for index, energy in enumerate(energies):
            for name in ("legacy", "spheroid_n1", "spheroid_full"):
                response = branches[name]
                values = linear[name]
                row: dict[str, object] = {
                    "orientation": orientation,
                    "model": name,
                    "energy_eV": float(energy),
                    "quasistatic_work_loss_cm2": float(work_loss[name][index]),
                    "bare_mnp_work_loss_cm2": float(bare_work[index]),
                }
                row.update(_complex_columns("alpha_effective_au3", values.alpha_effective_au3[index]))
                row.update(_complex_columns("mnp_dipole_over_field_au3", values.mnp_dipole_over_field_au3[index]))
                row.update(_complex_columns("qd_dipole_over_field_au3", values.qd_dipole_over_field_au3[index]))
                row.update(_complex_columns("denominator", values.denominator[index]))
                spectrum_rows.append(row)

            ratio_B = full.B[index] / legacy.B[index]
            ratio_K = full.K_au_minus3[index] / legacy.K_au_minus3[index]
            coefficient_row: dict[str, object] = {
                "orientation": orientation,
                "energy_eV": float(energy),
                "n_max": n_max,
                "higher_mode_fraction_abs": float(
                    abs(full.K_higher_au_minus3[index])
                    / max(abs(full.K_au_minus3[index]), np.finfo(float).tiny)
                ),
                "half_order_relative_change": float(
                    half_order_change_grid[index]
                ),
                "tail_block_relative_mass": float(tail_block_grid[index]),
            }
            for prefix, value in (
                ("A_au3", full.A_au3[index]),
                ("B_exact", full.B[index]),
                ("B_legacy", legacy.B[index]),
                ("B_exact_over_legacy", ratio_B),
                ("K_full_au_minus3", full.K_au_minus3[index]),
                ("K_bright_au_minus3", full.K_bright_au_minus3[index]),
                ("K_legacy_au_minus3", legacy.K_au_minus3[index]),
                ("K_full_over_legacy", ratio_K),
                ("K_higher_au_minus3", full.K_higher_au_minus3[index]),
            ):
                coefficient_row.update(_complex_columns(prefix, value))
            coefficient_rows.append(coefficient_row)

        target_epsilon = complex(params.material.epsilon_at(target_energy_eV))
        target_beta = qd_linear_polarizability_from_params(
            params,
            np.asarray(target_energy_eV),
        )
        target_full = kernel.response_from_epsilon(target_epsilon)
        target_bright = target_full.truncate(1)
        target_legacy = legacy_dipole_response_from_A(
            target_full.A_au3,
            kernel.geometry,
        )
        target_linear_full = solve_linear_hybrid_response(
            target_full,
            target_beta,
            eps_m=params.eps_m,
        )
        for order in orders:
            truncated = target_full.truncate(order)
            truncated_linear = solve_linear_hybrid_response(
                truncated,
                target_beta,
                eps_m=params.eps_m,
            )
            row = {
                "orientation": orientation,
                "energy_eV": float(target_energy_eV),
                "spatial_order_max": order,
                "K_relative_error_vs_n_max": float(
                    abs(truncated.K_au_minus3 - target_full.K_au_minus3)
                    / max(abs(complex(target_full.K_au_minus3)), np.finfo(float).tiny)
                ),
                "alpha_relative_error_vs_n_max": float(
                    abs(
                        truncated_linear.alpha_effective_au3
                        - target_linear_full.alpha_effective_au3
                    )
                    / max(
                        abs(complex(target_linear_full.alpha_effective_au3)),
                        np.finfo(float).tiny,
                    )
                ),
            }
            row.update(_complex_columns("K_au_minus3", truncated.K_au_minus3))
            row.update(
                _complex_columns(
                    "K_cumulative_au_minus3",
                    truncated.K_au_minus3,
                )
            )
            row.update(
                _complex_columns(
                    "K_order_contribution_au_minus3",
                    target_full.K_by_degree_au_minus3[order - 1],
                )
            )
            row.update(
                _complex_columns(
                    "alpha_effective_au3",
                    truncated_linear.alpha_effective_au3,
                )
            )
            convergence_rows.append(row)

        target_A = target_full.A_au3
        directional_semiaxis_nm = c_nm if qd_position == "tip" else a_nm
        for gap_nm in gaps_nm:
            separation_nm = directional_semiaxis_nm + qd_radius_nm + gap_nm
            gap_params = replace(params, R_au=float(nm_to_au(separation_nm)))
            gap_geometry = ProlateSpheroidGeometry.from_params(gap_params)
            gap_kernel = SpheroidGreenInteraction(gap_geometry, n_max=n_max)
            gap_full = gap_kernel.response_from_epsilon(target_epsilon)
            gap_bright = gap_full.truncate(1)
            gap_legacy = legacy_dipole_response_from_A(target_A, gap_geometry)
            gap_half_order_change = float(
                gap_full.relative_half_order_change()
            )
            gap_tail_block_mass = float(gap_full.relative_tail_block())
            gap_converged = bool(
                gap_half_order_change <= convergence_rtol
                and gap_tail_block_mass <= convergence_rtol
            )
            if not gap_converged:
                _apply_convergence_policy(
                    convergence_policy,
                    f"The spheroidal Green series is not converged for "
                    f"orientation={orientation!r}, qd_position={qd_position!r}, "
                    f"surface_gap_nm={gap_nm:g}: "
                    f"relative N/2-to-N change={gap_half_order_change:.6g}, "
                    f"tail-block mass={gap_tail_block_mass:.6g}, tolerance="
                    f"{convergence_rtol:.6g}, n_max={n_max}.",
                )
            gap_branches = {
                "legacy": gap_legacy,
                "spheroid_n1": gap_bright,
                "spheroid_full": gap_full,
            }
            gap_linear = {
                name: solve_linear_hybrid_response(
                    response,
                    target_beta,
                    eps_m=params.eps_m,
                )
                for name, response in gap_branches.items()
            }
            target_omega = np.asarray(eV_to_au(target_energy_eV))
            for name, response in gap_branches.items():
                work = quasistatic_work_loss_cross_section_cm2(
                    gap_linear[name].alpha_effective_au3,
                    target_omega,
                    params.eps_m,
                )
                row = {
                    "orientation": orientation,
                    "qd_position": qd_position,
                    "field_polarization": params.field_polarization,
                    "model": name,
                    "energy_eV": float(target_energy_eV),
                    "surface_gap_nm": float(gap_nm),
                    "center_distance_nm": float(separation_nm),
                    "quasistatic_work_loss_cm2": float(work),
                    "asymptotic_order_ratio": float(gap_kernel.asymptotic_order_ratio),
                    "full_half_order_relative_change": float(
                        gap_half_order_change
                    ),
                    "full_tail_block_relative_mass": float(
                        gap_tail_block_mass
                    ),
                    "full_series_converged": gap_converged,
                }
                row.update(_complex_columns("B", response.B))
                row.update(_complex_columns("K_au_minus3", response.K_au_minus3))
                row.update(
                    _complex_columns(
                        "alpha_effective_au3",
                        gap_linear[name].alpha_effective_au3,
                    )
                )
                gap_rows.append(row)

        common_physical_parameters = params_to_physical_dict(params, orientation)
        legacy_coupling_label = common_physical_parameters.pop("coupling_model")
        orientation_metadata[orientation] = {
            "common_physical_parameters": common_physical_parameters,
            "coupling_at_target_energy": {
                "energy_eV": float(target_energy_eV),
                "material_response": "direct_piecewise_linear_n_k",
                "legacy": {
                    "spatial_kernel": legacy_coupling_label,
                    "spatial_degrees_of_freedom": "one central induced point dipole",
                    "exact_spheroidal_projection": False,
                    "J_au_minus3": float(legacy_model.J),
                    "B": _complex_metadata(target_legacy.B),
                    "K_au_minus3": _complex_metadata(
                        target_legacy.K_au_minus3
                    ),
                    "identity": "B=A*J; K=A*J^2",
                },
                "spheroid_n1": {
                    "spatial_kernel": (
                        "exact_sphere_bright_projection"
                        if kernel.is_spherical
                        else "exact_prolate_spheroid_bright_projection"
                    ),
                    "retained_spheroidal_orders": [1],
                    "B": _complex_metadata(target_bright.B),
                    "K_au_minus3": _complex_metadata(
                        target_bright.K_au_minus3
                    ),
                    "identity": "K_1=B^2/A without complex conjugation",
                },
                "spheroid_full": {
                    "spatial_kernel": (
                        "analytic_sphere_green_series"
                        if kernel.is_spherical
                        else "analytic_prolate_spheroid_green_series"
                    ),
                    "retained_spheroidal_orders": [1, n_max],
                    "retained_order_semantics": "all integer n in the inclusive range",
                    "retained_azimuthal_orders": [
                        int(np.min(kernel.azimuthal_orders)),
                        int(np.max(kernel.azimuthal_orders)),
                    ],
                    "retained_azimuthal_semantics": (
                        "single order m for an axial QD; every m of parity "
                        "matching the QD dipole for an equatorial QD"
                    ),
                    "retained_mode_count": int(kernel.mode_count),
                    "uniform_laser_drive_orders": [1],
                    "point_qd_reaction_orders": [1, n_max],
                    "B": _complex_metadata(target_full.B),
                    "K_au_minus3": _complex_metadata(
                        target_full.K_au_minus3
                    ),
                    "K_bright_au_minus3": _complex_metadata(
                        target_full.K_bright_au_minus3
                    ),
                    "K_higher_au_minus3": _complex_metadata(
                        target_full.K_higher_au_minus3
                    ),
                },
            },
            "n_max": n_max,
            "asymptotic_order_ratio": kernel.asymptotic_order_ratio,
            "max_half_order_relative_change": float(
                max_half_order_change
            ),
            "max_tail_block_relative_mass": float(
                max_tail_block_mass
            ),
            "series_converged_on_energy_grid": energy_grid_converged,
            "max_relative_A_disagreement_legacy_vs_spheroid": float(
                np.max(
                    np.abs(full.A_au3 - legacy.A_au3)
                    / np.maximum(np.abs(full.A_au3), np.finfo(float).tiny)
                )
            ),
            "L_bright": float(kernel.depolarization_by_degree[0]),
        }
        plot_data[orientation] = {
            "energies": energies,
            "linear": linear,
            "work_loss": work_loss,
            "bare_work": bare_work,
            "legacy": legacy,
            "full": full,
        }

    run_dir = _create_unique_run_dir(output_dir)
    _write_csv(run_dir / "linear_spectrum.csv", spectrum_rows)
    _write_csv(run_dir / "response_coefficients.csv", coefficient_rows)
    _write_csv(run_dir / "multipole_convergence.csv", convergence_rows)
    _write_csv(run_dir / "gap_sweep.csv", gap_rows)

    metadata = {
        "comparison_schema_version": COMPARISON_SCHEMA_VERSION,
        "created_at": datetime.now().astimezone().isoformat(),
        "implementation": {
            "legacy": "LegacyDipoleInteraction over HybridQDPlasmonModel",
            "spheroid_n1": "SpheroidGreenInteraction truncated to n=1",
            "spheroid_full": "SpheroidGreenInteraction summed through n_max",
        },
        "material_response": "direct_piecewise_linear_n_k",
        "models": {
            "legacy": {
                "mnp_spatial_response": "central point dipole",
                "exact_spheroidal_projection": False,
                "coupling": "B=A*J, K=A*J^2, J=G/(eps_m*R^3)",
            },
            "spheroid_n1": {
                "mnp_spatial_response": "exact spheroidal bright order n=1",
                "coupling": "exact A/B projection and K=K_1=B^2/A",
            },
            "spheroid_full": {
                "mnp_spatial_response": "analytic spheroidal orders 1..n_max",
                "coupling": "exact bright A/B and K=sum(n=1..n_max) K_n",
            },
        },
        "units": {
            "A": "a0^3",
            "B": "dimensionless",
            "K": "a0^-3",
            "energy": "eV",
            "cross_section": "cm^2",
        },
        "normalization": {
            "effective_alpha": "(p_M+p_D)/(eps_m*E_inc)",
            "reciprocal_identity": "K_1=B^2/A without complex conjugation",
            "qd_source_for_green_kernel": "externally visible host dipole",
        },
        "numerical_settings": {
            "energy_window_eV": list(energy_window_eV),
            "energy_points": energy_points,
            "target_energy_eV": target_energy_eV,
            "n_max": n_max,
            "multipole_orders": list(orders),
            "multipole_orders_source": (
                "automatic" if multipole_orders is None else "explicit"
            ),
            "gaps_nm": list(gaps_nm),
            "convergence_rtol": convergence_rtol,
            "convergence_policy": convergence_policy,
            "n_legacy_fit_modes": n_legacy_fit_modes,
            "make_plots": make_plots,
        },
        "scope_limits": [
            "strict quasistatics: no retardation or radiation reaction",
            "homogeneous prolate spheroid or its exact spherical limit",
            "point QD on the positive symmetry axis",
            "local frequency-dependent particle permittivity",
            "no geometry-derived Purcell/Lindblad-rate correction",
            "work-loss curve is not labelled pure metal absorption",
        ],
        "orientation_diagnostics": orientation_metadata,
    }
    with (run_dir / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)

    figures: tuple[Figure, ...] = ()
    if make_plots:
        figures = _make_plots(
            run_dir,
            orientations=orientations,
            plot_data=plot_data,
            convergence_rows=convergence_rows,
            gap_rows=gap_rows,
            target_energy_eV=target_energy_eV,
        )
    if show and make_plots:
        plt.show()
    else:
        for figure in figures:
            plt.close(figure)
    return run_dir


def _make_plots(
    run_dir: Path,
    *,
    orientations: tuple[str, ...],
    plot_data: dict[str, dict[str, object]],
    convergence_rows: list[dict[str, object]],
    gap_rows: list[dict[str, object]],
    target_energy_eV: float,
) -> tuple[Figure, ...]:
    colours = {
        "legacy": "tab:gray",
        "spheroid_n1": "tab:blue",
        "spheroid_full": "tab:red",
    }
    labels = {
        "legacy": "legacy dipole",
        "spheroid_n1": "exact spheroid n=1",
        "spheroid_full": "full spheroid",
    }

    figures: list[Figure] = []
    figure, axes = plt.subplots(len(orientations), 1, figsize=(8.0, 4.2 * len(orientations)), squeeze=False)
    for axis, orientation in zip(axes[:, 0], orientations):
        data = plot_data[orientation]
        energies = np.asarray(data["energies"])
        for name in ("legacy", "spheroid_n1", "spheroid_full"):
            axis.plot(
                energies,
                np.asarray(data["work_loss"][name]),
                color=colours[name],
                label=labels[name],
            )
        axis.plot(energies, np.asarray(data["bare_work"]), "k--", label="bare MNP")
        axis.axvline(target_energy_eV, color="0.75", linewidth=1.0)
        axis.set_ylabel("QS work loss, cm$^2$")
        axis.set_title(f"{orientation} orientation")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[-1, 0].set_xlabel("Photon energy, eV")
    figure.tight_layout()
    figure.savefig(run_dir / "linear_spectrum.png", dpi=180)
    figures.append(figure)

    figure, axes = plt.subplots(len(orientations), 2, figsize=(11.0, 4.0 * len(orientations)), squeeze=False)
    for row_axes, orientation in zip(axes, orientations):
        data = plot_data[orientation]
        energies = np.asarray(data["energies"])
        responses = {
            "legacy": data["legacy"],
            "spheroid_n1": data["full"].truncate(1),
            "spheroid_full": data["full"],
        }
        for name, response in responses.items():
            K = np.asarray(response.K_au_minus3)
            row_axes[0].plot(
                energies,
                K.real,
                color=colours[name],
                label=labels[name],
            )
            row_axes[1].plot(
                energies,
                K.imag,
                color=colours[name],
                label=labels[name],
            )
        row_axes[0].set_ylabel(r"Re $K$, a$_0^{-3}$")
        row_axes[1].set_ylabel(r"Im $K$, a$_0^{-3}$")
        for axis in row_axes:
            axis.set_title(f"{orientation} orientation")
            axis.set_xlabel("Photon energy, eV")
            axis.grid(alpha=0.25)
            axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        row_axes[0].legend()
    figure.tight_layout()
    figure.savefig(run_dir / "response_coefficients.png", dpi=180)
    figures.append(figure)

    figure, axes = plt.subplots(
        len(orientations),
        3,
        figsize=(15.0, 4.0 * len(orientations)),
        squeeze=False,
    )
    for row_axes, orientation in zip(axes, orientations):
        rows = [row for row in convergence_rows if row["orientation"] == orientation]
        order = np.asarray([row["spatial_order_max"] for row in rows], dtype=float)
        K_real = np.asarray(
            [row["K_cumulative_au_minus3_real"] for row in rows]
        )
        K_imag = np.asarray(
            [row["K_cumulative_au_minus3_imag"] for row in rows]
        )
        error_K = np.asarray([row["K_relative_error_vs_n_max"] for row in rows])
        error_alpha = np.asarray([row["alpha_relative_error_vs_n_max"] for row in rows])
        row_axes[0].plot(order, K_real, "o-", color="tab:blue")
        row_axes[0].set_ylabel(r"Re $\sum_{n\leq N}K_n$, a$_0^{-3}$")
        row_axes[1].plot(order, K_imag, "o-", color="tab:red")
        row_axes[1].set_ylabel(r"Im $\sum_{n\leq N}K_n$, a$_0^{-3}$")
        row_axes[2].semilogy(
            order,
            np.maximum(error_K, 1e-18),
            "o-",
            label="K",
        )
        row_axes[2].semilogy(
            order,
            np.maximum(error_alpha, 1e-18),
            "s-",
            label="effective alpha",
        )
        row_axes[2].set_ylabel("Relative error vs final N")
        row_axes[2].legend()
        for axis in row_axes:
            axis.set_xlabel("Cumulative maximum order N")
            axis.set_title(f"{orientation}, {target_energy_eV:g} eV")
            axis.grid(alpha=0.25, which="both")
        row_axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        row_axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    figure.tight_layout()
    figure.savefig(run_dir / "multipole_convergence.png", dpi=180)
    figures.append(figure)

    figure, axes = plt.subplots(1, len(orientations), figsize=(6.0 * len(orientations), 4.3), squeeze=False)
    for axis, orientation in zip(axes[0], orientations):
        rows = [row for row in gap_rows if row["orientation"] == orientation]
        by_model = {
            name: [row for row in rows if row["model"] == name]
            for name in ("legacy", "spheroid_n1", "spheroid_full")
        }
        for name, model_rows in by_model.items():
            axis.loglog(
                [row["surface_gap_nm"] for row in model_rows],
                [row["K_au_minus3_abs"] for row in model_rows],
                "o-",
                color=colours[name],
                label=labels[name],
            )
        axis.set_xlabel("Surface gap, nm")
        axis.set_ylabel(r"$|K|$, a$_0^{-3}$")
        axis.set_title(f"{orientation}, {target_energy_eV:g} eV")
        axis.grid(alpha=0.25, which="both")
        axis.legend()
    figure.tight_layout()
    figure.savefig(run_dir / "gap_sweep.png", dpi=180)
    figures.append(figure)
    return tuple(figures)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/spheroid_green_comparison")
    parser.add_argument("--orientations", nargs="+", choices=("long", "trans"), default=("long", "trans"))
    parser.add_argument(
        "--qd-position",
        choices=("tip", "equatorial"),
        default="tip",
        help=(
            "QD centre: tip is (0,0,c+h) on the long axis, equatorial is "
            "(a+h,0,0) beside the particle. Independent of the polarization."
        ),
    )
    parser.add_argument("--energy-min-ev", type=float, default=1.8)
    parser.add_argument("--energy-max-ev", type=float, default=2.3)
    parser.add_argument("--energy-points", type=int, default=1001)
    parser.add_argument("--target-energy-ev", type=float, default=2.042)
    parser.add_argument(
        "--n-max",
        type=int,
        default=80,
        help="Largest retained spheroidal order.",
    )
    parser.add_argument(
        "--multipole-orders",
        nargs="+",
        type=int,
        default=None,
        metavar="N",
        help="Cumulative orders used in the convergence plot. By default they "
        "are generated automatically from --n-max.",
    )
    parser.add_argument("--gaps-nm", nargs="+", type=float, default=(1.0, 2.0, 5.0, 10.0, 20.0, 50.0))
    parser.add_argument("--c-nm", type=float, default=15.0)
    parser.add_argument("--a-nm", type=float, default=7.0)
    parser.add_argument("--r-nm", type=float, default=18.0)
    parser.add_argument("--qd-radius-nm", type=float, default=2.0)
    parser.add_argument("--eps-m", type=float, default=1.0)
    parser.add_argument("--eps-qd", type=float, default=6.0)
    parser.add_argument("--d-debye", type=float)
    parser.add_argument("--omega0-ev", type=float, default=2.042)
    parser.add_argument("--gamma-population-mev", type=float)
    parser.add_argument("--gamma2-coherence-mev", type=float)
    parser.add_argument(
        "--qd-dipole-convention",
        choices=("effective_external", "bare_internal"),
        default="effective_external",
    )
    parser.add_argument("--convergence-rtol", type=float, default=1.0e-8)
    parser.add_argument(
        "--convergence-policy",
        choices=tuple(sorted(POLICIES)),
        default="raise",
        help="Action when the N/2-to-N Green-series check fails on the energy "
        "grid or at any requested gap.",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = run_comparison(
        output_dir=args.output_dir,
        orientations=tuple(args.orientations),
        qd_position=args.qd_position,
        energy_window_eV=(args.energy_min_ev, args.energy_max_ev),
        energy_points=args.energy_points,
        target_energy_eV=args.target_energy_ev,
        n_max=args.n_max,
        multipole_orders=(
            None
            if args.multipole_orders is None
            else tuple(args.multipole_orders)
        ),
        gaps_nm=tuple(args.gaps_nm),
        c_nm=args.c_nm,
        a_nm=args.a_nm,
        r_nm=args.r_nm,
        qd_radius_nm=args.qd_radius_nm,
        eps_m=args.eps_m,
        eps_qd=args.eps_qd,
        d_debye=args.d_debye,
        omega0_eV=args.omega0_ev,
        gamma_population_meV=args.gamma_population_mev,
        gamma2_coherence_meV=args.gamma2_coherence_mev,
        qd_dipole_convention=args.qd_dipole_convention,
        convergence_rtol=args.convergence_rtol,
        convergence_policy=args.convergence_policy,
        make_plots=not args.no_plots,
        show=args.show,
    )
    print(f"Saved spheroid Green-response comparison to {run_dir}")


if __name__ == "__main__":
    main()
