"""Side-by-side laser-pulse comparison of legacy and full-QS QD--MNP models."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime
import json
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from qd_mnp_full_qs_model import (
    FullQSSpheroidPulseModel,
    build_positive_dark_reduction,
)
from qd_mnp_linear_spectrum import linear_coupled_alpha_au
from qd_mnp_pulse_absorption_sweep import spectral_effective_alpha_au
from qd_mnp_rational_fit import (
    GaussianPulse,
    HybridQDPlasmonModel,
    au_to_fs,
    eV_to_au,
    fs_to_au,
    make_params_with_overrides,
    params_to_physical_dict,
    response_tail_ratio,
    timestamped_run_dir,
)
from qd_mnp_spheroid_green import (
    LegacyDipoleInteraction,
    SpheroidGreenInteraction,
    qd_linear_polarizability_from_params,
    solve_linear_hybrid_response,
)
from qd_mnp_spheroid_equatorial import EquatorialSpheroidGreenInteraction


PULSE_COMPARISON_SCHEMA_VERSION = 2
POLICIES = {"raise", "warn", "ignore"}
SIDE_DIRECT_REFERENCE_ORDER_MAX = 8


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table {path.name}.")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _complex_metadata(value: complex) -> dict[str, float]:
    scalar = complex(value)
    return {
        "real": float(scalar.real),
        "imag": float(scalar.imag),
        "abs": float(abs(scalar)),
    }


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


def _apply_policy(policy: str, message: str) -> None:
    if policy == "raise":
        raise RuntimeError(message)
    if policy == "warn":
        warnings.warn(message, RuntimeWarning, stacklevel=3)


def _legacy_tail_ratio(
    model: HybridQDPlasmonModel,
    result,
    *,
    tail_fraction: float,
) -> float:
    """Component-wise legacy tail, including coherence and the reaction field."""

    W_index = 2 * model.n_modes
    Q = result.y[W_index + 1]
    P = result.y[W_index + 2]
    mnp_field = model.J * result.mu_p_au
    return max(
        response_tail_ratio(
            result.mu_total_au,
            result.t_au,
            result.mu_p_au,
            result.mu_d_au,
            tail_fraction=tail_fraction,
        ),
        response_tail_ratio(
            mnp_field,
            result.t_au,
            tail_fraction=tail_fraction,
        ),
        response_tail_ratio(
            np.hypot(Q, P),
            result.t_au,
            tail_fraction=tail_fraction,
        ),
    )


def run_pulse_comparison(
    *,
    output_dir: str | Path = "results/spheroid_pulse_comparison",
    orientation: str = "long",
    qd_placement: str = "axis",
    side_transverse_alignment: str | None = None,
    spatial_order_max: int = 80,
    material_fit_modes: int = 9,
    reduction_fit_grid_points: int = 1001,
    reduction_audit_grid_points: int = 1601,
    reduction_rms_tolerance: float = 1.0e-6,
    reduction_max_tolerance: float = 1.0e-4,
    reduction_max_nodes: int | None = None,
    reduction_policy: str = "raise",
    pulse_energy_eV: float = 2.042,
    pulse_tau_fs: float = 5.0,
    pulse_E0_au: float = 1.0e-5,
    pulse_tau_kind: str = "fwhm_intensity",
    start_sigma: float = 10.0,
    post_fs: float | None = None,
    common_time_points: int = 2001,
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
    method: str = "DOP853",
    rtol: float = 1.0e-8,
    atol: float = 1.0e-10,
    spectral_window_policy: str = "raise",
    max_spectral_leakage: float = 1.0e-3,
    positivity_policy: str = "raise",
    positivity_tolerance: float = 1.0e-7,
    tail_policy: str = "raise",
    tail_ratio_tolerance: float = 1.0e-4,
    tail_window_fraction: float = 0.05,
    max_auto_tail_extensions: int = 3,
    fit_quality_policy: str = "raise",
    max_modal_normalized_rms: float = 0.03,
    max_modal_relative_error: float = 0.06,
    spatial_convergence_policy: str = "raise",
    spatial_convergence_rtol: float = 1.0e-8,
    work_passivity_policy: str = "raise",
    concurrent: bool = True,
    make_plots: bool = True,
    show: bool = False,
) -> Path:
    """Propagate the same pulse through the old and new model APIs."""

    if orientation not in {"long", "trans"}:
        raise ValueError("orientation must be 'long' or 'trans'.")
    if qd_placement not in {"axis", "side"}:
        raise ValueError("qd_placement must be 'axis' or 'side'.")
    if side_transverse_alignment not in {None, "radial", "tangential"}:
        raise ValueError(
            "side_transverse_alignment must be None, 'radial' or 'tangential'."
        )
    if qd_placement == "axis" and side_transverse_alignment is not None:
        raise ValueError(
            "side_transverse_alignment must be omitted when qd_placement='axis'."
        )
    if qd_placement == "side" and orientation == "long":
        if side_transverse_alignment is not None:
            raise ValueError(
                "side_transverse_alignment applies only to a side/transverse channel."
            )
    if (
        qd_placement == "side"
        and orientation == "trans"
        and side_transverse_alignment is None
    ):
        raise ValueError(
            "A side/transverse channel requires side_transverse_alignment="
            "'radial' or 'tangential'."
        )
    if spatial_order_max < 1:
        raise ValueError("spatial_order_max must be at least 1.")
    if material_fit_modes < 1:
        raise ValueError("material_fit_modes must be at least 1.")
    if common_time_points < 101:
        raise ValueError("common_time_points must be at least 101.")
    if not np.isfinite(start_sigma) or start_sigma < 6.0:
        raise ValueError("start_sigma must be finite and at least 6.")
    if post_fs is not None and (not np.isfinite(post_fs) or post_fs <= 0.0):
        raise ValueError("post_fs must be finite and positive or None.")
    for policy_name, policy in (
        ("spectral_window_policy", spectral_window_policy),
        ("positivity_policy", positivity_policy),
        ("tail_policy", tail_policy),
        ("fit_quality_policy", fit_quality_policy),
        ("spatial_convergence_policy", spatial_convergence_policy),
        ("work_passivity_policy", work_passivity_policy),
        ("reduction_policy", reduction_policy),
    ):
        if policy not in POLICIES:
            raise ValueError(
                f"{policy_name} must be 'raise', 'warn' or 'ignore'."
            )
    if not np.isfinite(max_spectral_leakage) or not (
        0.0 <= max_spectral_leakage < 1.0
    ):
        raise ValueError("max_spectral_leakage must lie in [0, 1).")
    if not np.isfinite(positivity_tolerance) or positivity_tolerance < 0.0:
        raise ValueError("positivity_tolerance must be finite and non-negative.")
    if not np.isfinite(tail_ratio_tolerance) or not (
        0.0 < tail_ratio_tolerance < 1.0
    ):
        raise ValueError("tail_ratio_tolerance must lie in (0, 1).")
    if not np.isfinite(tail_window_fraction) or not (
        0.0 < tail_window_fraction <= 1.0
    ):
        raise ValueError("tail_window_fraction must lie in (0, 1].")
    if (
        isinstance(max_auto_tail_extensions, bool)
        or not isinstance(max_auto_tail_extensions, (int, np.integer))
        or max_auto_tail_extensions < 0
    ):
        raise ValueError("max_auto_tail_extensions must be a non-negative integer.")
    max_auto_tail_extensions = int(max_auto_tail_extensions)
    if not np.isfinite(spatial_convergence_rtol) or not (
        0.0 < spatial_convergence_rtol < 1.0
    ):
        raise ValueError("spatial_convergence_rtol must lie in (0, 1).")
    for name, value in (
        ("max_modal_normalized_rms", max_modal_normalized_rms),
        ("max_modal_relative_error", max_modal_relative_error),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    for name, value in (
        ("reduction_fit_grid_points", reduction_fit_grid_points),
        ("reduction_audit_grid_points", reduction_audit_grid_points),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < 101
        ):
            raise ValueError(f"{name} must be an integer at least 101.")
    if reduction_fit_grid_points == reduction_audit_grid_points:
        raise ValueError(
            "reduction_fit_grid_points and reduction_audit_grid_points must differ."
        )
    for name, value in (
        ("reduction_rms_tolerance", reduction_rms_tolerance),
        ("reduction_max_tolerance", reduction_max_tolerance),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    if reduction_max_nodes is not None and (
        isinstance(reduction_max_nodes, (bool, np.bool_))
        or not isinstance(reduction_max_nodes, (int, np.integer))
        or reduction_max_nodes < 1
    ):
        raise ValueError("reduction_max_nodes must be a positive integer or None.")

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
        orientation=orientation,
        qd_placement=qd_placement,
        side_transverse_alignment=side_transverse_alignment,
    )
    legacy_model = HybridQDPlasmonModel(
        params,
        orientation=orientation,
        n_modes=material_fit_modes,
        radiative_consistency_policy="ignore",
        verbose=False,
    )
    if qd_placement == "side":
        kernel = EquatorialSpheroidGreenInteraction.from_params(
            params,
            orientation=orientation,
            n_max=spatial_order_max,
        )
    else:
        kernel = SpheroidGreenInteraction.from_params(
            params,
            orientation=orientation,
            n_max=spatial_order_max,
        )
    dark_reduction = None
    if (
        qd_placement == "side"
        and spatial_order_max > SIDE_DIRECT_REFERENCE_ORDER_MAX
    ):
        dark_reduction = build_positive_dark_reduction(
            legacy_model,
            kernel,
            fit_grid_points=int(reduction_fit_grid_points),
            audit_grid_points=int(reduction_audit_grid_points),
            rms_tolerance=float(reduction_rms_tolerance),
            max_tolerance=float(reduction_max_tolerance),
            max_nodes=(
                None if reduction_max_nodes is None else int(reduction_max_nodes)
            ),
            policy=reduction_policy,
        )
        if not dark_reduction.diagnostics.accepted:
            raise RuntimeError(
                "The production side-QD time backend refuses an uncertified "
                "dark-kernel reduction, irrespective of the diagnostic "
                "reduction_policy. Increase reduction_max_nodes or relax only "
                "an explicitly justified and recorded accuracy tolerance."
            )
    full_model = FullQSSpheroidPulseModel(
        legacy_model,
        kernel,
        dark_reduction=dark_reduction,
        fit_quality_policy=fit_quality_policy,
        max_modal_normalized_rms=max_modal_normalized_rms,
        max_modal_relative_error=max_modal_relative_error,
        spatial_convergence_policy=spatial_convergence_policy,
        spatial_convergence_rtol=spatial_convergence_rtol,
        max_reduction_normalized_rms=float(reduction_rms_tolerance),
        max_reduction_normalized_error=float(reduction_max_tolerance),
    )
    pulse = GaussianPulse(
        E0_au=pulse_E0_au,
        omegaL_au=float(eV_to_au(pulse_energy_eV)),
        tau_au=float(fs_to_au(pulse_tau_fs)),
        tau_kind=pulse_tau_kind,
    )
    start_au = -start_sigma * pulse.sigma_t_au
    automatic_post = post_fs is None
    if post_fs is None:
        end_au = max(
            start_sigma * pulse.sigma_t_au,
            legacy_model.recommended_post_pulse_time_au(),
            full_model.recommended_post_pulse_time_au(),
        )
    else:
        end_au = float(fs_to_au(post_fs))
    initial_end_au = float(end_au)
    tail_extension_count = 0
    while True:
        t_span = (float(start_au), float(end_au))
        legacy_kwargs = dict(
            t_span_au=t_span,
            method=method,
            rtol=rtol,
            atol=atol,
            positivity_policy=positivity_policy,
            positivity_tol=positivity_tolerance,
            spectral_window_policy=spectral_window_policy,
            max_spectral_leakage=max_spectral_leakage,
        )
        full_kwargs = dict(
            t_span_au=t_span,
            method=method,
            rtol=rtol,
            atol=atol,
            positivity_policy=positivity_policy,
            positivity_tolerance=positivity_tolerance,
            spectral_window_policy=spectral_window_policy,
            max_spectral_leakage=max_spectral_leakage,
            work_passivity_policy=work_passivity_policy,
            response_tail_policy="ignore",
            response_tail_tolerance=tail_ratio_tolerance,
            response_tail_window_fraction=tail_window_fraction,
        )
        if concurrent:
            with ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="qd-mnp",
            ) as executor:
                legacy_future = executor.submit(
                    legacy_model.solve,
                    pulse,
                    **legacy_kwargs,
                )
                full_future = executor.submit(
                    full_model.solve,
                    pulse,
                    **full_kwargs,
                )
                legacy_result = legacy_future.result()
                full_result = full_future.result()
        else:
            legacy_result = legacy_model.solve(pulse, **legacy_kwargs)
            full_result = full_model.solve(pulse, **full_kwargs)

        tail_ratio_by_model = {
            "legacy": _legacy_tail_ratio(
                legacy_model,
                legacy_result,
                tail_fraction=tail_window_fraction,
            ),
            "spheroid_full": full_result.diagnostics.response_tail_ratio,
        }
        tails_converged = all(
            np.isfinite(value) and value <= tail_ratio_tolerance
            for value in tail_ratio_by_model.values()
        )
        if (
            not automatic_post
            or tails_converged
            or tail_extension_count >= max_auto_tail_extensions
        ):
            break
        end_au *= 2.0
        tail_extension_count += 1

    if not tails_converged:
        details = ", ".join(
            f"{name}={value:.6g}"
            for name, value in tail_ratio_by_model.items()
        )
        _apply_policy(
            tail_policy,
            "The common post-pulse window did not satisfy the component-wise "
            f"dipole-tail tolerance {tail_ratio_tolerance:.6g}: {details}.",
        )

    legacy_W = legacy_result.y[2 * legacy_model.n_modes]
    legacy_Q = legacy_result.y[2 * legacy_model.n_modes + 1]
    legacy_P = legacy_result.y[2 * legacy_model.n_modes + 2]
    legacy_rho22 = 0.5 * (legacy_W + 1.0)
    legacy_mnp_field = legacy_model.J * legacy_result.mu_p_au
    legacy_effective_field = params.qd_local_field_factor * (
        pulse.field(legacy_result.t_au) + legacy_mnp_field
    )

    required_common_time_points = int(
        np.ceil(
            (t_span[1] - t_span[0])
            / min(
                legacy_result.diagnostics.max_step_limit_au,
                full_result.diagnostics.max_step_limit_au,
            )
        )
        + 1
    )
    resolved_common_time_points = max(
        int(common_time_points),
        required_common_time_points,
    )
    common_time = np.linspace(
        t_span[0],
        t_span[1],
        resolved_common_time_points,
    )
    traces = {
        "time_au": common_time,
        "time_fs": au_to_fs(common_time),
        "incident_field_au": pulse.field(common_time),
        "legacy_W": np.interp(common_time, legacy_result.t_au, legacy_W),
        "legacy_Q": np.interp(common_time, legacy_result.t_au, legacy_Q),
        "legacy_P": np.interp(common_time, legacy_result.t_au, legacy_P),
        "legacy_rho22": np.interp(common_time, legacy_result.t_au, legacy_rho22),
        "legacy_mu_mnp_au": np.interp(
            common_time, legacy_result.t_au, legacy_result.mu_p_au
        ),
        "legacy_mu_qd_au": np.interp(
            common_time, legacy_result.t_au, legacy_result.mu_d_au
        ),
        "legacy_mu_total_au": np.interp(
            common_time, legacy_result.t_au, legacy_result.mu_total_au
        ),
        "legacy_mnp_field_at_qd_au": np.interp(
            common_time, legacy_result.t_au, legacy_mnp_field
        ),
        "legacy_effective_qd_field_au": np.interp(
            common_time, legacy_result.t_au, legacy_effective_field
        ),
        "full_W": np.interp(common_time, full_result.t_au, full_result.W),
        "full_Q": np.interp(common_time, full_result.t_au, full_result.Q),
        "full_P": np.interp(common_time, full_result.t_au, full_result.P),
        "full_rho22": np.interp(common_time, full_result.t_au, full_result.rho22),
        "full_mu_mnp_au": np.interp(
            common_time, full_result.t_au, full_result.mu_p_au
        ),
        "full_mu_qd_au": np.interp(
            common_time, full_result.t_au, full_result.mu_d_au
        ),
        "full_mu_total_au": np.interp(
            common_time, full_result.t_au, full_result.mu_total_au
        ),
        "full_mnp_field_at_qd_au": np.interp(
            common_time, full_result.t_au, full_result.mnp_field_at_qd_au
        ),
        "full_effective_qd_field_au": np.interp(
            common_time, full_result.t_au, full_result.effective_qd_field_au
        ),
    }

    alpha_time_legacy = spectral_effective_alpha_au(
        legacy_result,
        pulse,
        params.eps_m,
    )
    alpha_time_full = spectral_effective_alpha_au(
        full_result,
        pulse,
        params.eps_m,
    )
    alpha_linear_legacy = linear_coupled_alpha_au(
        legacy_model,
        np.asarray([pulse_energy_eV]),
        mnp_response="fit",
    )[0][0]
    full_frequency_response = full_model.frequency_response_from_fit(
        np.asarray([pulse_energy_eV])
    )
    legacy_frequency_response = LegacyDipoleInteraction(
        legacy_model
    ).frequency_response(
        np.asarray([pulse_energy_eV]),
        mnp_response="fit",
    )
    beta = qd_linear_polarizability_from_params(
        params,
        np.asarray([pulse_energy_eV]),
    )
    alpha_linear_full = solve_linear_hybrid_response(
        full_frequency_response,
        beta,
        eps_m=params.eps_m,
    ).alpha_effective_au3[0]

    summary_rows = []
    for name, result, rho22, alpha_time, alpha_linear in (
        (
            "legacy",
            legacy_result,
            legacy_rho22,
            alpha_time_legacy,
            alpha_linear_legacy,
        ),
        (
            "spheroid_full",
            full_result,
            full_result.rho22,
            alpha_time_full,
            alpha_linear_full,
        ),
    ):
        maximum_index = int(np.argmax(rho22))
        diagnostics = result.diagnostics
        row = {
            "model": name,
            "excited_population_max": float(np.max(rho22)),
            "excited_population_final": float(rho22[-1]),
            "time_of_population_max_fs": float(
                au_to_fs(result.t_au[maximum_index])
            ),
            "work_from_incident_field_j": float(result.work_from_incident_field_j),
            "sigma_energy_transfer_cm2": float(result.sigma_energy_transfer_cm2),
            "max_bloch_radius": float(diagnostics.max_bloch_radius),
            "min_density_eigenvalue": float(diagnostics.min_density_eigenvalue),
            "solver_steps": int(diagnostics.n_steps),
            "solver_nfev": int(diagnostics.nfev),
            "response_tail_ratio": float(tail_ratio_by_model[name]),
            "response_tail_converged": bool(
                tail_ratio_by_model[name] <= tail_ratio_tolerance
            ),
            "alpha_time_real": float(alpha_time.real),
            "alpha_time_imag": float(alpha_time.imag),
            "alpha_linear_real": float(alpha_linear.real),
            "alpha_linear_imag": float(alpha_linear.imag),
            "carrier_fourier_vs_weak_linear_relative_difference": float(
                abs(alpha_time - alpha_linear)
                / max(abs(alpha_linear), np.finfo(float).tiny)
            ),
        }
        summary_rows.append(row)

    rho_difference = traces["full_rho22"] - traces["legacy_rho22"]
    mu_difference = traces["full_mu_total_au"] - traces["legacy_mu_total_au"]
    direct_response = kernel.response_from_material(
        params.material,
        np.asarray([pulse_energy_eV]),
    )
    fit_response = full_frequency_response
    common_physical_parameters = params_to_physical_dict(params, orientation)
    common_physical_parameters["directional_mnp_radius_to_separation"] = float(
        (params.a_au if qd_placement == "side" else params.c_au) / params.R_au
    )
    legacy_coupling_label = common_physical_parameters.pop("coupling_model")
    exact_spatial_mode_count = int(full_model.exact_spatial_mode_count)
    dynamic_spatial_mode_count = int(full_model.n_spatial_modes)
    reduced_dark_node_count = (
        0 if dark_reduction is None else int(dark_reduction.node_count)
    )
    if qd_placement == "side":
        exact_mode_table = [
            {
                "index": index,
                "n": int(degree),
                "m": int(order),
                "sector": str(sector),
            }
            for index, (degree, order, sector) in enumerate(
                zip(
                    kernel.mode_degrees,
                    kernel.mode_orders,
                    kernel.mode_sectors,
                )
            )
        ]
        bright_index = int(kernel.bright_mode_index)
        bright_mode = exact_mode_table[bright_index]
        spatial_kernel_name = (
            "analytic_sphere_equatorial_green_series"
            if kernel.is_spherical
            else "analytic_prolate_spheroid_equatorial_green_series"
        )
        retained_mode_semantics = (
            "all symmetry-allowed real (n,m,sector) modes through n_max"
        )
    else:
        exact_mode_table = None
        bright_index = int(getattr(kernel, "bright_mode_index", 0))
        bright_mode = {"index": bright_index, "n": 1}
        spatial_kernel_name = (
            "analytic_sphere_green_series"
            if kernel.is_spherical
            else "analytic_prolate_spheroid_green_series"
        )
        retained_mode_semantics = "all integer n in the inclusive range"

    if dark_reduction is None:
        dynamic_mode_table = [
            {
                "full_modal_outputs_au_row": dynamic_index,
                "role": (
                    "bright"
                    if dynamic_index == int(full_model.bright_mode_index)
                    else "exact_dark"
                ),
                "depolarization": float(
                    full_model.modal_depolarization[dynamic_index]
                ),
                "reaction_weight_au_minus3": float(
                    full_model.modal_reaction_weights_au_minus3[dynamic_index]
                ),
                "source_mode_indices": [dynamic_index],
            }
            for dynamic_index in range(dynamic_spatial_mode_count)
        ]
    else:
        dynamic_mode_table = [
            {
                "full_modal_outputs_au_row": 0,
                "role": "bright",
                "depolarization": float(full_model.modal_depolarization[0]),
                "reaction_weight_au_minus3": float(
                    full_model.modal_reaction_weights_au_minus3[0]
                ),
                "source_mode_indices": [bright_index],
            }
        ]
        dynamic_mode_table.extend(
            {
                "full_modal_outputs_au_row": node_index + 1,
                "role": "reduced_dark_node",
                "depolarization": float(depolarization),
                "reaction_weight_au_minus3": float(weight),
                "source_mode_indices": [
                    int(source_index) for source_index in source_indices
                ],
            }
            for node_index, (depolarization, weight, source_indices) in enumerate(
                zip(
                    dark_reduction.depolarization_nodes,
                    dark_reduction.weights_au_minus3,
                    dark_reduction.source_mode_indices,
                )
            )
        )

    if dark_reduction is None:
        dark_reduction_metadata = {
            "applied": False,
            "backend": "direct_full_modal_reference",
            "selection_rule": (
                "axis backward-compatible direct model"
                if qd_placement == "axis"
                else "side direct reference for n_max <= "
                f"{SIDE_DIRECT_REFERENCE_ORDER_MAX}"
            ),
            "fit_grid_points": int(reduction_fit_grid_points),
            "audit_grid_points": int(reduction_audit_grid_points),
            "rms_tolerance": float(reduction_rms_tolerance),
            "max_tolerance": float(reduction_max_tolerance),
            "policy": reduction_policy,
            "certificate": None,
        }
    else:
        reduction_diagnostics = dark_reduction.diagnostics
        reduction_reaudit = full_model.dark_reduction_reaudit_diagnostics
        if reduction_reaudit is None or not reduction_reaudit.accepted:
            raise RuntimeError(
                "Internal error: the active material transfer lacks an accepted "
                "dark-reduction re-audit certificate."
            )
        dark_reduction_metadata = {
            "applied": True,
            "backend": "exact_bright_plus_positive_dark_measure_reduction",
            "selection_rule": "side production model for n_max > "
            f"{SIDE_DIRECT_REFERENCE_ORDER_MAX}",
            "fit_grid_points": int(reduction_fit_grid_points),
            "audit_grid_points": int(reduction_audit_grid_points),
            "rms_tolerance": float(reduction_rms_tolerance),
            "max_tolerance": float(reduction_max_tolerance),
            "maximum_nodes": (
                None if reduction_max_nodes is None else int(reduction_max_nodes)
            ),
            "policy": reduction_policy,
            "certificate": {
                "source_measure_sha256": dark_reduction.source_measure_sha256,
                "accepted": bool(reduction_diagnostics.accepted),
                "passive_on_audit_grid": bool(
                    reduction_diagnostics.passive_on_audit_grid
                ),
                "fit_normalized_rms": float(
                    reduction_diagnostics.fit_normalized_rms
                ),
                "fit_max_normalized_error": float(
                    reduction_diagnostics.fit_max_normalized_error
                ),
                "audit_normalized_rms": float(
                    reduction_diagnostics.audit_normalized_rms
                ),
                "audit_max_normalized_error": float(
                    reduction_diagnostics.audit_max_normalized_error
                ),
                "max_normalized_rms": float(
                    reduction_diagnostics.max_normalized_rms
                ),
                "max_normalized_error": float(
                    reduction_diagnostics.max_normalized_error
                ),
                "total_weight_relative_error": float(
                    reduction_diagnostics.total_weight_relative_error
                ),
                "first_moment_relative_error": float(
                    reduction_diagnostics.first_moment_relative_error
                ),
                "original_dark_mode_count": int(
                    reduction_diagnostics.original_dark_mode_count
                ),
                "positive_dark_mode_count": int(
                    reduction_diagnostics.positive_dark_mode_count
                ),
                "reduced_node_count": int(
                    reduction_diagnostics.reduced_node_count
                ),
                "depolarization_nodes": [
                    float(value)
                    for value in dark_reduction.depolarization_nodes
                ],
                "weights_au_minus3": [
                    float(value) for value in dark_reduction.weights_au_minus3
                ],
                "source_mode_indices": [
                    [int(index) for index in group]
                    for group in dark_reduction.source_mode_indices
                ],
                "current_transfer_reaudit": {
                    "accepted": bool(reduction_reaudit.accepted),
                    "passive_on_audit_grid": bool(
                        reduction_reaudit.passive_on_audit_grid
                    ),
                    "normalized_rms": float(reduction_reaudit.normalized_rms),
                    "max_normalized_error": float(
                        reduction_reaudit.max_normalized_error
                    ),
                    "minimum_imaginary_part": float(
                        reduction_reaudit.minimum_imaginary_part
                    ),
                    "audit_grid_points": int(
                        reduction_reaudit.audit_grid_points
                    ),
                    "rms_tolerance": float(reduction_reaudit.rms_tolerance),
                    "max_tolerance": float(reduction_reaudit.max_tolerance),
                    "energy_window_eV": [
                        float(value)
                        for value in reduction_reaudit.energy_window_eV
                    ],
                },
            },
        }
    metadata = {
        "pulse_comparison_schema_version": PULSE_COMPARISON_SCHEMA_VERSION,
        "created_at": datetime.now().astimezone().isoformat(),
        "execution": "concurrent_threads" if concurrent else "sequential",
        "implementation": {
            "legacy": "HybridQDPlasmonModel through its unchanged public API",
            "spheroid_full": "FullQSSpheroidPulseModel",
            "spheroid_full_time_backend": dark_reduction_metadata["backend"],
        },
        "models": ["legacy", "spheroid_full"],
        "physical_parameters": common_physical_parameters,
        "coupling_by_model": {
            "legacy": {
                "spatial_kernel": legacy_coupling_label,
                "frequency_response_source": "causal_fit_used_in_time_domain",
                "spatial_degrees_of_freedom": "one central induced point dipole",
                "exact_spheroidal_projection": False,
                "J_au_minus3": float(legacy_model.J),
                "B_at_carrier": _complex_metadata(
                    legacy_frequency_response.B[0]
                ),
                "K_at_carrier_au_minus3": _complex_metadata(
                    legacy_frequency_response.K_au_minus3[0]
                ),
                "identity": "B=A*J; K=A*J^2",
            },
            "spheroid_full": {
                "spatial_kernel": spatial_kernel_name,
                "qd_placement": qd_placement,
                "side_transverse_alignment": side_transverse_alignment,
                "frequency_response_source": (
                    "same_causal_bright_fit_transformed_by_exact_or_reduced_spatial_mode"
                ),
                "retained_spheroidal_orders": [1, spatial_order_max],
                "retained_order_semantics": retained_mode_semantics,
                "exact_spatial_mode_count": exact_spatial_mode_count,
                "dynamic_spatial_mode_count": dynamic_spatial_mode_count,
                "reduced_dark_node_count": reduced_dark_node_count,
                "bright_mode": bright_mode,
                "exact_equatorial_mode_table": exact_mode_table,
                "full_modal_outputs_au_row_metadata": (
                    "full_qs.full_modal_outputs_au_rows"
                ),
                "uniform_laser_drive_orders": [1],
                "point_qd_reaction_orders": [1, spatial_order_max],
                "reported_mnp_dipole_orders": [1],
                "bright_source_coupling_au_minus3": float(
                    full_model.bright_coupling_au_minus3
                ),
                "B_at_carrier": _complex_metadata(fit_response.B[0]),
                "K_at_carrier_au_minus3": _complex_metadata(
                    fit_response.K_au_minus3[0]
                ),
                "K_bright_at_carrier_au_minus3": _complex_metadata(
                    fit_response.K_bright_au_minus3[0]
                ),
                "K_higher_at_carrier_au_minus3": _complex_metadata(
                    fit_response.K_higher_au_minus3[0]
                ),
            },
        },
        "pulse": {
            "energy_eV": pulse_energy_eV,
            "E0_au": pulse_E0_au,
            "tau_fs": pulse_tau_fs,
            "tau_kind": pulse_tau_kind,
            "fluence_j_cm2": pulse.fluence_j_cm2(eps_m=params.eps_m),
            "peak_intensity_w_cm2": pulse.peak_intensity_w_cm2(
                eps_m=params.eps_m
            ),
        },
        "time_window": {
            "start_fs": float(au_to_fs(t_span[0])),
            "end_fs": float(au_to_fs(t_span[1])),
            "requested_post_fs": None if post_fs is None else float(post_fs),
            "initial_post_fs": float(au_to_fs(initial_end_au)),
            "post_was_automatic": automatic_post,
            "automatic_extension_count": tail_extension_count,
            "maximum_automatic_extensions": max_auto_tail_extensions,
            "tail_window_fraction": tail_window_fraction,
            "tail_ratio_tolerance": tail_ratio_tolerance,
            "tail_policy": tail_policy,
            "tail_ratio_by_model": tail_ratio_by_model,
            "all_model_tails_converged": tails_converged,
            "common_time_points_requested": common_time_points,
            "common_time_points_required_by_frequency_ceiling": (
                required_common_time_points
            ),
            "common_time_points_resolved": resolved_common_time_points,
        },
        "common_solver_policies": {
            "spectral_window_policy": spectral_window_policy,
            "max_spectral_leakage": max_spectral_leakage,
            "positivity_policy": positivity_policy,
            "positivity_tolerance": positivity_tolerance,
            "tail_policy": tail_policy,
            "tail_ratio_tolerance": tail_ratio_tolerance,
            "tail_window_fraction": tail_window_fraction,
        },
        "model_specific_policies": {
            "legacy": {
                "work_passivity_policy": "raise (enforced by legacy core)",
            },
            "spheroid_full": {
                "fit_quality_policy": fit_quality_policy,
                "max_modal_normalized_rms": max_modal_normalized_rms,
                "max_modal_relative_error": max_modal_relative_error,
                "spatial_convergence_policy": spatial_convergence_policy,
                "work_passivity_policy": work_passivity_policy,
            },
        },
        "full_qs": {
            "spatial_order_max": spatial_order_max,
            "exact_spatial_mode_count": exact_spatial_mode_count,
            "dynamic_spatial_mode_count": dynamic_spatial_mode_count,
            "reduced_dark_node_count": reduced_dark_node_count,
            "full_modal_outputs_au_rows": dynamic_mode_table,
            "dark_reduction": dark_reduction_metadata,
            "spatial_convergence_policy": spatial_convergence_policy,
            "spatial_convergence_rtol": spatial_convergence_rtol,
            "material_fit_modes_per_spatial_order": material_fit_modes,
            "modal_fit_normalized_rms_limit": max_modal_normalized_rms,
            "modal_fit_relative_error_limit": max_modal_relative_error,
            "asymptotic_order_ratio": kernel.asymptotic_order_ratio,
            "carrier_half_order_relative_change": float(
                direct_response.relative_half_order_change()[0]
            ),
            "carrier_tail_block_relative_mass": float(
                direct_response.relative_tail_block()[0]
            ),
            "modal_fit_max_normalized_rms": (
                full_model.modal_fit_diagnostics.max_normalized_rms
            ),
            "modal_fit_max_relative_error": (
                full_model.modal_fit_diagnostics.max_relative_error
            ),
            "modal_fit_K_normalized_rms": (
                full_model.modal_fit_diagnostics.K_normalized_rms
            ),
            "modal_fit_K_max_relative_error": (
                full_model.modal_fit_diagnostics.K_max_relative_error
            ),
            "modal_fit_passive_on_audit_grid": (
                full_model.modal_fit_diagnostics.passive_on_audit_grid
            ),
            "modal_fit_accepted": full_model.modal_fit_diagnostics.accepted,
            "fit_window_max_half_order_relative_change": (
                full_model.spatial_convergence_diagnostics.max_half_order_relative_change
            ),
            "fit_window_max_tail_block_relative_mass": (
                full_model.spatial_convergence_diagnostics.max_tail_block_relative_mass
            ),
            "spatial_series_converged": (
                full_model.spatial_convergence_diagnostics.accepted
            ),
            "carrier_fit_vs_material_K_relative_error": float(
                abs(
                    fit_response.K_au_minus3[0]
                    - direct_response.K_au_minus3[0]
                )
                / max(abs(direct_response.K_au_minus3[0]), np.finfo(float).tiny)
            ),
            "coupled_spectral_abscissa_au": (
                full_model.coupled_stability.spectral_abscissa_au
            ),
            "coupled_spectral_abscissa_available": (
                full_model.coupled_stability.spectral_abscissa_available
            ),
            "coupled_spectral_abscissa_is_bound": (
                full_model.coupled_stability.spectral_abscissa_is_bound
            ),
            "decay_rate_estimate_au": (
                full_model.coupled_stability.decay_rate_estimate_au
            ),
            "decay_rate_estimate_is_exact": (
                full_model.coupled_stability.decay_rate_estimate_is_exact
            ),
            "coupled_stability_certificate": (
                full_model.coupled_stability.eigensolver
            ),
        },
        "common_grid_metrics": {
            "rho22_rms_difference": float(np.sqrt(np.mean(rho_difference**2))),
            "rho22_max_absolute_difference": float(np.max(np.abs(rho_difference))),
            "mu_total_rms_difference_au": float(
                np.sqrt(np.mean(mu_difference**2))
            ),
        },
        "scope_limits": [
            "strict quasistatics and point QD",
            "the selected spatial_order_max must be checked for convergence",
            "external work is not a pure-metal absorption partition",
            "no automatic geometry-derived spontaneous-emission correction",
        ],
    }

    run_dir = _create_unique_run_dir(output_dir)
    _write_csv(run_dir / "pulse_summary.csv", summary_rows)
    trace_rows = [
        {name: float(values[index]) for name, values in traces.items()}
        for index in range(resolved_common_time_points)
    ]
    _write_csv(run_dir / "pulse_traces_common_grid.csv", trace_rows)
    np.savez_compressed(run_dir / "pulse_traces.npz", **traces)
    np.savez_compressed(
        run_dir / "pulse_traces_adaptive.npz",
        legacy_t_au=legacy_result.t_au,
        legacy_W=legacy_W,
        legacy_Q=legacy_Q,
        legacy_P=legacy_P,
        legacy_rho22=legacy_rho22,
        legacy_mu_mnp_au=legacy_result.mu_p_au,
        legacy_mu_qd_au=legacy_result.mu_d_au,
        legacy_mu_total_au=legacy_result.mu_total_au,
        legacy_mnp_field_at_qd_au=legacy_mnp_field,
        legacy_effective_qd_field_au=legacy_effective_field,
        full_t_au=full_result.t_au,
        full_W=full_result.W,
        full_Q=full_result.Q,
        full_P=full_result.P,
        full_rho22=full_result.rho22,
        full_mu_mnp_au=full_result.mu_p_au,
        full_mu_qd_au=full_result.mu_d_au,
        full_mu_total_au=full_result.mu_total_au,
        full_mnp_field_at_qd_au=full_result.mnp_field_at_qd_au,
        full_effective_qd_field_au=full_result.effective_qd_field_au,
        full_modal_outputs_au=full_result.modal_outputs_au,
    )
    with (run_dir / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)

    figure: Figure | None = None
    if make_plots:
        figure = _plot_pulse_comparison(run_dir, traces)
    if show and make_plots:
        plt.show()
    elif figure is not None:
        plt.close(figure)
    return run_dir


def _plot_pulse_comparison(
    run_dir: Path,
    traces: dict[str, np.ndarray],
) -> Figure:
    time_fs = traces["time_fs"]
    figure, axes = plt.subplots(3, 1, figsize=(9.0, 10.0), sharex=True)
    axes[0].plot(time_fs, traces["legacy_rho22"], color="tab:gray", label="legacy")
    axes[0].plot(time_fs, traces["full_rho22"], color="tab:red", label="full spheroid")
    axes[0].set_ylabel(r"$\rho_{22}$")
    axes[0].legend()

    axes[1].plot(
        time_fs,
        traces["legacy_mu_total_au"],
        color="tab:gray",
        label="legacy",
    )
    axes[1].plot(
        time_fs,
        traces["full_mu_total_au"],
        color="tab:red",
        label="full spheroid",
    )
    axes[1].set_ylabel("Total dipole, au")

    axes[2].plot(
        time_fs,
        traces["legacy_effective_qd_field_au"],
        color="tab:gray",
        label="legacy",
    )
    axes[2].plot(
        time_fs,
        traces["full_effective_qd_field_au"],
        color="tab:red",
        label="full spheroid",
    )
    axes[2].set_ylabel("Microscopic QD field, au")
    axes[2].set_xlabel("Time, fs")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(run_dir / "pulse_dynamics.png", dpi=180)
    return figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/spheroid_pulse_comparison")
    parser.add_argument("--orientation", choices=("long", "trans"), default="long")
    parser.add_argument(
        "--qd-placement",
        choices=("axis", "side"),
        default="axis",
        help="Place the QD on the spheroid major axis or on an equatorial side axis.",
    )
    parser.add_argument(
        "--side-transverse-alignment",
        choices=("radial", "tangential"),
        help="For --qd-placement side --orientation trans, select E parallel "
        "to R (radial) or perpendicular to both R and the spheroid axis "
        "(tangential).",
    )
    parser.add_argument("--spatial-order-max", type=int, default=80)
    parser.add_argument("--material-fit-modes", type=int, default=9)
    parser.add_argument("--reduction-fit-grid-points", type=int, default=1001)
    parser.add_argument("--reduction-audit-grid-points", type=int, default=1601)
    parser.add_argument("--reduction-rms-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--reduction-max-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--reduction-max-nodes", type=int)
    parser.add_argument(
        "--reduction-policy",
        choices=tuple(sorted(POLICIES)),
        default="raise",
        help="Diagnostic action if reduction misses its audit gate; the "
        "production backend never propagates an uncertified reduction.",
    )
    parser.add_argument("--pulse-energy-ev", type=float, default=2.042)
    parser.add_argument("--pulse-tau-fs", type=float, default=5.0)
    parser.add_argument("--pulse-e0-au", type=float, default=1.0e-5)
    post_group = parser.add_mutually_exclusive_group()
    post_group.add_argument(
        "--post-fs",
        type=float,
        default=None,
        help="Explicit final time after the pulse centre in fs. By default a "
        "shared old/new window is extended until both dipole tails converge.",
    )
    post_group.add_argument(
        "--auto-post",
        action="store_true",
        help="Request the default automatic common tail (compatibility flag).",
    )
    parser.add_argument("--common-time-points", type=int, default=2001)
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
    parser.add_argument("--method", choices=("DOP853", "RK45", "Radau", "BDF", "LSODA"), default="DOP853")
    parser.add_argument("--rtol", type=float, default=1.0e-8)
    parser.add_argument("--atol", type=float, default=1.0e-10)
    parser.add_argument(
        "--spectral-window-policy",
        choices=tuple(sorted(POLICIES)),
        default="raise",
        help="Common old/new action for response outside the fitted material window.",
    )
    parser.add_argument("--max-spectral-leakage", type=float, default=1.0e-3)
    parser.add_argument(
        "--positivity-policy",
        choices=tuple(sorted(POLICIES)),
        default="raise",
        help="Common old/new action if the QD density matrix leaves the Bloch ball.",
    )
    parser.add_argument("--positivity-tolerance", type=float, default=1.0e-7)
    parser.add_argument(
        "--tail-policy",
        choices=tuple(sorted(POLICIES)),
        default="raise",
        help="Common old/new action if either component-wise dipole tail fails.",
    )
    parser.add_argument("--tail-ratio-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--tail-window-fraction", type=float, default=0.05)
    parser.add_argument("--max-auto-tail-extensions", type=int, default=3)
    parser.add_argument(
        "--fit-quality-policy",
        choices=tuple(sorted(POLICIES)),
        default="raise",
    )
    parser.add_argument("--max-modal-normalized-rms", type=float, default=0.03)
    parser.add_argument("--max-modal-relative-error", type=float, default=0.06)
    parser.add_argument(
        "--spatial-convergence-policy",
        choices=tuple(sorted(POLICIES)),
        default="raise",
        help="Action if the retained full-QS spatial series fails its fit-window audit.",
    )
    parser.add_argument("--spatial-convergence-rtol", type=float, default=1.0e-8)
    parser.add_argument(
        "--work-passivity-policy",
        choices=tuple(sorted(POLICIES)),
        default="raise",
    )
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = run_pulse_comparison(
        output_dir=args.output_dir,
        orientation=args.orientation,
        qd_placement=args.qd_placement,
        side_transverse_alignment=args.side_transverse_alignment,
        spatial_order_max=args.spatial_order_max,
        material_fit_modes=args.material_fit_modes,
        reduction_fit_grid_points=args.reduction_fit_grid_points,
        reduction_audit_grid_points=args.reduction_audit_grid_points,
        reduction_rms_tolerance=args.reduction_rms_tolerance,
        reduction_max_tolerance=args.reduction_max_tolerance,
        reduction_max_nodes=args.reduction_max_nodes,
        reduction_policy=args.reduction_policy,
        pulse_energy_eV=args.pulse_energy_ev,
        pulse_tau_fs=args.pulse_tau_fs,
        pulse_E0_au=args.pulse_e0_au,
        post_fs=None if args.auto_post else args.post_fs,
        common_time_points=args.common_time_points,
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
        method=args.method,
        rtol=args.rtol,
        atol=args.atol,
        spectral_window_policy=args.spectral_window_policy,
        max_spectral_leakage=args.max_spectral_leakage,
        positivity_policy=args.positivity_policy,
        positivity_tolerance=args.positivity_tolerance,
        tail_policy=args.tail_policy,
        tail_ratio_tolerance=args.tail_ratio_tolerance,
        tail_window_fraction=args.tail_window_fraction,
        max_auto_tail_extensions=args.max_auto_tail_extensions,
        fit_quality_policy=args.fit_quality_policy,
        max_modal_normalized_rms=args.max_modal_normalized_rms,
        max_modal_relative_error=args.max_modal_relative_error,
        spatial_convergence_policy=args.spatial_convergence_policy,
        spatial_convergence_rtol=args.spatial_convergence_rtol,
        work_passivity_policy=args.work_passivity_policy,
        concurrent=not args.sequential,
        make_plots=not args.no_plots,
        show=args.show,
    )
    print(f"Saved pulse comparison to {run_dir}")


if __name__ == "__main__":
    main()
