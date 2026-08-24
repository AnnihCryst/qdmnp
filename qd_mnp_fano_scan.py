"""Search weak-field Fano-like suppression dips inside the native QD--MNP model.

The quasistatic dipole tensor stays physical: ``G=2`` for a dipole parallel
to the QD--MNP centre axis and ``G=-1`` for a transverse dipole.  The scan
varies the centre separation ``R`` and therefore the physical coupling
``J=G/(eps_m*R**3)`` instead of using an unbounded phenomenological ``G``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import warnings

import numpy as np

from qd_mnp_linear_spectrum import (
    MODAL_OBSERVABLE_NORMALIZED_MAX_ERROR_LIMIT,
    enforce_passive_work_loss_cm2,
    qd_linear_polarizability_au,
    quasistatic_work_loss_cross_section_cm2,
)
from qd_mnp_params import DEBYE_C_M, make_params_with_overrides
from qd_mnp_rational_fit import (
    AU_DIPOLE_C_M,
    HybridQDPlasmonModel,
    NATIVE_MODEL_PROFILE,
    SCHEMA_VERSION,
    au_to_eV,
    au_to_nm,
    eV_to_au,
    homogeneous_radiative_decay_rate_au,
    nm_to_au,
    params_to_physical_dict,
)


FANO_MODAL_TARGET_RATIO_ABSOLUTE_ERROR_LIMIT = 0.01
FANO_MODAL_MIN_RATIO_ABSOLUTE_ERROR_LIMIT = 0.01
FANO_MODAL_MINIMUM_ENERGY_SHIFT_LIMIT_EV = 0.0015
FANO_MAX_SCAN_ENERGY_STEP_EV = 0.0005
FANO_MAX_SCAN_STEP_FRACTION_OF_GAMMA2 = 0.25


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
    r_min_nm: float,
    r_max_nm: float,
    r_points: int,
    r_spacing: str,
    fit_window_ev: tuple[float, float],
    weight_center_ev: float | None,
    weight_sigma_ev: float | None,
    eps_m: float | None,
    eps_qd: float | None = None,
    qd_dipole_convention: str = "effective_external",
    qd_radius_nm: float | None = None,
    c_nm: float | None = None,
    a_nm: float | None = None,
    gamma_population_mev: float | None = None,
    orientation: str = "long",
    n_modes: int = 9,
) -> list[dict[str, float | str | bool]]:
    """Return stable Fano-like suppression candidates for physical parameters.

    A low coupled/bare ratio is a necessary interference signature but does
    not by itself fit an asymmetric Fano line shape or determine a Fano q
    parameter.  Rows are therefore candidates, not automatically established
    Fano resonances.  Numerical modal/material convergence, actual suppression
    below unity, and conservative physical-applicability guides are exported
    as separate flags; every finite stable row remains available for audit.

    The historical ``gamma_dephasing_mev_values`` alias is retained, but it
    denotes the total coherence width Gamma2, not pure dephasing.
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
    if grid_points < 3 or grid_points % 2 == 0 or omega0_points < 1 or r_points < 1:
        raise ValueError(
            "grid_points must be odd and >=3 so target_ev is sampled exactly; "
            "omega0_points/r_points must be >=1."
        )
    if not np.isfinite(target_ev) or target_ev <= 0.0:
        raise ValueError("target_ev must be finite and positive.")
    if not np.isfinite(window_ev) or window_ev <= 0.0:
        raise ValueError("window_ev must be finite and positive.")
    if target_ev - window_ev < fit_window_ev[0] or target_ev + window_ev > fit_window_ev[1]:
        raise ValueError("The scanned energy interval must lie inside fit_window_ev.")
    if not (
        np.isfinite(omega0_min_ev)
        and np.isfinite(omega0_max_ev)
        and 0.0 < omega0_min_ev <= omega0_max_ev
    ):
        raise ValueError("QD transition-energy bounds must be finite, positive and ordered.")
    dipoles = np.asarray(d_debye_values, dtype=float)
    if dipoles.size == 0 or np.any(~np.isfinite(dipoles)) or np.any(dipoles <= 0.0):
        raise ValueError("d_debye_values must contain positive finite dipoles.")
    if not (
        np.isfinite(r_min_nm)
        and np.isfinite(r_max_nm)
        and 0.0 < r_min_nm <= r_max_nm
    ):
        raise ValueError("R bounds must be finite, positive and ordered.")
    if r_spacing not in {"linear", "log"}:
        raise ValueError("r_spacing must be 'linear' or 'log'.")
    if orientation not in {"long", "trans"}:
        raise ValueError("orientation must be 'long' or 'trans'.")

    # The material fit is independent of R.  Construct at the weakest scanned
    # coupling, then evaluate a candidate-specific full Jacobian below.
    params = make_params_with_overrides(
        eps_m=eps_m,
        eps_qd=eps_qd,
        qd_dipole_convention=qd_dipole_convention,
        r_nm=r_max_nm,
        qd_radius_nm=qd_radius_nm,
        c_nm=c_nm,
        a_nm=a_nm,
        d_debye=0.0,
        omega0_ev=target_ev,
        gamma_population_mev=gamma_population_mev,
        gamma2_coherence_mev=float(gamma2_values[0]),
        orientation=orientation,
    )
    contact_distance_nm = float(au_to_nm(params.c_au + params.qd_radius_au))
    if r_min_nm <= contact_distance_nm:
        raise ValueError(
            "Every scanned separation must preserve a positive surface gap: "
            f"R_min={r_min_nm:g} nm, c+r_QD={contact_distance_nm:g} nm."
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
        orientation=orientation,
        n_modes=n_modes,
        fit_window_eV=fit_window_ev,
        weight_center_eV=weight_center_ev,
        weight_sigma_eV=weight_sigma_ev,
        alpha_objective_weight=1.0,
        inv_alpha_objective_weight=1.2,
        verbose=True,
    )

    energies = np.linspace(target_ev - window_ev, target_ev + window_ev, grid_points)
    energies[grid_points // 2] = target_ev
    omega = np.asarray(eV_to_au(energies), dtype=float)
    scan_energy_step_ev = float(np.max(np.diff(energies)))
    scan_absolute_grid_resolved = bool(
        scan_energy_step_ev
        <= FANO_MAX_SCAN_ENERGY_STEP_EV * (1.0 + 1.0e-12)
    )
    target_index = grid_points // 2
    alpha_p = model.C * model.alpha_from_fit(energies)
    alpha_p_material = model.C * model.alpha_from_material(energies)
    sigma_qs_work_bare = quasistatic_work_loss_cross_section_cm2(
        alpha_p / params.eps_m,
        omega,
        params.eps_m,
    )
    sigma_qs_work_bare_material = quasistatic_work_loss_cross_section_cm2(
        alpha_p_material / params.eps_m,
        omega,
        params.eps_m,
    )
    if np.any(~np.isfinite(sigma_qs_work_bare)) or np.any(sigma_qs_work_bare <= 0.0):
        raise RuntimeError(
            "The passive bare-MNP reference must have finite positive QS work loss "
            "throughout the Fano scan window."
        )
    if (
        np.any(~np.isfinite(sigma_qs_work_bare_material))
        or np.any(sigma_qs_work_bare_material <= 0.0)
    ):
        raise RuntimeError(
            "The direct material-polarizability reference must have finite "
            "positive QS work loss throughout the Fano scan window."
        )
    omega0_values = np.linspace(omega0_min_ev, omega0_max_ev, omega0_points)
    applicability_energy_ev = float(np.max(energies))
    applicability = model.dipole_applicability_diagnostics(
        energy_eV=applicability_energy_ev
    )
    if r_spacing == "log":
        r_values_nm = np.logspace(np.log10(r_min_nm), np.log10(r_max_nm), r_points)
    else:
        r_values_nm = np.linspace(r_min_nm, r_max_nm, r_points)

    rows: list[dict[str, float | str | bool]] = []
    physical_provenance = params_to_physical_dict(
        params,
        orientation=orientation,
    )
    unstable_candidate_count = 0
    modal_observable_unconverged_count = 0
    radiatively_inconsistent_pairs: set[tuple[float, float]] = set()
    for d_debye in dipoles:
        d_au = float(d_debye * DEBYE_C_M / AU_DIPOLE_C_M)
        for omega0_ev in omega0_values:
            omega0_au = float(eV_to_au(omega0_ev))
            d_external_au = float(params.qd_local_field_factor * d_au)
            gamma_rad_au = homogeneous_radiative_decay_rate_au(
                d_external_au,
                omega0_au,
                params.eps_m,
            )
            gamma1_over_gamma_rad = (
                np.inf
                if gamma_rad_au == 0.0
                else float(params.gamma_au / gamma_rad_au)
            )
            homogeneous_host_consistent = bool(
                gamma1_over_gamma_rad >= 1.0 - 1.0e-10
            )
            if not homogeneous_host_consistent:
                radiatively_inconsistent_pairs.add(
                    (float(d_debye), float(omega0_ev))
                )
            for gamma2_mev in gamma2_values:
                gamma2_au = float(eV_to_au(gamma2_mev / 1000.0))
                scan_gamma2_relative_max_energy_step_ev = float(
                    FANO_MAX_SCAN_STEP_FRACTION_OF_GAMMA2
                    * gamma2_mev
                    / 1000.0
                )
                scan_effective_max_energy_step_ev = min(
                    FANO_MAX_SCAN_ENERGY_STEP_EV,
                    scan_gamma2_relative_max_energy_step_ev,
                )
                scan_gamma2_relative_grid_resolved = bool(
                    scan_energy_step_ev
                    <= scan_gamma2_relative_max_energy_step_ev
                    * (1.0 + 1.0e-12)
                )
                scan_grid_resolved = bool(
                    scan_absolute_grid_resolved
                    and scan_gamma2_relative_grid_resolved
                )
                beta = qd_linear_polarizability_au(
                    omega,
                    d_au,
                    omega0_au,
                    gamma2_au,
                    params.qd_local_field_factor,
                )
                for separation_nm in r_values_nm:
                    separation_au = float(nm_to_au(separation_nm))
                    stability = model.linearized_ground_state_stability(
                        d_au=d_au,
                        omega0_au=omega0_au,
                        gamma1_au=params.gamma_au,
                        gamma2_au=gamma2_au,
                        g_factor=params.G,
                        R_au=separation_au,
                    )
                    if not stability.stable:
                        unstable_candidate_count += 1
                        continue

                    j_coupling = params.G / (params.eps_m * separation_au**3)
                    denominator = 1.0 - j_coupling**2 * alpha_p * beta
                    mu_p_over_e = alpha_p * (1.0 + j_coupling * beta) / denominator
                    mu_d_over_e = beta * (1.0 + j_coupling * alpha_p) / denominator
                    alpha_eff = (mu_p_over_e + mu_d_over_e) / params.eps_m
                    denominator_material = (
                        1.0 - j_coupling**2 * alpha_p_material * beta
                    )
                    mu_p_over_e_material = (
                        alpha_p_material * (1.0 + j_coupling * beta)
                        / denominator_material
                    )
                    mu_d_over_e_material = (
                        beta * (1.0 + j_coupling * alpha_p_material)
                        / denominator_material
                    )
                    alpha_eff_material = (
                        mu_p_over_e_material + mu_d_over_e_material
                    ) / params.eps_m
                    sigma_qs_work_coupled = quasistatic_work_loss_cross_section_cm2(
                        alpha_eff,
                        omega,
                        params.eps_m,
                    )
                    sigma_qs_work_coupled_material = (
                        quasistatic_work_loss_cross_section_cm2(
                            alpha_eff_material,
                            omega,
                            params.eps_m,
                        )
                    )
                    sigma_qs_work_coupled = enforce_passive_work_loss_cm2(
                        sigma_qs_work_coupled,
                        label="modal coupled Fano-scan response",
                    )
                    sigma_qs_work_coupled_material = (
                        enforce_passive_work_loss_cm2(
                            sigma_qs_work_coupled_material,
                            label="direct-material coupled Fano-scan response",
                        )
                    )
                    ratio = np.divide(
                        sigma_qs_work_coupled,
                        sigma_qs_work_bare,
                        out=np.full_like(
                            sigma_qs_work_coupled, np.nan, dtype=float
                        ),
                        where=sigma_qs_work_bare != 0.0,
                    )
                    ratio_material = np.divide(
                        sigma_qs_work_coupled_material,
                        sigma_qs_work_bare_material,
                        out=np.full_like(
                            sigma_qs_work_coupled_material,
                            np.nan,
                            dtype=float,
                        ),
                        where=sigma_qs_work_bare_material != 0.0,
                    )
                    finite = (
                        np.isfinite(ratio)
                        & np.isfinite(sigma_qs_work_coupled)
                        & np.isfinite(ratio_material)
                        & np.isfinite(sigma_qs_work_coupled_material)
                    )
                    if not finite[target_index] or not np.any(finite):
                        continue
                    finite_indices = np.flatnonzero(finite)
                    min_local_index = int(finite_indices[np.argmin(ratio[finite])])
                    min_material_index = int(
                        finite_indices[np.argmin(ratio_material[finite])]
                    )
                    observable_scale = max(
                        float(np.max(np.abs(sigma_qs_work_coupled_material[finite]))),
                        float(np.max(np.abs(sigma_qs_work_bare_material[finite]))),
                        np.finfo(float).tiny,
                    )
                    modal_observable_normalized_max_error = float(
                        max(
                            np.max(
                                np.abs(
                                    sigma_qs_work_coupled[finite]
                                    - sigma_qs_work_coupled_material[finite]
                                )
                            ),
                            np.max(
                                np.abs(
                                    sigma_qs_work_bare[finite]
                                    - sigma_qs_work_bare_material[finite]
                                )
                            ),
                        )
                        / observable_scale
                    )
                    modal_target_ratio_absolute_error = float(
                        abs(
                            ratio[target_index]
                            - ratio_material[target_index]
                        )
                    )
                    modal_minimum_energy_shift_ev = float(
                        energies[min_local_index] - energies[min_material_index]
                    )
                    modal_min_ratio_absolute_error = float(
                        abs(
                            ratio[min_local_index]
                            - ratio_material[min_material_index]
                        )
                    )
                    modal_observable_converged = bool(
                        modal_observable_normalized_max_error
                        <= MODAL_OBSERVABLE_NORMALIZED_MAX_ERROR_LIMIT
                        and modal_target_ratio_absolute_error
                        <= FANO_MODAL_TARGET_RATIO_ABSOLUTE_ERROR_LIMIT
                        and modal_min_ratio_absolute_error
                        <= FANO_MODAL_MIN_RATIO_ABSOLUTE_ERROR_LIMIT
                        and abs(modal_minimum_energy_shift_ev)
                        <= FANO_MODAL_MINIMUM_ENERGY_SHIFT_LIMIT_EV
                        and scan_grid_resolved
                    )
                    if not modal_observable_converged:
                        modal_observable_unconverged_count += 1
                    gap_nm = float(separation_nm - contact_distance_nm)
                    c_over_r = float(au_to_nm(params.c_au) / separation_nm)
                    qd_over_r = float(au_to_nm(params.qd_radius_au) / separation_nm)
                    kR = float(
                        applicability.medium_size_parameter_kc
                        * separation_au
                        / params.c_au
                    )
                    particle_quasistatic = bool(
                        applicability.particle_quasistatic_guide_satisfied
                    )
                    near_field_coupling = bool(kR <= applicability.guide_threshold)
                    mnp_point_dipole = bool(
                        c_over_r <= applicability.guide_threshold
                    )
                    qd_point_dipole = bool(
                        qd_over_r <= applicability.guide_threshold
                    )
                    point_dipole = bool(mnp_point_dipole and qd_point_dipole)
                    quantitative_physical_applicability = bool(
                        particle_quasistatic
                        and near_field_coupling
                        and point_dipole
                        and homogeneous_host_consistent
                    )
                    suppression_at_target = bool(ratio[target_index] < 1.0)
                    material_reference_suppression_at_target = bool(
                        ratio_material[target_index] < 1.0
                    )
                    suppression_at_target_confirmed_by_material_reference = bool(
                        suppression_at_target
                        and material_reference_suppression_at_target
                    )
                    dip_in_window = bool(ratio[min_local_index] < 1.0)
                    material_reference_dip_in_window = bool(
                        ratio_material[min_material_index] < 1.0
                    )
                    accepted_for_fano_like_suppression_ranking = bool(
                        modal_observable_converged
                        and suppression_at_target_confirmed_by_material_reference
                    )
                    rows.append(
                        {
                            **physical_provenance,
                            "schema_version": SCHEMA_VERSION,
                            "model_profile": NATIVE_MODEL_PROFILE,
                            "target_ev": float(target_ev),
                            "applicability_diagnostic_energy_ev": applicability_energy_ev,
                            "orientation": orientation,
                            "G": float(params.G),
                            "R_nm": float(separation_nm),
                            "qd_radius_nm": float(au_to_nm(params.qd_radius_au)),
                            "surface_gap_nm": gap_nm,
                            "mnp_size_to_separation_ratio_c_over_R": c_over_r,
                            "qd_size_to_separation_ratio_rqd_over_R": qd_over_r,
                            "medium_size_parameter_kc": float(
                                applicability.medium_size_parameter_kc
                            ),
                            "medium_separation_parameter_kR": kR,
                            "particle_quasistatic_guide_satisfied": particle_quasistatic,
                            "near_field_coupling_guide_satisfied": near_field_coupling,
                            "quasistatic_guide_satisfied": bool(
                                particle_quasistatic and near_field_coupling
                            ),
                            "mnp_point_dipole_guide_satisfied": mnp_point_dipole,
                            "qd_point_dipole_guide_satisfied": qd_point_dipole,
                            "point_dipole_guide_satisfied": point_dipole,
                            "quantitative_physical_applicability": (
                                quantitative_physical_applicability
                            ),
                            "mnp_fit_n_modes": int(model.n_modes),
                            "mnp_fit_alpha_inf": float(model.fit.alpha_inf),
                            "mnp_fit_window_min_ev": float(fit_window_ev[0]),
                            "mnp_fit_window_max_ev": float(fit_window_ev[1]),
                            "mnp_fit_normalized_rms_alpha": float(
                                model.fit.normalized_rms_alpha
                            ),
                            "mnp_fit_normalized_rms_inv_alpha": float(
                                model.fit.normalized_rms_inv_alpha
                            ),
                            "mnp_fit_max_relative_alpha_error": float(
                                model.fit.max_normalized_alpha_error
                            ),
                            "mnp_fit_nonnegative_imaginary_part_all_positive_frequencies": bool(
                                model.fit.nonnegative_imaginary_part_all_positive_frequencies
                            ),
                            "mnp_fit_globally_passive": bool(
                                model.fit.nonnegative_imaginary_part_all_positive_frequencies
                            ),
                            "modal_observable_reference": (
                                "direct_interpolated_material_alpha"
                            ),
                            "modal_observable_normalized_max_error": (
                                modal_observable_normalized_max_error
                            ),
                            "modal_observable_error_limit": (
                                MODAL_OBSERVABLE_NORMALIZED_MAX_ERROR_LIMIT
                            ),
                            "modal_target_ratio_absolute_error_limit": (
                                FANO_MODAL_TARGET_RATIO_ABSOLUTE_ERROR_LIMIT
                            ),
                            "modal_min_ratio_absolute_error_limit": (
                                FANO_MODAL_MIN_RATIO_ABSOLUTE_ERROR_LIMIT
                            ),
                            "modal_minimum_energy_shift_limit_ev": (
                                FANO_MODAL_MINIMUM_ENERGY_SHIFT_LIMIT_EV
                            ),
                            "scan_energy_step_ev": scan_energy_step_ev,
                            "scan_absolute_grid_resolved": (
                                scan_absolute_grid_resolved
                            ),
                            "scan_gamma2_relative_grid_resolved": (
                                scan_gamma2_relative_grid_resolved
                            ),
                            "scan_absolute_max_energy_step_ev": (
                                FANO_MAX_SCAN_ENERGY_STEP_EV
                            ),
                            "scan_gamma2_relative_max_energy_step_ev": (
                                scan_gamma2_relative_max_energy_step_ev
                            ),
                            "scan_max_energy_step_ev": (
                                scan_effective_max_energy_step_ev
                            ),
                            "scan_max_step_fraction_of_gamma2": (
                                FANO_MAX_SCAN_STEP_FRACTION_OF_GAMMA2
                            ),
                            "scan_grid_resolved": scan_grid_resolved,
                            "modal_observable_converged": (
                                modal_observable_converged
                            ),
                            "accepted_for_modal_numerical_ranking": (
                                modal_observable_converged
                            ),
                            "suppression_at_target": suppression_at_target,
                            "material_reference_suppression_at_target": (
                                material_reference_suppression_at_target
                            ),
                            "suppression_at_target_confirmed_by_material_reference": (
                                suppression_at_target_confirmed_by_material_reference
                            ),
                            "dip_in_window": dip_in_window,
                            "material_reference_dip_in_window": (
                                material_reference_dip_in_window
                            ),
                            "accepted_for_fano_like_suppression_ranking": (
                                accepted_for_fano_like_suppression_ranking
                            ),
                            "accepted_for_quantitative_ranking": (
                                accepted_for_fano_like_suppression_ranking
                                and quantitative_physical_applicability
                            ),
                            "linearized_ground_state_stable": True,
                            "linearized_ground_state_spectral_abscissa_au": float(
                                stability.spectral_abscissa_au
                            ),
                            "ratio_at_target": float(ratio[target_index]),
                            "ratio_qs_work_loss_at_target": float(
                                ratio[target_index]
                            ),
                            "ratio_qs_work_loss_material_reference_at_target": float(
                                ratio_material[target_index]
                            ),
                            "modal_vs_material_target_ratio_absolute_error": float(
                                modal_target_ratio_absolute_error
                            ),
                            "sigma_qs_work_loss_coupled_at_target_cm2": float(
                                sigma_qs_work_coupled[target_index]
                            ),
                            "sigma_qs_work_loss_bare_at_target_cm2": float(
                                sigma_qs_work_bare[target_index]
                            ),
                            "sigma_qs_work_loss_coupled_material_reference_at_target_cm2": float(
                                sigma_qs_work_coupled_material[target_index]
                            ),
                            "sigma_qs_work_loss_bare_material_reference_at_target_cm2": float(
                                sigma_qs_work_bare_material[target_index]
                            ),
                            "delta_sigma_qs_work_loss_at_target_cm2": float(
                                sigma_qs_work_coupled[target_index]
                                - sigma_qs_work_bare[target_index]
                            ),
                            "min_ratio_in_window": float(ratio[min_local_index]),
                            "min_ratio_energy_ev": float(energies[min_local_index]),
                            "material_reference_min_ratio_in_window": float(
                                ratio_material[min_material_index]
                            ),
                            "modal_vs_material_min_ratio_absolute_error": (
                                modal_min_ratio_absolute_error
                            ),
                            "material_reference_min_ratio_energy_ev": float(
                                energies[min_material_index]
                            ),
                            "modal_vs_material_minimum_energy_shift_ev": float(
                                modal_minimum_energy_shift_ev
                            ),
                            "min_delta_sigma_qs_work_loss_cm2": float(
                                sigma_qs_work_coupled[min_local_index]
                                - sigma_qs_work_bare[min_local_index]
                            ),
                            "omega0_ev": float(omega0_ev),
                            "gamma_population_mev": gamma1_population_mev,
                            "gamma2_coherence_mev": float(gamma2_mev),
                            "gamma_pure_dephasing_mev": float(
                                gamma2_mev - 0.5 * gamma1_population_mev
                            ),
                            "d_debye": float(d_debye),
                            "qd_external_dipole_debye": float(
                                d_external_au
                                * AU_DIPOLE_C_M
                                / DEBYE_C_M
                            ),
                            "homogeneous_radiative_decay_mev": float(
                                au_to_eV(gamma_rad_au) * 1000.0
                            ),
                            "gamma1_over_homogeneous_radiative_rate": float(
                                gamma1_over_gamma_rad
                            ),
                            "gamma1_at_or_above_homogeneous_reference_radiative_rate": homogeneous_host_consistent,
                            "qd_dipole_convention": params.qd_dipole_convention,
                            "eps_m": float(params.eps_m),
                            "eps_qd": float(params.eps_qd),
                            "qd_local_field_factor": float(params.qd_local_field_factor),
                            # Historical aliases: ``ext`` meant the QS
                            # work-loss proxy, not an optical-theorem partition.
                            "sigma_ext_coupled_at_target_cm2": float(
                                sigma_qs_work_coupled[target_index]
                            ),
                            "sigma_ext_bare_at_target_cm2": float(
                                sigma_qs_work_bare[target_index]
                            ),
                            "delta_sigma_ext_at_target_cm2": float(
                                sigma_qs_work_coupled[target_index]
                                - sigma_qs_work_bare[target_index]
                            ),
                            "min_delta_sigma_ext_cm2": float(
                                sigma_qs_work_coupled[min_local_index]
                                - sigma_qs_work_bare[min_local_index]
                            ),
                            "sigma_coupled_at_target_cm2": float(
                                sigma_qs_work_coupled[target_index]
                            ),
                            "sigma_bare_at_target_cm2": float(
                                sigma_qs_work_bare[target_index]
                            ),
                            "delta_at_target_cm2": float(
                                sigma_qs_work_coupled[target_index]
                                - sigma_qs_work_bare[target_index]
                            ),
                            "min_delta_cm2": float(
                                sigma_qs_work_coupled[min_local_index]
                                - sigma_qs_work_bare[min_local_index]
                            ),
                            "gamma_dephasing_mev": float(gamma2_mev),
                        }
                    )

    if unstable_candidate_count:
        warnings.warn(
            f"Excluded {unstable_candidate_count} Fano-like candidate(s) whose full "
            "field-free Jacobian has a pole in the unstable half-plane.",
            RuntimeWarning,
            stacklevel=2,
        )
    if modal_observable_unconverged_count:
        warnings.warn(
            f"{modal_observable_unconverged_count} Fano-like candidate(s) have "
            "not converged against the same coupled algebra evaluated with the "
            "direct interpolated material polarizability. Limits are: normalized "
            f"work-loss error <= {MODAL_OBSERVABLE_NORMALIZED_MAX_ERROR_LIMIT:.3g}, "
            "target-ratio absolute error <= "
            f"{FANO_MODAL_TARGET_RATIO_ABSOLUTE_ERROR_LIMIT:.3g}, minimum-ratio "
            f"absolute error <= {FANO_MODAL_MIN_RATIO_ABSOLUTE_ERROR_LIMIT:.3g}, "
            f"dip-position shift <= {1e3 * FANO_MODAL_MINIMUM_ENERGY_SHIFT_LIMIT_EV:.3g} "
            f"meV, and scan step <= {1e3 * FANO_MAX_SCAN_ENERGY_STEP_EV:.3g} meV "
            f"and <= Gamma2/{1.0 / FANO_MAX_SCAN_STEP_FRACTION_OF_GAMMA2:g}. "
            "Do not use failed rows quantitatively; increase n_modes or improve "
            "the modal fit.",
            RuntimeWarning,
            stacklevel=2,
        )
    if radiatively_inconsistent_pairs:
        warnings.warn(
            f"{len(radiatively_inconsistent_pairs)} unique (d, omega0) scan "
            "pair(s) have gamma1 below the homogeneous-host reference "
            "radiative rate implied by the same external dipole. The scan is "
            "retained as a phenomenological structured-environment result, "
            "but quantitative use requires a sourced d/gamma1 convention.",
            RuntimeWarning,
            stacklevel=2,
        )
    rows.sort(
        key=lambda row: (
            not bool(row["accepted_for_fano_like_suppression_ranking"]),
            float(row["ratio_at_target"]),
        )
    )
    return rows


def write_csv(rows: list[dict[str, float | str | bool]], path: Path) -> None:
    if not rows:
        raise ValueError("The Fano-like scan produced no finite candidates; no CSV was written.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_top(rows: list[dict[str, float | str | bool]], n: int) -> None:
    accepted_rows = [
        row
        for row in rows
        if bool(row.get("accepted_for_fano_like_suppression_ranking", False))
    ]
    print(
        "\n=== Best numerically converged suppression candidates at target "
        "within the native model ==="
    )
    if not accepted_rows:
        print(
            "No row both passed the modal/material and scan-resolution gates "
            "and suppressed the target signal in both the modal and direct-"
            "material evaluations (ratio_at_target < 1)."
        )
        return
    for index, row in enumerate(accepted_rows[:n], start=1):
        print(
            f"{index:2d}. ratio@target={float(row['ratio_at_target']):.4g}, "
            f"min={float(row['min_ratio_in_window']):.4g} at "
            f"{float(row['min_ratio_energy_ev']):.5f} eV, "
            f"omega0={float(row['omega0_ev']):.5f} eV, "
            f"Gamma2={float(row['gamma2_coherence_mev']):.3g} meV, "
            f"d={float(row['d_debye']):.3g} D, "
            f"R={float(row['R_nm']):.3g} nm, "
            f"gap={float(row['surface_gap_nm']):.3g} nm, "
            f"G={float(row['G']):g}, "
            "physical_applicability="
            f"{'PASS' if bool(row['quantitative_physical_applicability']) else 'FAIL'}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search weak-field Fano-like suppression dips by varying physical "
            "QD--MNP separation."
        )
    )
    parser.add_argument("--target-ev", type=float, default=2.042)
    parser.add_argument("--window-ev", type=float, default=0.08)
    parser.add_argument("--grid-points", type=int, default=501)
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
    parser.add_argument("--gamma-population-mev", type=float, default=None)
    parser.add_argument(
        "--d-debye-values",
        nargs="+",
        type=float,
        default=[22.5],
        help="Bare/internal QD dipoles unless the effective convention is selected.",
    )
    parser.add_argument("--r-min-nm", type=float, default=18.0)
    parser.add_argument("--r-max-nm", type=float, default=40.0)
    parser.add_argument("--r-points", type=int, default=61)
    parser.add_argument("--r-spacing", choices=["linear", "log"], default="linear")
    parser.add_argument("--orientation", choices=["long", "trans"], default="long")
    parser.add_argument("--n-modes", type=int, default=9)
    parser.add_argument("--fit-min-ev", type=float, default=0.8)
    parser.add_argument("--fit-max-ev", type=float, default=3.0)
    parser.add_argument("--weight-center-ev", type=float, default=None)
    parser.add_argument("--weight-sigma-ev", type=float, default=None)
    parser.add_argument("--eps-m", type=float, default=None)
    parser.add_argument("--eps-qd", type=float, default=None)
    parser.add_argument(
        "--qd-dipole-convention",
        choices=["bare_internal", "effective_external"],
        default="effective_external",
    )
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
        r_min_nm=args.r_min_nm,
        r_max_nm=args.r_max_nm,
        r_points=args.r_points,
        r_spacing=args.r_spacing,
        fit_window_ev=(args.fit_min_ev, args.fit_max_ev),
        weight_center_ev=args.weight_center_ev,
        weight_sigma_ev=args.weight_sigma_ev,
        eps_m=args.eps_m,
        eps_qd=args.eps_qd,
        qd_dipole_convention=args.qd_dipole_convention,
        qd_radius_nm=args.qd_radius_nm,
        c_nm=args.c_nm,
        a_nm=args.a_nm,
        gamma_population_mev=args.gamma_population_mev,
        orientation=args.orientation,
        n_modes=args.n_modes,
    )
    write_csv(rows, args.csv)
    print(f"Wrote {len(rows)} rows to {args.csv}")
    modal_count = sum(
        bool(row.get("accepted_for_modal_numerical_ranking", False)) for row in rows
    )
    suppression_count = sum(
        bool(row.get("accepted_for_fano_like_suppression_ranking", False))
        for row in rows
    )
    quantitative_count = sum(
        bool(row.get("accepted_for_quantitative_ranking", False)) for row in rows
    )
    print(
        f"Modal/material-converged rows: {modal_count}; "
        f"numerically ranked suppression candidates: {suppression_count}; "
        f"physically applicable quantitative candidates: {quantitative_count}; "
        f"all saved rows: {len(rows)}"
    )
    print_top(rows, args.top)


if __name__ == "__main__":
    main()
