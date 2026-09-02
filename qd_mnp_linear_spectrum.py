"""Слабополевой линейный спектр QS work-loss отклика системы КТ-МНЧ.

Скрипт не интегрирует импульсную динамику. Он берет подогнанную поляризуемость
МНЧ из ``qd_mnp_rational_fit.py`` и линейный отклик двухуровневой КТ,
после чего вычисляет ведущую квазистатическую оценку работы
``k Im(alpha_QS)/eps0`` и отдельную рэлеевскую оценку рассеяния для связанной
системы, голой МНЧ и изолированной КТ. Их разность сохраняется только как
optical-theorem diagnostic: без radiation-reaction она не является строгим
материальным поглощением.

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
    au_to_eV,
    au_to_nm,
    quasistatic_dipole_cross_section_estimates_cm2,
    eV_to_au,
    params_to_physical_dict,
)
from qd_mnp_params import make_params_with_overrides


# The canonical N=9 realization reproduces the direct interpolated-material
# work-loss scale within the regression limits in the native 0.8--3 eV fit.
# Seven percent is
# therefore a regression/acceptance ceiling, not a claim of sub-percent
# publication accuracy; the actual error is always exported.
MODAL_OBSERVABLE_NORMALIZED_MAX_ERROR_LIMIT = 0.07
MODAL_OBSERVABLE_LOCAL_RELATIVE_ERROR_LIMIT = 0.10
MODAL_OBSERVABLE_LOCAL_RELATIVE_FLOOR_FRACTION = 1.0e-3
LINEAR_MAX_ENERGY_STEP_EV = 0.0005
LINEAR_MAX_STEP_FRACTION_OF_GAMMA2 = 0.25


def enforce_passive_work_loss_cm2(
    values: np.ndarray,
    *,
    label: str,
    relative_tolerance: float = 1.0e-10,
) -> np.ndarray:
    """Validate a passive work-loss spectrum and clip roundoff only.

    A reciprocal passive linear QD--MNP system cannot deliver net work to the
    external field.  Negative values larger than a scale-relative numerical
    tolerance therefore indicate an inconsistent response, not a Fano dip.
    """

    work = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(work)):
        raise RuntimeError(f"{label} contains non-finite QS work-loss values.")
    scale = max(float(np.max(np.abs(work))), np.finfo(float).tiny)
    tolerance = relative_tolerance * scale
    minimum = float(np.min(work))
    if minimum < -tolerance:
        raise RuntimeError(
            f"{label} violates passive non-negative QS work loss: "
            f"minimum={minimum:.6g} cm^2, tolerance={tolerance:.6g} cm^2."
        )
    if minimum < 0.0:
        warnings.warn(
            f"Clipped a sub-tolerance negative value in {label} caused by "
            "floating-point roundoff.",
            RuntimeWarning,
            stacklevel=2,
        )
        work = np.maximum(work, 0.0)
    return work


def qd_linear_polarizability_au(
    omega_au: np.ndarray,
    d_au: float,
    omega0_au: float,
    gamma2_au: float,
    local_field_factor: float = 1.0,
) -> np.ndarray:
    """Externally visible linear QD response to the macroscopic field.

    The expression follows from the weak-field Bloch equations with W=-1 and
    the exp(-i omega t) frequency convention.  ``d_au`` is the unscreened
    interband matrix element when ``local_field_factor=l_QD``.  One factor
    screens the microscopic drive and another converts the transition dipole
    to the external dipole, hence beta_eff=l_QD**2 beta_0.  If ``d_au`` is
    already an effective external dipole, callers must pass one.
    """
    if not np.isfinite(local_field_factor) or local_field_factor <= 0.0:
        raise ValueError("local_field_factor must be finite and positive.")
    beta_bare = 2.0 * d_au**2 * omega0_au / (
        omega0_au**2 + (gamma2_au - 1j * omega_au) ** 2
    )
    return local_field_factor**2 * beta_bare


def quasistatic_work_loss_cross_section_cm2(
    alpha_au: np.ndarray,
    omega_au: np.ndarray,
    eps_m: float,
) -> np.ndarray:
    """Leading strict-QS work-loss estimate k*Im(alpha_eff)/eps0 in cm^2."""

    return quasistatic_dipole_cross_section_estimates_cm2(
        alpha_au, omega_au, eps_m
    ).quasistatic_work_loss_cm2


def rayleigh_scattering_estimate_cross_section_cm2(
    alpha_au: np.ndarray,
    omega_au: np.ndarray,
    eps_m: float,
) -> np.ndarray:
    """Separate coherent Rayleigh-dipole radiation estimate in cm^2."""

    return quasistatic_dipole_cross_section_estimates_cm2(
        alpha_au, omega_au, eps_m
    ).rayleigh_scattering_estimate_cm2


def optical_theorem_residual_cross_section_cm2(
    alpha_au: np.ndarray,
    omega_au: np.ndarray,
    eps_m: float,
) -> np.ndarray:
    """Formal QS-work minus Rayleigh estimate; not material absorption."""

    return quasistatic_dipole_cross_section_estimates_cm2(
        alpha_au, omega_au, eps_m
    ).optical_theorem_residual_cm2


def extinction_cross_section_cm2(
    alpha_au: np.ndarray,
    omega_au: np.ndarray,
    eps_m: float,
) -> np.ndarray:
    """Legacy name for ``quasistatic_work_loss_cross_section_cm2``."""

    return quasistatic_work_loss_cross_section_cm2(alpha_au, omega_au, eps_m)


def scattering_cross_section_cm2(
    alpha_au: np.ndarray,
    omega_au: np.ndarray,
    eps_m: float,
) -> np.ndarray:
    """Legacy name for the separate Rayleigh scattering estimate."""

    return rayleigh_scattering_estimate_cross_section_cm2(
        alpha_au, omega_au, eps_m
    )


def absorption_cross_section_cm2(
    alpha_au: np.ndarray,
    omega_au: np.ndarray,
    eps_m: float,
) -> np.ndarray:
    """Legacy name for the optical-theorem residual, not absorption."""

    return optical_theorem_residual_cross_section_cm2(
        alpha_au, omega_au, eps_m
    )


def cross_section_cm2(alpha_au: np.ndarray, omega_au: np.ndarray, eps_m: float) -> np.ndarray:
    """Schema-1 compatibility alias; the old function always returned extinction."""
    warnings.warn(
        "cross_section_cm2() was ambiguous; use "
        "quasistatic_work_loss_cross_section_cm2().",
        DeprecationWarning,
        stacklevel=2,
    )
    return quasistatic_work_loss_cross_section_cm2(alpha_au, omega_au, eps_m)


def linear_coupled_alpha_au(
    model: HybridQDPlasmonModel,
    energies_ev: np.ndarray,
    *,
    mnp_response: str = "fit",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return effective coupled, bare-MNP, isolated-QD, and coupled-minus-bare alphas.

    All returned quantities are in the same effective convention used for
    spectral cross sections: alpha_eff = mu_total / (eps_m E_inc).
    """
    p = model.params
    omega = eV_to_au(energies_ev)

    if mnp_response == "fit":
        alpha_mnp_dimless = model.alpha_from_fit(energies_ev)
    elif mnp_response == "material":
        alpha_mnp_dimless = model.alpha_from_material(energies_ev)
    else:
        raise ValueError("mnp_response must be 'fit' or 'material'.")
    alpha_p = model.C * alpha_mnp_dimless
    beta_qd = qd_linear_polarizability_au(
        omega,
        p.d_au,
        p.omega0_au,
        p.Gamma_au,
        p.qd_local_field_factor,
    )

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
    eps_qd: float | None = None,
    orientation: str = "long",
    qd_dipole_convention: str = "effective_external",
) -> list[dict[str, float | str | bool]]:
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
        eps_qd=eps_qd,
        qd_dipole_convention=qd_dipole_convention,
        orientation=orientation,
    )
    energy_step_ev = float((energy_max_ev - energy_min_ev) / (points - 1))
    gamma2_energy_ev = float(au_to_eV(params.Gamma_au))
    gamma2_relative_max_energy_step_ev = float(
        LINEAR_MAX_STEP_FRACTION_OF_GAMMA2 * gamma2_energy_ev
    )
    effective_max_energy_step_ev = min(
        LINEAR_MAX_ENERGY_STEP_EV,
        gamma2_relative_max_energy_step_ev,
    )
    spectral_grid_resolved = bool(
        energy_step_ev
        <= effective_max_energy_step_ev * (1.0 + 1.0e-12)
    )
    if not spectral_grid_resolved:
        required_points = int(
            np.ceil(
                (energy_max_ev - energy_min_ev)
                / effective_max_energy_step_ev
            )
            + 1
        )
        raise ValueError(
            "The requested linear-spectrum grid cannot resolve the narrow "
            "QD response: energy step="
            f"{1.0e3 * energy_step_ev:.6g} meV, but it must be <= "
            f"min(0.5 meV, Gamma2/4)="
            f"{1.0e3 * effective_max_energy_step_ev:.6g} meV. "
            f"Use at least {required_points} points over this interval."
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

    energies = np.linspace(energy_min_ev, energy_max_ev, points)
    omega = eV_to_au(energies)
    alpha_coupled, alpha_bare, alpha_qd, _ = linear_coupled_alpha_au(
        model, energies
    )
    alpha_coupled_material, alpha_bare_material, _, _ = linear_coupled_alpha_au(
        model,
        energies,
        mnp_response="material",
    )

    coupled = quasistatic_dipole_cross_section_estimates_cm2(
        alpha_coupled, omega, params.eps_m
    )
    bare = quasistatic_dipole_cross_section_estimates_cm2(
        alpha_bare, omega, params.eps_m
    )
    isolated_qd = quasistatic_dipole_cross_section_estimates_cm2(
        alpha_qd, omega, params.eps_m
    )
    coupled_material = quasistatic_dipole_cross_section_estimates_cm2(
        alpha_coupled_material, omega, params.eps_m
    )
    bare_material = quasistatic_dipole_cross_section_estimates_cm2(
        alpha_bare_material, omega, params.eps_m
    )
    modal_work = enforce_passive_work_loss_cm2(
        coupled.quasistatic_work_loss_cm2,
        label="modal coupled QD--MNP response",
    )
    material_work = enforce_passive_work_loss_cm2(
        coupled_material.quasistatic_work_loss_cm2,
        label="direct-material coupled QD--MNP response",
    )
    modal_bare_work = enforce_passive_work_loss_cm2(
        bare.quasistatic_work_loss_cm2,
        label="modal bare-MNP response",
    )
    material_bare_work = enforce_passive_work_loss_cm2(
        bare_material.quasistatic_work_loss_cm2,
        label="direct-material bare-MNP response",
    )
    isolated_qd_work = enforce_passive_work_loss_cm2(
        isolated_qd.quasistatic_work_loss_cm2,
        label="isolated-QD response",
    )
    observable_scale = max(
        float(np.max(np.abs(material_work))),
        float(
            np.max(
                np.abs(bare_material.quasistatic_work_loss_cm2)
            )
        ),
        np.finfo(float).tiny,
    )
    coupled_modal_normalized_max_error = float(
        np.max(np.abs(modal_work - material_work)) / observable_scale
    )
    bare_modal_normalized_max_error = float(
        np.max(np.abs(modal_bare_work - material_bare_work))
        / observable_scale
    )
    modal_observable_normalized_max_error = max(
        coupled_modal_normalized_max_error,
        bare_modal_normalized_max_error,
    )
    combined_modal_error = np.concatenate(
        [modal_work - material_work, modal_bare_work - material_bare_work]
    )
    combined_material_work = np.concatenate(
        [material_work, material_bare_work]
    )
    modal_observable_normalized_rms_error = float(
        np.sqrt(np.mean(combined_modal_error**2))
        / max(float(np.sqrt(np.mean(combined_material_work**2))), np.finfo(float).tiny)
    )
    local_relative_floor = float(
        MODAL_OBSERVABLE_LOCAL_RELATIVE_FLOOR_FRACTION * observable_scale
    )
    coupled_local_relative_errors = np.abs(modal_work - material_work) / np.maximum(
        np.abs(material_work),
        local_relative_floor,
    )
    bare_local_relative_errors = np.abs(
        modal_bare_work - material_bare_work
    ) / np.maximum(
        np.abs(material_bare_work),
        local_relative_floor,
    )
    modal_observable_local_relative_max_error = float(
        max(
            np.max(coupled_local_relative_errors),
            np.max(bare_local_relative_errors),
        )
    )
    modal_observable_pointwise_converged = bool(
        modal_observable_local_relative_max_error
        <= MODAL_OBSERVABLE_LOCAL_RELATIVE_ERROR_LIMIT
    )
    modal_observable_converged = bool(
        modal_observable_normalized_max_error
        <= MODAL_OBSERVABLE_NORMALIZED_MAX_ERROR_LIMIT
        and modal_observable_pointwise_converged
    )

    rows: list[dict[str, float | str | bool]] = []
    physical_provenance = params_to_physical_dict(
        params,
        orientation=orientation,
    )
    for idx, e in enumerate(energies):
        applicability = model.dipole_applicability_diagnostics(energy_eV=float(e))
        work_c = float(modal_work[idx])
        rayleigh_c = float(coupled.rayleigh_scattering_estimate_cm2[idx])
        residual_c = work_c - rayleigh_c
        work_b = float(modal_bare_work[idx])
        rayleigh_b = float(bare.rayleigh_scattering_estimate_cm2[idx])
        residual_b = work_b - rayleigh_b
        work_q = float(isolated_qd_work[idx])
        rayleigh_q = float(isolated_qd.rayleigh_scattering_estimate_cm2[idx])
        residual_q = work_q - rayleigh_q
        work_c_material = float(material_work[idx])
        work_b_material = float(material_bare_work[idx])
        ratio_work = float(work_c / work_b) if work_b != 0.0 else np.nan
        rows.append(
            {
                **physical_provenance,
                "schema_version": SCHEMA_VERSION,
                "model_profile": "quasistatic_ellipsoid_tls",
                "orientation": orientation,
                "G": float(params.G),
                "R_nm": float(au_to_nm(params.R_au)),
                "surface_gap_nm": float(au_to_nm(params.surface_gap_au)),
                "eps_qd": float(params.eps_qd),
                "qd_local_field_factor": float(params.qd_local_field_factor),
                "qd_dipole_convention": params.qd_dipole_convention,
                "medium_size_parameter_kc": float(
                    applicability.medium_size_parameter_kc
                ),
                "medium_separation_parameter_kR": float(
                    applicability.medium_separation_parameter_kR
                ),
                "mnp_size_to_separation_ratio_c_over_R": float(
                    applicability.mnp_size_to_separation_ratio
                ),
                "qd_size_to_separation_ratio_rqd_over_R": float(
                    applicability.qd_size_to_separation_ratio
                ),
                "particle_quasistatic_guide_satisfied": bool(
                    applicability.particle_quasistatic_guide_satisfied
                ),
                "near_field_coupling_guide_satisfied": bool(
                    applicability.near_field_coupling_guide_satisfied
                ),
                "quasistatic_guide_satisfied": bool(
                    applicability.quasistatic_guide_satisfied
                ),
                "mnp_point_dipole_guide_satisfied": bool(
                    applicability.mnp_point_dipole_guide_satisfied
                ),
                "qd_point_dipole_guide_satisfied": bool(
                    applicability.qd_point_dipole_guide_satisfied
                ),
                "point_dipole_guide_satisfied": bool(
                    applicability.point_dipole_guide_satisfied
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
                # Schema-3 compatibility alias; the explicit key above is the
                # physically precise statement proved by f_k,gamma_k>=0.
                "mnp_fit_globally_passive": bool(
                    model.fit.nonnegative_imaginary_part_all_positive_frequencies
                ),
                "modal_observable_reference": "direct_interpolated_material_alpha",
                "modal_observable_normalized_max_error": (
                    modal_observable_normalized_max_error
                ),
                "modal_observable_global_scale_normalized_max_error": (
                    modal_observable_normalized_max_error
                ),
                "modal_coupled_work_normalized_max_error": (
                    coupled_modal_normalized_max_error
                ),
                "modal_bare_work_normalized_max_error": (
                    bare_modal_normalized_max_error
                ),
                "modal_observable_normalized_rms_error": (
                    modal_observable_normalized_rms_error
                ),
                "modal_observable_local_relative_max_error": (
                    modal_observable_local_relative_max_error
                ),
                "modal_observable_local_relative_error_limit": (
                    MODAL_OBSERVABLE_LOCAL_RELATIVE_ERROR_LIMIT
                ),
                "modal_observable_local_relative_floor_cm2": (
                    local_relative_floor
                ),
                "modal_observable_pointwise_converged": (
                    modal_observable_pointwise_converged
                ),
                "modal_observable_error_limit": (
                    MODAL_OBSERVABLE_NORMALIZED_MAX_ERROR_LIMIT
                ),
                "modal_observable_converged": modal_observable_converged,
                "spectral_grid_energy_step_ev": energy_step_ev,
                "spectral_grid_absolute_max_energy_step_ev": (
                    LINEAR_MAX_ENERGY_STEP_EV
                ),
                "spectral_grid_gamma2_relative_max_energy_step_ev": (
                    gamma2_relative_max_energy_step_ev
                ),
                "spectral_grid_max_energy_step_ev": effective_max_energy_step_ev,
                "spectral_grid_max_step_fraction_of_gamma2": (
                    LINEAR_MAX_STEP_FRACTION_OF_GAMMA2
                ),
                "spectral_grid_resolved": spectral_grid_resolved,
                "energy_ev": float(e),
                "linearized_ground_state_stable": bool(model.linear_stability.stable),
                "linearized_ground_state_spectral_abscissa_au": float(
                    model.linear_stability.spectral_abscissa_au
                ),
                "sigma_qs_work_loss_coupled_cm2": work_c,
                "sigma_qs_work_loss_coupled_material_reference_cm2": (
                    work_c_material
                ),
                "modal_vs_material_coupled_local_relative_error": float(
                    coupled_local_relative_errors[idx]
                ),
                "sigma_rayleigh_sca_estimate_coupled_cm2": rayleigh_c,
                "sigma_optical_theorem_residual_coupled_cm2": residual_c,
                "sigma_qs_work_loss_bare_mnp_cm2": work_b,
                "sigma_qs_work_loss_bare_mnp_material_reference_cm2": (
                    work_b_material
                ),
                "modal_vs_material_bare_local_relative_error": float(
                    bare_local_relative_errors[idx]
                ),
                "sigma_rayleigh_sca_estimate_bare_mnp_cm2": rayleigh_b,
                "sigma_optical_theorem_residual_bare_mnp_cm2": residual_b,
                "sigma_qs_work_loss_isolated_qd_cm2": work_q,
                "sigma_rayleigh_sca_estimate_isolated_qd_cm2": rayleigh_q,
                "sigma_optical_theorem_residual_isolated_qd_cm2": residual_q,
                "delta_sigma_qs_work_loss_cm2": work_c - work_b,
                "delta_sigma_rayleigh_sca_estimate_cm2": rayleigh_c - rayleigh_b,
                "delta_sigma_optical_theorem_residual_cm2": residual_c - residual_b,
                "ratio_qs_work_loss_coupled_to_bare": ratio_work,
                # Historical aliases. ``ext`` meant the QS work-loss proxy and
                # ``abs`` meant only the optical-theorem residual.
                "sigma_ext_coupled_cm2": work_c,
                "sigma_sca_coupled_cm2": rayleigh_c,
                "sigma_abs_coupled_cm2": residual_c,
                "sigma_ext_bare_mnp_cm2": work_b,
                "sigma_sca_bare_mnp_cm2": rayleigh_b,
                "sigma_abs_bare_mnp_cm2": residual_b,
                "sigma_ext_isolated_qd_cm2": work_q,
                "sigma_sca_isolated_qd_cm2": rayleigh_q,
                "sigma_abs_isolated_qd_cm2": residual_q,
                "delta_sigma_ext_cm2": work_c - work_b,
                "delta_sigma_sca_cm2": rayleigh_c - rayleigh_b,
                "delta_sigma_abs_cm2": residual_c - residual_b,
                "ratio_ext_coupled_to_bare": ratio_work,
                "sigma_coupled_cm2": work_c,
                "sigma_bare_mnp_cm2": work_b,
                "sigma_isolated_qd_cm2": work_q,
                "sigma_coupled_minus_bare_cm2": work_c - work_b,
                "ratio_coupled_to_bare": ratio_work,
            }
        )
    if not modal_observable_converged:
        warnings.warn(
            "The modal realization does not reproduce the same linear "
            "work-loss spectrum evaluated with the direct interpolated "
            "material polarizability within both acceptance limits: global-scale "
            f"maximum error={modal_observable_normalized_max_error:.6g} "
            f"(limit {MODAL_OBSERVABLE_NORMALIZED_MAX_ERROR_LIMIT:.6g}), "
            "local relative maximum error="
            f"{modal_observable_local_relative_max_error:.6g} "
            f"(limit {MODAL_OBSERVABLE_LOCAL_RELATIVE_ERROR_LIMIT:.6g}, with "
            f"floor {local_relative_floor:.3g} cm^2). Increase n_modes or improve "
            "the fit before quantitative pointwise comparison.",
            RuntimeWarning,
            stacklevel=2,
        )
    return rows


def write_csv(rows: list[dict[str, float | str | bool]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_spectrum(rows: list[dict[str, float]], output_path: Path | None, show: bool) -> None:
    energy = np.array([row["energy_ev"] for row in rows])
    coupled = np.array([row["sigma_qs_work_loss_coupled_cm2"] for row in rows])
    bare = np.array([row["sigma_qs_work_loss_bare_mnp_cm2"] for row in rows])
    delta = np.array([row["delta_sigma_qs_work_loss_cm2"] for row in rows])
    ratio = np.array(
        [row["ratio_qs_work_loss_coupled_to_bare"] for row in rows]
    )

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(energy, bare, lw=2.0, label="bare MNP")
    axes[0].plot(energy, coupled, lw=2.0, label="coupled QD+MNP")
    axes[0].set_ylabel(r"$k\,\mathrm{Im}\,\alpha_{QS}/\epsilon_0$, cm$^2$")
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
    minimum = min(rows, key=lambda row: row["ratio_qs_work_loss_coupled_to_bare"])
    maximum = max(rows, key=lambda row: row["ratio_qs_work_loss_coupled_to_bare"])

    print("\n=== Linear-spectrum diagnostics ===")
    print(
        f"At {closest['energy_ev']:.6f} eV: "
        f"qs_work_coupled={closest['sigma_qs_work_loss_coupled_cm2']:.6e} cm^2, "
        f"qs_work_bare={closest['sigma_qs_work_loss_bare_mnp_cm2']:.6e} cm^2, "
        f"ratio={closest['ratio_qs_work_loss_coupled_to_bare']:.6g}, "
        f"delta_qs_work={closest['delta_sigma_qs_work_loss_cm2']:.6e} cm^2"
    )
    print(
        f"Deepest relative dip: {minimum['energy_ev']:.6f} eV, "
        f"ratio={minimum['ratio_qs_work_loss_coupled_to_bare']:.6g}, "
        f"delta_qs_work={minimum['delta_sigma_qs_work_loss_cm2']:.6e} cm^2"
    )
    print(
        f"Largest relative enhancement: {maximum['energy_ev']:.6f} eV, "
        f"ratio={maximum['ratio_qs_work_loss_coupled_to_bare']:.6g}, "
        f"delta_qs_work={maximum['delta_sigma_qs_work_loss_cm2']:.6e} cm^2"
    )

    min_residual = min(
        row["sigma_optical_theorem_residual_coupled_cm2"] for row in rows
    )
    if min_residual < 0.0:
        print(
            "NOTE: the formal optical-theorem residual "
            "sigma_qs_work-sigma_Rayleigh is negative "
            f"(minimum {min_residual:.6e} cm^2). The native alpha_QS is not "
            "radiatively dressed, so this residual is diagnostic only and is not "
            "material absorption."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Weak-field linear QD-MNP quasistatic work-loss spectrum."
    )
    parser.add_argument("--energy-min-ev", type=float, default=2.0)
    parser.add_argument("--energy-max-ev", type=float, default=2.08)
    parser.add_argument("--points", type=int, default=201)
    parser.add_argument("--target-ev", type=float, default=2.042)
    parser.add_argument("--n-modes", type=int, default=9)
    parser.add_argument("--fit-min-ev", type=float, default=0.8)
    parser.add_argument("--fit-max-ev", type=float, default=3.0)
    parser.add_argument("--weight-center-ev", type=float, default=None)
    parser.add_argument("--weight-sigma-ev", type=float, default=None)
    parser.add_argument("--c-nm", type=float, default=None)
    parser.add_argument("--a-nm", type=float, default=None)
    parser.add_argument("--r-nm", type=float, default=None)
    parser.add_argument("--qd-radius-nm", type=float, default=None)
    parser.add_argument("--G", dest="g_factor", type=float, default=None)
    parser.add_argument("--eps-m", type=float, default=None)
    parser.add_argument("--eps-qd", type=float, default=None)
    parser.add_argument("--orientation", choices=["long", "trans"], default="long")
    parser.add_argument(
        "--qd-dipole-convention",
        choices=["bare_internal", "effective_external"],
        default="effective_external",
    )
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
        eps_qd=args.eps_qd,
        orientation=args.orientation,
        qd_dipole_convention=args.qd_dipole_convention,
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
