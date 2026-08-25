"""Frequency-domain Cheng reference around the project's native Au response.

The paper studies one prolate Au nanoparticle and one CdTe quantum dot.  The
incident field is *always* parallel to the long MNP axis, so the MNP response
is always the longitudinal polarizability.  The QD is placed either

* at the end of that axis, with dipole tensor factor ``g=2``; or
* beside the particle, with ``g=-1``.

``HybridQDPlasmonModel`` deliberately couples ``orientation='trans'`` to both
``g=-1`` and the transverse MNP polarizability.  That is correct for its own
geometry but is not Cheng's side geometry.  This module is therefore a small
geometry adapter: it evaluates the unchanged core's Johnson--Christy
longitudinal quasistatic polarizability and applies ``g`` and the directional
surface gap independently.  No core file is changed.

Figures 2 and 3 of Cheng are frequency-domain rate formulas and Figure 4 is
given only as a function of pulse area.  Only alpha(omega_L) is needed; no MNP
time state appears.  The rate sweeps consequently use the direct material
response that the core's passive Lorentz fit targets.  Cached fits generated
by the real ``HybridQDPlasmonModel`` for q=1,3,4 provide a separate modal audit
at all three article wavelengths.  Re-fitting hundreds of aspect ratios would
merely approximate the same carrier-frequency samples.

Two reproducibility gaps are kept visible rather than silently guessed:

* Cheng uses Palik Au data; this project bundles Johnson--Christy data.
* Cheng does not give the QD transition dipole.  Nonradiative rates are saved
  both for a radiatively consistent project profile and for a clearly labelled
  1.82 D value inferred from the scale of Figure 3.

The pulse shape, duration and drive amplitude for dissipative Figure 4 are
also absent.  The script therefore exports only the strong-drive,
zero-feedback area-theorem proxy ``rho22=sin^2(|f| theta/2)``.  This additionally
assumes ``gamma10/Omega0 -> 0`` and ``|G_feedback|/Omega0 -> 0`` and is never
labelled an exact reproduction of the dissipative curve.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Literal

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c as C_SI, e as E_CHARGE, epsilon_0, h, hbar

from qd_mnp_params import make_params_with_overrides
from qd_mnp_rational_fit import (
    AU_DIPOLE_C_M,
    AU_TIME_S,
    DEBYE_C_M,
    DEFAULT_AU_MATERIAL,
    HybridQDPlasmonModel,
    MaterialDispersion,
    RationalLorentzFit,
    au_to_eV,
    eV_to_au,
    homogeneous_radiative_decay_rate_au,
    nm_to_au,
)


ChengGeometry = Literal["end", "side"]
DipoleProfile = Literal["radiative_consistent", "figure_calibrated"]
FieldConvention = Literal["cheng", "native"]
MAX_MODAL_AUDIT_RELATIVE_ERROR = 0.06
MAX_MODAL_AUDIT_LOSS_CHANNEL_ERROR = 0.10
_CHENG_PLOT_ARTIFACTS = (
    "figure_2_radiative_rates.png",
    "figure_3_nonradiative_rates_radiative_consistent.png",
    "figure_3_nonradiative_rates_figure_calibrated.png",
    "figure_4_zero_feedback_area_proxy.png",
)


# Deterministic N=9 caches produced by HybridQDPlasmonModel on 0.8--3.0 eV
# for Cheng's b=6 nm, eps_h=2.25 longitudinal spheroids.  These are modal
# audits for the q values explicitly used in Figure 4, not new material data.
_CHENG_MODAL_CACHE: dict[float, dict[str, object]] = {
    1.0: {
        "alpha_inf": -0.6818181818181818,
        "strengths": np.asarray([
            0.0017231482277107143, 0.00242096687291871,
            0.0012473179182225047, 0.0009665205881885139,
            0.0009630612225068187, 0.0011338111334509582,
            0.0014210258470453764, 0.0017205927117578403,
            0.1035412680305021,
        ]),
        "omega_ev": np.asarray([
            2.252180811785979, 2.3345049655458516, 2.425717089571707,
            2.5555848111259967, 2.6886525029678565, 2.819932882728319,
            2.9606194309688583, 3.0973126984006987, 5.680651877435022,
        ]),
        "gamma_ev": np.asarray([
            0.16077171039812888, 0.1460343015675781,
            0.16609695370853375, 0.2004106296558959,
            0.2207736184713587, 0.23494741641345687,
            0.22040216109376884, 0.07163793225804883,
            0.008000000000000021,
        ]),
    },
    3.0: {
        "alpha_inf": -0.5912644689583699,
        "strengths": np.asarray([
            0.0014249396024107624, 7.9544423972571e-05,
            0.027574487331877867, 0.0007026829670539394,
            0.0003818080129828513, 0.006730115116139726,
            0.002580694065213637, 0.000383552388846399,
            0.13927968621816714,
        ]),
        "omega_ev": np.asarray([
            0.28000000000000036, 0.9678440208491241, 1.7082278147906487,
            1.9833840455203917, 2.209876080881402, 2.7165142902489054,
            2.998572602581941, 3.054573015686625, 6.135648132932103,
        ]),
        "gamma_ev": np.asarray([
            0.008000000000000021, 0.18594566608478108,
            0.07255820830702324, 0.25996019252663916,
            0.26620003166891065, 0.7543062258404197,
            0.39186944291439435, 0.04237566280354355,
            0.008000000000003674,
        ]),
    },
    4.0: {
        "alpha_inf": -0.5798470368072401,
        "strengths": np.asarray([
            0.001470729655853052, 0.031137727457076044,
            0.0025988757967028765, 0.0006818979425785612,
            2.1849962892883215e-05, 0.004316214683273804,
            0.0032262569022904427, 0.00046876160706813973,
            0.10960529651044007,
        ]),
        "omega_ev": np.asarray([
            0.28000000000000036, 1.475665723303964, 1.5372114686827576,
            2.1818498227998235, 2.526194874827084, 2.6997166709835727,
            2.9865965760892847, 3.063227153321997, 5.712191269666465,
        ]),
        "gamma_ev": np.asarray([
            0.008000000000000021, 0.06782120914185019,
            0.27673293427835516, 0.5651726244064005,
            0.09757151490601318, 0.6142804575566024,
            0.4434198783003142, 0.0629671769531926,
            0.008000000000000021,
        ]),
    },
}


def photon_energy_ev(wavelength_nm: float | np.ndarray) -> np.ndarray:
    wavelength = np.asarray(wavelength_nm, dtype=float)
    if np.any(~np.isfinite(wavelength)) or np.any(wavelength <= 0.0):
        raise ValueError("wavelength_nm must be finite and positive.")
    return h * C_SI / (wavelength * 1.0e-9 * E_CHARGE)


def longitudinal_depolarization_factor(aspect_ratio: float | np.ndarray) -> np.ndarray:
    """Long-axis depolarization factor of a prolate spheroid, q=c/a >= 1."""

    q = np.asarray(aspect_ratio, dtype=float)
    if np.any(~np.isfinite(q)) or np.any(q < 1.0):
        raise ValueError("The Cheng prolate aspect ratio q must satisfy q >= 1.")
    result = np.empty_like(q)
    eccentricity_squared = 1.0 - 1.0 / q**2
    near_sphere = eccentricity_squared < 1.0e-3
    if np.any(near_sphere):
        e2 = eccentricity_squared[near_sphere]
        result[near_sphere] = (
            1.0 / 3.0
            - 2.0 * e2 / 15.0
            - 2.0 * e2**2 / 35.0
            - 2.0 * e2**3 / 63.0
            - 2.0 * e2**4 / 99.0
        )
    nonsphere = ~near_sphere
    if np.any(nonsphere):
        eccentricity = np.sqrt(eccentricity_squared[nonsphere])
        result[nonsphere] = (
            (1.0 - eccentricity**2)
            * (np.arctanh(eccentricity) - eccentricity)
            / eccentricity**3
        )
    return result


@dataclass(frozen=True)
class Cheng2007Profile:
    """Constants printed by Cheng plus explicit comparison assumptions."""

    mnp_semiminor_nm: float = 6.0
    eps_environment: float = 2.25
    eps_qd: float = 10.0
    isolated_radiative_rate_ns_inv: float = 0.08
    extra_population_decay_ns_inv: float = 10.0
    wavelengths_nm: tuple[float, ...] = (600.0, 750.0, 900.0)
    qd_radii_nm: tuple[float, ...] = (3.2, 6.6, 9.5)
    gaps_nm: tuple[float, ...] = (2.0, 4.0, 6.0)
    calibrated_internal_dipole_debye: float = 1.82

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.mnp_semiminor_nm,
                self.eps_environment,
                self.eps_qd,
                self.isolated_radiative_rate_ns_inv,
                self.extra_population_decay_ns_inv,
                self.calibrated_internal_dipole_debye,
            ],
            dtype=float,
        )
        if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("All Cheng profile constants must be finite and positive.")
        if len(self.wavelengths_nm) != len(self.qd_radii_nm):
            raise ValueError("Each Cheng wavelength needs one stated QD radius.")
        if any(
            not np.isfinite(value) or value <= 0.0
            for value in self.wavelengths_nm + self.qd_radii_nm
        ):
            raise ValueError("Wavelengths and QD radii must be positive.")
        if len(self.gaps_nm) == 0 or any(
            not np.isfinite(value) or value <= 0.0 for value in self.gaps_nm
        ):
            raise ValueError("The Cheng surface gaps must be positive.")

    @property
    def epsilon_eff(self) -> float:
        return (2.0 * self.eps_environment + self.eps_qd) / 3.0

    @property
    def qd_local_field_factor(self) -> float:
        return self.eps_environment / self.epsilon_eff

    def qd_radius_nm(self, wavelength_nm: float) -> float:
        for wavelength, radius in zip(self.wavelengths_nm, self.qd_radii_nm):
            if np.isclose(wavelength_nm, wavelength, rtol=0.0, atol=1.0e-10):
                return float(radius)
        raise ValueError(
            "Cheng gives QD radii only for wavelengths 600, 750 and 900 nm."
        )

    @staticmethod
    def orientation_factor(geometry: ChengGeometry) -> float:
        if geometry == "end":
            return 2.0
        if geometry == "side":
            return -1.0
        raise ValueError("geometry must be 'end' or 'side'.")

    def semimajor_nm(self, aspect_ratio: float) -> float:
        if not np.isfinite(aspect_ratio) or aspect_ratio < 1.0:
            raise ValueError("aspect_ratio must satisfy q >= 1.")
        return float(aspect_ratio * self.mnp_semiminor_nm)

    def center_distance_nm(
        self,
        *,
        aspect_ratio: float,
        wavelength_nm: float,
        gap_nm: float,
        geometry: ChengGeometry,
    ) -> float:
        if not np.isfinite(gap_nm) or gap_nm <= 0.0:
            raise ValueError("gap_nm must be finite and positive.")
        qd_radius = self.qd_radius_nm(wavelength_nm)
        if geometry == "end":
            mnp_directional_radius = self.semimajor_nm(aspect_ratio)
        elif geometry == "side":
            mnp_directional_radius = self.mnp_semiminor_nm
        else:
            raise ValueError("geometry must be 'end' or 'side'.")
        return float(mnp_directional_radius + qd_radius + gap_nm)

    def reconstructed_gap_nm(
        self,
        *,
        aspect_ratio: float,
        wavelength_nm: float,
        center_distance_nm: float,
        geometry: ChengGeometry,
    ) -> float:
        qd_radius = self.qd_radius_nm(wavelength_nm)
        directional_radius = (
            self.semimajor_nm(aspect_ratio)
            if geometry == "end"
            else self.mnp_semiminor_nm
        )
        return float(center_distance_nm - directional_radius - qd_radius)

    def provenance(self) -> dict[str, object]:
        return {
            "direct_from_cheng2007": {
                "mnp_semiminor_nm": self.mnp_semiminor_nm,
                "eps_environment": self.eps_environment,
                "eps_qd": self.eps_qd,
                "isolated_radiative_rate_ns_inv": self.isolated_radiative_rate_ns_inv,
                "extra_population_decay_ns_inv_figure_4": self.extra_population_decay_ns_inv,
                "wavelengths_nm": list(self.wavelengths_nm),
                "qd_radii_nm": list(self.qd_radii_nm),
                "gaps_nm": list(self.gaps_nm),
            },
            "missing_from_article": [
                "numerical QD transition dipole",
                "full Palik table and interpolation convention",
                "Figure 4 pulse amplitude, duration and temporal shape",
            ],
            "project_model_choices": {
                "gold_data": "Johnson-Christy table bundled with the project",
                "article_gold_data": "Palik",
                "calibrated_internal_dipole_debye": self.calibrated_internal_dipole_debye,
                "figure_4": (
                    "strong-drive zero-feedback area-theorem proxy only"
                ),
            },
        }


def longitudinal_alpha_au(
    profile: Cheng2007Profile,
    *,
    aspect_ratio: float | np.ndarray,
    wavelength_nm: float,
    material: MaterialDispersion = DEFAULT_AU_MATERIAL,
) -> np.ndarray:
    """Project's direct-material longitudinal MNP polarizability in a.u."""

    q = np.asarray(aspect_ratio, dtype=float)
    energy_ev = float(photon_energy_ev(wavelength_nm))
    alpha_dimless = _longitudinal_alpha_dimless_at_energy(
        profile,
        aspect_ratio=q,
        energy_ev=energy_ev,
        material=material,
    )
    semiminor_au = float(nm_to_au(profile.mnp_semiminor_nm))
    semimajor_au = q * semiminor_au
    core_C = profile.eps_environment * semiminor_au**2 * semimajor_au / 3.0
    return np.asarray(core_C * alpha_dimless, dtype=complex)


def _longitudinal_alpha_dimless_at_energy(
    profile: Cheng2007Profile,
    *,
    aspect_ratio: float | np.ndarray,
    energy_ev: float | np.ndarray,
    material: MaterialDispersion = DEFAULT_AU_MATERIAL,
) -> np.ndarray:
    q = np.asarray(aspect_ratio, dtype=float)
    depolarization = longitudinal_depolarization_factor(q)
    epsilon_gold = material.epsilon_at(energy_ev)
    contrast = epsilon_gold - profile.eps_environment
    return np.asarray(contrast / (
        profile.eps_environment + depolarization * contrast
    ), dtype=complex)


def _evaluate_modal_fit_dimless(
    fit: RationalLorentzFit,
    energies_ev: float | np.ndarray,
) -> np.ndarray:
    omega = np.asarray(eV_to_au(energies_ev), dtype=float)
    response = np.full_like(omega, fit.alpha_inf, dtype=complex)
    for strength, mode_omega, mode_gamma in zip(
        fit.strengths_au2,
        fit.omega_modes_au,
        fit.gamma_modes_au,
    ):
        response += strength / (
            mode_omega**2 - omega**2 - 1j * mode_gamma * omega
        )
    return response


def _cached_cheng_modal_fit(
    profile: Cheng2007Profile,
    aspect_ratio: float,
) -> RationalLorentzFit:
    try:
        cached = _CHENG_MODAL_CACHE[float(aspect_ratio)]
    except KeyError as exc:
        raise ValueError("Cached Cheng modal audits exist only for q=1,3,4.") from exc
    mask = (
        (DEFAULT_AU_MATERIAL.energy_eV >= 0.8)
        & (DEFAULT_AU_MATERIAL.energy_eV <= 3.0)
    )
    energies = np.unique(
        np.concatenate(
            (
                [0.8, 3.0],
                np.linspace(0.8, 3.0, max(1025, 256 * 9)),
                DEFAULT_AU_MATERIAL.energy_eV[mask],
            )
        )
    )
    target = _longitudinal_alpha_dimless_at_energy(
        profile,
        aspect_ratio=aspect_ratio,
        energy_ev=energies,
    )
    temporary_fit = RationalLorentzFit(
        alpha_inf=float(cached["alpha_inf"]),
        strengths_au2=np.asarray(cached["strengths"], dtype=float),
        omega_modes_au=np.asarray(eV_to_au(cached["omega_ev"]), dtype=float),
        gamma_modes_au=np.asarray(eV_to_au(cached["gamma_ev"]), dtype=float),
        energies_used_eV=energies,
        alpha_used=target,
        rms_alpha=1.0,
        rms_inv_alpha=1.0,
        cost=1.0,
    )
    fitted = _evaluate_modal_fit_dimless(temporary_fit, energies)
    inverse_target = 1.0 / target
    inverse_fitted = 1.0 / fitted
    alpha_error = fitted - target
    inverse_error = inverse_fitted - inverse_target
    rms_alpha = float(np.sqrt(np.mean(np.abs(alpha_error) ** 2)))
    rms_inverse = float(np.sqrt(np.mean(np.abs(inverse_error) ** 2)))
    normalized_rms_alpha = rms_alpha / float(
        np.sqrt(np.mean(np.abs(target) ** 2))
    )
    normalized_rms_inverse = rms_inverse / float(
        np.sqrt(np.mean(np.abs(inverse_target) ** 2))
    )
    maximum_relative_error = float(
        np.max(np.abs(alpha_error) / np.abs(target))
    )
    residual_parts = [
        alpha_error.real / max(float(np.max(np.abs(target.real))), 1.0e-12),
        alpha_error.imag / max(float(np.max(np.abs(target.imag))), 1.0e-12),
        np.sqrt(1.2)
        * inverse_error.real
        / max(float(np.max(np.abs(inverse_target.real))), 1.0e-12),
        np.sqrt(1.2)
        * inverse_error.imag
        / max(float(np.max(np.abs(inverse_target.imag))), 1.0e-12),
    ]
    score = float(np.sqrt(np.mean(np.concatenate(residual_parts) ** 2)))
    passivity_energies = np.linspace(0.8, 3.0, max(1024, 256 * 9))
    passivity_response = _evaluate_modal_fit_dimless(
        temporary_fit,
        passivity_energies,
    )
    return RationalLorentzFit(
        alpha_inf=float(cached["alpha_inf"]),
        strengths_au2=np.asarray(cached["strengths"], dtype=float),
        omega_modes_au=np.asarray(eV_to_au(cached["omega_ev"]), dtype=float),
        gamma_modes_au=np.asarray(eV_to_au(cached["gamma_ev"]), dtype=float),
        energies_used_eV=energies,
        alpha_used=target,
        rms_alpha=rms_alpha,
        rms_inv_alpha=rms_inverse,
        cost=score,
        normalized_rms_alpha=normalized_rms_alpha,
        normalized_rms_inv_alpha=normalized_rms_inverse,
        max_normalized_alpha_error=maximum_relative_error,
        min_imag_alpha_fit_window=float(np.min(passivity_response.imag)),
        passivity_grid_points=int(passivity_energies.size),
        passive_on_fit_window=bool(np.min(passivity_response.imag) >= 0.0),
        passive_for_all_positive_frequencies=True,
    )


def build_cheng_modal_fit(
    profile: Cheng2007Profile,
    aspect_ratio: float,
    *,
    refit: bool = False,
) -> RationalLorentzFit:
    """Return a cached core fit or regenerate it through HybridQDPlasmonModel."""

    if not refit:
        return _cached_cheng_modal_fit(profile, aspect_ratio)
    wavelength_nm = 750.0
    qd_radius_nm = profile.qd_radius_nm(wavelength_nm)
    semimajor_nm = profile.semimajor_nm(aspect_ratio)
    _, internal_debye = radiatively_consistent_dipoles_debye(
        profile,
        wavelength_nm,
    )
    gamma1_ns_inv = (
        profile.isolated_radiative_rate_ns_inv
        + profile.extra_population_decay_ns_inv
    )
    mev_to_ns_inv = 1.0e-3 * E_CHARGE / hbar * 1.0e-9
    params = make_params_with_overrides(
        c_nm=semimajor_nm,
        a_nm=profile.mnp_semiminor_nm,
        r_nm=semimajor_nm + qd_radius_nm + 6.0,
        qd_radius_nm=qd_radius_nm,
        eps_m=profile.eps_environment,
        eps_qd=profile.eps_qd,
        d_debye=internal_debye,
        omega0_ev=float(photon_energy_ev(wavelength_nm)),
        gamma_population_mev=gamma1_ns_inv / mev_to_ns_inv,
        gamma2_coherence_mev=0.5 * gamma1_ns_inv / mev_to_ns_inv,
        orientation="long",
        qd_dipole_convention="bare_internal",
    )
    model = HybridQDPlasmonModel(
        params,
        orientation="long",
        n_modes=9,
        fit_window_eV=(0.8, 3.0),
        max_fit_normalized_rms=0.04,
        max_fit_pointwise_relative_error=0.08,
        radiative_consistency_policy="ignore",
        verbose=True,
    )
    return model.fit


def cheng_modal_alpha_au(
    profile: Cheng2007Profile,
    *,
    aspect_ratio: float,
    wavelength_nm: float | np.ndarray,
    fit: RationalLorentzFit | None = None,
) -> np.ndarray:
    selected_fit = fit or build_cheng_modal_fit(profile, aspect_ratio)
    energies_ev = photon_energy_ev(wavelength_nm)
    alpha_dimless = _evaluate_modal_fit_dimless(selected_fit, energies_ev)
    semiminor_au = float(nm_to_au(profile.mnp_semiminor_nm))
    semimajor_au = aspect_ratio * semiminor_au
    core_C = profile.eps_environment * semiminor_au**2 * semimajor_au / 3.0
    return np.asarray(core_C * alpha_dimless, dtype=complex)


def modal_poles_au(fit: RationalLorentzFit) -> np.ndarray:
    poles: list[complex] = []
    for omega_mode, gamma_mode in zip(
        fit.omega_modes_au,
        fit.gamma_modes_au,
    ):
        poles.extend(np.roots([1.0, gamma_mode, omega_mode**2]))
    return np.asarray(poles, dtype=complex)


def radiatively_consistent_dipoles_debye(
    profile: Cheng2007Profile,
    wavelength_nm: float,
) -> tuple[float, float]:
    """Return native external and corresponding bare-internal QD dipoles."""

    omega_si = 2.0 * np.pi * C_SI / (wavelength_nm * 1.0e-9)
    rate_s_inv = profile.isolated_radiative_rate_ns_inv * 1.0e9
    external_si = np.sqrt(
        rate_s_inv
        * 3.0
        * np.pi
        * epsilon_0
        * hbar
        * C_SI**3
        / (np.sqrt(profile.eps_environment) * omega_si**3)
    )
    external_debye = float(external_si / DEBYE_C_M)
    internal_debye = external_debye / profile.qd_local_field_factor
    return external_debye, float(internal_debye)


def homogeneous_rate_from_internal_dipole_ns_inv(
    profile: Cheng2007Profile,
    wavelength_nm: float,
    internal_dipole_debye_value: float,
) -> float:
    """Homogeneous-host rate implied by an internal dipole and native screening."""

    if (
        not np.isfinite(internal_dipole_debye_value)
        or internal_dipole_debye_value < 0.0
    ):
        raise ValueError("internal_dipole_debye_value must be finite and non-negative.")
    external_au = (
        internal_dipole_debye_value
        * profile.qd_local_field_factor
        * DEBYE_C_M
        / AU_DIPOLE_C_M
    )
    gamma_au = homogeneous_radiative_decay_rate_au(
        external_au,
        float(eV_to_au(photon_energy_ev(wavelength_nm))),
        profile.eps_environment,
    )
    return float(gamma_au / AU_TIME_S * 1.0e-9)


def internal_dipole_debye(
    profile: Cheng2007Profile,
    wavelength_nm: float,
    dipole_profile: DipoleProfile,
) -> float:
    if dipole_profile == "radiative_consistent":
        return radiatively_consistent_dipoles_debye(profile, wavelength_nm)[1]
    if dipole_profile == "figure_calibrated":
        return profile.calibrated_internal_dipole_debye
    raise ValueError(
        "dipole_profile must be 'radiative_consistent' or 'figure_calibrated'."
    )


@dataclass(frozen=True)
class ChengRatePoint:
    aspect_ratio: float
    wavelength_nm: float
    qd_radius_nm: float
    gap_nm: float
    geometry: ChengGeometry
    orientation_factor: float
    center_distance_nm: float
    reconstructed_gap_nm: float
    alpha_mnp_au: complex
    cheng_field_factor: complex
    native_field_factor: complex
    gamma_radiative_cheng_ns_inv: float
    radiative_rate_from_native_field_ratio_estimate_ns_inv: float
    internal_dipole_debye: float
    external_dipole_debye: float
    gamma_nonradiative_ns_inv: float
    feedback_ns_inv: complex
    mnp_long_axis_to_distance_ratio: float
    qd_radius_to_distance_ratio: float
    host_wavenumber_times_semimajor: float
    gamma1_with_zeta_ns_inv: float
    gamma2_cp_ns_inv: float


def evaluate_rate_point(
    profile: Cheng2007Profile,
    *,
    aspect_ratio: float,
    wavelength_nm: float,
    gap_nm: float,
    geometry: ChengGeometry,
    dipole_profile: DipoleProfile = "radiative_consistent",
) -> ChengRatePoint:
    """Evaluate Cheng rate formulas with the current project's Au response."""

    g = profile.orientation_factor(geometry)
    distance_nm = profile.center_distance_nm(
        aspect_ratio=aspect_ratio,
        wavelength_nm=wavelength_nm,
        gap_nm=gap_nm,
        geometry=geometry,
    )
    reconstructed_gap = profile.reconstructed_gap_nm(
        aspect_ratio=aspect_ratio,
        wavelength_nm=wavelength_nm,
        center_distance_nm=distance_nm,
        geometry=geometry,
    )
    if reconstructed_gap <= 0.0:
        raise ValueError("The directional QD-MNP gap must be strictly positive.")
    distance_au = float(nm_to_au(distance_nm))
    alpha = complex(
        longitudinal_alpha_au(
            profile,
            aspect_ratio=aspect_ratio,
            wavelength_nm=wavelength_nm,
        )
    )
    f_cheng = 1.0 + g * alpha / (profile.epsilon_eff * distance_au**3)
    f_native = 1.0 + g * alpha / (
        profile.eps_environment * distance_au**3
    )
    gamma_rad_cheng = profile.isolated_radiative_rate_ns_inv * abs(f_cheng) ** 2
    gamma_rad_native = profile.isolated_radiative_rate_ns_inv * abs(f_native) ** 2

    internal_debye = internal_dipole_debye(
        profile,
        wavelength_nm,
        dipole_profile,
    )
    internal_au = internal_debye * DEBYE_C_M / AU_DIPOLE_C_M
    external_debye = internal_debye * profile.qd_local_field_factor
    feedback_au = (
        2.0
        * g**2
        * internal_au**2
        * alpha
        / (profile.epsilon_eff**2 * distance_au**6)
    )
    feedback_ns_inv = complex(feedback_au / AU_TIME_S * 1.0e-9)
    gamma_nonradiative = float(feedback_ns_inv.imag)
    tolerance = 1.0e-11 * max(abs(feedback_ns_inv), 1.0)
    if gamma_nonradiative < -tolerance:
        raise RuntimeError(
            "Passive Johnson-Christy alpha produced a negative nonradiative rate."
        )
    gamma_nonradiative = max(gamma_nonradiative, 0.0)

    qd_radius = profile.qd_radius_nm(wavelength_nm)
    gamma1_with_zeta = gamma_rad_cheng + profile.extra_population_decay_ns_inv
    return ChengRatePoint(
        aspect_ratio=float(aspect_ratio),
        wavelength_nm=float(wavelength_nm),
        qd_radius_nm=qd_radius,
        gap_nm=float(gap_nm),
        geometry=geometry,
        orientation_factor=g,
        center_distance_nm=distance_nm,
        reconstructed_gap_nm=reconstructed_gap,
        alpha_mnp_au=alpha,
        cheng_field_factor=complex(f_cheng),
        native_field_factor=complex(f_native),
        gamma_radiative_cheng_ns_inv=float(gamma_rad_cheng),
        radiative_rate_from_native_field_ratio_estimate_ns_inv=float(
            gamma_rad_native
        ),
        internal_dipole_debye=float(internal_debye),
        external_dipole_debye=float(external_debye),
        gamma_nonradiative_ns_inv=gamma_nonradiative,
        feedback_ns_inv=feedback_ns_inv,
        mnp_long_axis_to_distance_ratio=float(
            profile.semimajor_nm(aspect_ratio) / distance_nm
        ),
        qd_radius_to_distance_ratio=float(qd_radius / distance_nm),
        host_wavenumber_times_semimajor=float(
            2.0
            * np.pi
            * np.sqrt(profile.eps_environment)
            * profile.semimajor_nm(aspect_ratio)
            / wavelength_nm
        ),
        gamma1_with_zeta_ns_inv=float(gamma1_with_zeta),
        gamma2_cp_ns_inv=float(0.5 * gamma1_with_zeta),
    )


def zero_feedback_area_population(
    pulse_area_rad: float | np.ndarray,
    field_factor: complex,
) -> np.ndarray:
    """Strong-drive, zero-feedback area-theorem proxy for Cheng Figure 4.

    Besides negligible population/coherence decay during the pulse, this
    expression requires resonant bare-QD excitation and
    ``|G_feedback|/Omega0 -> 0``.  In particular it does not retain the
    feedback-induced detuning present in Cheng's full Bloch equations.
    """

    theta = np.asarray(pulse_area_rad, dtype=float)
    if np.any(~np.isfinite(theta)):
        raise ValueError("pulse_area_rad must be finite.")
    return np.sin(0.5 * abs(field_factor) * theta) ** 2


def zero_feedback_period_pi_units(field_factor: complex) -> float:
    if not np.isfinite(field_factor) or abs(field_factor) == 0.0:
        return float("inf")
    return float(2.0 / abs(field_factor))


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _pdf_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _default_run_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("results") / "cheng2007_comparison" / stamp


def _rate_row(point: ChengRatePoint, dipole_profile: DipoleProfile) -> dict[str, object]:
    return {
        "dipole_profile": dipole_profile,
        "aspect_ratio": point.aspect_ratio,
        "wavelength_nm": point.wavelength_nm,
        "qd_radius_nm": point.qd_radius_nm,
        "gap_nm": point.gap_nm,
        "geometry": point.geometry,
        "orientation_factor": point.orientation_factor,
        "center_distance_nm": point.center_distance_nm,
        "reconstructed_gap_nm": point.reconstructed_gap_nm,
        "alpha_real_au": point.alpha_mnp_au.real,
        "alpha_imag_au": point.alpha_mnp_au.imag,
        "cheng_field_factor_real": point.cheng_field_factor.real,
        "cheng_field_factor_imag": point.cheng_field_factor.imag,
        "native_field_factor_real": point.native_field_factor.real,
        "native_field_factor_imag": point.native_field_factor.imag,
        "gamma_radiative_cheng_ns_inv": point.gamma_radiative_cheng_ns_inv,
        "radiative_rate_from_native_field_ratio_estimate_ns_inv": (
            point.radiative_rate_from_native_field_ratio_estimate_ns_inv
        ),
        "internal_dipole_debye": point.internal_dipole_debye,
        "external_dipole_debye": point.external_dipole_debye,
        "gamma_nonradiative_ns_inv": point.gamma_nonradiative_ns_inv,
        "feedback_real_ns_inv": point.feedback_ns_inv.real,
        "feedback_imag_ns_inv": point.feedback_ns_inv.imag,
        "mnp_long_axis_to_distance_ratio": point.mnp_long_axis_to_distance_ratio,
        "qd_radius_to_distance_ratio": point.qd_radius_to_distance_ratio,
        "host_wavenumber_times_semimajor": (
            point.host_wavenumber_times_semimajor
        ),
        "gamma1_with_zeta_ns_inv": point.gamma1_with_zeta_ns_inv,
        "gamma2_cp_ns_inv": point.gamma2_cp_ns_inv,
    }


def _build_modal_audit(
    profile: Cheng2007Profile,
    *,
    refit: bool,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Compare direct material alpha with the unchanged core's modal fit."""

    rows: list[dict[str, object]] = []
    fit_summaries: dict[str, object] = {}
    for aspect_ratio in (1.0, 3.0, 4.0):
        fit = build_cheng_modal_fit(profile, aspect_ratio, refit=refit)
        poles = modal_poles_au(fit)
        stable_poles = bool(np.all(poles.real < 0.0))
        structural_passivity = bool(
            np.all(fit.strengths_au2 >= 0.0)
            and np.all(fit.gamma_modes_au > 0.0)
            and fit.passive_for_all_positive_frequencies
        )
        if not fit.passive_on_fit_window or not structural_passivity:
            raise RuntimeError(
                f"The Cheng q={aspect_ratio:g} Lorentz audit is not passive."
            )
        if not stable_poles:
            raise RuntimeError(
                f"The Cheng q={aspect_ratio:g} Lorentz audit has unstable poles."
            )
        fit_summary: dict[str, object] = {
            "n_modes": int(fit.strengths_au2.size),
            "normalized_rms_alpha": float(fit.normalized_rms_alpha),
            "normalized_rms_inverse_alpha": float(
                fit.normalized_rms_inv_alpha
            ),
            "maximum_fit_window_relative_alpha_error": float(
                fit.max_normalized_alpha_error
            ),
            "minimum_imaginary_alpha_fit_window": float(
                fit.min_imag_alpha_fit_window
            ),
            "minimum_strength_au2": float(np.min(fit.strengths_au2)),
            "minimum_gamma_au": float(np.min(fit.gamma_modes_au)),
            "maximum_pole_real_part_au": float(np.max(poles.real)),
            "passive": True,
            "stable_poles": True,
            "source": "regenerated_with_core" if refit else "cached_core_fit",
        }
        direct_imaginary: list[float] = []
        modal_imaginary: list[float] = []
        for wavelength_nm in profile.wavelengths_nm:
            direct = complex(
                longitudinal_alpha_au(
                    profile,
                    aspect_ratio=aspect_ratio,
                    wavelength_nm=wavelength_nm,
                )
            )
            modal = complex(
                cheng_modal_alpha_au(
                    profile,
                    aspect_ratio=aspect_ratio,
                    wavelength_nm=wavelength_nm,
                    fit=fit,
                )
            )
            relative_error = abs(modal - direct) / max(
                abs(direct), np.finfo(float).tiny
            )
            direct_imaginary.append(direct.imag)
            modal_imaginary.append(modal.imag)
            rows.append(
                {
                    "aspect_ratio": aspect_ratio,
                    "wavelength_nm": wavelength_nm,
                    "photon_energy_ev": float(photon_energy_ev(wavelength_nm)),
                    "direct_alpha_real_au": direct.real,
                    "direct_alpha_imag_au": direct.imag,
                    "modal_alpha_real_au": modal.real,
                    "modal_alpha_imag_au": modal.imag,
                    "relative_complex_alpha_error": relative_error,
                    "modal_alpha_passive_at_carrier": modal.imag >= 0.0,
                }
            )
        direct_imaginary_array = np.asarray(direct_imaginary, dtype=float)
        modal_imaginary_array = np.asarray(modal_imaginary, dtype=float)
        loss_channel_error = float(
            np.max(np.abs(modal_imaginary_array - direct_imaginary_array))
            / max(
                float(np.max(np.abs(direct_imaginary_array))),
                np.finfo(float).tiny,
            )
        )
        if loss_channel_error > MAX_MODAL_AUDIT_LOSS_CHANNEL_ERROR:
            raise RuntimeError(
                "The Cheng modal loss-channel audit failed for "
                f"q={aspect_ratio:g}: normalized error={loss_channel_error:.6g}, "
                f"limit={MAX_MODAL_AUDIT_LOSS_CHANNEL_ERROR:.6g}."
            )
        fit_summary["carrier_loss_channel_normalized_error"] = (
            loss_channel_error
        )
        fit_summary["carrier_loss_channel_acceptance_limit"] = (
            MAX_MODAL_AUDIT_LOSS_CHANNEL_ERROR
        )
        fit_summaries[f"q={aspect_ratio:g}"] = fit_summary

    maximum_error = max(
        float(row["relative_complex_alpha_error"]) for row in rows
    )
    if maximum_error > MAX_MODAL_AUDIT_RELATIVE_ERROR:
        raise RuntimeError(
            "The Cheng modal audit does not reproduce the direct material "
            f"response: max relative alpha error={maximum_error:.6g}, "
            f"limit={MAX_MODAL_AUDIT_RELATIVE_ERROR:.6g}."
        )
    return rows, {
        "aspect_ratios": [1.0, 3.0, 4.0],
        "wavelengths_nm": list(profile.wavelengths_nm),
        "maximum_carrier_relative_complex_alpha_error": maximum_error,
        "acceptance_limit": MAX_MODAL_AUDIT_RELATIVE_ERROR,
        "all_carrier_samples_accepted": True,
        "fits": fit_summaries,
    }


def run_comparison(
    *,
    output_dir: str | Path | None = None,
    q_points: int = 321,
    theta_points: int = 501,
    dipole_profiles: tuple[DipoleProfile, ...] = (
        "radiative_consistent",
        "figure_calibrated",
    ),
    refit_modal_audit: bool = False,
    make_plots: bool = True,
) -> Path:
    """Calculate Cheng rate analogues and the zero-feedback area proxy."""

    if q_points < 21 or theta_points < 21:
        raise ValueError("q_points and theta_points must both be at least 21.")
    if len(dipole_profiles) == 0:
        raise ValueError("At least one dipole profile must be requested.")
    for name in dipole_profiles:
        if name not in ("radiative_consistent", "figure_calibrated"):
            raise ValueError(f"Unsupported dipole profile: {name!r}.")

    profile = Cheng2007Profile()
    modal_audit_rows, modal_audit_summary = _build_modal_audit(
        profile,
        refit=refit_modal_audit,
    )
    run_dir = Path(output_dir) if output_dir is not None else _default_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        run_dir / "modal_audit.csv",
        list(modal_audit_rows[0]),
        modal_audit_rows,
    )
    q_grid = np.linspace(1.0, 9.0, q_points)

    rate_rows: list[dict[str, object]] = []
    points_by_key: dict[
        tuple[DipoleProfile, ChengGeometry, float, float],
        list[ChengRatePoint],
    ] = {}
    for dipole_profile in dipole_profiles:
        for geometry in ("end", "side"):
            for wavelength in profile.wavelengths_nm:
                for gap in profile.gaps_nm:
                    key = (dipole_profile, geometry, wavelength, gap)
                    collection: list[ChengRatePoint] = []
                    for aspect_ratio in q_grid:
                        point = evaluate_rate_point(
                            profile,
                            aspect_ratio=float(aspect_ratio),
                            wavelength_nm=wavelength,
                            gap_nm=gap,
                            geometry=geometry,
                            dipole_profile=dipole_profile,
                        )
                        collection.append(point)
                        rate_rows.append(_rate_row(point, dipole_profile))
                    points_by_key[key] = collection
    _write_csv(run_dir / "figures_2_3_rates.csv", list(rate_rows[0]), rate_rows)

    theta = np.linspace(0.0, 5.0 * np.pi, theta_points)
    area_rows: list[dict[str, object]] = []
    for geometry in ("end", "side"):
        for aspect_ratio in (1.0, 3.0, 4.0):
            # f does not contain the unknown QD dipole, so either dipole profile
            # gives exactly the same area-theorem proxy.  Evaluate it only once.
            point = evaluate_rate_point(
                profile,
                aspect_ratio=aspect_ratio,
                wavelength_nm=750.0,
                gap_nm=6.0,
                geometry=geometry,
                dipole_profile="radiative_consistent",
            )
            cheng_population = zero_feedback_area_population(
                theta, point.cheng_field_factor
            )
            native_population = zero_feedback_area_population(
                theta, point.native_field_factor
            )
            for index, theta_value in enumerate(theta):
                area_rows.append(
                    {
                        "geometry": geometry,
                        "aspect_ratio": aspect_ratio,
                        "theta_rad": theta_value,
                        "theta_pi_units": theta_value / np.pi,
                        "population_zero_feedback_cheng_convention": (
                            cheng_population[index]
                        ),
                        "population_zero_feedback_native_convention": (
                            native_population[index]
                        ),
                        "cheng_period_pi_units": zero_feedback_period_pi_units(
                            point.cheng_field_factor
                        ),
                        "native_period_pi_units": zero_feedback_period_pi_units(
                            point.native_field_factor
                        ),
                    }
                )
    _write_csv(
        run_dir / "figure_4_zero_feedback_area_proxy.csv",
        list(area_rows[0]),
        area_rows,
    )

    q_period = np.linspace(1.0, 5.0, q_points)
    period_rows: list[dict[str, object]] = []
    for geometry in ("end", "side"):
        for aspect_ratio in q_period:
            point = evaluate_rate_point(
                profile,
                aspect_ratio=float(aspect_ratio),
                wavelength_nm=750.0,
                gap_nm=6.0,
                geometry=geometry,
                dipole_profile="radiative_consistent",
            )
            period_rows.append(
                {
                    "geometry": geometry,
                    "aspect_ratio": aspect_ratio,
                    "cheng_period_pi_units": zero_feedback_period_pi_units(
                        point.cheng_field_factor
                    ),
                    "native_period_pi_units": zero_feedback_period_pi_units(
                        point.native_field_factor
                    ),
                    "mnp_long_axis_to_distance_ratio": (
                        point.mnp_long_axis_to_distance_ratio
                    ),
                }
            )
    _write_csv(
        run_dir / "figure_4_zero_feedback_periods.csv",
        list(period_rows[0]),
        period_rows,
    )

    peak_summary: dict[str, object] = {}
    for dipole_profile in dipole_profiles:
        profile_peaks: dict[str, object] = {}
        for wavelength in profile.wavelengths_nm:
            for geometry in ("end", "side"):
                key = (dipole_profile, geometry, wavelength, 2.0)
                collection = points_by_key[key]
                radiative = np.asarray(
                    [point.gamma_radiative_cheng_ns_inv for point in collection]
                )
                nonradiative = np.asarray(
                    [point.gamma_nonradiative_ns_inv for point in collection]
                )
                profile_peaks[f"{geometry}_{wavelength:g}nm"] = {
                    "radiative_peak_q": float(q_grid[np.argmax(radiative)]),
                    "radiative_peak_ns_inv": float(np.max(radiative)),
                    "nonradiative_peak_q": float(q_grid[np.argmax(nonradiative)]),
                    "nonradiative_peak_ns_inv": float(np.max(nonradiative)),
                }
        peak_summary[dipole_profile] = profile_peaks

    external_dipoles = {
        f"{wavelength:g}": {
            "external_debye": radiatively_consistent_dipoles_debye(
                profile, wavelength
            )[0],
            "internal_debye": radiatively_consistent_dipoles_debye(
                profile, wavelength
            )[1],
        }
        for wavelength in profile.wavelengths_nm
    }
    calibrated_dipole_consistency = {
        f"{wavelength:g}": {
            "wavelength_nm": wavelength,
            "internal_dipole_debye": profile.calibrated_internal_dipole_debye,
            "external_dipole_debye": (
                profile.calibrated_internal_dipole_debye
                * profile.qd_local_field_factor
            ),
            "implied_homogeneous_radiative_rate_ns_inv": (
                homogeneous_rate_from_internal_dipole_ns_inv(
                    profile,
                    wavelength,
                    profile.calibrated_internal_dipole_debye,
                )
            ),
            "ratio_to_article_gamma_rad0": (
                homogeneous_rate_from_internal_dipole_ns_inv(
                    profile,
                    wavelength,
                    profile.calibrated_internal_dipole_debye,
                )
                / profile.isolated_radiative_rate_ns_inv
            ),
        }
        for wavelength in profile.wavelengths_nm
    }
    resolved_plot_profile: DipoleProfile | None = None
    if make_plots:
        resolved_plot_profile = (
            "figure_calibrated"
            if "figure_calibrated" in dipole_profiles
            else dipole_profiles[0]
        )
    metadata = {
        "comparison": (
            "Cheng 2007 direct-frequency reference plus unchanged-core modal audit"
        ),
        "article": {
            "title": "Coherent exciton-plasmon interaction in the hybrid semiconductor quantum dot and metal nanoparticle complex",
            "doi": "10.1364/OL.32.002125",
            "local_pdf": "articles/cheng2007.pdf",
            "local_pdf_sha256": _pdf_sha256(Path("articles/cheng2007.pdf")),
        },
        "mathematical_core_modified": False,
        "numerical_settings": {
            "q_points": q_points,
            "figures_2_3_q_interval": [1.0, 9.0],
            "figure_3_display_q_interval": [1.0, 7.0],
            "theta_points": theta_points,
            "area_proxy_theta_pi_interval": [0.0, 5.0],
            "period_q_interval": [1.0, 5.0],
            "refit_modal_audit": refit_modal_audit,
            "make_plots": make_plots,
            "resolved_plot_dipole_profile": resolved_plot_profile,
        },
        "response_used": {
            "figure_2_3_and_area_proxy": (
                "direct Johnson-Christy target of the core passive modal fit"
            ),
            "modal_audit": (
                "unchanged HybridQDPlasmonModel N=9 passive Lorentz decomposition"
            ),
        },
        "why_no_q_sweep_refit": (
            "Figures 2--3 and the area-only Figure 4 use alpha only at one carrier. "
            "A separate Lorentz refit for every q would approximate the same direct "
            "frequency sample and add fit error without adding dynamics."
        ),
        "geometry_adapter": {
            "longitudinal_alpha_for_both_positions": True,
            "end_orientation_factor": 2.0,
            "side_orientation_factor": -1.0,
            "reason": (
                "Cheng keeps the incident field along the long MNP axis while moving "
                "the QD from the end to the side."
            ),
        },
        "profile_provenance": profile.provenance(),
        "dipole_profiles_exported": list(dipole_profiles),
        "radiatively_consistent_dipoles": external_dipoles,
        "figure_calibrated_dipole_consistency_diagnostic": (
            calibrated_dipole_consistency
        ),
        "modal_audit": modal_audit_summary,
        "local_field_conventions": {
            "epsilon_eff": profile.epsilon_eff,
            "native_qd_local_field_factor": profile.qd_local_field_factor,
            "cheng_f": "1 + g alpha/(epsilon_eff d^3)",
            "native_drive_ratio": "1 + g alpha/(epsilon_host d^3)",
            "both_are_exported": True,
        },
        "figure_4_scope": {
            "calculated": "strong-drive zero-feedback area-theorem proxy",
            "not_calculated": "article's under-specified dissipative pulse dynamics",
            "population_formula": "sin^2(|f| theta/2)",
            "period_pi_units": "2/|f|",
            "required_limits": [
                "bare detuning delta/Omega0 -> 0 (resonant excitation)",
                "gamma10/Omega0 -> 0",
                "Gamma2/Omega0 -> 0",
                "|G_feedback|/Omega0 -> 0",
            ],
            "independent_of_unknown_qd_dipole": True,
        },
        "peak_summary_by_dipole_profile": peak_summary,
        "invariants": {
            "minimum_alpha_imag_au": float(
                min(float(row["alpha_imag_au"]) for row in rate_rows)
            ),
            "minimum_gamma_radiative_ns_inv": float(
                min(float(row["gamma_radiative_cheng_ns_inv"]) for row in rate_rows)
            ),
            "minimum_gamma_nonradiative_ns_inv": float(
                min(float(row["gamma_nonradiative_ns_inv"]) for row in rate_rows)
            ),
            "minimum_reconstructed_gap_nm": float(
                min(float(row["reconstructed_gap_nm"]) for row in rate_rows)
            ),
            "maximum_mnp_long_axis_to_distance_ratio": float(
                max(float(row["mnp_long_axis_to_distance_ratio"]) for row in rate_rows)
            ),
            "maximum_host_wavenumber_times_semimajor": float(
                max(
                    float(row["host_wavenumber_times_semimajor"])
                    for row in rate_rows
                )
            ),
            "Gamma2_equals_gamma1_over_2_for_figure_4_profile": True,
            "modal_audit_maximum_relative_complex_alpha_error": (
                modal_audit_summary[
                    "maximum_carrier_relative_complex_alpha_error"
                ]
            ),
            "all_modal_audit_fits_passive_and_stable": True,
        },
        "limitations": [
            "The paper uses Palik Au data; the native project uses Johnson-Christy.",
            "The QD transition dipole is not printed, so Figure 3 has no unique amplitude.",
            "The article text and Figure 3(a) disagree on the reported g=2 maximum.",
            "Figure 4 pulse shape, duration and amplitude are not printed.",
            "For large q, especially in side geometry, one-MNP point-dipole applicability is poor.",
            (
                "At q=9 and 600 nm, k_host*a is about 0.85 rather than much "
                "smaller than one, so quasistatic no-retardation accuracy is "
                "also uncontrolled at the far end of the article sweep."
            ),
            "The electrostatic model does not derive an MNP-modified spontaneous-emission Lindblad rate.",
            (
                "The figure-calibrated 1.82 D dipole is only a Figure-3 scale "
                "diagnostic: at 750 nm it implies about 0.0008 ns^-1 in the "
                "homogeneous host, a factor of about 100 below the separately "
                "reported gamma_rad0=0.08 ns^-1."
            ),
            (
                "The same calibrated 1.82 D is applied to the three different "
                "QD radii only because the article does not print their dipoles."
            ),
            (
                "The Figure-4 proxy removes both dissipation and coherent "
                "feedback; it is not the general lossless limit of the full "
                "Cheng Bloch equations."
            ),
        ],
    }
    with (run_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    # Reusing an explicit output directory must not leave plots from a previous
    # dipole profile or from a run that enabled plotting.  Only this script's
    # exact, known optional filenames are removed; CSV/JSON are regenerated.
    for plot_name in _CHENG_PLOT_ARTIFACTS:
        (run_dir / plot_name).unlink(missing_ok=True)
    if make_plots:
        assert resolved_plot_profile is not None
        plot_profile = resolved_plot_profile
        colors = {2.0: "C0", 4.0: "C1", 6.0: "C2"}
        wavelength_colors = {600.0: "C0", 750.0: "C1", 900.0: "C2"}

        fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.4), sharex=True)
        for column, geometry in enumerate(("end", "side")):
            for gap in profile.gaps_nm:
                collection = points_by_key[(plot_profile, geometry, 750.0, gap)]
                axes[0, column].plot(
                    q_grid,
                    [point.gamma_radiative_cheng_ns_inv for point in collection],
                    color=colors[gap],
                    label=rf"$\Delta={gap:g}$ nm",
                )
            for wavelength in profile.wavelengths_nm:
                collection = points_by_key[(plot_profile, geometry, wavelength, 2.0)]
                axes[1, column].plot(
                    q_grid,
                    [point.gamma_radiative_cheng_ns_inv for point in collection],
                    color=wavelength_colors[wavelength],
                    label=f"{wavelength:g} nm",
                )
            axes[0, column].set_title("to-the-end, g=2" if geometry == "end" else "by-the-side, g=-1")
            axes[1, column].set_xlabel("aspect ratio q")
            for row in (0, 1):
                axes[row, column].grid(alpha=0.25)
                axes[row, column].set_xlim(1.0, 9.0)
        axes[0, 0].set_ylabel(r"$\gamma_{rad}$ (ns$^{-1}$)")
        axes[1, 0].set_ylabel(r"$\gamma_{rad}$ (ns$^{-1}$)")
        axes[0, 0].legend()
        axes[1, 0].legend()
        fig.suptitle("Cheng Fig. 2 analogue: JC alpha, Cheng local-field convention")
        fig.tight_layout()
        fig.savefig(run_dir / "figure_2_radiative_rates.png", dpi=180)
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.4), sharex=True)
        for column, geometry in enumerate(("end", "side")):
            for gap in profile.gaps_nm:
                collection = points_by_key[(plot_profile, geometry, 750.0, gap)]
                axes[0, column].plot(
                    q_grid,
                    [point.gamma_nonradiative_ns_inv for point in collection],
                    color=colors[gap],
                    label=rf"$\Delta={gap:g}$ nm",
                )
            for wavelength in profile.wavelengths_nm:
                collection = points_by_key[(plot_profile, geometry, wavelength, 2.0)]
                axes[1, column].plot(
                    q_grid,
                    [point.gamma_nonradiative_ns_inv for point in collection],
                    color=wavelength_colors[wavelength],
                    label=f"{wavelength:g} nm",
                )
            axes[0, column].set_title("to-the-end, g=2" if geometry == "end" else "by-the-side, g=-1")
            axes[1, column].set_xlabel("aspect ratio q")
            for row in (0, 1):
                axes[row, column].grid(alpha=0.25)
                axes[row, column].set_xlim(1.0, 7.0)
        axes[0, 0].set_ylabel(r"$\gamma_{nonrad}$ (ns$^{-1}$)")
        axes[1, 0].set_ylabel(r"$\gamma_{nonrad}$ (ns$^{-1}$)")
        axes[0, 0].legend()
        axes[1, 0].legend()
        fig.suptitle(
            "Cheng Fig. 3 analogue: "
            + ("figure-calibrated 1.82 D" if plot_profile == "figure_calibrated" else "radiatively consistent dipole")
        )
        fig.tight_layout()
        fig.savefig(run_dir / f"figure_3_nonradiative_rates_{plot_profile}.png", dpi=180)
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.4))
        for row, geometry in enumerate(("end", "side")):
            for aspect_ratio in (1.0, 3.0, 4.0):
                point = evaluate_rate_point(
                    profile,
                    aspect_ratio=aspect_ratio,
                    wavelength_nm=750.0,
                    gap_nm=6.0,
                    geometry=geometry,
                    dipole_profile=plot_profile,
                )
                axes[row, 0].plot(
                    theta / np.pi,
                    zero_feedback_area_population(
                        theta, point.cheng_field_factor
                    ),
                    label=f"q={aspect_ratio:g}",
                )
            period_values = []
            for aspect_ratio in q_period:
                point = evaluate_rate_point(
                    profile,
                    aspect_ratio=float(aspect_ratio),
                    wavelength_nm=750.0,
                    gap_nm=6.0,
                    geometry=geometry,
                    dipole_profile=plot_profile,
                )
                period_values.append(
                    zero_feedback_period_pi_units(point.cheng_field_factor)
                )
            axes[row, 1].plot(q_period, period_values)
            axes[row, 0].set_ylabel(r"$\rho_{22}$")
            axes[row, 1].set_ylabel(r"period / $\pi$")
            axes[row, 0].set_title("g=2" if geometry == "end" else "g=-1")
            axes[row, 1].set_title("g=2" if geometry == "end" else "g=-1")
            axes[row, 0].grid(alpha=0.25)
            axes[row, 1].grid(alpha=0.25)
        axes[0, 0].legend()
        axes[1, 0].legend()
        axes[1, 0].set_xlabel(r"input area $\theta/\pi$")
        axes[1, 1].set_xlabel("aspect ratio q")
        fig.suptitle(
            "Cheng Fig. 4: strong-drive zero-feedback area proxy "
            "(not dissipative replica)"
        )
        fig.tight_layout()
        fig.savefig(run_dir / "figure_4_zero_feedback_area_proxy.png", dpi=180)
        plt.close(fig)

    return run_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate Cheng-2007 figure analogues with the unchanged native "
            "longitudinal quasistatic MNP response."
        )
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--q-points", type=int, default=321)
    parser.add_argument("--theta-points", type=int, default=501)
    parser.add_argument(
        "--dipole-profile",
        choices=("both", "radiative_consistent", "figure_calibrated"),
        default="both",
    )
    parser.add_argument(
        "--refit-modal-audit",
        action="store_true",
        help=(
            "Regenerate the q=1,3,4 N=9 passive fits through the unchanged "
            "HybridQDPlasmonModel instead of using deterministic cached fits."
        ),
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    dipole_profiles: tuple[DipoleProfile, ...]
    if args.dipole_profile == "both":
        dipole_profiles = ("radiative_consistent", "figure_calibrated")
    else:
        dipole_profiles = (args.dipole_profile,)
    run_dir = run_comparison(
        output_dir=args.output_dir,
        q_points=args.q_points,
        theta_points=args.theta_points,
        dipole_profiles=dipole_profiles,
        refit_modal_audit=args.refit_modal_audit,
        make_plots=not args.no_plots,
    )
    print(f"Saved Cheng-2007 comparison to {run_dir}")


if __name__ == "__main__":
    main()
