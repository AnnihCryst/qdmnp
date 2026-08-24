"""Основной решатель новой модели взаимодействия квантовой точки и МНЧ.

Модуль задает физические параметры системы КТ-МНЧ, строит квазистатическую
поляризуемость вытянутой металлической наночастицы по табличным оптическим
данным золота и аппроксимирует ее устойчивой рациональной моделью - суммой
лоренцевых мод ``alpha(omega)``.

Эта аппроксимация затем используется во временной системе ОДУ: моды МНЧ
описывают плазмонный диполь, а КТ описывается уравнениями Блоха для
двухуровневой системы. Модуль можно использовать как библиотеку для расчетных
скриптов ``qd_mnp_linear_spectrum.py``, ``qd_mnp_fano_scan.py`` и
``qd_mnp_pulse_absorption_sweep.py``. При прямом запуске файл выполняет
диагностический пример и строит проверочные графики аппроксимации/динамики.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Iterable, Literal

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c as C_SI, epsilon_0, e as E_CHARGE, hbar, physical_constants
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.special import erf

# ================================================================
# Atomic units
# ================================================================
AU_LENGTH_M = physical_constants['Bohr radius'][0]
AU_TIME_S = physical_constants['atomic unit of time'][0]
AU_ENERGY_J = physical_constants['Hartree energy'][0]
AU_ENERGY_EV = physical_constants['Hartree energy in eV'][0]
AU_FIELD_V_M = AU_ENERGY_J / (E_CHARGE * AU_LENGTH_M)
AU_DIPOLE_C_M = E_CHARGE * AU_LENGTH_M
AU_SPEED_OF_LIGHT = C_SI * AU_TIME_S / AU_LENGTH_M
DEBYE_C_M = 3.33564e-30
SCHEMA_VERSION = 3
NATIVE_MODEL_PROFILE = 'quasistatic_ellipsoid_tls'
MATERIAL_INTERPOLATION = 'piecewise_linear_n_k_no_extrapolation'
MATERIAL_HIGH_FREQUENCY_EPSILON = 1.0

DipoleOrientation = Literal['long', 'trans']
QDDipoleConvention = Literal['bare_internal', 'effective_external']
ORIENTATION_FACTORS: dict[DipoleOrientation, float] = {
    'long': 2.0,
    'trans': -1.0,
}

# Deterministic initial guesses for the bundled Johnson--Christy table,
# default aspect ratio and 0.8--3.0 eV window.  These are warm starts only:
# least_squares still refits them to the selected objective and the result must
# pass the same accuracy/passivity gates.  Generic parameters fall back to the
# continuation fitter below.
_CANONICAL_PASSIVE_N9_SEEDS: dict[DipoleOrientation, tuple[float, np.ndarray, np.ndarray, np.ndarray]] = {
    'long': (
        0.0,
        np.asarray([
            0.00182692289943019,
            0.00367553157594101,
            0.00326181017535397,
            0.00170965263072298,
            0.0018253478998092,
            0.0026589672879989,
            0.00316106916302188,
            0.00381834684294691,
            0.16316300479420298,
        ]),
        np.asarray([
            0.0803943949286627,
            0.08322114062055991,
            0.0858598716468505,
            0.08934839500233095,
            0.09421600524605818,
            0.1003647685083918,
            0.10724697895721329,
            0.11410184481086819,
            0.20925461512521068,
        ]),
        np.asarray([
            0.00472655604494313,
            0.00424854917096931,
            0.00471251493221726,
            0.00612156287181602,
            0.00853883569244901,
            0.01086850337000413,
            0.01117455319447792,
            0.0057933128198453,
            0.00029399457740532,
        ]),
    ),
    'trans': (
        0.0,
        np.asarray([
            0.000046436978239069719,
            0.00012129927641792417,
            0.00025058513895763208,
            0.00036364973079320439,
            0.00039659424332527546,
            0.00046000865045366877,
            0.00054942597615988988,
            0.00076073342700850514,
            0.12104933574946282,
        ]),
        np.asarray([
            0.08128491714874338,
            0.08548313502945135,
            0.08865405251984387,
            0.09162538404093074,
            0.09562941136993196,
            0.10082269909492125,
            0.10679296092816247,
            0.11315620070290949,
            0.24254552635938523,
        ]),
        np.asarray([
            0.00443594368282125,
            0.00428625635222494,
            0.00437703099834203,
            0.00512272659885935,
            0.00675913058169198,
            0.00844356992258622,
            0.00906793752017132,
            0.00485854150372238,
            0.00029399457740532,
        ]),
    ),
}


def orientation_factor(orientation: DipoleOrientation) -> float:
    """Return the quasistatic dipole-tensor factor for an orientation."""
    try:
        return ORIENTATION_FACTORS[orientation]
    except KeyError as exc:
        raise ValueError("orientation must be either 'long' or 'trans'.") from exc


def eV_to_au(value: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(value) / AU_ENERGY_EV


def au_to_eV(value: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(value) * AU_ENERGY_EV


def nm_to_au(value: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(value) * 1e-9 / AU_LENGTH_M


def au_to_nm(value: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(value) * AU_LENGTH_M * 1e9


def fs_to_au(value: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(value) * 1e-15 / AU_TIME_S


def homogeneous_radiative_decay_rate_au(
    d_external_au: float,
    omega0_au: float,
    eps_m: float,
) -> float:
    """Macroscopic homogeneous-host electric-dipole population decay rate.

    ``d_external_au`` is the dipole visible in the host.  Thus callers using a
    bare internal QD matrix element must apply the same electrostatic local-
    field factor that converts it to the emitted external dipole.  The host is
    the lossless, nondispersive medium assumed everywhere else in this model.
    """

    values = np.asarray([d_external_au, omega0_au, eps_m], dtype=float)
    if np.any(~np.isfinite(values)):
        raise ValueError('Radiative-rate inputs must be finite.')
    if d_external_au < 0.0 or omega0_au <= 0.0 or eps_m <= 0.0:
        raise ValueError(
            'Radiative-rate dipole must be non-negative and frequency/eps_m positive.'
        )
    dipole_si = d_external_au * AU_DIPOLE_C_M
    omega_si = omega0_au / AU_TIME_S
    rate_s = (
        np.sqrt(eps_m)
        * omega_si**3
        * dipole_si**2
        / (3.0 * np.pi * epsilon_0 * hbar * C_SI**3)
    )
    return float(rate_s * AU_TIME_S)


def ns_to_au(value: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(value) * 1e-9 / AU_TIME_S


def au_to_fs(value: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(value) * AU_TIME_S * 1e15


def field_si_to_au(value: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(value) / AU_FIELD_V_M


def field_au_to_si(value: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(value) * AU_FIELD_V_M


def dipole_si_to_au(value: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(value) / AU_DIPOLE_C_M


def dipole_au_to_debye(value: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(value) * AU_DIPOLE_C_M / DEBYE_C_M


def timestamped_run_dir(base_dir: str | Path) -> Path:
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return Path(base_dir) / stamp


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ================================================================
# Material data
# ================================================================
@dataclass(frozen=True)
class MaterialDispersion:
    energy_eV: np.ndarray
    n: np.ndarray
    k: np.ndarray

    def __post_init__(self) -> None:
        energy = np.array(self.energy_eV, dtype=float, order='C', copy=True)
        n = np.array(self.n, dtype=float, order='C', copy=True)
        k = np.array(self.k, dtype=float, order='C', copy=True)
        if energy.ndim != 1 or n.ndim != 1 or k.ndim != 1:
            raise ValueError('Material arrays energy_eV, n and k must be one-dimensional.')
        if not (len(energy) == len(n) == len(k)) or len(energy) < 2:
            raise ValueError('Material arrays energy_eV, n and k must have the same length >= 2.')
        if not (np.all(np.isfinite(energy)) and np.all(np.isfinite(n)) and np.all(np.isfinite(k))):
            raise ValueError('Material arrays must contain only finite values.')
        if np.any(np.diff(energy) <= 0.0):
            raise ValueError('Material energy_eV values must be strictly increasing.')
        if np.any(energy <= 0.0) or np.any(n < 0.0) or np.any(k < 0.0):
            raise ValueError('Material energies must be positive and optical constants non-negative.')
        for values in (energy, n, k):
            values.setflags(write=False)
        object.__setattr__(self, 'energy_eV', energy)
        object.__setattr__(self, 'n', n)
        object.__setattr__(self, 'k', k)

    @property
    def epsilon(self) -> np.ndarray:
        return (self.n + 1j * self.k) ** 2

    def optical_constants_at(
        self,
        energy_eV: float | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Linearly interpolate the tabulated optical constants ``n`` and ``k``.

        The Johnson--Christy data are discrete measurements.  Every continuous
        spectrum and every modal quality gate therefore needs one explicit
        interpolation convention.  Piecewise-linear interpolation is used
        because it is local, deterministic, preserves non-negative ``n`` and
        ``k``, and cannot introduce spline overshoot between measurements.
        Extrapolation is deliberately forbidden.
        """

        energy = np.asarray(energy_eV, dtype=float)
        if np.any(~np.isfinite(energy)):
            raise ValueError('Requested material energies must be finite.')
        scale = max(abs(float(self.energy_eV[0])), abs(float(self.energy_eV[-1])), 1.0)
        tolerance = 1e-12 * scale
        if np.any(energy < self.energy_eV[0] - tolerance) or np.any(
            energy > self.energy_eV[-1] + tolerance
        ):
            raise ValueError(
                'Requested material energy lies outside the tabulated interval '
                f'[{self.energy_eV[0]:g}, {self.energy_eV[-1]:g}] eV.'
            )
        clipped = np.clip(energy, self.energy_eV[0], self.energy_eV[-1])
        return (
            np.asarray(np.interp(clipped, self.energy_eV, self.n), dtype=float),
            np.asarray(np.interp(clipped, self.energy_eV, self.k), dtype=float),
        )

    def epsilon_at(self, energy_eV: float | np.ndarray) -> np.ndarray:
        """Return dielectric data under the project's explicit interpolation rule."""

        n_interp, k_interp = self.optical_constants_at(energy_eV)
        return np.asarray((n_interp + 1j * k_interp) ** 2, dtype=complex)


DEFAULT_AU_MATERIAL = MaterialDispersion(
    energy_eV=np.array([
        0.64, 0.77, 0.89, 1.02, 1.14, 1.26, 1.39, 1.51, 1.64, 1.76, 1.88, 2.01,
        2.13, 2.26, 2.38, 2.50, 2.63, 2.75, 2.88, 3.00, 3.12, 3.25, 3.37, 3.50,
        3.62, 3.74, 3.87, 3.99, 4.12, 4.24, 4.36, 4.49, 4.61, 4.74, 4.86, 4.98,
        5.11, 5.23, 5.36, 5.48, 5.60, 5.73, 5.85, 5.98, 6.10, 6.22, 6.35, 6.47, 6.60,
    ]),
    n=np.array([
        0.92, 0.56, 0.43, 0.35, 0.27, 0.22, 0.17, 0.16, 0.14, 0.13, 0.14, 0.21,
        0.29, 0.43, 0.62, 1.04, 1.31, 1.38, 1.45, 1.46, 1.47, 1.46, 1.48, 1.50,
        1.48, 1.48, 1.54, 1.53, 1.53, 1.49, 1.47, 1.43, 1.38, 1.35, 1.33, 1.33,
        1.32, 1.32, 1.30, 1.31, 1.30, 1.30, 1.30, 1.30, 1.33, 1.33, 1.34, 1.32, 1.28,
    ]),
    k=np.array([
        13.78, 11.21, 9.519, 8.145, 7.15, 6.35, 5.663, 5.083, 4.542, 4.103, 3.697, 3.272,
        2.863, 2.455, 2.081, 1.833, 1.849, 1.914, 1.948, 1.958, 1.952, 1.933, 1.895, 1.866,
        1.871, 1.883, 1.898, 1.893, 1.889, 1.878, 1.869, 1.847, 1.803, 1.749, 1.688, 1.631,
        1.577, 1.536, 1.497, 1.460, 1.427, 1.387, 1.350, 1.304, 1.277, 1.251, 1.226, 1.203, 1.188,
    ]),
)


# ================================================================
# Physical parameters
# ================================================================
@dataclass(frozen=True)
class RadiativeRateDiagnostics:
    d_external_au: float
    homogeneous_radiative_decay_au: float
    gamma1_au: float
    gamma1_over_homogeneous_radiative_rate: float
    homogeneous_host_consistent: bool


@dataclass(frozen=True)
class HybridSystemParams:
    c_au: float
    a_au: float
    R_au: float
    G: float
    eps_m: float
    d_au: float
    omega0_au: float
    gamma_au: float
    Gamma_au: float
    qd_radius_au: float = field(default=0.0, kw_only=True)
    eps_qd: float = field(default=1.0, kw_only=True)
    qd_dipole_convention: QDDipoleConvention = field(
        default='effective_external',
        kw_only=True,
    )
    material: MaterialDispersion = field(default_factory=lambda: DEFAULT_AU_MATERIAL)

    @property
    def axial_surface_gap_au(self) -> float:
        """Surface-to-surface gap for a QD on the ellipsoid long axis."""
        return float(self.R_au - self.c_au - self.qd_radius_au)

    @property
    def pure_dephasing_au(self) -> float:
        """Pure-dephasing contribution implied by Gamma2=gamma1/2+gamma_phi."""
        return float(self.Gamma_au - 0.5 * self.gamma_au)

    @property
    def qd_local_field_factor(self) -> float:
        """Electrostatic screening of a spherical QD embedded in the host.

        For ``bare_internal``, ``d_au`` is the unscreened interband transition
        matrix element.  Both the microscopic drive and the externally visible
        QD dipole acquire the electrostatic factor.  For
        ``effective_external``, ``d_au`` already contains this screening and
        the factor is one, preventing double counting.
        """
        if (
            not np.isreal(self.eps_m)
            or not np.isreal(self.eps_qd)
            or not np.isfinite(self.eps_m)
            or not np.isfinite(self.eps_qd)
            or self.eps_m <= 0.0
            or self.eps_qd <= 0.0
        ):
            raise ValueError(
                'eps_m and eps_qd must be finite, positive real background '
                'permittivities.'
            )
        if self.qd_dipole_convention == 'effective_external':
            return 1.0
        if self.qd_dipole_convention != 'bare_internal':
            raise ValueError(
                "qd_dipole_convention must be 'bare_internal' or "
                "'effective_external'."
            )
        return float(3.0 * self.eps_m / (self.eps_qd + 2.0 * self.eps_m))

    @property
    def qd_external_dipole_au(self) -> float:
        """Dipole coupled to/radiating into the macroscopic host."""

        return float(self.qd_local_field_factor * self.d_au)

    @property
    def homogeneous_radiative_decay_au(self) -> float:
        """Homogeneous-host reference radiative rate implied by d and omega0."""

        return homogeneous_radiative_decay_rate_au(
            self.qd_external_dipole_au,
            self.omega0_au,
            self.eps_m,
        )

    @property
    def radiative_rate_diagnostics(self) -> RadiativeRateDiagnostics:
        """Consistency of phenomenological gamma1 with a homogeneous host.

        This is a diagnostic, not an MNP Purcell calculation.  The present
        electrostatic density-matrix model does not contain vacuum noise and
        therefore cannot derive environment-modified spontaneous emission.
        """

        radiative_rate = self.homogeneous_radiative_decay_au
        if radiative_rate == 0.0:
            ratio = np.inf
            consistent = True
        else:
            ratio = float(self.gamma_au / radiative_rate)
            consistent = bool(ratio >= 1.0 - 1.0e-10)
        return RadiativeRateDiagnostics(
            d_external_au=self.qd_external_dipole_au,
            homogeneous_radiative_decay_au=radiative_rate,
            gamma1_au=float(self.gamma_au),
            gamma1_over_homogeneous_radiative_rate=ratio,
            homogeneous_host_consistent=consistent,
        )


@dataclass(frozen=True)
class GaussianPulse:
    E0_au: float
    omegaL_au: float
    tau_au: float
    tau_kind: Literal['sigma', 'fwhm_intensity'] = 'fwhm_intensity'

    def __post_init__(self) -> None:
        scalars = (self.E0_au, self.omegaL_au, self.tau_au)
        if not all(np.isfinite(value) for value in scalars):
            raise ValueError('Pulse amplitude, carrier frequency and duration must be finite.')
        if self.E0_au == 0.0:
            raise ValueError('Pulse amplitude E0_au must be non-zero because cross sections divide by fluence.')
        if self.omegaL_au <= 0.0:
            raise ValueError('Pulse carrier frequency omegaL_au must be positive.')
        if self.tau_au <= 0.0:
            raise ValueError('Pulse duration tau_au must be positive.')
        if self.tau_kind not in {'sigma', 'fwhm_intensity'}:
            raise ValueError(f'Unsupported tau_kind={self.tau_kind!r}')

    @property
    def sigma_t_au(self) -> float:
        if self.tau_kind == 'sigma':
            return float(self.tau_au)
        if self.tau_kind == 'fwhm_intensity':
            return float(self.tau_au) / (2.0 * np.sqrt(np.log(2.0)))
        raise ValueError(f'Unsupported tau_kind={self.tau_kind!r}')

    def envelope(self, t_au: float | np.ndarray) -> np.ndarray:
        t = np.asarray(t_au, dtype=float)
        sigma = self.sigma_t_au
        return np.exp(-0.5 * (t / sigma) ** 2)

    def field(self, t_au: float | np.ndarray) -> np.ndarray:
        t = np.asarray(t_au, dtype=float)
        return self.E0_au * self.envelope(t) * np.cos(self.omegaL_au * t)

    def field_dot(self, t_au: float | np.ndarray) -> np.ndarray:
        t = np.asarray(t_au, dtype=float)
        sigma = self.sigma_t_au
        env = self.envelope(t)
        return self.E0_au * env * (
            -(t / sigma**2) * np.cos(self.omegaL_au * t)
            - self.omegaL_au * np.sin(self.omegaL_au * t)
        )

    @staticmethod
    def _refractive_index(eps_m: float) -> float:
        if not np.isfinite(eps_m) or eps_m <= 0.0:
            raise ValueError('The real host permittivity eps_m must be finite and positive.')
        return float(np.sqrt(eps_m))

    def peak_intensity_w_cm2(
        self,
        cycle_averaged: bool = True,
        *,
        eps_m: float = 1.0,
    ) -> float:
        """Peak intensity in a nonmagnetic, lossless host with n=sqrt(eps_m)."""
        E0_si = float(field_au_to_si(self.E0_au))
        prefactor = 0.5 if cycle_averaged else 1.0
        n_m = self._refractive_index(eps_m)
        return prefactor * n_m * epsilon_0 * C_SI * E0_si**2 * 1e-4

    def fluence_j_cm2(self, *, eps_m: float = 1.0) -> float:
        """Exact integral of n*eps0*c*E(t)^2 for the real Gaussian carrier."""
        E0_si = float(field_au_to_si(self.E0_au))
        sigma_s = self.sigma_t_au * AU_TIME_S
        osc = np.exp(-(self.omegaL_au * self.sigma_t_au) ** 2)
        integral_E2 = 0.5 * np.sqrt(np.pi) * sigma_s * E0_si**2 * (1.0 + osc)
        n_m = self._refractive_index(eps_m)
        fluence_j_m2 = n_m * epsilon_0 * C_SI * integral_E2
        return fluence_j_m2 * 1e-4

    def positive_frequency_spectral_fraction(
        self,
        energy_window_eV: tuple[float, float],
    ) -> float:
        """Exact fraction of real-pulse spectral energy inside a positive window.

        The Fourier amplitude of ``envelope(t)*cos(omegaL*t)`` is the sum of
        two Gaussian lobes centered at +/-omegaL.  Their overlap is retained,
        so this remains valid even for few-cycle pulses.
        """
        if len(energy_window_eV) != 2 or not np.all(np.isfinite(energy_window_eV)):
            raise ValueError('energy_window_eV must contain two finite values.')
        e_min, e_max = (float(value) for value in energy_window_eV)
        if e_min < 0.0 or e_max <= e_min:
            raise ValueError('energy_window_eV must satisfy 0 <= min < max.')

        sigma = self.sigma_t_au
        omega_l = self.omegaL_au
        omega_min = float(eV_to_au(e_min))
        omega_max = float(eV_to_au(e_max))

        def integral_between(lower: float, upper: float) -> float:
            shifted_positive = erf(sigma * (upper - omega_l)) - erf(
                sigma * (lower - omega_l)
            )
            shifted_negative = erf(sigma * (upper + omega_l)) - erf(
                sigma * (lower + omega_l)
            )
            overlap = 2.0 * np.exp(-(sigma * omega_l) ** 2) * (
                erf(sigma * upper) - erf(sigma * lower)
            )
            return float(shifted_positive + shifted_negative + overlap)

        numerator = integral_between(omega_min, omega_max)
        denominator = 2.0 * (1.0 + np.exp(-(sigma * omega_l) ** 2))
        return float(np.clip(numerator / denominator, 0.0, 1.0))

    def spectral_leakage_fraction(
        self,
        energy_window_eV: tuple[float, float],
    ) -> float:
        """Positive-frequency pulse energy outside ``energy_window_eV``."""
        return float(1.0 - self.positive_frequency_spectral_fraction(energy_window_eV))


def sampled_positive_frequency_spectral_fraction(
    t_au: np.ndarray,
    signal: np.ndarray,
    energy_window_eV: tuple[float, float],
    *,
    highest_resolved_omega_au: float,
    samples_per_fastest_cycle: int = 20,
) -> float:
    """Estimate the positive-frequency energy fraction of a sampled signal.

    ``solve_ivp`` returns a nonuniform grid, so the signal is first linearly
    resampled at a rate fixed by the fastest carrier/modal frequency and then
    audited with a one-sided FFT.  This is a post-solve validity diagnostic,
    not a replacement for the exact analytic incident-Gaussian check above.
    """

    t = np.asarray(t_au, dtype=float)
    values = np.asarray(signal, dtype=float)
    if (
        t.ndim != 1
        or values.ndim != 1
        or t.size != values.size
        or t.size < 2
        or np.any(~np.isfinite(t))
        or np.any(~np.isfinite(values))
        or np.any(np.diff(t) <= 0.0)
    ):
        raise ValueError('Spectral-audit time and signal arrays must be finite, aligned and monotone.')
    if (
        not np.isfinite(highest_resolved_omega_au)
        or highest_resolved_omega_au <= 0.0
    ):
        raise ValueError('highest_resolved_omega_au must be finite and positive.')
    if samples_per_fastest_cycle < 8:
        raise ValueError('samples_per_fastest_cycle must be at least 8.')
    if len(energy_window_eV) != 2 or not np.all(np.isfinite(energy_window_eV)):
        raise ValueError('energy_window_eV must contain two finite values.')
    e_min, e_max = (float(value) for value in energy_window_eV)
    if e_min < 0.0 or e_max <= e_min:
        raise ValueError('energy_window_eV must satisfy 0 <= min < max.')

    duration = float(t[-1] - t[0])
    target_dt = float(
        2.0 * np.pi
        / (samples_per_fastest_cycle * highest_resolved_omega_au)
    )
    n_uniform = max(2048, int(np.ceil(duration / target_dt)) + 1)
    if n_uniform > 1_000_001:
        raise RuntimeError(
            'The response-spectrum audit would require more than one million '
            'uniform samples; shorten the time window or audit the trace offline.'
        )
    uniform_t = np.linspace(t[0], t[-1], n_uniform)
    uniform_signal = np.interp(uniform_t, t, values)
    spectrum = np.fft.rfft(uniform_signal)
    omega = 2.0 * np.pi * np.fft.rfftfreq(
        n_uniform,
        d=float(uniform_t[1] - uniform_t[0]),
    )
    # Parseval weights for a one-sided transform.  Keeping the DC bin in the
    # denominator makes a low-frequency/static component count as uncovered by
    # an optical fit window whose lower bound is nonzero.
    spectral_energy = np.abs(spectrum) ** 2
    weights = np.full(spectral_energy.shape, 2.0)
    weights[0] = 1.0
    if n_uniform % 2 == 0:
        weights[-1] = 1.0
    spectral_energy *= weights
    total = float(np.sum(spectral_energy))
    if not np.isfinite(total) or total <= np.finfo(float).tiny:
        raise RuntimeError('The response-spectrum audit found no finite positive-frequency energy.')
    energies_eV = np.asarray(au_to_eV(omega), dtype=float)
    inside = (energies_eV >= e_min) & (energies_eV <= e_max)
    return float(np.clip(np.sum(spectral_energy[inside]) / total, 0.0, 1.0))


def response_tail_ratio(
    mu_total_au: np.ndarray,
    t_au: np.ndarray | None = None,
    mu_p_au: np.ndarray | None = None,
    mu_d_au: np.ndarray | None = None,
    *,
    tail_fraction: float = 0.05,
) -> float:
    """Return the residual dipole amplitude near the final integration time.

    With a time grid, the metric is the largest time-weighted RMS/peak ratio
    among the total, MNP and QD dipoles.  Testing the components separately is
    essential: a slowly decaying MNP and QD response can cancel in the total
    dipole while the physical state is still far from its post-pulse limit.

    Omitting ``t_au`` retains the legacy sample-weighted total-dipole metric.
    """

    values = np.asarray(mu_total_au, dtype=float)
    if t_au is None:
        if mu_p_au is not None or mu_d_au is not None:
            raise ValueError('t_au is required when component dipoles are supplied.')
        if values.size == 0:
            return np.nan
        peak = float(np.max(np.abs(values)))
        if peak == 0.0:
            return 0.0
        n_tail = min(max(8, int(np.ceil(tail_fraction * values.size))), values.size)
        return float(np.sqrt(np.mean(values[-n_tail:] ** 2)) / peak)

    times = np.asarray(t_au, dtype=float)
    if not np.isfinite(tail_fraction) or not 0.0 < tail_fraction <= 1.0:
        raise ValueError('tail_fraction must be finite and lie in (0, 1].')
    if times.ndim != 1 or times.size < 2:
        raise ValueError('t_au must be a one-dimensional array with at least two samples.')
    if np.any(~np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
        raise ValueError('t_au must contain finite, strictly increasing values.')

    responses = [values]
    responses.extend(
        np.asarray(component, dtype=float)
        for component in (mu_p_au, mu_d_au)
        if component is not None
    )
    if any(response.ndim != 1 or response.size != times.size for response in responses):
        raise ValueError('Each dipole response must be one-dimensional and aligned with t_au.')
    if any(np.any(~np.isfinite(response)) for response in responses):
        raise ValueError('Dipole responses must contain only finite values.')

    window_start = float(times[-1] - tail_fraction * (times[-1] - times[0]))
    start_index = int(np.searchsorted(times, window_start, side='left'))
    ratios: list[float] = []
    for response in responses:
        peak = float(np.max(np.abs(response)))
        if peak == 0.0:
            ratios.append(0.0)
            continue

        if times[start_index] == window_start:
            tail_times = times[start_index:]
            tail_values = response[start_index:]
        else:
            left = start_index - 1
            weight = (window_start - times[left]) / (times[start_index] - times[left])
            value_at_start = response[left] + weight * (response[start_index] - response[left])
            tail_times = np.concatenate(([window_start], times[start_index:]))
            tail_values = np.concatenate(([value_at_start], response[start_index:]))

        duration = float(tail_times[-1] - tail_times[0])
        mean_square = float(np.trapezoid(tail_values**2, tail_times) / duration)
        ratios.append(float(np.sqrt(max(mean_square, 0.0)) / peak))

    return max(ratios)


# ================================================================
# Fit and solution containers
# ================================================================
@dataclass(frozen=True)
class RationalLorentzFit:
    alpha_inf: float
    strengths_au2: np.ndarray
    omega_modes_au: np.ndarray
    gamma_modes_au: np.ndarray
    energies_used_eV: np.ndarray
    alpha_used: np.ndarray
    rms_alpha: float
    rms_inv_alpha: float
    cost: float
    normalized_rms_alpha: float = 0.0
    normalized_rms_inv_alpha: float = 0.0
    max_normalized_alpha_error: float = 0.0
    min_imag_alpha_fit_window: float = 0.0
    passivity_grid_points: int = 0
    passive_on_fit_window: bool = True
    passive_for_all_positive_frequencies: bool = True

    def __post_init__(self) -> None:
        alpha_inf_scalar = np.asarray(self.alpha_inf)
        if alpha_inf_scalar.ndim != 0:
            raise ValueError('alpha_inf must be a finite real scalar.')
        alpha_inf_complex = complex(alpha_inf_scalar)
        if not np.isfinite(alpha_inf_complex) or alpha_inf_complex.imag != 0.0:
            raise ValueError('alpha_inf must be finite and real.')
        strengths = np.array(self.strengths_au2, dtype=float, copy=True)
        frequencies = np.array(self.omega_modes_au, dtype=float, copy=True)
        dampings = np.array(self.gamma_modes_au, dtype=float, copy=True)
        energies = np.array(self.energies_used_eV, dtype=float, copy=True)
        alpha_used = np.array(self.alpha_used, dtype=complex, copy=True)
        if not (
            strengths.ndim
            == frequencies.ndim
            == dampings.ndim
            == energies.ndim
            == alpha_used.ndim
            == 1
        ):
            raise ValueError('Rational-fit arrays must be one-dimensional.')
        if not (strengths.size == frequencies.size == dampings.size):
            raise ValueError('Every Lorentz mode needs one strength, frequency and damping.')
        if energies.size != alpha_used.size:
            raise ValueError('energies_used_eV and alpha_used must have equal lengths.')
        if (
            np.any(~np.isfinite(strengths))
            or np.any(~np.isfinite(frequencies))
            or np.any(~np.isfinite(dampings))
            or np.any(~np.isfinite(energies))
            or np.any(~np.isfinite(alpha_used))
        ):
            raise ValueError('Rational-fit arrays must contain only finite values.')
        if np.any(strengths < 0.0):
            raise ValueError('Passive Lorentz strengths must satisfy f_k >= 0.')
        if np.any(frequencies <= 0.0) or np.any(dampings <= 0.0):
            raise ValueError('Lorentz frequencies and damping rates must be positive.')
        for values in (strengths, frequencies, dampings, energies, alpha_used):
            values.setflags(write=False)
        object.__setattr__(self, 'alpha_inf', float(alpha_inf_complex.real))
        object.__setattr__(self, 'strengths_au2', strengths)
        object.__setattr__(self, 'omega_modes_au', frequencies)
        object.__setattr__(self, 'gamma_modes_au', dampings)
        object.__setattr__(self, 'energies_used_eV', energies)
        object.__setattr__(self, 'alpha_used', alpha_used)

    @property
    def nonnegative_imaginary_part_all_positive_frequencies(self) -> bool:
        """Whether the fit has nonnegative harmonic loss for every omega>0.

        The historical ``passive_for_all_positive_frequencies`` field is kept
        for artifact/API compatibility.  The explicit name avoids claiming
        that a finite-band excess-polarizability realization proves every
        possible finite-time/global passivity property.
        """

        return bool(self.passive_for_all_positive_frequencies)


@dataclass(frozen=True)
class DipoleCrossSections:
    """Strict-QS dipole estimates plus schema-compatible historical names.

    For an *undressed electrostatic* polarizability,
    ``quasistatic_work_loss_cm2`` stores the leading-order
    ``k Im(alpha)/eps0`` work-loss estimate,
    ``rayleigh_scattering_estimate_cm2`` is a separate radiation estimate, and
    ``optical_theorem_residual_cm2`` is only their formal difference.  It is
    not a self-consistent material-absorption partition unless a radiatively
    dressed polarizability is supplied.
    """

    quasistatic_work_loss_cm2: np.ndarray
    rayleigh_scattering_estimate_cm2: np.ndarray
    optical_theorem_residual_cm2: np.ndarray

    @property
    def extinction_cm2(self) -> np.ndarray:
        """Schema-compatibility alias for ``quasistatic_work_loss_cm2``."""

        return self.quasistatic_work_loss_cm2

    @property
    def scattering_cm2(self) -> np.ndarray:
        """Schema-compatibility alias for the separate Rayleigh estimate."""

        return self.rayleigh_scattering_estimate_cm2

    @property
    def absorption_cm2(self) -> np.ndarray:
        """Legacy name for the residual; not strict-QS material absorption."""

        return self.optical_theorem_residual_cm2


@dataclass(frozen=True)
class LinearStabilityDiagnostics:
    """Poles of the full field-free Jacobian at the QD ground state."""

    poles_au: np.ndarray
    spectral_abscissa_au: float
    tolerance_au: float
    stable: bool


@dataclass(frozen=True)
class DipoleApplicabilityDiagnostics:
    """Dimensionless diagnostics, not hard validity boundaries.

    A quasistatic particle response needs k_m*c << 1, while the non-retarded
    coupling J proportional to 1/R**3 also needs k_m*R << 1.  Replacing the MNP
    by one dipole and the field across a finite spherical QD by its centre value
    additionally needs c/R << 1 and r_QD/R << 1.  The reported 0.3 guide is a
    warning level, not a theorem or a substitute for a convergence study.
    """

    energy_eV: float
    medium_size_parameter_kc: float
    medium_separation_parameter_kR: float
    mnp_size_to_separation_ratio: float
    qd_size_to_separation_ratio: float
    guide_threshold: float
    particle_quasistatic_guide_satisfied: bool
    near_field_coupling_guide_satisfied: bool
    quasistatic_guide_satisfied: bool
    mnp_point_dipole_guide_satisfied: bool
    qd_point_dipole_guide_satisfied: bool
    point_dipole_guide_satisfied: bool


def quasistatic_dipole_cross_section_estimates_cm2(
    alpha_eff_au: float | complex | np.ndarray,
    omega_au: float | np.ndarray,
    eps_m: float,
) -> DipoleCrossSections:
    """Return strict-QS work-loss and separate Rayleigh-radiation estimates.

    ``alpha_eff`` is the undressed electrostatic response used by the native
    model.  Consequently ``k Im(alpha_eff)/eps0`` must not be interpreted
    simultaneously as an exact extinction (absorption+scattering) while the
    Rayleigh scattering term is subtracted from it.  The returned historical
    fields are retained for schema compatibility; use the explicit properties
    on ``DipoleCrossSections`` for new analysis.
    """
    if not np.isfinite(eps_m) or eps_m <= 0.0:
        raise ValueError('The real host permittivity eps_m must be finite and positive.')
    alpha_au = np.asarray(alpha_eff_au, dtype=complex)
    omega = np.asarray(omega_au, dtype=float)
    if np.any(~np.isfinite(alpha_au)) or np.any(~np.isfinite(omega)) or np.any(omega < 0.0):
        raise ValueError('Polarizability and non-negative angular frequencies must be finite.')

    alpha_si = alpha_au * (AU_DIPOLE_C_M / AU_FIELD_V_M)
    omega_si = omega / AU_TIME_S
    k_si = np.sqrt(eps_m) * omega_si / C_SI
    work_loss_m2 = (k_si / epsilon_0) * alpha_si.imag
    rayleigh_estimate_m2 = (
        k_si**4 * np.abs(alpha_si) ** 2 / (6.0 * np.pi * epsilon_0**2)
    )
    optical_theorem_residual_m2 = work_loss_m2 - rayleigh_estimate_m2
    return DipoleCrossSections(
        quasistatic_work_loss_cm2=np.asarray(work_loss_m2 * 1e4),
        rayleigh_scattering_estimate_cm2=np.asarray(rayleigh_estimate_m2 * 1e4),
        optical_theorem_residual_cm2=np.asarray(
            optical_theorem_residual_m2 * 1e4
        ),
    )


def dipole_cross_sections_cm2(
    alpha_eff_au: float | complex | np.ndarray,
    omega_au: float | np.ndarray,
    eps_m: float,
) -> DipoleCrossSections:
    """Schema-compatible name for QS dipole cross-section estimates."""

    return quasistatic_dipole_cross_section_estimates_cm2(
        alpha_eff_au,
        omega_au,
        eps_m,
    )


@dataclass(frozen=True)
class HybridSolveDiagnostics:
    solver_success: bool
    solver_status: int
    solver_message: str
    n_steps: int
    nfev: int
    njev: int | None
    nlu: int | None
    t_final_reached: bool
    state_is_finite: bool
    min_step_au: float
    max_step_au: float
    max_step_limit_au: float
    integration_frequency_ceiling_au: float
    boundary_envelope_fraction: float
    W_min: float
    W_max: float
    excited_population_min: float
    excited_population_max: float
    max_bloch_radius: float
    min_density_eigenvalue: float
    pulse_spectral_fraction_in_fit_window: float = 1.0
    pulse_spectral_leakage: float = 0.0
    mnp_drive_spectral_fraction_in_fit_window: float = 1.0
    mnp_drive_spectral_leakage: float = 0.0
    mnp_dipole_spectral_fraction_in_fit_window: float = 1.0
    mnp_dipole_spectral_leakage: float = 0.0
    work_passivity_checked: bool = False
    work_passivity_tolerance_au: float = 0.0
    work_nonnegative_within_tolerance: bool = True
    incident_peak_rabi_frequency_au: float = 0.0
    observed_peak_rabi_frequency_au: float = 0.0
    rabi_step_refinement_count: int = 0


@dataclass(frozen=True)
class HybridSolveResult:
    t_au: np.ndarray
    y: np.ndarray
    mu_p_au: np.ndarray
    mu_d_au: np.ndarray
    mu_total_au: np.ndarray
    mu_dot_total_au: np.ndarray
    sigma_energy_transfer_cm2: float
    work_from_incident_field_j: float
    fluence_j_cm2: float
    peak_intensity_w_cm2: float
    solve_ivp_result: object
    diagnostics: HybridSolveDiagnostics

    @property
    def sigma_abs_cm2(self) -> float:
        """Compatibility alias; historically this was mislabeled absorption."""
        warnings.warn(
            'sigma_abs_cm2 was mislabeled; use sigma_energy_transfer_cm2.',
            DeprecationWarning,
            stacklevel=2,
        )
        return self.sigma_energy_transfer_cm2

    @property
    def absorbed_energy_j(self) -> float:
        """Compatibility alias for work_from_incident_field_j."""
        warnings.warn(
            'absorbed_energy_j was mislabeled; use work_from_incident_field_j.',
            DeprecationWarning,
            stacklevel=2,
        )
        return self.work_from_incident_field_j

    @property
    def max_bloch_radius(self) -> float:
        return self.diagnostics.max_bloch_radius

    @property
    def min_density_eigenvalue(self) -> float:
        return self.diagnostics.min_density_eigenvalue


# ================================================================
# Main model
# ================================================================
class HybridQDPlasmonModel:
    """
    Полная перепись модели с заменой полинома для 1/alpha(omega)
    на устойчивую рациональную аппроксимацию alpha(omega) в виде суммы
    лоренцевских мод.

    Это устраняет главный источник ошибки исходного кода: мнимая часть
    1/alpha(omega) больше не вынуждена быть почти линейным нечётным
    полиномом по малой частоте omega в атомных единицах.
    """

    def __init__(
        self,
        params: HybridSystemParams,
        *,
        orientation: DipoleOrientation = 'long',
        n_modes: int = 9,
        fit_window_eV: tuple[float, float] = (0.8, 3.0),
        weight_center_eV: float | None = None,
        weight_sigma_eV: float | None = None,
        alpha_objective_weight: float = 1.0,
        inv_alpha_objective_weight: float = 1.2,
        max_fit_normalized_rms: float | None = 0.025,
        max_fit_pointwise_relative_error: float | None = 0.05,
        radiative_consistency_policy: Literal['raise', 'warn', 'ignore'] = 'warn',
        seed: int = 12345,
        verbose: bool = True,
    ) -> None:
        if n_modes < 1:
            raise ValueError('n_modes must be >= 1')
        expected_g = orientation_factor(orientation)
        if len(fit_window_eV) != 2 or not np.all(np.isfinite(fit_window_eV)):
            raise ValueError('fit_window_eV must contain two finite values.')
        if fit_window_eV[0] <= 0.0 or fit_window_eV[1] <= fit_window_eV[0]:
            raise ValueError('fit_window_eV must satisfy 0 < min < max.')
        material_min = float(params.material.energy_eV[0])
        material_max = float(params.material.energy_eV[-1])
        if fit_window_eV[0] < material_min or fit_window_eV[1] > material_max:
            raise ValueError(
                'fit_window_eV must lie inside the tabulated material-data interval '
                f'[{material_min:g}, {material_max:g}] eV.'
            )
        if (weight_center_eV is None) != (weight_sigma_eV is None):
            raise ValueError(
                'weight_center_eV and weight_sigma_eV must be specified together.'
            )
        if weight_center_eV is not None and not np.isfinite(weight_center_eV):
            raise ValueError('weight_center_eV must be finite when specified.')
        if weight_sigma_eV is not None and (
            not np.isfinite(weight_sigma_eV) or weight_sigma_eV <= 0.0
        ):
            raise ValueError('weight_sigma_eV must be finite and positive when specified.')
        if alpha_objective_weight < 0.0 or inv_alpha_objective_weight < 0.0:
            raise ValueError('Fit objective weights must be non-negative.')
        if alpha_objective_weight == 0.0 and inv_alpha_objective_weight == 0.0:
            raise ValueError('At least one fit objective weight must be positive.')
        if max_fit_normalized_rms is not None and (
            not np.isfinite(max_fit_normalized_rms)
            or max_fit_normalized_rms <= 0.0
        ):
            raise ValueError('max_fit_normalized_rms must be positive or None.')
        if max_fit_pointwise_relative_error is not None and (
            not np.isfinite(max_fit_pointwise_relative_error)
            or max_fit_pointwise_relative_error <= 0.0
        ):
            raise ValueError(
                'max_fit_pointwise_relative_error must be positive or None.'
            )
        if radiative_consistency_policy not in {'raise', 'warn', 'ignore'}:
            raise ValueError(
                "radiative_consistency_policy must be 'raise', 'warn' or 'ignore'."
            )

        self.params = params
        self.orientation = orientation
        self.n_modes = int(n_modes)
        self.fit_window_eV = fit_window_eV
        self.weight_center_eV = weight_center_eV
        self.weight_sigma_eV = weight_sigma_eV
        self.alpha_objective_weight = float(alpha_objective_weight)
        self.inv_alpha_objective_weight = float(inv_alpha_objective_weight)
        self.max_fit_normalized_rms = (
            None
            if max_fit_normalized_rms is None
            else float(max_fit_normalized_rms)
        )
        self.max_fit_pointwise_relative_error = (
            None
            if max_fit_pointwise_relative_error is None
            else float(max_fit_pointwise_relative_error)
        )
        self.radiative_consistency_policy = radiative_consistency_policy
        self.seed = int(seed)
        self.verbose = bool(verbose)
        self._validate_physical_parameters()
        self.radiative_rate_diagnostics = params.radiative_rate_diagnostics
        if not self.radiative_rate_diagnostics.homogeneous_host_consistent:
            message = (
                'The supplied d and phenomenological gamma1 are inconsistent '
                'with an isolated emitter in the assumed homogeneous host: '
                f'gamma1/gamma_rad='
                f'{self.radiative_rate_diagnostics.gamma1_over_homogeneous_radiative_rate:.6g}. '
                'The electrostatic MNP model does not calculate spontaneous '
                'Purcell/nonradiative decay. Quantitative population dynamics '
                'therefore requires a sourced dipole convention and decay rate.'
            )
            if radiative_consistency_policy == 'raise':
                raise ValueError(message)
            if radiative_consistency_policy == 'warn':
                warnings.warn(message, RuntimeWarning, stacklevel=2)
        if not np.isclose(self.params.G, expected_g, rtol=0.0, atol=1e-12):
            raise ValueError(
                f'Inconsistent quasistatic orientation: orientation={orientation!r} '
                f'requires G={expected_g:g}, got G={self.params.G:g}.'
            )

        self.L_long, self.L_trans = self._depolarization_factors()
        self.L = self.L_long if orientation == 'long' else self.L_trans
        self.physical_alpha_infinity = float(
            (MATERIAL_HIGH_FREQUENCY_EPSILON - self.params.eps_m)
            / (
                self.params.eps_m
                + self.L
                * (MATERIAL_HIGH_FREQUENCY_EPSILON - self.params.eps_m)
            )
        )
        self.C = self.params.eps_m * self.params.a_au**2 * self.params.c_au / 3.0
        self.J = self.params.G / (self.params.eps_m * self.params.R_au**3)

        self.energy_au = eV_to_au(self.params.material.energy_eV)
        self.alpha_tab = self._alpha_dimless(self.L)
        self.inv_alpha_tab = 1.0 / self.alpha_tab
        self.fit = self._fit_rational_alpha()
        self.linear_stability = self.assert_linearized_ground_state_stable()
        self.applicability = self.dipole_applicability_diagnostics(
            energy_eV=float(self.fit_window_eV[1])
        )

        if self.verbose:
            self.print_fit_summary()

    # ------------------------------------------------------------
    # Geometry and tabulated optical response
    # ------------------------------------------------------------
    def _validate_physical_parameters(self) -> None:
        p = self.params
        scalar_values = {
            'c_au': p.c_au,
            'a_au': p.a_au,
            'R_au': p.R_au,
            'qd_radius_au': p.qd_radius_au,
            'G': p.G,
            'eps_m': p.eps_m,
            'eps_qd': p.eps_qd,
            'd_au': p.d_au,
            'omega0_au': p.omega0_au,
            'gamma_au': p.gamma_au,
            'Gamma_au': p.Gamma_au,
        }
        nonfinite = [name for name, value in scalar_values.items() if not np.isfinite(value)]
        if nonfinite:
            raise ValueError(f'Physical parameters must be finite; invalid: {", ".join(nonfinite)}.')
        if p.a_au <= 0.0 or p.c_au <= 0.0 or p.R_au <= 0.0:
            raise ValueError('MNP semiaxes and center-to-center distance R must be positive.')
        if p.c_au < p.a_au:
            raise ValueError('This implementation supports prolate/spherical MNPs and requires c >= a.')
        if p.qd_radius_au < 0.0:
            raise ValueError('QD radius must be non-negative; zero denotes the point-QD limit.')
        if p.axial_surface_gap_au <= 0.0:
            gap_nm = float(au_to_nm(p.axial_surface_gap_au))
            raise ValueError(
                'Non-positive QD-MNP surface gap: require R > c + qd_radius '
                f'(current gap={gap_nm:.6g} nm). The particles overlap or touch.'
            )
        if not np.isreal(p.eps_m) or p.eps_m <= 0.0:
            raise ValueError('The real host permittivity eps_m must be positive.')
        if not np.isreal(p.eps_qd) or p.eps_qd <= 0.0:
            raise ValueError('The real QD background permittivity eps_qd must be positive.')
        if p.qd_dipole_convention not in {'bare_internal', 'effective_external'}:
            raise ValueError(
                "qd_dipole_convention must be 'bare_internal' or "
                "'effective_external'."
            )
        if not any(
            np.isclose(p.G, value, rtol=0.0, atol=1e-12)
            for value in ORIENTATION_FACTORS.values()
        ):
            raise ValueError('Quasistatic dipole factor G must be exactly 2 (long) or -1 (trans).')
        if p.d_au < 0.0 or p.omega0_au <= 0.0:
            raise ValueError('QD transition-dipole magnitude must be non-negative and omega0 positive.')
        if p.gamma_au < 0.0 or p.Gamma_au < 0.0:
            raise ValueError('Relaxation rates gamma1 and Gamma2 must be non-negative.')
        if p.Gamma_au < 0.5 * p.gamma_au:
            raise ValueError(
                'Unphysical coherence decay: Gamma2 must satisfy Gamma2 >= gamma1/2. '
                'Gamma_au is the total coherence-decay rate, not pure dephasing.'
            )

    def _depolarization_factors(self) -> tuple[float, float]:
        c_au = self.params.c_au
        a_au = self.params.a_au
        eccentricity_squared = float(1.0 - (a_au / c_au) ** 2)
        if eccentricity_squared <= 0.0:
            return 1.0 / 3.0, 1.0 / 3.0
        if eccentricity_squared < 1.0e-3:
            # Stable expansion of
            # (1-e**2)*(atanh(e)-e)/e**3 near the spherical limit.  The direct
            # expression suffers catastrophic cancellation when e is tiny.
            e2 = eccentricity_squared
            L_long = (
                1.0 / 3.0
                - 2.0 * e2 / 15.0
                - 2.0 * e2**2 / 35.0
                - 2.0 * e2**3 / 63.0
                - 2.0 * e2**4 / 99.0
            )
        else:
            eccentricity = float(np.sqrt(eccentricity_squared))
            L_long = (
                (1.0 - eccentricity_squared)
                * (np.arctanh(eccentricity) - eccentricity)
                / eccentricity**3
            )
        L_trans = 0.5 * (1.0 - L_long)
        if not (0.0 < L_long < 1.0 and 0.0 < L_trans < 1.0):
            raise RuntimeError('Invalid ellipsoid depolarization factors.')
        return float(L_long), float(L_trans)

    def dipole_applicability_diagnostics(
        self,
        *,
        energy_eV: float,
        guide_threshold: float = 0.3,
    ) -> DipoleApplicabilityDiagnostics:
        """Return scale ratios controlling the selected approximation.

        This intentionally does not reject a calculation: the user selected a
        quasistatic point-dipole model.  It records when its output should be
        interpreted qualitatively rather than as a quantitatively converged
        electrodynamic prediction.
        """
        if not np.isfinite(energy_eV) or energy_eV <= 0.0:
            raise ValueError('Applicability energy_eV must be finite and positive.')
        if not np.isfinite(guide_threshold) or guide_threshold <= 0.0:
            raise ValueError('guide_threshold must be finite and positive.')
        omega_au = float(eV_to_au(energy_eV))
        kc = float(
            np.sqrt(self.params.eps_m)
            * omega_au
            * self.params.c_au
            / AU_SPEED_OF_LIGHT
        )
        c_over_r = float(self.params.c_au / self.params.R_au)
        qd_over_r = float(self.params.qd_radius_au / self.params.R_au)
        kR = float(
            np.sqrt(self.params.eps_m)
            * omega_au
            * self.params.R_au
            / AU_SPEED_OF_LIGHT
        )
        particle_quasistatic = bool(kc <= guide_threshold)
        near_field_coupling = bool(kR <= guide_threshold)
        return DipoleApplicabilityDiagnostics(
            energy_eV=float(energy_eV),
            medium_size_parameter_kc=kc,
            medium_separation_parameter_kR=kR,
            mnp_size_to_separation_ratio=c_over_r,
            qd_size_to_separation_ratio=qd_over_r,
            guide_threshold=float(guide_threshold),
            particle_quasistatic_guide_satisfied=particle_quasistatic,
            near_field_coupling_guide_satisfied=near_field_coupling,
            quasistatic_guide_satisfied=bool(
                particle_quasistatic and near_field_coupling
            ),
            mnp_point_dipole_guide_satisfied=bool(c_over_r <= guide_threshold),
            qd_point_dipole_guide_satisfied=bool(qd_over_r <= guide_threshold),
            point_dipole_guide_satisfied=bool(
                max(c_over_r, qd_over_r) <= guide_threshold
            ),
        )

    def _alpha_dimless(self, L: float) -> np.ndarray:
        eps = self.params.material.epsilon
        eps_m = self.params.eps_m
        return (eps - eps_m) / (eps_m + L * (eps - eps_m))

    def _alpha_dimless_at(
        self,
        energies_eV: float | np.ndarray,
        L: float | None = None,
    ) -> np.ndarray:
        """Continuous quasistatic target based on interpolated material data."""

        eps = self.params.material.epsilon_at(energies_eV)
        eps_m = self.params.eps_m
        depolarization = self.L if L is None else float(L)
        return np.asarray(
            (eps - eps_m) / (eps_m + depolarization * (eps - eps_m)),
            dtype=complex,
        )

    def _fit_weights(self, energies_eV: np.ndarray) -> np.ndarray:
        if self.weight_center_eV is None or self.weight_sigma_eV is None:
            return np.ones_like(energies_eV, dtype=float)
        x = (energies_eV - self.weight_center_eV) / self.weight_sigma_eV
        return np.exp(-0.5 * x**2)

    # ------------------------------------------------------------
    # Stable rational fit for alpha(omega)
    # ------------------------------------------------------------
    def _alpha_model_from_params(
        self,
        omega_au: np.ndarray,
        u: np.ndarray,
        *,
        n_modes: int | None = None,
    ) -> np.ndarray:
        n = self.n_modes if n_modes is None else int(n_modes)
        if np.asarray(u).size != 1 + 3 * n:
            raise ValueError(f'Expected {1 + 3 * n} fit parameters for {n} modes.')
        alpha_inf = u[0]
        strengths = u[1 : 1 + n]
        omega_modes = np.exp(u[1 + n : 1 + 2 * n])
        gamma_modes = np.exp(u[1 + 2 * n : 1 + 3 * n])

        alpha = np.full_like(omega_au, fill_value=alpha_inf, dtype=complex)
        for f_k, w_k, g_k in zip(strengths, omega_modes, gamma_modes):
            denom = (w_k**2 - omega_au**2) - 1j * g_k * omega_au
            alpha += f_k / denom
        return alpha

    def _fit_rational_alpha(self) -> RationalLorentzFit:
        e_min, e_max = self.fit_window_eV
        mask = (self.params.material.energy_eV >= e_min) & (self.params.material.energy_eV <= e_max)
        tabulated_point_count = int(np.count_nonzero(mask))
        # The dense interpolation below prevents between-node resonances but
        # does not create independent material information.  Count
        # identifiability from the original complex n,k samples: N passive
        # Lorentz terms have 3N real modal parameters and the implementation
        # carries one (physically fixed) alpha_inf coordinate.  Each complex
        # sample supplies two real constraints.  Require four further real
        # constraints as a small overdetermination margin rather than the old
        # unrelated 2N+2-node heuristic, which incorrectly rejected the
        # canonical nine-mode fit despite 36 real data values for 28 fit
        # coordinates.
        fit_coordinate_count = 1 + 3 * self.n_modes
        minimum_real_constraint_count = fit_coordinate_count + 4
        minimum_tabulated_point_count = max(
            5,
            int(np.ceil(0.5 * minimum_real_constraint_count)),
        )
        if tabulated_point_count < minimum_tabulated_point_count:
            raise ValueError(
                'Too few independent tabulated points inside fit_window_eV '
                'for the requested n_modes: '
                f'{tabulated_point_count} complex points provide '
                f'{2 * tabulated_point_count} real constraints, but at least '
                f'{minimum_real_constraint_count} are required for '
                f'{fit_coordinate_count} fit coordinates plus the '
                'overdetermination margin.'
            )

        # Fitting and quality gates must constrain the response *between* the
        # sparse Johnson--Christy samples.  A sparse-node objective can accept
        # narrow artificial Lorentz resonances which are invisible at every
        # measurement node but dominate a continuous time/frequency run.
        dense_energies = np.linspace(e_min, e_max, max(1025, 256 * self.n_modes))
        table_energies = self.params.material.energy_eV[mask]
        energies = np.unique(
            np.concatenate(([e_min, e_max], dense_energies, table_energies))
        )
        omega = np.asarray(eV_to_au(energies), dtype=float)
        alpha_true = self._alpha_dimless_at(energies)
        inv_true = 1.0 / alpha_true
        weights = self._fit_weights(energies)

        scale_alpha_re = max(np.max(np.abs(alpha_true.real)), 1e-12)
        scale_alpha_im = max(np.max(np.abs(alpha_true.imag)), 1e-12)
        scale_inv_re = max(np.max(np.abs(inv_true.real)), 1e-12)
        scale_inv_im = max(np.max(np.abs(inv_true.imag)), 1e-12)

        omega_peak = float(omega[np.argmax(np.abs(alpha_true.imag))])
        omega_min = float(np.min(omega))
        omega_max = float(np.max(omega))
        omega_span = max(omega_max - omega_min, 1e-4)
        alpha_med = float(np.median(alpha_true.real))
        alpha_scale = max(np.max(np.abs(alpha_true)), 1e-3)
        strength_scale = max(alpha_scale * omega_peak**2, 1e-4)

        passivity_omega = np.asarray(
            eV_to_au(np.linspace(e_min, e_max, max(1024, 256 * self.n_modes))),
            dtype=float,
        )
        passivity_scale = max(float(np.max(np.abs(alpha_true.imag))), 1.0)
        passivity_tol = 1e-9 * passivity_scale

        # A finite-band optimizer must not invent an arbitrary constant that
        # survives at |omega| -> infinity.  For a metal, epsilon(omega) -> 1;
        # inserting that limit in the same ellipsoid formula fixes the direct
        # term.  In vacuum this is exactly zero.  A tiny numerical interval is
        # used because scipy requires strict lower < upper bounds; accepted
        # candidates are projected back to the exact asymptote below.
        physical_alpha_inf = self.physical_alpha_infinity
        alpha_inf_tolerance = 1.0e-12 * max(1.0, abs(physical_alpha_inf))
        alpha_inf_lo = physical_alpha_inf - alpha_inf_tolerance
        alpha_inf_hi = physical_alpha_inf + alpha_inf_tolerance
        # With the exp(-i omega t) convention every independent Lorentz mode
        # is passive for all omega>0 iff f_k>=0, omega_k>0 and gamma_k>0.
        # Negative residues may make a sampled sum look passive through pole
        # cancellation, but they cannot represent independent passive plasmon
        # oscillators in the time-domain realization.
        omega_lo_scalar = max(0.35 * omega_min, 1e-5)
        omega_hi_scalar = float(eV_to_au(self.params.material.energy_eV[-1]))
        gamma_lo_scalar = max(1e-4, 0.01 * omega_min)
        gamma_hi_scalar = max(2.0 * omega_max, 0.20)

        def parameter_bounds(n_modes: int) -> tuple[np.ndarray, np.ndarray]:
            lower = np.concatenate([
                [alpha_inf_lo],
                np.zeros(n_modes),
                np.full(n_modes, np.log(omega_lo_scalar)),
                np.full(n_modes, np.log(gamma_lo_scalar)),
            ])
            upper = np.concatenate([
                [alpha_inf_hi],
                np.full(n_modes, 100.0 * strength_scale),
                np.full(n_modes, np.log(omega_hi_scalar)),
                np.full(n_modes, np.log(gamma_hi_scalar)),
            ])
            return lower, upper

        evaluation_cache: dict[
            int,
            tuple[np.ndarray, np.ndarray, np.ndarray],
        ] = {}

        def residual_and_jacobian(
            u: np.ndarray,
            n_modes: int,
        ) -> tuple[np.ndarray, np.ndarray]:
            """Return the fit residual and its exact parameter Jacobian."""

            u = np.asarray(u, dtype=float)
            cached = evaluation_cache.get(n_modes)
            if cached is not None and np.array_equal(cached[0], u):
                return cached[1], cached[2]

            alpha_fit = self._alpha_model_from_params(omega, u, n_modes=n_modes)
            inv_fit = 1.0 / alpha_fit

            r_alpha = alpha_fit - alpha_true
            r_inv = inv_fit - inv_true

            alpha_inf_index = 0
            strength_start = 1
            omega_start = 1 + n_modes
            gamma_start = 1 + 2 * n_modes
            strengths = u[strength_start:omega_start]
            omega_modes = np.exp(u[omega_start:gamma_start])
            gamma_modes = np.exp(u[gamma_start : gamma_start + n_modes])
            derivatives = np.empty(
                (omega.size, 1 + 3 * n_modes),
                dtype=complex,
            )
            derivatives[:, alpha_inf_index] = 1.0
            for index, (strength, mode_omega, mode_gamma) in enumerate(
                zip(strengths, omega_modes, gamma_modes)
            ):
                denominator = (
                    mode_omega**2
                    - omega**2
                    - 1j * mode_gamma * omega
                )
                denominator_squared = denominator**2
                derivatives[:, strength_start + index] = 1.0 / denominator
                derivatives[:, omega_start + index] = (
                    -2.0 * strength * mode_omega**2 / denominator_squared
                )
                derivatives[:, gamma_start + index] = (
                    1j
                    * strength
                    * mode_gamma
                    * omega
                    / denominator_squared
                )
            inverse_derivatives = -derivatives / alpha_fit[:, None] ** 2

            alpha_scale_factor = (
                np.sqrt(self.alpha_objective_weight) * weights
            )
            inverse_scale_factor = (
                np.sqrt(self.inv_alpha_objective_weight) * weights
            )
            parts = [
                alpha_scale_factor * r_alpha.real / scale_alpha_re,
                alpha_scale_factor * r_alpha.imag / scale_alpha_im,
                inverse_scale_factor * r_inv.real / scale_inv_re,
                inverse_scale_factor * r_inv.imag / scale_inv_im,
            ]
            jacobian_parts = [
                alpha_scale_factor[:, None] * derivatives.real / scale_alpha_re,
                alpha_scale_factor[:, None] * derivatives.imag / scale_alpha_im,
                inverse_scale_factor[:, None] * inverse_derivatives.real / scale_inv_re,
                inverse_scale_factor[:, None] * inverse_derivatives.imag / scale_inv_im,
            ]
            residual_values = np.concatenate(parts)
            jacobian_values = np.vstack(jacobian_parts)
            evaluation_cache[n_modes] = (
                u.copy(),
                residual_values,
                jacobian_values,
            )
            return residual_values, jacobian_values

        def residual(u: np.ndarray, n_modes: int) -> np.ndarray:
            return residual_and_jacobian(u, n_modes)[0]

        def jacobian(u: np.ndarray, n_modes: int) -> np.ndarray:
            return residual_and_jacobian(u, n_modes)[1]

        def build_spread_start(
            n_modes: int,
            alpha_inf_guess: float,
            mode_shift: float,
            gamma_factor: float,
        ) -> np.ndarray:
            centers = np.linspace(-0.5, 0.5, n_modes)
            w_guess = omega_peak + mode_shift * omega_span * centers
            w_guess = np.clip(
                w_guess,
                1.02 * omega_lo_scalar,
                omega_hi_scalar / 1.02,
            )
            g_guess = np.clip(
                gamma_factor * np.maximum(w_guess, 0.05 * omega_peak),
                1.05 * gamma_lo_scalar,
                gamma_hi_scalar / 1.05,
            )

            strengths = [
                strength_scale * (1.0 + 0.35 * idx)
                for idx, _ in enumerate(w_guess)
            ]

            return np.concatenate([
                [alpha_inf_guess],
                np.asarray(strengths, dtype=float),
                np.log(w_guess),
                np.log(g_guess),
            ])

        def candidate_metrics(n_modes: int, u: np.ndarray) -> dict[str, object] | None:
            u = np.asarray(u, dtype=float).copy()
            lower, upper = parameter_bounds(n_modes)
            bound_tolerance = 1.0e-12 * np.maximum(1.0, np.maximum(np.abs(lower), np.abs(upper)))
            if np.any(u < lower - bound_tolerance) or np.any(u > upper + bound_tolerance):
                return None
            u[0] = physical_alpha_inf
            omega_modes = np.exp(u[1 + n_modes : 1 + 2 * n_modes])
            order = np.argsort(omega_modes)
            u[1 : 1 + n_modes] = u[1 : 1 + n_modes][order]
            u[1 + n_modes : 1 + 2 * n_modes] = u[1 + n_modes : 1 + 2 * n_modes][order]
            u[1 + 2 * n_modes : 1 + 3 * n_modes] = u[1 + 2 * n_modes : 1 + 3 * n_modes][order]

            alpha_fit = self._alpha_model_from_params(omega, u, n_modes=n_modes)
            inv_fit = 1.0 / alpha_fit
            if not (np.all(np.isfinite(alpha_fit)) and np.all(np.isfinite(inv_fit))):
                return None
            alpha_passivity = self._alpha_model_from_params(
                passivity_omega,
                u,
                n_modes=n_modes,
            )
            min_imag_alpha = float(np.min(alpha_passivity.imag))
            if min_imag_alpha < -passivity_tol:
                return None
            rms_alpha = float(np.sqrt(np.mean(np.abs(alpha_fit - alpha_true) ** 2)))
            rms_inv = float(np.sqrt(np.mean(np.abs(inv_fit - inv_true) ** 2)))
            normalized_rms_alpha = rms_alpha / max(
                float(np.sqrt(np.mean(np.abs(alpha_true) ** 2))),
                1e-15,
            )
            normalized_rms_inv = rms_inv / max(
                float(np.sqrt(np.mean(np.abs(inv_true) ** 2))),
                1e-15,
            )
            max_normalized_alpha_error = float(
                np.max(
                    np.abs(alpha_fit - alpha_true)
                    / np.maximum(
                        np.abs(alpha_true),
                        1e-15 * max(float(np.max(np.abs(alpha_true))), 1.0),
                    )
                )
            )
            score = float(np.sqrt(np.mean(residual(u, n_modes) ** 2)))
            return {
                'u': u,
                'score': score,
                'rms_alpha': rms_alpha,
                'rms_inv': rms_inv,
                'normalized_rms_alpha': normalized_rms_alpha,
                'normalized_rms_inv': normalized_rms_inv,
                'max_normalized_alpha_error': max_normalized_alpha_error,
                'min_imag_alpha': min_imag_alpha,
            }

        def optimize_starts(
            n_modes: int,
            starts: list[np.ndarray],
            *,
            max_nfev: int,
        ) -> dict[str, object] | None:
            lower, upper = parameter_bounds(n_modes)
            best_stage: dict[str, object] | None = None
            for u0 in starts:
                u0 = np.clip(np.asarray(u0, dtype=float), lower, upper)
                try:
                    result = least_squares(
                        lambda trial: residual(trial, n_modes),
                        jac=lambda trial: jacobian(trial, n_modes),
                        x0=u0,
                        bounds=(lower, upper),
                        method='trf',
                        loss='soft_l1',
                        f_scale=0.3,
                        x_scale='jac',
                        max_nfev=max_nfev,
                        ftol=2e-11,
                        xtol=2e-11,
                        gtol=2e-11,
                    )
                except (FloatingPointError, ValueError):
                    continue
                candidate = candidate_metrics(n_modes, result.x)
                if candidate is not None and (
                    best_stage is None or candidate['score'] < best_stage['score']
                ):
                    best_stage = candidate
            return best_stage

        def meets_fast_quality_gate(candidate: dict[str, object] | None) -> bool:
            if candidate is None:
                return False
            rms_limit = min(
                0.025,
                0.025
                if self.max_fit_normalized_rms is None
                else self.max_fit_normalized_rms,
            )
            point_limit = min(
                0.05,
                0.05
                if self.max_fit_pointwise_relative_error is None
                else self.max_fit_pointwise_relative_error,
            )
            return bool(
                candidate['normalized_rms_alpha'] <= rms_limit
                and candidate['normalized_rms_inv'] <= rms_limit
                and candidate['max_normalized_alpha_error'] <= point_limit
            )

        best: dict[str, object] | None = None
        canonical_aspect = 15.0 / 7.0
        has_bundled_n9_warm_start = bool(
            self.n_modes == 9
            and self.params.material is DEFAULT_AU_MATERIAL
            and np.allclose(self.fit_window_eV, (0.8, 3.0), rtol=0.0, atol=1e-14)
        )
        if has_bundled_n9_warm_start:
            alpha_seed, strength_seed, omega_seed, gamma_seed = (
                _CANONICAL_PASSIVE_N9_SEEDS[self.orientation]
            )
            warm_u = np.concatenate([
                [alpha_seed],
                strength_seed,
                np.log(omega_seed),
                np.log(gamma_seed),
            ])
            exact_canonical_problem = bool(
                np.isclose(self.params.eps_m, 1.0, rtol=0.0, atol=1e-14)
                and np.isclose(
                    self.params.c_au / self.params.a_au,
                    canonical_aspect,
                    rtol=0.0,
                    atol=1e-12,
                )
                and self.weight_center_eV is None
                and self.weight_sigma_eV is None
                and np.isclose(
                    self.alpha_objective_weight,
                    1.0,
                    rtol=0.0,
                    atol=1e-14,
                )
                and np.isclose(
                    self.inv_alpha_objective_weight,
                    1.2,
                    rtol=0.0,
                    atol=1e-14,
                )
            )
            warm_best = candidate_metrics(self.n_modes, warm_u)
            if not exact_canonical_problem or not meets_fast_quality_gate(warm_best):
                warm_best = optimize_starts(
                    self.n_modes,
                    [warm_u],
                    max_nfev=750,
                )
            if meets_fast_quality_gate(warm_best):
                best = warm_best

        if best is None:
            # A direct many-mode optimization is highly non-convex.  Start
            # with a smaller identifiable model and insert one weak passive
            # pole at a time.  This slower path supports custom material,
            # geometry, medium and fit windows without using a signed fit.
            base_n = min(4, self.n_modes)
            base_starts: list[np.ndarray] = [
                build_spread_start(base_n, alpha_med, 0.35, 0.10),
                build_spread_start(base_n, alpha_med, 0.55, 0.18),
                build_spread_start(base_n, alpha_med, 0.70, 0.28),
                build_spread_start(base_n, alpha_med + 0.5 * alpha_scale, 0.30, 0.07),
                build_spread_start(base_n, alpha_med - 0.5 * alpha_scale, 0.85, 0.35),
            ]
            if base_n == 4:
                for frequency_ratios in (
                    (0.75, 0.93, 1.11, 1.37),
                    (0.88, 1.02, 1.19, 1.50),
                    (0.44, 0.97, 1.24, 1.77),
                ):
                    w_guess = np.clip(
                        omega_peak * np.asarray(frequency_ratios),
                        1.02 * omega_lo_scalar,
                        omega_hi_scalar / 1.02,
                    )
                    g_guess = np.clip(
                        omega_peak * np.asarray((0.13, 0.09, 0.18, 0.35)),
                        1.05 * gamma_lo_scalar,
                        gamma_hi_scalar / 1.05,
                    )
                    base_starts.append(np.concatenate([
                        [alpha_med],
                        np.full(4, 0.15 * strength_scale),
                        np.log(w_guess),
                        np.log(g_guess),
                    ]))

            best = optimize_starts(base_n, base_starts, max_nfev=2000)
            if best is None:
                raise RuntimeError(
                    'A passive base Lorentz fit was not found on fit_window_eV.'
                )

            for n_modes in range(base_n + 1, self.n_modes + 1):
                previous_n = n_modes - 1
                previous = np.asarray(best['u'], dtype=float)
                alpha_inf_previous = previous[0]
                strengths_previous = previous[1 : 1 + previous_n]
                omega_logs_previous = previous[1 + previous_n : 1 + 2 * previous_n]
                gamma_logs_previous = previous[1 + 2 * previous_n : 1 + 3 * previous_n]
                tiny_strength = max(1e-7 * strength_scale, 1e-14)

                insertion_omegas = np.linspace(
                    max(1.02 * omega_lo_scalar, 0.5 * omega_min),
                    min(omega_hi_scalar / 1.02, 1.5 * omega_max),
                    9,
                )
                insertion_gammas = np.clip(
                    omega_peak * np.asarray((0.025, 0.10, 0.40)),
                    1.05 * gamma_lo_scalar,
                    gamma_hi_scalar / 1.05,
                )
                continuation_starts: list[np.ndarray] = []
                for inserted_omega in insertion_omegas:
                    for inserted_gamma in insertion_gammas:
                        continuation_starts.append(np.concatenate([
                            [alpha_inf_previous],
                            strengths_previous,
                            [tiny_strength],
                            omega_logs_previous,
                            [np.log(inserted_omega)],
                            gamma_logs_previous,
                            [np.log(inserted_gamma)],
                        ]))

                continued = optimize_starts(
                    n_modes,
                    continuation_starts,
                    max_nfev=1500,
                )
                if continued is None:
                    raise RuntimeError(
                        f'Passive Lorentz continuation failed while adding mode {n_modes}.'
                    )
                best = continued

        if best is None:
            raise RuntimeError('A stable and passive rational fit was not found on fit_window_eV.')

        u = best['u']
        alpha_inf = physical_alpha_inf
        strengths = np.asarray(u[1 : 1 + self.n_modes], dtype=float)
        omega_modes = np.exp(u[1 + self.n_modes : 1 + 2 * self.n_modes])
        gamma_modes = np.exp(u[1 + 2 * self.n_modes : 1 + 3 * self.n_modes])
        if np.any(strengths < 0.0):
            raise RuntimeError('Internal error: passive Lorentz fit produced a negative residue.')
        if self.max_fit_normalized_rms is not None and (
            best['normalized_rms_alpha'] > self.max_fit_normalized_rms
            or best['normalized_rms_inv'] > self.max_fit_normalized_rms
        ):
            raise RuntimeError(
                'Passive Lorentz fit did not reach the requested accuracy: '
                f"NRMS(alpha)={best['normalized_rms_alpha']:.4g}, "
                f"NRMS(1/alpha)={best['normalized_rms_inv']:.4g}, "
                f'limit={self.max_fit_normalized_rms:.4g}. Increase n_modes, '
                'adjust the fit window/weights, or relax the explicitly recorded limit.'
            )
        if (
            self.max_fit_pointwise_relative_error is not None
            and best['max_normalized_alpha_error']
            > self.max_fit_pointwise_relative_error
        ):
            raise RuntimeError(
                'Passive Lorentz fit did not reach the pointwise accuracy gate: '
                f"max relative alpha error={best['max_normalized_alpha_error']:.4g}, "
                f'limit={self.max_fit_pointwise_relative_error:.4g}. Increase n_modes '
                'or adjust the fit window/weights.'
            )

        return RationalLorentzFit(
            alpha_inf=alpha_inf,
            strengths_au2=strengths,
            omega_modes_au=omega_modes,
            gamma_modes_au=gamma_modes,
            energies_used_eV=energies.copy(),
            alpha_used=alpha_true.copy(),
            rms_alpha=float(best['rms_alpha']),
            rms_inv_alpha=float(best['rms_inv']),
            cost=float(best['score']),
            normalized_rms_alpha=float(best['normalized_rms_alpha']),
            normalized_rms_inv_alpha=float(best['normalized_rms_inv']),
            max_normalized_alpha_error=float(best['max_normalized_alpha_error']),
            min_imag_alpha_fit_window=float(best['min_imag_alpha']),
            passivity_grid_points=int(passivity_omega.size),
            passive_on_fit_window=True,
            passive_for_all_positive_frequencies=True,
        )

    # ------------------------------------------------------------
    # Frequency-domain reconstruction
    # ------------------------------------------------------------
    def alpha_from_material(self, energies_eV: float | np.ndarray) -> np.ndarray:
        """Interpolated quasistatic ellipsoid response used as the fit target."""

        return self._alpha_dimless_at(energies_eV)

    def alpha_from_fit(
        self,
        energies_eV: np.ndarray,
        *,
        allow_extrapolation: bool = False,
    ) -> np.ndarray:
        energies = np.asarray(energies_eV, dtype=float)
        if np.any(~np.isfinite(energies)):
            raise ValueError('Requested fit energies must be finite.')
        e_min, e_max = self.fit_window_eV
        scale = max(abs(e_min), abs(e_max), 1.0)
        tolerance = 1e-12 * scale
        if not allow_extrapolation and (
            np.any(energies < e_min - tolerance) or np.any(energies > e_max + tolerance)
        ):
            raise ValueError(
                f'Requested energy lies outside fit_window_eV={self.fit_window_eV}; '
                'refit on a wider window instead of extrapolating the plasmon modes.'
            )
        omega = eV_to_au(energies)
        u = np.concatenate([
            [self.fit.alpha_inf],
            self.fit.strengths_au2,
            np.log(self.fit.omega_modes_au),
            np.log(self.fit.gamma_modes_au),
        ])
        return self._alpha_model_from_params(omega, u)

    def inv_alpha_from_fit(
        self,
        energies_eV: np.ndarray,
        *,
        allow_extrapolation: bool = False,
    ) -> np.ndarray:
        return 1.0 / self.alpha_from_fit(
            energies_eV,
            allow_extrapolation=allow_extrapolation,
        )

    def transfer_polynomials_desc(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Возвращает полиномы D(s) и N(s) в порядке убывания степеней,
        такие что alpha(s) = N(s) / D(s).

        Тогда 1/alpha(s) = D(s) / N(s).
        """
        quads_desc = [np.array([1.0, g, w**2], dtype=float)
                      for w, g in zip(self.fit.omega_modes_au, self.fit.gamma_modes_au)]

        D = np.array([1.0], dtype=float)
        for q in quads_desc:
            D = np.polymul(D, q)

        N = self.fit.alpha_inf * D.copy()
        for k, strength in enumerate(self.fit.strengths_au2):
            prod = np.array([1.0], dtype=float)
            for j, q in enumerate(quads_desc):
                if j == k:
                    continue
                prod = np.polymul(prod, q)
            N = np.polyadd(N, strength * prod)

        return D, N

    def inv_alpha_from_transfer(self, energies_eV: np.ndarray) -> np.ndarray:
        omega = eV_to_au(np.asarray(energies_eV, dtype=float))
        D, N = self.transfer_polynomials_desc()
        s = -1j * omega
        Dv = np.polyval(D, s)
        Nv = np.polyval(N, s)
        return Dv / Nv

    def print_fit_summary(self) -> None:
        print('\n=== Stable rational fit for alpha(omega) ===')
        print(f'orientation       : {self.orientation}')
        print(f'n_modes           : {self.n_modes}')
        print(f'fit_window_eV     : {self.fit_window_eV}')
        print(f'alpha_inf         : {self.fit.alpha_inf:.8g}')
        print(f'RMS alpha error   : {self.fit.rms_alpha:.6e}')
        print(f'RMS 1/alpha error : {self.fit.rms_inv_alpha:.6e}')
        print(f'NRMS alpha error  : {self.fit.normalized_rms_alpha:.6e}')
        print(f'NRMS 1/alpha error: {self.fit.normalized_rms_inv_alpha:.6e}')
        print(f'max norm alpha err: {self.fit.max_normalized_alpha_error:.6e}')
        print(f'weighted score    : {self.fit.cost:.6e}')
        print(f'min Im[alpha]     : {self.fit.min_imag_alpha_fit_window:.6e} (fit window)')
        print(f'passivity grid    : {self.fit.passivity_grid_points} points')
        print(
            'Im[alpha]>=0 all w: '
            f'{self.fit.nonnegative_imaginary_part_all_positive_frequencies}'
        )
        print(f'max Re[pole]      : {self.linear_stability.spectral_abscissa_au:.6e} au')
        print(f'coupled stable    : {self.linear_stability.stable}')
        print(
            'applicability     : '
            f'k_m*c={self.applicability.medium_size_parameter_kc:.4g}, '
            f'k_m*R={self.applicability.medium_separation_parameter_kR:.4g}, '
            f'c/R={self.applicability.mnp_size_to_separation_ratio:.4g}, '
            f'r_QD/R={self.applicability.qd_size_to_separation_ratio:.4g} '
            f'(guide <<1; warning level {self.applicability.guide_threshold:g})'
        )
        if not self.applicability.quasistatic_guide_satisfied:
            print(
                'WARNING applicability: k_m*c or k_m*R exceeds the selected '
                'quasistatic warning guide; retardation may be material.'
            )
        if not self.applicability.point_dipole_guide_satisfied:
            print(
                'WARNING applicability: c/R or r_QD/R exceeds the point-dipole '
                'warning guide. Interpret this run qualitatively; a multipole/'
                'finite-size convergence study is required for quantitative use.'
            )
        for idx, (f_k, w_k, g_k) in enumerate(zip(self.fit.strengths_au2, self.fit.omega_modes_au, self.fit.gamma_modes_au), start=1):
            print(
                f'mode {idx}: strength={f_k:.6e} au^2, '
                f'omega0={au_to_eV(w_k):.6f} eV, gamma={au_to_eV(g_k):.6f} eV'
            )

    # ------------------------------------------------------------
    # Time-domain coupled dynamics
    # ------------------------------------------------------------
    def initial_state(self) -> np.ndarray:
        # [q1, v1, q2, v2, ..., W, Q, P]
        y0 = np.zeros(2 * self.n_modes + 3, dtype=float)
        y0[2 * self.n_modes] = -1.0
        return y0

    def default_time_span(
        self,
        pulse: GaussianPulse,
        n_sigma: float = 8.0,
        decay_times: float = 8.0,
    ) -> tuple[float, float]:
        """Return a pulse-covered span with a coherent post-pulse tail."""
        if not np.isfinite(n_sigma) or n_sigma <= 0.0:
            raise ValueError('n_sigma must be finite and positive.')
        sigma = pulse.sigma_t_au
        post_au = max(
            n_sigma * sigma,
            self.recommended_post_pulse_time_au(decay_times=decay_times),
        )
        return -n_sigma * sigma, float(post_au)

    def recommended_post_pulse_time_au(self, decay_times: float = 8.0) -> float:
        """Tail duration needed for coherent QD/MNP dipoles to decay.

        Lorentz oscillator amplitudes decay as exp(-gamma_k*t/2), whereas QD
        coherence decays as exp(-Gamma2*t).  The full coupled coherent Jacobian
        is also inspected so a slowly decaying hybrid pole controls the window.
        The population lifetime is not used because population alone has no
        optical dipole after the field is off.
        """
        if not np.isfinite(decay_times) or decay_times <= 0.0:
            raise ValueError('decay_times must be finite and positive.')
        rates = [float(self.params.Gamma_au)]
        rates.extend(float(gamma) / 2.0 for gamma in self.fit.gamma_modes_au)

        # Conservative uncoupled rates are not enough close to a feedback
        # instability, where a hybrid coherent pole can decay much more slowly.
        # Remove W: at the ground-state tail the population mode is linearly
        # decoupled and carries no optical dipole, whereas Q/P and q_k/v_k do.
        jacobian = self.linearized_ground_state_jacobian()
        W_index = 2 * self.n_modes
        coherent_jacobian = np.delete(np.delete(jacobian, W_index, axis=0), W_index, axis=1)
        coherent_poles = np.linalg.eigvals(coherent_jacobian)
        coherent_rate = -float(np.max(np.real(coherent_poles)))
        rates.append(coherent_rate)
        if any(not np.isfinite(rate) or rate <= 0.0 for rate in rates):
            raise ValueError('A finite post-pulse tail cannot be inferred with an undamped coherent mode.')
        return float(decay_times / min(rates))

    def linearized_ground_state_jacobian(
        self,
        *,
        d_au: float | None = None,
        omega0_au: float | None = None,
        gamma1_au: float | None = None,
        gamma2_au: float | None = None,
        g_factor: float | None = None,
        R_au: float | None = None,
    ) -> np.ndarray:
        """Return the full field-free Jacobian at W=-1, Q=P=q_k=v_k=0.

        Optional values let parameter scans test each candidate without
        refitting the MNP response.
        """
        p = self.params
        d = float(p.d_au if d_au is None else d_au)
        omega0 = float(p.omega0_au if omega0_au is None else omega0_au)
        gamma1 = float(p.gamma_au if gamma1_au is None else gamma1_au)
        gamma2 = float(p.Gamma_au if gamma2_au is None else gamma2_au)
        coupling_factor = float(p.G if g_factor is None else g_factor)
        separation = float(p.R_au if R_au is None else R_au)
        scalars = np.asarray(
            [d, omega0, gamma1, gamma2, coupling_factor, separation],
            dtype=float,
        )
        if np.any(~np.isfinite(scalars)):
            raise ValueError('Linear-stability parameters must be finite.')
        if d < 0.0 or omega0 <= 0.0 or gamma1 < 0.0 or gamma2 < 0.0:
            raise ValueError('Linear-stability dipole/frequency/rates must be physically non-negative.')
        if separation <= p.c_au + p.qd_radius_au:
            raise ValueError('Linear-stability separation must preserve a positive QD-MNP gap.')
        if not any(
            np.isclose(coupling_factor, value, rtol=0.0, atol=1e-12)
            for value in ORIENTATION_FACTORS.values()
        ):
            raise ValueError('Linear-stability G must be exactly 2 (long) or -1 (trans).')
        expected_coupling_factor = orientation_factor(self.orientation)
        if not np.isclose(
            coupling_factor,
            expected_coupling_factor,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f'orientation={self.orientation!r} requires '
                f'G={expected_coupling_factor:g} in the linear-stability '
                'Jacobian; mixing a transverse coupling with a longitudinal '
                'MNP polarizability (or vice versa) is outside this model.'
            )
        if gamma2 < 0.5 * gamma1:
            raise ValueError('Linear-stability check requires Gamma2 >= gamma1/2.')

        n = self.n_modes
        size = 2 * n + 3
        W_index, Q_index, P_index = 2 * n, 2 * n + 1, 2 * n + 2
        jacobian = np.zeros((size, size), dtype=float)
        J = coupling_factor / (p.eps_m * separation**3)
        local_field = p.qd_local_field_factor

        for k, (strength, mode_omega, mode_gamma) in enumerate(
            zip(self.fit.strengths_au2, self.fit.omega_modes_au, self.fit.gamma_modes_au)
        ):
            q_index = 2 * k
            v_index = q_index + 1
            jacobian[q_index, v_index] = 1.0
            jacobian[v_index, q_index] = -(mode_omega**2)
            jacobian[v_index, v_index] = -mode_gamma
            jacobian[v_index, P_index] = strength * J * local_field * d

        jacobian[W_index, W_index] = -gamma1
        jacobian[Q_index, 0 : 2 * n : 2] = 2.0 * d * local_field * J * self.C
        jacobian[Q_index, Q_index] = -gamma2
        jacobian[Q_index, P_index] = (
            -omega0
            + 2.0 * d**2 * local_field**2 * J**2 * self.C * self.fit.alpha_inf
        )
        jacobian[P_index, Q_index] = omega0
        jacobian[P_index, P_index] = -gamma2
        return jacobian

    def linearized_ground_state_stability(self, **overrides) -> LinearStabilityDiagnostics:
        """Evaluate whether the coupled ground state has an unstable pole."""
        jacobian = self.linearized_ground_state_jacobian(**overrides)
        poles = np.asarray(np.linalg.eigvals(jacobian), dtype=complex)
        poles.setflags(write=False)
        spectral_abscissa = float(np.max(poles.real))
        scale = max(
            abs(float(self.params.omega0_au if overrides.get('omega0_au') is None else overrides['omega0_au'])),
            float(np.max(np.abs(self.fit.omega_modes_au))) if self.n_modes else 0.0,
            float(np.max(np.abs(self.fit.gamma_modes_au))) if self.n_modes else 0.0,
            1e-15,
        )
        tolerance = 1e-10 * scale
        return LinearStabilityDiagnostics(
            poles_au=poles,
            spectral_abscissa_au=spectral_abscissa,
            tolerance_au=float(tolerance),
            stable=bool(spectral_abscissa <= tolerance),
        )

    def assert_linearized_ground_state_stable(self, **overrides) -> LinearStabilityDiagnostics:
        """Raise before propagation when the passive ground state self-excites."""
        diagnostics = self.linearized_ground_state_stability(**overrides)
        if not diagnostics.stable:
            raise RuntimeError(
                'Unstable coupled QD-MNP ground state: the full field-free Jacobian '
                f'has max Re(lambda)={diagnostics.spectral_abscissa_au:.6e} au, '
                f'above tolerance {diagnostics.tolerance_au:.6e} au. Reduce coupling '
                'or revise the rational fit/model applicability.'
            )
        return diagnostics

    def _unpack_mode_states(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = y[: 2 * self.n_modes : 2]
        v = y[1 : 2 * self.n_modes : 2]
        return q, v

    def rhs(self, t_au: float, y: np.ndarray, pulse: GaussianPulse) -> np.ndarray:
        p = self.params
        q, v = self._unpack_mode_states(y)
        W, Q, P = y[2 * self.n_modes : 2 * self.n_modes + 3]

        E = float(pulse.field(t_au))
        local_field = p.qd_local_field_factor
        mu_d = local_field * p.d_au * P
        F = E + self.J * mu_d

        mu_p = self.C * (self.fit.alpha_inf * F + np.sum(q))
        E_eff_qd = local_field * (E + self.J * mu_p)
        Omega_rabi = 2.0 * p.d_au * E_eff_qd

        dydt = np.zeros_like(y)
        for k, (q_k, v_k, f_k, w_k, g_k) in enumerate(zip(q, v, self.fit.strengths_au2, self.fit.omega_modes_au, self.fit.gamma_modes_au)):
            dydt[2 * k] = v_k
            dydt[2 * k + 1] = f_k * F - g_k * v_k - (w_k**2) * q_k

        dydt[2 * self.n_modes] = Omega_rabi * Q - p.gamma_au * (W + 1.0)
        dydt[2 * self.n_modes + 1] = -p.omega0_au * P - Omega_rabi * W - p.Gamma_au * Q
        dydt[2 * self.n_modes + 2] = p.omega0_au * Q - p.Gamma_au * P
        return dydt

    def solve(
        self,
        pulse: GaussianPulse,
        *,
        method: Literal['Radau', 'BDF', 'LSODA', 'RK45', 'DOP853'] = 'Radau',
        rtol: float = 1e-8,
        atol: float = 1e-10,
        max_step_au: float | None = None,
        t_span_au: tuple[float, float] | None = None,
        positivity_tol: float = 1e-6,
        positivity_policy: Literal['raise', 'warn', 'ignore'] = 'raise',
        spectral_window_policy: Literal['raise', 'warn', 'ignore'] = 'raise',
        max_spectral_leakage: float = 1e-3,
    ) -> HybridSolveResult:
        # Recheck here as callers can replace params after model construction.
        self._validate_physical_parameters()
        linear_stability = self.assert_linearized_ground_state_stable()
        if positivity_policy not in {'raise', 'warn', 'ignore'}:
            raise ValueError("positivity_policy must be 'raise', 'warn' or 'ignore'.")
        if spectral_window_policy not in {'raise', 'warn', 'ignore'}:
            raise ValueError("spectral_window_policy must be 'raise', 'warn' or 'ignore'.")
        if not np.isfinite(positivity_tol) or positivity_tol < 0.0:
            raise ValueError('positivity_tol must be finite and non-negative.')
        if not np.isfinite(max_spectral_leakage) or not 0.0 <= max_spectral_leakage < 1.0:
            raise ValueError('max_spectral_leakage must be finite and lie in [0, 1).')

        fit_window = getattr(self, 'fit_window_eV', None)
        if fit_window is None:
            spectral_fraction = 1.0
            spectral_leakage = 0.0
        else:
            spectral_fraction = pulse.positive_frequency_spectral_fraction(fit_window)
            spectral_leakage = 1.0 - spectral_fraction
            if spectral_leakage > max_spectral_leakage:
                message = (
                    f'Pulse spectrum is not covered by fit_window_eV={fit_window}: '
                    f'{spectral_leakage:.6g} of positive-frequency spectral energy lies outside, '
                    f'above the allowed {max_spectral_leakage:.6g}. Widen the material-fit window.'
                )
                if spectral_window_policy == 'raise':
                    raise ValueError(message)
                if spectral_window_policy == 'warn':
                    warnings.warn(message, RuntimeWarning, stacklevel=2)
        if t_span_au is None:
            t_span_au = self.default_time_span(pulse)
        if (
            len(t_span_au) != 2
            or not np.all(np.isfinite(t_span_au))
            or t_span_au[1] <= t_span_au[0]
        ):
            raise ValueError('t_span_au must contain finite values with t_end > t_start.')

        boundary_envelope = float(
            np.max(pulse.envelope(np.asarray(t_span_au, dtype=float)))
        )
        if boundary_envelope > 1.0e-6:
            raise ValueError(
                't_span_au truncates the incident pulse: the Gaussian envelope '
                f'is {boundary_envelope:.6g} of its peak at a boundary, above '
                '1e-6. Start from the field-free ground/modal state only after '
                'moving both boundaries into the negligible-envelope tails.'
            )

        requested_max_step_au = max_step_au
        if requested_max_step_au is not None and (
            not np.isfinite(requested_max_step_au) or requested_max_step_au <= 0.0
        ):
            raise ValueError('max_step_au must be finite and positive.')
        frequency_candidates = [
            float(pulse.omegaL_au),
            float(self.params.omega0_au),
        ]
        frequency_candidates.extend(
            float(value) for value in self.fit.omega_modes_au
        )
        frequency_candidates.extend(
            float(abs(value)) for value in linear_stability.poles_au
        )
        incident_peak_rabi_frequency = float(
            2.0
            * abs(self.params.d_au)
            * self.params.qd_local_field_factor
            * abs(pulse.E0_au)
        )
        frequency_candidates.append(incident_peak_rabi_frequency)
        integration_frequency_ceiling = max(frequency_candidates)
        p = self.params
        rabi_step_refinement_count = 0
        max_rabi_refinements = 3
        while True:
            resolution_step_limit = float(
                2.0 * np.pi / (20.0 * integration_frequency_ceiling)
            )
            max_step_au = (
                resolution_step_limit
                if requested_max_step_au is None
                else min(float(requested_max_step_au), resolution_step_limit)
            )

            sol = solve_ivp(
                fun=lambda t, y: self.rhs(t, y, pulse),
                t_span=t_span_au,
                y0=self.initial_state(),
                method=method,
                rtol=rtol,
                atol=atol,
                max_step=max_step_au,
                dense_output=False,
            )
            if not sol.success:
                raise RuntimeError(sol.message)

            t = sol.t
            y = sol.y
            state_is_finite = bool(
                np.all(np.isfinite(t)) and np.all(np.isfinite(y))
            )
            if not state_is_finite:
                raise RuntimeError('solve_ivp returned non-finite times or state values.')
            if len(t) < 2 or np.any(np.diff(t) <= 0.0):
                raise RuntimeError('solve_ivp returned an invalid or non-monotone time grid.')
            end_scale = max(1.0, abs(float(t_span_au[1])))
            t_final_reached = bool(
                abs(float(t[-1]) - float(t_span_au[1])) <= 1e-10 * end_scale
            )
            if not t_final_reached:
                raise RuntimeError(
                    'solve_ivp reported success but did not reach the requested final time.'
                )

            q = y[: 2 * self.n_modes : 2]
            v = y[1 : 2 * self.n_modes : 2]
            W, Q, P = y[2 * self.n_modes : 2 * self.n_modes + 3]
            E = pulse.field(t)
            E_dot = pulse.field_dot(t)
            local_field = p.qd_local_field_factor
            mu_d = local_field * p.d_au * P
            mu_d_dot = local_field * p.d_au * (
                p.omega0_au * Q - p.Gamma_au * P
            )
            F = E + self.J * mu_d
            F_dot = E_dot + self.J * mu_d_dot
            mu_p = self.C * (self.fit.alpha_inf * F + np.sum(q, axis=0))
            mu_p_dot = self.C * (
                self.fit.alpha_inf * F_dot + np.sum(v, axis=0)
            )
            observed_peak_rabi_frequency = float(
                np.max(
                    np.abs(
                        2.0 * p.d_au * local_field * (E + self.J * mu_p)
                    )
                )
            )
            required_frequency_ceiling = max(
                integration_frequency_ceiling,
                observed_peak_rabi_frequency,
            )
            required_step_limit = float(
                2.0 * np.pi / (20.0 * required_frequency_ceiling)
            )
            if max_step_au <= required_step_limit * (1.0 + 1.0e-12):
                integration_frequency_ceiling = required_frequency_ceiling
                resolution_step_limit = required_step_limit
                max_step_au = (
                    required_step_limit
                    if requested_max_step_au is None
                    else min(float(requested_max_step_au), required_step_limit)
                )
                break
            if rabi_step_refinement_count >= max_rabi_refinements:
                raise RuntimeError(
                    'The self-consistent local-field Rabi frequency did not '
                    'converge after automatic max-step refinement. Reduce the '
                    'field amplitude or provide a smaller max_step_au.'
                )
            integration_frequency_ceiling = required_frequency_ceiling
            rabi_step_refinement_count += 1

        bloch_radius = np.sqrt(W**2 + Q**2 + P**2)
        max_bloch_radius = float(np.max(bloch_radius))
        min_density_eigenvalue = float(np.min(0.5 * (1.0 - bloch_radius)))
        if min_density_eigenvalue < -positivity_tol:
            message = (
                'Density-matrix positivity violation: trajectory left the Bloch ball; '
                f'max radius={max_bloch_radius:.9g}, min eigenvalue={min_density_eigenvalue:.9g}. '
                'Check Gamma2>=gamma1/2 and tighten solver tolerances.'
            )
            if positivity_policy == 'raise':
                raise RuntimeError(message)
            if positivity_policy == 'warn':
                warnings.warn(message, RuntimeWarning, stacklevel=2)

        mu_total = mu_p + mu_d
        mu_total_dot = mu_p_dot + mu_d_dot

        work_from_incident_field_au = np.trapezoid(mu_total_dot * E, t)
        absolute_work_exchange_au = float(np.trapezoid(np.abs(mu_total_dot * E), t))
        work_passivity_tolerance_au = float(
            1.0e-9 * max(absolute_work_exchange_au, np.finfo(float).tiny)
        )
        work_passivity_checked = True
        work_nonnegative = bool(
            work_from_incident_field_au >= -work_passivity_tolerance_au
        )
        if work_passivity_checked and not work_nonnegative:
            raise RuntimeError(
                'A stable passive QD--MNP run produced materially negative '
                'external-field work after a pulse-complete time window: '
                f'W={work_from_incident_field_au:.6g} au, tolerance='
                f'{work_passivity_tolerance_au:.6g} au. Inspect the modal fit, '
                'time window and energy-balance conventions.'
            )
        work_from_incident_field_j = float(work_from_incident_field_au * AU_ENERGY_J)
        fluence_j_cm2 = float(pulse.fluence_j_cm2(eps_m=p.eps_m))
        sigma_energy_transfer_cm2 = work_from_incident_field_j / fluence_j_cm2
        reconstructed = (mu_p, mu_d, mu_total, mu_total_dot)
        if not all(np.all(np.isfinite(values)) for values in reconstructed):
            raise RuntimeError('Non-finite reconstructed dipole or dipole derivative.')

        if fit_window is None:
            drive_spectral_fraction = 1.0
            drive_spectral_leakage = 0.0
            mnp_dipole_spectral_fraction = 1.0
            mnp_dipole_spectral_leakage = 0.0
        else:
            drive_spectral_fraction = sampled_positive_frequency_spectral_fraction(
                t,
                F,
                fit_window,
                highest_resolved_omega_au=integration_frequency_ceiling,
            )
            drive_spectral_leakage = 1.0 - drive_spectral_fraction
            if drive_spectral_leakage > max_spectral_leakage:
                message = (
                    'The self-consistent field driving the MNP is not covered by '
                    f'fit_window_eV={fit_window}: {drive_spectral_leakage:.6g} of '
                    'its sampled spectral energy lies outside, above the allowed '
                    f'{max_spectral_leakage:.6g}. Nonlinear frequency generation '
                    'requires a wider validated material fit.'
                )
                if spectral_window_policy == 'raise':
                    raise ValueError(message)
                if spectral_window_policy == 'warn':
                    warnings.warn(message, RuntimeWarning, stacklevel=2)
            mnp_dipole_spectral_fraction = sampled_positive_frequency_spectral_fraction(
                t,
                mu_p,
                fit_window,
                highest_resolved_omega_au=integration_frequency_ceiling,
            )
            mnp_dipole_spectral_leakage = 1.0 - mnp_dipole_spectral_fraction
            if mnp_dipole_spectral_leakage > max_spectral_leakage:
                message = (
                    'The MNP dipole response is not covered by '
                    f'fit_window_eV={fit_window}: '
                    f'{mnp_dipole_spectral_leakage:.6g} of its sampled spectral '
                    'energy lies outside, above the allowed '
                    f'{max_spectral_leakage:.6g}. The result is sensitive to an '
                    'unvalidated continuation of the material response.'
                )
                if spectral_window_policy == 'raise':
                    raise ValueError(message)
                if spectral_window_policy == 'warn':
                    warnings.warn(message, RuntimeWarning, stacklevel=2)
        if not np.isfinite(work_from_incident_field_j) or not np.isfinite(sigma_energy_transfer_cm2):
            raise RuntimeError('Non-finite external-field work or pulse cross section.')

        steps = np.diff(t)
        diagnostics = HybridSolveDiagnostics(
            solver_success=bool(sol.success),
            solver_status=int(sol.status),
            solver_message=str(sol.message),
            n_steps=int(len(t)),
            nfev=int(sol.nfev),
            njev=None if getattr(sol, 'njev', None) is None else int(sol.njev),
            nlu=None if getattr(sol, 'nlu', None) is None else int(sol.nlu),
            t_final_reached=t_final_reached,
            state_is_finite=state_is_finite,
            min_step_au=float(np.min(steps)),
            max_step_au=float(np.max(steps)),
            max_step_limit_au=float(max_step_au),
            integration_frequency_ceiling_au=float(
                integration_frequency_ceiling
            ),
            boundary_envelope_fraction=boundary_envelope,
            W_min=float(np.min(W)),
            W_max=float(np.max(W)),
            excited_population_min=float(np.min(0.5 * (W + 1.0))),
            excited_population_max=float(np.max(0.5 * (W + 1.0))),
            max_bloch_radius=max_bloch_radius,
            min_density_eigenvalue=min_density_eigenvalue,
            pulse_spectral_fraction_in_fit_window=float(spectral_fraction),
            pulse_spectral_leakage=float(spectral_leakage),
            mnp_drive_spectral_fraction_in_fit_window=float(
                drive_spectral_fraction
            ),
            mnp_drive_spectral_leakage=float(drive_spectral_leakage),
            mnp_dipole_spectral_fraction_in_fit_window=float(
                mnp_dipole_spectral_fraction
            ),
            mnp_dipole_spectral_leakage=float(mnp_dipole_spectral_leakage),
            work_passivity_checked=work_passivity_checked,
            work_passivity_tolerance_au=work_passivity_tolerance_au,
            work_nonnegative_within_tolerance=work_nonnegative,
            incident_peak_rabi_frequency_au=incident_peak_rabi_frequency,
            observed_peak_rabi_frequency_au=observed_peak_rabi_frequency,
            rabi_step_refinement_count=rabi_step_refinement_count,
        )

        return HybridSolveResult(
            t_au=t,
            y=y,
            mu_p_au=mu_p,
            mu_d_au=mu_d,
            mu_total_au=mu_total,
            mu_dot_total_au=mu_total_dot,
            sigma_energy_transfer_cm2=float(sigma_energy_transfer_cm2),
            work_from_incident_field_j=work_from_incident_field_j,
            fluence_j_cm2=fluence_j_cm2,
            peak_intensity_w_cm2=float(pulse.peak_intensity_w_cm2(eps_m=p.eps_m)),
            solve_ivp_result=sol,
            diagnostics=diagnostics,
        )

    def sweep_energy_transfer_vs_peak_intensity(
        self,
        tau_fs: float,
        E0_values_V_m: Iterable[float],
        *,
        omegaL_eV: float | None = None,
        tau_kind: Literal['sigma', 'fwhm_intensity'] = 'fwhm_intensity',
        method: Literal['Radau', 'BDF', 'LSODA'] = 'Radau',
        rtol: float = 1e-8,
        atol: float = 1e-10,
    ) -> tuple[np.ndarray, np.ndarray, list[HybridSolveResult]]:
        I_peaks = []
        sigmas = []
        results: list[HybridSolveResult] = []

        omegaL_au = self.params.omega0_au if omegaL_eV is None else float(eV_to_au(omegaL_eV))
        tau_au = float(fs_to_au(tau_fs))

        for E0_V_m in E0_values_V_m:
            pulse = GaussianPulse(
                E0_au=float(field_si_to_au(E0_V_m)),
                omegaL_au=omegaL_au,
                tau_au=tau_au,
                tau_kind=tau_kind,
            )
            res = self.solve(pulse, method=method, rtol=rtol, atol=atol)
            I_peaks.append(res.peak_intensity_w_cm2)
            sigmas.append(res.sigma_energy_transfer_cm2)
            results.append(res)

        return np.asarray(I_peaks), np.asarray(sigmas), results

    def sweep_absorption_vs_peak_intensity(self, *args, **kwargs):
        """Compatibility alias for sweep_energy_transfer_vs_peak_intensity."""
        warnings.warn(
            'sweep_absorption_vs_peak_intensity() was mislabeled; use '
            'sweep_energy_transfer_vs_peak_intensity().',
            DeprecationWarning,
            stacklevel=2,
        )
        return self.sweep_energy_transfer_vs_peak_intensity(*args, **kwargs)


# ================================================================
# Convenience helpers
# ================================================================
def make_default_params(orientation: DipoleOrientation = 'long') -> HybridSystemParams:
    default_d_au = float(dipole_si_to_au(7.5e-29))
    default_omega0_au = float(eV_to_au(2.042))
    return HybridSystemParams(
        c_au=float(nm_to_au(15.0)),
        a_au=float(nm_to_au(7.0)),
        R_au=float(nm_to_au(18.0)),
        G=orientation_factor(orientation), eps_m=1.0,
        d_au=default_d_au,
        omega0_au=default_omega0_au,
        # Retain the legacy 30 ns value as provenance, but model construction
        # emits a radiative-consistency warning: it is too slow for the legacy
        # external dipole to be a homogeneous-host total decay rate.
        gamma_au=float(1.0 / ns_to_au(30.0)),
        Gamma_au=float(1.0 / fs_to_au(330.0)),
        qd_radius_au=float(nm_to_au(2.0)),
        eps_qd=6.0,
        # The legacy project never recorded whether 7.5e-29 C m was a bare
        # interband matrix element.  Preserve its original external-response
        # normalization until a literature profile supplies a sourced bare d.
        qd_dipole_convention='effective_external',
    )


def make_params_with_overrides(
    *,
    c_nm: float | None = None,
    a_nm: float | None = None,
    r_nm: float | None = None,
    qd_radius_nm: float | None = None,
    g_factor: float | None = None,
    eps_m: float | None = None,
    d_debye: float | None = None,
    omega0_ev: float | None = None,
    gamma_population_mev: float | None = None,
    gamma2_coherence_mev: float | None = None,
    gamma_dephasing_mev: float | None = None,
    eps_qd: float | None = None,
    qd_dipole_convention: QDDipoleConvention | None = None,
    orientation: DipoleOrientation = 'long',
) -> HybridSystemParams:
    params = make_default_params(orientation)
    updates = {}
    if c_nm is not None:
        updates['c_au'] = float(nm_to_au(c_nm))
    if a_nm is not None:
        updates['a_au'] = float(nm_to_au(a_nm))
    if r_nm is not None:
        updates['R_au'] = float(nm_to_au(r_nm))
    if qd_radius_nm is not None:
        updates['qd_radius_au'] = float(nm_to_au(qd_radius_nm))
    if g_factor is not None:
        expected_g = orientation_factor(orientation)
        if not np.isclose(float(g_factor), expected_g, rtol=0.0, atol=1e-12):
            raise ValueError(
                f'orientation={orientation!r} requires G={expected_g:g}; '
                f'phenomenological G={float(g_factor):g} is outside this model.'
            )
        updates['G'] = expected_g
    if eps_m is not None:
        updates['eps_m'] = float(eps_m)
    if eps_qd is not None:
        updates['eps_qd'] = float(eps_qd)
    if qd_dipole_convention is not None:
        updates['qd_dipole_convention'] = qd_dipole_convention
    if d_debye is not None:
        updates['d_au'] = float(d_debye * DEBYE_C_M / AU_DIPOLE_C_M)
    if omega0_ev is not None:
        updates['omega0_au'] = float(eV_to_au(omega0_ev))
    if gamma_population_mev is not None:
        updates['gamma_au'] = float(eV_to_au(gamma_population_mev / 1000.0))
    if gamma2_coherence_mev is not None and gamma_dephasing_mev is not None:
        if float(gamma2_coherence_mev) != float(gamma_dephasing_mev):
            raise ValueError(
                'gamma2_coherence_mev and legacy gamma_dephasing_mev must agree when both are provided.'
            )
    gamma2_mev = gamma2_coherence_mev if gamma2_coherence_mev is not None else gamma_dephasing_mev
    if gamma2_mev is not None:
        updates['Gamma_au'] = float(eV_to_au(float(gamma2_mev) / 1000.0))
    return replace(params, **updates)


def params_to_physical_dict(
    params: HybridSystemParams,
    orientation: str = 'long',
) -> dict[str, float | str | bool | None]:
    radiative = params.radiative_rate_diagnostics
    radiative_ratio = radiative.gamma1_over_homogeneous_radiative_rate
    return {
        'model_profile': NATIVE_MODEL_PROFILE,
        'coupling_model': 'quasistatic_point_dipole',
        'material_interpolation': MATERIAL_INTERPOLATION,
        'material_high_frequency_epsilon': MATERIAL_HIGH_FREQUENCY_EPSILON,
        'c_nm': float(au_to_nm(params.c_au)),
        'a_nm': float(au_to_nm(params.a_au)),
        'R_nm': float(au_to_nm(params.R_au)),
        'qd_radius_nm': float(au_to_nm(params.qd_radius_au)),
        'surface_gap_nm': float(au_to_nm(params.axial_surface_gap_au)),
        'mnp_size_to_separation_ratio_c_over_R': float(params.c_au / params.R_au),
        'G': float(params.G),
        'eps_m': float(params.eps_m),
        'eps_qd': float(params.eps_qd),
        'qd_local_field_factor': float(params.qd_local_field_factor),
        'qd_dipole_convention': params.qd_dipole_convention,
        'd_debye': float(dipole_au_to_debye(params.d_au)),
        'qd_external_dipole_debye': float(
            dipole_au_to_debye(params.qd_external_dipole_au)
        ),
        'omega0_ev': float(au_to_eV(params.omega0_au)),
        'gamma_population_mev': float(au_to_eV(params.gamma_au) * 1000.0),
        'homogeneous_radiative_decay_mev': float(
            au_to_eV(radiative.homogeneous_radiative_decay_au) * 1000.0
        ),
        'gamma1_at_or_above_homogeneous_reference_radiative_rate': bool(
            radiative.homogeneous_host_consistent
        ),
        'gamma1_over_homogeneous_radiative_rate': (
            None if not np.isfinite(radiative_ratio) else float(radiative_ratio)
        ),
        'gamma2_coherence_mev': float(au_to_eV(params.Gamma_au) * 1000.0),
        'gamma_pure_dephasing_mev': float(au_to_eV(params.pure_dephasing_au) * 1000.0),
        # Schema-1 compatibility alias: this has always been the total Gamma2.
        'gamma_dephasing_mev': float(au_to_eV(params.Gamma_au) * 1000.0),
        'orientation': orientation,
    }


def plot_inv_alpha_rational_family(
    params: HybridSystemParams,
    modes_list: tuple[int, ...] = (1, 2, 3, 4),
    orientation: Literal['long', 'trans'] = 'long',
    fit_window_eV: tuple[float, float] = (0.8, 3.0),
    weight_center_eV: float | None = None,
    weight_sigma_eV: float | None = None,
    energy_plot_range: tuple[float, float] = (0.8, 3.0),
    n_plot: int = 800,
) -> None:
    energies = np.linspace(energy_plot_range[0], energy_plot_range[1], n_plot)

    ref = HybridQDPlasmonModel(
        params,
        orientation=orientation,
        n_modes=1,
        fit_window_eV=fit_window_eV,
        weight_center_eV=weight_center_eV,
        weight_sigma_eV=weight_sigma_eV,
        max_fit_normalized_rms=None,
        max_fit_pointwise_relative_error=None,
        verbose=False,
    )
    inv_alpha_true = ref.inv_alpha_tab

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1.plot(params.material.energy_eV, inv_alpha_true.real, 'ko', ms=4, label='Tabulated Re[1/alpha]', zorder=20)
    ax2.plot(params.material.energy_eV, inv_alpha_true.imag, 'ko', ms=4, label='Tabulated Im[1/alpha]', zorder=20)

    styles = {
        1: dict(ls='-', lw=2.6, marker='s', markevery=70, ms=5),
        2: dict(ls='--', lw=2.6, marker='^', markevery=80, ms=5),
        3: dict(ls='-.', lw=2.6, marker='D', markevery=90, ms=5),
        4: dict(ls=':', lw=2.8, marker='v', markevery=100, ms=5),
    }

    for n_modes in modes_list:
        model = HybridQDPlasmonModel(
            params,
            orientation=orientation,
            n_modes=n_modes,
            fit_window_eV=fit_window_eV,
            weight_center_eV=weight_center_eV,
            weight_sigma_eV=weight_sigma_eV,
            max_fit_normalized_rms=None,
            max_fit_pointwise_relative_error=None,
            verbose=False,
        )
        inv_fit = model.inv_alpha_from_fit(energies)
        label = f'rational fit, modes={n_modes}, RMS={model.fit.rms_inv_alpha:.3e}'
        st = styles.get(n_modes, dict(ls='-', lw=2.4))
        ax1.plot(energies, inv_fit.real, label=label, **st)
        ax2.plot(energies, inv_fit.imag, label=label, **st)

    for ax in (ax1, ax2):
        ax.axvspan(fit_window_eV[0], fit_window_eV[1], alpha=0.12, color='gray')
        if weight_center_eV is not None:
            ax.axvline(weight_center_eV, ls=':', lw=1.5)
        ax.grid(True)
        ax.legend(fontsize=9)

    ax1.set_ylabel('Re[1/alpha]')
    ax1.set_title('Stable rational fit: tabulated 1/alpha vs reconstructed 1/alpha')
    ax2.set_ylabel('Im[1/alpha]')
    ax2.set_xlabel('Energy (eV)')
    plt.tight_layout()
    plt.show()


def plot_time_dynamics(result: HybridSolveResult) -> None:
    t_fs = au_to_fs(result.t_au)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(t_fs, result.mu_p_au, label='mu_p(t)')
    axes[0].set_ylabel('MNP dipole (a.u.)')
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(t_fs, result.mu_d_au, label='mu_d(t)')
    axes[1].set_ylabel('QD dipole (a.u.)')
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(t_fs, result.mu_total_au, label='mu_total(t)')
    axes[2].set_xlabel('Time (fs)')
    axes[2].set_ylabel('Total dipole (a.u.)')
    axes[2].grid(True)
    axes[2].legend()
    plt.tight_layout()
    plt.show()


def plot_fit_diagnostics_from_data(data, output_path: Path | None = None, show: bool = True) -> None:
    energy_table = data['energy_table_ev']
    inv_alpha_table = data['inv_alpha_table']
    energy_plot = data['energy_plot_ev']
    modes_list = data['modes_list']
    inv_alpha_fit_by_mode = data['inv_alpha_fit_by_mode']
    rms_inv_alpha_by_mode = data['rms_inv_alpha_by_mode']

    styles = [
        dict(ls='-', lw=2.4),
        dict(ls='--', lw=2.4),
        dict(ls='-.', lw=2.4),
        dict(ls=':', lw=2.7),
        dict(ls=(0, (3, 1, 1, 1)), lw=2.4),
    ]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1.plot(energy_table, inv_alpha_table.real, 'ko', ms=4, label='tabulated Re[1/alpha]', zorder=20)
    ax2.plot(energy_table, inv_alpha_table.imag, 'ko', ms=4, label='tabulated Im[1/alpha]', zorder=20)

    for idx, n_modes in enumerate(modes_list):
        st = styles[idx % len(styles)]
        label = f'rational fit, modes={int(n_modes)}, RMS={rms_inv_alpha_by_mode[idx]:.3e}'
        ax1.plot(energy_plot, inv_alpha_fit_by_mode[idx].real, label=label, **st)
        ax2.plot(energy_plot, inv_alpha_fit_by_mode[idx].imag, label=label, **st)

    ax1.set_ylabel('Re[1/alpha]')
    ax1.set_title('Stable rational fit: tabulated 1/alpha vs reconstructed 1/alpha')
    ax2.set_ylabel('Im[1/alpha]')
    ax2.set_xlabel('Energy (eV)')
    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_time_dynamics_from_data(data, output_path: Path | None = None, show: bool = True) -> None:
    t_fs = data['t_fs']
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(t_fs, data['mu_p_au'], label='mu_p(t)')
    axes[0].set_ylabel('MNP dipole (a.u.)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(t_fs, data['mu_d_au'], label='mu_d(t)')
    axes[1].set_ylabel('QD dipole (a.u.)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(t_fs, data['mu_total_au'], label='mu_total(t)')
    axes[2].set_xlabel('Time (fs)')
    axes[2].set_ylabel('Total dipole (a.u.)')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)


def build_rational_fit_artifact(args: argparse.Namespace) -> Path:
    run_dir = args.run_dir if args.run_dir is not None else timestamped_run_dir('results/rational_fit_runs')
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    orientation = args.orientation
    params = make_params_with_overrides(
        c_nm=args.c_nm,
        a_nm=args.a_nm,
        r_nm=args.r_nm,
        qd_radius_nm=args.qd_radius_nm,
        g_factor=args.g_factor,
        eps_m=args.eps_m,
        eps_qd=args.eps_qd,
        qd_dipole_convention=args.qd_dipole_convention,
        d_debye=args.d_debye,
        omega0_ev=args.omega0_ev,
        gamma_population_mev=args.gamma_population_mev,
        gamma2_coherence_mev=args.gamma2_coherence_mev,
        orientation=orientation,
    )
    fit_window_ev = (args.fit_min_ev, args.fit_max_ev)
    energy_plot = np.linspace(args.energy_min_ev, args.energy_max_ev, args.points)
    modes_list = np.asarray(args.modes, dtype=int)

    inv_alpha_fit_by_mode = []
    alpha_fit_by_mode = []
    rms_alpha_by_mode = []
    rms_inv_alpha_by_mode = []
    normalized_rms_alpha_by_mode = []
    normalized_rms_inv_alpha_by_mode = []
    max_relative_alpha_error_by_mode = []
    fit_cost_by_mode = []
    fit_min_imag_alpha_by_mode = []
    fit_passivity_grid_points_by_mode = []
    fit_target_grid_points_by_mode = []
    models_by_mode: dict[int, HybridQDPlasmonModel] = {}
    inv_alpha_table = None
    energy_table = params.material.energy_eV.copy()

    for n_modes in modes_list:
        model = HybridQDPlasmonModel(
            params,
            orientation=orientation,
            n_modes=int(n_modes),
            fit_window_eV=fit_window_ev,
            weight_center_eV=args.weight_center_ev,
            weight_sigma_eV=args.weight_sigma_ev,
            alpha_objective_weight=1.0,
            inv_alpha_objective_weight=1.2,
            max_fit_normalized_rms=None,
            max_fit_pointwise_relative_error=None,
            verbose=True,
        )
        if inv_alpha_table is None:
            inv_alpha_table = model.inv_alpha_tab.copy()
        models_by_mode[int(n_modes)] = model
        alpha_fit = model.alpha_from_fit(energy_plot)
        alpha_fit_by_mode.append(alpha_fit)
        inv_alpha_fit_by_mode.append(1.0 / alpha_fit)
        rms_alpha_by_mode.append(model.fit.rms_alpha)
        rms_inv_alpha_by_mode.append(model.fit.rms_inv_alpha)
        normalized_rms_alpha_by_mode.append(model.fit.normalized_rms_alpha)
        normalized_rms_inv_alpha_by_mode.append(model.fit.normalized_rms_inv_alpha)
        max_relative_alpha_error_by_mode.append(model.fit.max_normalized_alpha_error)
        fit_cost_by_mode.append(model.fit.cost)
        fit_min_imag_alpha_by_mode.append(model.fit.min_imag_alpha_fit_window)
        fit_passivity_grid_points_by_mode.append(model.fit.passivity_grid_points)
        fit_target_grid_points_by_mode.append(model.fit.energies_used_eV.size)

    dynamics_model = models_by_mode.get(int(args.dynamics_n_modes))
    if dynamics_model is None:
        dynamics_model = HybridQDPlasmonModel(
            params,
            orientation=orientation,
            n_modes=args.dynamics_n_modes,
            fit_window_eV=fit_window_ev,
            weight_center_eV=args.weight_center_ev,
            weight_sigma_eV=args.weight_sigma_ev,
            alpha_objective_weight=1.0,
            inv_alpha_objective_weight=1.2,
            verbose=True,
        )
    elif (
        dynamics_model.fit.normalized_rms_alpha > 0.025
        or dynamics_model.fit.normalized_rms_inv_alpha > 0.025
        or dynamics_model.fit.max_normalized_alpha_error > 0.05
    ):
        raise RuntimeError(
            'The selected diagnostic fit does not pass the production dynamics '
            'quality gates; choose a more accurate --dynamics-n-modes value.'
        )
    pulse = GaussianPulse(
        E0_au=float(field_si_to_au(args.pulse_e0_v_m)),
        omegaL_au=float(eV_to_au(args.omega_l_ev)),
        tau_au=float(fs_to_au(args.tau_fs)),
        tau_kind='fwhm_intensity',
    )
    result = dynamics_model.solve(pulse, method=args.method, rtol=args.rtol, atol=args.atol)
    dynamics_tail_ratio = response_tail_ratio(
        result.mu_total_au,
        result.t_au,
        result.mu_p_au,
        result.mu_d_au,
    )
    dynamics_tail_tolerance = 1.0e-3
    dynamics_tail_converged = bool(
        np.isfinite(dynamics_tail_ratio)
        and dynamics_tail_ratio <= dynamics_tail_tolerance
    )
    if not dynamics_tail_converged:
        raise RuntimeError(
            'The rational-fit demonstration solve retained an unconverged '
            f'dipole tail ratio {dynamics_tail_ratio:.6g}, above '
            f'{dynamics_tail_tolerance:.6g}; no production artifact was written.'
        )

    metadata = {
        'schema_version': SCHEMA_VERSION,
        'script': 'qd_mnp_rational_fit.py',
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'run_dir': str(run_dir),
        'physical': params_to_physical_dict(params, orientation=orientation),
        'fit': {
            'modes_list': [int(x) for x in modes_list],
            'fit_window_ev': [float(fit_window_ev[0]), float(fit_window_ev[1])],
            'weight_center_ev': None if args.weight_center_ev is None else float(args.weight_center_ev),
            'weight_sigma_ev': None if args.weight_sigma_ev is None else float(args.weight_sigma_ev),
            'energy_plot_range_ev': [float(args.energy_min_ev), float(args.energy_max_ev)],
            'points': int(args.points),
            'alpha_objective_weight': 1.0,
            'inv_alpha_objective_weight': 1.2,
            'harmonic_loss_requirement': (
                'Nonnegative Im(alpha) for every positive frequency: '
                'f_k >= 0, omega_k > 0, gamma_k > 0; the dense grid is an assertion. '
                'The direct term is fixed by epsilon(omega->infinity)=1.'
            ),
            'material_interpolation': MATERIAL_INTERPOLATION,
            'production_quality_gates': {
                'normalized_rms_alpha_max': 0.025,
                'normalized_rms_inv_alpha_max': 0.025,
                'max_pointwise_relative_alpha_error': 0.05,
            },
            'passivity_grid_points_by_mode': [int(x) for x in fit_passivity_grid_points_by_mode],
            'target_grid_points_by_mode': [int(x) for x in fit_target_grid_points_by_mode],
        },
        'dynamics': {
            'n_modes': int(args.dynamics_n_modes),
            'pulse_e0_v_m': float(args.pulse_e0_v_m),
            'omega_l_ev': float(args.omega_l_ev),
            'tau_fs': float(args.tau_fs),
            'tau_kind': 'fwhm_intensity',
            'linearized_ground_state_stability': {
                'stable': bool(dynamics_model.linear_stability.stable),
                'spectral_abscissa_au': float(dynamics_model.linear_stability.spectral_abscissa_au),
                'tolerance_au': float(dynamics_model.linear_stability.tolerance_au),
            },
            'applicability': {
                'diagnostic_energy_ev': float(fit_window_ev[1]),
                'medium_size_parameter_kc': float(
                    dynamics_model.applicability.medium_size_parameter_kc
                ),
                'medium_separation_parameter_kR': float(
                    dynamics_model.applicability.medium_separation_parameter_kR
                ),
                'mnp_size_to_separation_ratio_c_over_R': float(
                    dynamics_model.applicability.mnp_size_to_separation_ratio
                ),
                'qd_size_to_separation_ratio_rqd_over_R': float(
                    dynamics_model.applicability.qd_size_to_separation_ratio
                ),
                'quasistatic_guide_satisfied': bool(
                    dynamics_model.applicability.quasistatic_guide_satisfied
                ),
                'point_dipole_guide_satisfied': bool(
                    dynamics_model.applicability.point_dipole_guide_satisfied
                ),
            },
            'modal_parameters': {
                'alpha_inf': float(dynamics_model.fit.alpha_inf),
                'strengths_au2': [
                    float(value) for value in dynamics_model.fit.strengths_au2
                ],
                'omega_modes_au': [
                    float(value) for value in dynamics_model.fit.omega_modes_au
                ],
                'gamma_modes_au': [
                    float(value) for value in dynamics_model.fit.gamma_modes_au
                ],
            },
        },
        'solver': {
            'method': args.method,
            'rtol': float(args.rtol),
            'atol': float(args.atol),
            'success': bool(result.diagnostics.solver_success),
            't_final_reached': bool(result.diagnostics.t_final_reached),
            'n_steps': int(result.diagnostics.n_steps),
            'nfev': int(result.diagnostics.nfev),
            'max_step_limit_au': float(result.diagnostics.max_step_limit_au),
            'integration_frequency_ceiling_au': float(
                result.diagnostics.integration_frequency_ceiling_au
            ),
            'incident_peak_rabi_frequency_au': float(
                result.diagnostics.incident_peak_rabi_frequency_au
            ),
            'observed_peak_rabi_frequency_au': float(
                result.diagnostics.observed_peak_rabi_frequency_au
            ),
            'rabi_step_refinement_count': int(
                result.diagnostics.rabi_step_refinement_count
            ),
        },
        'validation': {
            'response_tail_ratio': float(dynamics_tail_ratio),
            'response_tail_tolerance': dynamics_tail_tolerance,
            'response_tail_converged': dynamics_tail_converged,
            'max_bloch_radius': float(result.max_bloch_radius),
            'min_density_eigenvalue': float(result.min_density_eigenvalue),
            'pulse_spectral_fraction_in_fit_window': float(
                result.diagnostics.pulse_spectral_fraction_in_fit_window
            ),
            'mnp_drive_spectral_fraction_in_fit_window': float(
                result.diagnostics.mnp_drive_spectral_fraction_in_fit_window
            ),
            'mnp_dipole_spectral_fraction_in_fit_window': float(
                result.diagnostics.mnp_dipole_spectral_fraction_in_fit_window
            ),
            'work_passivity_checked': bool(
                result.diagnostics.work_passivity_checked
            ),
            'work_passivity_tolerance_au': float(
                result.diagnostics.work_passivity_tolerance_au
            ),
            'work_nonnegative_within_tolerance': bool(
                result.diagnostics.work_nonnegative_within_tolerance
            ),
            'homogeneous_host_radiative_consistency': bool(
                params.radiative_rate_diagnostics.homogeneous_host_consistent
            ),
            'gamma1_over_homogeneous_radiative_rate': float(
                params.radiative_rate_diagnostics.gamma1_over_homogeneous_radiative_rate
            ),
        },
        'observables': {
            'sigma_energy_transfer_cm2': 'external-field work divided by incident fluence',
            'work_from_incident_field_j': 'integral E_inc(t) * d(mu_total)/dt dt',
            'legacy_aliases': {
                'sigma_abs_cm2': 'sigma_energy_transfer_cm2',
                'absorbed_energy_j': 'work_from_incident_field_j',
            },
        },
    }

    data_path = run_dir / 'data.npz'
    np.savez_compressed(
        data_path,
        energy_table_ev=energy_table,
        inv_alpha_table=np.asarray(inv_alpha_table),
        energy_plot_ev=energy_plot,
        modes_list=modes_list,
        inv_alpha_fit_by_mode=np.asarray(inv_alpha_fit_by_mode),
        alpha_fit_by_mode=np.asarray(alpha_fit_by_mode),
        rms_alpha_by_mode=np.asarray(rms_alpha_by_mode, dtype=float),
        rms_inv_alpha_by_mode=np.asarray(rms_inv_alpha_by_mode, dtype=float),
        normalized_rms_alpha_by_mode=np.asarray(
            normalized_rms_alpha_by_mode,
            dtype=float,
        ),
        normalized_rms_inv_alpha_by_mode=np.asarray(
            normalized_rms_inv_alpha_by_mode,
            dtype=float,
        ),
        max_relative_alpha_error_by_mode=np.asarray(
            max_relative_alpha_error_by_mode,
            dtype=float,
        ),
        fit_cost_by_mode=np.asarray(fit_cost_by_mode, dtype=float),
        fit_min_imag_alpha_by_mode=np.asarray(fit_min_imag_alpha_by_mode, dtype=float),
        fit_passivity_grid_points_by_mode=np.asarray(fit_passivity_grid_points_by_mode, dtype=np.int64),
        fit_target_grid_points_by_mode=np.asarray(fit_target_grid_points_by_mode, dtype=np.int64),
        linearized_ground_state_poles_au=dynamics_model.linear_stability.poles_au,
        dynamics_fit_alpha_inf=np.asarray(dynamics_model.fit.alpha_inf),
        dynamics_fit_strengths_au2=dynamics_model.fit.strengths_au2,
        dynamics_fit_omega_modes_au=dynamics_model.fit.omega_modes_au,
        dynamics_fit_gamma_modes_au=dynamics_model.fit.gamma_modes_au,
        dynamics_fit_target_energy_eV=dynamics_model.fit.energies_used_eV,
        dynamics_fit_target_alpha=dynamics_model.fit.alpha_used,
        linearized_ground_state_spectral_abscissa_au=np.asarray(
            dynamics_model.linear_stability.spectral_abscissa_au
        ),
        t_au=result.t_au,
        t_fs=au_to_fs(result.t_au),
        e_field_au=pulse.field(result.t_au),
        mu_p_au=result.mu_p_au,
        mu_d_au=result.mu_d_au,
        mu_total_au=result.mu_total_au,
        mu_dot_total_au=result.mu_dot_total_au,
        y=result.y,
        sigma_energy_transfer_cm2=np.asarray(result.sigma_energy_transfer_cm2),
        work_from_incident_field_j=np.asarray(result.work_from_incident_field_j),
        # Schema-1 compatibility aliases.
        sigma_abs_cm2=np.asarray(result.sigma_energy_transfer_cm2),
        absorbed_energy_j=np.asarray(result.work_from_incident_field_j),
        fluence_j_cm2=np.asarray(result.fluence_j_cm2),
        peak_intensity_w_cm2=np.asarray(result.peak_intensity_w_cm2),
        max_bloch_radius=np.asarray(result.max_bloch_radius),
        min_density_eigenvalue=np.asarray(result.min_density_eigenvalue),
        solver_n_steps=np.asarray(result.diagnostics.n_steps),
        solver_nfev=np.asarray(result.diagnostics.nfev),
        solver_success=np.asarray(result.diagnostics.solver_success),
        t_final_reached=np.asarray(result.diagnostics.t_final_reached),
        solver_max_step_limit_au=np.asarray(result.diagnostics.max_step_limit_au),
        integration_frequency_ceiling_au=np.asarray(
            result.diagnostics.integration_frequency_ceiling_au
        ),
        incident_peak_rabi_frequency_au=np.asarray(
            result.diagnostics.incident_peak_rabi_frequency_au
        ),
        observed_peak_rabi_frequency_au=np.asarray(
            result.diagnostics.observed_peak_rabi_frequency_au
        ),
        rabi_step_refinement_count=np.asarray(
            result.diagnostics.rabi_step_refinement_count,
            dtype=np.int64,
        ),
        boundary_envelope_fraction=np.asarray(
            result.diagnostics.boundary_envelope_fraction
        ),
        response_tail_ratio=np.asarray(dynamics_tail_ratio),
        response_tail_converged=np.asarray(dynamics_tail_converged),
        pulse_spectral_fraction_in_fit_window=np.asarray(
            result.diagnostics.pulse_spectral_fraction_in_fit_window
        ),
        pulse_spectral_leakage=np.asarray(
            result.diagnostics.pulse_spectral_leakage
        ),
        mnp_drive_spectral_fraction_in_fit_window=np.asarray(
            result.diagnostics.mnp_drive_spectral_fraction_in_fit_window
        ),
        mnp_drive_spectral_leakage=np.asarray(
            result.diagnostics.mnp_drive_spectral_leakage
        ),
        mnp_dipole_spectral_fraction_in_fit_window=np.asarray(
            result.diagnostics.mnp_dipole_spectral_fraction_in_fit_window
        ),
        mnp_dipole_spectral_leakage=np.asarray(
            result.diagnostics.mnp_dipole_spectral_leakage
        ),
        work_passivity_checked=np.asarray(
            result.diagnostics.work_passivity_checked
        ),
        work_passivity_tolerance_au=np.asarray(
            result.diagnostics.work_passivity_tolerance_au
        ),
        work_nonnegative_within_tolerance=np.asarray(
            result.diagnostics.work_nonnegative_within_tolerance
        ),
    )
    write_json(run_dir / 'params.json', metadata)

    with np.load(data_path) as data:
        plot_fit_diagnostics_from_data(data, run_dir / 'fit_diagnostics.png', show=not args.no_show)
        plot_time_dynamics_from_data(data, run_dir / 'time_dynamics.png', show=not args.no_show)

    print(f'Wrote rational-fit artifact run to {run_dir}')
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Save reproducible QD-MNP rational-fit artifacts.')
    parser.add_argument('--run-dir', type=Path, default=None)
    parser.add_argument('--c-nm', type=float, default=None)
    parser.add_argument('--a-nm', type=float, default=None)
    parser.add_argument('--r-nm', type=float, default=None)
    parser.add_argument('--qd-radius-nm', type=float, default=None)
    parser.add_argument('--G', dest='g_factor', type=float, default=None)
    parser.add_argument('--eps-m', type=float, default=None)
    parser.add_argument('--eps-qd', type=float, default=None)
    parser.add_argument('--orientation', choices=['long', 'trans'], default='long')
    parser.add_argument(
        '--qd-dipole-convention',
        choices=['bare_internal', 'effective_external'],
        default='effective_external',
    )
    parser.add_argument('--d-debye', type=float, default=None)
    parser.add_argument('--omega0-ev', type=float, default=None)
    parser.add_argument('--gamma-population-mev', type=float, default=None)
    parser.add_argument(
        '--gamma2-coherence-mev',
        dest='gamma2_coherence_mev',
        metavar='GAMMA2_MEV',
        type=float,
        default=None,
        help='Total coherence HWHM hbar*Gamma2 in meV (not pure dephasing); requires Gamma2 >= gamma1/2.',
    )
    parser.add_argument(
        '--gamma-dephasing-mev',
        dest='gamma_dephasing_mev',
        metavar='GAMMA2_MEV',
        type=float,
        default=None,
        help='Deprecated alias for --gamma2-coherence-mev.',
    )
    parser.add_argument('--modes', nargs='+', type=int, default=[9])
    parser.add_argument('--fit-min-ev', type=float, default=0.8)
    parser.add_argument('--fit-max-ev', type=float, default=3.0)
    parser.add_argument('--weight-center-ev', type=float, default=None)
    parser.add_argument('--weight-sigma-ev', type=float, default=None)
    parser.add_argument('--energy-min-ev', type=float, default=0.8)
    parser.add_argument('--energy-max-ev', type=float, default=3.0)
    parser.add_argument('--points', type=int, default=800)
    parser.add_argument('--dynamics-n-modes', type=int, default=9)
    parser.add_argument('--pulse-e0-v-m', type=float, default=2.5e5)
    parser.add_argument('--omega-l-ev', type=float, default=2.042)
    parser.add_argument('--tau-fs', type=float, default=5.0)
    parser.add_argument('--method', choices=['Radau', 'BDF', 'LSODA', 'RK45', 'DOP853'], default='Radau')
    parser.add_argument('--rtol', type=float, default=1e-8)
    parser.add_argument('--atol', type=float, default=1e-10)
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    if (
        args.gamma2_coherence_mev is not None
        and args.gamma_dephasing_mev is not None
        and args.gamma2_coherence_mev != args.gamma_dephasing_mev
    ):
        parser.error('--gamma2-coherence-mev and --gamma-dephasing-mev must agree when both are supplied.')
    if args.gamma2_coherence_mev is None:
        args.gamma2_coherence_mev = args.gamma_dephasing_mev
    return args


if __name__ == '__main__':
    build_rational_fit_artifact(parse_args())
