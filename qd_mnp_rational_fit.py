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
from scipy.constants import c as C_SI, epsilon_0, e as E_CHARGE, physical_constants
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# ================================================================
# Atomic units
# ================================================================
AU_LENGTH_M = physical_constants['Bohr radius'][0]
AU_TIME_S = physical_constants['atomic unit of time'][0]
AU_ENERGY_J = physical_constants['Hartree energy'][0]
AU_ENERGY_EV = physical_constants['Hartree energy in eV'][0]
AU_FIELD_V_M = AU_ENERGY_J / (E_CHARGE * AU_LENGTH_M)
AU_DIPOLE_C_M = E_CHARGE * AU_LENGTH_M
DEBYE_C_M = 3.33564e-30
SCHEMA_VERSION = 2


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
    material: MaterialDispersion = field(default_factory=lambda: DEFAULT_AU_MATERIAL)

    @property
    def axial_surface_gap_au(self) -> float:
        """Surface-to-surface gap for a QD on the ellipsoid long axis."""
        return float(self.R_au - self.c_au - self.qd_radius_au)

    @property
    def pure_dephasing_au(self) -> float:
        """Pure-dephasing contribution implied by Gamma2=gamma1/2+gamma_phi."""
        return float(self.Gamma_au - 0.5 * self.gamma_au)


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
    min_imag_alpha_fit_window: float = 0.0
    passivity_grid_points: int = 0
    passive_on_fit_window: bool = True


@dataclass(frozen=True)
class DipoleCrossSections:
    """Dipole cross sections in the alpha_eff=mu/(eps_m E_inc) convention."""

    extinction_cm2: np.ndarray
    scattering_cm2: np.ndarray
    absorption_cm2: np.ndarray


@dataclass(frozen=True)
class LinearStabilityDiagnostics:
    """Poles of the full field-free Jacobian at the QD ground state."""

    poles_au: np.ndarray
    spectral_abscissa_au: float
    tolerance_au: float
    stable: bool


def dipole_cross_sections_cm2(
    alpha_eff_au: float | complex | np.ndarray,
    omega_au: float | np.ndarray,
    eps_m: float,
) -> DipoleCrossSections:
    """Return extinction, scattering and material absorption cross sections.

    No negative absorption is clipped.  It flags that the supplied effective
    dipole response does not satisfy the passive optical-theorem partition
    under these assumptions; possible causes include a missing radiative
    correction, nonlinear frequency conversion, a finite Fourier window or a
    poor rational approximation.
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
    extinction_m2 = (k_si / epsilon_0) * alpha_si.imag
    scattering_m2 = k_si**4 * np.abs(alpha_si) ** 2 / (6.0 * np.pi * epsilon_0**2)
    absorption_m2 = extinction_m2 - scattering_m2
    return DipoleCrossSections(
        extinction_cm2=np.asarray(extinction_m2 * 1e4),
        scattering_cm2=np.asarray(scattering_m2 * 1e4),
        absorption_cm2=np.asarray(absorption_m2 * 1e4),
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
    W_min: float
    W_max: float
    excited_population_min: float
    excited_population_max: float
    max_bloch_radius: float
    min_density_eigenvalue: float


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
        orientation: Literal['long', 'trans'] = 'long',
        n_modes: int = 3,
        fit_window_eV: tuple[float, float] = (0.8, 3.0),
        weight_center_eV: float | None = None,
        weight_sigma_eV: float | None = None,
        alpha_objective_weight: float = 1.0,
        inv_alpha_objective_weight: float = 1.0,
        seed: int = 12345,
        verbose: bool = True,
    ) -> None:
        if n_modes < 1:
            raise ValueError('n_modes must be >= 1')
        if orientation not in {'long', 'trans'}:
            raise ValueError("orientation must be either 'long' or 'trans'.")
        if len(fit_window_eV) != 2 or not np.all(np.isfinite(fit_window_eV)):
            raise ValueError('fit_window_eV must contain two finite values.')
        if fit_window_eV[0] <= 0.0 or fit_window_eV[1] <= fit_window_eV[0]:
            raise ValueError('fit_window_eV must satisfy 0 < min < max.')
        if weight_sigma_eV is not None and weight_sigma_eV <= 0.0:
            raise ValueError('weight_sigma_eV must be positive when specified.')
        if alpha_objective_weight < 0.0 or inv_alpha_objective_weight < 0.0:
            raise ValueError('Fit objective weights must be non-negative.')
        if alpha_objective_weight == 0.0 and inv_alpha_objective_weight == 0.0:
            raise ValueError('At least one fit objective weight must be positive.')

        self.params = params
        self.orientation = orientation
        self.n_modes = int(n_modes)
        self.fit_window_eV = fit_window_eV
        self.weight_center_eV = weight_center_eV
        self.weight_sigma_eV = weight_sigma_eV
        self.alpha_objective_weight = float(alpha_objective_weight)
        self.inv_alpha_objective_weight = float(inv_alpha_objective_weight)
        self.seed = int(seed)
        self.verbose = bool(verbose)
        self._validate_physical_parameters()

        self.L_long, self.L_trans = self._depolarization_factors()
        self.L = self.L_long if orientation == 'long' else self.L_trans
        self.C = self.params.eps_m * self.params.a_au**2 * self.params.c_au / 3.0
        self.J = self.params.G / (self.params.eps_m * self.params.R_au**3)

        self.energy_au = eV_to_au(self.params.material.energy_eV)
        self.alpha_tab = self._alpha_dimless(self.L)
        self.inv_alpha_tab = 1.0 / self.alpha_tab
        self.fit = self._fit_rational_alpha()
        self.linear_stability = self.assert_linearized_ground_state_stable()

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
        if p.eps_m <= 0.0:
            raise ValueError('The real host permittivity eps_m must be positive.')
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
        if np.isclose(c_au, a_au, rtol=1e-12, atol=0.0):
            return 1.0 / 3.0, 1.0 / 3.0
        xi2 = c_au**2 / (c_au**2 - a_au**2)
        xi = np.sqrt(xi2)
        log_term = np.log((xi + 1.0) / (xi - 1.0))
        L_long = (xi2 - 1.0) * (0.5 * xi * log_term - 1.0)
        L_trans = 0.5 * (1.0 - L_long)
        return float(L_long), float(L_trans)

    def _alpha_dimless(self, L: float) -> np.ndarray:
        eps = self.params.material.epsilon
        eps_m = self.params.eps_m
        return (eps - eps_m) / (eps_m + L * (eps - eps_m))

    def _fit_weights(self, energies_eV: np.ndarray) -> np.ndarray:
        if self.weight_center_eV is None or self.weight_sigma_eV is None:
            return np.ones_like(energies_eV, dtype=float)
        x = (energies_eV - self.weight_center_eV) / self.weight_sigma_eV
        return np.exp(-0.5 * x**2)

    # ------------------------------------------------------------
    # Stable rational fit for alpha(omega)
    # ------------------------------------------------------------
    def _alpha_model_from_params(self, omega_au: np.ndarray, u: np.ndarray) -> np.ndarray:
        alpha_inf = u[0]
        strengths = u[1 : 1 + self.n_modes]
        omega_modes = np.exp(u[1 + self.n_modes : 1 + 2 * self.n_modes])
        gamma_modes = np.exp(u[1 + 2 * self.n_modes : 1 + 3 * self.n_modes])

        alpha = np.full_like(omega_au, fill_value=alpha_inf, dtype=complex)
        for f_k, w_k, g_k in zip(strengths, omega_modes, gamma_modes):
            denom = (w_k**2 - omega_au**2) - 1j * g_k * omega_au
            alpha += f_k / denom
        return alpha

    def _fit_rational_alpha(self) -> RationalLorentzFit:
        e_min, e_max = self.fit_window_eV
        mask = (self.params.material.energy_eV >= e_min) & (self.params.material.energy_eV <= e_max)
        if np.count_nonzero(mask) < max(5, 2 * self.n_modes + 2):
            raise ValueError('Too few tabulated points inside fit_window_eV for the requested n_modes.')

        omega = self.energy_au[mask]
        energies = self.params.material.energy_eV[mask]
        alpha_true = self.alpha_tab[mask]
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

        rng = np.random.default_rng(self.seed)
        passivity_omega = np.asarray(
            eV_to_au(np.linspace(e_min, e_max, max(1024, 256 * self.n_modes))),
            dtype=float,
        )
        passivity_scale = max(float(np.max(np.abs(alpha_true.imag))), 1.0)
        passivity_tol = 1e-9 * passivity_scale

        alpha_inf_lo = alpha_med - 5.0 * alpha_scale
        alpha_inf_hi = alpha_med + 5.0 * alpha_scale
        strength_lo = np.full(self.n_modes, -100.0 * strength_scale)
        strength_hi = np.full(self.n_modes, +100.0 * strength_scale)
        omega_lo = np.full(self.n_modes, max(0.35 * omega_min, 1e-5))
        omega_hi = np.full(self.n_modes, max(2.50 * omega_max, 2.0 * omega_min))
        gamma_lo = np.full(self.n_modes, max(1e-4, 0.01 * omega_min))
        gamma_hi = np.full(self.n_modes, max(2.0 * omega_max, 0.20))

        lower = np.concatenate([
            [alpha_inf_lo],
            strength_lo,
            np.log(omega_lo),
            np.log(gamma_lo),
        ])
        upper = np.concatenate([
            [alpha_inf_hi],
            strength_hi,
            np.log(omega_hi),
            np.log(gamma_hi),
        ])

        def residual(u: np.ndarray) -> np.ndarray:
            alpha_fit = self._alpha_model_from_params(omega, u)
            inv_fit = 1.0 / alpha_fit

            r_alpha = alpha_fit - alpha_true
            r_inv = inv_fit - inv_true

            parts = [
                np.sqrt(self.alpha_objective_weight) * weights * r_alpha.real / scale_alpha_re,
                np.sqrt(self.alpha_objective_weight) * weights * r_alpha.imag / scale_alpha_im,
                np.sqrt(self.inv_alpha_objective_weight) * weights * r_inv.real / scale_inv_re,
                np.sqrt(self.inv_alpha_objective_weight) * weights * r_inv.imag / scale_inv_im,
            ]
            return np.concatenate(parts)

        def build_start(alpha_inf_guess: float, mode_shift: float, gamma_factor: float, sign_flip: bool = False) -> np.ndarray:
            centers = np.linspace(-0.5, 0.5, self.n_modes)
            w_guess = omega_peak + mode_shift * omega_span * centers
            w_guess = np.clip(w_guess, omega_lo * 1.02, omega_hi / 1.02)
            g_guess = np.clip(gamma_factor * np.maximum(w_guess, 0.05 * omega_peak), gamma_lo * 1.05, gamma_hi / 1.05)

            strengths = []
            for idx, w0 in enumerate(w_guess):
                sign = -1.0 if (sign_flip and idx % 2 == 1) else 1.0
                strengths.append(sign * strength_scale * (1.0 + 0.35 * idx))

            return np.concatenate([
                [alpha_inf_guess],
                np.asarray(strengths, dtype=float),
                np.log(w_guess),
                np.log(g_guess),
            ])

        starts: list[np.ndarray] = [
            build_start(alpha_med, 0.35, 0.10, False),
            build_start(alpha_med, 0.55, 0.18, False),
            build_start(alpha_med, 0.70, 0.28, True),
            build_start(alpha_med + 0.5 * alpha_scale, 0.30, 0.07, False),
            build_start(alpha_med - 0.5 * alpha_scale, 0.85, 0.35, True),
        ]

        extra_starts = 8 + 4 * self.n_modes
        for _ in range(extra_starts):
            u0 = np.empty(1 + 3 * self.n_modes, dtype=float)
            u0[0] = rng.uniform(alpha_inf_lo, alpha_inf_hi)
            u0[1 : 1 + self.n_modes] = rng.uniform(strength_lo, strength_hi)
            u0[1 + self.n_modes : 1 + 2 * self.n_modes] = rng.uniform(np.log(omega_lo), np.log(omega_hi))
            u0[1 + 2 * self.n_modes : 1 + 3 * self.n_modes] = rng.uniform(np.log(gamma_lo), np.log(gamma_hi))
            starts.append(u0)

        best = None
        for u0 in starts:
            u0 = np.clip(u0, lower, upper)
            res = least_squares(
                residual,
                x0=u0,
                bounds=(lower, upper),
                method='trf',
                loss='soft_l1',
                f_scale=0.3,
                max_nfev=6000,
                ftol=1e-11,
                xtol=1e-11,
                gtol=1e-11,
            )
            if not res.success:
                continue

            u = res.x.copy()
            omega_modes = np.exp(u[1 + self.n_modes : 1 + 2 * self.n_modes])
            order = np.argsort(omega_modes)
            u[1 : 1 + self.n_modes] = u[1 : 1 + self.n_modes][order]
            u[1 + self.n_modes : 1 + 2 * self.n_modes] = u[1 + self.n_modes : 1 + 2 * self.n_modes][order]
            u[1 + 2 * self.n_modes : 1 + 3 * self.n_modes] = u[1 + 2 * self.n_modes : 1 + 3 * self.n_modes][order]

            alpha_fit = self._alpha_model_from_params(omega, u)
            inv_fit = 1.0 / alpha_fit
            alpha_passivity = self._alpha_model_from_params(passivity_omega, u)
            min_imag_alpha = float(np.min(alpha_passivity.imag))
            if min_imag_alpha < -passivity_tol:
                continue
            rms_alpha = float(np.sqrt(np.mean(np.abs(alpha_fit - alpha_true) ** 2)))
            rms_inv = float(np.sqrt(np.mean(np.abs(inv_fit - inv_true) ** 2)))
            score = float(np.sqrt(np.mean(residual(u) ** 2)))

            if best is None or score < best['score']:
                best = {
                    'u': u,
                    'score': score,
                    'rms_alpha': rms_alpha,
                    'rms_inv': rms_inv,
                    'min_imag_alpha': min_imag_alpha,
                }

        if best is None:
            raise RuntimeError('A stable and passive rational fit was not found on fit_window_eV.')

        u = best['u']
        alpha_inf = float(u[0])
        strengths = np.asarray(u[1 : 1 + self.n_modes], dtype=float)
        omega_modes = np.exp(u[1 + self.n_modes : 1 + 2 * self.n_modes])
        gamma_modes = np.exp(u[1 + 2 * self.n_modes : 1 + 3 * self.n_modes])

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
            min_imag_alpha_fit_window=float(best['min_imag_alpha']),
            passivity_grid_points=int(passivity_omega.size),
            passive_on_fit_window=True,
        )

    # ------------------------------------------------------------
    # Frequency-domain reconstruction
    # ------------------------------------------------------------
    def alpha_from_fit(self, energies_eV: np.ndarray) -> np.ndarray:
        omega = eV_to_au(np.asarray(energies_eV, dtype=float))
        u = np.concatenate([
            [self.fit.alpha_inf],
            self.fit.strengths_au2,
            np.log(self.fit.omega_modes_au),
            np.log(self.fit.gamma_modes_au),
        ])
        return self._alpha_model_from_params(omega, u)

    def inv_alpha_from_fit(self, energies_eV: np.ndarray) -> np.ndarray:
        return 1.0 / self.alpha_from_fit(energies_eV)

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
        print(f'weighted score    : {self.fit.cost:.6e}')
        print(f'min Im[alpha]     : {self.fit.min_imag_alpha_fit_window:.6e} (fit window)')
        print(f'passivity grid    : {self.fit.passivity_grid_points} points')
        print(f'passive in window : {self.fit.passive_on_fit_window}')
        print(f'max Re[pole]      : {self.linear_stability.spectral_abscissa_au:.6e} au')
        print(f'coupled stable    : {self.linear_stability.stable}')
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

    def default_time_span(self, pulse: GaussianPulse, n_sigma: float = 10.0) -> tuple[float, float]:
        sigma = pulse.sigma_t_au
        return -0.5 * n_sigma * sigma, n_sigma * sigma

    def recommended_post_pulse_time_au(self, decay_times: float = 8.0) -> float:
        """Tail duration needed for coherent QD/MNP dipoles to decay.

        Lorentz oscillator amplitudes decay as exp(-gamma_k*t/2), whereas QD
        coherence decays as exp(-Gamma2*t). The population lifetime is not used
        because population alone has no optical dipole after the field is off.
        """
        if not np.isfinite(decay_times) or decay_times <= 0.0:
            raise ValueError('decay_times must be finite and positive.')
        rates = [float(self.params.Gamma_au)]
        rates.extend(float(gamma) / 2.0 for gamma in self.fit.gamma_modes_au)
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
        scalars = np.asarray([d, omega0, gamma1, gamma2, coupling_factor], dtype=float)
        if np.any(~np.isfinite(scalars)):
            raise ValueError('Linear-stability parameters must be finite.')
        if d < 0.0 or omega0 <= 0.0 or gamma1 < 0.0 or gamma2 < 0.0:
            raise ValueError('Linear-stability dipole/frequency/rates must be physically non-negative.')
        if gamma2 < 0.5 * gamma1:
            raise ValueError('Linear-stability check requires Gamma2 >= gamma1/2.')

        n = self.n_modes
        size = 2 * n + 3
        W_index, Q_index, P_index = 2 * n, 2 * n + 1, 2 * n + 2
        jacobian = np.zeros((size, size), dtype=float)
        J = coupling_factor / (p.eps_m * p.R_au**3)

        for k, (strength, mode_omega, mode_gamma) in enumerate(
            zip(self.fit.strengths_au2, self.fit.omega_modes_au, self.fit.gamma_modes_au)
        ):
            q_index = 2 * k
            v_index = q_index + 1
            jacobian[q_index, v_index] = 1.0
            jacobian[v_index, q_index] = -(mode_omega**2)
            jacobian[v_index, v_index] = -mode_gamma
            jacobian[v_index, P_index] = strength * J * d

        jacobian[W_index, W_index] = -gamma1
        jacobian[Q_index, 0 : 2 * n : 2] = 2.0 * d * J * self.C
        jacobian[Q_index, Q_index] = -gamma2
        jacobian[Q_index, P_index] = (
            -omega0 + 2.0 * d**2 * J**2 * self.C * self.fit.alpha_inf
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
        mu_d = p.d_au * P
        F = E + self.J * mu_d

        mu_p = self.C * (self.fit.alpha_inf * F + np.sum(q))
        E_eff_qd = E + self.J * mu_p
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
    ) -> HybridSolveResult:
        # Recheck here as callers can replace params after model construction.
        self._validate_physical_parameters()
        self.assert_linearized_ground_state_stable()
        if positivity_policy not in {'raise', 'warn', 'ignore'}:
            raise ValueError("positivity_policy must be 'raise', 'warn' or 'ignore'.")
        if not np.isfinite(positivity_tol) or positivity_tol < 0.0:
            raise ValueError('positivity_tol must be finite and non-negative.')
        if t_span_au is None:
            t_span_au = self.default_time_span(pulse)
        if (
            len(t_span_au) != 2
            or not np.all(np.isfinite(t_span_au))
            or t_span_au[1] <= t_span_au[0]
        ):
            raise ValueError('t_span_au must contain finite values with t_end > t_start.')

        if max_step_au is None:
            carrier_period = 2.0 * np.pi / pulse.omegaL_au
            max_step_au = 0.10 * carrier_period
        if not np.isfinite(max_step_au) or max_step_au <= 0.0:
            raise ValueError('max_step_au must be finite and positive.')

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
        state_is_finite = bool(np.all(np.isfinite(t)) and np.all(np.isfinite(y)))
        if not state_is_finite:
            raise RuntimeError('solve_ivp returned non-finite times or state values.')
        if len(t) < 2 or np.any(np.diff(t) <= 0.0):
            raise RuntimeError('solve_ivp returned an invalid or non-monotone time grid.')
        end_scale = max(1.0, abs(float(t_span_au[1])))
        t_final_reached = bool(abs(float(t[-1]) - float(t_span_au[1])) <= 1e-10 * end_scale)
        if not t_final_reached:
            raise RuntimeError('solve_ivp reported success but did not reach the requested final time.')

        p = self.params
        q = y[: 2 * self.n_modes : 2]
        v = y[1 : 2 * self.n_modes : 2]
        W, Q, P = y[2 * self.n_modes : 2 * self.n_modes + 3]
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

        E = pulse.field(t)
        E_dot = pulse.field_dot(t)
        mu_d = p.d_au * P
        mu_d_dot = p.d_au * (p.omega0_au * Q - p.Gamma_au * P)
        F = E + self.J * mu_d
        F_dot = E_dot + self.J * mu_d_dot

        mu_p = self.C * (self.fit.alpha_inf * F + np.sum(q, axis=0))
        mu_p_dot = self.C * (self.fit.alpha_inf * F_dot + np.sum(v, axis=0))
        mu_total = mu_p + mu_d
        mu_total_dot = mu_p_dot + mu_d_dot

        work_from_incident_field_au = np.trapezoid(mu_total_dot * E, t)
        work_from_incident_field_j = float(work_from_incident_field_au * AU_ENERGY_J)
        fluence_j_cm2 = float(pulse.fluence_j_cm2(eps_m=p.eps_m))
        sigma_energy_transfer_cm2 = work_from_incident_field_j / fluence_j_cm2
        reconstructed = (mu_p, mu_d, mu_total, mu_total_dot)
        if not all(np.all(np.isfinite(values)) for values in reconstructed):
            raise RuntimeError('Non-finite reconstructed dipole or dipole derivative.')
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
            W_min=float(np.min(W)),
            W_max=float(np.max(W)),
            excited_population_min=float(np.min(0.5 * (W + 1.0))),
            excited_population_max=float(np.max(0.5 * (W + 1.0))),
            max_bloch_radius=max_bloch_radius,
            min_density_eigenvalue=min_density_eigenvalue,
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
def make_default_params() -> HybridSystemParams:
    return HybridSystemParams(
        c_au=float(nm_to_au(15.0)),
        a_au=float(nm_to_au(7.0)),
        R_au=float(nm_to_au(18.0)),
        G=2.0, eps_m=1.0,
        d_au=float(dipole_si_to_au(7.5e-29)),
        omega0_au=float(eV_to_au(2.042)),
        gamma_au=float(1.0 / ns_to_au(30.0)),
        Gamma_au=float(1.0 / fs_to_au(330.0)),
        qd_radius_au=float(nm_to_au(2.0)),
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
) -> HybridSystemParams:
    params = make_default_params()
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
        updates['G'] = float(g_factor)
    if eps_m is not None:
        updates['eps_m'] = float(eps_m)
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


def params_to_physical_dict(params: HybridSystemParams, orientation: str = 'long') -> dict[str, float | str]:
    return {
        'c_nm': float(au_to_nm(params.c_au)),
        'a_nm': float(au_to_nm(params.a_au)),
        'R_nm': float(au_to_nm(params.R_au)),
        'qd_radius_nm': float(au_to_nm(params.qd_radius_au)),
        'surface_gap_nm': float(au_to_nm(params.axial_surface_gap_au)),
        'G': float(params.G),
        'eps_m': float(params.eps_m),
        'd_debye': float(dipole_au_to_debye(params.d_au)),
        'omega0_ev': float(au_to_eV(params.omega0_au)),
        'gamma_population_mev': float(au_to_eV(params.gamma_au) * 1000.0),
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

    orientation = 'long'
    params = make_params_with_overrides(
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
    fit_window_ev = (args.fit_min_ev, args.fit_max_ev)
    energy_plot = np.linspace(args.energy_min_ev, args.energy_max_ev, args.points)
    modes_list = np.asarray(args.modes, dtype=int)

    inv_alpha_fit_by_mode = []
    alpha_fit_by_mode = []
    rms_alpha_by_mode = []
    rms_inv_alpha_by_mode = []
    fit_cost_by_mode = []
    fit_min_imag_alpha_by_mode = []
    fit_passivity_grid_points_by_mode = []
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
            verbose=True,
        )
        if inv_alpha_table is None:
            inv_alpha_table = model.inv_alpha_tab.copy()
        alpha_fit = model.alpha_from_fit(energy_plot)
        alpha_fit_by_mode.append(alpha_fit)
        inv_alpha_fit_by_mode.append(1.0 / alpha_fit)
        rms_alpha_by_mode.append(model.fit.rms_alpha)
        rms_inv_alpha_by_mode.append(model.fit.rms_inv_alpha)
        fit_cost_by_mode.append(model.fit.cost)
        fit_min_imag_alpha_by_mode.append(model.fit.min_imag_alpha_fit_window)
        fit_passivity_grid_points_by_mode.append(model.fit.passivity_grid_points)

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
    pulse = GaussianPulse(
        E0_au=float(field_si_to_au(args.pulse_e0_v_m)),
        omegaL_au=float(eV_to_au(args.omega_l_ev)),
        tau_au=float(fs_to_au(args.tau_fs)),
        tau_kind='fwhm_intensity',
    )
    result = dynamics_model.solve(pulse, method=args.method, rtol=args.rtol, atol=args.atol)

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
            'passivity_requirement': (
                'Im(alpha_fit) >= -tol on a finite dense grid over fit_window_ev; '
                'tol = 1e-9 * max(max(abs(Im(alpha_table))), 1)'
            ),
            'passivity_grid_points_by_mode': [int(x) for x in fit_passivity_grid_points_by_mode],
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
        },
        'solver': {
            'method': args.method,
            'rtol': float(args.rtol),
            'atol': float(args.atol),
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
        fit_cost_by_mode=np.asarray(fit_cost_by_mode, dtype=float),
        fit_min_imag_alpha_by_mode=np.asarray(fit_min_imag_alpha_by_mode, dtype=float),
        fit_passivity_grid_points_by_mode=np.asarray(fit_passivity_grid_points_by_mode, dtype=np.int64),
        linearized_ground_state_poles_au=dynamics_model.linear_stability.poles_au,
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
    parser.add_argument('--modes', nargs='+', type=int, default=[1, 2, 3, 4])
    parser.add_argument('--fit-min-ev', type=float, default=0.8)
    parser.add_argument('--fit-max-ev', type=float, default=3.0)
    parser.add_argument('--weight-center-ev', type=float, default=2.35)
    parser.add_argument('--weight-sigma-ev', type=float, default=0.30)
    parser.add_argument('--energy-min-ev', type=float, default=0.8)
    parser.add_argument('--energy-max-ev', type=float, default=3.0)
    parser.add_argument('--points', type=int, default=800)
    parser.add_argument('--dynamics-n-modes', type=int, default=4)
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
