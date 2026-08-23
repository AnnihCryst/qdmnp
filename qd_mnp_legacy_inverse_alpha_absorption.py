"""Архивная неудачная реализация импульсного поглощения КТ-МНЧ.

Этот файл оставлен для истории и сравнения с новой реализацией. Здесь
динамика металлической наночастицы строится через полиномиальную аппроксимацию
обратной поляризуемости ``1/alpha(omega)`` и последующее высокопорядковое
дифференциальное уравнение для диполя МНЧ.

Идея соответствует ранней попытке напрямую реализовать уравнение из заготовки
статьи, но на практике этот путь оказался численно и физически неудобным:
подгонка ``1/alpha`` плохо контролирует устойчивость временного оператора,
возникают жесткость и чувствительность к порядку полинома, а расчет сечения
поглощения для коротких импульсов получается менее надежным.

Для текущей работы используй ``qd_mnp_rational_fit.py`` и запускаемые
скрипты ``qd_mnp_linear_spectrum.py``, ``qd_mnp_fano_scan.py`` и
``qd_mnp_pulse_absorption_sweep.py``. Этот модуль можно запускать только как
архивный диагностический пример, не как основную расчетную схему.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, Literal
import numpy as np
from numpy.polynomial import Polynomial
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.constants import c as C_SI, epsilon_0, e as E_CHARGE, physical_constants
#==================================================
from scipy.optimize import least_squares
#==================================================

# ----------------------------
# Atomic-unit conversion helpers
# ----------------------------
AU_LENGTH_M = physical_constants['Bohr radius'][0]
AU_TIME_S = physical_constants['atomic unit of time'][0]
AU_ENERGY_J = physical_constants['Hartree energy'][0]
AU_ENERGY_EV = physical_constants['Hartree energy in eV'][0]
AU_FIELD_V_M = AU_ENERGY_J / (E_CHARGE * AU_LENGTH_M)
AU_DIPOLE_C_M = E_CHARGE * AU_LENGTH_M
AU_AREA_CM2 = (AU_LENGTH_M * 100.0) ** 2


def eV_to_au(value: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(value) / AU_ENERGY_EV


def au_to_eV(value: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(value) * AU_ENERGY_EV


def nm_to_au(value: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(value) * 1e-9 / AU_LENGTH_M


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


# ----------------------------
# Input data
# ----------------------------
@dataclass(frozen=True)
class MaterialDispersion:
    energy_eV: np.ndarray
    n: np.ndarray
    k: np.ndarray

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


@dataclass(frozen=True)
class HybridSystemParams:
    # Geometry (atomic units)
    c_au: float
    a_au: float
    R_au: float
    G: float
    eps_m: float

    # QD (atomic units)
    d_au: float
    omega0_au: float
    gamma_au: float
    Gamma_au: float

    material: MaterialDispersion = field(default_factory=lambda: DEFAULT_AU_MATERIAL)


@dataclass(frozen=True)
class GaussianPulse:
    E0_au: float
    omegaL_au: float
    tau_au: float
    tau_kind: Literal['sigma', 'fwhm_intensity'] = 'fwhm_intensity'

    @property
    def sigma_t_au(self) -> float:
        if self.tau_kind == 'sigma':
            return float(self.tau_au)
        if self.tau_kind == 'fwhm_intensity':
            return float(self.tau_au) / (2.0 * np.sqrt(np.log(2.0)))
        raise ValueError(f'Unsupported tau_kind={self.tau_kind!r}')

    def field(self, t_au: float | np.ndarray) -> np.ndarray:
        t = np.asarray(t_au)
        sigma = self.sigma_t_au
        return self.E0_au * np.exp(-0.5 * (t / sigma) ** 2) * np.cos(self.omegaL_au * t)

    def peak_intensity_w_cm2(self, cycle_averaged: bool = True) -> float:
        E0_si = float(field_au_to_si(self.E0_au))
        prefactor = 0.5 if cycle_averaged else 1.0
        return prefactor * epsilon_0 * C_SI * E0_si**2 * 1e-4

    def fluence_j_cm2(self) -> float:
        # Exact integral of the real field squared for Eq. (18).
        E0_si = float(field_au_to_si(self.E0_au))
        sigma_s = self.sigma_t_au * AU_TIME_S
        osc = np.exp(-(self.omegaL_au * self.sigma_t_au) ** 2)
        integral_E2 = 0.5 * np.sqrt(np.pi) * sigma_s * E0_si**2 * (1.0 + osc)
        fluence_j_m2 = epsilon_0 * C_SI * integral_E2
        return fluence_j_m2 * 1e-4


@dataclass(frozen=True)
class FitResult:
    coeffs_time: np.ndarray
    coeffs_re_freq: np.ndarray
    coeffs_im_freq: np.ndarray
    energies_used_eV: np.ndarray
    inv_alpha_used: np.ndarray
    order: int


@dataclass(frozen=True)
class HybridSolveResult:
    t_au: np.ndarray
    y: np.ndarray
    mu_p_au: np.ndarray
    mu_d_au: np.ndarray
    mu_total_au: np.ndarray
    mu_dot_total_au: np.ndarray
    sigma_abs_cm2: float
    absorbed_energy_j: float
    fluence_j_cm2: float
    peak_intensity_w_cm2: float
    solve_ivp_result: object

#===============================================================================================
# Проверка коэффициентов на критерии устойчивости. Быстрая диагностика (условие Роута–Гурвица):
def print_stability_info(coeffs_time: np.ndarray):
    print("\nВызов функции print_stability_info():")
    roots = np.roots(coeffs_time[::-1])
    print("coeffs_time =", coeffs_time)
    print("roots =", roots)

    N = len(coeffs_time) - 1

    # Общий численный критерий: все корни должны иметь Re(r) < 0
    stable_by_roots = np.all(np.real(roots) < 0.0)
    print("Stable by roots:", stable_by_roots)

    if N == 2:
        a0, a1, a2 = coeffs_time
        if a2 < 0:
            a0, a1, a2 = -a0, -a1, -a2

        cond1 = a2 > 0
        cond2 = a1 > 0
        cond3 = a0 > 0

        print("Quadratic Routh-Hurwitz:")
        print("a0, a1, a2 =", a0, a1, a2)
        print("a2 > 0 :", cond1)
        print("a1 > 0 :", cond2)
        print("a0 > 0 :", cond3)
        print("Stable by Routh-Hurwitz:", cond1 and cond2 and cond3)

    elif N == 3:
        a0, a1, a2, a3 = coeffs_time
        if a3 < 0:
            a0, a1, a2, a3 = -a0, -a1, -a2, -a3

        cond1 = a3 > 0
        cond2 = a2 > 0
        cond3 = a1 > 0
        cond4 = a0 > 0
        cond5 = a2 * a1 > a3 * a0

        print("Cubic Routh-Hurwitz:")
        print("a0, a1, a2, a3 =", a0, a1, a2, a3)
        print("a3 > 0 :", cond1)
        print("a2 > 0 :", cond2)
        print("a1 > 0 :", cond3)
        print("a0 > 0 :", cond4)
        print("a2*a1 > a3*a0 :", cond5)
        print("Stable by Routh-Hurwitz:", cond1 and cond2 and cond3 and cond4 and cond5)

    elif N == 4:
        a0, a1, a2, a3, a4 = coeffs_time
        if a4 < 0:
            a0, a1, a2, a3, a4 = -a0, -a1, -a2, -a3, -a4

        cond1 = a4 > 0
        cond2 = a3 > 0
        cond3 = a2 > 0
        cond4 = a1 > 0
        cond5 = a0 > 0
        cond6 = a3 * a2 > a4 * a1
        cond7 = a3 * a2 * a1 > a4 * a1 * a1 + a3 * a3 * a0

        print("Quartic Routh-Hurwitz:")
        print("a0, a1, a2, a3, a4 =", a0, a1, a2, a3, a4)
        print("a4 > 0 :", cond1)
        print("a3 > 0 :", cond2)
        print("a2 > 0 :", cond3)
        print("a1 > 0 :", cond4)
        print("a0 > 0 :", cond5)
        print("a3*a2 > a4*a1 :", cond6)
        print("a3*a2*a1 > a4*a1*a1 + a3*a3*a0 :", cond7)
        print("Stable by Routh-Hurwitz:", cond1 and cond2 and cond3 and cond4 and cond5 and cond6 and cond7)

    else:
        print("Routh-Hurwitz report is implemented only for N=2,3,4.")

    print("-" * 70)
#===============================================================================================


class HybridQDPlasmonModel:
    def __init__(
        self,
        params: HybridSystemParams,
        *,
        orientation: Literal['long', 'trans'] = 'long',
        operator_order: int = 4,
        fit_window_eV: tuple[float, float] = (1.0, 3.0),
        weight_center_eV: float | None = 2.05,
        weight_sigma_eV: float | None = 0.35,
        #===================================================
        enforce_stability: bool = True,
        verbose_stability: bool = True
        #===================================================
    ) -> None:
        self.params = params
        self.orientation = orientation
        self.operator_order = int(operator_order)
        self.fit_window_eV = fit_window_eV
        self.weight_center_eV = weight_center_eV
        self.weight_sigma_eV = weight_sigma_eV

        self.L_long, self.L_trans = self._depolarization_factors()
        self.L = self.L_long if orientation == 'long' else self.L_trans
        self.C = self.params.eps_m * self.params.a_au**2 * self.params.c_au / 3.0
        self.J = self.params.G / (self.params.eps_m * self.params.R_au**3)

        self.energy_au = eV_to_au(self.params.material.energy_eV)
        self.inv_alpha = 1.0 / self._alpha_dimless(self.L)
        self.fit = self._fit_operator_coeffs()
        #============================================================================
        self.enforce_stability = enforce_stability
        self.verbose_stability = verbose_stability
        if self.verbose_stability:
            print_stability_info(self.fit.coeffs_time)
        if self.enforce_stability:
            self._check_operator_stability(self.fit.coeffs_time)
        #============================================================================
        self.jac_sparsity = self._build_jac_sparsity()

    def _depolarization_factors(self) -> tuple[float, float]:
        c_au = self.params.c_au
        a_au = self.params.a_au
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

    def _fit_weights(self, energies_eV: np.ndarray) -> np.ndarray | None:
        if self.weight_center_eV is None or self.weight_sigma_eV is None:
            return None
        return np.exp(-0.5 * ((energies_eV - self.weight_center_eV) / self.weight_sigma_eV) ** 2)

    #=================================================================================================
    def _fit_operator_coeffs(self) -> FitResult:
        N = self.operator_order
        e_min, e_max = self.fit_window_eV
        mask = (self.params.material.energy_eV >= e_min) & (self.params.material.energy_eV <= e_max)

        if np.count_nonzero(mask) < 3:
            raise ValueError('Too few material-data points inside fit_window_eV.')

        if N not in (2, 3, 4, 5, 6):
            raise NotImplementedError('Stable factorized fit is implemented only for N=2,3,4,5,6.')

        omega = self.energy_au[mask]
        inv_alpha = self.inv_alpha[mask]
        weights = self._fit_weights(self.params.material.energy_eV[mask])

        coeffs_time = self._stable_factorized_fit(omega, inv_alpha, weights, N)

        # Оставляем обратный перевод для совместимости со структурой FitResult
        coeffs_re_freq = np.zeros(N // 2 + 1, dtype=float)
        coeffs_im_freq = np.zeros((N + 1) // 2, dtype=float)

        for n, a in enumerate(coeffs_time):
            if n % 2 == 0:
                m = n // 2
                coeffs_re_freq[m] = ((-1) ** m) * a
            else:
                m = (n - 1) // 2
                coeffs_im_freq[m] = ((-1) ** (m + 1)) * a

        return FitResult(
            coeffs_time=coeffs_time,
            coeffs_re_freq=coeffs_re_freq,
            coeffs_im_freq=coeffs_im_freq,
            energies_used_eV=self.params.material.energy_eV[mask].copy(),
            inv_alpha_used=inv_alpha.copy(),
            order=N
        )

    def _check_operator_stability(self, coeffs_time: np.ndarray, tol: float = 1e-12) -> None:
        # coeffs_time = [a0, a1, ..., aN]
        # характеристический полином: aN*r^N + ... + a1*r + a0 = 0
        # np.roots ожидает коэффициенты в порядке убывания степеней:
        # [aN, aN-1, ..., a1, a0]
        print("\nВызов функции _check_operator_stability():")
        roots = np.roots(coeffs_time[::-1])
        bad = roots.real > tol
        if np.any(bad):
            raise ValueError("Unstable time-domain operator: roots with Re(r) > 0 found.\n"f"Roots: {roots}")
    

    def reconstructed_inv_alpha(self, energies_eV: np.ndarray) -> np.ndarray:
        omega = eV_to_au(np.asarray(energies_eV, dtype=float))
        inv_alpha_fit = np.zeros_like(omega, dtype=complex)

        for n, a_n in enumerate(self.fit.coeffs_time):
            inv_alpha_fit += a_n * (-1j * omega) ** n

        return inv_alpha_fit
    #=================================================================================================

    def _build_jac_sparsity(self) -> np.ndarray:
        N = self.operator_order
        n = N + 3  # [muP, muP', ..., muP^(N-1), W, Q, P]
        S = np.zeros((n, n), dtype=bool)
        for i in range(N - 1):
            S[i, i + 1] = True
        S[N - 1, :N] = True
        S[N - 1, N + 2] = True
        S[N, 0] = True
        S[N, N] = True
        S[N, N + 1] = True
        S[N + 1, 0] = True
        S[N + 1, N] = True
        S[N + 1, N + 1] = True
        S[N + 1, N + 2] = True
        S[N + 2, N + 1] = True
        S[N + 2, N + 2] = True
        return S

    def rhs(self, t_au: float, y: np.ndarray, pulse: GaussianPulse) -> np.ndarray:
        p = self.params
        N = self.operator_order
        a = self.fit.coeffs_time

        mu_chain = y[:N]
        W, Q, P = y[N : N + 3]

        mu_p = mu_chain[0]
        mu_d = p.d_au * P
        E = pulse.field(t_au)

        forcing_mnp = self.C * (E + self.J * mu_d)
        lower_terms = float(np.dot(a[:-1], mu_chain))
        mu_p_N = (forcing_mnp - lower_terms) / a[-1]

        E_eff_qd = E + self.J * mu_p
        Omega = 2.0 * p.d_au * E_eff_qd

        dydt = np.empty_like(y)
        dydt[: N - 1] = mu_chain[1:]
        dydt[N - 1] = mu_p_N
        dydt[N] = Omega * Q - p.gamma_au * (W + 1.0)
        dydt[N + 1] = -p.omega0_au * P - Omega * W - p.Gamma_au * Q
        dydt[N + 2] = p.omega0_au * Q - p.Gamma_au * P
        return dydt

    def initial_state(self) -> np.ndarray:
        y0 = np.zeros(self.operator_order + 3, dtype=float)
        y0[self.operator_order] = -1.0  # W(t0) = -1
        return y0

    def default_time_span(self, pulse: GaussianPulse, n_sigma: float = 5.0) -> tuple[float, float]:
        sigma = pulse.sigma_t_au
        return -0.5*n_sigma * sigma, n_sigma * sigma

    def solve(self, pulse: GaussianPulse, *,
        #method: Literal['BDF', 'Radau', 'LSODA'] = 'BDF',
        method: Literal['BDF', 'Radau', 'LSODA', 'RK45', 'DOP853'] = 'RK45',
        rtol: float = 1e-7,
        atol: float = 1e-9,
        max_step_au: float | None = None,
        t_span_au: tuple[float, float] | None = None) -> HybridSolveResult:
        if self.operator_order < 2:
            raise ValueError('Use operator_order >= 2 so that mu_p_dot is available directly from the state vector.')

        if t_span_au is None:
            t_span_au = self.default_time_span(pulse)

        if max_step_au is None:
            carrier_period = 2.0 * np.pi / pulse.omegaL_au
            max_step_au = 0.15 * carrier_period

        # if method in ('RK45', 'DOP853'):
        #     sol = solve_ivp(
        #         fun=lambda t, y: self.rhs(t, y, pulse),
        #         t_span=t_span_au,
        #         y0=self.initial_state(),
        #         method=method,
        #         rtol=rtol,
        #         atol=atol,
        #         max_step=max_step_au,
        #         dense_output=False,
        #     )
        # else:
        #     sol = solve_ivp(
        #         fun=lambda t, y: self.rhs(t, y, pulse),
        #         t_span=t_span_au,
        #         y0=self.initial_state(),
        #         method=method,
        #         rtol=rtol,
        #         atol=atol,
        #         max_step=max_step_au,
        #         jac=None if method == 'LSODA' else (lambda t, y: self.jac(t, y, pulse)),
        #         jac_sparsity=None if method == 'LSODA' else self.jac_sparsity,
        #         dense_output=False,
        #     )
        # if not sol.success:
        #     raise RuntimeError(sol.message)

        sol = solve_ivp(
        fun=lambda t, y: self.rhs(t, y, pulse),
        t_span=t_span_au,
        y0=self.initial_state(),
        method='Radau',
        rtol=1e-8,
        atol=1e-10,
        max_step=max_step_au,
        jac=None,
        jac_sparsity=self.jac_sparsity,
        dense_output=False,
        )

        t = sol.t
        y = sol.y
        p = self.params
        N = self.operator_order

        mu_p = y[0]
        W, Q, P = y[N : N + 3]
        mu_d = p.d_au * P
        mu_total = mu_p + mu_d

        mu_p_dot = y[1]
        P_dot = p.omega0_au * Q - p.Gamma_au * P
        mu_total_dot = mu_p_dot + p.d_au * P_dot

        E = pulse.field(t)
        absorbed_energy_au = np.trapezoid(mu_total_dot * E, t)
        absorbed_energy_j = float(absorbed_energy_au * AU_ENERGY_J)
        fluence_j_cm2 = pulse.fluence_j_cm2()
        sigma_abs_cm2 = absorbed_energy_j / fluence_j_cm2

        return HybridSolveResult(
            t_au=t,
            y=y,
            mu_p_au=mu_p,
            mu_d_au=mu_d,
            mu_total_au=mu_total,
            mu_dot_total_au=mu_total_dot,
            sigma_abs_cm2=float(sigma_abs_cm2),
            absorbed_energy_j=absorbed_energy_j,
            fluence_j_cm2=float(fluence_j_cm2),
            peak_intensity_w_cm2=float(pulse.peak_intensity_w_cm2()),
            solve_ivp_result=sol)

    def sweep_absorption_vs_peak_intensity(
        self,
        tau_fs: float,
        E0_values_V_m: Iterable[float],
        *,
        omegaL_eV: float | None = None,
        tau_kind: Literal['sigma', 'fwhm_intensity'] = 'fwhm_intensity',
        method: Literal['BDF', 'Radau', 'LSODA'] = 'BDF',
        rtol: float = 1e-7,
        atol: float = 1e-9,
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
                tau_kind=tau_kind)
            
            res = self.solve(pulse, method=method, rtol=rtol, atol=atol)
            I_peaks.append(res.peak_intensity_w_cm2)
            sigmas.append(res.sigma_abs_cm2)
            results.append(res)

        return np.asarray(I_peaks), np.asarray(sigmas), results
    
    #========================================================================================
    # Этот метод зависит от степени аппроксимирующего полинома:
    def _stable_factorized_fit(self, omega, inv_alpha, weights, N):
        omega = np.asarray(omega, dtype=float)
        inv_alpha = np.asarray(inv_alpha, dtype=complex)
        w = np.ones_like(omega) if weights is None else np.asarray(weights, dtype=float)

        scale_re = max(float(np.max(np.abs(inv_alpha.real))), 1e-12)
        scale_im = max(float(np.max(np.abs(inv_alpha.imag))), 1e-12)

        omega_min = float(np.min(omega))
        omega_max = float(np.max(omega))
        omega_c = float(np.median(omega))

        k_lo, k_hi = 1e-6, 1e4
        z_lo, z_hi = 0.02, 10.0
        w_lo = max(0.5 * omega_min, 1e-4)
        w_hi = max(3.0 * omega_max, 1.2 * w_lo)
        p_lo = max(0.2 * omega_min, 1e-4)
        p_hi = max(5.0 * omega_max, 1.5 * p_lo)

        rng = np.random.default_rng(12345 + N)

        def quad_desc(w0, z0):
            # коэффициенты [1, b, c] для s^2 + 2*z*w*s + w^2
            return np.array([1.0, 2.0 * z0 * w0, w0 * w0], dtype=float)

        def lin_desc(p0):
            # коэффициенты [1, p] для s + p
            return np.array([1.0, p0], dtype=float)

        def assemble_desc(factors_desc, k):
            poly = np.array([1.0], dtype=float)
            for f in factors_desc:
                poly = np.polymul(poly, f)   # descending powers
            return k * poly

        def asc_from_desc(poly_desc):
            return np.asarray(poly_desc[::-1], dtype=float)

        def A_from_coeffs(coeffs):
            A = np.zeros_like(inv_alpha, dtype=complex)
            for n, a in enumerate(coeffs):
                A += a * (-1j * omega) ** n
            return A

        def residual_from_coeffs(coeffs):
            A = A_from_coeffs(coeffs)
            r = A - inv_alpha
            return np.concatenate([
                w * r.real / scale_re,
                w * r.imag / scale_im,
            ])

        # ---------- parameterizations ----------
        if N == 2:
            # A(s) = k * Q1(s)
            def unpack(u):
                logk, logw1, logz1 = u
                k, w1, z1 = np.exp(logk), np.exp(logw1), np.exp(logz1)
                desc = assemble_desc([quad_desc(w1, z1)], k)
                return asc_from_desc(desc)

            lower = np.log([k_lo, w_lo, z_lo])
            upper = np.log([k_hi, w_hi, z_hi])

            initials = [
                np.log([1.0, omega_c, 0.20]),
                np.log([0.2, 0.85 * omega_c, 0.60]),
                np.log([5.0, 1.15 * omega_c, 0.08]),
            ]

        elif N == 3:
            # A(s) = k * L(s) * Q1(s)
            def unpack(u):
                logk, logp, logw1, logz1 = u
                k, p, w1, z1 = np.exp(logk), np.exp(logp), np.exp(logw1), np.exp(logz1)
                desc = assemble_desc([lin_desc(p), quad_desc(w1, z1)], k)
                return asc_from_desc(desc)

            lower = np.log([k_lo, p_lo, w_lo, z_lo])
            upper = np.log([k_hi, p_hi, w_hi, z_hi])

            initials = [
                np.log([1.0, 0.5 * omega_c, omega_c, 0.25]),
                np.log([0.2, 1.0 * omega_c, 0.90 * omega_c, 0.70]),
                np.log([5.0, 0.3 * omega_c, 1.15 * omega_c, 0.08]),
            ]

        elif N == 4:
            # A(s) = k * Q1(s) * Q2(s)
            def unpack(u):
                logk, logw1, logw2, logz1, logz2 = u
                k = np.exp(logk)
                w1, w2 = np.exp(logw1), np.exp(logw2)
                z1, z2 = np.exp(logz1), np.exp(logz2)
                desc = assemble_desc([quad_desc(w1, z1), quad_desc(w2, z2)], k)
                return asc_from_desc(desc)

            lower = np.log([k_lo, w_lo, w_lo, z_lo, z_lo])
            upper = np.log([k_hi, w_hi, w_hi, z_hi, z_hi])

            initials = [
                np.log([1.0, 0.85 * omega_c, 1.15 * omega_c, 0.25, 0.80]),
                np.log([0.2, 0.75 * omega_c, 1.35 * omega_c, 0.70, 0.15]),
                np.log([5.0, 0.95 * omega_c, 1.05 * omega_c, 0.10, 0.10]),
            ]

        elif N == 5:
            # A(s) = k * L(s) * Q1(s) * Q2(s)
            def unpack(u):
                logk, logp, logw1, logw2, logz1, logz2 = u
                k = np.exp(logk)
                p = np.exp(logp)
                w1, w2 = np.exp(logw1), np.exp(logw2)
                z1, z2 = np.exp(logz1), np.exp(logz2)
                desc = assemble_desc(
                    [lin_desc(p), quad_desc(w1, z1), quad_desc(w2, z2)],
                    k
                )
                return asc_from_desc(desc)

            lower = np.log([k_lo, p_lo, w_lo, w_lo, z_lo, z_lo])
            upper = np.log([k_hi, p_hi, w_hi, w_hi, z_hi, z_hi])

            initials = [
                np.log([1.0, 0.5 * omega_c, 0.85 * omega_c, 1.15 * omega_c, 0.25, 0.80]),
                np.log([0.2, 1.0 * omega_c, 0.75 * omega_c, 1.35 * omega_c, 0.70, 0.15]),
                np.log([5.0, 0.3 * omega_c, 0.95 * omega_c, 1.05 * omega_c, 0.10, 0.10]),
            ]

        elif N == 6:
            # A(s) = k * Q1(s) * Q2(s) * Q3(s)
            def unpack(u):
                logk, logw1, logw2, logw3, logz1, logz2, logz3 = u
                k = np.exp(logk)
                w1, w2, w3 = np.exp(logw1), np.exp(logw2), np.exp(logw3)
                z1, z2, z3 = np.exp(logz1), np.exp(logz2), np.exp(logz3)
                desc = assemble_desc(
                    [quad_desc(w1, z1), quad_desc(w2, z2), quad_desc(w3, z3)],
                    k
                )
                return asc_from_desc(desc)

            lower = np.log([k_lo, w_lo, w_lo, w_lo, z_lo, z_lo, z_lo])
            upper = np.log([k_hi, w_hi, w_hi, w_hi, z_hi, z_hi, z_hi])

            initials = [
                np.log([1.0, 0.80 * omega_c, 1.00 * omega_c, 1.20 * omega_c, 0.25, 0.50, 1.00]),
                np.log([0.2, 0.70 * omega_c, 0.95 * omega_c, 1.35 * omega_c, 0.70, 0.20, 0.10]),
                np.log([5.0, 0.90 * omega_c, 1.05 * omega_c, 1.15 * omega_c, 0.10, 0.10, 0.10]),
            ]

        else:
            raise NotImplementedError('Stable factorized fit is implemented only for N=2,3,4,5,6.')

        # больше стартов для 5 и 6 порядка
        extra_starts = 24 if N <= 4 else 48
        for _ in range(extra_starts):
            initials.append(rng.uniform(lower, upper))

        best = None
        for u0 in initials:
            u0 = np.clip(np.asarray(u0, dtype=float), lower, upper)

            res = least_squares(
                lambda u: residual_from_coeffs(unpack(u)),
                x0=u0,
                bounds=(lower, upper),
                max_nfev=80000,
                ftol=1e-12,
                xtol=1e-12,
                gtol=1e-12,
            )

            coeffs = unpack(res.x)
            roots = np.roots(coeffs[::-1])

            if np.any(np.real(roots) >= 1e-10):
                continue

            score = np.sqrt(np.mean(residual_from_coeffs(coeffs) ** 2))

            if best is None or score < best[0]:
                best = (score, coeffs, roots)

        if best is None:
            raise RuntimeError(f'Could not find a stable fit for N={N}.')

        return best[1]
    #========================================================================================





def make_default_params() -> HybridSystemParams:
    return HybridSystemParams(
        c_au=float(nm_to_au(15.0)),
        a_au=float(nm_to_au(10.0)),
        R_au=float(nm_to_au(17.0)),
        G=2.0,
        eps_m=1.0,
        d_au=float(dipole_si_to_au(7.5e-29)),
        omega0_au=float(eV_to_au(2.042)),
        gamma_au=float(1.0 / ns_to_au(30.0)),
        Gamma_au=float(1.0 / fs_to_au(330.0)))



def reconstruct_inv_alpha_from_time_operator(coeffs_time: np.ndarray, energies_eV: np.ndarray) -> np.ndarray:
    """
    Восстанавливает спектральную аппроксимацию 1/alpha(omega)
    напрямую из коэффициентов временного оператора:

        A(d/dt)  <->  A(-i*omega) = sum_n a_n (-i*omega)^n

    coeffs_time = [a0, a1, ..., aN]
    energies_eV : массив энергий в eV
    """
    omega = eV_to_au(np.asarray(energies_eV, dtype=float))
    inv_alpha_fit = np.zeros_like(omega, dtype=complex)

    for n, a_n in enumerate(coeffs_time):
        inv_alpha_fit += a_n * (-1j * omega) ** n

    return inv_alpha_fit


def plot_inv_alpha_new_method(
    params: HybridSystemParams,
    orders=(2, 3, 4, 5, 6),
    orientation='long',
    fit_window_eV=(0.8, 3.0),
    weight_center_eV=2.042,
    weight_sigma_eV=0.0035,
    energy_plot_range=(0.8, 3.0),
    n_plot=800,
):
    energies = np.linspace(energy_plot_range[0], energy_plot_range[1], n_plot)

    model_ref = HybridQDPlasmonModel(
        params,
        orientation=orientation,
        operator_order=2,
        fit_window_eV=fit_window_eV,
        weight_center_eV=weight_center_eV,
        weight_sigma_eV=weight_sigma_eV,
        enforce_stability=False,
        verbose_stability=False,
    )

    inv_alpha_true = 1.0 / model_ref._alpha_dimless(model_ref.L)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    # табличные данные
    ax1.plot(params.material.energy_eV, inv_alpha_true.real, 'ko', ms=4, label='Tabulated Re[1/alpha]', zorder=20)
    ax2.plot(params.material.energy_eV, inv_alpha_true.imag, 'ko', ms=4, label='Tabulated Im[1/alpha]', zorder=20)

    styles = {
        2: dict(ls='-',  marker='s', markevery=65, lw=2.6, ms=5, alpha=0.95, zorder=7),
        3: dict(ls='--', marker='^', markevery=75, lw=2.6, ms=5, alpha=0.95, zorder=8),
        4: dict(ls='-.', marker='D', markevery=85, lw=2.6, ms=5, alpha=0.95, zorder=9),
        5: dict(ls=':',  marker='v', markevery=95, lw=2.8, ms=5, alpha=0.95, zorder=10),
        6: dict(ls=(0, (3, 1, 1, 1)), marker='P', markevery=105, lw=2.8, ms=5, alpha=0.95, zorder=11),
    }

    for order in orders:
        model_fit = HybridQDPlasmonModel(
            params,
            orientation=orientation,
            operator_order=order,
            fit_window_eV=fit_window_eV,
            weight_center_eV=weight_center_eV,
            weight_sigma_eV=weight_sigma_eV,
            enforce_stability=False,
            verbose_stability=False,
        )

        inv_alpha_fit = model_fit.reconstructed_inv_alpha(energies)
        roots = np.roots(model_fit.fit.coeffs_time[::-1])
        stable = np.all(np.real(roots) < 0.0)

        st = styles.get(order, dict(ls='-', lw=2.4, alpha=0.95, zorder=6))
        label = f'new fit N={order} (stable={stable})'

        ax1.plot(energies, inv_alpha_fit.real, label=label, **st)
        ax2.plot(energies, inv_alpha_fit.imag, label=label, **st)

    for ax in (ax1, ax2):
        ax.axvspan(fit_window_eV[0], fit_window_eV[1], alpha=0.12, color='gray')
        if weight_center_eV is not None:
            ax.axvline(weight_center_eV, ls=':', lw=1.5, color='tab:blue')
        ax.grid(True)
        ax.legend(fontsize=9)

    ax1.set_ylabel('Re[1/alpha]')
    ax1.set_title('1/alpha(omega): tabulated data vs new-method fits')

    ax2.set_ylabel('Im[1/alpha]')
    ax2.set_xlabel('Energy (eV)')

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    
    params = make_default_params()

    plot_inv_alpha_new_method(
        params,
        orders=(2, 3, 4, 5, 6),
        orientation='long',
        fit_window_eV=(2.0, 2.5),
        weight_center_eV=None,
        weight_sigma_eV=None,
    )

    model = HybridQDPlasmonModel(
        params,
        orientation='long',
        operator_order=4,
        fit_window_eV=(1.8, 3.0),
        weight_center_eV=2.042,
        weight_sigma_eV=0.7,
        enforce_stability=True,
        verbose_stability=True
    )


    pulse = GaussianPulse(
        E0_au=float(field_si_to_au(2.5e5)),   # амплитуда поля, В/м -> а.е.
        omegaL_au=float(eV_to_au(2.042)),     # частота лазера
        tau_au=float(fs_to_au(5.0)),          # длительность 5 фс
        tau_kind='fwhm_intensity'
    )

    #res = model.solve(pulse, method='LSODA', rtol=1e-8, atol=1e-8)
    # res = model.solve(pulse, method='DOP853', rtol=1e-3, atol=1e-3)
    res = model.solve(pulse, method='Radau', rtol=1e-8, atol=1e-10)

    print('\nsigma_abs [cm^2] =', res.sigma_abs_cm2)
    print('fluence [J/cm^2] =', res.fluence_j_cm2)
    print(f'I_peak [W/cm^2] = {res.peak_intensity_w_cm2}\n')

    # Временная сетка
    t_fs = au_to_fs(res.t_au)

    # Дипольные моменты
    mu_p = res.mu_p_au         # МНЧ
    mu_d = res.mu_d_au         # КТ
    mu_total = res.mu_total_au # суммарный дипольный момент системы
    E_t = pulse.field(res.t_au)

    _,axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    # --- График 1: Поле импульса (для контекста) ---
    axes[0].plot(t_fs, E_t, color='gray', alpha=0.5, label='Лазерный импульс E(t)')
    axes[0].set_ylabel('Field Amplitude (a.u.)')
    axes[0].legend()
    axes[0].grid(True)
    # --- График 2: Дипольный момент МНЧ (Плазмон) ---
    axes[1].plot(t_fs, res.mu_p_au, color='red', label=fr'Диполь МНЧ $\mu_P(t)$')
    axes[1].set_ylabel('MNP Dipole (a.u.)')
    axes[1].legend()
    axes[1].grid(True)
    # --- График 3: Дипольный момент КТ ---
    axes[2].plot(t_fs, res.mu_d_au, color='green', label=fr'Диполь КТ $\mu_D(t)$')
    axes[2].set_xlabel('Время (фс)')
    axes[2].set_ylabel('QD Dipole (a.u.)')
    axes[2].legend()
    axes[2].grid(True)
    plt.suptitle('Сравнение динамики дипольных моментов')
    plt.tight_layout()
    plt.show()
