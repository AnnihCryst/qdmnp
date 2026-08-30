"""Shared physics for the McMillan et al. (2016) time-domain benchmark.

The benchmark paper is R. J. McMillan, L. Stella and M. Gruning,
Phys. Rev. B 94, 125312 (2016), DOI 10.1103/PhysRevB.94.125312.

Two different indices must not be conflated here:

``n``
    Spatial spheroidal/spherical-harmonic order.  The paper uses the dipole
    approximation, so the paper-matched calculation has ``n_max=1``.

``k``
    Pole of a causal material-response realization.  McMillan et al. use
    first-order complex PEOM poles.  The project core uses passive real
    second-order Lorentz oscillators.  This module fits the latter to the same
    Etchegoin gold polarizability before any time propagation is attempted.

The module deliberately contains no plotting.  One executable script is kept
for each reproduced article figure.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from scipy.constants import c as C_SI
from scipy.constants import epsilon_0
from scipy.integrate import quad
from scipy.optimize import nnls

from qd_mnp_full_qs_model import FullQSSpheroidPulseModel
from qd_mnp_rational_fit import (
    AU_ENERGY_EV,
    AU_FIELD_V_M,
    AU_TIME_S,
    GaussianPulse,
    HybridQDPlasmonModel,
    HybridSystemParams,
    MaterialDispersion,
    RationalLorentzFit,
    au_to_fs,
    eV_to_au,
    nm_to_au,
    ns_to_au,
)
from qd_mnp_spheroid_green import SpheroidGreenInteraction


PAPER_DOI: Final[str] = "10.1103/PhysRevB.94.125312"
PAPER_ARXIV_URL: Final[str] = "https://arxiv.org/abs/1607.06386"

# The journal article imports the particle size from its Ref. 17.  McMillan's
# thesis later writes "diameter of a=7.5 nm", although both the defining
# formula alpha=a^3(eps-1)/(eps+2) and Ref. 17 use ``a`` as the radius.  The
# benchmark therefore uses the unambiguous formula parameter a=7.5 nm; this
# choice also reproduces the published Fig. 4(c) trajectories.
MNP_RADIUS_NM: Final[float] = 7.5
QD_BARE_DIPOLE_E_NM: Final[float] = 0.65
EXCITON_ENERGY_EV: Final[float] = 2.5
QD_EPSILON: Final[float] = 6.0
HOST_EPSILON: Final[float] = 1.0
POPULATION_LIFETIME_NS: Final[float] = 0.8
COHERENCE_LIFETIME_NS: Final[float] = 0.3
LONGITUDINAL_G: Final[float] = 2.0

PULSE_CYCLES: Final[int] = 10
PULSE_AREA_PI: Final[float] = 5.0
FIT_WINDOW_EV: Final[tuple[float, float]] = (0.01, 10.0)


# Etchegoin, Le Ru and Meyer, J. Chem. Phys. 125, 164705 (2006).
# Energies are hbar*omega in eV.  Gamma_D=0.0729 eV is the value obtained
# from the published damping wavelength gamma_p=17000 nm.  The occasionally
# repeated 0.00729 eV value is incompatible with that table and with the
# published twelve-pole fit used as an independent check in McMillan's thesis.
ETCHEGOIN_EPS_INF: Final[float] = 1.53
ETCHEGOIN_PLASMA_EV: Final[float] = 8.55
ETCHEGOIN_DRUDE_GAMMA_EV: Final[float] = 0.0729
ETCHEGOIN_INTERBAND: Final[tuple[tuple[float, float, float, float], ...]] = (
    (0.94, -np.pi / 4.0, 2.65, 0.539),
    (1.36, -np.pi / 4.0, 3.74, 1.32),
)


@dataclass(frozen=True)
class PassiveFitAudit:
    """Numerical certificate attached to the script-built material fit."""

    selected_poles: int
    dictionary_poles: int
    normalized_rms_alpha: float
    normalized_rms_inverse_alpha: float
    max_pointwise_relative_alpha_error: float
    min_imaginary_alpha: float
    fit_window_eV: tuple[float, float]
    accepted: bool


@dataclass(frozen=True)
class PulseProfile:
    """Article pulse parameters in atomic and paper time coordinates."""

    pulse: "SechPulse"
    duration_au: float
    tau_p_au: float
    paper_center_au: float
    area_pi: float
    cycles: int
    local_field_amplitude_factor: float


class EtchegoinGoldMaterial(MaterialDispersion):
    """Tabulated-domain material object with an exact analytic ``epsilon_at``."""

    def epsilon_at(self, energy_eV: float | np.ndarray) -> np.ndarray:
        energy = np.asarray(energy_eV, dtype=float)
        if np.any(~np.isfinite(energy)):
            raise ValueError("Requested Etchegoin energies must be finite.")
        scale = max(abs(float(self.energy_eV[0])), abs(float(self.energy_eV[-1])), 1.0)
        tolerance = 1.0e-12 * scale
        if np.any(energy < self.energy_eV[0] - tolerance) or np.any(
            energy > self.energy_eV[-1] + tolerance
        ):
            raise ValueError(
                "Requested Etchegoin energy lies outside the declared interval "
                f"[{self.energy_eV[0]:g}, {self.energy_eV[-1]:g}] eV."
            )
        clipped = np.clip(energy, self.energy_eV[0], self.energy_eV[-1])
        return etchegoin_gold_epsilon(clipped)


@dataclass(frozen=True)
class SechPulse(GaussianPulse):
    """Real carrier with the McMillan hyperbolic-secant envelope.

    ``tau_au`` is :math:`tau_p` in Eq. (25) of the paper, not a Gaussian
    width.  The pulse center is shifted to zero for the core solver.  Because
    ``omega_L*T/2 = 2*pi*n`` for the paper definition of ``T``, this shift
    leaves the carrier phase unchanged for integer ``n``.
    """

    def envelope(self, t_au: float | np.ndarray) -> np.ndarray:
        x = np.asarray(t_au, dtype=float) / self.tau_au
        absolute = np.abs(x)
        exponential = np.exp(-absolute)
        return 2.0 * exponential / (1.0 + exponential**2)

    def field(self, t_au: float | np.ndarray) -> np.ndarray:
        t = np.asarray(t_au, dtype=float)
        return self.E0_au * self.envelope(t) * np.cos(self.omegaL_au * t)

    def field_dot(self, t_au: float | np.ndarray) -> np.ndarray:
        t = np.asarray(t_au, dtype=float)
        envelope = self.envelope(t)
        envelope_dot = -envelope * np.tanh(t / self.tau_au) / self.tau_au
        return self.E0_au * (
            envelope_dot * np.cos(self.omegaL_au * t)
            - self.omegaL_au * envelope * np.sin(self.omegaL_au * t)
        )

    @staticmethod
    def _stable_sech(value: float) -> float:
        absolute = abs(float(value))
        exponential = np.exp(-absolute)
        return float(2.0 * exponential / (1.0 + exponential**2))

    def _positive_spectral_density(self, omega_au: float) -> float:
        scale = 0.5 * np.pi * self.tau_au
        amplitude = self._stable_sech(scale * (omega_au - self.omegaL_au))
        amplitude += self._stable_sech(scale * (omega_au + self.omegaL_au))
        return float(amplitude**2)

    def positive_frequency_spectral_fraction(
        self,
        energy_window_eV: tuple[float, float],
    ) -> float:
        if len(energy_window_eV) != 2 or not np.all(np.isfinite(energy_window_eV)):
            raise ValueError("energy_window_eV must contain two finite values.")
        e_min, e_max = (float(value) for value in energy_window_eV)
        if e_min < 0.0 or e_max <= e_min:
            raise ValueError("energy_window_eV must satisfy 0 <= min < max.")

        omega_min = float(eV_to_au(e_min))
        omega_max = float(eV_to_au(e_max))
        integration_max = max(
            omega_max,
            self.omegaL_au + 50.0 / self.tau_au,
        )
        denominator = quad(
            self._positive_spectral_density,
            0.0,
            integration_max,
            epsabs=1.0e-13,
            epsrel=2.0e-11,
            limit=300,
        )[0]
        numerator = quad(
            self._positive_spectral_density,
            omega_min,
            min(omega_max, integration_max),
            epsabs=1.0e-13,
            epsrel=2.0e-11,
            limit=300,
        )[0]
        return float(np.clip(numerator / denominator, 0.0, 1.0))

    def fluence_j_cm2(self, *, eps_m: float = 1.0) -> float:
        if not np.isfinite(eps_m) or eps_m <= 0.0:
            raise ValueError("eps_m must be finite and positive.")
        e0_si = float(self.E0_au * AU_FIELD_V_M)
        tau_s = float(self.tau_au * AU_TIME_S)
        carrier_width = float(np.pi * self.omegaL_au * self.tau_au)
        oscillatory = carrier_width / np.sinh(carrier_width)
        integral_e2 = e0_si**2 * tau_s * (1.0 + oscillatory)
        fluence_j_m2 = np.sqrt(eps_m) * epsilon_0 * C_SI * integral_e2
        return float(fluence_j_m2 * 1.0e-4)


def etchegoin_gold_epsilon(energy_eV: float | np.ndarray) -> np.ndarray:
    """Return the causal analytic dielectric function used by McMillan et al."""

    energy = np.asarray(energy_eV, dtype=float)
    if np.any(~np.isfinite(energy)) or np.any(energy <= 0.0):
        raise ValueError("Etchegoin energies must be finite and strictly positive.")
    epsilon = ETCHEGOIN_EPS_INF - ETCHEGOIN_PLASMA_EV**2 / (
        energy**2 + 1j * ETCHEGOIN_DRUDE_GAMMA_EV * energy
    )
    for amplitude, phase, resonance, damping in ETCHEGOIN_INTERBAND:
        epsilon = epsilon + amplitude * resonance * (
            np.exp(-1j * phase) / (energy + resonance + 1j * damping)
            - np.exp(1j * phase) / (energy - resonance + 1j * damping)
        )
    return np.asarray(epsilon, dtype=complex)


def make_etchegoin_material() -> EtchegoinGoldMaterial:
    """Create the material-domain object used by both raw and fitted backends."""

    energies = np.unique(
        np.concatenate(
            (
                np.geomspace(FIT_WINDOW_EV[0], 0.5, 161),
                np.linspace(0.5, 12.0, 461),
                np.linspace(12.0, 60.0, 193),
            )
        )
    )
    epsilon = etchegoin_gold_epsilon(energies)
    if np.min(epsilon.imag) < -1.0e-12:
        raise RuntimeError("The selected Etchegoin convention produced optical gain.")
    refractive_index = np.sqrt(epsilon)
    if np.any(refractive_index.imag < 0.0):
        refractive_index = np.where(
            refractive_index.imag < 0.0,
            -refractive_index,
            refractive_index,
        )
    return EtchegoinGoldMaterial(
        energy_eV=energies,
        n=np.asarray(refractive_index.real, dtype=float),
        k=np.asarray(refractive_index.imag, dtype=float),
    )


def make_paper_params(
    separation_nm: float,
    *,
    material: MaterialDispersion | None = None,
) -> HybridSystemParams:
    """Map the CdSe/Au parameters of the paper to the native core units."""

    if not np.isfinite(separation_nm) or separation_nm <= MNP_RADIUS_NM:
        raise ValueError("separation_nm must place the point QD outside the MNP.")
    selected_material = make_etchegoin_material() if material is None else material
    return HybridSystemParams(
        c_au=float(nm_to_au(MNP_RADIUS_NM)),
        a_au=float(nm_to_au(MNP_RADIUS_NM)),
        R_au=float(nm_to_au(separation_nm)),
        G=LONGITUDINAL_G,
        eps_m=HOST_EPSILON,
        # In atomic units, a dipole of 0.65 e*nm is 0.65 nm/a0.
        d_au=float(nm_to_au(QD_BARE_DIPOLE_E_NM)),
        omega0_au=float(eV_to_au(EXCITON_ENERGY_EV)),
        gamma_au=float(1.0 / ns_to_au(POPULATION_LIFETIME_NS)),
        Gamma_au=float(1.0 / ns_to_au(COHERENCE_LIFETIME_NS)),
        qd_radius_au=0.0,
        eps_qd=QD_EPSILON,
        qd_dipole_convention="bare_internal",
        material=selected_material,
    )


def sphere_bright_susceptibility(energy_eV: np.ndarray) -> np.ndarray:
    """Dimensionless sphere susceptibility ``H`` used by the full-QS core."""

    epsilon = etchegoin_gold_epsilon(energy_eV)
    contrast = epsilon - HOST_EPSILON
    return np.asarray(
        contrast / (HOST_EPSILON + contrast / 3.0),
        dtype=complex,
    )


def fit_passive_lorentz_sphere(
    *,
    selected_poles: int = 80,
    fit_points: int = 1001,
    max_normalized_rms: float = 0.025,
    max_pointwise_relative_error: float = 0.06,
) -> tuple[RationalLorentzFit, PassiveFitAudit]:
    """Fit a passive second-order ADE to the paper's Au sphere response.

    A dense non-negative Lorentz dictionary is first fitted by NNLS.  The most
    important poles are retained and all retained non-negative residues are
    refitted.  The procedure is deterministic and takes under a few seconds;
    it does not reuse McMillan's signed first-order PEOM coefficients.
    """

    if selected_poles < 8:
        raise ValueError("selected_poles must be at least 8.")
    if fit_points < 501:
        raise ValueError("fit_points must be at least 501.")

    energies = np.linspace(FIT_WINDOW_EV[0], FIT_WINDOW_EV[1], fit_points)
    target = sphere_bright_susceptibility(energies)

    centers = np.unique(
        np.concatenate(
            (
                np.geomspace(0.02, 1.5, 18),
                np.linspace(1.55, 5.0, 55),
                np.linspace(5.2, 12.0, 20),
                np.asarray([15.0, 20.0, 25.0, 30.0, 35.0, 45.0]),
            )
        )
    )
    damping_ratios = np.asarray([0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8])
    omega_dictionary = np.repeat(centers, damping_ratios.size)
    gamma_dictionary = np.maximum(
        0.005,
        np.tile(damping_ratios, centers.size) * omega_dictionary,
    )
    if selected_poles > omega_dictionary.size:
        raise ValueError("selected_poles exceeds the Lorentz dictionary size.")

    basis = 1.0 / (
        omega_dictionary[None, :] ** 2
        - energies[:, None] ** 2
        - 1j * gamma_dictionary[None, :] * energies[:, None]
    )
    real_scale = max(float(np.sqrt(np.mean(target.real**2))), np.finfo(float).tiny)
    imag_scale = max(float(np.sqrt(np.mean(target.imag**2))), np.finfo(float).tiny)
    design = np.concatenate((basis.real / real_scale, basis.imag / imag_scale), axis=0)
    right_hand_side = np.concatenate(
        (target.real / real_scale, target.imag / imag_scale)
    )
    full_strengths, _ = nnls(design, right_hand_side, maxiter=10000)
    importance = np.sqrt(
        np.mean(np.abs(basis * full_strengths[None, :]) ** 2, axis=0)
    )
    retained = np.argsort(importance)[-selected_poles:]
    retained_basis = basis[:, retained]
    retained_design = np.concatenate(
        (retained_basis.real / real_scale, retained_basis.imag / imag_scale),
        axis=0,
    )
    strengths_eV2, residual_norm = nnls(
        retained_design,
        right_hand_side,
        maxiter=10000,
    )
    fitted = retained_basis @ strengths_eV2
    error = fitted - target
    inverse_error = 1.0 / fitted - 1.0 / target
    rms_alpha = float(np.sqrt(np.mean(np.abs(error) ** 2)))
    rms_inverse = float(np.sqrt(np.mean(np.abs(inverse_error) ** 2)))
    normalized_rms = rms_alpha / max(
        float(np.sqrt(np.mean(np.abs(target) ** 2))),
        np.finfo(float).tiny,
    )
    normalized_inverse_rms = rms_inverse / max(
        float(np.sqrt(np.mean(np.abs(1.0 / target) ** 2))),
        np.finfo(float).tiny,
    )
    target_scale = max(float(np.max(np.abs(target))), np.finfo(float).tiny)
    max_relative = float(
        np.max(
            np.abs(error)
            / np.maximum(np.abs(target), 1.0e-15 * target_scale)
        )
    )
    minimum_imaginary = float(np.min(fitted.imag))
    accepted = bool(
        normalized_rms <= max_normalized_rms
        and normalized_inverse_rms <= max_normalized_rms
        and max_relative <= max_pointwise_relative_error
        and minimum_imaginary >= -1.0e-12
        and np.all(strengths_eV2 >= 0.0)
    )

    audit = PassiveFitAudit(
        selected_poles=int(selected_poles),
        dictionary_poles=int(omega_dictionary.size),
        normalized_rms_alpha=float(normalized_rms),
        normalized_rms_inverse_alpha=float(normalized_inverse_rms),
        max_pointwise_relative_alpha_error=max_relative,
        min_imaginary_alpha=minimum_imaginary,
        fit_window_eV=FIT_WINDOW_EV,
        accepted=accepted,
    )
    if not accepted:
        raise RuntimeError(
            "The McMillan broad-band passive Lorentz fit failed its gate: "
            f"NRMS(alpha)={normalized_rms:.6g}, "
            f"NRMS(1/alpha)={normalized_inverse_rms:.6g}, "
            f"max-relative={max_relative:.6g}, min-Im={minimum_imaginary:.6g}."
        )

    omega_eV = omega_dictionary[retained]
    gamma_eV = gamma_dictionary[retained]
    fit = RationalLorentzFit(
        alpha_inf=0.0,
        strengths_au2=np.asarray(strengths_eV2 / AU_ENERGY_EV**2, dtype=float),
        omega_modes_au=np.asarray(omega_eV / AU_ENERGY_EV, dtype=float),
        gamma_modes_au=np.asarray(gamma_eV / AU_ENERGY_EV, dtype=float),
        energies_used_eV=energies,
        alpha_used=target,
        rms_alpha=rms_alpha,
        rms_inv_alpha=rms_inverse,
        cost=float(residual_norm),
        normalized_rms_alpha=float(normalized_rms),
        normalized_rms_inv_alpha=float(normalized_inverse_rms),
        max_normalized_alpha_error=max_relative,
        min_imag_alpha_fit_window=minimum_imaginary,
        passivity_grid_points=int(energies.size),
        passive_on_fit_window=True,
        passive_for_all_positive_frequencies=True,
    )
    return fit, audit


class _PrefittedHybridQDPlasmonModel(HybridQDPlasmonModel):
    """Thin literature adapter that lets the existing core audit a supplied fit.

    The native constructor currently owns its nonlinear fitting step.  This
    script-local subclass only replaces that protected construction hook; all
    subsequent frequency response, stability checks and time equations remain
    the unmodified project core.
    """

    def __init__(
        self,
        params: HybridSystemParams,
        prefitted: RationalLorentzFit,
    ) -> None:
        self._literature_prefitted = prefitted
        super().__init__(
            params,
            orientation="long",
            n_modes=int(prefitted.strengths_au2.size),
            fit_window_eV=FIT_WINDOW_EV,
            max_fit_normalized_rms=None,
            max_fit_pointwise_relative_error=None,
            radiative_consistency_policy="ignore",
            verbose=False,
        )

    def _fit_rational_alpha(self) -> RationalLorentzFit:
        return self._literature_prefitted


def build_paper_matched_model(
    params: HybridSystemParams,
    fit: RationalLorentzFit,
    *,
    spatial_orders: int = 1,
) -> FullQSSpheroidPulseModel:
    """Build the core model; ``spatial_orders=1`` matches McMillan's geometry."""

    if spatial_orders < 1:
        raise ValueError("spatial_orders must be positive.")
    bright = _PrefittedHybridQDPlasmonModel(params, fit)
    kernel = SpheroidGreenInteraction.from_params(
        params,
        orientation="long",
        n_max=spatial_orders,
    )
    return FullQSSpheroidPulseModel(
        bright,
        kernel,
        fit_quality_policy="raise",
        max_modal_normalized_rms=0.025,
        max_modal_relative_error=0.06,
        modal_audit_points=1001,
        # n=1 is intentionally the paper's dipole limit, not a claim that the
        # complete spatial series has converged at R=13 nm.
        spatial_convergence_policy=("ignore" if spatial_orders == 1 else "raise"),
        spatial_convergence_rtol=1.0e-6,
    )


def mnp_polarizability_au3_at_carrier(params: HybridSystemParams) -> complex:
    """Raw Clausius-Mossotti sphere polarizability used in paper Eq. (29)."""

    epsilon = complex(params.material.epsilon_at(EXCITON_ENERGY_EV))
    return complex(
        params.a_au**3
        * (epsilon - params.eps_m)
        / (epsilon + 2.0 * params.eps_m)
    )


def make_paper_pulse(
    params: HybridSystemParams,
    *,
    cycles: int = PULSE_CYCLES,
    area_pi: float = PULSE_AREA_PI,
) -> PulseProfile:
    """Construct the R-dependent 5-pi sech pulse from paper Eqs. (26)-(29)."""

    if cycles < 1:
        raise ValueError("cycles must be a positive integer.")
    if not np.isfinite(area_pi) or area_pi <= 0.0:
        raise ValueError("area_pi must be finite and positive.")
    omega_l = float(eV_to_au(EXCITON_ENERGY_EV))
    duration = float(4.0 * np.pi * cycles / omega_l)
    tau_p = float(duration / 30.0)
    screened_dipole = float(params.qd_local_field_factor * params.d_au)
    alpha_mnp = mnp_polarizability_au3_at_carrier(params)
    # Paper Eq. (29): in its vacuum convention the MNP correction is
    # G*alpha_MNP/R^3.  HOST_EPSILON is one for this benchmark, but keeping
    # the equation literally avoids suggesting an extra medium factor that
    # is not present in the published pulse-area definition.
    local_amplitude_factor = abs(1.0 + params.G * alpha_mnp / params.R_au**3)
    theta = float(area_pi * np.pi)
    e0 = float(theta / (np.pi * screened_dipole * tau_p * local_amplitude_factor))
    pulse = SechPulse(
        E0_au=e0,
        omegaL_au=omega_l,
        tau_au=tau_p,
        tau_kind="sigma",
    )
    return PulseProfile(
        pulse=pulse,
        duration_au=duration,
        tau_p_au=tau_p,
        paper_center_au=0.5 * duration,
        area_pi=float(area_pi),
        cycles=int(cycles),
        local_field_amplitude_factor=float(local_amplitude_factor),
    )


def paper_time_fs(centered_time_au: np.ndarray, profile: PulseProfile) -> np.ndarray:
    """Convert core-centered time to the paper's interval ``0 <= t <= T``."""

    return np.asarray(au_to_fs(centered_time_au + profile.paper_center_au), dtype=float)


def find_article_pdf(candidate: str | Path | None) -> Path | None:
    """Resolve an optional local PDF without downloading copyrighted content."""

    if candidate is None:
        return None
    path = Path(candidate).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Article PDF not found: {path}")
    return path
