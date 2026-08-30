"""Validate the full-QS modal ODE against Yan et al. Figs. 2(b) and 3.

This is a literature-facing driver, not a third mathematical core.  It uses
the public full-QS ``A/B/K`` response and the public time-domain ``rhs`` as
APIs, and keeps three comparisons separate:

1. ``material_rwa``: direct Johnson--Christy optical constants with the
   rotating-wave, steady-state QD response used in Yan et al.;
2. ``causal_fit_native``: the passive Lorentz realization and the full real-
   carrier weak-field QD response actually represented by the new ODE core;
3. ``time_ode``: numerical propagation of that same ODE realization.

The script reconstructs the total strict-quasistatic external-work spectrum.
Yan neglects Rayleigh scattering for these small particles, so this is the
quantity that can be compared with the paper's total energy absorption rate
``Q_tot``.  It is deliberately not labelled as a separately resolved metal
heating rate: the current core exports the work done by the incident field on
the whole coupled dipole, not a volume-loss decomposition by subsystem.

The full Maxwell--Bloch carrier ODE is initialized on its weak-field periodic
orbit.  This checks the modal ODE transfer function without pretending that a
multi-nanosecond switch-on transient was propagated.  A second, plasmon-only
test starts all Lorentz coordinates from zero and waits twelve modal decay
times; that test really does verify settling of the plasmon expansion.

Paper
-----
J.-Y. Yan, W. Zhang, S. Duan, X.-G. Zhao, and A. O. Govorov,
Phys. Rev. B 77, 165301 (2008), DOI 10.1103/PhysRevB.77.165301.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
import sys
from typing import Sequence

# Keep both ``python -m literature_reproductions...`` and direct execution of
# this file usable from an arbitrary current directory.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from scipy.constants import c as C_SI
from scipy.constants import elementary_charge, epsilon_0, hbar
from scipy.integrate import solve_ivp

from qd_mnp_full_qs_model import FullQSSpheroidPulseModel
from qd_mnp_linear_spectrum import quasistatic_work_loss_cross_section_cm2
from qd_mnp_rational_fit import (
    AU_ENERGY_EV,
    AU_ENERGY_J,
    AU_TIME_S,
    DEBYE_C_M,
    DEFAULT_AU_MATERIAL,
    GaussianPulse,
    HybridQDPlasmonModel,
    HybridSystemParams,
    MaterialDispersion,
    au_to_fs,
    eV_to_au,
    field_si_to_au,
    make_params_with_overrides,
    ns_to_au,
    params_to_physical_dict,
    timestamped_run_dir,
    write_json,
)
from qd_mnp_spheroid_green import (
    SpheroidGreenInteraction,
    qd_linear_polarizability_from_params,
    solve_linear_hybrid_response,
)


ARTICLE_DOI = "10.1103/PhysRevB.77.165301"
ARTICLE_URL = "https://doi.org/10.1103/PhysRevB.77.165301"
ZHANG_PARAMETER_URL = "https://doi.org/10.1103/PhysRevLett.97.146804"
JOHNSON_CHRISTY_URL = "https://doi.org/10.1103/PhysRevB.6.4370"
AG_DATA_URL = (
    "https://github.com/polyanskiy/refractiveindex.info-database/blob/"
    "main/database/data/main/Ag/nk/Johnson.yml"
)
SCHEMA_VERSION = 1
DEFAULT_ARTICLE_PDF = Path(__file__).resolve().parents[1] / "articles" / "yan2008.pdf"

FIG2B_SPECTRUM_PNG = "yan2008_fig2b_spectrum_full_qs.png"
FIG2B_DISTANCE_PNG = "yan2008_fig2b_distance_full_qs.png"
FIG3_PNG = "yan2008_fig3_full_qs.png"
MODAL_ODE_PNG = "yan2008_modal_ode_validation.png"
FIG2B_SPECTRUM_CSV = "yan2008_fig2b_spectrum.csv"
FIG2B_DISTANCE_CSV = "yan2008_fig2b_distance.csv"
FIG3_OVERVIEW_CSV = "yan2008_fig3_overview.csv"
FIG3_N1_CSV = "yan2008_fig3_N1_zoom.csv"
FIG3_N10_CSV = "yan2008_fig3_N10_zoom.csv"
TIME_AU_CSV = "yan2008_time_ode_Au_N10.csv"
TIME_AG_CSV = "yan2008_time_ode_Ag_N10.csv"
METADATA_FILENAME = "metadata.json"

# Figure readouts are intentionally low precision.  They are visual anchors,
# not author-supplied numerical data.  The uncertainties express axis/pixel
# readout precision and must not be interpreted as experimental error bars.
PAPER_RASTER_ANCHORS = {
    "fig2b_N1": {
        "detuning_meV": -0.0267,
        "detuning_uncertainty_meV": 0.0020,
        "power_1e10_W": 0.331,
        "power_uncertainty_1e10_W": 0.004,
        "printed_peak_label_meV": -0.025,
    },
    "fig2b_N10": {
        "detuning_meV": -0.210,
        "detuning_uncertainty_meV": 0.005,
        "power_1e10_W": 0.0930,
        "power_uncertainty_1e10_W": 0.0010,
        "printed_peak_label_meV": -0.200,
    },
    "fig3_plasmon": {
        "energy_eV": 3.230,
        "energy_uncertainty_eV": 0.003,
        "power_1e10_W": 5.05,
        "power_uncertainty_1e10_W": 0.08,
    },
    "fig3_N1": {
        "detuning_meV": 0.294,
        "detuning_uncertainty_meV": 0.020,
        "power_1e10_W": 2.32,
        "power_uncertainty_1e10_W": 0.04,
    },
    "fig3_N10": {
        "detuning_meV": -3.32,
        "detuning_uncertainty_meV": 0.10,
        "power_1e10_W": 0.531,
        "power_uncertainty_1e10_W": 0.003,
    },
}

# Marker centres digitized from the 600-dpi rendering of the local PDF.
# x/y uncertainties are approximately 0.0005/0.006 in the plotted logarithms.
PAPER_FIG2B_DISTANCE_RASTER = {
    "distance_nm": np.arange(18.0, 38.0, 2.0),
    "log10_distance_ratio": np.asarray(
        [0.07918, 0.12494, 0.16633, 0.20412, 0.23888,
         0.27107, 0.30103, 0.32906, 0.35539, 0.38021]
    ),
    "log10_abs_shift_N10_meV": np.asarray(
        [-0.062, -0.686, -1.159, -1.538, -1.864,
         -2.144, -2.389, -2.601, -2.770, -2.959]
    ),
    "log10_abs_shift_N1_meV": np.asarray(
        [-1.285, -1.571, -1.833, -2.070, -2.292,
         -2.481, -2.659, -2.825, -2.959, -3.099]
    ),
    "x_uncertainty": 0.0005,
    "y_uncertainty": 0.006,
}

HBAR_EV_S = hbar / elementary_charge
TRANSITION_DIPOLE_E_NM = 0.65
TRANSITION_DIPOLE_DEBYE = elementary_charge * TRANSITION_DIPOLE_E_NM * 1.0e-9 / DEBYE_C_M
T1_NS = 0.8
T20_NS = 0.3
GAMMA1_MEV = HBAR_EV_S / (T1_NS * 1.0e-9) * 1.0e3
GAMMA2_MEV = HBAR_EV_S / (T20_NS * 1.0e-9) * 1.0e3
DEFAULT_INTENSITY_W_CM2 = 1.0


@dataclass(frozen=True)
class YanProfile:
    r"""Physical parameters for one paper figure.

    ``radius_nm`` is the metal-sphere radius :math:`R_0`; ``distance_nm`` is
    the centre-to-centre distance :math:`R_d`; ``omega0_eV`` is the bare QD
    transition energy :math:`\hbar\omega_0`.
    """

    label: str
    material_name: str
    radius_nm: float
    distance_nm: float
    eps_host: float
    eps_qd: float
    omega0_eV: float
    intensity_w_cm2: float
    material: MaterialDispersion
    completion_note: str


@dataclass(frozen=True)
class SpectralCurve:
    energy_eV: np.ndarray
    detuning_meV: np.ndarray
    power_material_rwa_W: np.ndarray
    power_fit_native_W: np.ndarray | None
    alpha_material_rwa_au3: np.ndarray
    alpha_fit_native_au3: np.ndarray | None

    @property
    def material_peak_index(self) -> int:
        return int(np.argmax(self.power_material_rwa_W))

    @property
    def material_peak_detuning_meV(self) -> float:
        return _quadratic_peak_coordinate(
            self.detuning_meV,
            self.power_material_rwa_W,
            self.material_peak_index,
        )

    @property
    def material_peak_power_W(self) -> float:
        return _quadratic_peak_value(
            self.detuning_meV,
            self.power_material_rwa_W,
            self.material_peak_index,
        )

    @property
    def fit_peak_detuning_meV(self) -> float | None:
        if self.power_fit_native_W is None:
            return None
        index = int(np.argmax(self.power_fit_native_W))
        return _quadratic_peak_coordinate(
            self.detuning_meV,
            self.power_fit_native_W,
            index,
        )

    @property
    def fit_peak_power_W(self) -> float | None:
        if self.power_fit_native_W is None:
            return None
        index = int(np.argmax(self.power_fit_native_W))
        return _quadratic_peak_value(
            self.detuning_meV,
            self.power_fit_native_W,
            index,
        )


@dataclass(frozen=True)
class HarmonicStateResult:
    energy_eV: float
    omega_au: float
    state_over_field: np.ndarray
    alpha_ode_au3: complex
    alpha_abk_au3: complex
    qd_dipole_over_field_au3: complex
    mnp_dipole_over_field_au3: complex
    mnp_field_over_field: complex
    max_port_relative_error: float


@dataclass(frozen=True)
class PeriodicTimeResult:
    material_name: str
    spatial_order: int
    energy_eV: float
    t_fs: np.ndarray
    incident_field_au: np.ndarray
    total_dipole_au: np.ndarray
    instantaneous_power_au: np.ndarray
    alpha_time_au3: complex
    alpha_reference_au3: complex
    alpha_relative_error: float
    mean_power_accumulator_W: float
    mean_power_quadrature_W: float
    mean_power_reference_W: float
    accumulator_vs_quadrature_relative_error: float
    mean_power_relative_error: float
    max_excited_population: float
    nfev: int
    cycles: int
    points_per_cycle: int


@dataclass(frozen=True)
class PlasmonSettlingResult:
    energy_eV: float
    t_fs: np.ndarray
    phase_cycles: np.ndarray
    modal_outputs_over_field: np.ndarray
    expected_modal_outputs_over_field: np.ndarray
    phasor_by_degree: np.ndarray
    expected_phasor_by_degree: np.ndarray
    max_relative_error: float
    settling_time_fs: float
    slowest_modal_decay_time_fs: float
    nfev: int


def johnson_christy_silver() -> MaterialDispersion:
    r"""Return the Johnson--Christy Ag table used for Fig. 3.

    The source stores vacuum wavelength in micrometres followed by ``n`` and
    ``k``.  Energy is :math:`E=hc/\lambda`; the arrays are reversed after the
    conversion so that ``MaterialDispersion`` receives increasing energy.
    """

    table = np.asarray(
        [
            (0.1879, 1.07, 1.212), (0.1916, 1.10, 1.232),
            (0.1953, 1.12, 1.255), (0.1993, 1.14, 1.277),
            (0.2033, 1.15, 1.296), (0.2073, 1.18, 1.312),
            (0.2119, 1.20, 1.325), (0.2164, 1.22, 1.336),
            (0.2214, 1.25, 1.342), (0.2262, 1.26, 1.344),
            (0.2313, 1.28, 1.357), (0.2371, 1.28, 1.367),
            (0.2426, 1.30, 1.378), (0.2490, 1.31, 1.389),
            (0.2551, 1.33, 1.393), (0.2616, 1.35, 1.387),
            (0.2689, 1.38, 1.372), (0.2761, 1.41, 1.331),
            (0.2844, 1.41, 1.264), (0.2924, 1.39, 1.161),
            (0.3009, 1.34, 0.964), (0.3107, 1.13, 0.616),
            (0.3204, 0.81, 0.392), (0.3315, 0.17, 0.829),
            (0.3425, 0.14, 1.142), (0.3542, 0.10, 1.419),
            (0.3679, 0.07, 1.657), (0.3815, 0.05, 1.864),
            (0.3974, 0.05, 2.070), (0.4133, 0.05, 2.275),
            (0.4305, 0.04, 2.462), (0.4509, 0.04, 2.657),
            (0.4714, 0.05, 2.869), (0.4959, 0.05, 3.093),
            (0.5209, 0.05, 3.324), (0.5486, 0.06, 3.586),
            (0.5821, 0.05, 3.858), (0.6168, 0.06, 4.152),
            (0.6595, 0.05, 4.483), (0.7045, 0.04, 4.838),
            (0.7560, 0.03, 5.242), (0.8211, 0.04, 5.727),
            (0.8920, 0.04, 6.312), (0.9840, 0.04, 6.992),
            (1.0880, 0.04, 7.795), (1.2160, 0.09, 8.828),
            (1.3930, 0.13, 10.10), (1.6100, 0.15, 11.85),
            (1.9370, 0.24, 14.08),
        ],
        dtype=float,
    )
    hc_eV_um = hbar * 2.0 * np.pi * C_SI / elementary_charge * 1.0e6
    energy = hc_eV_um / table[:, 0]
    order = np.argsort(energy)
    return MaterialDispersion(
        energy_eV=energy[order],
        n=table[order, 1],
        k=table[order, 2],
    )


def yan_profiles() -> tuple[YanProfile, YanProfile]:
    common_note = (
        "eps_qd=6, d=e*0.65 nm, T1=0.8 ns and T20=0.3 ns are not all "
        "restated in the Yan figure captions; they complete the profile from "
        "Yan's cited Zhang et al. CdSe-Au model."
    )
    return (
        YanProfile(
            label="Fig. 2(b)", material_name="Au", radius_nm=15.0,
            distance_nm=20.0, eps_host=1.0, eps_qd=6.0,
            omega0_eV=2.5, intensity_w_cm2=DEFAULT_INTENSITY_W_CM2,
            material=DEFAULT_AU_MATERIAL, completion_note=common_note,
        ),
        YanProfile(
            label="Fig. 3", material_name="Ag", radius_nm=15.0,
            distance_nm=22.0, eps_host=1.8, eps_qd=6.0,
            omega0_eV=3.34, intensity_w_cm2=DEFAULT_INTENSITY_W_CM2,
            material=johnson_christy_silver(),
            completion_note=(
                common_note
                + " Fig. 3 explicitly supplies Ag, water, R0, Rd and omega0; "
                "the 1 W/cm^2 intensity is carried over as an explicit "
                "comparison assumption because its caption does not restate it."
            ),
        ),
    )


def make_profile_params(profile: YanProfile) -> HybridSystemParams:
    params = make_params_with_overrides(
        c_nm=profile.radius_nm,
        a_nm=profile.radius_nm,
        r_nm=profile.distance_nm,
        qd_radius_nm=0.0,
        eps_m=profile.eps_host,
        eps_qd=profile.eps_qd,
        d_debye=TRANSITION_DIPOLE_DEBYE,
        omega0_ev=profile.omega0_eV,
        gamma_population_mev=GAMMA1_MEV,
        gamma2_coherence_mev=GAMMA2_MEV,
        qd_dipole_convention="bare_internal",
        orientation="long",
    )
    return replace(params, material=profile.material)


def incident_field_amplitude_au(intensity_w_cm2: float, eps_host: float) -> float:
    """Peak real field from cycle-averaged intensity in a lossless host."""

    if not np.isfinite(intensity_w_cm2) or intensity_w_cm2 <= 0.0:
        raise ValueError("intensity_w_cm2 must be finite and positive.")
    if not np.isfinite(eps_host) or eps_host <= 0.0:
        raise ValueError("eps_host must be finite and positive.")
    field_si = np.sqrt(
        2.0 * intensity_w_cm2 * 1.0e4
        / (np.sqrt(eps_host) * epsilon_0 * C_SI)
    )
    return float(field_si_to_au(field_si))


def rwa_qd_polarizability_from_params(
    params: HybridSystemParams,
    energies_eV: float | np.ndarray,
) -> np.ndarray:
    r"""Yan's rotating-wave weak-field QD polarizability.

    .. math::

       \beta_{\rm RWA}(\omega)=
       \frac{l_{\rm QD}^2 d^2}{\omega_0-\omega-i\Gamma_2}.

    ``d`` is the bare interband transition dipole, ``l_QD`` is the static
    local-field factor of the QD, ``omega0`` is its bare transition frequency,
    and ``Gamma2=1/T20`` is the coherence-decay rate.
    """

    omega = np.asarray(eV_to_au(np.asarray(energies_eV, dtype=float)), dtype=float)
    return np.asarray(
        params.qd_local_field_factor**2 * params.d_au**2
        / (params.omega0_au - omega - 1j * params.Gamma_au),
        dtype=complex,
    )


def _quadratic_peak_coordinate(x: np.ndarray, y: np.ndarray, index: int) -> float:
    if index <= 0 or index >= len(x) - 1:
        return float(x[index])
    coefficients = np.polyfit(x[index - 1 : index + 2], y[index - 1 : index + 2], 2)
    if coefficients[0] >= 0.0 or not np.all(np.isfinite(coefficients)):
        return float(x[index])
    vertex = -coefficients[1] / (2.0 * coefficients[0])
    if vertex < x[index - 1] or vertex > x[index + 1]:
        return float(x[index])
    return float(vertex)


def _quadratic_peak_value(x: np.ndarray, y: np.ndarray, index: int) -> float:
    coordinate = _quadratic_peak_coordinate(x, y, index)
    if index <= 0 or index >= len(x) - 1 or coordinate == float(x[index]):
        return float(y[index])
    coefficients = np.polyfit(x[index - 1 : index + 2], y[index - 1 : index + 2], 2)
    return float(np.polyval(coefficients, coordinate))


def _power_from_alpha(
    alpha_au3: np.ndarray,
    energies_eV: np.ndarray,
    *,
    eps_host: float,
    intensity_w_cm2: float,
) -> np.ndarray:
    cross_section = quasistatic_work_loss_cross_section_cm2(
        np.asarray(alpha_au3, dtype=complex),
        np.asarray(eV_to_au(energies_eV), dtype=float),
        eps_host,
    )
    power = np.asarray(cross_section, dtype=float) * intensity_w_cm2
    scale = max(float(np.max(np.abs(power))), np.finfo(float).tiny)
    if float(np.min(power)) < -1.0e-10 * scale:
        raise RuntimeError("The coupled strict-QS work spectrum is not passive.")
    return np.maximum(power, 0.0)


def calculate_spectral_curve(
    profile: YanProfile,
    params: HybridSystemParams,
    kernel: SpheroidGreenInteraction,
    *,
    spatial_order: int,
    detuning_window_meV: tuple[float, float],
    points: int,
    full_model: FullQSSpheroidPulseModel | None = None,
) -> SpectralCurve:
    """Calculate one near-exciton paper curve and optional ODE-fit curve."""

    if points < 3 or not isinstance(points, (int, np.integer)):
        raise ValueError("points must be an integer at least 3.")
    if not 1 <= spatial_order <= kernel.n_max:
        raise ValueError("spatial_order must lie inside the kernel truncation.")
    detuning = np.linspace(*detuning_window_meV, int(points))
    energies = profile.omega0_eV + detuning * 1.0e-3
    response_material = kernel.response_from_material(profile.material, energies).truncate(
        spatial_order
    )
    beta_rwa = rwa_qd_polarizability_from_params(params, energies)
    material_solution = solve_linear_hybrid_response(
        response_material,
        beta_rwa,
        eps_m=profile.eps_host,
    )
    alpha_material = material_solution.alpha_effective_au3
    power_material = _power_from_alpha(
        alpha_material,
        energies,
        eps_host=profile.eps_host,
        intensity_w_cm2=profile.intensity_w_cm2,
    )

    alpha_fit: np.ndarray | None = None
    power_fit: np.ndarray | None = None
    if full_model is not None:
        if full_model.n_spatial_modes < spatial_order:
            raise ValueError("full_model has fewer spatial modes than requested.")
        response_fit = full_model.frequency_response_from_fit(energies).truncate(
            spatial_order
        )
        beta_native = qd_linear_polarizability_from_params(params, energies)
        fit_solution = solve_linear_hybrid_response(
            response_fit,
            beta_native,
            eps_m=profile.eps_host,
        )
        alpha_fit = fit_solution.alpha_effective_au3
        power_fit = _power_from_alpha(
            alpha_fit,
            energies,
            eps_host=profile.eps_host,
            intensity_w_cm2=profile.intensity_w_cm2,
        )
    return SpectralCurve(
        energy_eV=np.asarray(energies),
        detuning_meV=np.asarray(detuning),
        power_material_rwa_W=power_material,
        power_fit_native_W=power_fit,
        alpha_material_rwa_au3=np.asarray(alpha_material),
        alpha_fit_native_au3=None if alpha_fit is None else np.asarray(alpha_fit),
    )


def calculate_overview_curve(
    profile: YanProfile,
    params: HybridSystemParams,
    kernel: SpheroidGreenInteraction,
    *,
    spatial_order: int,
    energy_window_eV: tuple[float, float],
    points: int,
    full_model: FullQSSpheroidPulseModel | None,
) -> SpectralCurve:
    detuning_window = (
        (energy_window_eV[0] - profile.omega0_eV) * 1.0e3,
        (energy_window_eV[1] - profile.omega0_eV) * 1.0e3,
    )
    return calculate_spectral_curve(
        profile,
        params,
        kernel,
        spatial_order=spatial_order,
        detuning_window_meV=detuning_window,
        points=points,
        full_model=full_model,
    )


def build_full_qs_models(
    profile: YanProfile,
    params: HybridSystemParams,
    *,
    n_max: int = 10,
    modal_audit_points: int = 501,
) -> tuple[HybridQDPlasmonModel, FullQSSpheroidPulseModel, FullQSSpheroidPulseModel]:
    """Build the common material fit and the article N=1/N=10 ODEs."""

    if profile.material_name == "Au":
        n_modes = 9
        fit_window = (0.8, 3.0)
        weight_center = None
        weight_sigma = None
    elif profile.material_name == "Ag":
        # The Ag table is much sharper and sparser around 3.3 eV.  This narrow
        # causal realization is an ODE approximation audit, while the paper
        # curves always retain the direct material data as their authority.
        n_modes = 3
        fit_window = (2.8, 3.8)
        weight_center = 3.34
        weight_sigma = 0.32
    else:
        raise ValueError(f"Unsupported literature material {profile.material_name!r}.")

    bright = HybridQDPlasmonModel(
        params,
        orientation="long",
        n_modes=n_modes,
        fit_window_eV=fit_window,
        weight_center_eV=weight_center,
        weight_sigma_eV=weight_sigma,
        max_fit_normalized_rms=None,
        max_fit_pointwise_relative_error=None,
        radiative_consistency_policy="ignore",
        verbose=False,
    )
    kernel_nmax = SpheroidGreenInteraction.from_params(
        params, orientation="long", n_max=n_max
    )
    full_nmax = FullQSSpheroidPulseModel(
        bright,
        kernel_nmax,
        fit_quality_policy="ignore" if profile.material_name == "Ag" else "raise",
        spatial_convergence_policy="ignore",
        modal_audit_points=modal_audit_points,
    )
    full_n1 = FullQSSpheroidPulseModel(
        bright,
        SpheroidGreenInteraction.from_params(params, orientation="long", n_max=1),
        fit_quality_policy="ignore" if profile.material_name == "Ag" else "raise",
        spatial_convergence_policy="ignore",
        modal_audit_points=modal_audit_points,
    )
    return bright, full_n1, full_nmax


def _relative_error(value: complex | float, reference: complex | float) -> float:
    return float(abs(value - reference) / max(abs(reference), np.finfo(float).tiny))


def harmonic_state_from_modal_ode(
    model: FullQSSpheroidPulseModel,
    energy_eV: float,
) -> HarmonicStateResult:
    """Solve ``(-i*omega I-M)x=b`` for the public modal ODE realization."""

    omega = float(eV_to_au(energy_eV))
    unit_pulse = GaussianPulse(
        E0_au=1.0,
        omegaL_au=omega,
        tau_au=float(ns_to_au(1.0)),
        tau_kind="fwhm_intensity",
    )
    ground = model.initial_state()
    derivative = model.rhs(0.0, ground, unit_pulse)
    b = np.concatenate(
        (
            derivative[: model.mode_state_count],
            derivative[model.Q_index : model.P_index + 1],
        )
    )
    matrix = model.linearized_ground_state_matrix().toarray()
    state = np.linalg.solve(
        -1j * omega * np.eye(matrix.shape[0], dtype=complex) - matrix,
        b.astype(complex),
    )

    modal_state = state[: model.mode_state_count].reshape(
        model.n_spatial_modes, model.n_material_modes, 2
    )
    q_sum = np.sum(modal_state[:, :, 0], axis=1)
    P = state[-1]
    local = model.params.qd_local_field_factor
    mu_d = local * model.params.d_au * P
    external = np.full(model.n_spatial_modes, mu_d, dtype=complex)
    external[0] = 1.0 + model.bright_coupling_au_minus3 * mu_d
    internal = (external - model.delta_L * q_sum) / model.feedback_denominator
    outputs = model.alpha_inf * internal + q_sum
    mu_p = model.C * outputs[0]
    mnp_field = (
        model.bright_coupling_au_minus3 * mu_p
        + np.dot(model.reaction_weights_au_minus3[1:], outputs[1:])
    )
    alpha_ode = (mu_p + mu_d) / model.params.eps_m

    energy_array = np.asarray([energy_eV])
    response = model.frequency_response_from_fit(energy_array)
    beta = qd_linear_polarizability_from_params(model.params, energy_array)
    reference = solve_linear_hybrid_response(
        response, beta, eps_m=model.params.eps_m
    )
    alpha_reference = complex(reference.alpha_effective_au3[0])
    qd_reference = complex(reference.qd_dipole_over_field_au3[0])
    mnp_reference = complex(reference.mnp_dipole_over_field_au3[0])
    field_reference = complex(
        response.B[0] + response.K_au_minus3[0] * qd_reference
    )
    errors = (
        _relative_error(alpha_ode, alpha_reference),
        _relative_error(mu_d, qd_reference),
        _relative_error(mu_p, mnp_reference),
        _relative_error(mnp_field, field_reference),
    )
    return HarmonicStateResult(
        energy_eV=float(energy_eV),
        omega_au=omega,
        state_over_field=np.asarray(state),
        alpha_ode_au3=complex(alpha_ode),
        alpha_abk_au3=alpha_reference,
        qd_dipole_over_field_au3=complex(mu_d),
        mnp_dipole_over_field_au3=complex(mu_p),
        mnp_field_over_field=complex(mnp_field),
        max_port_relative_error=max(errors),
    )


def _state_observables(
    model: FullQSSpheroidPulseModel,
    times: np.ndarray,
    states: np.ndarray,
    pulse: GaussianPulse,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    incident = np.asarray(pulse.field(times), dtype=float)
    total_dipole = np.empty(times.size, dtype=float)
    work_dot = np.empty(times.size, dtype=float)
    for index, (time, state) in enumerate(zip(times, states.T)):
        derivative = model.rhs(float(time), state, pulse)
        # Reconstruct p_QD and p_MNP using only public model coefficients.
        modal = state[: model.mode_state_count].reshape(
            model.n_spatial_modes, model.n_material_modes, 2
        )
        q_sum = np.sum(modal[:, :, 0], axis=1)
        mu_d = model.params.qd_local_field_factor * model.params.d_au * state[
            model.P_index
        ]
        external = np.full(model.n_spatial_modes, mu_d, dtype=float)
        external[0] = incident[index] + model.bright_coupling_au_minus3 * mu_d
        internal = (external - model.delta_L * q_sum) / model.feedback_denominator
        outputs = model.alpha_inf * internal + q_sum
        mu_p = model.C * outputs[0]
        total_dipole[index] = mu_p + mu_d
        work_dot[index] = derivative[model.work_index]
    return incident, total_dipole, work_dot


def validate_periodic_time_ode(
    model: FullQSSpheroidPulseModel,
    *,
    material_name: str,
    energy_eV: float,
    intensity_w_cm2: float,
    cycles: int = 40,
    analysis_cycles: int = 20,
    points_per_cycle: int = 40,
) -> PeriodicTimeResult:
    """Integrate the real-carrier nonlinear RHS from its periodic weak-field orbit."""

    if cycles < analysis_cycles or analysis_cycles < 2:
        raise ValueError("Require cycles >= analysis_cycles >= 2.")
    if points_per_cycle < 16:
        raise ValueError("points_per_cycle must be at least 16.")
    harmonic = harmonic_state_from_modal_ode(model, energy_eV)
    E0 = incident_field_amplitude_au(intensity_w_cm2, model.params.eps_m)
    initial = model.initial_state()
    initial[: model.mode_state_count] = E0 * harmonic.state_over_field[
        : model.mode_state_count
    ].real
    initial[model.Q_index] = E0 * harmonic.state_over_field[-2].real
    initial[model.P_index] = E0 * harmonic.state_over_field[-1].real
    initial[model.work_index] = 0.0

    omega = harmonic.omega_au
    period = 2.0 * np.pi / omega
    pulse = GaussianPulse(
        E0_au=E0,
        omegaL_au=omega,
        tau_au=float(ns_to_au(1.0)),
        tau_kind="fwhm_intensity",
    )
    times = np.linspace(0.0, cycles * period, cycles * points_per_cycle + 1)
    solution = solve_ivp(
        lambda time, state: model.rhs(time, state, pulse),
        (0.0, float(times[-1])),
        initial,
        method="DOP853",
        t_eval=times,
        rtol=2.0e-9,
        atol=1.0e-12,
        max_step=period / points_per_cycle,
    )
    if not solution.success or solution.t.size != times.size:
        raise RuntimeError(f"Periodic full-QS solve failed: {solution.message}")
    incident, total_dipole, work_dot = _state_observables(
        model, solution.t, solution.y, pulse
    )
    start = (cycles - analysis_cycles) * points_per_cycle
    tail_t = solution.t[start:]
    duration = float(tail_t[-1] - tail_t[0])
    phase = np.exp(1j * omega * tail_t)
    E_hat = 2.0 / duration * np.trapezoid(incident[start:] * phase, tail_t)
    p_hat = 2.0 / duration * np.trapezoid(total_dipole[start:] * phase, tail_t)
    alpha_time = p_hat / E_hat / model.params.eps_m
    alpha_reference = harmonic.alpha_abk_au3

    work_accumulator = solution.y[model.work_index]
    mean_accumulator_au = float(
        (work_accumulator[-1] - work_accumulator[start]) / duration
    )
    mean_quadrature_au = float(
        np.trapezoid(work_dot[start:], tail_t) / duration
    )
    dipole_over_field_reference = model.params.eps_m * alpha_reference
    mean_reference_au = float(
        0.5
        * omega
        * dipole_over_field_reference.imag
        * abs(E_hat) ** 2
    )
    power_conversion = AU_ENERGY_J / AU_TIME_S
    rho22 = 0.5 * (solution.y[model.W_index] + 1.0)
    return PeriodicTimeResult(
        material_name=material_name,
        spatial_order=model.n_spatial_modes,
        energy_eV=float(energy_eV),
        t_fs=np.asarray(au_to_fs(solution.t)),
        incident_field_au=incident,
        total_dipole_au=total_dipole,
        instantaneous_power_au=work_dot,
        alpha_time_au3=complex(alpha_time),
        alpha_reference_au3=complex(alpha_reference),
        alpha_relative_error=_relative_error(alpha_time, alpha_reference),
        mean_power_accumulator_W=mean_accumulator_au * power_conversion,
        mean_power_quadrature_W=mean_quadrature_au * power_conversion,
        mean_power_reference_W=mean_reference_au * power_conversion,
        accumulator_vs_quadrature_relative_error=_relative_error(
            mean_accumulator_au, mean_quadrature_au
        ),
        mean_power_relative_error=_relative_error(
            mean_accumulator_au, mean_reference_au
        ),
        max_excited_population=float(np.max(rho22)),
        nfev=int(solution.nfev),
        cycles=int(cycles),
        points_per_cycle=int(points_per_cycle),
    )


def validate_plasmon_settling(
    model: FullQSSpheroidPulseModel,
    *,
    energy_eV: float,
    decay_times: float = 12.0,
    analysis_cycles: int = 20,
    points_per_cycle: int = 40,
) -> PlasmonSettlingResult:
    """Start Lorentz coordinates at zero and verify every spatial transfer.

    Each spatial order is driven by the same diagnostic harmonic source.  A
    uniform physical laser drives only the bright degree ``n=1``; applying a
    source to every order here is a transfer-function unit test that audits the
    whole spheroidal expansion, including the dark reaction modes.
    """

    omega = float(eV_to_au(energy_eV))
    period = 2.0 * np.pi / omega
    slowest_rate = -float(np.max(model.modal_poles_au.real))
    if not np.isfinite(slowest_rate) or slowest_rate <= 0.0:
        raise RuntimeError("No positive modal decay rate is available.")
    slowest_time = 1.0 / slowest_rate
    end = max(decay_times * slowest_time, (analysis_cycles + 2) * period)
    tail_start = end - analysis_cycles * period
    sample_times = np.linspace(
        tail_start,
        end,
        analysis_cycles * points_per_cycle + 1,
    )
    E0 = 1.0e-8
    state_size = model.mode_state_count

    def modal_rhs(time: float, state: np.ndarray) -> np.ndarray:
        reshaped = state.reshape(
            model.n_spatial_modes, model.n_material_modes, 2
        )
        q = reshaped[:, :, 0]
        velocity = reshaped[:, :, 1]
        q_sum = np.sum(q, axis=1)
        external = E0 * np.cos(omega * time)
        internal = (
            external - model.delta_L * q_sum
        ) / model.feedback_denominator
        derivative = np.zeros_like(reshaped)
        derivative[:, :, 0] = velocity
        derivative[:, :, 1] = (
            model.fit.strengths_au2[None, :] * internal[:, None]
            - model.fit.gamma_modes_au[None, :] * velocity
            - model.fit.omega_modes_au[None, :] ** 2 * q
        )
        return derivative.reshape(state_size)

    solution = solve_ivp(
        modal_rhs,
        (0.0, end),
        np.zeros(state_size),
        method="DOP853",
        t_eval=sample_times,
        rtol=2.0e-9,
        atol=1.0e-12,
        max_step=period / points_per_cycle,
    )
    if not solution.success or solution.t.size != sample_times.size:
        raise RuntimeError(f"Plasmon settling solve failed: {solution.message}")
    reshaped = solution.y.T.reshape(
        sample_times.size, model.n_spatial_modes, model.n_material_modes, 2
    )
    q_sum = np.sum(reshaped[:, :, :, 0], axis=2)
    incident = E0 * np.cos(omega * sample_times)
    internal = (
        incident[:, None] - q_sum * model.delta_L[None, :]
    ) / model.feedback_denominator[None, :]
    outputs = model.alpha_inf * internal + q_sum
    duration = float(sample_times[-1] - sample_times[0])
    phase = np.exp(1j * omega * sample_times)
    phasor = 2.0 / duration * np.trapezoid(
        outputs * phase[:, None], sample_times, axis=0
    ) / E0
    expected = model.modal_susceptibility_from_fit(np.asarray([energy_eV]))[:, 0]
    relative = np.abs(phasor - expected) / np.maximum(
        np.abs(expected), np.finfo(float).tiny
    )
    expected_time = np.real(
        expected[None, :] * np.exp(-1j * omega * sample_times[:, None])
    )
    return PlasmonSettlingResult(
        energy_eV=float(energy_eV),
        t_fs=np.asarray(au_to_fs(sample_times)),
        phase_cycles=np.asarray(sample_times / period),
        modal_outputs_over_field=np.asarray(outputs / E0),
        expected_modal_outputs_over_field=np.asarray(expected_time),
        phasor_by_degree=np.asarray(phasor),
        expected_phasor_by_degree=np.asarray(expected),
        max_relative_error=float(np.max(relative)),
        settling_time_fs=float(au_to_fs(end)),
        slowest_modal_decay_time_fs=float(au_to_fs(slowest_time)),
        nfev=int(solution.nfev),
    )


def calculate_fig2b_distance_dependence(
    profile: YanProfile,
    params: HybridSystemParams,
    *,
    distance_ratios: np.ndarray | None = None,
    points: int = 4401,
) -> dict[str, np.ndarray]:
    """Reconstruct Fig. 2(b)'s main distance-dependent peak-shift panel."""

    if distance_ratios is None:
        distance_ratios = PAPER_FIG2B_DISTANCE_RASTER["distance_nm"] / profile.radius_nm
    ratios = np.asarray(distance_ratios, dtype=float)
    if ratios.ndim != 1 or np.any(ratios <= 1.0):
        raise ValueError("distance_ratios must be a 1-D array above one.")
    shifts_n1 = np.empty(ratios.size)
    shifts_n10 = np.empty(ratios.size)
    for index, ratio in enumerate(ratios):
        distance_nm = profile.radius_nm * float(ratio)
        local_profile = replace(profile, distance_nm=distance_nm)
        local_params = replace(
            params,
            R_au=float(params.c_au * ratio),
        )
        kernel = SpheroidGreenInteraction.from_params(
            local_params, orientation="long", n_max=10
        )
        for order, destination in ((1, shifts_n1), (10, shifts_n10)):
            curve = calculate_spectral_curve(
                local_profile,
                local_params,
                kernel,
                spatial_order=order,
                detuning_window_meV=(-2.0, 0.2),
                points=points,
            )
            if curve.material_peak_index in {0, points - 1}:
                raise RuntimeError(
                    f"Peak search hit a boundary at Rd/R0={ratio:g}, N={order}."
                )
            destination[index] = curve.material_peak_detuning_meV
    return {
        "distance_ratio": ratios,
        "log10_distance_ratio": np.log10(ratios),
        "shift_N1_meV": shifts_n1,
        "shift_N10_meV": shifts_n10,
        "log10_abs_shift_N1_meV": np.log10(np.abs(shifts_n1)),
        "log10_abs_shift_N10_meV": np.log10(np.abs(shifts_n10)),
    }


def _write_curve_csv(path: Path, curve: SpectralCurve) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "energy_eV", "detuning_meV", "power_material_rwa_W",
                "power_causal_fit_native_W", "alpha_material_rwa_real_au3",
                "alpha_material_rwa_imag_au3", "alpha_fit_native_real_au3",
                "alpha_fit_native_imag_au3",
            ]
        )
        for index in range(curve.energy_eV.size):
            alpha_fit = None if curve.alpha_fit_native_au3 is None else curve.alpha_fit_native_au3[index]
            power_fit = None if curve.power_fit_native_W is None else curve.power_fit_native_W[index]
            writer.writerow(
                [
                    curve.energy_eV[index], curve.detuning_meV[index],
                    curve.power_material_rwa_W[index], power_fit,
                    curve.alpha_material_rwa_au3[index].real,
                    curve.alpha_material_rwa_au3[index].imag,
                    None if alpha_fit is None else alpha_fit.real,
                    None if alpha_fit is None else alpha_fit.imag,
                ]
            )


def _write_time_csv(path: Path, result: PeriodicTimeResult) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["time_fs", "incident_field_au", "total_dipole_au", "instantaneous_power_au"]
        )
        writer.writerows(
            zip(
                result.t_fs,
                result.incident_field_au,
                result.total_dipole_au,
                result.instantaneous_power_au,
            )
        )


def _plot_fig2b_spectrum(
    run_dir: Path,
    n1: SpectralCurve,
    n10: SpectralCurve,
) -> Figure:
    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for curve, order, color in ((n1, 1, "black"), (n10, 10, "tab:red")):
        axis.plot(
            curve.detuning_meV,
            curve.power_material_rwa_W / 1.0e-10,
            color=color,
            lw=2.0,
            label=f"N={order}: J&C + Yan RWA",
        )
        if curve.power_fit_native_W is not None:
            axis.plot(
                curve.detuning_meV,
                curve.power_fit_native_W / 1.0e-10,
                color=color,
                lw=1.2,
                ls="--",
                label=f"N={order}: causal ODE fit",
            )
        anchor = PAPER_RASTER_ANCHORS[f"fig2b_N{order}"]
        axis.errorbar(
            anchor["detuning_meV"], anchor["power_1e10_W"],
            xerr=anchor["detuning_uncertainty_meV"],
            yerr=anchor["power_uncertainty_1e10_W"],
            fmt="o" if order == 10 else "s", mfc="none", mec=color,
            ecolor=color, capsize=2.5, label=f"paper raster N={order}",
        )
    axis.set_xlabel(r"$\hbar(\omega-\omega_0)$ (meV)")
    axis.set_ylabel(r"total work rate ($10^{-10}$ W)")
    axis.set_title("Yan et al. (2008), Fig. 2(b) inset: Au sphere")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.savefig(run_dir / FIG2B_SPECTRUM_PNG, dpi=220)
    return figure


def _plot_fig2b_distance(run_dir: Path, distance: dict[str, np.ndarray]) -> Figure:
    figure, axis = plt.subplots(figsize=(6.7, 4.8), constrained_layout=True)
    axis.plot(
        distance["log10_distance_ratio"],
        distance["log10_abs_shift_N10_meV"],
        "o-", color="tab:red", ms=4, label="full-QS N=10",
    )
    axis.plot(
        distance["log10_distance_ratio"],
        distance["log10_abs_shift_N1_meV"],
        "s--", color="black", ms=4, label="dipole N=1",
    )
    raster = PAPER_FIG2B_DISTANCE_RASTER
    axis.errorbar(
        raster["log10_distance_ratio"],
        raster["log10_abs_shift_N10_meV"],
        xerr=raster["x_uncertainty"], yerr=raster["y_uncertainty"],
        fmt="o", mfc="none", mec="tab:red", ecolor="tab:red",
        ms=5, capsize=1.5, label="paper raster N=10",
    )
    axis.errorbar(
        raster["log10_distance_ratio"],
        raster["log10_abs_shift_N1_meV"],
        xerr=raster["x_uncertainty"], yerr=raster["y_uncertainty"],
        fmt="s", mfc="none", mec="black", ecolor="black",
        ms=5, capsize=1.5, label="paper raster N=1",
    )
    axis.set_xlabel(r"$\log_{10}(R_d/R_0)$")
    axis.set_ylabel(r"$\log_{10}(|\hbar(\omega_{peak}-\omega_0)|/\mathrm{meV})$")
    axis.set_title("Yan et al. (2008), Fig. 2(b): distance-dependent shift")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(run_dir / FIG2B_DISTANCE_PNG, dpi=220)
    return figure


def _plot_fig3(
    run_dir: Path,
    overview: SpectralCurve,
    n1: SpectralCurve,
    n10: SpectralCurve,
) -> Figure:
    figure = plt.figure(figsize=(9.2, 7.0), constrained_layout=True)
    grid = figure.add_gridspec(2, 2)
    top = figure.add_subplot(grid[0, :])
    left = figure.add_subplot(grid[1, 0])
    right = figure.add_subplot(grid[1, 1])
    top.plot(overview.energy_eV, overview.power_material_rwa_W / 1.0e-10, color="tab:red", lw=2)
    if overview.power_fit_native_W is not None:
        top.plot(overview.energy_eV, overview.power_fit_native_W / 1.0e-10, "--", color="tab:red", lw=1.2)
    anchor = PAPER_RASTER_ANCHORS["fig3_plasmon"]
    top.errorbar(
        anchor["energy_eV"], anchor["power_1e10_W"],
        xerr=anchor["energy_uncertainty_eV"], yerr=anchor["power_uncertainty_1e10_W"],
        fmt="o", mfc="none", color="black", capsize=2.5, label="paper raster",
    )
    top.set(xlabel=r"$\hbar\omega$ (eV)", ylabel=r"$Q_{tot}$ ($10^{-10}$ W)", title="Fig. 3: Ag plasmon overview, N=10")
    top.grid(alpha=0.25)
    top.legend(fontsize=8)

    for axis, curve, order, color in ((left, n10, 10, "tab:red"), (right, n1, 1, "tab:blue")):
        axis.plot(curve.detuning_meV, curve.power_material_rwa_W / 1.0e-10, color=color, lw=2, label="J&C + Yan RWA")
        if curve.power_fit_native_W is not None:
            axis.plot(curve.detuning_meV, curve.power_fit_native_W / 1.0e-10, "--", color=color, lw=1.2, label="causal ODE fit")
        anchor = PAPER_RASTER_ANCHORS[f"fig3_N{order}"]
        axis.errorbar(
            anchor["detuning_meV"], anchor["power_1e10_W"],
            xerr=anchor["detuning_uncertainty_meV"], yerr=anchor["power_uncertainty_1e10_W"],
            fmt="o", mfc="none", color="black", capsize=2.5, label="paper raster",
        )
        axis.set_xlabel(r"$\hbar(\omega-\omega_0)$ (meV)")
        axis.set_ylabel(r"$Q_{tot}$ ($10^{-10}$ W)")
        axis.set_title(f"exciton zoom, N={order}")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    figure.savefig(run_dir / FIG3_PNG, dpi=220)
    return figure


def _plot_modal_validation(
    run_dir: Path,
    settling: PlasmonSettlingResult,
    time_results: Sequence[PeriodicTimeResult],
) -> Figure:
    figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), constrained_layout=True)
    last_four = settling.phase_cycles >= settling.phase_cycles[-1] - 4.0
    for degree_index, color in ((0, "tab:blue"), (-1, "tab:red")):
        degree = degree_index + 1 if degree_index >= 0 else settling.modal_outputs_over_field.shape[1]
        axes[0, 0].plot(
            settling.phase_cycles[last_four],
            settling.modal_outputs_over_field[last_four, degree_index],
            color=color, lw=1.6, label=f"ODE n={degree}",
        )
        axes[0, 0].plot(
            settling.phase_cycles[last_four],
            settling.expected_modal_outputs_over_field[last_four, degree_index],
            color=color, ls="--", lw=1.0, label=f"transfer n={degree}",
        )
    axes[0, 0].set(xlabel="optical cycles", ylabel=r"$\chi_n(t)$", title="plasmon modes after zero-state settling")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7, ncol=2)

    axes[0, 1].semilogy(
        np.arange(1, settling.phasor_by_degree.size + 1),
        np.abs(settling.phasor_by_degree - settling.expected_phasor_by_degree)
        / np.maximum(np.abs(settling.expected_phasor_by_degree), np.finfo(float).tiny),
        "o-",
    )
    axes[0, 1].set(xlabel="spheroidal degree n", ylabel="relative phasor error", title="all plasmon transfer channels")
    axes[0, 1].grid(alpha=0.25)

    for column, result in enumerate(time_results[:2]):
        axis = axes[1, column]
        period_fs = (result.t_fs[-1] - result.t_fs[0]) / result.cycles
        tail = result.t_fs >= result.t_fs[-1] - 4.0 * period_fs
        E = result.incident_field_au[tail]
        p = result.total_dipole_au[tail]
        axis.plot(result.t_fs[tail], E / np.max(np.abs(E)), label="E/E0", lw=1.2)
        axis.plot(result.t_fs[tail], p / np.max(np.abs(p)), label="p/max|p|", lw=1.2)
        axis.set(
            xlabel="time (fs)", ylabel="normalized carrier",
            title=f"full RHS: {result.material_name}, N={result.spatial_order}",
        )
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.savefig(run_dir / MODAL_ODE_PNG, dpi=220)
    return figure


def _peak_payload(curve: SpectralCurve, anchor_key: str) -> dict[str, object]:
    anchor = PAPER_RASTER_ANCHORS[anchor_key]
    fit_detuning = curve.fit_peak_detuning_meV
    fit_power = curve.fit_peak_power_W
    return {
        "material_rwa": {
            "detuning_meV": curve.material_peak_detuning_meV,
            "power_W": curve.material_peak_power_W,
        },
        "causal_fit_native": None if fit_detuning is None else {
            "detuning_meV": fit_detuning,
            "power_W": fit_power,
        },
        "paper_raster": anchor,
        "material_minus_paper_detuning_meV": (
            curve.material_peak_detuning_meV - anchor["detuning_meV"]
        ),
        "material_minus_paper_power_W": (
            curve.material_peak_power_W - anchor["power_1e10_W"] * 1.0e-10
        ),
        "fit_vs_material_shift_relative": None if fit_detuning is None else (
            abs(fit_detuning - curve.material_peak_detuning_meV)
            / max(abs(curve.material_peak_detuning_meV), np.finfo(float).tiny)
        ),
    }


def _distance_comparison_payload(distance: dict[str, np.ndarray]) -> dict[str, object]:
    raster = PAPER_FIG2B_DISTANCE_RASTER
    calculated_x = distance["log10_distance_ratio"]
    target_x = raster["log10_distance_ratio"]
    payload: dict[str, object] = {
        "paper_x_uncertainty": raster["x_uncertainty"],
        "paper_y_uncertainty": raster["y_uncertainty"],
    }
    for order in (1, 10):
        calculated_y = np.interp(
            target_x,
            calculated_x,
            distance[f"log10_abs_shift_N{order}_meV"],
        )
        paper_y = raster[f"log10_abs_shift_N{order}_meV"]
        residual = calculated_y - paper_y
        payload[f"N{order}"] = {
            "rms_log10_residual": float(np.sqrt(np.mean(residual**2))),
            "max_abs_log10_residual": float(np.max(np.abs(residual))),
            "residual_in_raster_y_uncertainties": (
                residual / float(raster["y_uncertainty"])
            ).tolist(),
        }
    return payload


def _complex_payload(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def _harmonic_payload(result: HarmonicStateResult) -> dict[str, object]:
    return {
        "energy_eV": result.energy_eV,
        "spatial_state_dimension": int(result.state_over_field.size),
        "alpha_ode_au3": _complex_payload(result.alpha_ode_au3),
        "alpha_abk_au3": _complex_payload(result.alpha_abk_au3),
        "max_A_B_K_port_relative_error": result.max_port_relative_error,
    }


def _time_payload(result: PeriodicTimeResult) -> dict[str, object]:
    return {
        "material": result.material_name,
        "N": result.spatial_order,
        "energy_eV": result.energy_eV,
        "cycles": result.cycles,
        "points_per_cycle": result.points_per_cycle,
        "nfev": result.nfev,
        "alpha_time_au3": _complex_payload(result.alpha_time_au3),
        "alpha_reference_au3": _complex_payload(result.alpha_reference_au3),
        "alpha_relative_error": result.alpha_relative_error,
        "mean_power_accumulator_W": result.mean_power_accumulator_W,
        "mean_power_quadrature_W": result.mean_power_quadrature_W,
        "mean_power_reference_W": result.mean_power_reference_W,
        "accumulator_vs_quadrature_relative_error": result.accumulator_vs_quadrature_relative_error,
        "mean_power_relative_error": result.mean_power_relative_error,
        "max_excited_population": result.max_excited_population,
        "acceptance": {
            "alpha_below_1e-5": bool(result.alpha_relative_error < 1.0e-5),
            "power_below_1e-5": bool(result.mean_power_relative_error < 1.0e-5),
            "weak_field_rho22_below_1e-7": bool(result.max_excited_population < 1.0e-7),
        },
    }


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _create_unique_run_dir(output_dir: str | Path) -> Path:
    first = timestamped_run_dir(output_dir)
    for suffix in range(1000):
        candidate = first if suffix == 0 else first.with_name(f"{first.name}_{suffix:02d}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError("Could not reserve a unique timestamped output directory.")


def run_validation(
    *,
    output_dir: str | Path = "results/literature_yan2008_full_qs",
    article_pdf: str | Path = DEFAULT_ARTICLE_PDF,
    make_plots: bool = True,
    show: bool = False,
    run_time_domain: bool = True,
    quick: bool = False,
) -> Path:
    """Run both paper reconstructions and the modal/time-domain audits."""

    run_dir = _create_unique_run_dir(output_dir)
    au_profile, ag_profile = yan_profiles()
    au_params = make_profile_params(au_profile)
    ag_params = make_profile_params(ag_profile)

    # Build the ODE realizations first, then use exactly those fit objects for
    # the dashed frequency references.
    au_bright, au_n1_model, au_n10_model = build_full_qs_models(
        au_profile, au_params, modal_audit_points=201 if quick else 501
    )
    ag_bright, ag_n1_model, ag_n10_model = build_full_qs_models(
        ag_profile, ag_params, modal_audit_points=201 if quick else 501
    )
    au_kernel = au_n10_model.kernel
    ag_kernel = ag_n10_model.kernel

    fig2_points = 1201 if quick else 3001
    fig2_n1 = calculate_spectral_curve(
        au_profile, au_params, au_kernel, spatial_order=1,
        detuning_window_meV=(-0.5, 0.1), points=fig2_points,
        full_model=au_n10_model,
    )
    fig2_n10 = calculate_spectral_curve(
        au_profile, au_params, au_kernel, spatial_order=10,
        detuning_window_meV=(-0.5, 0.1), points=fig2_points,
        full_model=au_n10_model,
    )
    distance = calculate_fig2b_distance_dependence(
        au_profile, au_params,
        distance_ratios=(
            PAPER_FIG2B_DISTANCE_RASTER["distance_nm"][[0, 2, 4, 6, 8, 9]]
            / au_profile.radius_nm
            if quick
            else PAPER_FIG2B_DISTANCE_RASTER["distance_nm"] / au_profile.radius_nm
        ),
        points=1801 if quick else 4401,
    )

    fig3_overview = calculate_overview_curve(
        ag_profile, ag_params, ag_kernel, spatial_order=10,
        energy_window_eV=(3.1, 3.5), points=801 if quick else 2001,
        full_model=ag_n10_model,
    )
    fig3_n10 = calculate_spectral_curve(
        ag_profile, ag_params, ag_kernel, spatial_order=10,
        detuning_window_meV=(-10.0, 10.0), points=4001 if quick else 10001,
        full_model=ag_n10_model,
    )
    fig3_n1 = calculate_spectral_curve(
        ag_profile, ag_params, ag_kernel, spatial_order=1,
        detuning_window_meV=(-2.0, 2.0), points=8001 if quick else 20001,
        full_model=ag_n10_model,
    )

    harmonic_checks = {
        "Au_N1": harmonic_state_from_modal_ode(
            au_n1_model, fig2_n1.fit_peak_detuning_meV * 1.0e-3 + au_profile.omega0_eV
        ),
        "Au_N10": harmonic_state_from_modal_ode(
            au_n10_model, fig2_n10.fit_peak_detuning_meV * 1.0e-3 + au_profile.omega0_eV
        ),
        "Ag_N1": harmonic_state_from_modal_ode(
            ag_n1_model, fig3_n1.fit_peak_detuning_meV * 1.0e-3 + ag_profile.omega0_eV
        ),
        "Ag_N10": harmonic_state_from_modal_ode(
            ag_n10_model, fig3_n10.fit_peak_detuning_meV * 1.0e-3 + ag_profile.omega0_eV
        ),
    }
    worst_harmonic_error = max(
        result.max_port_relative_error for result in harmonic_checks.values()
    )
    if worst_harmonic_error >= 1.0e-8:
        raise RuntimeError(
            "The modal state-space realization failed the A/B/K port check: "
            f"max relative error={worst_harmonic_error:.6g}."
        )

    settling: PlasmonSettlingResult | None = None
    time_results: list[PeriodicTimeResult] = []
    if run_time_domain:
        settling = validate_plasmon_settling(
            au_n10_model,
            energy_eV=au_profile.omega0_eV,
            decay_times=8.0 if quick else 12.0,
            analysis_cycles=10 if quick else 20,
            points_per_cycle=24 if quick else 40,
        )
        for profile, model, curve in (
            (au_profile, au_n10_model, fig2_n10),
            (ag_profile, ag_n10_model, fig3_n10),
        ):
            fit_detuning = curve.fit_peak_detuning_meV
            assert fit_detuning is not None
            time_results.append(
                validate_periodic_time_ode(
                    model,
                    material_name=profile.material_name,
                    energy_eV=profile.omega0_eV + fit_detuning * 1.0e-3,
                    intensity_w_cm2=profile.intensity_w_cm2,
                    cycles=20 if quick else 40,
                    analysis_cycles=10 if quick else 20,
                    points_per_cycle=24 if quick else 40,
                )
            )
        if settling.max_relative_error >= 1.0e-5:
            raise RuntimeError(
                "The zero-state plasmon settling check failed: max relative "
                f"transfer error={settling.max_relative_error:.6g}."
            )
        for result in time_results:
            if (
                result.alpha_relative_error >= 1.0e-5
                or result.mean_power_relative_error >= 1.0e-5
                or result.accumulator_vs_quadrature_relative_error >= 2.0e-5
                or result.max_excited_population >= 1.0e-7
            ):
                raise RuntimeError(
                    f"The periodic full-RHS validation failed for {result.material_name}."
                )

    _write_curve_csv(run_dir / FIG2B_SPECTRUM_CSV, fig2_n10)
    # Add N=1 columns in a second pass-friendly companion file name.
    _write_curve_csv(run_dir / FIG2B_SPECTRUM_CSV.replace(".csv", "_N1.csv"), fig2_n1)
    with (run_dir / FIG2B_DISTANCE_CSV).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(distance.keys())
        writer.writerows(zip(*distance.values()))
    _write_curve_csv(run_dir / FIG3_OVERVIEW_CSV, fig3_overview)
    _write_curve_csv(run_dir / FIG3_N1_CSV, fig3_n1)
    _write_curve_csv(run_dir / FIG3_N10_CSV, fig3_n10)
    for result, filename in zip(time_results, (TIME_AU_CSV, TIME_AG_CSV)):
        _write_time_csv(run_dir / filename, result)

    figures: list[Figure] = []
    if make_plots:
        figures.extend(
            (
                _plot_fig2b_spectrum(run_dir, fig2_n1, fig2_n10),
                _plot_fig2b_distance(run_dir, distance),
                _plot_fig3(run_dir, fig3_overview, fig3_n1, fig3_n10),
            )
        )
        if settling is not None:
            figures.append(_plot_modal_validation(run_dir, settling, time_results))

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "article": {
            "doi": ARTICLE_DOI,
            "url": ARTICLE_URL,
            "figures": ["2(b)", "3"],
            "pdf_path": str(Path(article_pdf).resolve()),
            "pdf_sha256": _sha256(Path(article_pdf)),
            "raster_anchors": PAPER_RASTER_ANCHORS,
            "fig2b_distance_raster": {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in PAPER_FIG2B_DISTANCE_RASTER.items()
            },
            "raster_anchor_semantics": (
                "visual readout from a 600-dpi rendering, not author raw data"
            ),
        },
        "physical_definitions": {
            "R0": "metal nanosphere radius",
            "Rd": "centre-to-centre QD-MNP distance",
            "omega": "angular frequency of the monochromatic external field",
            "omega0": "bare QD transition angular frequency",
            "N": "largest retained spherical/spheroidal electrostatic degree n",
            "n": "one spatial multipole degree; n=1 is the bright dipole term",
            "T1": "QD population-relaxation time; gamma1=1/T1",
            "T20": "bare QD coherence time; Gamma2=1/T20",
            "A": "MNP dipole response to the spatially uniform incident field",
            "B": "reciprocal bright QD-to-MNP and MNP-to-QD coupling channel",
            "K": "QD reaction-field coefficient, K=sum_n K_n",
            "alpha_effective": "total QD+MNP dipole divided by eps_host*incident field",
            "Q_tot": "cycle-averaged external work rate of the coupled system in strict QS",
        },
        "equations": {
            "coupled_response": "p_QD/E0=beta*(1+B)/(1-beta*K); p_MNP/E0=A+B*p_QD/E0",
            "yan_rwa_beta": "beta=l_QD^2*d^2/(omega0-omega-i*Gamma2)",
            "native_beta": "beta=l_QD^2*2*d^2*omega0/(omega0^2+(Gamma2-i*omega)^2)",
            "work_spectrum": "Q_tot=I0*C_work; C_work=k*Im(alpha_effective)/epsilon0 (strict QS)",
            "harmonic_ode": "(-i*omega*I-M)*x=b*E0",
            "harmonic_mean_power": "<P>=omega*Im[(p/E0)]*|E0|^2/2",
        },
        "profiles": {
            "fig2b": {
                **params_to_physical_dict(au_params, "long"),
                "material": "Au Johnson-Christy",
                "intensity_w_cm2": au_profile.intensity_w_cm2,
                "completion_note": au_profile.completion_note,
            },
            "fig3": {
                **params_to_physical_dict(ag_params, "long"),
                "material": "Ag Johnson-Christy",
                "intensity_w_cm2": ag_profile.intensity_w_cm2,
                "completion_note": ag_profile.completion_note,
            },
        },
        "material_sources": {
            "Au": JOHNSON_CHRISTY_URL,
            "Ag": {"paper": JOHNSON_CHRISTY_URL, "machine_readable_table": AG_DATA_URL},
        },
        "comparisons": {
            "fig2b_N1": _peak_payload(fig2_n1, "fig2b_N1"),
            "fig2b_N10": _peak_payload(fig2_n10, "fig2b_N10"),
            "fig2b_distance": _distance_comparison_payload(distance),
            "fig3_N1": _peak_payload(fig3_n1, "fig3_N1"),
            "fig3_N10": _peak_payload(fig3_n10, "fig3_N10"),
            "fig3_plasmon": {
                "material_peak_energy_eV": float(
                    fig3_overview.energy_eV[fig3_overview.material_peak_index]
                ),
                "material_peak_power_W": fig3_overview.material_peak_power_W,
                "paper_raster": PAPER_RASTER_ANCHORS["fig3_plasmon"],
            },
            "harmonic_state_space_vs_A_B_K": {
                key: _harmonic_payload(value) for key, value in harmonic_checks.items()
            },
            "plasmon_zero_state_settling": None if settling is None else {
                "energy_eV": settling.energy_eV,
                "decay_times_propagated": settling.settling_time_fs / settling.slowest_modal_decay_time_fs,
                "settling_time_fs": settling.settling_time_fs,
                "slowest_modal_decay_time_fs": settling.slowest_modal_decay_time_fs,
                "max_relative_transfer_error": settling.max_relative_error,
                "nfev": settling.nfev,
            },
            "periodic_full_rhs": [_time_payload(result) for result in time_results],
        },
        "fit_diagnostics": {
            "Au": {
                "n_material_modes": au_bright.n_modes,
                "fit_window_eV": au_bright.fit_window_eV,
                "normalized_rms_alpha": au_bright.fit.normalized_rms_alpha,
                "max_relative_alpha_error": au_bright.fit.max_normalized_alpha_error,
                "full_qs_modal_max_normalized_rms": au_n10_model.modal_fit_diagnostics.max_normalized_rms,
                "full_qs_modal_max_relative_error": au_n10_model.modal_fit_diagnostics.max_relative_error,
            },
            "Ag": {
                "n_material_modes": ag_bright.n_modes,
                "fit_window_eV": ag_bright.fit_window_eV,
                "normalized_rms_alpha": ag_bright.fit.normalized_rms_alpha,
                "max_relative_alpha_error": ag_bright.fit.max_normalized_alpha_error,
                "full_qs_modal_max_normalized_rms": ag_n10_model.modal_fit_diagnostics.max_normalized_rms,
                "full_qs_modal_max_relative_error": ag_n10_model.modal_fit_diagnostics.max_relative_error,
                "quality_policy": "reported, not hidden; direct material curves remain authoritative",
            },
        },
        "spatial_truncation": {
            "article_order": 10,
            "article_order_is_claimed_converged": False,
            "Au_N10_spatial_audit": {
                "accepted_at_production_tolerance": au_n10_model.spatial_convergence_diagnostics.accepted,
                "max_half_order_relative_change": au_n10_model.spatial_convergence_diagnostics.max_half_order_relative_change,
                "max_tail_block_relative_mass": au_n10_model.spatial_convergence_diagnostics.max_tail_block_relative_mass,
            },
            "note": "N=10 reproduces Yan's truncation; it is not silently relabelled as N->infinity convergence.",
        },
        "limitations": [
            "The paper does not provide author raw curve data; plot points are raster readouts.",
            "Several CdSe linewidth/dipole parameters are completed from Yan's cited Zhang model and are marked as such.",
            "The Ag passive Lorentz fit is a finite-band approximation; material-vs-fit disagreement is exported separately.",
            "Periodic initialization validates the steady harmonic ODE response, not the slow switch-on transient controlled by T20.",
            "The plasmon-only zero-state test is a diagnostic drive of every spatial port; a uniform laser physically drives n=1 directly.",
            "Strict QS external work is comparable to Q_tot when scattering is neglected, but is not a subsystem-resolved metal-heating observable.",
            "N=10 is the article truncation, not a demonstrated converged infinite-multipole result.",
        ],
        "validation_conclusion": {
            "Fig_2b": (
                "qualified quantitative agreement in peak scale, sign and distance trend; "
                "the direct N=10 inset peak is about -0.183 meV versus the rounded "
                "paper label -0.2 meV"
            ),
            "Fig_3": (
                "qualitative sign reversal only for the traceable completed profile; "
                "the paper omits parameters/material-processing detail needed for its "
                "+0.294 and -3.32 meV quantitative peak positions"
            ),
            "modal_ODE": (
                "state-space resolvent, zero-state plasmon settling and periodic full-RHS "
                "are separate checks; passing them validates the implemented causal ODE "
                "against its A/B/K transfer function, not the missing Fig. 3 inputs"
            ),
        },
        "run_options": {"quick": bool(quick), "time_domain": bool(run_time_domain)},
    }
    write_json(run_dir / METADATA_FILENAME, metadata)

    if show and figures:
        plt.show()
    else:
        for figure in figures:
            plt.close(figure)
    return run_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct Yan 2008 Figs. 2(b)/3 and validate the full-QS modal ODE."
    )
    parser.add_argument("--output-dir", default="results/literature_yan2008_full_qs")
    parser.add_argument("--article-pdf", type=Path, default=DEFAULT_ARTICLE_PDF)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--skip-time-domain", action="store_true")
    parser.add_argument(
        "--quick", action="store_true",
        help="Use reduced grids/cycles for a smoke run; do not use for final paper comparison.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = run_validation(
        output_dir=args.output_dir,
        article_pdf=args.article_pdf,
        make_plots=not args.no_plots,
        show=args.show,
        run_time_domain=not args.skip_time_domain,
        quick=args.quick,
    )
    print(f"Yan 2008 full-QS validation artifacts: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
