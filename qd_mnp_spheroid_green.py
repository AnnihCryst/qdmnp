"""Analytic quasistatic QD--prolate-spheroid interaction kernel.

The historical model in :mod:`qd_mnp_rational_fit` replaces the metal
nanoparticle (MNP) by one point dipole at its centre.  In that approximation
the three response channels needed by the semiclassical QD--MNP equations are

``A``
    MNP dipole induced by a spatially uniform incident field, ``p_M=A E``;
``B``
    reciprocal cross channel, ``p_M=B p_D`` and ``E_sc(QD)=B E``;
``K``
    MNP-mediated reaction field at the QD, ``E_reac=K p_D``.

The point-dipole model imposes ``B=A*J`` and ``K=A*J**2``.  This module keeps
the same three-port description but evaluates it from the separated solution
of Laplace's equation in prolate spheroidal coordinates.  A point QD on the
positive symmetry axis excites all spatial orders.  The uniform field and the
total MNP dipole occupy only the bright order n=1, whereas ``K`` contains the
full converged series.

Conventions
-----------
* The host and particle use relative permittivities ``eps_m`` and ``eps_p``.
* Length, field and dipole values follow the atomic-unit convention of the
  existing project.  Consequently A has units a0**3, B is dimensionless and K
  has units a0**-3.  The usual SI factor 4*pi*epsilon_0 is absent.
* Time-harmonic quantities use exp(-i*omega*t), as in the existing linear
  spectrum code.  For a passive local material Im(eps_p) >= 0.
* ``long`` means a QD dipole parallel to the spheroid axis; ``trans`` means a
  dipole perpendicular to that axis while the QD centre remains on the axis.

This is a strictly quasistatic, local-response and point-QD kernel.  It does
not add retardation, radiation reaction, nonlocal metal response or a
geometry-derived spontaneous-emission rate to the Bloch equations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.special import gammaln, hyp2f1, legendre_p_all, lqn

from qd_mnp_rational_fit import (
    DipoleOrientation,
    FieldPolarization,
    HybridQDPlasmonModel,
    HybridSystemParams,
    MaterialDispersion,
    QDPosition,
    eV_to_au,
    geometric_coupling_factor,
    orientation_from_field_polarization,
    resolve_field_polarization,
    validate_qd_position,
)


InteractionModel = Literal["legacy", "spheroid_n1", "spheroid_full"]
MAX_SUPPORTED_SPATIAL_DEGREE = 512


def _readonly_array(value: np.ndarray, *, dtype=None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ProlateSpheroidGeometry:
    """Geometry of one external point QD beside a prolate/spherical particle.

    ``SpheroidGreenInteraction`` treats ``c_au == a_au`` with the exact
    spherical multipole limit.  Values with ``c_au > a_au`` use prolate
    spheroidal coordinates; oblate particles remain outside this API.

    ``qd_position`` selects the QD centre and ``field_polarization`` selects the
    incident polarization; the two are independent.  ``R_au`` is always the
    centre-to-centre distance measured along the QD direction, so a tip QD sits
    at ``(0, 0, R_au)`` and an equatorial QD at ``(R_au, 0, 0)``.
    ``orientation`` is the legacy alias of ``field_polarization``.
    """

    a_au: float
    c_au: float
    R_au: float
    eps_m: float
    orientation: DipoleOrientation | None = None
    qd_radius_au: float = 0.0
    qd_position: QDPosition = "tip"
    field_polarization: FieldPolarization | None = None

    def __post_init__(self) -> None:
        polarization = resolve_field_polarization(
            self.orientation,
            self.field_polarization,
        )
        object.__setattr__(self, "field_polarization", polarization)
        object.__setattr__(
            self,
            "orientation",
            orientation_from_field_polarization(polarization),
        )
        validate_qd_position(self.qd_position)
        values = np.asarray(
            [
                self.a_au,
                self.c_au,
                self.R_au,
                self.eps_m,
                self.qd_radius_au,
            ],
            dtype=float,
        )
        if np.any(~np.isfinite(values)):
            raise ValueError("Spheroid geometry values must be finite.")
        if self.a_au <= 0.0 or self.c_au <= 0.0 or self.R_au <= 0.0:
            raise ValueError("Spheroid semiaxes and QD centre distance must be positive.")
        if self.c_au < self.a_au:
            raise ValueError(
                "The analytic interaction requires a prolate or spherical "
                "particle, c_au >= a_au."
            )
        if self.eps_m <= 0.0:
            raise ValueError("The lossless host permittivity eps_m must be positive.")
        if self.qd_radius_au < 0.0:
            raise ValueError("qd_radius_au must be non-negative.")
        if self.R_au <= self.directional_semiaxis_au + self.qd_radius_au:
            semiaxis = "c_au" if self.qd_position == "tip" else "a_au"
            raise ValueError(
                "The QD must lie strictly outside the spheroid: require "
                f"R_au > {semiaxis} + qd_radius_au for "
                f"qd_position={self.qd_position!r}."
            )

    @classmethod
    def from_params(
        cls,
        params: HybridSystemParams,
        *,
        orientation: DipoleOrientation | None = None,
        qd_position: QDPosition | None = None,
        field_polarization: FieldPolarization | None = None,
    ) -> "ProlateSpheroidGeometry":
        """Mirror the geometry already carried by ``params``.

        The optional arguments only let a caller spell the same choice out; a
        value that contradicts ``params`` is rejected.
        """
        polarization = resolve_field_polarization(
            orientation,
            field_polarization,
            default=params.field_polarization,
        )
        if polarization != params.field_polarization:
            raise ValueError(
                f"field_polarization={polarization!r} contradicts the "
                f"parameters' {params.field_polarization!r}."
            )
        position = params.qd_position if qd_position is None else qd_position
        if position != params.qd_position:
            raise ValueError(
                f"qd_position={position!r} contradicts the parameters' "
                f"{params.qd_position!r}."
            )
        return cls(
            a_au=float(params.a_au),
            c_au=float(params.c_au),
            R_au=float(params.R_au),
            eps_m=float(params.eps_m),
            qd_radius_au=float(params.qd_radius_au),
            qd_position=position,
            field_polarization=polarization,
        )

    @property
    def directional_semiaxis_au(self) -> float:
        """MNP surface radius along the QD direction: c at the tip, a at the equator."""
        return float(self.c_au if self.qd_position == "tip" else self.a_au)

    @property
    def qd_position_vector_au(self) -> np.ndarray:
        """QD centre r_D in the frame whose z axis is the long MNP axis."""
        if self.qd_position == "tip":
            return np.asarray([0.0, 0.0, self.R_au], dtype=float)
        return np.asarray([self.R_au, 0.0, 0.0], dtype=float)

    @property
    def geometric_coupling_factor(self) -> float:
        """Point-dipole tensor factor G of this position/polarization pair."""
        return geometric_coupling_factor(self.qd_position, self.field_polarization)

    @property
    def focal_length_au(self) -> float:
        # Product form avoids cancellation for a weakly prolate particle.
        return float(np.sqrt((self.c_au - self.a_au) * (self.c_au + self.a_au)))

    @property
    def xi_surface(self) -> float:
        if self.c_au == self.a_au:
            return float("inf")
        return float(self.c_au / self.focal_length_au)

    @property
    def xi_qd(self) -> float:
        """Radial spheroidal coordinate of the QD centre.

        A tip QD lies on the axis, where ``z = f*xi*eta`` with ``eta=1`` gives
        ``xi = R/f``.  An equatorial QD lies in ``z = 0``, where
        ``x = f*sqrt(xi**2-1)`` gives ``xi = sqrt(1+(R/f)**2)``.
        """
        if self.c_au == self.a_au:
            return float("inf")
        focal = self.focal_length_au
        if self.qd_position == "tip":
            return float(self.R_au / focal)
        return float(np.hypot(1.0, self.R_au / focal))

    @property
    def eta_qd(self) -> float:
        """Angular QD coordinate: 1 on the long axis, 0 in the equatorial plane."""
        return 1.0 if self.qd_position == "tip" else 0.0

    @property
    def surface_gap_au(self) -> float:
        return float(self.R_au - self.directional_semiaxis_au - self.qd_radius_au)


@dataclass(frozen=True)
class QuasistaticInteractionResponse:
    """Frequency-domain A/B/K response in the project's atomic-unit convention.

    Modes are labelled by a spatial degree ``n`` and an azimuthal order ``m``.
    A QD on the symmetry axis excites one order only, so ``degrees`` is then the
    contiguous sequence 1..n_max.  An equatorial QD excites every ``m`` with the
    parity its dipole selects, and ``degrees`` repeats each ``n`` accordingly.
    Modes are always ordered by ``(n, m)``, so index 0 is the bright mode.
    """

    model: InteractionModel
    orientation: DipoleOrientation
    eps_m: float
    A_au3: np.ndarray
    B: np.ndarray
    K_au_minus3: np.ndarray
    degrees: np.ndarray
    K_by_degree_au_minus3: np.ndarray
    modal_susceptibility_by_degree: np.ndarray
    reaction_weight_by_degree_au_minus3: np.ndarray
    depolarization_by_degree: np.ndarray
    geometric_factor_by_degree: np.ndarray
    log_abs_geometric_factor_by_degree: np.ndarray | None = None
    azimuthal_orders: np.ndarray | None = None
    qd_position: QDPosition = "tip"

    def __post_init__(self) -> None:
        A = _readonly_array(self.A_au3, dtype=complex)
        B = _readonly_array(self.B, dtype=complex)
        K = _readonly_array(self.K_au_minus3, dtype=complex)
        degrees = _readonly_array(self.degrees, dtype=int)
        K_by_degree = _readonly_array(self.K_by_degree_au_minus3, dtype=complex)
        modal_susceptibility = _readonly_array(
            self.modal_susceptibility_by_degree,
            dtype=complex,
        )
        reaction_weight = _readonly_array(
            self.reaction_weight_by_degree_au_minus3,
            dtype=float,
        )
        depolarization = _readonly_array(self.depolarization_by_degree, dtype=float)
        geometric = _readonly_array(self.geometric_factor_by_degree, dtype=float)
        if self.log_abs_geometric_factor_by_degree is None:
            with np.errstate(divide="ignore", invalid="ignore"):
                log_abs_geometric = np.log(np.abs(geometric))
        else:
            log_abs_geometric = _readonly_array(
                self.log_abs_geometric_factor_by_degree,
                dtype=float,
            )

        if not (A.shape == B.shape == K.shape):
            raise ValueError("A, B and K must have identical frequency shapes.")
        if A.ndim > 1 or A.size == 0:
            raise ValueError(
                "Frequency responses must be scalar or non-empty one-dimensional arrays."
            )
        if degrees.ndim != 1 or degrees.size < 1:
            raise ValueError("degrees must be a non-empty one-dimensional array.")
        validate_qd_position(self.qd_position)
        if self.azimuthal_orders is None:
            # A QD on the symmetry axis couples to a single azimuthal order:
            # m=0 for an axial dipole and m=1 for a transverse one.
            single_order = 0 if self.orientation == "long" else 1
            orders = np.full(degrees.size, single_order, dtype=int)
        else:
            orders = _readonly_array(self.azimuthal_orders, dtype=int)
        expected_mode_shape = (degrees.size,) + A.shape
        if K_by_degree.shape != expected_mode_shape:
            raise ValueError("K_by_degree has an inconsistent mode/frequency shape.")
        if modal_susceptibility.shape != expected_mode_shape:
            raise ValueError("modal_susceptibility_by_degree has an inconsistent shape.")
        if not (
            depolarization.shape
            == geometric.shape
            == reaction_weight.shape
            == log_abs_geometric.shape
            == degrees.shape
        ):
            raise ValueError("Mode geometry arrays must have one value per degree.")
        if orders.shape != degrees.shape:
            raise ValueError("azimuthal_orders must have one value per mode.")
        if degrees[0] != 1 or np.any(np.diff(degrees) < 0) or np.any(np.diff(degrees) > 1):
            raise ValueError(
                "Spatial degrees must start at 1 and increase by at most one, "
                "so that every degree up to n_max is represented."
            )
        if np.any(orders < 0) or np.any(orders > degrees):
            raise ValueError("Azimuthal orders must satisfy 0 <= m <= n.")
        mode_labels = np.stack([degrees, orders], axis=1)
        if np.any(np.all(mode_labels[1:] <= mode_labels[:-1], axis=1)):
            raise ValueError("Modes must be strictly ordered by (degree, order).")
        finite_arrays = (
            A,
            B,
            K,
            K_by_degree,
            modal_susceptibility,
            reaction_weight,
            depolarization,
        )
        if any(np.any(~np.isfinite(values)) for values in finite_arrays):
            raise FloatingPointError(
                "Non-finite spheroidal response; a lossless modal pole or numerical "
                "overflow may have been sampled."
            )
        if np.any(np.isnan(geometric)) or np.any(np.isnan(log_abs_geometric)):
            raise FloatingPointError("Invalid geometric-factor diagnostics.")
        if not np.isfinite(self.eps_m) or self.eps_m <= 0.0:
            raise ValueError("eps_m must be finite and positive.")

        object.__setattr__(self, "A_au3", A)
        object.__setattr__(self, "B", B)
        object.__setattr__(self, "K_au_minus3", K)
        object.__setattr__(self, "degrees", degrees)
        object.__setattr__(self, "K_by_degree_au_minus3", K_by_degree)
        object.__setattr__(
            self,
            "modal_susceptibility_by_degree",
            modal_susceptibility,
        )
        object.__setattr__(
            self,
            "reaction_weight_by_degree_au_minus3",
            reaction_weight,
        )
        object.__setattr__(self, "depolarization_by_degree", depolarization)
        object.__setattr__(self, "geometric_factor_by_degree", geometric)
        orders.setflags(write=False)
        object.__setattr__(self, "azimuthal_orders", orders)
        object.__setattr__(
            self,
            "log_abs_geometric_factor_by_degree",
            log_abs_geometric,
        )
        object.__setattr__(self, "eps_m", float(self.eps_m))

    @property
    def n_max(self) -> int:
        return int(self.degrees[-1])

    @property
    def mode_count(self) -> int:
        return int(self.degrees.size)

    @property
    def K_bright_au_minus3(self) -> np.ndarray:
        return self.K_by_degree_au_minus3[0]

    @property
    def K_higher_au_minus3(self) -> np.ndarray:
        return self.K_au_minus3 - self.K_bright_au_minus3

    @property
    def cumulative_K_au_minus3(self) -> np.ndarray:
        return np.cumsum(self.K_by_degree_au_minus3, axis=0)

    def relative_half_order_change(self, *, floor_scale: float = 1.0e-14) -> np.ndarray:
        """Return |K_N-K_floor(N/2)| with a global-scale relative floor."""

        if not np.isfinite(floor_scale) or floor_scale <= 0.0:
            raise ValueError("floor_scale must be finite and positive.")
        cumulative = self.cumulative_K_au_minus3
        fine = cumulative[-1]
        if self.n_max == 1:
            # K_0 is the empty modal sum.  Returning zero here would compare
            # K_1 with itself and falsely label a one-mode calculation as
            # converged.
            coarse = np.zeros_like(fine)
        else:
            # Compare against every mode up to half the retained spatial
            # degree, which is the mode count itself for a single-order family.
            coarse_index = int(
                np.searchsorted(self.degrees, self.n_max // 2, side="right")
            ) - 1
            coarse = cumulative[coarse_index]
        global_scale = max(float(np.max(np.abs(fine))), np.finfo(float).tiny)
        return np.abs(fine - coarse) / np.maximum(
            np.abs(fine), floor_scale * global_scale
        )

    def relative_tail_block(
        self,
        *,
        block_size: int | None = None,
        floor_scale: float = 1.0e-14,
    ) -> np.ndarray:
        """Conservative absolute modal mass in the final spatial-order block."""

        if block_size is None:
            block_size = min(self.mode_count, max(4, self.mode_count // 8))
        if not isinstance(block_size, (int, np.integer)) or not (
            1 <= block_size <= self.mode_count
        ):
            raise ValueError("block_size must be an integer in [1, mode_count].")
        if not np.isfinite(floor_scale) or floor_scale <= 0.0:
            raise ValueError("floor_scale must be finite and positive.")
        absolute_modal_sum = np.sum(np.abs(self.K_by_degree_au_minus3), axis=0)
        tail_mass = np.sum(
            np.abs(self.K_by_degree_au_minus3[-block_size:]),
            axis=0,
        )
        denominator = np.maximum(
            np.abs(self.K_au_minus3),
            floor_scale * np.maximum(absolute_modal_sum, np.finfo(float).tiny),
        )
        return tail_mass / denominator

    def truncate(self, n_max: int) -> "QuasistaticInteractionResponse":
        """Keep the exact A/B channels and truncate only the reaction series."""

        if (
            isinstance(n_max, (bool, np.bool_))
            or not isinstance(n_max, (int, np.integer))
            or n_max < 1
            or n_max > self.n_max
        ):
            raise ValueError(f"n_max must lie in [1, {self.n_max}].")
        n_max = int(n_max)
        # Truncation is by spatial degree, which keeps every azimuthal order of
        # the retained degrees together.
        mode_slice = self.degrees <= n_max
        if self.model == "legacy":
            model: InteractionModel = "legacy"
        else:
            model = "spheroid_n1" if n_max == 1 else "spheroid_full"
        modes = self.K_by_degree_au_minus3[mode_slice]
        return QuasistaticInteractionResponse(
            model=model,
            orientation=self.orientation,
            A_au3=self.A_au3,
            B=self.B,
            K_au_minus3=np.sum(modes, axis=0),
            degrees=self.degrees[mode_slice],
            K_by_degree_au_minus3=modes,
            modal_susceptibility_by_degree=(
                self.modal_susceptibility_by_degree[mode_slice]
            ),
            reaction_weight_by_degree_au_minus3=(
                self.reaction_weight_by_degree_au_minus3[mode_slice]
            ),
            depolarization_by_degree=self.depolarization_by_degree[mode_slice],
            geometric_factor_by_degree=self.geometric_factor_by_degree[mode_slice],
            eps_m=self.eps_m,
            log_abs_geometric_factor_by_degree=(
                self.log_abs_geometric_factor_by_degree[mode_slice]
            ),
            azimuthal_orders=self.azimuthal_orders[mode_slice],
            qd_position=self.qd_position,
        )


@dataclass(frozen=True)
class LinearHybridResponse:
    """Weak-field solution obtained from any reciprocal A/B/K interaction."""

    alpha_effective_au3: np.ndarray
    mnp_dipole_over_field_au3: np.ndarray
    qd_dipole_over_field_au3: np.ndarray
    denominator: np.ndarray

    def __post_init__(self) -> None:
        arrays = [
            _readonly_array(self.alpha_effective_au3, dtype=complex),
            _readonly_array(self.mnp_dipole_over_field_au3, dtype=complex),
            _readonly_array(self.qd_dipole_over_field_au3, dtype=complex),
            _readonly_array(self.denominator, dtype=complex),
        ]
        if len({values.shape for values in arrays}) != 1:
            raise ValueError("All linear-response arrays must have the same shape.")
        if any(np.any(~np.isfinite(values)) for values in arrays):
            raise FloatingPointError("The coupled linear response contains non-finite values.")
        (
            alpha_effective,
            mnp_dipole,
            qd_dipole,
            denominator,
        ) = arrays
        object.__setattr__(self, "alpha_effective_au3", alpha_effective)
        object.__setattr__(self, "mnp_dipole_over_field_au3", mnp_dipole)
        object.__setattr__(self, "qd_dipole_over_field_au3", qd_dipole)
        object.__setattr__(self, "denominator", denominator)


def _log_legendre_p_and_derivative(
    n_max: int,
    x: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return log(P_n(x)) and log(P_n'(x)) for n=1..n_max.

    Direct SciPy values are retained whenever representable.  If the dominant
    P solution overflows for a nearly spherical coordinate system (large x),
    a positive forward-ratio recurrence supplies the logarithms without ever
    constructing P_n itself.
    """

    with np.errstate(over="ignore", invalid="ignore"):
        p_all = np.asarray(legendre_p_all(n_max, x, diff_n=1), dtype=float)
    P = p_all[0]
    P_prime = p_all[1]
    valid_P = np.isfinite(P) & (P > 0.0)
    valid_P_prime = np.isfinite(P_prime) & (P_prime > 0.0)
    if np.all(valid_P) and np.all(valid_P_prime[1:]):
        return np.log(P[1:]), np.log(P_prime[1:])

    log_P = np.empty(n_max + 1, dtype=float)
    log_P_prime = np.full(n_max + 1, np.nan, dtype=float)
    log_P[0] = 0.0
    ratio = float(x)
    log_x_minus_one = float(np.log(x - 1.0))
    log_x_plus_one = float(np.log(x + 1.0))
    for degree in range(1, n_max + 1):
        if degree > 1:
            ratio = (
                (2.0 * degree - 1.0) * x - (degree - 1.0) / ratio
            ) / degree
        if not np.isfinite(ratio) or ratio <= 0.0:
            raise FloatingPointError("Positive Legendre-P ratio recurrence failed.")
        log_P[degree] = log_P[degree - 1] + np.log(ratio)
        derivative_numerator = x - 1.0 / ratio
        if derivative_numerator <= 0.0 or not np.isfinite(derivative_numerator):
            raise FloatingPointError("Legendre-P logarithmic derivative failed.")
        log_derivative_ratio = (
            np.log(float(degree))
            + np.log(derivative_numerator)
            - log_x_minus_one
            - log_x_plus_one
        )
        log_P_prime[degree] = log_P[degree] + log_derivative_ratio

    # Preserve the generally more accurate direct values at all orders where
    # they did not overflow.
    direct_log_P = np.full(n_max + 1, np.nan, dtype=float)
    direct_log_P_prime = np.full(n_max + 1, np.nan, dtype=float)
    direct_log_P[valid_P] = np.log(P[valid_P])
    direct_log_P_prime[valid_P_prime] = np.log(P_prime[valid_P_prime])
    log_P[valid_P] = direct_log_P[valid_P]
    log_P_prime[valid_P_prime] = direct_log_P_prime[valid_P_prime]
    return log_P[1:], log_P_prime[1:]


def _hypergeometric_log_Q(degrees: np.ndarray, x: float) -> np.ndarray:
    """Logarithm of Q_n(x), using its positive x>1 hypergeometric form."""

    degrees_float = np.asarray(degrees, dtype=float)
    inverse_x = 1.0 / x
    z = inverse_x * inverse_x
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        hypergeometric = hyp2f1(
            0.5 * (degrees_float + 1.0),
            0.5 * (degrees_float + 2.0),
            degrees_float + 1.5,
            z,
        )
        result = (
            0.5 * np.log(np.pi)
            + gammaln(degrees_float + 1.0)
            - gammaln(degrees_float + 1.5)
            - (degrees_float + 1.0) * np.log(2.0)
            - (degrees_float + 1.0) * np.log(x)
            + np.log(hypergeometric)
        )
    if np.any(~np.isfinite(result)):
        raise FloatingPointError(
            "Legendre-Q underflow could not be recovered by the scaled "
            "hypergeometric representation."
        )
    return np.asarray(result, dtype=float)


def _log_legendre_q_and_derivative(
    n_max: int,
    x: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return log(Q_n(x)) and log(-Q_n'(x)) for n=1..n_max."""

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        Q, Q_prime = (np.asarray(values, dtype=float) for values in lqn(n_max, x))
    valid_Q = np.isfinite(Q) & (Q > 0.0)
    valid_Q_prime = np.isfinite(Q_prime) & (Q_prime < 0.0)
    log_Q = np.full(n_max + 1, np.nan, dtype=float)
    log_minus_Q_prime = np.full(n_max + 1, np.nan, dtype=float)
    log_Q[valid_Q] = np.log(Q[valid_Q])
    log_minus_Q_prime[valid_Q_prime] = np.log(-Q_prime[valid_Q_prime])

    missing_Q = ~np.isfinite(log_Q)
    if np.any(missing_Q):
        missing_degrees = np.flatnonzero(missing_Q)
        log_Q[missing_Q] = _hypergeometric_log_Q(missing_degrees, x)

    log_x_minus_one = float(np.log(x - 1.0))
    log_x_plus_one = float(np.log(x + 1.0))
    for degree in range(1, n_max + 1):
        if np.isfinite(log_minus_Q_prime[degree]):
            continue
        previous_over_current = np.exp(log_Q[degree - 1] - log_Q[degree])
        derivative_factor = previous_over_current - x
        if derivative_factor <= 0.0 or not np.isfinite(derivative_factor):
            raise FloatingPointError("Legendre-Q logarithmic derivative failed.")
        log_minus_Q_prime[degree] = (
            log_Q[degree]
            + np.log(float(degree))
            + np.log(derivative_factor)
            - log_x_minus_one
            - log_x_plus_one
        )

    selected_Q = log_Q[1:]
    selected_Q_prime = log_minus_Q_prime[1:]
    if np.any(~np.isfinite(selected_Q)) or np.any(~np.isfinite(selected_Q_prime)):
        raise FloatingPointError("Scaled Legendre-Q evaluation failed.")
    return selected_Q, selected_Q_prime


@dataclass(frozen=True)
class _AssociatedLegendreLogTable:
    """Log magnitudes of P_n^m, Q_n^m and their xi-derivatives for xi>1.

    Hobson's convention is used throughout, ``F_n^m(x)=(x**2-1)**(m/2)*d^m
    F_n/dx^m``, without the Condon--Shortley phase.  For ``x>1`` this makes
    ``P_n^m`` and ``P_n^m'`` positive, while ``Q_n^m`` has sign ``(-1)**m`` and
    ``Q_n^m'`` has sign ``(-1)**(m+1)``.  Only magnitudes are stored; the
    kernel reintroduces the signs analytically, where they cancel against the
    ``(-1)**m`` of the expansion coefficient.

    Entries with ``m > n`` are NaN.  Row ``i`` holds degree ``n = i+1``.
    """

    x: float
    n_max: int
    log_P: np.ndarray
    log_P_prime: np.ndarray
    log_abs_Q: np.ndarray
    log_abs_Q_prime: np.ndarray


def _associated_legendre_log_table(
    n_max: int,
    x: float,
) -> _AssociatedLegendreLogTable:
    """Build the (n, m) log table by cancellation-free positive recurrences.

    The ``m=0`` column reuses the hardened axial routines above.  Higher orders
    follow

    * ``P``: the closed form ``P_m^m=(2m-1)!!*(x**2-1)**(m/2)`` plus the usual
      degree recurrence written for the ratio ``P_n^m/P_{n-1}^m``;
    * ``Q``: the order recurrence
      ``q_n^{m+2}=2(m+1)x/s*q_n^{m+1}+(n-m)(n+m+1)*q_n^m`` for the positive
      magnitudes ``q_n^m=(-1)**m*Q_n^m``, whose terms are all positive, seeded
      by ``q_n^0=Q_n`` and ``q_n^1=s*|Q_n'|`` with ``s=sqrt(x**2-1)``.

    Both derivatives use ``(x**2-1)*F_n^m'=m*x*F_n^m+s*F_n^{m+1}``, which is
    free of cancellation for ``P`` term by term and, for ``Q``, reduces via the
    order recurrence to ``m*x+s*(n-m+1)(n+m)/rho_m`` with ``rho_m`` the
    positive ratio ``q_n^m/q_n^{m-1}``.
    """

    if (
        isinstance(n_max, (bool, np.bool_))
        or not isinstance(n_max, (int, np.integer))
        or n_max < 1
    ):
        raise ValueError("n_max must be an integer >= 1.")
    n_max = int(n_max)
    x = float(x)
    if not np.isfinite(x) or x <= 1.0:
        raise ValueError("The prolate radial coordinate must satisfy x > 1.")

    log_x_minus_one = float(np.log(x - 1.0))
    log_x_plus_one = float(np.log(x + 1.0))
    log_x2_minus_one = log_x_minus_one + log_x_plus_one
    half_log_x2_minus_one = 0.5 * log_x2_minus_one
    sqrt_x2_minus_one = float(np.sqrt((x - 1.0) * (x + 1.0)))

    shape = (n_max, n_max + 1)
    log_P = np.full(shape, np.nan, dtype=float)
    log_P_prime = np.full(shape, np.nan, dtype=float)
    log_abs_Q = np.full(shape, np.nan, dtype=float)
    log_abs_Q_prime = np.full(shape, np.nan, dtype=float)

    axial_log_P, axial_log_P_prime = _log_legendre_p_and_derivative(n_max, x)
    axial_log_Q, axial_log_minus_Q_prime = _log_legendre_q_and_derivative(n_max, x)
    log_P[:, 0] = axial_log_P
    log_P_prime[:, 0] = axial_log_P_prime
    log_abs_Q[:, 0] = axial_log_Q
    log_abs_Q_prime[:, 0] = axial_log_minus_Q_prime
    if n_max >= 1:
        log_abs_Q[:, 1] = half_log_x2_minus_one + axial_log_minus_Q_prime

    orders = np.arange(1, n_max + 1, dtype=float)
    log_double_factorial = (
        gammaln(2.0 * orders + 1.0)
        - orders * np.log(2.0)
        - gammaln(orders + 1.0)
    )
    for m in range(1, n_max + 1):
        log_P[m - 1, m] = (
            log_double_factorial[m - 1] + m * half_log_x2_minus_one
        )
        ratio = 0.0
        for n in range(m + 1, n_max + 1):
            if n == m + 1:
                ratio = (2.0 * m + 1.0) * x
            else:
                ratio = (
                    (2.0 * n - 1.0) * x - (n + m - 1.0) / ratio
                ) / (n - m)
            if not np.isfinite(ratio) or ratio <= 0.0:
                raise FloatingPointError(
                    "Positive associated-Legendre-P ratio recurrence failed at "
                    f"n={n}, m={m}, x={x!r}."
                )
            log_P[n - 1, m] = log_P[n - 2, m] + np.log(ratio)

    for m in range(1, n_max + 1):
        degrees = np.arange(m, n_max + 1, dtype=float)
        if m < n_max:
            with np.errstate(over="ignore"):
                order_ratio = np.exp(
                    half_log_x2_minus_one
                    + log_P[m:, m + 1]
                    - log_P[m:, m]
                )
            bracket = np.empty(degrees.size, dtype=float)
            # The diagonal has P_n^{n+1}=0, so only m*x survives there.
            bracket[0] = m * x
            bracket[1:] = m * x + order_ratio
        else:
            bracket = np.asarray([m * x], dtype=float)
        if np.any(~np.isfinite(bracket)) or np.any(bracket <= 0.0):
            raise FloatingPointError(
                f"Associated-Legendre-P derivative bracket failed at m={m}."
            )
        log_P_prime[m - 1 :, m] = (
            log_P[m - 1 :, m] + np.log(bracket) - log_x2_minus_one
        )

    for n in range(1, n_max + 1):
        rho = np.empty(n + 2, dtype=float)
        rho[0] = np.nan
        rho[1] = float(np.exp(log_abs_Q[n - 1, 1] - log_abs_Q[n - 1, 0]))
        for k in range(1, n):
            rho[k + 1] = (
                2.0 * k * x / sqrt_x2_minus_one
                + (n - k + 1.0) * (n + k) / rho[k]
            )
            if not np.isfinite(rho[k + 1]) or rho[k + 1] <= 0.0:
                raise FloatingPointError(
                    "Positive associated-Legendre-Q order recurrence failed at "
                    f"n={n}, m={k + 1}, x={x!r}."
                )
            log_abs_Q[n - 1, k + 1] = log_abs_Q[n - 1, k] + np.log(rho[k + 1])
        for m in range(1, n + 1):
            bracket = (
                m * x
                + sqrt_x2_minus_one * (n - m + 1.0) * (n + m) / rho[m]
            )
            if not np.isfinite(bracket) or bracket <= 0.0:
                raise FloatingPointError(
                    "Associated-Legendre-Q derivative bracket failed at "
                    f"n={n}, m={m}."
                )
            log_abs_Q_prime[n - 1, m] = (
                log_abs_Q[n - 1, m] + np.log(bracket) - log_x2_minus_one
            )

    upper = np.triu(np.ones(shape, dtype=bool), k=2)
    tables = (log_P, log_P_prime, log_abs_Q, log_abs_Q_prime)
    for table in tables:
        if np.any(~np.isfinite(table[~upper])):
            raise FloatingPointError(
                "The associated-Legendre log table contains non-finite entries."
            )
        table.setflags(write=False)
    return _AssociatedLegendreLogTable(
        x=x,
        n_max=n_max,
        log_P=log_P,
        log_P_prime=log_P_prime,
        log_abs_Q=log_abs_Q,
        log_abs_Q_prime=log_abs_Q_prime,
    )


def _log_abs_ferrers_at_zero(
    degrees: np.ndarray,
    orders: np.ndarray,
) -> np.ndarray:
    """log|P_n^m(0)| for the equatorial plane; requires n-m even.

    ``P_n^m(0)=2**m*Gamma((n+m+1)/2)/(sqrt(pi)*Gamma((n-m)/2+1))`` up to the
    sign that squares away in every reaction weight.
    """

    n = np.asarray(degrees, dtype=float)
    m = np.asarray(orders, dtype=float)
    if np.any((n - m) % 2 != 0):
        raise ValueError("P_n^m(0) vanishes unless n-m is even.")
    return (
        m * np.log(2.0)
        + gammaln(0.5 * (n + m + 1.0))
        - 0.5 * np.log(np.pi)
        - gammaln(0.5 * (n - m) + 1.0)
    )


def _log_abs_ferrers_derivative_at_zero(
    degrees: np.ndarray,
    orders: np.ndarray,
) -> np.ndarray:
    """log|dP_n^m/d(eta)| at eta=0; requires n-m odd.

    Differentiating ``P_n^m(eta)=(1-eta**2)**(m/2)*d^m P_n/d(eta)^m`` at zero
    leaves ``P_n^{m+1}(0)``, so the closed form above applies with m -> m+1.
    """

    n = np.asarray(degrees, dtype=float)
    m = np.asarray(orders, dtype=float)
    if np.any((n - m) % 2 != 1):
        raise ValueError("dP_n^m/d(eta) vanishes at eta=0 unless n-m is odd.")
    return (
        (m + 1.0) * np.log(2.0)
        + gammaln(0.5 * (n + m) + 1.0)
        - 0.5 * np.log(np.pi)
        - gammaln(0.5 * (n - m + 1.0))
    )


def _exp_representable(log_values: np.ndarray, *, quantity: str) -> np.ndarray:
    """Exponentiate a physical nonnegative quantity, allowing true underflow."""

    values = np.asarray(log_values, dtype=float)
    if np.any(np.isnan(values)) or np.any(values > np.log(np.finfo(float).max)):
        raise FloatingPointError(f"{quantity} exceeds double-precision range.")
    with np.errstate(under="ignore"):
        return np.asarray(np.exp(values), dtype=float)


class SpheroidGreenInteraction:
    """Analytic modal projection of the spheroid's quasistatic Green tensor."""

    name = "spheroid_full"

    def __init__(
        self,
        geometry: ProlateSpheroidGeometry,
        *,
        n_max: int = 80,
    ) -> None:
        if (
            isinstance(n_max, (bool, np.bool_))
            or not isinstance(n_max, (int, np.integer))
            or n_max < 1
        ):
            raise ValueError("n_max must be an integer >= 1.")
        if n_max > MAX_SUPPORTED_SPATIAL_DEGREE:
            raise ValueError(
                "n_max exceeds MAX_SUPPORTED_SPATIAL_DEGREE="
                f"{MAX_SUPPORTED_SPATIAL_DEGREE}; higher orders are outside "
                "the guarded double-precision implementation."
            )
        self.geometry = geometry
        self.n_max = int(n_max)
        self.is_spherical = bool(self.geometry.c_au == self.geometry.a_au)
        # ``degrees``/``azimuthal_orders`` carry one entry per retained mode.
        # An axial QD, and every spherical particle, needs a single order per
        # degree; an equatorial QD on a spheroid needs the full (n, m) family.
        self.degrees = np.arange(1, self.n_max + 1, dtype=int)
        self.azimuthal_orders = np.full(
            self.n_max,
            0 if self.geometry.field_polarization == "longitudinal" else 1,
            dtype=int,
        )
        if self.is_spherical:
            self._initialize_spherical_limit()
        elif self.geometry.qd_position == "equatorial":
            self._initialize_equatorial_prolate()
        else:
            self._initialize_prolate_scaled()
        self.degrees.setflags(write=False)
        self.azimuthal_orders.setflags(write=False)

        if (
            np.any(~np.isfinite(self.depolarization_by_degree))
            or np.any(~np.isfinite(self.reaction_weight_by_degree_au_minus3))
            or np.any(self.depolarization_by_degree <= 0.0)
            or np.any(self.depolarization_by_degree >= 1.0)
            or np.any(self.reaction_weight_by_degree_au_minus3 < 0.0)
            or self.reaction_weight_by_degree_au_minus3[0] <= 0.0
            or not np.isfinite(self.bright_source_coupling_au_minus3)
        ):
            raise FloatingPointError("Invalid spheroidal modal geometry coefficients.")

    def _initialize_spherical_limit(self) -> None:
        """Exact dielectric-sphere multipoles, avoiding the singular f -> 0 basis.

        A sphere has no distinguished axis, so only the angle between the QD
        dipole and the QD radius vector matters: the tip/equatorial choice
        enters solely through ``G``.  Every azimuthal order of one degree also
        shares ``L_n = n/(2n+1)``, so the azimuthal sum is carried exactly by a
        single mode per degree.
        """

        degree = self.degrees.astype(float)
        a = self.geometry.a_au
        R = self.geometry.R_au
        eps_m = self.geometry.eps_m
        self.depolarization_by_degree = degree / (2.0 * degree + 1.0)
        common_log_weight = (
            (2.0 * degree + 1.0) * np.log(a)
            - (2.0 * degree + 4.0) * np.log(R)
            - np.log(eps_m)
            - np.log(2.0 * degree + 1.0)
        )
        coupling_factor = self.geometry.geometric_coupling_factor
        if coupling_factor > 0.0:
            # QD dipole along the radius vector.
            log_weight = (
                common_log_weight
                + np.log(degree)
                + 2.0 * np.log(degree + 1.0)
            )
        else:
            # QD dipole tangent to the sphere.
            log_weight = (
                common_log_weight
                + 2.0 * np.log(degree)
                + np.log(degree + 1.0)
                - np.log(2.0)
            )
        self.reaction_weight_by_degree_au_minus3 = _exp_representable(
            log_weight,
            quantity="Spherical reaction weight",
        )
        self.bright_source_coupling_au_minus3 = coupling_factor / (eps_m * R**3)
        # The unscaled spheroidal g_nm diverges as f -> 0 and has no finite
        # spherical value.  It is diagnostic only; A/B/K use the finite weights
        # above.  Zero and -inf explicitly mark the different normalization.
        self.geometric_factor_by_degree = np.zeros(self.n_max, dtype=float)
        self.log_abs_geometric_factor_by_degree = np.full(
            self.n_max,
            -np.inf,
            dtype=float,
        )

    def _initialize_prolate_scaled(self) -> None:
        """Build prolate coefficients from log-amplitudes, without P*Q overflow."""

        xi0 = self.geometry.xi_surface
        xi_d = self.geometry.xi_qd
        log_P, log_P_prime = _log_legendre_p_and_derivative(self.n_max, xi0)
        log_Q, log_minus_Q_prime = _log_legendre_q_and_derivative(
            self.n_max,
            xi0,
        )
        _, log_minus_Q_prime_qd = _log_legendre_q_and_derivative(
            self.n_max,
            xi_d,
        )
        log_P_derivative_ratio = log_P_prime - log_P
        log_Q_derivative_ratio = log_minus_Q_prime - log_Q
        degree = self.degrees.astype(float)

        if self.geometry.field_polarization == "longitudinal":
            log_denominator = np.logaddexp(
                log_P_derivative_ratio,
                log_Q_derivative_ratio,
            )
            log_L = log_P_derivative_ratio - log_denominator
            log_abs_geometric = log_L + log_P - log_Q
            geometric_sign = -1.0
        else:
            P_derivative_ratio = np.exp(log_P_derivative_ratio)
            minus_Q_derivative_ratio = np.exp(log_Q_derivative_ratio)
            xi_denominator = (xi0 - 1.0) * (xi0 + 1.0)
            radial_P_derivative_ratio = (
                degree * (degree + 1.0) / P_derivative_ratio - xi0
            ) / xi_denominator
            minus_radial_Q_derivative_ratio = (
                xi0 + degree * (degree + 1.0) / minus_Q_derivative_ratio
            ) / xi_denominator
            if (
                np.any(radial_P_derivative_ratio <= 0.0)
                or np.any(minus_radial_Q_derivative_ratio <= 0.0)
                or np.any(~np.isfinite(radial_P_derivative_ratio))
                or np.any(~np.isfinite(minus_radial_Q_derivative_ratio))
            ):
                raise FloatingPointError("Associated-Legendre log derivative failed.")
            log_radial_P_derivative_ratio = np.log(radial_P_derivative_ratio)
            log_minus_radial_Q_derivative_ratio = np.log(
                minus_radial_Q_derivative_ratio
            )
            log_denominator = np.logaddexp(
                log_radial_P_derivative_ratio,
                log_minus_radial_Q_derivative_ratio,
            )
            log_L = log_radial_P_derivative_ratio - log_denominator
            # sqrt(xi^2-1) cancels in P_n^1/Q_n^1.
            log_abs_geometric = log_L + log_P_prime - log_minus_Q_prime
            geometric_sign = 1.0

        self.depolarization_by_degree = np.exp(log_L)
        self.log_abs_geometric_factor_by_degree = np.asarray(
            log_abs_geometric,
            dtype=float,
        )
        with np.errstate(over="ignore", under="ignore"):
            self.geometric_factor_by_degree = geometric_sign * np.exp(
                log_abs_geometric
            )

        f = self.geometry.focal_length_au
        common_log_weight = (
            np.log(2.0 * degree + 1.0)
            + log_abs_geometric
            + 2.0 * log_minus_Q_prime_qd
            - np.log(self.geometry.eps_m)
            - 3.0 * np.log(f)
        )
        if self.geometry.field_polarization == "transverse":
            common_log_weight = common_log_weight - np.log(2.0)
        self.reaction_weight_by_degree_au_minus3 = _exp_representable(
            common_log_weight,
            quantity="Spheroidal reaction weight",
        )

        log_bright_coupling = (
            log_minus_Q_prime_qd[0]
            - np.log(self.geometry.eps_m)
            - 3.0 * np.log(f)
        )
        if self.geometry.field_polarization == "longitudinal":
            log_bright_coupling += np.log(3.0)
            bright_sign = 1.0
        else:
            log_bright_coupling += np.log(1.5)
            bright_sign = -1.0
        self.bright_source_coupling_au_minus3 = bright_sign * float(
            _exp_representable(
                np.asarray(log_bright_coupling),
                quantity="Bright source coupling",
            )
        )

    def _initialize_equatorial_prolate(self) -> None:
        """Full (n, m) spheroidal reaction kernel for a QD at (a+h, 0, 0).

        At ``eta_D=0`` the QD is off the symmetry axis, so every azimuthal
        order contributes.  Writing the scattered potential of a unit source as

        ``Phi_sc = sum_nm c_nm*u_nm(r)*u_nm(r_D)``,
        ``u_nm = Q_n^m(xi)*P_n^m(eta)*cos(m*phi)``,
        ``c_nm = A_nm*chi_nm*g_nm/(eps_m*f)``,

        with ``A_nm=(-1)**m*(2-delta_m0)*(2n+1)*((n-m)!/(n+m)!)**2`` the
        coefficient of the prolate expansion of ``1/|r-r'|``, the reaction
        field projected on the QD dipole is
        ``K = -sum_nm c_nm*(e_d.grad u_nm(r_D))**2``.  The sign of ``A_nm``
        cancels against the sign of ``Q_n^m``, leaving weights that are
        manifestly non-negative.

        The equatorial point has ``e_x = e_xi`` and ``e_z = e_eta``, so a
        transverse (radial) QD dipole differentiates ``Q_n^m`` and keeps
        ``P_n^m(0)``, which is non-zero for even ``n-m``, whereas a
        longitudinal (tangential) dipole keeps ``Q_n^m`` and differentiates
        ``P_n^m``, which is non-zero for odd ``n-m``.
        """

        geometry = self.geometry
        xi0 = geometry.xi_surface
        xi_d = geometry.xi_qd
        focal = geometry.focal_length_au
        surface = _associated_legendre_log_table(self.n_max, xi0)
        qd = _associated_legendre_log_table(self.n_max, xi_d)
        radial = geometry.field_polarization == "transverse"
        required_parity = 0 if radial else 1

        degrees_list: list[int] = []
        orders_list: list[int] = []
        for degree in range(1, self.n_max + 1):
            for order in range(0, degree + 1):
                if (degree - order) % 2 == required_parity:
                    degrees_list.append(degree)
                    orders_list.append(order)
        degrees = np.asarray(degrees_list, dtype=int)
        orders = np.asarray(orders_list, dtype=int)
        rows = degrees - 1
        degree_float = degrees.astype(float)
        order_float = orders.astype(float)

        log_P = surface.log_P[rows, orders]
        log_P_derivative_ratio = surface.log_P_prime[rows, orders] - log_P
        log_abs_Q = surface.log_abs_Q[rows, orders]
        log_Q_derivative_ratio = surface.log_abs_Q_prime[rows, orders] - log_abs_Q
        log_denominator = np.logaddexp(
            log_P_derivative_ratio,
            log_Q_derivative_ratio,
        )
        log_L = log_P_derivative_ratio - log_denominator
        log_abs_geometric = log_L + log_P - log_abs_Q

        log_half_xi_factor = 0.5 * np.log((xi_d - 1.0) * (xi_d + 1.0))
        if radial:
            log_abs_directional_derivative = (
                log_half_xi_factor
                - np.log(focal)
                - np.log(xi_d)
                + qd.log_abs_Q_prime[rows, orders]
                + _log_abs_ferrers_at_zero(degrees, orders)
            )
        else:
            log_abs_directional_derivative = (
                -np.log(focal)
                - np.log(xi_d)
                + qd.log_abs_Q[rows, orders]
                + _log_abs_ferrers_derivative_at_zero(degrees, orders)
            )

        log_abs_expansion_coefficient = (
            np.where(orders == 0, 0.0, np.log(2.0))
            + np.log(2.0 * degree_float + 1.0)
            + 2.0
            * (
                gammaln(degree_float - order_float + 1.0)
                - gammaln(degree_float + order_float + 1.0)
            )
        )

        self.degrees = degrees
        self.azimuthal_orders = orders
        self.depolarization_by_degree = np.exp(log_L)
        self.log_abs_geometric_factor_by_degree = np.asarray(
            log_abs_geometric,
            dtype=float,
        )
        # Q_n^m carries the sign (-1)**m for xi>1, so g_nm=-L*P_n^m/Q_n^m has
        # the sign (-1)**(m+1).
        with np.errstate(over="ignore", under="ignore"):
            self.geometric_factor_by_degree = np.where(
                orders % 2 == 0,
                -1.0,
                1.0,
            ) * np.exp(log_abs_geometric)

        self.reaction_weight_by_degree_au_minus3 = _exp_representable(
            log_abs_expansion_coefficient
            + log_abs_geometric
            + 2.0 * log_abs_directional_derivative
            - np.log(geometry.eps_m)
            - np.log(focal),
            quantity="Equatorial spheroidal reaction weight",
        )

        # The bright channel is the n=1 harmonic of the incident polarization:
        # m=0 for a longitudinal field and m=1 for a transverse one.  Both have
        # |P_1^m(0)| = |dP_1^0/d(eta)(0)| = 1.
        log_common_bright = (
            -np.log(geometry.eps_m)
            - 3.0 * np.log(focal)
            - np.log(xi_d)
        )
        if geometry.field_polarization == "longitudinal":
            log_bright_coupling = (
                np.log(3.0) + qd.log_abs_Q[0, 0] + log_common_bright
            )
            bright_sign = -1.0
        else:
            log_bright_coupling = (
                np.log(1.5)
                + log_half_xi_factor
                + qd.log_abs_Q_prime[0, 1]
                + log_common_bright
            )
            bright_sign = 1.0
        self.bright_source_coupling_au_minus3 = bright_sign * float(
            _exp_representable(
                np.asarray(log_bright_coupling),
                quantity="Bright source coupling",
            )
        )

    @property
    def mode_count(self) -> int:
        """Number of retained (n, m) modes, which is n_max for an axial QD."""
        return int(self.degrees.size)

    @property
    def asymptotic_order_ratio(self) -> float:
        """Leading geometric decay ratio of successive high-order K terms."""

        if self.is_spherical:
            return float((self.geometry.a_au / self.geometry.R_au) ** 2)
        exponent = -2.0 * (
            np.arccosh(self.geometry.xi_qd)
            - np.arccosh(self.geometry.xi_surface)
        )
        return float(np.exp(exponent))

    @classmethod
    def from_params(
        cls,
        params: HybridSystemParams,
        *,
        orientation: DipoleOrientation | None = None,
        qd_position: QDPosition | None = None,
        field_polarization: FieldPolarization | None = None,
        n_max: int = 80,
    ) -> "SpheroidGreenInteraction":
        return cls(
            ProlateSpheroidGeometry.from_params(
                params,
                orientation=orientation,
                qd_position=qd_position,
                field_polarization=field_polarization,
            ),
            n_max=n_max,
        )

    def response_from_epsilon(
        self,
        epsilon_particle: complex | np.ndarray,
    ) -> QuasistaticInteractionResponse:
        """Evaluate A, B and every K_n for one or more particle permittivities."""

        eps = np.asarray(epsilon_particle, dtype=complex)
        if eps.ndim > 1 or eps.size == 0:
            raise ValueError(
                "epsilon_particle must be scalar or a non-empty one-dimensional array."
            )
        if np.any(~np.isfinite(eps)):
            raise ValueError("epsilon_particle must contain only finite values.")
        original_shape = eps.shape
        eps_flat = eps.reshape(-1)
        eps_m = self.geometry.eps_m
        delta = eps_flat - eps_m

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            modal_susceptibility = delta[None, :] / (
                eps_m
                + self.depolarization_by_degree[:, None] * delta[None, :]
            )

        C = eps_m * self.geometry.a_au**2 * self.geometry.c_au / 3.0
        A_flat = C * modal_susceptibility[0]
        B_flat = self.bright_source_coupling_au_minus3 * A_flat
        K_by_degree_flat = (
            self.reaction_weight_by_degree_au_minus3[:, None]
            * modal_susceptibility
        )
        K_flat = np.sum(K_by_degree_flat, axis=0)

        frequency_shape = original_shape
        mode_shape = (self.mode_count,) + frequency_shape
        return QuasistaticInteractionResponse(
            model="spheroid_n1" if self.n_max == 1 else "spheroid_full",
            orientation=self.geometry.orientation,
            eps_m=eps_m,
            A_au3=A_flat.reshape(frequency_shape),
            B=B_flat.reshape(frequency_shape),
            K_au_minus3=K_flat.reshape(frequency_shape),
            degrees=self.degrees,
            K_by_degree_au_minus3=K_by_degree_flat.reshape(mode_shape),
            modal_susceptibility_by_degree=modal_susceptibility.reshape(mode_shape),
            reaction_weight_by_degree_au_minus3=(
                self.reaction_weight_by_degree_au_minus3
            ),
            depolarization_by_degree=self.depolarization_by_degree,
            geometric_factor_by_degree=self.geometric_factor_by_degree,
            log_abs_geometric_factor_by_degree=(
                self.log_abs_geometric_factor_by_degree
            ),
            azimuthal_orders=self.azimuthal_orders,
            qd_position=self.geometry.qd_position,
        )

    def response_from_material(
        self,
        material: MaterialDispersion,
        energies_eV: float | np.ndarray,
    ) -> QuasistaticInteractionResponse:
        energies = np.asarray(energies_eV, dtype=float)
        if energies.ndim > 1 or energies.size == 0:
            raise ValueError(
                "energies_eV must be scalar or a non-empty one-dimensional array."
            )
        if np.any(~np.isfinite(energies)):
            raise ValueError("energies_eV must contain only finite values.")
        return self.response_from_epsilon(material.epsilon_at(energies))


class LegacyDipoleInteraction:
    """Adapter exposing the existing point-dipole model through the A/B/K API."""

    name = "legacy"

    def __init__(self, model: HybridQDPlasmonModel) -> None:
        self.model = model

    def frequency_response(
        self,
        energies_eV: float | np.ndarray,
        *,
        mnp_response: Literal["material", "fit"] = "material",
    ) -> QuasistaticInteractionResponse:
        energies = np.asarray(energies_eV, dtype=float)
        if energies.ndim > 1 or energies.size == 0:
            raise ValueError(
                "energies_eV must be scalar or a non-empty one-dimensional array."
            )
        if np.any(~np.isfinite(energies)):
            raise ValueError("energies_eV must contain only finite values.")
        if mnp_response == "material":
            alpha_dimensionless = self.model.alpha_from_material(energies)
        elif mnp_response == "fit":
            alpha_dimensionless = self.model.alpha_from_fit(energies)
        else:
            raise ValueError("mnp_response must be 'material' or 'fit'.")
        A = self.model.C * np.asarray(alpha_dimensionless, dtype=complex)
        geometry = ProlateSpheroidGeometry.from_params(self.model.params)
        return legacy_dipole_response_from_A(A, geometry)


def legacy_dipole_response_from_A(
    A_au3: complex | np.ndarray,
    geometry: ProlateSpheroidGeometry,
) -> QuasistaticInteractionResponse:
    """Build the old central point-dipole channels from a supplied MNP A."""

    A = np.asarray(A_au3, dtype=complex)
    if A.ndim > 1 or A.size == 0:
        raise ValueError(
            "A_au3 must be scalar or a non-empty one-dimensional array."
        )
    if np.any(~np.isfinite(A)):
        raise ValueError("A_au3 must contain only finite values.")
    J = geometry.geometric_coupling_factor / (
        geometry.eps_m * geometry.R_au**3
    )
    B = A * J
    K = A * J**2
    mode_shape = (1,) + A.shape
    C = geometry.eps_m * geometry.a_au**2 * geometry.c_au / 3.0
    modal_susceptibility = (A / C).reshape(mode_shape)
    reaction_weight = np.asarray([C * J**2], dtype=float)
    eccentricity_squared = float(1.0 - (geometry.a_au / geometry.c_au) ** 2)
    if eccentricity_squared <= 0.0:
        L_long = 1.0 / 3.0
    elif eccentricity_squared < 1.0e-3:
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
    depolarization = np.asarray(
        [
            L_long
            if geometry.field_polarization == "longitudinal"
            else 0.5 * (1.0 - L_long)
        ],
        dtype=float,
    )
    # A central point dipole has no spheroidal-harmonic normalization g_nm.
    # Zero/-inf explicitly mark this diagnostic as unavailable.  The exported
    # susceptibility and reaction weight, in contrast, are physical metadata
    # and obey K_1 = w_1 chi_1 exactly.
    geometric = np.zeros(1, dtype=float)
    return QuasistaticInteractionResponse(
        model="legacy",
        orientation=geometry.orientation,
        eps_m=geometry.eps_m,
        A_au3=A,
        B=B,
        K_au_minus3=K,
        degrees=np.asarray([1]),
        K_by_degree_au_minus3=K.reshape(mode_shape),
        modal_susceptibility_by_degree=modal_susceptibility,
        reaction_weight_by_degree_au_minus3=reaction_weight,
        depolarization_by_degree=depolarization,
        geometric_factor_by_degree=geometric,
        log_abs_geometric_factor_by_degree=np.full(1, -np.inf, dtype=float),
        azimuthal_orders=np.asarray(
            [0 if geometry.field_polarization == "longitudinal" else 1],
            dtype=int,
        ),
        qd_position=geometry.qd_position,
    )


def solve_linear_hybrid_response(
    response: QuasistaticInteractionResponse,
    beta_qd_au3: complex | np.ndarray,
    *,
    eps_m: float,
) -> LinearHybridResponse:
    """Solve the reciprocal weak-field QD--MNP equations for arbitrary A/B/K."""

    if not np.isfinite(eps_m) or eps_m <= 0.0:
        raise ValueError("eps_m must be finite and positive.")
    if not np.isclose(eps_m, response.eps_m, rtol=1.0e-13, atol=0.0):
        raise ValueError(
            "eps_m must match the host permittivity stored in the interaction response."
        )
    beta = np.asarray(beta_qd_au3, dtype=complex)
    if beta.shape != response.A_au3.shape:
        raise ValueError("beta_qd_au3 must have the same shape as A, B and K.")
    denominator = 1.0 - beta * response.K_au_minus3
    qd_dipole = beta * (1.0 + response.B) / denominator
    mnp_dipole = response.A_au3 + response.B * qd_dipole
    # The response is the authoritative source after the compatibility check
    # above; this prevents two almost-equal host values from entering one solve.
    alpha_effective = (mnp_dipole + qd_dipole) / response.eps_m
    return LinearHybridResponse(
        alpha_effective_au3=alpha_effective,
        mnp_dipole_over_field_au3=mnp_dipole,
        qd_dipole_over_field_au3=qd_dipole,
        denominator=denominator,
    )


def qd_linear_polarizability_from_params(
    params: HybridSystemParams,
    energies_eV: float | np.ndarray,
) -> np.ndarray:
    """Externally visible weak-field QD polarizability used by the old core."""

    energies = np.asarray(energies_eV, dtype=float)
    if energies.ndim > 1 or energies.size == 0:
        raise ValueError(
            "energies_eV must be scalar or a non-empty one-dimensional array."
        )
    if np.any(~np.isfinite(energies)):
        raise ValueError("energies_eV must contain only finite values.")
    omega = np.asarray(eV_to_au(energies), dtype=float)
    beta_bare = 2.0 * params.d_au**2 * params.omega0_au / (
        params.omega0_au**2 + (params.Gamma_au - 1j * omega) ** 2
    )
    return np.asarray(params.qd_local_field_factor**2 * beta_bare, dtype=complex)
