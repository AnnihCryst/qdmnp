"""Quasistatic Green response for a QD beside a prolate spheroid.

The spheroid symmetry axis is the Cartesian ``z`` axis and the QD centre is
fixed on the positive equatorial ``x`` axis.  Three scalar, mirror-symmetric
polarization channels are supported:

``orientation='long'``
    The field and both dipoles are parallel to ``z``.
``orientation='trans', side_transverse_alignment='radial'``
    The field and both dipoles are parallel to the centre line ``x``.
``orientation='trans', side_transverse_alignment='tangential'``
    The field and both dipoles are parallel to the azimuthal tangent ``y``.

Unlike the axial kernel, an equatorial point dipole excites several azimuthal
orders ``m`` at each spatial degree ``n``.  The spheroid remains diagonal in
the separated ``(n, m)`` basis, so the existing reciprocal A/B/K reduction is
retained exactly.  Length, field and dipole values follow the atomic-unit
normalization used by the rest of the project.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.special import gammaln, hyp2f1

from qd_mnp_rational_fit import (
    DipoleOrientation,
    HybridSystemParams,
    MaterialDispersion,
    SideTransverseAlignment,
    interaction_factor,
)


EquatorialModeSector = Literal["cos", "sin"]
EquatorialChannel = Literal[
    "long",
    "transverse_radial",
    "transverse_tangential",
]
EquatorialInteractionModel = Literal[
    "spheroid_equatorial_n1",
    "spheroid_equatorial_full",
]

MAX_SUPPORTED_EQUATORIAL_SPATIAL_DEGREE = 80
_LOG_TWO = float(np.log(2.0))
_LOG_PI = float(np.log(np.pi))
_LOG_FLOAT_MAX = float(np.log(np.finfo(float).max))


def _readonly_array(value: np.ndarray, *, dtype=None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _exp_nonnegative(log_values: np.ndarray, *, quantity: str) -> np.ndarray:
    values = np.asarray(log_values, dtype=float)
    if np.any(np.isnan(values)) or np.any(values > _LOG_FLOAT_MAX):
        raise FloatingPointError(f"{quantity} exceeds double-precision range.")
    with np.errstate(under="ignore"):
        return np.asarray(np.exp(values), dtype=float)


def _signed_exp(log_values: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """Exponentiate diagnostic signed amplitudes, retaining valid infinities."""

    logs = np.asarray(log_values, dtype=float)
    sign_values = np.asarray(signs, dtype=float)
    with np.errstate(over="ignore", under="ignore"):
        return sign_values * np.exp(logs)


def _log_double_factorial(values: np.ndarray) -> np.ndarray:
    """Return log(k!!) for integer k >= -1, with (-1)!! = 0!! = 1."""

    integers = np.asarray(values, dtype=int)
    if np.any(integers < -1):
        raise ValueError("Double-factorial arguments must be >= -1.")
    result = np.zeros(integers.shape, dtype=float)
    positive = integers > 0
    even = positive & (integers % 2 == 0)
    odd = positive & ~even
    half_even = integers[even] // 2
    result[even] = half_even * _LOG_TWO + gammaln(half_even + 1.0)
    half_odd = (integers[odd] + 1) // 2
    result[odd] = (
        half_odd * _LOG_TWO
        + gammaln(half_odd + 0.5)
        - 0.5 * _LOG_PI
    )
    return result


def _radial_p_log_and_derivative_ratio(
    n_max: int,
    x: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return log(P_n^m) and P_n^m'/P_n^m for real radial functions.

    The positive forward-ratio recurrence avoids constructing the very large
    associated Legendre functions that occur near the spherical limit.
    """

    if not np.isfinite(x) or x <= 1.0:
        raise ValueError("The prolate radial coordinate must be finite and > 1.")
    log_p = np.full((n_max + 1, n_max + 1), np.nan, dtype=float)
    derivative_ratio = np.full_like(log_p, np.nan)
    x_squared_minus_one = (x - 1.0) * (x + 1.0)

    for order in range(n_max + 1):
        log_base = (
            gammaln(2.0 * order + 1.0)
            - order * _LOG_TWO
            - gammaln(order + 1.0)
            + 0.5 * order * np.log(x_squared_minus_one)
        )
        log_p[order, order] = log_base
        derivative_ratio[order, order] = (
            0.0
            if order == 0
            else order * x / x_squared_minus_one
        )
        if order == n_max:
            continue

        ratio = (2.0 * order + 1.0) * x
        degree = order + 1
        log_p[order, degree] = log_base + np.log(ratio)
        derivative_ratio[order, degree] = (
            degree * x - (degree + order) / ratio
        ) / x_squared_minus_one

        for degree in range(order + 1, n_max):
            next_ratio = (
                (2.0 * degree + 1.0) * x
                - (degree + order) / ratio
            ) / (degree - order + 1.0)
            if not np.isfinite(next_ratio) or next_ratio <= 0.0:
                raise FloatingPointError(
                    "Associated-Legendre P ratio recurrence failed."
                )
            ratio = next_ratio
            next_degree = degree + 1
            log_p[order, next_degree] = (
                log_p[order, degree] + np.log(ratio)
            )
            derivative_ratio[order, next_degree] = (
                next_degree * x - (next_degree + order) / ratio
            ) / x_squared_minus_one

    # Arrays are indexed as [order, degree], hence physical entries satisfy
    # order <= degree and occupy the upper triangle.
    valid = np.triu(np.ones_like(log_p, dtype=bool))
    if (
        np.any(~np.isfinite(log_p[valid]))
        or np.any(~np.isfinite(derivative_ratio[valid]))
        or np.any(derivative_ratio[valid & (np.indices(log_p.shape)[1] > 0)] <= 0.0)
    ):
        raise FloatingPointError("Associated-Legendre P evaluation failed.")
    return log_p, derivative_ratio


def _radial_q_log_and_minus_derivative_ratio(
    degrees: np.ndarray,
    orders: np.ndarray,
    x: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return log(abs(Q_n^m)) and -Q_n^m'/Q_n^m for x > 1.

    Euler's hypergeometric transformation extracts the potentially large
    ``(1-x**-2)**(-m)`` factor analytically.  The remaining hypergeometric
    function is well scaled both near the sphere and near the surface.
    """

    if not np.isfinite(x) or x <= 1.0:
        raise ValueError("The prolate radial coordinate must be finite and > 1.")
    degree = np.asarray(degrees, dtype=float)
    order = np.asarray(orders, dtype=float)
    z = 1.0 / (x * x)
    a = 0.5 * (degree - order + 1.0)
    b = 0.5 * (degree - order + 2.0)
    c = degree + 1.5
    transformed = np.asarray(hyp2f1(a, b, c, z), dtype=float)
    transformed_prime = np.asarray(
        hyp2f1(a + 1.0, b + 1.0, c + 1.0, z),
        dtype=float,
    )
    x_squared_minus_one = (x - 1.0) * (x + 1.0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        log_q = (
            0.5 * _LOG_PI
            + gammaln(degree + order + 1.0)
            - (degree + 1.0) * _LOG_TWO
            - gammaln(degree + 1.5)
            - (degree - order + 1.0) * np.log(x)
            - 0.5 * order * np.log(x_squared_minus_one)
            + np.log(transformed)
        )
        derivative_ratio = (
            -(degree - order + 1.0) / x
            - order * x / x_squared_minus_one
            - (2.0 / x**3)
            * (a * b / c)
            * (transformed_prime / transformed)
        )
        minus_derivative_ratio = -derivative_ratio
    if (
        np.any(~np.isfinite(log_q))
        or np.any(~np.isfinite(minus_derivative_ratio))
        or np.any(minus_derivative_ratio <= 0.0)
    ):
        raise FloatingPointError("Associated-Legendre Q evaluation failed.")
    return np.asarray(log_q), np.asarray(minus_derivative_ratio)


def _angular_log_amplitude(
    degrees: np.ndarray,
    orders: np.ndarray,
    *,
    derivative: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return log magnitude and sign of P_n^m(0) or its derivative."""

    degree = np.asarray(degrees, dtype=int)
    order = np.asarray(orders, dtype=int)
    parity = degree + order
    if derivative:
        if np.any(parity % 2 != 1):
            raise ValueError("P_n^m'(0) is nonzero only when n+m is odd.")
        log_abs = _log_double_factorial(parity) - _log_double_factorial(
            degree - order - 1
        )
        exponent = (parity - 1) // 2
    else:
        if np.any(parity % 2 != 0):
            raise ValueError("P_n^m(0) is nonzero only when n+m is even.")
        log_abs = _log_double_factorial(parity - 1) - _log_double_factorial(
            degree - order
        )
        exponent = parity // 2
    signs = np.where(exponent % 2 == 0, 1.0, -1.0)
    return np.asarray(log_abs, dtype=float), np.asarray(signs, dtype=float)


def _shape_function(t: float) -> float:
    """Stable (atanh(t)-t)/t**3 for 0 <= t < 1."""

    if not np.isfinite(t) or not 0.0 <= t < 1.0:
        raise ValueError("The prolate shape argument must satisfy 0 <= t < 1.")
    if t < 1.0e-3:
        t_squared = t * t
        return float(
            1.0 / 3.0
            + t_squared / 5.0
            + t_squared**2 / 7.0
            + t_squared**3 / 9.0
            + t_squared**4 / 11.0
        )
    return float((np.arctanh(t) - t) / t**3)


@dataclass(frozen=True)
class EquatorialSpheroidGeometry:
    """One point QD on the positive equatorial axis of a prolate spheroid."""

    a_au: float
    c_au: float
    R_au: float
    eps_m: float
    orientation: DipoleOrientation = "long"
    side_transverse_alignment: SideTransverseAlignment | None = None
    qd_radius_au: float = 0.0

    def __post_init__(self) -> None:
        values = np.asarray(
            [self.a_au, self.c_au, self.R_au, self.eps_m, self.qd_radius_au],
            dtype=float,
        )
        if np.any(~np.isfinite(values)):
            raise ValueError("Equatorial spheroid geometry values must be finite.")
        if self.a_au <= 0.0 or self.c_au <= 0.0 or self.R_au <= 0.0:
            raise ValueError("Spheroid semiaxes and QD centre distance must be positive.")
        if self.c_au < self.a_au:
            raise ValueError("The equatorial kernel requires c_au >= a_au.")
        if self.eps_m <= 0.0:
            raise ValueError("eps_m must be positive.")
        if self.qd_radius_au < 0.0:
            raise ValueError("qd_radius_au must be non-negative.")
        if self.R_au <= self.a_au + self.qd_radius_au:
            raise ValueError(
                "The side QD must lie strictly outside the spheroid: require "
                "R_au > a_au + qd_radius_au."
            )
        interaction_factor(
            self.orientation,
            qd_placement="side",
            side_transverse_alignment=self.side_transverse_alignment,
        )

    @classmethod
    def from_params(
        cls,
        params: HybridSystemParams,
        *,
        orientation: DipoleOrientation,
    ) -> "EquatorialSpheroidGeometry":
        if params.qd_placement != "side":
            raise ValueError(
                "EquatorialSpheroidGeometry requires params.qd_placement='side'."
            )
        expected_g = interaction_factor(
            orientation,
            qd_placement="side",
            side_transverse_alignment=params.side_transverse_alignment,
        )
        if not np.isclose(params.G, expected_g, rtol=0.0, atol=1.0e-12):
            raise ValueError(
                "The parameter coupling factor G is inconsistent with the "
                "selected side geometry."
            )
        return cls(
            a_au=float(params.a_au),
            c_au=float(params.c_au),
            R_au=float(params.R_au),
            eps_m=float(params.eps_m),
            orientation=orientation,
            side_transverse_alignment=params.side_transverse_alignment,
            qd_radius_au=float(params.qd_radius_au),
        )

    @property
    def channel(self) -> EquatorialChannel:
        if self.orientation == "long":
            return "long"
        if self.side_transverse_alignment == "radial":
            return "transverse_radial"
        return "transverse_tangential"

    @property
    def qd_placement(self) -> Literal["side"]:
        """Placement tag shared with the native model configuration API."""

        return "side"

    @property
    def focal_length_au(self) -> float:
        return float(np.sqrt((self.c_au - self.a_au) * (self.c_au + self.a_au)))

    @property
    def xi_surface(self) -> float:
        if self.c_au == self.a_au:
            return float("inf")
        return float(self.c_au / self.focal_length_au)

    @property
    def rho_qd_au(self) -> float:
        return float(np.hypot(self.R_au, self.focal_length_au))

    @property
    def xi_qd(self) -> float:
        if self.c_au == self.a_au:
            return float("inf")
        return float(self.rho_qd_au / self.focal_length_au)

    @property
    def surface_gap_au(self) -> float:
        return float(self.R_au - self.a_au - self.qd_radius_au)


@dataclass(frozen=True)
class EquatorialQuasistaticInteractionResponse:
    """Frequency-domain A/B/K response resolved by ``(n,m,sector)`` mode."""

    model: EquatorialInteractionModel
    channel: EquatorialChannel
    orientation: DipoleOrientation
    side_transverse_alignment: SideTransverseAlignment | None
    eps_m: float
    A_au3: np.ndarray
    B: np.ndarray
    K_au_minus3: np.ndarray
    mode_degrees: np.ndarray
    mode_orders: np.ndarray
    mode_sectors: np.ndarray
    K_by_mode_au_minus3: np.ndarray
    K_by_degree_au_minus3: np.ndarray
    modal_susceptibility_by_mode: np.ndarray
    reaction_weight_by_mode_au_minus3: np.ndarray
    depolarization_by_mode: np.ndarray
    geometric_factor_by_mode: np.ndarray
    log_abs_geometric_factor_by_mode: np.ndarray
    source_derivative_by_mode_au_minus1: np.ndarray
    bright_mode_index: int = 0

    def __post_init__(self) -> None:
        A = _readonly_array(self.A_au3, dtype=complex)
        B = _readonly_array(self.B, dtype=complex)
        K = _readonly_array(self.K_au_minus3, dtype=complex)
        degree = _readonly_array(self.mode_degrees, dtype=int)
        order = _readonly_array(self.mode_orders, dtype=int)
        sector = _readonly_array(self.mode_sectors, dtype="U3")
        K_mode = _readonly_array(self.K_by_mode_au_minus3, dtype=complex)
        K_degree = _readonly_array(self.K_by_degree_au_minus3, dtype=complex)
        modal = _readonly_array(self.modal_susceptibility_by_mode, dtype=complex)
        weight = _readonly_array(self.reaction_weight_by_mode_au_minus3, dtype=float)
        depolarization = _readonly_array(self.depolarization_by_mode, dtype=float)
        geometric = _readonly_array(self.geometric_factor_by_mode, dtype=float)
        log_geometric = _readonly_array(
            self.log_abs_geometric_factor_by_mode,
            dtype=float,
        )
        source_derivative = _readonly_array(
            self.source_derivative_by_mode_au_minus1,
            dtype=float,
        )

        if not (A.shape == B.shape == K.shape):
            raise ValueError("A, B and K must have identical frequency shapes.")
        if A.ndim > 1 or A.size == 0:
            raise ValueError(
                "Frequency responses must be scalar or non-empty one-dimensional arrays."
            )
        if degree.ndim != 1 or degree.size < 1:
            raise ValueError("Equatorial mode metadata must be non-empty arrays.")
        mode_count = degree.size
        if not (
            order.shape
            == sector.shape
            == weight.shape
            == depolarization.shape
            == geometric.shape
            == log_geometric.shape
            == source_derivative.shape
            == degree.shape
        ):
            raise ValueError("Every equatorial mode must have complete metadata.")
        expected_mode_shape = (mode_count,) + A.shape
        if K_mode.shape != expected_mode_shape or modal.shape != expected_mode_shape:
            raise ValueError("By-mode arrays have inconsistent frequency shapes.")
        n_max = int(np.max(degree))
        if K_degree.shape != (n_max,) + A.shape:
            raise ValueError("K_by_degree has an inconsistent frequency shape.")
        if not 0 <= self.bright_mode_index < mode_count:
            raise ValueError("bright_mode_index is outside the mode array.")
        if degree[self.bright_mode_index] != 1:
            raise ValueError("The bright equatorial mode must have degree one.")
        if np.any(order < 0) or np.any(order > degree):
            raise ValueError("Equatorial orders must satisfy 0 <= m <= n.")
        if np.any(~np.isin(sector, np.asarray(["cos", "sin"]))):
            raise ValueError("Mode sectors must be 'cos' or 'sin'.")
        finite_arrays = (A, B, K, K_mode, K_degree, modal, weight, depolarization)
        if any(np.any(~np.isfinite(values)) for values in finite_arrays):
            raise FloatingPointError("Non-finite equatorial spheroid response.")
        if (
            np.any(weight < 0.0)
            or np.any(depolarization <= 0.0)
            or np.any(depolarization >= 1.0)
            or np.any(np.isnan(geometric))
            or np.any(np.isnan(log_geometric))
            or np.any(np.isnan(source_derivative))
        ):
            raise FloatingPointError("Invalid equatorial mode geometry.")
        if not np.isfinite(self.eps_m) or self.eps_m <= 0.0:
            raise ValueError("eps_m must be finite and positive.")

        object.__setattr__(self, "A_au3", A)
        object.__setattr__(self, "B", B)
        object.__setattr__(self, "K_au_minus3", K)
        object.__setattr__(self, "mode_degrees", degree)
        object.__setattr__(self, "mode_orders", order)
        object.__setattr__(self, "mode_sectors", sector)
        object.__setattr__(self, "K_by_mode_au_minus3", K_mode)
        object.__setattr__(self, "K_by_degree_au_minus3", K_degree)
        object.__setattr__(self, "modal_susceptibility_by_mode", modal)
        object.__setattr__(self, "reaction_weight_by_mode_au_minus3", weight)
        object.__setattr__(self, "depolarization_by_mode", depolarization)
        object.__setattr__(self, "geometric_factor_by_mode", geometric)
        object.__setattr__(
            self,
            "log_abs_geometric_factor_by_mode",
            log_geometric,
        )
        object.__setattr__(
            self,
            "source_derivative_by_mode_au_minus1",
            source_derivative,
        )
        object.__setattr__(self, "eps_m", float(self.eps_m))
        object.__setattr__(self, "bright_mode_index", int(self.bright_mode_index))

    @property
    def mode_count(self) -> int:
        return int(self.mode_degrees.size)

    @property
    def n_max(self) -> int:
        return int(np.max(self.mode_degrees))

    @property
    def modes(self) -> tuple[tuple[int, int, str], ...]:
        return tuple(
            (int(n), int(m), str(sector))
            for n, m, sector in zip(
                self.mode_degrees,
                self.mode_orders,
                self.mode_sectors,
            )
        )

    @property
    def K_bright_au_minus3(self) -> np.ndarray:
        return self.K_by_mode_au_minus3[self.bright_mode_index]

    @property
    def K_higher_au_minus3(self) -> np.ndarray:
        return self.K_au_minus3 - self.K_bright_au_minus3

    @property
    def cumulative_K_au_minus3(self) -> np.ndarray:
        return np.cumsum(self.K_by_degree_au_minus3, axis=0)

    def relative_half_order_change(self, *, floor_scale: float = 1.0e-14) -> np.ndarray:
        if not np.isfinite(floor_scale) or floor_scale <= 0.0:
            raise ValueError("floor_scale must be finite and positive.")
        cumulative = self.cumulative_K_au_minus3
        fine = cumulative[-1]
        coarse = (
            np.zeros_like(fine)
            if self.n_max == 1
            else cumulative[self.n_max // 2 - 1]
        )
        global_scale = max(float(np.max(np.abs(fine))), np.finfo(float).tiny)
        return np.abs(fine - coarse) / np.maximum(
            np.abs(fine),
            floor_scale * global_scale,
        )

    def relative_tail_block(
        self,
        *,
        block_size: int | None = None,
        floor_scale: float = 1.0e-14,
    ) -> np.ndarray:
        if block_size is None:
            block_size = min(self.n_max, max(4, self.n_max // 8))
        if not isinstance(block_size, (int, np.integer)) or not (
            1 <= block_size <= self.n_max
        ):
            raise ValueError("block_size must be an integer in [1, n_max].")
        if not np.isfinite(floor_scale) or floor_scale <= 0.0:
            raise ValueError("floor_scale must be finite and positive.")
        absolute_modal_sum = np.sum(np.abs(self.K_by_mode_au_minus3), axis=0)
        # Sum absolute *modal* contributions from the final complete degree
        # shells.  Taking abs only after aggregating all m at fixed n would
        # let cancellation inside a shell produce a false convergence
        # certificate.
        tail_mode_mask = self.mode_degrees > self.n_max - block_size
        tail_mass = np.sum(
            np.abs(self.K_by_mode_au_minus3[tail_mode_mask]),
            axis=0,
        )
        denominator = np.maximum(
            np.abs(self.K_au_minus3),
            floor_scale * np.maximum(absolute_modal_sum, np.finfo(float).tiny),
        )
        return tail_mass / denominator

    def truncate(self, n_max: int) -> "EquatorialQuasistaticInteractionResponse":
        if (
            isinstance(n_max, (bool, np.bool_))
            or not isinstance(n_max, (int, np.integer))
            or n_max < 1
            or n_max > self.n_max
        ):
            raise ValueError(f"n_max must lie in [1, {self.n_max}].")
        n_max = int(n_max)
        mask = self.mode_degrees <= n_max
        K_mode = self.K_by_mode_au_minus3[mask]
        return EquatorialQuasistaticInteractionResponse(
            model=(
                "spheroid_equatorial_n1"
                if n_max == 1
                else "spheroid_equatorial_full"
            ),
            channel=self.channel,
            orientation=self.orientation,
            side_transverse_alignment=self.side_transverse_alignment,
            eps_m=self.eps_m,
            A_au3=self.A_au3,
            B=self.B,
            K_au_minus3=np.sum(K_mode, axis=0),
            mode_degrees=self.mode_degrees[mask],
            mode_orders=self.mode_orders[mask],
            mode_sectors=self.mode_sectors[mask],
            K_by_mode_au_minus3=K_mode,
            K_by_degree_au_minus3=self.K_by_degree_au_minus3[:n_max],
            modal_susceptibility_by_mode=(
                self.modal_susceptibility_by_mode[mask]
            ),
            reaction_weight_by_mode_au_minus3=(
                self.reaction_weight_by_mode_au_minus3[mask]
            ),
            depolarization_by_mode=self.depolarization_by_mode[mask],
            geometric_factor_by_mode=self.geometric_factor_by_mode[mask],
            log_abs_geometric_factor_by_mode=(
                self.log_abs_geometric_factor_by_mode[mask]
            ),
            source_derivative_by_mode_au_minus1=(
                self.source_derivative_by_mode_au_minus1[mask]
            ),
            bright_mode_index=0,
        )


class EquatorialSpheroidGreenInteraction:
    """Separated full-QS Green interaction for a QD beside a spheroid."""

    name = "spheroid_equatorial_full"

    def __init__(
        self,
        geometry: EquatorialSpheroidGeometry,
        *,
        n_max: int = 80,
    ) -> None:
        if (
            isinstance(n_max, (bool, np.bool_))
            or not isinstance(n_max, (int, np.integer))
            or n_max < 1
        ):
            raise ValueError("n_max must be an integer >= 1.")
        if n_max > MAX_SUPPORTED_EQUATORIAL_SPATIAL_DEGREE:
            raise ValueError(
                "n_max exceeds MAX_SUPPORTED_EQUATORIAL_SPATIAL_DEGREE="
                f"{MAX_SUPPORTED_EQUATORIAL_SPATIAL_DEGREE}."
            )
        self.geometry = geometry
        self.n_max = int(n_max)
        self.is_spherical = bool(geometry.c_au == geometry.a_au)
        self.C_au3 = float(geometry.eps_m * geometry.a_au**2 * geometry.c_au / 3.0)

        degrees, orders, sectors = self._enumerate_modes()
        self.mode_degrees = _readonly_array(degrees, dtype=int)
        self.mode_orders = _readonly_array(orders, dtype=int)
        self.mode_sectors = _readonly_array(sectors, dtype="U3")
        self.mode_count = int(self.mode_degrees.size)
        self.bright_mode_index = 0

        if self.is_spherical:
            self._initialize_spherical_limit()
        else:
            self._initialize_prolate()

        self.depolarization_by_mode = _readonly_array(
            self.depolarization_by_mode,
            dtype=float,
        )
        self.reaction_weight_by_mode_au_minus3 = _readonly_array(
            self.reaction_weight_by_mode_au_minus3,
            dtype=float,
        )
        self.geometric_factor_by_mode = _readonly_array(
            self.geometric_factor_by_mode,
            dtype=float,
        )
        self.log_abs_geometric_factor_by_mode = _readonly_array(
            self.log_abs_geometric_factor_by_mode,
            dtype=float,
        )
        self.source_derivative_by_mode_au_minus1 = _readonly_array(
            self.source_derivative_by_mode_au_minus1,
            dtype=float,
        )
        self.bright_source_coupling_au_minus3 = float(
            self.bright_source_coupling_au_minus3
        )

        if (
            np.any(~np.isfinite(self.depolarization_by_mode))
            or np.any(self.depolarization_by_mode <= 0.0)
            or np.any(self.depolarization_by_mode >= 1.0)
            or np.any(~np.isfinite(self.reaction_weight_by_mode_au_minus3))
            or np.any(self.reaction_weight_by_mode_au_minus3 < 0.0)
            or self.reaction_weight_by_mode_au_minus3[0] <= 0.0
            or not np.isfinite(self.bright_source_coupling_au_minus3)
        ):
            raise FloatingPointError("Invalid equatorial modal coefficients.")

    @classmethod
    def from_params(
        cls,
        params: HybridSystemParams,
        *,
        orientation: DipoleOrientation,
        n_max: int = 80,
    ) -> "EquatorialSpheroidGreenInteraction":
        return cls(
            EquatorialSpheroidGeometry.from_params(
                params,
                orientation=orientation,
            ),
            n_max=n_max,
        )

    @property
    def channel(self) -> EquatorialChannel:
        return self.geometry.channel

    @property
    def modes(self) -> tuple[tuple[int, int, str], ...]:
        return tuple(
            (int(n), int(m), str(sector))
            for n, m, sector in zip(
                self.mode_degrees,
                self.mode_orders,
                self.mode_sectors,
            )
        )

    @property
    def asymptotic_order_ratio(self) -> float:
        if self.is_spherical:
            return float((self.geometry.a_au / self.geometry.R_au) ** 2)
        exponent = -2.0 * (
            np.arccosh(self.geometry.xi_qd)
            - np.arccosh(self.geometry.xi_surface)
        )
        return float(np.exp(exponent))

    def _enumerate_modes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        degrees: list[int] = []
        orders: list[int] = []
        sectors: list[str] = []
        for degree in range(1, self.n_max + 1):
            if self.channel == "long":
                selected_orders = [
                    order
                    for order in range(degree + 1)
                    if (degree + order) % 2 == 1
                ]
                sector = "cos"
            elif self.channel == "transverse_radial":
                selected_orders = [
                    order
                    for order in range(degree + 1)
                    if (degree + order) % 2 == 0
                ]
                sector = "cos"
            else:
                selected_orders = [
                    order
                    for order in range(1, degree + 1)
                    if (degree + order) % 2 == 0
                ]
                sector = "sin"
            for order in selected_orders:
                degrees.append(degree)
                orders.append(order)
                sectors.append(sector)
        return (
            np.asarray(degrees, dtype=int),
            np.asarray(orders, dtype=int),
            np.asarray(sectors, dtype="U3"),
        )

    def _bright_coupling(self) -> float:
        eps_m = self.geometry.eps_m
        R = self.geometry.R_au
        if self.is_spherical:
            factor = 2.0 if self.channel == "transverse_radial" else -1.0
            return float(factor / (eps_m * R**3))

        f = self.geometry.focal_length_au
        rho = self.geometry.rho_qd_au
        t = f / rho
        shape = _shape_function(t)
        inverse = 1.0 / ((1.0 - t) * (1.0 + t))
        common = 3.0 / (eps_m * rho**3)
        if self.channel == "long":
            return float(-common * shape)
        if self.channel == "transverse_radial":
            return float(0.5 * common * (shape + inverse))
        return float(-0.5 * common * (inverse - shape))

    def _initialize_spherical_limit(self) -> None:
        degree = self.mode_degrees.astype(float)
        order = self.mode_orders.astype(float)
        degree_int = self.mode_degrees
        order_int = self.mode_orders
        log_multiplicity = np.where(order_int == 0, 0.0, _LOG_TWO)
        log_a_coefficient = (
            gammaln(2.0 * degree + 1.0)
            - degree * _LOG_TWO
            - gammaln(degree + 1.0)
            - gammaln(degree - order + 1.0)
        )
        log_b_coefficient = (
            (degree + 1.0) * _LOG_TWO
            + gammaln(degree + 2.0)
            + gammaln(degree + order + 1.0)
            - gammaln(2.0 * degree + 3.0)
        )
        log_factorial_ratio = (
            gammaln(degree - order + 1.0)
            - gammaln(degree + order + 1.0)
        )
        if self.channel == "long":
            log_angular, _ = _angular_log_amplitude(
                degree_int,
                order_int,
                derivative=True,
            )
        else:
            log_angular, _ = _angular_log_amplitude(
                degree_int,
                order_int,
                derivative=False,
            )
            if self.channel == "transverse_radial":
                log_angular = log_angular + np.log(degree + 1.0)
            else:
                log_angular = log_angular + np.log(order)

        log_weight = (
            log_multiplicity
            + np.log(degree)
            + log_a_coefficient
            + log_b_coefficient
            + 2.0 * log_factorial_ratio
            + 2.0 * log_angular
            + (2.0 * degree + 1.0) * np.log(self.geometry.a_au)
            - (2.0 * degree + 4.0) * np.log(self.geometry.R_au)
            - np.log(self.geometry.eps_m)
        )
        self.depolarization_by_mode = degree / (2.0 * degree + 1.0)
        self.reaction_weight_by_mode_au_minus3 = _exp_nonnegative(
            log_weight,
            quantity="Spherical equatorial reaction weight",
        )
        self.bright_source_coupling_au_minus3 = self._bright_coupling()
        self.reaction_weight_by_mode_au_minus3[0] = (
            self.C_au3 * self.bright_source_coupling_au_minus3**2
        )
        self.geometric_factor_by_mode = np.zeros(self.mode_count, dtype=float)
        self.log_abs_geometric_factor_by_mode = np.full(
            self.mode_count,
            -np.inf,
            dtype=float,
        )
        self.source_derivative_by_mode_au_minus1 = np.zeros(
            self.mode_count,
            dtype=float,
        )

    def _initialize_prolate(self) -> None:
        degree = self.mode_degrees
        order = self.mode_orders
        degree_float = degree.astype(float)
        order_float = order.astype(float)
        f = self.geometry.focal_length_au
        rho = self.geometry.rho_qd_au
        R = self.geometry.R_au
        xi0 = self.geometry.xi_surface
        xi_d = self.geometry.xi_qd

        log_p_all, p_derivative_ratio_all = _radial_p_log_and_derivative_ratio(
            self.n_max,
            xi0,
        )
        log_p = log_p_all[order, degree]
        p_derivative_ratio = p_derivative_ratio_all[order, degree]
        log_q_surface, minus_q_derivative_ratio_surface = (
            _radial_q_log_and_minus_derivative_ratio(degree, order, xi0)
        )
        log_q_qd, minus_q_derivative_ratio_qd = (
            _radial_q_log_and_minus_derivative_ratio(degree, order, xi_d)
        )

        log_p_ratio = np.log(p_derivative_ratio)
        log_q_ratio = np.log(minus_q_derivative_ratio_surface)
        log_denominator = np.logaddexp(log_p_ratio, log_q_ratio)
        log_L = log_p_ratio - log_denominator
        self.depolarization_by_mode = np.exp(log_L)
        log_abs_geometric = log_L + log_p - log_q_surface
        geometric_sign = np.where(order % 2 == 0, -1.0, 1.0)
        self.log_abs_geometric_factor_by_mode = log_abs_geometric
        self.geometric_factor_by_mode = _signed_exp(
            log_abs_geometric,
            geometric_sign,
        )

        log_H = (
            np.log(2.0 * degree_float + 1.0)
            + np.where(order == 0, 0.0, _LOG_TWO)
            + 2.0
            * (
                gammaln(degree_float - order_float + 1.0)
                - gammaln(degree_float + order_float + 1.0)
            )
        )
        if self.channel == "long":
            log_angular, angular_sign = _angular_log_amplitude(
                degree,
                order,
                derivative=True,
            )
            log_D = log_q_qd + log_angular - np.log(rho)
            D_sign = np.where(
                (order + (degree + order - 1) // 2) % 2 == 0,
                1.0,
                -1.0,
            )
            del angular_sign
        elif self.channel == "transverse_radial":
            log_angular, angular_sign = _angular_log_amplitude(
                degree,
                order,
                derivative=False,
            )
            log_D = (
                np.log(R)
                + log_q_qd
                + np.log(minus_q_derivative_ratio_qd)
                + log_angular
                - np.log(f)
                - np.log(rho)
            )
            D_sign = np.where(
                (order + 1 + (degree + order) // 2) % 2 == 0,
                1.0,
                -1.0,
            )
            del angular_sign
        else:
            log_angular, angular_sign = _angular_log_amplitude(
                degree,
                order,
                derivative=False,
            )
            log_D = (
                np.log(order_float)
                + log_q_qd
                + log_angular
                - np.log(R)
            )
            D_sign = np.where(
                (order + (degree + order) // 2) % 2 == 0,
                1.0,
                -1.0,
            )
            del angular_sign

        self.source_derivative_by_mode_au_minus1 = _signed_exp(log_D, D_sign)
        log_weight = (
            log_H
            + log_abs_geometric
            + 2.0 * log_D
            - np.log(self.geometry.eps_m)
            - np.log(f)
        )
        self.reaction_weight_by_mode_au_minus3 = _exp_nonnegative(
            log_weight,
            quantity="Prolate equatorial reaction weight",
        )
        self.bright_source_coupling_au_minus3 = self._bright_coupling()
        self.reaction_weight_by_mode_au_minus3[0] = (
            self.C_au3 * self.bright_source_coupling_au_minus3**2
        )

    def response_from_modal_susceptibility(
        self,
        modal_susceptibility: np.ndarray,
    ) -> EquatorialQuasistaticInteractionResponse:
        """Assemble A/B/K from one susceptibility per active spatial mode."""

        modal = np.asarray(modal_susceptibility, dtype=complex)
        if modal.ndim not in (1, 2) or modal.shape[0] != self.mode_count:
            raise ValueError(
                "modal_susceptibility must have shape (mode_count,) or "
                "(mode_count, frequency_count)."
            )
        if modal.ndim == 2 and modal.shape[1] == 0:
            raise ValueError("The modal frequency axis must be non-empty.")
        if np.any(~np.isfinite(modal)):
            raise FloatingPointError("Modal susceptibility contains non-finite values.")

        frequency_shape = modal.shape[1:]
        weight_shape = (self.mode_count,) + (1,) * len(frequency_shape)
        K_by_mode = (
            self.reaction_weight_by_mode_au_minus3.reshape(weight_shape) * modal
        )
        K_by_degree = np.zeros((self.n_max,) + frequency_shape, dtype=complex)
        for mode_index, degree in enumerate(self.mode_degrees):
            K_by_degree[degree - 1] += K_by_mode[mode_index]
        A = self.C_au3 * modal[self.bright_mode_index]
        B = self.bright_source_coupling_au_minus3 * A
        K = np.sum(K_by_degree, axis=0)
        return EquatorialQuasistaticInteractionResponse(
            model=(
                "spheroid_equatorial_n1"
                if self.n_max == 1
                else "spheroid_equatorial_full"
            ),
            channel=self.channel,
            orientation=self.geometry.orientation,
            side_transverse_alignment=(
                self.geometry.side_transverse_alignment
            ),
            eps_m=self.geometry.eps_m,
            A_au3=A,
            B=B,
            K_au_minus3=K,
            mode_degrees=self.mode_degrees,
            mode_orders=self.mode_orders,
            mode_sectors=self.mode_sectors,
            K_by_mode_au_minus3=K_by_mode,
            K_by_degree_au_minus3=K_by_degree,
            modal_susceptibility_by_mode=modal,
            reaction_weight_by_mode_au_minus3=(
                self.reaction_weight_by_mode_au_minus3
            ),
            depolarization_by_mode=self.depolarization_by_mode,
            geometric_factor_by_mode=self.geometric_factor_by_mode,
            log_abs_geometric_factor_by_mode=(
                self.log_abs_geometric_factor_by_mode
            ),
            source_derivative_by_mode_au_minus1=(
                self.source_derivative_by_mode_au_minus1
            ),
            bright_mode_index=self.bright_mode_index,
        )

    def response_from_epsilon(
        self,
        epsilon_particle: complex | np.ndarray,
    ) -> EquatorialQuasistaticInteractionResponse:
        eps = np.asarray(epsilon_particle, dtype=complex)
        if eps.ndim > 1 or eps.size == 0:
            raise ValueError(
                "epsilon_particle must be scalar or a non-empty one-dimensional array."
            )
        if np.any(~np.isfinite(eps)):
            raise ValueError("epsilon_particle must contain only finite values.")
        original_shape = eps.shape
        delta = eps.reshape(-1) - self.geometry.eps_m
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            modal = delta[None, :] / (
                self.geometry.eps_m
                + self.depolarization_by_mode[:, None] * delta[None, :]
            )
        return self.response_from_modal_susceptibility(
            modal.reshape((self.mode_count,) + original_shape)
        )

    def response_from_material(
        self,
        material: MaterialDispersion,
        energies_eV: float | np.ndarray,
    ) -> EquatorialQuasistaticInteractionResponse:
        energies = np.asarray(energies_eV, dtype=float)
        if energies.ndim > 1 or energies.size == 0:
            raise ValueError(
                "energies_eV must be scalar or a non-empty one-dimensional array."
            )
        if np.any(~np.isfinite(energies)):
            raise ValueError("energies_eV must contain only finite values.")
        return self.response_from_epsilon(material.epsilon_at(energies))
