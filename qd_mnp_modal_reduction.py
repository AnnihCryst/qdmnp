"""Passive reduction of a full-QS dark spatial spectrum.

The full spheroidal reaction kernel is a positive weighted sum of modal
susceptibilities.  This module reduces only the dark part of that sum by
clustering its positive spectral measure in depolarization-factor space.  It
does not fit arbitrary poles and therefore preserves the common material
response used by the bright mode.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Literal

import numpy as np


ReductionPolicy = Literal["raise", "warn", "ignore"]


def _readonly(value: np.ndarray, *, dtype=None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _validated_vector(name: str, value: np.ndarray, *, dtype) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    return result


def modal_measure_sha256(
    depolarization: np.ndarray,
    reaction_weights_au_minus3: np.ndarray,
    *,
    bright_index: int,
) -> str:
    """Return a canonical fingerprint of one ordered spatial modal measure."""

    L = _validated_vector("depolarization", depolarization, dtype=float)
    weights = _validated_vector(
        "reaction_weights_au_minus3",
        reaction_weights_au_minus3,
        dtype=float,
    )
    if L.size == 0 or weights.shape != L.shape:
        raise ValueError("Modal depolarizations and weights must be aligned.")
    if isinstance(bright_index, (bool, np.bool_)) or not isinstance(
        bright_index,
        (int, np.integer),
    ) or not (0 <= int(bright_index) < L.size):
        raise ValueError("bright_index must identify one spatial mode.")
    digest = hashlib.sha256()
    digest.update(b"qdmnp.modal-measure.v1\0")
    digest.update(
        np.asarray([L.size, int(bright_index)], dtype="<i8").tobytes(order="C")
    )
    digest.update(np.asarray(L, dtype="<f8").tobytes(order="C"))
    digest.update(np.asarray(weights, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class DarkKernelReductionDiagnostics:
    """Accuracy and passivity certificate for a positive dark-kernel reduction."""

    fit_normalized_rms: float
    fit_max_normalized_error: float
    audit_normalized_rms: float
    audit_max_normalized_error: float
    max_normalized_rms: float
    max_normalized_error: float
    passive_on_audit_grid: bool
    total_weight_relative_error: float
    first_moment_relative_error: float
    accepted: bool
    fit_grid_points: int
    audit_grid_points: int
    original_dark_mode_count: int
    positive_dark_mode_count: int
    reduced_node_count: int
    rms_tolerance: float
    max_tolerance: float


@dataclass(frozen=True)
class PositiveDarkKernelReduction:
    """Positive quadrature representation of the dark spatial measure."""

    bright_depolarization: float
    depolarization_nodes: np.ndarray
    weights_au_minus3: np.ndarray
    source_mode_indices: tuple[tuple[int, ...], ...]
    source_measure_sha256: str
    diagnostics: DarkKernelReductionDiagnostics

    def __post_init__(self) -> None:
        nodes = _readonly(self.depolarization_nodes, dtype=float)
        weights = _readonly(self.weights_au_minus3, dtype=float)
        if nodes.ndim != 1 or weights.shape != nodes.shape:
            raise ValueError("Reduction nodes and weights must be aligned vectors.")
        if np.any((nodes <= 0.0) | (nodes >= 1.0)):
            raise ValueError("Reduced depolarization nodes must lie in (0, 1).")
        if np.any(weights <= 0.0):
            raise ValueError("Reduced spectral weights must be strictly positive.")
        groups = tuple(tuple(int(index) for index in group) for group in self.source_mode_indices)
        if len(groups) != nodes.size or any(len(group) == 0 for group in groups):
            raise ValueError("Every reduced node must have a non-empty source group.")
        if not np.isfinite(self.bright_depolarization) or not (
            0.0 < self.bright_depolarization < 1.0
        ):
            raise ValueError("bright_depolarization must lie in (0, 1).")
        fingerprint = str(self.source_measure_sha256).lower()
        if len(fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in fingerprint
        ):
            raise ValueError("source_measure_sha256 must be a SHA-256 hex digest.")
        object.__setattr__(self, "depolarization_nodes", nodes)
        object.__setattr__(self, "weights_au_minus3", weights)
        object.__setattr__(self, "source_mode_indices", groups)
        object.__setattr__(self, "bright_depolarization", float(self.bright_depolarization))
        object.__setattr__(self, "source_measure_sha256", fingerprint)

    @property
    def node_count(self) -> int:
        return int(self.depolarization_nodes.size)

    def modal_susceptibility_from_bright(
        self,
        bright_susceptibility: complex | np.ndarray,
    ) -> np.ndarray:
        """Return reduced dark susceptibilities from the common bright response."""

        H = np.asarray(bright_susceptibility, dtype=complex)
        if H.ndim > 1 or H.size == 0 or np.any(~np.isfinite(H)):
            raise ValueError(
                "bright_susceptibility must be a finite scalar or one-dimensional array."
            )
        delta = self.depolarization_nodes - self.bright_depolarization
        return H[None, ...] / (
            1.0 + delta.reshape((self.node_count,) + (1,) * H.ndim) * H[None, ...]
        )

    def evaluate_from_bright(
        self,
        bright_susceptibility: complex | np.ndarray,
    ) -> np.ndarray:
        modal = self.modal_susceptibility_from_bright(bright_susceptibility)
        return np.sum(
            self.weights_au_minus3.reshape(
                (self.node_count,) + (1,) * (modal.ndim - 1)
            )
            * modal,
            axis=0,
        )


@dataclass(frozen=True)
class ReducedInteractionResponse:
    """Frequency response of the exact bright plus reduced dark realization."""

    orientation: str
    eps_m: float
    A_au3: np.ndarray
    B: np.ndarray
    K_au_minus3: np.ndarray
    K_by_mode_au_minus3: np.ndarray
    modal_susceptibility_by_mode: np.ndarray
    depolarization_by_mode: np.ndarray
    reaction_weight_by_mode_au_minus3: np.ndarray
    bright_mode_index: int
    spatial_order_max: int
    exact_mode_count: int
    reduction: PositiveDarkKernelReduction
    model: str = "spheroid_full_reduced"

    def __post_init__(self) -> None:
        A = _readonly(self.A_au3, dtype=complex)
        B = _readonly(self.B, dtype=complex)
        K = _readonly(self.K_au_minus3, dtype=complex)
        K_by_mode = _readonly(self.K_by_mode_au_minus3, dtype=complex)
        modal = _readonly(self.modal_susceptibility_by_mode, dtype=complex)
        L = _readonly(self.depolarization_by_mode, dtype=float)
        weights = _readonly(self.reaction_weight_by_mode_au_minus3, dtype=float)
        if not (A.shape == B.shape == K.shape):
            raise ValueError("A, B and K must have identical frequency shapes.")
        if A.ndim > 1 or A.size == 0:
            raise ValueError("Frequency responses must be scalar or one-dimensional.")
        if L.ndim != 1 or weights.shape != L.shape or L.size < 1:
            raise ValueError("Reduced modal geometry arrays are inconsistent.")
        expected = (L.size,) + A.shape
        if K_by_mode.shape != expected or modal.shape != expected:
            raise ValueError("Reduced modal response arrays have inconsistent shapes.")
        if np.any(~np.isfinite(A)) or np.any(~np.isfinite(B)) or np.any(~np.isfinite(K)):
            raise FloatingPointError("Reduced A/B/K contains a non-finite value.")
        if np.any(~np.isfinite(K_by_mode)) or np.any(~np.isfinite(modal)):
            raise FloatingPointError("Reduced modal response contains a non-finite value.")
        if not np.isfinite(self.eps_m) or self.eps_m <= 0.0:
            raise ValueError("eps_m must be finite and positive.")
        if not (0 <= int(self.bright_mode_index) < L.size):
            raise ValueError("bright_mode_index is outside the reduced mode table.")
        object.__setattr__(self, "A_au3", A)
        object.__setattr__(self, "B", B)
        object.__setattr__(self, "K_au_minus3", K)
        object.__setattr__(self, "K_by_mode_au_minus3", K_by_mode)
        object.__setattr__(self, "modal_susceptibility_by_mode", modal)
        object.__setattr__(self, "depolarization_by_mode", L)
        object.__setattr__(self, "reaction_weight_by_mode_au_minus3", weights)
        object.__setattr__(self, "bright_mode_index", int(self.bright_mode_index))
        object.__setattr__(self, "spatial_order_max", int(self.spatial_order_max))
        object.__setattr__(self, "exact_mode_count", int(self.exact_mode_count))
        object.__setattr__(self, "eps_m", float(self.eps_m))

    @property
    def K_bright_au_minus3(self) -> np.ndarray:
        return self.K_by_mode_au_minus3[self.bright_mode_index]

    @property
    def K_higher_au_minus3(self) -> np.ndarray:
        return self.K_au_minus3 - self.K_bright_au_minus3


def _dark_response(
    depolarization: np.ndarray,
    weights: np.ndarray,
    bright_depolarization: float,
    H: np.ndarray,
) -> np.ndarray:
    if depolarization.size == 0:
        return np.zeros_like(H, dtype=complex)
    transformed = H[None, :] / (
        1.0
        + (depolarization - bright_depolarization)[:, None] * H[None, :]
    )
    return np.sum(weights[:, None] * transformed, axis=0)


def _error_metrics(target: np.ndarray, approximation: np.ndarray) -> tuple[float, float]:
    error = approximation - target
    tiny = np.finfo(float).tiny
    rms_scale = max(float(np.sqrt(np.mean(np.abs(target) ** 2))), tiny)
    max_scale = max(float(np.max(np.abs(target))), tiny)
    return (
        float(np.sqrt(np.mean(np.abs(error) ** 2)) / rms_scale),
        float(np.max(np.abs(error)) / max_scale),
    )


def _group_nodes_and_weights(
    groups: list[np.ndarray],
    depolarization: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    node_values: list[float] = []
    group_weights: list[float] = []
    for group in groups:
        group_weight = float(np.sum(weights[group]))
        if group_weight <= 0.0:
            raise RuntimeError("Internal reduction group has non-positive weight.")
        node_values.append(float(np.dot(weights[group], depolarization[group]) / group_weight))
        group_weights.append(group_weight)
    return np.asarray(node_values), np.asarray(group_weights)


def _weighted_split(group: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if group.size < 2:
        raise ValueError("A singleton reduction group cannot be split.")
    cumulative = np.cumsum(weights[group])
    split = int(np.searchsorted(cumulative, 0.5 * cumulative[-1], side="right"))
    split = min(max(split, 1), group.size - 1)
    return group[:split], group[split:]


def reduce_positive_dark_measure(
    depolarization: np.ndarray,
    reaction_weights_au_minus3: np.ndarray,
    *,
    bright_index: int,
    bright_susceptibility_fit: np.ndarray,
    bright_susceptibility_audit: np.ndarray,
    rms_tolerance: float = 1.0e-6,
    max_tolerance: float = 1.0e-4,
    max_nodes: int | None = None,
    zero_weight_relative_tolerance: float = 0.0,
    policy: ReductionPolicy = "raise",
) -> PositiveDarkKernelReduction:
    """Adaptively cluster a positive dark spatial spectrum.

    The construction and audit arrays must be sampled on different frequency
    grids by the caller.  Groups are contiguous after sorting by
    depolarization factor and are split at their weighted median.
    """

    L = _validated_vector("depolarization", depolarization, dtype=float)
    weights = _validated_vector(
        "reaction_weights_au_minus3", reaction_weights_au_minus3, dtype=float
    )
    H_fit = _validated_vector(
        "bright_susceptibility_fit", bright_susceptibility_fit, dtype=complex
    )
    H_audit = _validated_vector(
        "bright_susceptibility_audit", bright_susceptibility_audit, dtype=complex
    )
    if L.size == 0 or weights.shape != L.shape:
        raise ValueError("Modal depolarizations and weights must be non-empty and aligned.")
    if np.any((L <= 0.0) | (L >= 1.0)):
        raise ValueError("Every modal depolarization factor must lie in (0, 1).")
    if np.any(weights < 0.0):
        raise ValueError("Reaction weights must be non-negative.")
    if isinstance(bright_index, (bool, np.bool_)) or not isinstance(
        bright_index, (int, np.integer)
    ) or not (0 <= int(bright_index) < L.size):
        raise ValueError("bright_index must identify one retained spatial mode.")
    bright_index = int(bright_index)
    if policy not in {"raise", "warn", "ignore"}:
        raise ValueError("policy must be 'raise', 'warn' or 'ignore'.")
    for name, value in (
        ("rms_tolerance", rms_tolerance),
        ("max_tolerance", max_tolerance),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    if (
        not np.isfinite(zero_weight_relative_tolerance)
        or zero_weight_relative_tolerance < 0.0
    ):
        raise ValueError(
            "zero_weight_relative_tolerance must be finite and non-negative."
        )

    dark_mask = np.ones(L.size, dtype=bool)
    dark_mask[bright_index] = False
    dark_indices = np.flatnonzero(dark_mask)
    total_dark_weight = float(np.sum(weights[dark_indices]))
    cutoff = float(zero_weight_relative_tolerance) * max(
        total_dark_weight,
        float(weights[bright_index]),
        np.finfo(float).tiny,
    )
    positive_indices = dark_indices[weights[dark_indices] > cutoff]
    if positive_indices.size == 0:
        diagnostics = DarkKernelReductionDiagnostics(
            fit_normalized_rms=0.0,
            fit_max_normalized_error=0.0,
            audit_normalized_rms=0.0,
            audit_max_normalized_error=0.0,
            max_normalized_rms=0.0,
            max_normalized_error=0.0,
            passive_on_audit_grid=True,
            total_weight_relative_error=0.0,
            first_moment_relative_error=0.0,
            accepted=True,
            fit_grid_points=int(H_fit.size),
            audit_grid_points=int(H_audit.size),
            original_dark_mode_count=int(dark_indices.size),
            positive_dark_mode_count=0,
            reduced_node_count=0,
            rms_tolerance=float(rms_tolerance),
            max_tolerance=float(max_tolerance),
        )
        return PositiveDarkKernelReduction(
            bright_depolarization=float(L[bright_index]),
            depolarization_nodes=np.empty(0, dtype=float),
            weights_au_minus3=np.empty(0, dtype=float),
            source_mode_indices=tuple(),
            source_measure_sha256=modal_measure_sha256(
                L,
                weights,
                bright_index=bright_index,
            ),
            diagnostics=diagnostics,
        )

    order = positive_indices[np.argsort(L[positive_indices], kind="stable")]
    if max_nodes is None:
        max_nodes = int(order.size)
    if isinstance(max_nodes, (bool, np.bool_)) or not isinstance(
        max_nodes, (int, np.integer)
    ) or not (1 <= int(max_nodes) <= order.size):
        raise ValueError(f"max_nodes must be an integer in [1, {order.size}].")
    max_nodes = int(max_nodes)

    exact_fit = _dark_response(
        L[order], weights[order], float(L[bright_index]), H_fit
    )
    exact_audit = _dark_response(
        L[order], weights[order], float(L[bright_index]), H_audit
    )
    groups = [order]
    while True:
        nodes, group_weights = _group_nodes_and_weights(groups, L, weights)
        reduced_fit = _dark_response(
            nodes, group_weights, float(L[bright_index]), H_fit
        )
        fit_rms, fit_max = _error_metrics(exact_fit, reduced_fit)
        construction_accepted = bool(
            fit_rms <= rms_tolerance and fit_max <= max_tolerance
        )
        if construction_accepted or len(groups) >= max_nodes:
            break

        scores: list[float] = []
        for group in groups:
            if group.size < 2:
                scores.append(-np.inf)
                continue
            group_node, group_weight = _group_nodes_and_weights([group], L, weights)
            group_exact = _dark_response(
                L[group], weights[group], float(L[bright_index]), H_fit
            )
            group_reduced = _dark_response(
                group_node, group_weight, float(L[bright_index]), H_fit
            )
            scores.append(float(np.max(np.abs(group_exact - group_reduced))))
        split_index = int(np.argmax(scores))
        if not np.isfinite(scores[split_index]):
            break
        left, right = _weighted_split(groups[split_index], weights)
        groups[split_index : split_index + 1] = [left, right]

    # The audit grid is a genuine holdout: it is first consulted only after
    # construction has fixed both the groups and their node count.
    reduced_audit = _dark_response(
        nodes, group_weights, float(L[bright_index]), H_audit
    )
    audit_rms, audit_max = _error_metrics(exact_audit, reduced_audit)
    total_weight_error = abs(
        float(np.sum(group_weights)) - float(np.sum(weights[order]))
    )
    total_weight_error /= max(
        float(np.sum(weights[order])),
        np.finfo(float).tiny,
    )
    exact_moment = float(np.dot(weights[order], L[order]))
    reduced_moment = float(np.dot(group_weights, nodes))
    moment_error = abs(reduced_moment - exact_moment) / max(
        abs(exact_moment),
        np.finfo(float).tiny,
    )
    passive = bool(
        np.min(reduced_audit.imag)
        >= -1.0e-12 * max(float(np.max(np.abs(reduced_audit))), 1.0)
    )
    accepted = bool(
        fit_rms <= rms_tolerance
        and fit_max <= max_tolerance
        and audit_rms <= rms_tolerance
        and audit_max <= max_tolerance
        and passive
    )
    diagnostics = DarkKernelReductionDiagnostics(
        fit_normalized_rms=fit_rms,
        fit_max_normalized_error=fit_max,
        audit_normalized_rms=audit_rms,
        audit_max_normalized_error=audit_max,
        max_normalized_rms=max(fit_rms, audit_rms),
        max_normalized_error=max(fit_max, audit_max),
        passive_on_audit_grid=passive,
        total_weight_relative_error=total_weight_error,
        first_moment_relative_error=moment_error,
        accepted=accepted,
        fit_grid_points=int(H_fit.size),
        audit_grid_points=int(H_audit.size),
        original_dark_mode_count=int(dark_indices.size),
        positive_dark_mode_count=int(order.size),
        reduced_node_count=len(groups),
        rms_tolerance=float(rms_tolerance),
        max_tolerance=float(max_tolerance),
    )
    if not diagnostics.accepted:
        message = (
            "Positive dark-kernel reduction did not reach its accuracy/passivity "
            f"gate with {diagnostics.reduced_node_count} nodes: max NRMS="
            f"{diagnostics.max_normalized_rms:.6g}, max normalized error="
            f"{diagnostics.max_normalized_error:.6g}, passive="
            f"{diagnostics.passive_on_audit_grid}."
        )
        if policy == "raise":
            raise RuntimeError(message)
        if policy == "warn":
            import warnings

            warnings.warn(message, RuntimeWarning, stacklevel=2)

    nodes, group_weights = _group_nodes_and_weights(groups, L, weights)
    return PositiveDarkKernelReduction(
        bright_depolarization=float(L[bright_index]),
        depolarization_nodes=nodes,
        weights_au_minus3=group_weights,
        source_mode_indices=tuple(tuple(int(index) for index in group) for group in groups),
        source_measure_sha256=modal_measure_sha256(
            L,
            weights,
            bright_index=bright_index,
        ),
        diagnostics=diagnostics,
    )
