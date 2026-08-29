"""Causal Maxwell--Bloch pulse model using the full-QS spheroid kernel.

The spatial kernel in :mod:`qd_mnp_spheroid_green` is frequency dependent and
therefore cannot be inserted into a time-domain Bloch RHS as an instantaneous
number.  This module realizes every retained spatial susceptibility through
the *same* passive Lorentz model already fitted by ``HybridQDPlasmonModel``.

If H(omega) is the fitted bright susceptibility with depolarization L_1, then
the susceptibility for any spheroidal order is exactly related by

    chi_n = H / (1 + (L_n - L_1) H).

The relation follows by eliminating the common material permittivity from
chi_L=(eps_p-eps_m)/(eps_m+L*(eps_p-eps_m)).  It is implemented as a causal
feedback realization, not as independent fits of A, B and K.  Consequently
the bright laser/QD channels remain reciprocal and K_1=B**2/A by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import warnings

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import ArpackNoConvergence, eigs

from qd_mnp_rational_fit import (
    AU_ENERGY_J,
    GaussianPulse,
    HybridQDPlasmonModel,
    HybridSystemParams,
    sampled_positive_frequency_spectral_fraction,
)
from qd_mnp_spheroid_green import (
    QuasistaticInteractionResponse,
    SpheroidGreenInteraction,
)


Policy = Literal["raise", "warn", "ignore"]


def _readonly(value: np.ndarray, *, dtype=None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ModalTransformDiagnostics:
    normalized_rms_by_degree: np.ndarray
    max_relative_error_by_degree: np.ndarray
    minimum_imaginary_part_by_degree: np.ndarray
    K_normalized_rms: float
    K_max_relative_error: float
    max_normalized_rms: float
    max_relative_error: float
    passive_on_audit_grid: bool
    accepted: bool
    audit_grid_points: int

    def __post_init__(self) -> None:
        for name in (
            "normalized_rms_by_degree",
            "max_relative_error_by_degree",
            "minimum_imaginary_part_by_degree",
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name), dtype=float))


@dataclass(frozen=True)
class SpatialConvergenceDiagnostics:
    max_half_order_relative_change: float
    max_tail_block_relative_mass: float
    accepted: bool
    tolerance: float
    audit_grid_points: int
    energy_window_eV: tuple[float, float]
    half_order_relative_change_by_energy: np.ndarray
    tail_block_relative_mass_by_energy: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "half_order_relative_change_by_energy",
            _readonly(self.half_order_relative_change_by_energy, dtype=float),
        )
        object.__setattr__(
            self,
            "tail_block_relative_mass_by_energy",
            _readonly(self.tail_block_relative_mass_by_energy, dtype=float),
        )
        object.__setattr__(
            self,
            "energy_window_eV",
            tuple(float(value) for value in self.energy_window_eV),
        )


@dataclass(frozen=True)
class CoupledStabilityDiagnostics:
    rightmost_poles_au: np.ndarray
    largest_magnitude_poles_au: np.ndarray
    spectral_abscissa_au: float | None
    spectral_abscissa_available: bool
    spectral_abscissa_is_bound: bool
    decay_rate_estimate_au: float
    decay_rate_estimate_is_exact: bool
    spectral_radius_au: float
    tolerance_au: float
    stable: bool
    eigensolver: str
    coherent_state_dimension: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rightmost_poles_au",
            _readonly(self.rightmost_poles_au, dtype=complex),
        )
        object.__setattr__(
            self,
            "largest_magnitude_poles_au",
            _readonly(self.largest_magnitude_poles_au, dtype=complex),
        )


@dataclass(frozen=True)
class FullQSSolveDiagnostics:
    solver_success: bool
    solver_status: int
    solver_message: str
    n_steps: int
    nfev: int
    min_step_au: float
    max_step_au: float
    max_step_limit_au: float
    integration_frequency_ceiling_au: float
    t_final_reached: bool
    state_is_finite: bool
    boundary_envelope_fraction: float
    excited_population_min: float
    excited_population_max: float
    max_bloch_radius: float
    min_density_eigenvalue: float
    pulse_spectral_fraction_in_fit_window: float
    pulse_spectral_leakage: float
    qd_source_spectral_fraction_in_fit_window: float
    qd_source_spectral_leakage: float
    mnp_drive_spectral_fraction_in_fit_window: float
    mnp_drive_spectral_leakage: float
    mnp_dipole_spectral_fraction_in_fit_window: float
    mnp_dipole_spectral_leakage: float
    mnp_field_spectral_fraction_in_fit_window: float
    mnp_field_spectral_leakage: float
    response_tail_ratio: float
    response_tail_tolerance: float
    response_tail_converged: bool
    response_tail_window_fraction: float
    work_nonnegative_within_tolerance: bool
    work_passivity_tolerance_au: float
    spatial_order_max: int
    material_poles_per_spatial_order: int
    spectral_abscissa_au: float | None
    spectral_abscissa_available: bool
    spectral_abscissa_is_bound: bool
    decay_rate_estimate_au: float
    decay_rate_estimate_is_exact: bool
    spectral_radius_au: float
    incident_peak_rabi_frequency_au: float
    observed_peak_rabi_frequency_au: float
    rabi_step_refinement_count: int
    modal_fit_max_normalized_rms: float
    modal_fit_max_relative_error: float


@dataclass(frozen=True)
class FullQSSolveResult:
    t_au: np.ndarray
    y: np.ndarray
    W: np.ndarray
    Q: np.ndarray
    P: np.ndarray
    rho22: np.ndarray
    mu_p_au: np.ndarray
    mu_d_au: np.ndarray
    mu_total_au: np.ndarray
    mu_dot_total_au: np.ndarray
    incident_field_au: np.ndarray
    mnp_field_at_qd_au: np.ndarray
    effective_qd_field_au: np.ndarray
    modal_outputs_au: np.ndarray
    sigma_energy_transfer_cm2: float
    work_from_incident_field_j: float
    fluence_j_cm2: float
    peak_intensity_w_cm2: float
    solve_ivp_result: object
    diagnostics: FullQSSolveDiagnostics

    @property
    def max_bloch_radius(self) -> float:
        return self.diagnostics.max_bloch_radius

    @property
    def min_density_eigenvalue(self) -> float:
        return self.diagnostics.min_density_eigenvalue


class FullQSSpheroidPulseModel:
    """Semiclassical TLS plus causal full-QS prolate-spheroid response."""

    def __init__(
        self,
        bright_model: HybridQDPlasmonModel,
        spheroid_kernel: SpheroidGreenInteraction,
        *,
        fit_quality_policy: Policy = "raise",
        max_modal_normalized_rms: float = 0.03,
        max_modal_relative_error: float = 0.06,
        modal_audit_points: int = 2001,
        spatial_convergence_policy: Policy = "raise",
        spatial_convergence_rtol: float = 1.0e-8,
    ) -> None:
        if fit_quality_policy not in {"raise", "warn", "ignore"}:
            raise ValueError("fit_quality_policy must be 'raise', 'warn' or 'ignore'.")
        if spatial_convergence_policy not in {"raise", "warn", "ignore"}:
            raise ValueError(
                "spatial_convergence_policy must be 'raise', 'warn' or 'ignore'."
            )
        if modal_audit_points < 101:
            raise ValueError("modal_audit_points must be at least 101.")
        for name, value in (
            ("max_modal_normalized_rms", max_modal_normalized_rms),
            ("max_modal_relative_error", max_modal_relative_error),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if not np.isfinite(spatial_convergence_rtol) or not (
            0.0 < spatial_convergence_rtol < 1.0
        ):
            raise ValueError("spatial_convergence_rtol must lie in (0, 1).")

        self.bright_model = bright_model
        self.kernel = spheroid_kernel
        self.params: HybridSystemParams = bright_model.params
        self.orientation = bright_model.orientation
        self.n_spatial_modes = spheroid_kernel.n_max
        self.n_material_modes = bright_model.n_modes
        self.fit = bright_model.fit
        self.fit_window_eV = bright_model.fit_window_eV
        self.C = float(bright_model.C)
        self.alpha_inf = float(self.fit.alpha_inf)

        self._validate_shared_configuration()
        self.spatial_convergence_diagnostics = self._audit_spatial_convergence(
            points=modal_audit_points,
            tolerance=float(spatial_convergence_rtol),
        )
        if not self.spatial_convergence_diagnostics.accepted:
            spatial = self.spatial_convergence_diagnostics
            message = (
                "The retained spheroidal spatial series is not converged on the "
                "material-fit window: max half-order relative change="
                f"{spatial.max_half_order_relative_change:.6g}, max tail-block "
                f"relative mass={spatial.max_tail_block_relative_mass:.6g}, "
                f"tolerance={spatial.tolerance:.6g}, n_max={self.n_spatial_modes}."
            )
            if spatial_convergence_policy == "raise":
                raise RuntimeError(message)
            if spatial_convergence_policy == "warn":
                warnings.warn(message, RuntimeWarning, stacklevel=2)
        self.delta_L = np.asarray(
            self.kernel.depolarization_by_degree
            - self.kernel.depolarization_by_degree[0],
            dtype=float,
        )
        self.feedback_denominator = 1.0 + self.delta_L * self.alpha_inf
        if np.any(~np.isfinite(self.feedback_denominator)) or np.any(
            np.abs(self.feedback_denominator) < 1.0e-10
        ):
            raise ValueError(
                "The modal feedthrough feedback 1+(L_n-L_1)*alpha_inf is singular."
            )

        self.bright_coupling_au_minus3 = float(
            self.kernel.bright_source_coupling_au_minus3
        )
        self.reaction_weights_au_minus3 = np.asarray(
            self.kernel.reaction_weight_by_degree_au_minus3,
            dtype=float,
        )
        expected_bright_weight = self.C * self.bright_coupling_au_minus3**2
        if not np.isclose(
            self.reaction_weights_au_minus3[0],
            expected_bright_weight,
            rtol=5.0e-13,
            atol=0.0,
        ):
            raise RuntimeError("Bright reciprocity identity w_1=C*lambda^2 failed.")

        self.modal_fit_diagnostics = self._audit_modal_transform(
            points=modal_audit_points,
            max_normalized_rms=max_modal_normalized_rms,
            max_relative_error=max_modal_relative_error,
        )
        if not self.modal_fit_diagnostics.accepted:
            message = (
                "The transformed common-material realization does not meet the "
                "full-QS modal accuracy/passivity gate: max NRMS="
                f"{self.modal_fit_diagnostics.max_normalized_rms:.5g}, max relative="
                f"{self.modal_fit_diagnostics.max_relative_error:.5g}, passive="
                f"{self.modal_fit_diagnostics.passive_on_audit_grid}."
            )
            if fit_quality_policy == "raise":
                raise RuntimeError(message)
            if fit_quality_policy == "warn":
                warnings.warn(message, RuntimeWarning, stacklevel=2)

        self.modal_poles_au = self._modal_transfer_poles()
        modal_tolerance = 1.0e-10 * max(
            float(np.max(self.fit.omega_modes_au)),
            1.0e-15,
        )
        if float(np.max(self.modal_poles_au.real)) > modal_tolerance:
            raise RuntimeError(
                "A transformed material mode has an unstable causal pole: "
                f"max Re(lambda)={float(np.max(self.modal_poles_au.real)):.6e} au."
            )
        self.coupled_stability = self.linearized_ground_state_stability()
        if not self.coupled_stability.stable:
            abscissa = self.coupled_stability.spectral_abscissa_au
            abscissa_text = (
                "unavailable" if abscissa is None else f"{abscissa:.6e} au"
            )
            raise RuntimeError(
                "The full-QS coupled ground state failed its stability audit: "
                f"spectral abscissa={abscissa_text}, certificate="
                f"{self.coupled_stability.eigensolver}."
            )

    def _validate_shared_configuration(self) -> None:
        geometry = self.kernel.geometry
        p = self.params
        checks = {
            "a_au": (geometry.a_au, p.a_au),
            "c_au": (geometry.c_au, p.c_au),
            "R_au": (geometry.R_au, p.R_au),
            "eps_m": (geometry.eps_m, p.eps_m),
            "qd_radius_au": (geometry.qd_radius_au, p.qd_radius_au),
        }
        mismatched = [
            name
            for name, (left, right) in checks.items()
            if not np.isclose(left, right, rtol=2.0e-14, atol=0.0)
        ]
        if mismatched:
            raise ValueError(
                "The bright model and spheroid kernel use different parameters: "
                + ", ".join(mismatched)
            )
        if geometry.orientation != self.orientation:
            raise ValueError("The bright model and spheroid kernel orientations differ.")
        if not np.isclose(
            self.kernel.depolarization_by_degree[0],
            self.bright_model.L,
            rtol=1.0e-12,
            atol=1.0e-14,
        ):
            raise ValueError("The bright depolarization factor differs between backends.")

    def _audit_modal_transform(
        self,
        *,
        points: int,
        max_normalized_rms: float,
        max_relative_error: float,
    ) -> ModalTransformDiagnostics:
        energies = np.linspace(self.fit_window_eV[0], self.fit_window_eV[1], points)
        H = np.asarray(self.bright_model.alpha_from_fit(energies), dtype=complex)
        fitted = H[None, :] / (1.0 + self.delta_L[:, None] * H[None, :])
        epsilon = self.params.material.epsilon_at(energies)
        delta_epsilon = epsilon - self.params.eps_m
        target = delta_epsilon[None, :] / (
            self.params.eps_m
            + self.kernel.depolarization_by_degree[:, None] * delta_epsilon[None, :]
        )
        error = fitted - target
        target_rms = np.sqrt(np.mean(np.abs(target) ** 2, axis=1))
        normalized_rms = np.sqrt(np.mean(np.abs(error) ** 2, axis=1)) / np.maximum(
            target_rms,
            np.finfo(float).tiny,
        )
        target_scale = np.max(np.abs(target), axis=1)
        relative = np.max(
            np.abs(error)
            / np.maximum(
                np.abs(target),
                1.0e-15 * np.maximum(target_scale[:, None], np.finfo(float).tiny),
            ),
            axis=1,
        )
        minimum_imaginary = np.min(fitted.imag, axis=1)

        K_fit = np.sum(self.reaction_weights_au_minus3[:, None] * fitted, axis=0)
        K_target = np.sum(self.reaction_weights_au_minus3[:, None] * target, axis=0)
        K_error = K_fit - K_target
        K_nrms = float(
            np.sqrt(np.mean(np.abs(K_error) ** 2))
            / max(np.sqrt(np.mean(np.abs(K_target) ** 2)), np.finfo(float).tiny)
        )
        K_scale = max(float(np.max(np.abs(K_target))), np.finfo(float).tiny)
        K_max_relative = float(
            np.max(
                np.abs(K_error)
                / np.maximum(np.abs(K_target), 1.0e-15 * K_scale)
            )
        )
        passive = bool(np.min(minimum_imaginary) >= -1.0e-13)
        accepted = bool(
            float(np.max(normalized_rms)) <= max_normalized_rms
            and float(np.max(relative)) <= max_relative_error
            and K_nrms <= max_normalized_rms
            and K_max_relative <= max_relative_error
            and passive
        )
        return ModalTransformDiagnostics(
            normalized_rms_by_degree=normalized_rms,
            max_relative_error_by_degree=relative,
            minimum_imaginary_part_by_degree=minimum_imaginary,
            K_normalized_rms=K_nrms,
            K_max_relative_error=K_max_relative,
            max_normalized_rms=float(np.max(normalized_rms)),
            max_relative_error=float(np.max(relative)),
            passive_on_audit_grid=passive,
            accepted=accepted,
            audit_grid_points=points,
        )

    def _single_modal_state_matrix(self, spatial_index: int) -> np.ndarray:
        count = self.n_material_modes
        matrix = np.zeros((2 * count, 2 * count), dtype=float)
        feedback = self.delta_L[spatial_index] / self.feedback_denominator[spatial_index]
        for material_index, (strength, frequency, damping) in enumerate(
            zip(
                self.fit.strengths_au2,
                self.fit.omega_modes_au,
                self.fit.gamma_modes_au,
            )
        ):
            q_index = 2 * material_index
            v_index = q_index + 1
            matrix[q_index, v_index] = 1.0
            matrix[v_index, q_index] = -frequency**2
            matrix[v_index, v_index] = -damping
            matrix[v_index, 0 : 2 * count : 2] -= strength * feedback
        return matrix

    def _modal_transfer_poles(self) -> np.ndarray:
        poles = np.empty(
            (self.n_spatial_modes, 2 * self.n_material_modes),
            dtype=complex,
        )
        for index in range(self.n_spatial_modes):
            poles[index] = np.linalg.eigvals(self._single_modal_state_matrix(index))
        return _readonly(poles, dtype=complex)

    @property
    def mode_state_count(self) -> int:
        return 2 * self.n_spatial_modes * self.n_material_modes

    @property
    def state_size(self) -> int:
        # Spatial/material oscillator states, W/Q/P and an external-work accumulator.
        return self.mode_state_count + 4

    @property
    def W_index(self) -> int:
        return self.mode_state_count

    @property
    def Q_index(self) -> int:
        return self.mode_state_count + 1

    @property
    def P_index(self) -> int:
        return self.mode_state_count + 2

    @property
    def work_index(self) -> int:
        return self.mode_state_count + 3

    def initial_state(self) -> np.ndarray:
        state = np.zeros(self.state_size, dtype=float)
        state[self.W_index] = -1.0
        return state

    def _validated_state(self, state: np.ndarray) -> np.ndarray:
        raw = np.asarray(state)
        if raw.shape != (self.state_size,):
            raise ValueError(
                f"state must have shape ({self.state_size},), got {raw.shape}."
            )
        if not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
            raise TypeError("state must be a real numeric one-dimensional array.")
        values = np.asarray(raw, dtype=float)
        if np.any(~np.isfinite(values)):
            raise ValueError("state must contain only finite values.")
        return values

    def _audit_spatial_convergence(
        self,
        *,
        points: int,
        tolerance: float,
    ) -> SpatialConvergenceDiagnostics:
        energies = np.linspace(self.fit_window_eV[0], self.fit_window_eV[1], points)
        direct_response = self.kernel.response_from_material(
            self.params.material,
            energies,
        )
        half_order = np.asarray(
            direct_response.relative_half_order_change(),
            dtype=float,
        )
        tail_block = np.asarray(
            direct_response.relative_tail_block(),
            dtype=float,
        )
        max_half_order = float(np.max(half_order))
        max_tail_block = float(np.max(tail_block))
        return SpatialConvergenceDiagnostics(
            max_half_order_relative_change=max_half_order,
            max_tail_block_relative_mass=max_tail_block,
            accepted=bool(
                max_half_order <= tolerance and max_tail_block <= tolerance
            ),
            tolerance=float(tolerance),
            audit_grid_points=int(points),
            energy_window_eV=(
                float(self.fit_window_eV[0]),
                float(self.fit_window_eV[1]),
            ),
            half_order_relative_change_by_energy=half_order,
            tail_block_relative_mass_by_energy=tail_block,
        )

    @staticmethod
    def _validated_time_and_pulse(
        t_au: float,
        pulse: GaussianPulse,
    ) -> tuple[float, GaussianPulse]:
        time = np.asarray(t_au)
        if time.ndim != 0 or np.iscomplexobj(time):
            raise TypeError("t_au must be a finite real scalar.")
        try:
            time_value = float(time)
        except (TypeError, ValueError) as exc:
            raise TypeError("t_au must be a finite real scalar.") from exc
        if not np.isfinite(time_value):
            raise ValueError("t_au must be a finite real scalar.")
        if not isinstance(pulse, GaussianPulse):
            raise TypeError("pulse must be a GaussianPulse instance.")
        return time_value, pulse

    def _unpack_modal_states(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        modal = state[: self.mode_state_count].reshape(
            self.n_spatial_modes,
            self.n_material_modes,
            2,
        )
        return modal[:, :, 0], modal[:, :, 1]

    def _solution_observables(
        self,
        t_au: np.ndarray,
        y: np.ndarray,
        pulse: GaussianPulse,
    ) -> dict[str, np.ndarray]:
        """Vectorized reconstruction used by Rabi, spectrum and tail audits."""

        point_count = t_au.size
        modal = y[: self.mode_state_count].reshape(
            self.n_spatial_modes,
            self.n_material_modes,
            2,
            point_count,
        )
        q = modal[:, :, 0, :]
        velocity = modal[:, :, 1, :]
        W = y[self.W_index]
        Q_bloch = y[self.Q_index]
        P_bloch = y[self.P_index]
        incident = np.asarray(pulse.field(t_au), dtype=float)
        incident_dot = np.asarray(pulse.field_dot(t_au), dtype=float)
        local = self.params.qd_local_field_factor
        mu_d = local * self.params.d_au * P_bloch

        external_inputs = np.broadcast_to(
            mu_d,
            (self.n_spatial_modes, point_count),
        ).copy()
        external_inputs[0] = incident + self.bright_coupling_au_minus3 * mu_d
        q_sum = np.sum(q, axis=1)
        velocity_sum = np.sum(velocity, axis=1)
        internal_drives = (
            external_inputs - self.delta_L[:, None] * q_sum
        ) / self.feedback_denominator[:, None]
        modal_outputs = self.alpha_inf * internal_drives + q_sum

        _, field_weights = self._input_and_field_weights()
        mu_p = self.C * modal_outputs[0]
        mnp_field = np.sum(field_weights[:, None] * modal_outputs, axis=0)
        effective_field = local * (incident + mnp_field)
        rabi = 2.0 * self.params.d_au * effective_field

        dP = self.params.omega0_au * Q_bloch - self.params.Gamma_au * P_bloch
        dmu_d = local * self.params.d_au * dP
        external_input_dots = np.broadcast_to(
            dmu_d,
            (self.n_spatial_modes, point_count),
        ).copy()
        external_input_dots[0] = (
            incident_dot + self.bright_coupling_au_minus3 * dmu_d
        )
        internal_drive_dots = (
            external_input_dots - self.delta_L[:, None] * velocity_sum
        ) / self.feedback_denominator[:, None]
        modal_output_dots = self.alpha_inf * internal_drive_dots + velocity_sum
        dmu_p = self.C * modal_output_dots[0]

        return {
            "q": q,
            "velocity": velocity,
            "W": W,
            "Q": Q_bloch,
            "P": P_bloch,
            "incident": incident,
            "qd_source": mu_d,
            "mnp_drive": external_inputs[0],
            "internal_drives": internal_drives,
            "modal_outputs": modal_outputs,
            "mu_d": mu_d,
            "mu_p": mu_p,
            "mu_total": mu_d + mu_p,
            "mu_dot_total": dmu_d + dmu_p,
            "mnp_field": mnp_field,
            "effective_field": effective_field,
            "rabi": rabi,
        }

    def _windowed_response_tail_ratio(
        self,
        t_au: np.ndarray,
        observables: dict[str, np.ndarray],
        *,
        window_fraction: float,
        relative_channel_floor: float = 1.0e-12,
    ) -> float:
        """Audit the final-time tail without cancellation between channels.

        Dipoles, reaction-field contributions and TLS coherence are normalized
        within separate physical-unit families.  Each modal output and each
        material ``q`` state is weighted by its contribution to the field at
        the QD; velocity is additionally divided by its oscillator frequency.
        Channels whose peak is below ``relative_channel_floor`` times their
        family's largest peak are omitted, so a geometrically negligible high
        spheroidal order cannot dictate the post-pulse window.
        """

        times = np.asarray(t_au, dtype=float)
        window_start = float(
            times[-1] - window_fraction * (times[-1] - times[0])
        )
        start_index = int(np.searchsorted(times, window_start, side="left"))

        def family_ratio(signals: list[np.ndarray]) -> float:
            finite_signals = [np.asarray(signal, dtype=float) for signal in signals]
            peaks = np.asarray(
                [float(np.max(np.abs(signal))) for signal in finite_signals],
                dtype=float,
            )
            family_peak = float(np.max(peaks, initial=0.0))
            if family_peak == 0.0:
                return 0.0
            ratios: list[float] = []
            for signal, peak in zip(finite_signals, peaks):
                if peak < relative_channel_floor * family_peak:
                    continue
                if times[start_index] == window_start:
                    tail_times = times[start_index:]
                    tail_signal = signal[start_index:]
                else:
                    left = start_index - 1
                    interpolation_weight = (
                        (window_start - times[left])
                        / (times[start_index] - times[left])
                    )
                    signal_at_start = signal[left] + interpolation_weight * (
                        signal[start_index] - signal[left]
                    )
                    tail_times = np.concatenate(([window_start], times[start_index:]))
                    tail_signal = np.concatenate(
                        ([signal_at_start], signal[start_index:])
                    )
                duration = float(tail_times[-1] - tail_times[0])
                mean_square = float(
                    np.trapezoid(tail_signal**2, tail_times) / duration
                )
                ratios.append(float(np.sqrt(max(mean_square, 0.0)) / peak))
            return max(ratios, default=0.0)

        dipole_ratio = family_ratio(
            [
                observables["mu_total"],
                observables["mu_p"],
                observables["mu_d"],
            ]
        )
        _, field_weights = self._input_and_field_weights()
        weighted_outputs = (
            field_weights[:, None] * observables["modal_outputs"]
        )
        weighted_q = field_weights[:, None, None] * observables["q"]
        weighted_velocity = (
            field_weights[:, None, None]
            * observables["velocity"]
            / self.fit.omega_modes_au[None, :, None]
        )
        field_signals = [observables["mnp_field"]]
        field_signals.extend(weighted_outputs)
        field_signals.extend(weighted_q.reshape(-1, times.size))
        field_signals.extend(weighted_velocity.reshape(-1, times.size))
        field_ratio = family_ratio(field_signals)
        coherence_ratio = family_ratio(
            [np.hypot(observables["Q"], observables["P"])]
        )
        return float(max(dipole_ratio, field_ratio, coherence_ratio))

    def _input_and_field_weights(self) -> tuple[np.ndarray, np.ndarray]:
        local = self.params.qd_local_field_factor
        input_coefficients = np.full(
            self.n_spatial_modes,
            local * self.params.d_au,
            dtype=float,
        )
        input_coefficients[0] *= self.bright_coupling_au_minus3
        field_weights = self.reaction_weights_au_minus3.copy()
        field_weights[0] = self.C * self.bright_coupling_au_minus3
        return input_coefficients, field_weights

    def linearized_ground_state_matrix(self) -> csr_matrix:
        """Sparse field-free coherent Jacobian (population/work modes excluded)."""

        oscillator_count = self.mode_state_count
        Q_index = oscillator_count
        P_index = oscillator_count + 1
        size = oscillator_count + 2
        matrix = lil_matrix((size, size), dtype=float)
        input_coefficients, field_weights = self._input_and_field_weights()

        for spatial_index in range(self.n_spatial_modes):
            denominator = self.feedback_denominator[spatial_index]
            feedback = self.delta_L[spatial_index] / denominator
            spatial_offset = 2 * self.n_material_modes * spatial_index
            q_indices = [
                spatial_offset + 2 * material_index
                for material_index in range(self.n_material_modes)
            ]
            for material_index, (strength, frequency, damping) in enumerate(
                zip(
                    self.fit.strengths_au2,
                    self.fit.omega_modes_au,
                    self.fit.gamma_modes_au,
                )
            ):
                q_index = spatial_offset + 2 * material_index
                v_index = q_index + 1
                matrix[q_index, v_index] = 1.0
                matrix[v_index, q_index] = -frequency**2
                matrix[v_index, v_index] = -damping
                for coupled_q in q_indices:
                    matrix[v_index, coupled_q] -= strength * feedback
                matrix[v_index, P_index] += (
                    strength * input_coefficients[spatial_index] / denominator
                )

            q_to_Q = (
                2.0
                * self.params.d_au
                * self.params.qd_local_field_factor
                * field_weights[spatial_index]
                / denominator
            )
            for q_index in q_indices:
                matrix[Q_index, q_index] += q_to_Q

        feedthrough_P = float(
            np.sum(
                field_weights
                * self.alpha_inf
                * input_coefficients
                / self.feedback_denominator
            )
        )
        matrix[Q_index, Q_index] = -self.params.Gamma_au
        matrix[Q_index, P_index] = (
            -self.params.omega0_au
            + 2.0
            * self.params.d_au
            * self.params.qd_local_field_factor
            * feedthrough_P
        )
        matrix[P_index, Q_index] = self.params.omega0_au
        matrix[P_index, P_index] = -self.params.Gamma_au
        return matrix.tocsr()

    def linearized_ground_state_stability(self) -> CoupledStabilityDiagnostics:
        """Audit coupled poles or a separately labelled passivity certificate.

        For large systems the passivity/no-soft-mode argument certifies
        stability but does *not* bound the decay rate of the coupled system.
        In that branch ``spectral_abscissa_au`` is therefore unavailable and
        the uncoupled subsystem rate is exported only as a heuristic proxy.
        """

        matrix = self.linearized_ground_state_matrix()
        size = matrix.shape[0]
        if size <= 320:
            rightmost_poles = np.linalg.eigvals(matrix.toarray())
            largest_magnitude_poles = rightmost_poles
            eigensolver = "dense"
            spectral_abscissa_available = True
            spectral_abscissa_is_bound = False
        else:
            # Real matrices produce conjugate pairs, so two Ritz values are
            # sufficient for both extrema and avoid a costly clustered solve
            # at the production spatial order N=80.
            requested = min(2, size - 2)
            krylov_dimension = min(size, max(40, 12 * requested + 1))
            matrix_scale = max(
                float(np.max(np.asarray(np.abs(matrix).sum(axis=1)).ravel())),
                np.finfo(float).tiny,
            )

            def certified_sparse_poles(which: str) -> np.ndarray:
                try:
                    values, vectors = eigs(
                        matrix,
                        k=requested,
                        which=which,
                        return_eigenvectors=True,
                        tol=1.0e-9,
                        ncv=krylov_dimension,
                        maxiter=max(10_000, 40 * size),
                    )
                except ArpackNoConvergence as exc:
                    raise RuntimeError(
                        "Sparse stability eigensolver did not fully converge for "
                        f"which={which!r}; a partial ARPACK spectrum cannot certify "
                        "coupled stability."
                    ) from exc
                values = np.asarray(values, dtype=complex)
                vectors = np.asarray(vectors, dtype=complex)
                if (
                    values.shape != (requested,)
                    or vectors.shape != (size, requested)
                    or np.any(~np.isfinite(values))
                    or np.any(~np.isfinite(vectors))
                ):
                    raise RuntimeError(
                        "Sparse stability eigensolver returned an invalid eigensystem."
                    )
                residual = matrix @ vectors - vectors * values[None, :]
                vector_norm = np.linalg.norm(vectors, axis=0)
                relative_residual = np.linalg.norm(residual, axis=0) / np.maximum(
                    (matrix_scale + np.abs(values)) * vector_norm,
                    np.finfo(float).tiny,
                )
                if float(np.max(relative_residual)) > 1.0e-7:
                    raise RuntimeError(
                        "Sparse stability eigensystem failed its residual "
                        f"certification for which={which!r}: max relative residual="
                        f"{float(np.max(relative_residual)):.6e}."
                    )
                return values

            largest_magnitude_poles = certified_sparse_poles("LM")
            if size <= 512:
                rightmost_poles = certified_sparse_poles("LR")
                eigensolver = "sparse_LR_LM_certified"
                spectral_abscissa_available = True
                spectral_abscissa_is_bound = False
            else:
                # At N=80 the passive Lorentz blocks form a large, nearly
                # degenerate cluster for which ARPACK's algebraic ``LR`` mode
                # is prohibitively slow.  Stability can instead be certified
                # from the passive reciprocal realization: all dissipative
                # subsystems must be strictly stable and their static coupled
                # stiffness must retain a positive (no-soft-mode) margin.
                H_static = float(
                    self.alpha_inf
                    + np.sum(
                        self.fit.strengths_au2 / self.fit.omega_modes_au**2
                    )
                )
                modal_static = H_static / (1.0 + self.delta_L * H_static)
                K_static = float(
                    np.dot(self.reaction_weights_au_minus3, modal_static)
                )
                static_stiffness = float(
                    self.params.omega0_au
                    - 2.0
                    * self.params.d_au**2
                    * self.params.qd_local_field_factor**2
                    * K_static
                )
                # The exact damped TLS static term is
                # omega0 + Gamma**2/omega0.  Omitting its positive second term
                # makes this no-soft-mode test deliberately conservative.
                static_tolerance = 1.0e-12 * max(
                    self.params.omega0_au,
                    1.0e-15,
                )
                passive_subsystems = bool(
                    self.params.Gamma_au > 0.0
                    and np.all(self.fit.strengths_au2 > 0.0)
                    and np.all(self.fit.gamma_modes_au > 0.0)
                    and float(np.max(self.modal_poles_au.real)) < 0.0
                )
                if (
                    not passive_subsystems
                    or not np.isfinite(static_stiffness)
                    or static_stiffness <= static_tolerance
                ):
                    raise RuntimeError(
                        "The large sparse coupled system failed its passive "
                        "interconnection stability certificate: passive="
                        f"{passive_subsystems}, static stiffness="
                        f"{static_stiffness:.6e} au, required above "
                        f"{static_tolerance:.6e} au."
                    )
                flat_modal_poles = self.modal_poles_au.reshape(-1)
                modal_indices = np.argsort(flat_modal_poles.real)[-2:]
                tls_reference_poles = np.asarray(
                    [
                        complex(-self.params.Gamma_au, self.params.omega0_au),
                        complex(-self.params.Gamma_au, -self.params.omega0_au),
                    ]
                )
                # These uncoupled poles provide only a practical initial
                # decay-rate estimate.  Passive coupling can create a much
                # slower pole near a soft-mode threshold, so they are not a
                # bound on the coupled spectral abscissa.
                subsystem_proxy_poles = np.concatenate(
                    (tls_reference_poles, flat_modal_poles[modal_indices])
                )
                rightmost_poles = np.empty(0, dtype=complex)
                eigensolver = "passive_no_soft_mode_certificate+LM_certified"
                spectral_abscissa_available = False
                spectral_abscissa_is_bound = False
        rightmost_poles = np.asarray(rightmost_poles, dtype=complex)
        largest_magnitude_poles = np.asarray(
            largest_magnitude_poles,
            dtype=complex,
        )
        if spectral_abscissa_available:
            spectral_abscissa: float | None = float(np.max(rightmost_poles.real))
            decay_rate_estimate = max(
                -spectral_abscissa,
                np.finfo(float).tiny,
            )
            decay_rate_estimate_is_exact = True
        else:
            spectral_abscissa = None
            decay_rate_estimate = float(-np.max(subsystem_proxy_poles.real))
            decay_rate_estimate_is_exact = False
        if not np.isfinite(decay_rate_estimate) or decay_rate_estimate <= 0.0:
            raise RuntimeError("No finite positive decay-rate proxy is available.")
        spectral_radius = float(np.max(np.abs(largest_magnitude_poles)))
        scale = max(
            self.params.omega0_au,
            float(np.max(np.abs(self.modal_poles_au))),
            spectral_radius,
            1.0e-15,
        )
        tolerance = 1.0e-10 * scale
        stable = True if spectral_abscissa is None else bool(
            spectral_abscissa <= tolerance
        )
        return CoupledStabilityDiagnostics(
            rightmost_poles_au=rightmost_poles,
            largest_magnitude_poles_au=largest_magnitude_poles,
            spectral_abscissa_au=spectral_abscissa,
            spectral_abscissa_available=spectral_abscissa_available,
            spectral_abscissa_is_bound=spectral_abscissa_is_bound,
            decay_rate_estimate_au=decay_rate_estimate,
            decay_rate_estimate_is_exact=decay_rate_estimate_is_exact,
            spectral_radius_au=spectral_radius,
            tolerance_au=tolerance,
            stable=stable,
            eigensolver=eigensolver,
            coherent_state_dimension=size,
        )

    def modal_susceptibility_from_fit(
        self,
        energies_eV: float | np.ndarray,
    ) -> np.ndarray:
        H = np.asarray(self.bright_model.alpha_from_fit(energies_eV), dtype=complex)
        return H[None, ...] / (
            1.0
            + self.delta_L.reshape((self.n_spatial_modes,) + (1,) * H.ndim)
            * H[None, ...]
        )

    def frequency_response_from_fit(
        self,
        energies_eV: float | np.ndarray,
    ) -> QuasistaticInteractionResponse:
        modal = self.modal_susceptibility_from_fit(energies_eV)
        frequency_shape = modal.shape[1:]
        A = self.C * modal[0]
        B = self.C * self.bright_coupling_au_minus3 * modal[0]
        K_by_degree = (
            self.reaction_weights_au_minus3.reshape(
                (self.n_spatial_modes,) + (1,) * len(frequency_shape)
            )
            * modal
        )
        return QuasistaticInteractionResponse(
            model="spheroid_n1" if self.n_spatial_modes == 1 else "spheroid_full",
            orientation=self.orientation,
            A_au3=A,
            B=B,
            K_au_minus3=np.sum(K_by_degree, axis=0),
            degrees=self.kernel.degrees,
            K_by_degree_au_minus3=K_by_degree,
            modal_susceptibility_by_degree=modal,
            reaction_weight_by_degree_au_minus3=self.reaction_weights_au_minus3,
            depolarization_by_degree=self.kernel.depolarization_by_degree,
            geometric_factor_by_degree=self.kernel.geometric_factor_by_degree,
            eps_m=self.params.eps_m,
            log_abs_geometric_factor_by_degree=(
                self.kernel.log_abs_geometric_factor_by_degree
            ),
        )

    def _evaluate_state(
        self,
        t_au: float,
        state: np.ndarray,
        pulse: GaussianPulse,
    ) -> dict[str, object]:
        time, validated_pulse = self._validated_time_and_pulse(t_au, pulse)
        return self._evaluate_state_unchecked(
            time,
            self._validated_state(state),
            validated_pulse,
        )

    def _evaluate_state_unchecked(
        self,
        t_au: float,
        state: np.ndarray,
        pulse: GaussianPulse,
    ) -> dict[str, object]:
        q, velocity = self._unpack_modal_states(state)
        W, Q_bloch, P_bloch = state[self.W_index : self.P_index + 1]
        incident = float(pulse.field(t_au))
        incident_dot = float(pulse.field_dot(t_au))
        local = self.params.qd_local_field_factor
        mu_d = local * self.params.d_au * P_bloch

        external_inputs = np.full(self.n_spatial_modes, mu_d, dtype=float)
        external_inputs[0] = incident + self.bright_coupling_au_minus3 * mu_d
        q_sum = np.sum(q, axis=1)
        velocity_sum = np.sum(velocity, axis=1)
        internal_drives = (
            external_inputs - self.delta_L * q_sum
        ) / self.feedback_denominator
        modal_outputs = self.alpha_inf * internal_drives + q_sum

        mu_p = self.C * modal_outputs[0]
        mnp_field = (
            self.bright_coupling_au_minus3 * mu_p
            + float(
                np.dot(
                    self.reaction_weights_au_minus3[1:],
                    modal_outputs[1:],
                )
            )
        )
        effective_field = local * (incident + mnp_field)
        rabi = 2.0 * self.params.d_au * effective_field
        dW = rabi * Q_bloch - self.params.gamma_au * (W + 1.0)
        dQ = (
            -self.params.omega0_au * P_bloch
            - rabi * W
            - self.params.Gamma_au * Q_bloch
        )
        dP = self.params.omega0_au * Q_bloch - self.params.Gamma_au * P_bloch

        dmu_d = local * self.params.d_au * dP
        external_input_dots = np.full(self.n_spatial_modes, dmu_d, dtype=float)
        external_input_dots[0] = (
            incident_dot + self.bright_coupling_au_minus3 * dmu_d
        )
        internal_drive_dots = (
            external_input_dots - self.delta_L * velocity_sum
        ) / self.feedback_denominator
        modal_output_dots = self.alpha_inf * internal_drive_dots + velocity_sum
        dmu_p = self.C * modal_output_dots[0]
        dmu_total = dmu_p + dmu_d

        return {
            "q": q,
            "velocity": velocity,
            "W": float(W),
            "Q": float(Q_bloch),
            "P": float(P_bloch),
            "incident": incident,
            "internal_drives": internal_drives,
            "modal_outputs": modal_outputs,
            "mu_d": float(mu_d),
            "mu_p": float(mu_p),
            "mnp_field": float(mnp_field),
            "effective_field": float(effective_field),
            "dW": float(dW),
            "dQ": float(dQ),
            "dP": float(dP),
            "dmu_total": float(dmu_total),
            "work_dot": float(incident * dmu_total),
        }

    def _rhs_unchecked(
        self,
        t_au: float,
        state: np.ndarray,
        pulse: GaussianPulse,
    ) -> np.ndarray:
        values = self._evaluate_state_unchecked(t_au, state, pulse)
        q = values["q"]
        velocity = values["velocity"]
        internal_drives = values["internal_drives"]
        derivative = np.zeros(self.state_size, dtype=float)
        modal_derivative = derivative[: self.mode_state_count].reshape(
            self.n_spatial_modes,
            self.n_material_modes,
            2,
        )
        modal_derivative[:, :, 0] = velocity
        modal_derivative[:, :, 1] = (
            self.fit.strengths_au2[None, :] * internal_drives[:, None]
            - self.fit.gamma_modes_au[None, :] * velocity
            - self.fit.omega_modes_au[None, :] ** 2 * q
        )
        derivative[self.W_index] = values["dW"]
        derivative[self.Q_index] = values["dQ"]
        derivative[self.P_index] = values["dP"]
        derivative[self.work_index] = values["work_dot"]
        return derivative

    def rhs(self, t_au: float, state: np.ndarray, pulse: GaussianPulse) -> np.ndarray:
        """Return a finite, real derivative for one correctly sized state."""

        time, validated_pulse = self._validated_time_and_pulse(t_au, pulse)
        return self._rhs_unchecked(
            time,
            self._validated_state(state),
            validated_pulse,
        )

    def recommended_post_pulse_time_au(self, *, decay_times: float = 10.0) -> float:
        """Return an initial post-pulse window estimate.

        For a large Jacobian the coupled spectral abscissa is intentionally
        not claimed.  The subsystem rate used here is then only a heuristic;
        callers must retain the component-wise tail audit and extend the
        window when necessary.
        """

        if not np.isfinite(decay_times) or decay_times <= 0.0:
            raise ValueError("decay_times must be finite and positive.")
        coherent_rate = self.coupled_stability.decay_rate_estimate_au
        modal_rate = -float(np.max(self.modal_poles_au.real))
        rates = [coherent_rate, modal_rate, self.params.Gamma_au]
        if any(not np.isfinite(rate) or rate <= 0.0 for rate in rates):
            raise ValueError("No finite decaying full-QS coherent tail can be inferred.")
        return float(decay_times / min(rates))

    def default_time_span(
        self,
        pulse: GaussianPulse,
        *,
        n_sigma: float = 8.0,
        decay_times: float = 10.0,
    ) -> tuple[float, float]:
        if not np.isfinite(n_sigma) or n_sigma <= 0.0:
            raise ValueError("n_sigma must be finite and positive.")
        return (
            -n_sigma * pulse.sigma_t_au,
            max(
                n_sigma * pulse.sigma_t_au,
                self.recommended_post_pulse_time_au(decay_times=decay_times),
            ),
        )

    def solve(
        self,
        pulse: GaussianPulse,
        *,
        t_span_au: tuple[float, float] | None = None,
        method: Literal["DOP853", "RK45", "Radau", "BDF", "LSODA"] = "DOP853",
        rtol: float = 1.0e-8,
        atol: float = 1.0e-10,
        max_step_au: float | None = None,
        points_per_fastest_cycle: float = 20.0,
        spectral_window_policy: Policy = "raise",
        max_spectral_leakage: float = 1.0e-3,
        positivity_policy: Policy = "raise",
        positivity_tolerance: float = 1.0e-7,
        work_passivity_policy: Policy = "warn",
        response_tail_policy: Policy = "raise",
        response_tail_tolerance: float = 1.0e-4,
        response_tail_window_fraction: float = 0.05,
    ) -> FullQSSolveResult:
        if not isinstance(pulse, GaussianPulse):
            raise TypeError("pulse must be a GaussianPulse instance.")
        supported_methods = {"DOP853", "RK45", "Radau", "BDF", "LSODA"}
        if method not in supported_methods:
            raise ValueError(
                "method must be one of " + ", ".join(sorted(supported_methods)) + "."
            )
        for policy_name, policy in (
            ("spectral_window_policy", spectral_window_policy),
            ("positivity_policy", positivity_policy),
            ("work_passivity_policy", work_passivity_policy),
            ("response_tail_policy", response_tail_policy),
        ):
            if policy not in {"raise", "warn", "ignore"}:
                raise ValueError(f"{policy_name} must be 'raise', 'warn' or 'ignore'.")
        for tolerance_name, tolerance_value in (("rtol", rtol), ("atol", atol)):
            if not np.isscalar(tolerance_value) or not np.isfinite(
                tolerance_value
            ) or tolerance_value <= 0.0:
                raise ValueError(f"{tolerance_name} must be a finite positive scalar.")
        if not np.isfinite(points_per_fastest_cycle) or points_per_fastest_cycle < 8.0:
            raise ValueError("points_per_fastest_cycle must be finite and at least 8.")
        if not np.isfinite(max_spectral_leakage) or not 0.0 <= max_spectral_leakage < 1.0:
            raise ValueError("max_spectral_leakage must lie in [0, 1).")
        if not np.isfinite(positivity_tolerance) or positivity_tolerance < 0.0:
            raise ValueError("positivity_tolerance must be finite and non-negative.")
        if (
            not np.isfinite(response_tail_tolerance)
            or not 0.0 < response_tail_tolerance < 1.0
        ):
            raise ValueError("response_tail_tolerance must lie in (0, 1).")
        if (
            not np.isfinite(response_tail_window_fraction)
            or not 0.0 < response_tail_window_fraction <= 1.0
        ):
            raise ValueError("response_tail_window_fraction must lie in (0, 1].")
        requested_max_step = max_step_au
        if requested_max_step is not None and (
            not np.isscalar(requested_max_step)
            or not np.isfinite(requested_max_step)
            or requested_max_step <= 0.0
        ):
            raise ValueError("max_step_au must be a finite positive scalar.")

        leakage = float(pulse.spectral_leakage_fraction(self.fit_window_eV))
        if leakage > max_spectral_leakage:
            message = (
                "Pulse spectrum is not covered by the validated modal fit window: "
                f"leakage={leakage:.6g}, limit={max_spectral_leakage:.6g}."
            )
            if spectral_window_policy == "raise":
                raise ValueError(message)
            if spectral_window_policy == "warn":
                warnings.warn(message, RuntimeWarning, stacklevel=2)

        if t_span_au is None:
            t_span_au = self.default_time_span(pulse)
        span = np.asarray(t_span_au)
        if (
            span.shape != (2,)
            or np.iscomplexobj(span)
            or not np.issubdtype(span.dtype, np.number)
        ):
            raise ValueError("t_span_au must contain finite increasing endpoints.")
        span = np.asarray(span, dtype=float)
        if np.any(~np.isfinite(span)) or span[1] <= span[0]:
            raise ValueError("t_span_au must contain finite increasing endpoints.")
        if not span[0] < 0.0 < span[1]:
            raise ValueError(
                "t_span_au must straddle t=0, the center of the Gaussian pulse."
            )
        t_span_au = (float(span[0]), float(span[1]))
        boundary_envelope = float(
            np.max(pulse.envelope(np.asarray(t_span_au, dtype=float)))
        )
        if boundary_envelope > 1.0e-6:
            raise ValueError(
                "t_span_au truncates the incident pulse: the Gaussian envelope "
                f"is {boundary_envelope:.6g} of its peak at a boundary, above 1e-6."
            )

        incident_peak_rabi = float(
            2.0
            * abs(self.params.d_au)
            * self.params.qd_local_field_factor
            * abs(pulse.E0_au)
        )
        frequency_ceiling = max(
            pulse.omegaL_au,
            self.params.omega0_au,
            float(np.max(np.abs(self.modal_poles_au))),
            self.coupled_stability.spectral_radius_au,
            incident_peak_rabi,
        )
        rabi_step_refinement_count = 0
        max_rabi_refinements = 3
        # A small headroom prevents repeated solves merely because the more
        # resolved trajectory raises the sampled peak by roundoff/a few percent.
        rabi_refinement_safety_factor = 1.10
        while True:
            step_limit = float(
                2.0 * np.pi / (points_per_fastest_cycle * frequency_ceiling)
            )
            effective_max_step = (
                step_limit
                if requested_max_step is None
                else min(float(requested_max_step), step_limit)
            )
            solution = solve_ivp(
                lambda time, state: self._rhs_unchecked(time, state, pulse),
                t_span=t_span_au,
                y0=self.initial_state(),
                method=method,
                rtol=float(rtol),
                atol=float(atol),
                max_step=effective_max_step,
            )
            if not solution.success:
                raise RuntimeError(f"Full-QS solve_ivp failed: {solution.message}")
            if not (
                solution.t.size >= 2
                and np.all(np.isfinite(solution.t))
                and np.all(np.isfinite(solution.y))
                and np.all(np.diff(solution.t) > 0.0)
            ):
                raise RuntimeError(
                    "Full-QS solve_ivp returned a non-finite or invalid grid."
                )
            final_tolerance = 1.0e-10 * max(abs(t_span_au[1]), 1.0)
            if abs(solution.t[-1] - t_span_au[1]) > final_tolerance:
                raise RuntimeError(
                    "Full-QS solve_ivp did not reach the requested final time."
                )

            observables = self._solution_observables(
                solution.t,
                solution.y,
                pulse,
            )
            observed_peak_rabi = float(np.max(np.abs(observables["rabi"])))
            required_frequency_ceiling = max(
                frequency_ceiling,
                observed_peak_rabi,
            )
            required_step_limit = float(
                2.0
                * np.pi
                / (points_per_fastest_cycle * required_frequency_ceiling)
            )
            if effective_max_step <= required_step_limit * (1.0 + 1.0e-12):
                frequency_ceiling = required_frequency_ceiling
                step_limit = required_step_limit
                effective_max_step = (
                    required_step_limit
                    if requested_max_step is None
                    else min(float(requested_max_step), required_step_limit)
                )
                break
            if rabi_step_refinement_count >= max_rabi_refinements:
                raise RuntimeError(
                    "The self-consistent local-field Rabi frequency did not "
                    "converge after automatic max-step refinement. Reduce the "
                    "field amplitude or provide a smaller max_step_au."
                )
            frequency_ceiling = max(
                required_frequency_ceiling,
                rabi_refinement_safety_factor * observed_peak_rabi,
            )
            rabi_step_refinement_count += 1

        W = observables["W"]
        Q_bloch = observables["Q"]
        P_bloch = observables["P"]
        incident = observables["incident"]
        mu_d = observables["mu_d"]
        mu_p = observables["mu_p"]
        mu_total = observables["mu_total"]
        mu_dot = observables["mu_dot_total"]
        mnp_field = observables["mnp_field"]
        effective_field = observables["effective_field"]
        modal_outputs = observables["modal_outputs"]
        rho22 = 0.5 * (W + 1.0)
        bloch_radius = np.sqrt(W**2 + Q_bloch**2 + P_bloch**2)
        max_bloch_radius = float(np.max(bloch_radius))
        min_density_eigenvalue = float(0.5 * (1.0 - max_bloch_radius))
        if min_density_eigenvalue < -positivity_tolerance:
            message = (
                "Full-QS density matrix left the Bloch ball: minimum eigenvalue="
                f"{min_density_eigenvalue:.6e}."
            )
            if positivity_policy == "raise":
                raise RuntimeError(message)
            if positivity_policy == "warn":
                warnings.warn(message, RuntimeWarning, stacklevel=2)

        work_au = float(solution.y[self.work_index, -1])
        absolute_work_scale = float(
            np.trapezoid(np.abs(incident * mu_dot), solution.t)
        )
        work_tolerance = 1.0e-8 * max(absolute_work_scale, np.finfo(float).tiny)
        work_nonnegative = bool(work_au >= -work_tolerance)
        if not work_nonnegative:
            message = (
                "Full-QS external work is negative beyond the integration tolerance: "
                f"work={work_au:.6e} au, tolerance={work_tolerance:.6e} au."
            )
            if work_passivity_policy == "raise":
                raise RuntimeError(message)
            if work_passivity_policy == "warn":
                warnings.warn(message, RuntimeWarning, stacklevel=2)

        work_j = work_au * AU_ENERGY_J
        fluence = pulse.fluence_j_cm2(eps_m=self.params.eps_m)
        sigma = work_j / fluence
        steps = np.diff(solution.t)

        def spectral_fraction(signal: np.ndarray) -> float:
            if float(np.max(np.abs(signal))) <= np.finfo(float).tiny:
                return 1.0
            return float(
                sampled_positive_frequency_spectral_fraction(
                    solution.t,
                    signal,
                    self.fit_window_eV,
                    highest_resolved_omega_au=frequency_ceiling,
                )
            )

        spectral_signals = {
            "QD source": observables["qd_source"],
            "MNP drive": observables["mnp_drive"],
            "MNP dipole": mu_p,
            "MNP field": mnp_field,
        }
        response_spectral_fractions = {
            name: spectral_fraction(signal)
            for name, signal in spectral_signals.items()
        }
        for name, fraction in response_spectral_fractions.items():
            response_leakage = float(1.0 - fraction)
            if response_leakage > max_spectral_leakage:
                message = (
                    f"{name} spectrum is not covered by the validated modal fit "
                    f"window: leakage={response_leakage:.6g}, "
                    f"limit={max_spectral_leakage:.6g}."
                )
                if spectral_window_policy == "raise":
                    raise ValueError(message)
                if spectral_window_policy == "warn":
                    warnings.warn(message, RuntimeWarning, stacklevel=2)

        tail_ratio = self._windowed_response_tail_ratio(
            solution.t,
            observables,
            window_fraction=response_tail_window_fraction,
        )
        tail_converged = bool(tail_ratio <= response_tail_tolerance)
        if not tail_converged:
            message = (
                "Full-QS response has not decayed within the requested time "
                f"window: tail ratio={tail_ratio:.6g}, "
                f"tolerance={response_tail_tolerance:.6g}."
            )
            if response_tail_policy == "raise":
                raise RuntimeError(message)
            if response_tail_policy == "warn":
                warnings.warn(message, RuntimeWarning, stacklevel=2)

        qd_source_fraction = response_spectral_fractions["QD source"]
        mnp_drive_fraction = response_spectral_fractions["MNP drive"]
        mnp_dipole_fraction = response_spectral_fractions["MNP dipole"]
        mnp_field_fraction = response_spectral_fractions["MNP field"]
        diagnostics = FullQSSolveDiagnostics(
            solver_success=bool(solution.success),
            solver_status=int(solution.status),
            solver_message=str(solution.message),
            n_steps=int(solution.t.size - 1),
            nfev=int(solution.nfev),
            min_step_au=float(np.min(steps)),
            max_step_au=float(np.max(steps)),
            max_step_limit_au=float(effective_max_step),
            integration_frequency_ceiling_au=float(frequency_ceiling),
            t_final_reached=True,
            state_is_finite=True,
            boundary_envelope_fraction=boundary_envelope,
            excited_population_min=float(np.min(rho22)),
            excited_population_max=float(np.max(rho22)),
            max_bloch_radius=max_bloch_radius,
            min_density_eigenvalue=min_density_eigenvalue,
            pulse_spectral_fraction_in_fit_window=float(1.0 - leakage),
            pulse_spectral_leakage=leakage,
            qd_source_spectral_fraction_in_fit_window=qd_source_fraction,
            qd_source_spectral_leakage=float(1.0 - qd_source_fraction),
            mnp_drive_spectral_fraction_in_fit_window=mnp_drive_fraction,
            mnp_drive_spectral_leakage=float(1.0 - mnp_drive_fraction),
            mnp_dipole_spectral_fraction_in_fit_window=mnp_dipole_fraction,
            mnp_dipole_spectral_leakage=float(1.0 - mnp_dipole_fraction),
            mnp_field_spectral_fraction_in_fit_window=mnp_field_fraction,
            mnp_field_spectral_leakage=float(1.0 - mnp_field_fraction),
            response_tail_ratio=tail_ratio,
            response_tail_tolerance=float(response_tail_tolerance),
            response_tail_converged=tail_converged,
            response_tail_window_fraction=float(response_tail_window_fraction),
            work_nonnegative_within_tolerance=work_nonnegative,
            work_passivity_tolerance_au=work_tolerance,
            spatial_order_max=self.n_spatial_modes,
            material_poles_per_spatial_order=self.n_material_modes,
            spectral_abscissa_au=self.coupled_stability.spectral_abscissa_au,
            spectral_abscissa_available=(
                self.coupled_stability.spectral_abscissa_available
            ),
            spectral_abscissa_is_bound=(
                self.coupled_stability.spectral_abscissa_is_bound
            ),
            decay_rate_estimate_au=(
                self.coupled_stability.decay_rate_estimate_au
            ),
            decay_rate_estimate_is_exact=(
                self.coupled_stability.decay_rate_estimate_is_exact
            ),
            spectral_radius_au=self.coupled_stability.spectral_radius_au,
            incident_peak_rabi_frequency_au=incident_peak_rabi,
            observed_peak_rabi_frequency_au=observed_peak_rabi,
            rabi_step_refinement_count=rabi_step_refinement_count,
            modal_fit_max_normalized_rms=(
                self.modal_fit_diagnostics.max_normalized_rms
            ),
            modal_fit_max_relative_error=self.modal_fit_diagnostics.max_relative_error,
        )
        return FullQSSolveResult(
            t_au=_readonly(solution.t, dtype=float),
            y=_readonly(solution.y, dtype=float),
            W=_readonly(W, dtype=float),
            Q=_readonly(Q_bloch, dtype=float),
            P=_readonly(P_bloch, dtype=float),
            rho22=_readonly(rho22, dtype=float),
            mu_p_au=_readonly(mu_p, dtype=float),
            mu_d_au=_readonly(mu_d, dtype=float),
            mu_total_au=_readonly(mu_total, dtype=float),
            mu_dot_total_au=_readonly(mu_dot, dtype=float),
            incident_field_au=_readonly(incident, dtype=float),
            mnp_field_at_qd_au=_readonly(mnp_field, dtype=float),
            effective_qd_field_au=_readonly(effective_field, dtype=float),
            modal_outputs_au=_readonly(modal_outputs, dtype=float),
            sigma_energy_transfer_cm2=float(sigma),
            work_from_incident_field_j=float(work_j),
            fluence_j_cm2=float(fluence),
            peak_intensity_w_cm2=float(
                pulse.peak_intensity_w_cm2(eps_m=self.params.eps_m)
            ),
            solve_ivp_result=solution,
            diagnostics=diagnostics,
        )
