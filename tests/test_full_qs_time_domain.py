"""Causality, reciprocity and weak-field tests for the full-QS pulse backend."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse.linalg import ArpackNoConvergence

from qd_mnp_full_qs_model import FullQSSpheroidPulseModel
from qd_mnp_pulse_absorption_sweep import spectral_effective_alpha_au
from qd_mnp_rational_fit import (
    AU_ENERGY_J,
    GaussianPulse,
    HybridQDPlasmonModel,
    eV_to_au,
    fs_to_au,
    make_default_params,
)
from qd_mnp_spheroid_green import (
    SpheroidGreenInteraction,
    qd_linear_polarizability_from_params,
    solve_linear_hybrid_response,
)


def _one_material_pole_model(*, spatial_orders: int = 2) -> FullQSSpheroidPulseModel:
    params = replace(
        make_default_params("long"),
        gamma_au=float(eV_to_au(0.020)),
        Gamma_au=float(eV_to_au(0.020)),
    )
    bright = HybridQDPlasmonModel(
        params,
        orientation="long",
        n_modes=1,
        max_fit_normalized_rms=None,
        max_fit_pointwise_relative_error=None,
        radiative_consistency_policy="ignore",
        verbose=False,
    )
    kernel = SpheroidGreenInteraction.from_params(
        params,
        orientation="long",
        n_max=spatial_orders,
    )
    return FullQSSpheroidPulseModel(
        bright,
        kernel,
        # The one-pole fixture is deliberately an algebra test, not a
        # production-quality representation of tabulated gold.
        fit_quality_policy="ignore",
        spatial_convergence_policy="ignore",
        modal_audit_points=201,
    )


class FullQSTransferRealizationTests(unittest.TestCase):
    def test_transformed_susceptibilities_obey_the_common_material_identity(self) -> None:
        model = _one_material_pole_model(spatial_orders=4)
        energies = np.asarray([1.9, 2.042, 2.2])
        H = model.bright_model.alpha_from_fit(energies)
        expected = H[None, :] / (1.0 + model.delta_L[:, None] * H[None, :])
        np.testing.assert_allclose(
            model.modal_susceptibility_from_fit(energies),
            expected,
            rtol=2.0e-15,
            atol=0.0,
        )

    def test_frequency_backend_preserves_bright_reciprocity(self) -> None:
        model = _one_material_pole_model(spatial_orders=4)
        response = model.frequency_response_from_fit(np.asarray([1.9, 2.042, 2.2]))
        np.testing.assert_allclose(
            response.K_bright_au_minus3,
            response.B**2 / response.A_au3,
            rtol=5.0e-14,
            atol=0.0,
        )
        np.testing.assert_allclose(
            response.K_au_minus3,
            np.sum(response.K_by_degree_au_minus3, axis=0),
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(response.eps_m, model.params.eps_m)
        np.testing.assert_allclose(
            response.log_abs_geometric_factor_by_degree,
            model.kernel.log_abs_geometric_factor_by_degree,
            rtol=0.0,
            atol=0.0,
        )

    def test_field_free_ground_state_and_transformed_material_are_stable(self) -> None:
        model = _one_material_pole_model(spatial_orders=4)
        pulse = GaussianPulse(
            E0_au=1.0e-8,
            omegaL_au=float(eV_to_au(2.042)),
            tau_au=float(fs_to_au(5.0)),
            tau_kind="fwhm_intensity",
        )
        derivative = model.rhs(1.0e9, model.initial_state(), pulse)
        np.testing.assert_allclose(derivative, np.zeros_like(derivative), atol=0.0)
        self.assertTrue(model.coupled_stability.stable)
        self.assertLessEqual(
            model.coupled_stability.spectral_abscissa_au,
            model.coupled_stability.tolerance_au,
        )
        self.assertLessEqual(float(np.max(model.modal_poles_au.real)), 0.0)
        dense_poles = np.linalg.eigvals(
            model.linearized_ground_state_matrix().toarray()
        )
        self.assertAlmostEqual(
            model.coupled_stability.spectral_radius_au,
            float(np.max(np.abs(dense_poles))),
            places=13,
        )
        self.assertTrue(model.coupled_stability.spectral_abscissa_available)
        self.assertFalse(model.coupled_stability.spectral_abscissa_is_bound)
        self.assertTrue(model.coupled_stability.decay_rate_estimate_is_exact)
        self.assertAlmostEqual(
            model.coupled_stability.decay_rate_estimate_au,
            -model.coupled_stability.spectral_abscissa_au,
            places=13,
        )
        self.assertFalse(model.coupled_stability.rightmost_poles_au.flags.writeable)
        self.assertFalse(
            model.coupled_stability.largest_magnitude_poles_au.flags.writeable
        )

    def test_production_nine_pole_transform_passes_accuracy_and_passivity_gate(self) -> None:
        params = make_default_params("trans")
        bright = HybridQDPlasmonModel(
            params,
            orientation="trans",
            n_modes=9,
            radiative_consistency_policy="ignore",
            verbose=False,
        )
        model = FullQSSpheroidPulseModel(
            bright,
            SpheroidGreenInteraction.from_params(
                params,
                orientation="trans",
                n_max=80,
            ),
            modal_audit_points=501,
        )
        diagnostics = model.modal_fit_diagnostics
        self.assertTrue(diagnostics.accepted)
        self.assertTrue(diagnostics.passive_on_audit_grid)
        self.assertLessEqual(diagnostics.max_normalized_rms, 0.03)
        self.assertLessEqual(diagnostics.max_relative_error, 0.06)
        spatial = model.spatial_convergence_diagnostics
        self.assertTrue(spatial.accepted)
        self.assertLessEqual(spatial.max_half_order_relative_change, 1.0e-8)
        self.assertLessEqual(spatial.max_tail_block_relative_mass, 1.0e-8)
        self.assertIsNone(model.coupled_stability.spectral_abscissa_au)
        self.assertFalse(model.coupled_stability.spectral_abscissa_available)
        self.assertFalse(model.coupled_stability.spectral_abscissa_is_bound)
        self.assertEqual(model.coupled_stability.rightmost_poles_au.size, 0)
        self.assertGreater(model.coupled_stability.decay_rate_estimate_au, 0.0)
        self.assertFalse(model.coupled_stability.decay_rate_estimate_is_exact)
        self.assertIn(
            "passive_no_soft_mode_certificate",
            model.coupled_stability.eigensolver,
        )

    def test_low_order_fixture_requires_an_explicit_spatial_policy_override(self) -> None:
        params = replace(
            make_default_params("long"),
            gamma_au=float(eV_to_au(0.020)),
            Gamma_au=float(eV_to_au(0.020)),
        )
        bright = HybridQDPlasmonModel(
            params,
            orientation="long",
            n_modes=1,
            max_fit_normalized_rms=None,
            max_fit_pointwise_relative_error=None,
            radiative_consistency_policy="ignore",
            verbose=False,
        )
        kernel = SpheroidGreenInteraction.from_params(
            params,
            orientation="long",
            n_max=2,
        )
        with self.assertRaisesRegex(RuntimeError, "spatial series is not converged"):
            FullQSSpheroidPulseModel(
                bright,
                kernel,
                fit_quality_policy="ignore",
                modal_audit_points=201,
            )
        ignored = FullQSSpheroidPulseModel(
            bright,
            kernel,
            fit_quality_policy="ignore",
            spatial_convergence_policy="ignore",
            modal_audit_points=201,
        )
        self.assertFalse(ignored.spatial_convergence_diagnostics.accepted)

    def test_partial_sparse_arpack_result_is_not_accepted_as_a_certificate(self) -> None:
        params = make_default_params("trans")
        bright = HybridQDPlasmonModel(
            params,
            orientation="trans",
            n_modes=9,
            radiative_consistency_policy="ignore",
            verbose=False,
        )
        partial = ArpackNoConvergence(
            "deliberate partial result",
            np.asarray([-1.0e-4 + 0.08j]),
            np.ones((326, 1), dtype=complex),
        )
        with patch("qd_mnp_full_qs_model.eigs", side_effect=partial):
            with self.assertRaisesRegex(RuntimeError, "partial ARPACK spectrum"):
                FullQSSpheroidPulseModel(
                    bright,
                    SpheroidGreenInteraction.from_params(
                        params,
                        orientation="trans",
                        n_max=18,
                    ),
                    spatial_convergence_policy="ignore",
                    modal_audit_points=201,
                )


class FullQSPulseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = _one_material_pole_model(spatial_orders=2)
        cls.pulse = GaussianPulse(
            E0_au=1.0e-8,
            omegaL_au=float(eV_to_au(2.042)),
            tau_au=float(fs_to_au(5.0)),
            tau_kind="fwhm_intensity",
        )
        cls.result = cls.model.solve(
            cls.pulse,
            method="DOP853",
            rtol=3.0e-9,
            atol=1.0e-11,
        )

    def test_weak_pulse_trajectory_matches_its_frequency_response(self) -> None:
        alpha_time = spectral_effective_alpha_au(
            self.result,
            self.pulse,
            self.model.params.eps_m,
        )
        response = self.model.frequency_response_from_fit(np.asarray([2.042]))
        beta = qd_linear_polarizability_from_params(
            self.model.params,
            np.asarray([2.042]),
        )
        alpha_frequency = solve_linear_hybrid_response(
            response,
            beta,
            eps_m=self.model.params.eps_m,
        ).alpha_effective_au3[0]
        self.assertLess(
            abs(alpha_time - alpha_frequency) / abs(alpha_frequency),
            5.0e-6,
        )
        self.assertLess(self.result.diagnostics.excited_population_max, 1.0e-7)

    def test_explicit_observables_and_work_accumulator_are_self_consistent(self) -> None:
        np.testing.assert_allclose(
            self.result.mu_total_au,
            self.result.mu_p_au + self.result.mu_d_au,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            self.result.rho22,
            0.5 * (self.result.W + 1.0),
            rtol=0.0,
            atol=0.0,
        )
        work_quadrature_au = np.trapezoid(
            self.result.incident_field_au * self.result.mu_dot_total_au,
            self.result.t_au,
        )
        work_accumulator_au = self.result.work_from_incident_field_j / AU_ENERGY_J
        self.assertTrue(
            np.isclose(
                work_quadrature_au,
                work_accumulator_au,
                rtol=2.0e-5,
                atol=1.0e-14,
            )
        )
        self.assertTrue(self.result.diagnostics.work_nonnegative_within_tolerance)
        self.assertGreater(self.result.work_from_incident_field_j, 0.0)
        self.assertGreater(self.result.sigma_energy_transfer_cm2, 0.0)
        self.assertGreaterEqual(self.result.diagnostics.min_density_eigenvalue, -1.0e-12)

    def test_default_window_response_spectra_and_tail_are_audited(self) -> None:
        diagnostics = self.result.diagnostics
        expected_span = self.model.default_time_span(self.pulse)
        self.assertAlmostEqual(self.result.t_au[0], expected_span[0], places=12)
        self.assertAlmostEqual(self.result.t_au[-1], expected_span[1], places=12)
        self.assertEqual(diagnostics.n_steps, self.result.t_au.size - 1)
        self.assertTrue(diagnostics.response_tail_converged)
        self.assertLessEqual(
            diagnostics.response_tail_ratio,
            diagnostics.response_tail_tolerance,
        )
        self.assertEqual(diagnostics.response_tail_tolerance, 1.0e-4)
        self.assertEqual(diagnostics.response_tail_window_fraction, 0.05)
        for fraction_name, leakage_name in (
            (
                "qd_source_spectral_fraction_in_fit_window",
                "qd_source_spectral_leakage",
            ),
            (
                "mnp_drive_spectral_fraction_in_fit_window",
                "mnp_drive_spectral_leakage",
            ),
            (
                "mnp_dipole_spectral_fraction_in_fit_window",
                "mnp_dipole_spectral_leakage",
            ),
            (
                "mnp_field_spectral_fraction_in_fit_window",
                "mnp_field_spectral_leakage",
            ),
        ):
            fraction = getattr(diagnostics, fraction_name)
            leakage = getattr(diagnostics, leakage_name)
            self.assertGreaterEqual(fraction, 0.0)
            self.assertLessEqual(fraction, 1.0)
            self.assertAlmostEqual(fraction + leakage, 1.0, places=15)
            self.assertLessEqual(leakage, 1.0e-3)

    def test_rhs_validates_shape_type_and_finiteness_without_integer_truncation(self) -> None:
        integer_state = self.model.initial_state().astype(int)
        derivative = self.model.rhs(0.0, integer_state, self.pulse)
        self.assertEqual(derivative.shape, (self.model.state_size,))
        self.assertEqual(derivative.dtype, np.dtype(float))
        self.assertGreater(float(np.max(np.abs(derivative))), 0.0)

        invalid_states = (
            np.zeros(self.model.state_size - 1),
            np.zeros((self.model.state_size, 1)),
            np.zeros(self.model.state_size, dtype=complex),
        )
        for state in invalid_states:
            with self.subTest(shape=state.shape, dtype=state.dtype):
                with self.assertRaises((TypeError, ValueError)):
                    self.model.rhs(0.0, state, self.pulse)
        nonfinite = self.model.initial_state()
        nonfinite[0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            self.model.rhs(0.0, nonfinite, self.pulse)
        with self.assertRaises((TypeError, ValueError)):
            self.model.rhs(np.nan, self.model.initial_state(), self.pulse)
        with self.assertRaises(TypeError):
            self.model.rhs(0.0, self.model.initial_state(), object())

    def test_solve_api_rejects_invalid_controls_and_non_straddling_windows(self) -> None:
        for kwargs in (
            {"method": "Euler"},
            {"rtol": 0.0},
            {"atol": np.inf},
            {"max_step_au": 0.0},
            {"points_per_fastest_cycle": 7.99},
            {"spectral_window_policy": "bad"},
            {"positivity_policy": "bad"},
            {"work_passivity_policy": "bad"},
            {"response_tail_policy": "bad"},
            {"max_spectral_leakage": 1.0},
            {"positivity_tolerance": -1.0},
            {"response_tail_tolerance": 0.0},
            {"response_tail_window_fraction": 0.0},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises((TypeError, ValueError)):
                    self.model.solve(self.pulse, **kwargs)

        remote_tail = 10.0 * self.pulse.sigma_t_au
        for span in (
            (remote_tail, 2.0 * remote_tail),
            (-2.0 * remote_tail, -remote_tail),
            (0.0, remote_tail),
            (-remote_tail, 0.0),
        ):
            with self.subTest(t_span_au=span):
                with self.assertRaisesRegex(ValueError, "straddle"):
                    self.model.solve(self.pulse, t_span_au=span)

    def test_tail_audit_detects_significant_modal_state_but_ignores_roundoff_channel(self) -> None:
        times = np.linspace(0.0, 100.0, 101)
        zero = np.zeros_like(times)
        early_field = zero.copy()
        early_field[10] = 1.0
        base = {
            "mu_total": zero,
            "mu_p": zero,
            "mu_d": zero,
            "mnp_field": early_field,
            "modal_outputs": np.zeros((self.model.n_spatial_modes, times.size)),
            "q": np.zeros(
                (
                    self.model.n_spatial_modes,
                    self.model.n_material_modes,
                    times.size,
                )
            ),
            "velocity": np.zeros(
                (
                    self.model.n_spatial_modes,
                    self.model.n_material_modes,
                    times.size,
                )
            ),
            "Q": zero,
            "P": zero,
        }
        _, field_weights = self.model._input_and_field_weights()
        base["q"][0, 0, -10:] = 1.0e-13 / abs(field_weights[0])
        negligible = self.model._windowed_response_tail_ratio(
            times,
            base,
            window_fraction=0.1,
        )
        self.assertEqual(negligible, 0.0)
        base["q"][0, 0, -10:] = 1.0e-9 / abs(field_weights[0])
        significant = self.model._windowed_response_tail_ratio(
            times,
            base,
            window_fraction=0.1,
        )
        self.assertGreater(significant, 0.8)

    def test_strong_field_refines_step_from_observed_local_rabi_frequency(self) -> None:
        model = _one_material_pole_model(spatial_orders=1)
        pulse = GaussianPulse(
            E0_au=1.0e-2,
            omegaL_au=float(eV_to_au(2.042)),
            tau_au=float(fs_to_au(1.0)),
            tau_kind="fwhm_intensity",
        )
        result = model.solve(
            pulse,
            t_span_au=(-8.0 * pulse.sigma_t_au, 8.0 * pulse.sigma_t_au),
            rtol=1.0e-7,
            atol=1.0e-9,
            spectral_window_policy="ignore",
            work_passivity_policy="ignore",
            response_tail_policy="ignore",
        )
        diagnostics = result.diagnostics
        self.assertGreaterEqual(diagnostics.rabi_step_refinement_count, 1)
        self.assertGreater(
            diagnostics.observed_peak_rabi_frequency_au,
            diagnostics.incident_peak_rabi_frequency_au,
        )
        required_step = (
            2.0
            * np.pi
            / (
                20.0
                * diagnostics.observed_peak_rabi_frequency_au
            )
        )
        self.assertLessEqual(
            diagnostics.max_step_limit_au,
            required_step * (1.0 + 1.0e-12),
        )


if __name__ == "__main__":
    unittest.main()
