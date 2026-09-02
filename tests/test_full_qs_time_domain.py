"""Causality, reciprocity and weak-field tests for the full-QS pulse backend."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse.linalg import ArpackNoConvergence

from qd_mnp_full_qs_model import (
    FullQSSpheroidPulseModel,
    build_positive_dark_reduction,
)
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
from qd_mnp_spheroid_equatorial import EquatorialSpheroidGreenInteraction


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


def _one_material_pole_side_model(
    orientation: str,
    alignment: str | None,
    *,
    spatial_orders: int = 3,
) -> FullQSSpheroidPulseModel:
    params = replace(
        make_default_params(
            orientation,
            qd_placement="side",
            side_transverse_alignment=alignment,
        ),
        gamma_au=float(eV_to_au(0.020)),
        Gamma_au=float(eV_to_au(0.020)),
    )
    bright = HybridQDPlasmonModel(
        params,
        orientation=orientation,
        n_modes=1,
        max_fit_normalized_rms=None,
        max_fit_pointwise_relative_error=None,
        radiative_consistency_policy="ignore",
        verbose=False,
    )
    kernel = EquatorialSpheroidGreenInteraction.from_params(
        params,
        orientation=orientation,
        n_max=spatial_orders,
    )
    return FullQSSpheroidPulseModel(
        bright,
        kernel,
        fit_quality_policy="ignore",
        spatial_convergence_policy="ignore",
        modal_audit_points=201,
    )


class FullQSTransferRealizationTests(unittest.TestCase):
    def test_positive_dark_reduction_preserves_frequency_response(self) -> None:
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
            n_max=8,
        )
        reduction = build_positive_dark_reduction(
            bright,
            kernel,
            fit_grid_points=201,
            audit_grid_points=307,
            rms_tolerance=1.0e-7,
            max_tolerance=1.0e-6,
        )
        direct = FullQSSpheroidPulseModel(
            bright,
            kernel,
            fit_quality_policy="ignore",
            spatial_convergence_policy="ignore",
            modal_audit_points=201,
        )
        reduced = FullQSSpheroidPulseModel(
            bright,
            kernel,
            dark_reduction=reduction,
            fit_quality_policy="ignore",
            spatial_convergence_policy="ignore",
            modal_audit_points=201,
        )
        reaudit = reduced.dark_reduction_reaudit_diagnostics
        self.assertIsNotNone(reaudit)
        self.assertTrue(reaudit.accepted)
        self.assertTrue(reaudit.passive_on_audit_grid)
        self.assertEqual(reaudit.audit_grid_points, 1709)
        self.assertLessEqual(reaudit.normalized_rms, reaudit.rms_tolerance)
        self.assertLessEqual(
            reaudit.max_normalized_error,
            reaudit.max_tolerance,
        )
        energies = np.linspace(1.81, 2.29, 173)
        exact_response = direct.frequency_response_from_fit(energies)
        reduced_response = reduced.frequency_response_from_fit(energies)
        relative = np.max(
            np.abs(reduced_response.K_au_minus3 - exact_response.K_au_minus3)
        ) / np.max(np.abs(exact_response.K_au_minus3))
        self.assertLess(relative, 1.0e-6)
        np.testing.assert_allclose(
            reduced_response.A_au3,
            exact_response.A_au3,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            reduced_response.B,
            exact_response.B,
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(reduced.exact_spatial_mode_count, 8)
        self.assertEqual(reduced.n_spatial_modes, 1 + reduction.node_count)
        self.assertLessEqual(reduced.state_size, direct.state_size)

        audit_energies = np.linspace(
            reduced.fit_window_eV[0],
            reduced.fit_window_eV[1],
            201,
        )
        audit_fit = reduced.frequency_response_from_fit(audit_energies)
        audit_exact = kernel.response_from_material(
            params.material,
            audit_energies,
        )
        audit_error = audit_fit.K_au_minus3 - audit_exact.K_au_minus3
        expected_nrms = float(
            np.sqrt(np.mean(np.abs(audit_error) ** 2))
            / np.sqrt(np.mean(np.abs(audit_exact.K_au_minus3) ** 2))
        )
        expected_max = float(
            np.max(
                np.abs(audit_error)
                / np.maximum(
                    np.abs(audit_exact.K_au_minus3),
                    1.0e-15 * np.max(np.abs(audit_exact.K_au_minus3)),
                )
            )
        )
        self.assertAlmostEqual(
            reduced.modal_fit_diagnostics.K_normalized_rms,
            expected_nrms,
            places=15,
        )
        self.assertAlmostEqual(
            reduced.modal_fit_diagnostics.K_max_relative_error,
            expected_max,
            places=15,
        )

    def test_reduction_certificate_cannot_be_reused_for_another_kernel(self) -> None:
        target_params = replace(
            make_default_params("long"),
            gamma_au=float(eV_to_au(0.020)),
            Gamma_au=float(eV_to_au(0.020)),
        )
        source_params = replace(
            target_params,
            R_au=1.5 * target_params.R_au,
        )
        source_bright = HybridQDPlasmonModel(
            source_params,
            orientation="long",
            n_modes=1,
            max_fit_normalized_rms=None,
            max_fit_pointwise_relative_error=None,
            radiative_consistency_policy="ignore",
            verbose=False,
        )
        source_kernel = SpheroidGreenInteraction.from_params(
            source_params,
            orientation="long",
            n_max=6,
        )
        foreign_reduction = build_positive_dark_reduction(
            source_bright,
            source_kernel,
            fit_grid_points=201,
            audit_grid_points=307,
        )
        target_bright = HybridQDPlasmonModel(
            target_params,
            orientation="long",
            n_modes=1,
            max_fit_normalized_rms=None,
            max_fit_pointwise_relative_error=None,
            radiative_consistency_policy="ignore",
            verbose=False,
        )
        target_kernel = SpheroidGreenInteraction.from_params(
            target_params,
            orientation="long",
            n_max=6,
        )
        self.assertEqual(source_kernel.n_max, target_kernel.n_max)
        np.testing.assert_array_equal(
            source_kernel.depolarization_by_degree,
            target_kernel.depolarization_by_degree,
        )
        with self.assertRaisesRegex(ValueError, "different full modal measure"):
            FullQSSpheroidPulseModel(
                target_bright,
                target_kernel,
                dark_reduction=foreign_reduction,
                fit_quality_policy="ignore",
                spatial_convergence_policy="ignore",
                modal_audit_points=201,
            )

    def test_reduction_is_reaudited_with_the_current_bright_transfer(self) -> None:
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
            n_max=9,
        )

        class ArtificialCertificateTransfer:
            fit_window_eV = bright.fit_window_eV

            @staticmethod
            def alpha_from_fit(energies_eV):
                energies = np.asarray(energies_eV, dtype=float)
                return np.full(energies.shape, 1.0e-9 + 1.0e-12j)

        misleading_reduction = build_positive_dark_reduction(
            ArtificialCertificateTransfer(),
            kernel,
            fit_grid_points=201,
            audit_grid_points=307,
        )
        self.assertTrue(misleading_reduction.diagnostics.accepted)
        self.assertEqual(misleading_reduction.node_count, 1)
        with self.assertRaisesRegex(
            ValueError,
            "not accurate/passive for the current bright material transfer",
        ):
            FullQSSpheroidPulseModel(
                bright,
                kernel,
                dark_reduction=misleading_reduction,
                fit_quality_policy="ignore",
                spatial_convergence_policy="ignore",
                modal_audit_points=201,
            )

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


class EquatorialFullQSTimeDomainTests(unittest.TestCase):
    channels = (
        ("long", None),
        ("trans", "radial"),
        ("trans", "tangential"),
    )

    @classmethod
    def setUpClass(cls) -> None:
        cls.pulse = GaussianPulse(
            E0_au=1.0e-8,
            omegaL_au=float(eV_to_au(2.042)),
            tau_au=float(fs_to_au(5.0)),
            tau_kind="fwhm_intensity",
        )
        cls.models_and_results = []
        for orientation, alignment in cls.channels:
            model = _one_material_pole_side_model(orientation, alignment)
            result = model.solve(
                cls.pulse,
                method="DOP853",
                rtol=3.0e-9,
                atol=1.0e-11,
            )
            cls.models_and_results.append((model, result))

    def test_all_three_side_channels_match_their_frequency_response(self) -> None:
        for (orientation, alignment), (model, result) in zip(
            self.channels,
            self.models_and_results,
        ):
            with self.subTest(orientation=orientation, alignment=alignment):
                alpha_time = spectral_effective_alpha_au(
                    result,
                    self.pulse,
                    model.params.eps_m,
                )
                response = model.frequency_response_from_fit(np.asarray([2.042]))
                beta = qd_linear_polarizability_from_params(
                    model.params,
                    np.asarray([2.042]),
                )
                alpha_frequency = solve_linear_hybrid_response(
                    response,
                    beta,
                    eps_m=model.params.eps_m,
                ).alpha_effective_au3[0]
                self.assertLess(
                    abs(alpha_time - alpha_frequency) / abs(alpha_frequency),
                    5.0e-6,
                )
                self.assertLess(result.diagnostics.excited_population_max, 1.0e-7)

    def test_side_trajectories_remain_physical_and_report_mode_counts(self) -> None:
        for (orientation, alignment), (model, result) in zip(
            self.channels,
            self.models_and_results,
        ):
            with self.subTest(orientation=orientation, alignment=alignment):
                diagnostics = result.diagnostics
                self.assertTrue(diagnostics.work_nonnegative_within_tolerance)
                self.assertGreaterEqual(diagnostics.min_density_eigenvalue, -1.0e-12)
                self.assertLessEqual(diagnostics.max_bloch_radius, 1.0 + 1.0e-10)
                self.assertEqual(diagnostics.spatial_order_max, 3)
                self.assertEqual(
                    diagnostics.exact_spatial_mode_count,
                    model.kernel.mode_count,
                )
                self.assertEqual(
                    diagnostics.dynamic_spatial_mode_count,
                    model.kernel.mode_count,
                )
                self.assertEqual(diagnostics.reduced_dark_node_count, 0)

    def test_n80_positive_reduction_covers_every_dark_mode(self) -> None:
        for orientation, alignment in self.channels:
            with self.subTest(orientation=orientation, alignment=alignment):
                params = make_default_params(
                    orientation,
                    qd_placement="side",
                    side_transverse_alignment=alignment,
                )
                bright = HybridQDPlasmonModel(
                    params,
                    orientation=orientation,
                    n_modes=9,
                    radiative_consistency_policy="ignore",
                    verbose=False,
                )
                kernel = EquatorialSpheroidGreenInteraction.from_params(
                    params,
                    orientation=orientation,
                    n_max=80,
                )
                reduction = build_positive_dark_reduction(
                    bright,
                    kernel,
                    fit_grid_points=201,
                    audit_grid_points=307,
                )
                max_modal_relative_error = (
                    0.075 if alignment == "radial" else 0.06
                )
                reduced_model = FullQSSpheroidPulseModel(
                    bright,
                    kernel,
                    dark_reduction=reduction,
                    max_modal_relative_error=max_modal_relative_error,
                    modal_audit_points=201,
                )

                energies = np.linspace(0.8, 3.0, 257)
                bright_susceptibility = bright.alpha_from_fit(energies)
                exact_modal = bright_susceptibility[None, :] / (
                    1.0
                    + (
                        kernel.depolarization_by_mode
                        - kernel.depolarization_by_mode[kernel.bright_mode_index]
                    )[:, None]
                    * bright_susceptibility[None, :]
                )
                exact = kernel.response_from_modal_susceptibility(exact_modal)
                reduced = reduced_model.frequency_response_from_fit(energies)
                relative_K_error = float(
                    np.max(np.abs(reduced.K_au_minus3 - exact.K_au_minus3))
                    / np.max(np.abs(exact.K_au_minus3))
                )

                self.assertTrue(reduction.diagnostics.accepted)
                self.assertEqual(
                    reduction.diagnostics.positive_dark_mode_count,
                    kernel.mode_count - 1,
                )
                self.assertLess(relative_K_error, 1.0e-4)
                self.assertLess(reduction.node_count, kernel.mode_count // 20)
                self.assertTrue(reduced_model.modal_fit_diagnostics.accepted)
                self.assertTrue(
                    reduced_model.modal_fit_diagnostics.passive_on_audit_grid
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
