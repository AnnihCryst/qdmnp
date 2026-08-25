"""Physics contracts for the Sadeghi-2009 native-model adapter."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from qd_mnp_rational_fit import AU_TIME_S, au_to_nm, nm_to_au
from qd_mnp_sadeghi2009_comparison import (
    MEV_TO_NS_INV,
    Sadeghi2009Profile,
    build_sadeghi_adapter,
    build_sadeghi_params,
    isolated_steady_state,
    logistic_switch_envelope,
    run_comparison,
    solve_steady_state,
    solve_switch_on_trace,
    steady_state_jacobian_ns_inv,
)


class SadeghiProfileAndMappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = Sadeghi2009Profile()
        cls.adapter = build_sadeghi_adapter(cls.profile)

    def test_direct_inferred_and_project_parameters_have_separate_provenance(self) -> None:
        provenance = self.profile.provenance()

        self.assertIn("direct_from_sadeghi2009", provenance)
        self.assertIn("inferred_from_sadeghi_reference_9", provenance)
        self.assertIn("project_model_choices", provenance)
        self.assertEqual(
            provenance["project_model_choices"]["gold_data"],
            "Johnson-Christy table bundled with the project",
        )
        self.assertEqual(
            provenance["project_model_choices"]["paper_reference_gold_data"],
            "Palik",
        )
        self.assertEqual(provenance["project_model_choices"]["qd_radius_nm"], 0.0)
        self.assertNotIn(
            "qd_radius_nm",
            provenance["inferred_from_sadeghi_reference_9"],
        )

    def test_spherical_longitudinal_core_mapping_and_cp_rates(self) -> None:
        params = build_sadeghi_params(self.profile)

        self.assertAlmostEqual(float(au_to_nm(params.c_au)), 7.5, places=13)
        self.assertAlmostEqual(params.c_au, params.a_au, places=13)
        self.assertEqual(params.G, 2.0)
        self.assertEqual(params.qd_dipole_convention, "bare_internal")
        self.assertAlmostEqual(params.qd_local_field_factor, 3.0 / 8.0, places=14)
        self.assertGreaterEqual(
            self.profile.coherence_decay_ns_inv,
            0.5 * self.profile.population_decay_ns_inv,
        )
        self.assertAlmostEqual(self.profile.pure_dephasing_ns_inv, 2.075, places=14)

    def test_article_intensities_reconstruct_printed_rabi_frequencies(self) -> None:
        weak = self.profile.isolated_rabi_ns_inv(1.0)
        strong = self.profile.isolated_rabi_ns_inv(1.0e3)

        self.assertAlmostEqual(weak, 0.5082520542533421, places=13)
        self.assertAlmostEqual(strong, 16.07234116900031, places=12)
        self.assertAlmostEqual(strong / weak, np.sqrt(1000.0), places=13)

    def test_all_article_distances_have_positive_point_qd_gap(self) -> None:
        expected = (92.5, 12.5, 9.5, 7.5)

        for distance, gap in zip(self.profile.distances_nm, expected):
            with self.subTest(distance_nm=distance):
                self.assertAlmostEqual(self.profile.surface_gap_nm(distance), gap)
                coupling = self.adapter.coupling(
                    laser_energy_ev=self.profile.transition_energy_ev,
                    distance_nm=distance,
                )
                self.assertGreater(coupling.point_qd_surface_gap_nm, 0.0)

        with self.assertRaisesRegex(ValueError, "strictly positive"):
            self.adapter.coupling(
                laser_energy_ev=self.profile.transition_energy_ev,
                distance_nm=self.profile.sphere_radius_nm,
            )

    def test_core_sphere_alpha_equals_eps_a3_gamma(self) -> None:
        energy = self.profile.transition_energy_ev
        epsilon_gold = complex(self.adapter.params.material.epsilon_at(energy))
        gamma = (epsilon_gold - self.profile.eps_environment) / (
            epsilon_gold + 2.0 * self.profile.eps_environment
        )
        radius_au = float(nm_to_au(self.profile.sphere_radius_nm))
        expected = self.profile.eps_environment * radius_au**3 * gamma
        actual = complex(
            self.adapter.alpha_physical_au(energy, response="material")
        )

        self.assertAlmostEqual(actual.real / expected.real, 1.0, places=13)
        self.assertAlmostEqual(actual.imag / expected.imag, 1.0, places=13)

    def test_direct_and_feedback_terms_scale_as_R_minus_3_and_R_minus_6(self) -> None:
        first = self.adapter.coupling(
            laser_energy_ev=self.profile.transition_energy_ev,
            distance_nm=30.0,
            response="material",
        )
        second = self.adapter.coupling(
            laser_energy_ev=self.profile.transition_energy_ev,
            distance_nm=60.0,
            response="material",
        )

        self.assertAlmostEqual(
            abs(first.direct_factor - 1.0) / abs(second.direct_factor - 1.0),
            8.0,
            places=12,
        )
        self.assertAlmostEqual(
            abs(first.feedback_ns_inv) / abs(second.feedback_ns_inv),
            64.0,
            places=12,
        )

    def test_direct_and_feedback_coefficients_equal_article_formulas(self) -> None:
        distance_nm = 30.0
        distance_au = float(nm_to_au(distance_nm))
        radius_au = float(nm_to_au(self.profile.sphere_radius_nm))
        epsilon_gold = complex(
            self.adapter.params.material.epsilon_at(
                self.profile.transition_energy_ev
            )
        )
        gamma = (epsilon_gold - self.profile.eps_environment) / (
            epsilon_gold + 2.0 * self.profile.eps_environment
        )
        expected_direct = (
            1.0
            + 2.0 * gamma * radius_au**3 / distance_au**3
        )
        expected_feedback_au = (
            4.0
            * gamma
            * self.adapter.params.d_au**2
            * self.profile.qd_local_field_factor**2
            * radius_au**3
            / (self.profile.eps_environment * distance_au**6)
        )
        expected_feedback_ns_inv = (
            expected_feedback_au / AU_TIME_S * 1.0e-9
        )
        actual = self.adapter.coupling(
            laser_energy_ev=self.profile.transition_energy_ev,
            distance_nm=distance_nm,
            response="material",
        )

        np.testing.assert_allclose(
            actual.direct_factor,
            expected_direct,
            rtol=2.0e-15,
            atol=2.0e-15,
        )
        np.testing.assert_allclose(
            actual.feedback_ns_inv,
            expected_feedback_ns_inv,
            rtol=2.0e-15,
            atol=2.0e-15,
        )

    def test_cached_modal_fit_is_passive_and_matches_local_material_response(self) -> None:
        energies = np.linspace(2.49, 2.51, 81)
        modal = self.adapter.alpha_physical_au(energies, response="modal")
        material = self.adapter.alpha_physical_au(energies, response="material")
        relative_error = np.abs(modal - material) / np.abs(material)

        self.assertTrue(self.adapter.fit.passive_for_all_positive_frequencies)
        self.assertTrue(np.all(self.adapter.fit.strengths_au2 >= 0.0))
        self.assertGreaterEqual(float(np.min(modal.imag)), 0.0)
        self.assertLess(float(np.max(relative_error)), 0.025)

        cached_modal = self.adapter.alpha_dimless(
            self.adapter.fit.energies_used_eV,
            response="modal",
        )
        cached_target = self.adapter.fit.alpha_used
        cached_inverse_error = 1.0 / cached_modal - 1.0 / cached_target
        expected_rms_alpha = float(
            np.sqrt(np.mean(np.abs(cached_modal - cached_target) ** 2))
        )
        expected_rms_inverse = float(
            np.sqrt(np.mean(np.abs(cached_inverse_error) ** 2))
        )
        expected_normalized_rms_alpha = expected_rms_alpha / float(
            np.sqrt(np.mean(np.abs(cached_target) ** 2))
        )
        expected_normalized_rms_inverse = expected_rms_inverse / float(
            np.sqrt(np.mean(np.abs(1.0 / cached_target) ** 2))
        )
        alpha_error = cached_modal - cached_target
        inverse_target = 1.0 / cached_target
        expected_maximum_relative_error = float(
            np.max(np.abs(alpha_error) / np.abs(cached_target))
        )
        residual_parts = [
            alpha_error.real
            / max(float(np.max(np.abs(cached_target.real))), 1.0e-12),
            alpha_error.imag
            / max(float(np.max(np.abs(cached_target.imag))), 1.0e-12),
            np.sqrt(1.2)
            * cached_inverse_error.real
            / max(float(np.max(np.abs(inverse_target.real))), 1.0e-12),
            np.sqrt(1.2)
            * cached_inverse_error.imag
            / max(float(np.max(np.abs(inverse_target.imag))), 1.0e-12),
        ]
        expected_cost = float(
            np.sqrt(np.mean(np.concatenate(residual_parts) ** 2))
        )
        passivity_energies = np.linspace(
            *self.adapter.fit_window_ev,
            self.adapter.fit.passivity_grid_points,
        )
        expected_minimum_imaginary = float(
            np.min(
                self.adapter.alpha_dimless(
                    passivity_energies,
                    response="modal",
                ).imag
            )
        )
        self.assertAlmostEqual(self.adapter.fit.rms_alpha, expected_rms_alpha, places=14)
        self.assertAlmostEqual(
            self.adapter.fit.rms_inv_alpha,
            expected_rms_inverse,
            places=14,
        )
        self.assertAlmostEqual(
            self.adapter.fit.normalized_rms_alpha,
            expected_normalized_rms_alpha,
            places=14,
        )
        self.assertAlmostEqual(
            self.adapter.fit.normalized_rms_inv_alpha,
            expected_normalized_rms_inverse,
            places=14,
        )
        self.assertAlmostEqual(self.adapter.fit.cost, expected_cost, places=14)
        self.assertAlmostEqual(
            self.adapter.fit.max_normalized_alpha_error,
            expected_maximum_relative_error,
            places=14,
        )
        self.assertAlmostEqual(
            self.adapter.fit.min_imag_alpha_fit_window,
            expected_minimum_imaginary,
            places=14,
        )


class SadeghiDensityMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = Sadeghi2009Profile()
        cls.adapter = build_sadeghi_adapter(cls.profile)

    def test_far_distance_steady_state_matches_isolated_analytic_solution(self) -> None:
        omega = self.profile.isolated_rabi_ns_inv(
            self.profile.weak_intensity_w_cm2
        )
        expected_population, expected_coherence = isolated_steady_state(
            omega_ns_inv=omega,
            detuning_ns_inv=0.0,
            gamma1_ns_inv=self.profile.population_decay_ns_inv,
            gamma2_ns_inv=self.profile.coherence_decay_ns_inv,
        )
        state = solve_steady_state(
            self.adapter,
            detuning_mev=0.0,
            distance_nm=1.0e6,
            intensity_w_cm2=self.profile.weak_intensity_w_cm2,
        )

        self.assertAlmostEqual(state.excited_population, expected_population, places=10)
        self.assertAlmostEqual(state.coherence_21.real, expected_coherence.real, places=10)
        self.assertAlmostEqual(state.coherence_21.imag, expected_coherence.imag, places=10)
        closed_form = 2.0 * omega**2 / (
            self.profile.population_decay_ns_inv
            * self.profile.coherence_decay_ns_inv
            + 4.0 * omega**2
        )
        self.assertAlmostEqual(expected_population, closed_form, places=14)

    def test_analytic_steady_jacobian_matches_finite_differences(self) -> None:
        direct_drive = 0.7 + 0.2j
        feedback = 1.1 + 0.4j
        detuning = 3.0
        gamma1 = 1.25
        gamma2 = 2.7
        state = np.asarray([0.2, 0.1, -0.05], dtype=float)

        def rhs(vector: np.ndarray) -> np.ndarray:
            population, x, y = vector
            coherence = complex(x, y)
            omega = direct_drive + feedback * coherence
            coherence_dot = (
                -(gamma2 + 1j * detuning) * coherence
                + 1j * omega * (1.0 - 2.0 * population)
            )
            return np.asarray(
                [
                    2.0 * np.imag(np.conj(omega) * coherence)
                    - gamma1 * population,
                    coherence_dot.real,
                    coherence_dot.imag,
                ]
            )

        analytic = steady_state_jacobian_ns_inv(
            population=float(state[0]),
            coherence_21=complex(state[1], state[2]),
            direct_drive_ns_inv=direct_drive,
            feedback_ns_inv=feedback,
            detuning_ns_inv=detuning,
            gamma1_ns_inv=gamma1,
            gamma2_ns_inv=gamma2,
        )
        step = 1.0e-6
        numerical = np.column_stack(
            [
                (rhs(state + step * np.eye(3)[index])
                 - rhs(state - step * np.eye(3)[index]))
                / (2.0 * step)
                for index in range(3)
            ]
        )
        np.testing.assert_allclose(analytic, numerical, rtol=2.0e-10, atol=2.0e-10)

    def test_weak_resonant_population_is_suppressed_at_short_distance(self) -> None:
        far = solve_steady_state(
            self.adapter,
            detuning_mev=0.0,
            distance_nm=100.0,
            intensity_w_cm2=self.profile.weak_intensity_w_cm2,
        )
        close = solve_steady_state(
            self.adapter,
            detuning_mev=0.0,
            distance_nm=15.0,
            intensity_w_cm2=self.profile.weak_intensity_w_cm2,
        )

        self.assertLess(close.excited_population, 0.1 * far.excited_population)
        for state in (far, close):
            self.assertGreaterEqual(state.excited_population, 0.0)
            self.assertLessEqual(state.excited_population, 0.5)
            self.assertLessEqual(state.bloch_radius, 1.0 + 1.0e-8)
            self.assertTrue(state.locally_stable)
            self.assertLess(state.max_jacobian_real_part_ns_inv, 0.0)
            self.assertAlmostEqual(
                state.negated_physical_excitation_ns_inv,
                -state.physical_excitation_ns_inv,
                places=14,
            )
            self.assertAlmostEqual(
                state.literal_paper_G_ns_inv,
                -2.0
                * np.imag(state.normalized_rabi_ns_inv * state.coherence_21),
                places=14,
            )
            self.assertAlmostEqual(
                state.physical_excitation_ns_inv,
                self.profile.population_decay_ns_inv * state.excited_population,
                places=10,
            )

    def test_switch_ramp_definition_and_time_density_matrix_positivity(self) -> None:
        half_width = 0.5 * self.profile.switch_rise_10_90_ns
        values = logistic_switch_envelope(
            np.asarray(
                [
                    self.profile.switch_center_ns - half_width,
                    self.profile.switch_center_ns,
                    self.profile.switch_center_ns + half_width,
                ]
            ),
            center_ns=self.profile.switch_center_ns,
            rise_10_90_ns=self.profile.switch_rise_10_90_ns,
        )
        np.testing.assert_allclose(values, [0.1, 0.5, 0.9], rtol=0.0, atol=2.0e-15)

        trace = solve_switch_on_trace(
            self.adapter,
            distance_nm=15.0,
            detuning_mev=0.0,
            points=301,
        )
        self.assertTrue(trace.solver_success)
        self.assertLessEqual(trace.max_bloch_radius, 1.0 + 2.0e-6)
        self.assertGreaterEqual(trace.min_density_eigenvalue, -1.0e-6)
        self.assertGreaterEqual(float(np.min(trace.excited_population)), -1.0e-9)
        self.assertLessEqual(float(np.max(trace.excited_population)), 1.0 + 1.0e-9)
        np.testing.assert_allclose(
            trace.negated_physical_excitation_ns_inv,
            -trace.physical_excitation_ns_inv,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            trace.literal_paper_G_ns_inv,
            -2.0
            * np.imag(trace.normalized_rabi_ns_inv * trace.coherence_21),
            rtol=0.0,
            atol=0.0,
        )

    def test_mev_to_inverse_nanoseconds_conversion_is_angular_frequency(self) -> None:
        self.assertAlmostEqual(MEV_TO_NS_INV, 1519.267448, places=6)

    def test_time_solution_converges_when_tolerances_are_tightened(self) -> None:
        coarse = solve_switch_on_trace(
            self.adapter,
            distance_nm=15.0,
            detuning_mev=0.0,
            points=201,
            rtol=2.0e-7,
            atol=2.0e-9,
        )
        fine = solve_switch_on_trace(
            self.adapter,
            distance_nm=15.0,
            detuning_mev=0.0,
            points=201,
            rtol=2.0e-9,
            atol=2.0e-11,
        )

        self.assertLess(
            float(np.max(np.abs(coarse.excited_population - fine.excited_population))),
            2.0e-7,
        )
        self.assertLess(
            float(np.max(np.abs(coarse.normalized_rabi_ns_inv - fine.normalized_rabi_ns_inv))),
            2.0e-5,
        )

    def test_underresolved_production_grid_is_rejected_before_artifacts(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "must_not_exist"
            with self.assertRaisesRegex(ValueError, "under-resolves"):
                run_comparison(
                    output_dir=output,
                    weak_points=801,
                    strong_points=5001,
                    time_points=5,
                    make_plots=False,
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
