"""Physical and time-domain regressions for the McMillan Fig. 4(c) benchmark.

The article-matched run deliberately uses one spatial order (the spherical
dipole limit) and 80 passive Lorentz material poles.  Consequently, these
tests validate the material-memory/Bloch part of the full-QS time core in the
model intersection with McMillan et al.; they do not claim convergence of the
higher spatial multipoles.
"""

import unittest

import numpy as np

from literature_reproductions.mcmillan2016_common import (
    FIT_WINDOW_EV,
    au_to_fs,
    build_paper_matched_model,
    etchegoin_gold_epsilon,
    fit_passive_lorentz_sphere,
    make_etchegoin_material,
    make_paper_params,
    make_paper_pulse,
    paper_time_fs,
)


class McMillan2016Fig4CFullQSTests(unittest.TestCase):
    """One shared fit and propagation keep the literature regression quick."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.material = make_etchegoin_material()
        cls.fit, cls.fit_audit = fit_passive_lorentz_sphere(
            selected_poles=80,
            fit_points=1001,
        )
        cls.params = {
            separation: make_paper_params(separation, material=cls.material)
            for separation in (80.0, 20.0, 13.0)
        }
        cls.profiles = {
            separation: make_paper_pulse(params)
            for separation, params in cls.params.items()
        }

        cls.model = build_paper_matched_model(
            cls.params[13.0],
            cls.fit,
            spatial_orders=1,
        )
        profile = cls.profiles[13.0]
        cls.result = cls.model.solve(
            profile.pulse,
            t_span_au=(-0.5 * profile.duration_au, 0.5 * profile.duration_au),
            method="DOP853",
            rtol=3.0e-8,
            atol=1.0e-10,
            points_per_fastest_cycle=20.0,
            spectral_window_policy="raise",
            max_spectral_leakage=0.01,
            positivity_policy="raise",
            positivity_tolerance=1.0e-7,
            work_passivity_policy="warn",
            # The article stops at T although T1 and T2 are hundreds of ps/ns,
            # so a decayed-response tail is not expected at this endpoint.
            response_tail_policy="ignore",
        )
        cls.paper_time_fs = paper_time_fs(cls.result.t_au, profile)

    def test_etchegoin_gold_at_the_qd_transition_energy(self) -> None:
        epsilon = complex(etchegoin_gold_epsilon(2.5))
        self.assertAlmostEqual(epsilon.real, -2.4301391455946004, places=12)
        self.assertAlmostEqual(epsilon.imag, 3.3553414844634740, places=12)
        self.assertGreater(epsilon.imag, 0.0)
        self.assertAlmostEqual(
            complex(self.material.epsilon_at(2.5)).real,
            epsilon.real,
            places=14,
        )
        self.assertAlmostEqual(
            complex(self.material.epsilon_at(2.5)).imag,
            epsilon.imag,
            places=14,
        )

    def test_cdse_screening_and_article_pulse_mapping(self) -> None:
        for params in self.params.values():
            # For a spherical CdSe QD with eps_s=6, the internal-field factor
            # is 3/(eps_s+2)=3/8 and multiplies the bare 0.65 e*nm dipole.
            self.assertEqual(params.qd_dipole_convention, "bare_internal")
            self.assertAlmostEqual(params.qd_local_field_factor, 3.0 / 8.0)

        expected_amplitudes = {
            80.0: 0.023764633370430357,
            20.0: 0.021234471937599978,
            13.0: 0.016231561562524040,
        }
        expected_local_amplitude_factors = {
            80.0: 1.0018348303509006,
            20.0: 1.1212069464679786,
            13.0: 1.4667866273560326,
        }
        for separation, profile in self.profiles.items():
            self.assertEqual(profile.cycles, 10)
            self.assertEqual(profile.area_pi, 5.0)
            self.assertAlmostEqual(
                float(au_to_fs(profile.duration_au)),
                33.08534157539163,
                places=11,
            )
            self.assertAlmostEqual(
                float(au_to_fs(profile.tau_p_au)),
                1.102844719179721,
                places=12,
            )
            self.assertAlmostEqual(
                profile.local_field_amplitude_factor,
                expected_local_amplitude_factors[separation],
                places=12,
            )
            self.assertAlmostEqual(
                profile.pulse.E0_au,
                expected_amplitudes[separation],
                places=12,
            )
            self.assertLess(
                profile.pulse.spectral_leakage_fraction(FIT_WINDOW_EV),
                1.0e-6,
            )

    def test_80_pole_passive_fit_meets_the_broadband_quality_gate(self) -> None:
        audit = self.fit_audit
        self.assertTrue(audit.accepted)
        self.assertEqual(audit.selected_poles, 80)
        self.assertEqual(audit.dictionary_poles, 693)
        self.assertEqual(self.fit.strengths_au2.size, 80)
        self.assertTrue(np.all(self.fit.strengths_au2 >= 0.0))
        self.assertTrue(self.fit.passive_on_fit_window)
        self.assertTrue(self.fit.passive_for_all_positive_frequencies)
        self.assertGreater(audit.normalized_rms_alpha, 0.0090)
        self.assertLess(audit.normalized_rms_alpha, 0.0100)
        self.assertGreater(audit.normalized_rms_inverse_alpha, 0.0060)
        self.assertLess(audit.normalized_rms_inverse_alpha, 0.0070)
        self.assertGreater(audit.max_pointwise_relative_alpha_error, 0.020)
        self.assertLess(audit.max_pointwise_relative_alpha_error, 0.024)
        self.assertGreaterEqual(audit.min_imaginary_alpha, 0.0)

    def test_r13_time_domain_population_positivity_and_bandwidth_regression(self) -> None:
        diagnostics = self.result.diagnostics
        self.assertTrue(diagnostics.solver_success, diagnostics.solver_message)
        self.assertTrue(diagnostics.t_final_reached)
        self.assertTrue(diagnostics.state_is_finite)
        self.assertEqual(self.model.n_spatial_modes, 1)
        self.assertEqual(self.model.n_material_modes, 80)
        self.assertEqual(self.model.state_size, 164)

        post_pulse = self.paper_time_fs >= 18.0
        post_indices = np.flatnonzero(post_pulse)
        peak_index = int(
            post_indices[np.argmax(self.result.rho22[post_pulse])]
        )
        peak_population = float(self.result.rho22[peak_index])
        peak_time_fs = float(self.paper_time_fs[peak_index])

        self.assertAlmostEqual(float(self.result.rho22[-1]), 0.591437773, delta=0.002)
        self.assertAlmostEqual(peak_population, 0.942134643, delta=0.002)
        self.assertAlmostEqual(peak_time_fs, 20.8922463, delta=0.15)
        self.assertGreaterEqual(float(np.min(self.result.rho22)), -1.0e-12)
        self.assertLessEqual(float(np.max(self.result.rho22)), 1.0 + 1.0e-12)
        self.assertGreaterEqual(diagnostics.min_density_eigenvalue, -1.0e-12)

        self.assertLess(diagnostics.pulse_spectral_leakage, 1.0e-6)
        # The nonlinear QD trajectory has a small, explicitly accepted tail
        # outside the article's 0.01--10 eV material-fit interval.
        self.assertGreater(diagnostics.qd_source_spectral_leakage, 0.0030)
        self.assertLess(diagnostics.qd_source_spectral_leakage, 0.0037)
        for leakage in (
            diagnostics.mnp_drive_spectral_leakage,
            diagnostics.mnp_dipole_spectral_leakage,
            diagnostics.mnp_field_spectral_leakage,
        ):
            self.assertGreaterEqual(leakage, 0.0)
            self.assertLess(leakage, 3.0e-6)


if __name__ == "__main__":
    unittest.main()
