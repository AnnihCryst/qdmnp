"""Fast physics contracts for the native Shah-2013 dimer comparison.

These tests deliberately use the cached passive N=10 modal fit.  They validate
the geometry adapter and the reduced symmetric two-MNP equations without
rerunning the nonlinear fit or producing the article-comparison figures.
"""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from qd_mnp_rational_fit import (
    HybridQDPlasmonModel,
    au_to_eV,
    au_to_nm,
    eV_to_au,
    field_au_to_si,
)
from qd_mnp_shah2013_comparison import (
    Shah2013Profile,
    build_native_dimer_model,
    build_shah_params,
    pulse_envelope_spectral_alpha_au,
    pulse_from_fluence,
    run_comparison,
    solve_cw_harmonic_state,
    solve_pulse_envelope_trace,
)


class ShahGeometryAndRateTests(unittest.TestCase):
    def test_article_geometry_has_a_centered_qd_and_positive_one_nm_gaps(self) -> None:
        profile = Shah2013Profile()
        model = build_native_dimer_model(profile)

        self.assertAlmostEqual(profile.mnp_center_distance_nm, 36.0, places=14)
        self.assertAlmostEqual(profile.qd_to_mnp_center_nm, 18.0, places=14)
        self.assertAlmostEqual(
            profile.mnp_center_distance_nm - 2.0 * profile.mnp_semimajor_nm,
            6.0,
            places=14,
        )
        self.assertAlmostEqual(profile.qd_mnp_surface_gap_nm, 1.0, places=14)
        self.assertAlmostEqual(
            float(au_to_nm(model.params.axial_surface_gap_au)),
            1.0,
            places=13,
        )

        with self.assertRaisesRegex(ValueError, "strictly positive surface gap"):
            Shah2013Profile(qd_radius_nm=3.0)

    def test_article_fluences_reconstruct_in_the_dielectric_host(self) -> None:
        profile = Shah2013Profile()
        expected_peak_fields_v_m = np.asarray(
            [
                1.08614686e6,
                3.43469795e6,
                1.08614686e7,
                3.43469795e7,
            ]
        )
        actual_peak_fields_v_m = []

        for fluence in profile.pulse_fluences_j_cm2:
            pulse = pulse_from_fluence(profile, fluence)
            self.assertAlmostEqual(
                pulse.fluence_j_cm2(eps_m=profile.eps_environment) / fluence,
                1.0,
                places=13,
            )
            actual_peak_fields_v_m.append(float(field_au_to_si(pulse.E0_au)))

        np.testing.assert_allclose(
            actual_peak_fields_v_m,
            expected_peak_fields_v_m,
            rtol=1.0e-8,
            atol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(actual_peak_fields_v_m[1:])
            / np.asarray(actual_peak_fields_v_m[:-1]),
            np.sqrt(10.0),
            rtol=2.0e-14,
            atol=0.0,
        )

    def test_supported_rate_profiles_are_completely_positive(self) -> None:
        profile = Shah2013Profile()

        for decay_profile in ("naive", "purcell_cp"):
            with self.subTest(decay_profile=decay_profile):
                gamma1_mev, gamma2_mev = profile.rates_mev(decay_profile)
                params = build_shah_params(profile, decay_profile)

                self.assertGreaterEqual(gamma2_mev, 0.5 * gamma1_mev)
                self.assertAlmostEqual(
                    gamma2_mev - 0.5 * gamma1_mev,
                    profile.qd_thermal_dephasing_mev,
                    places=14,
                )
                self.assertAlmostEqual(
                    float(au_to_eV(params.gamma_au)) * 1000.0,
                    gamma1_mev,
                    places=13,
                )
                self.assertAlmostEqual(
                    float(au_to_eV(params.pure_dephasing_au)) * 1000.0,
                    profile.qd_thermal_dephasing_mev,
                    places=13,
                )

    def test_raw_corrected_shah_rate_pair_is_rejected_by_the_core(self) -> None:
        profile = Shah2013Profile()
        params = build_shah_params(profile, "purcell_cp")

        # Shah's corrected semiclassical table pairs the Purcell-enhanced
        # gamma1 with Gamma2=1.265 meV.  Interpreting that number as the *total*
        # coherence decay gives Gamma2 < gamma1/2 and is not a positive
        # two-level Lindblad generator.
        self.assertLess(
            profile.qd_thermal_dephasing_mev,
            0.5 * profile.effective_population_decay_mev,
        )
        raw_article_pair = replace(
            params,
            Gamma_au=float(eV_to_au(profile.qd_thermal_dephasing_mev / 1000.0)),
        )

        with self.assertRaisesRegex(ValueError, "Gamma2.*gamma1/2"):
            HybridQDPlasmonModel(
                raw_article_pair,
                orientation="long",
                n_modes=1,
                radiative_consistency_policy="ignore",
                verbose=False,
            )


class SymmetricDimerReductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = Shah2013Profile()
        cls.model = build_native_dimer_model(cls.profile)

    def test_frequency_response_matches_explicit_two_dipole_solve(self) -> None:
        energies_ev = np.asarray([1.96, 2.042, 2.12])
        single = self.model.C * self.model.alpha_single_dimless(energies_ev)
        reduced = self.model.alpha_bare_dimer_au(energies_ev)
        explicit = np.empty_like(reduced)

        for index, alpha_single in enumerate(single):
            matrix = np.asarray(
                [
                    [1.0, -self.model.K * alpha_single],
                    [-self.model.K * alpha_single, 1.0],
                ],
                dtype=complex,
            )
            dipoles = np.linalg.solve(
                matrix,
                np.asarray([alpha_single, alpha_single], dtype=complex),
            )
            explicit[index] = np.sum(dipoles)

        np.testing.assert_allclose(reduced, explicit, rtol=5.0e-15, atol=1.0e-7)

    def test_zero_mnp_coupling_reduces_to_two_independent_particles(self) -> None:
        energies_ev = np.linspace(1.94, 2.14, 17)
        expected = 2.0 * self.model.C * self.model.alpha_single_dimless(energies_ev)

        with patch.object(self.model, "K", 0.0):
            uncoupled = self.model.alpha_bare_dimer_au(energies_ev)

        np.testing.assert_allclose(uncoupled, expected, rtol=0.0, atol=0.0)

    def test_cached_modal_dimer_is_passive_and_both_symmetry_branches_are_stable(
        self,
    ) -> None:
        energies_ev = np.linspace(1.94, 2.14, 401)

        self.assertTrue(self.model.fit.passive_for_all_positive_frequencies)
        self.assertTrue(np.all(self.model.fit.strengths_au2 >= 0.0))
        self.assertTrue(self.model.stability.symmetric_stable)
        self.assertTrue(self.model.stability.antisymmetric_stable)
        self.assertLessEqual(
            float(np.max(self.model.stability.symmetric_poles_au.real)),
            self.model.stability.tolerance_au,
        )
        self.assertLessEqual(
            float(np.max(self.model.stability.antisymmetric_poles_au.real)),
            self.model.stability.tolerance_au,
        )

        for response in ("modal", "material"):
            with self.subTest(response=response):
                single = self.model.alpha_single_dimless(
                    energies_ev,
                    response=response,
                )
                bare_dimer = self.model.alpha_bare_dimer_au(
                    energies_ev,
                    response=response,
                )
                coupled = self.model.linear_coupled_alpha_au(
                    energies_ev,
                    response=response,
                )[0]
                self.assertGreaterEqual(float(np.min(single.imag)), -1.0e-14)
                self.assertGreaterEqual(float(np.min(bare_dimer.imag)), -1.0e-7)
                self.assertGreaterEqual(float(np.min(coupled.imag)), -1.0e-7)

    def test_weak_cw_solution_matches_the_analytic_linear_response(self) -> None:
        energy_ev = self.profile.transition_energy_ev
        state = solve_cw_harmonic_state(
            self.model,
            energy_ev=energy_ev,
            intensity_w_cm2=1.0e-6,
        )
        linear_alpha = self.model.linear_coupled_alpha_au(
            np.asarray([energy_ev])
        )[0][0]

        relative_error = abs(state.effective_alpha_au - linear_alpha) / abs(
            linear_alpha
        )
        self.assertLess(relative_error, 1.0e-9)
        self.assertAlmostEqual(state.inversion, -1.0, places=10)
        self.assertLessEqual(state.max_periodic_bloch_radius, 1.0 + 1.0e-10)


class ShahPulseEnvelopeTests(unittest.TestCase):
    def test_weak_rwa_pulse_is_physical_converged_and_matches_linear_spectrum(
        self,
    ) -> None:
        profile = Shah2013Profile()
        model = build_native_dimer_model(profile)
        # This is three orders of magnitude below the smallest article fluence,
        # so the remaining spectrum mismatch measures the rotating-envelope
        # and finite-time approximations rather than nonlinear saturation.
        trace = solve_pulse_envelope_trace(model, profile, 5.0e-12)

        self.assertLessEqual(trace.max_bloch_radius, 1.0 + 1.0e-10)
        self.assertGreaterEqual(trace.min_density_eigenvalue, -1.0e-10)
        self.assertLess(trace.tail_ratio, 5.0e-4)
        self.assertTrue(np.all(np.isfinite(trace.state)))
        self.assertGreaterEqual(float(np.min(trace.inversion)), -1.0 - 1.0e-10)
        self.assertLessEqual(float(np.max(trace.inversion)), 1.0 + 1.0e-10)

        energies_ev = np.linspace(1.94, 2.14, 101)
        pulse_alpha = pulse_envelope_spectral_alpha_au(
            trace,
            profile,
            energies_ev,
        )
        linear_alpha = model.linear_coupled_alpha_au(energies_ev)[0]
        error = np.abs(pulse_alpha - linear_alpha)
        scale = float(np.max(np.abs(linear_alpha)))
        normalized_max_error = float(np.nanmax(error) / scale)
        normalized_rms_error = float(
            np.sqrt(np.nanmean(error**2))
            / np.sqrt(np.mean(np.abs(linear_alpha) ** 2))
        )

        # Measured values for the current strict first-order RWA are about
        # 2.44% and 1.01%, respectively.  The gates leave numerical headroom
        # but would catch a Fourier-sign error or broken dimer/local-field coupling.
        self.assertLess(normalized_max_error, 0.03)
        self.assertLess(normalized_rms_error, 0.012)
        self.assertGreaterEqual(float(np.nanmin(pulse_alpha.imag)), -1.0e-7)


class ShahComparisonGridTests(unittest.TestCase):
    def test_underresolved_energy_grid_is_rejected_before_calculations(self) -> None:
        profile = Shah2013Profile()
        with TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory) / "comparison"
            with self.assertRaisesRegex(ValueError, "resolve Gamma2/4"):
                run_comparison(
                    profile=profile,
                    output_dir=output_directory,
                    article_pdf=Path(temporary_directory) / "missing.pdf",
                    decay_profile="purcell_cp",
                    response="modal",
                    refit=False,
                    energy_points=101,
                    intensity_points=9,
                    fluence_points=9,
                    verbose=False,
                )
            self.assertFalse(output_directory.exists())


if __name__ == "__main__":
    unittest.main()
