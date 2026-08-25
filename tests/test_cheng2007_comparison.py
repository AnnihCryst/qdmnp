"""Physics contracts for the Cheng-2007 native-model geometry adapter."""

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import numpy as np

from qd_mnp_cheng2007_comparison import (
    Cheng2007Profile,
    MAX_MODAL_AUDIT_LOSS_CHANNEL_ERROR,
    MAX_MODAL_AUDIT_RELATIVE_ERROR,
    build_cheng_modal_fit,
    cheng_modal_alpha_au,
    evaluate_rate_point,
    homogeneous_rate_from_internal_dipole_ns_inv,
    longitudinal_alpha_au,
    longitudinal_depolarization_factor,
    modal_poles_au,
    photon_energy_ev,
    radiatively_consistent_dipoles_debye,
    run_comparison,
    zero_feedback_area_population,
    zero_feedback_period_pi_units,
)
from qd_mnp_rational_fit import (
    AU_DIPOLE_C_M,
    AU_TIME_S,
    DEBYE_C_M,
    DEFAULT_AU_MATERIAL,
    eV_to_au,
    homogeneous_radiative_decay_rate_au,
    nm_to_au,
)


class ChengGeometryAndPolarizabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = Cheng2007Profile()

    def test_article_geometry_reconstructs_directional_gaps(self) -> None:
        for geometry, expected_g in (("end", 2.0), ("side", -1.0)):
            distance = self.profile.center_distance_nm(
                aspect_ratio=3.0,
                wavelength_nm=750.0,
                gap_nm=2.0,
                geometry=geometry,
            )
            reconstructed = self.profile.reconstructed_gap_nm(
                aspect_ratio=3.0,
                wavelength_nm=750.0,
                center_distance_nm=distance,
                geometry=geometry,
            )
            self.assertEqual(self.profile.orientation_factor(geometry), expected_g)
            self.assertAlmostEqual(reconstructed, 2.0, places=14)

        self.assertAlmostEqual(
            self.profile.center_distance_nm(
                aspect_ratio=3.0,
                wavelength_nm=750.0,
                gap_nm=2.0,
                geometry="end",
            ),
            26.6,
        )
        self.assertAlmostEqual(
            self.profile.center_distance_nm(
                aspect_ratio=3.0,
                wavelength_nm=750.0,
                gap_nm=2.0,
                geometry="side",
            ),
            14.6,
        )

    def test_longitudinal_response_is_independent_of_qd_position(self) -> None:
        end = evaluate_rate_point(
            self.profile,
            aspect_ratio=3.0,
            wavelength_nm=750.0,
            gap_nm=2.0,
            geometry="end",
            dipole_profile="figure_calibrated",
        )
        side = evaluate_rate_point(
            self.profile,
            aspect_ratio=3.0,
            wavelength_nm=750.0,
            gap_nm=2.0,
            geometry="side",
            dipole_profile="figure_calibrated",
        )

        self.assertEqual(end.alpha_mnp_au, side.alpha_mnp_au)
        self.assertEqual(end.orientation_factor, 2.0)
        self.assertEqual(side.orientation_factor, -1.0)

    def test_sphere_limit_matches_the_core_material_formula(self) -> None:
        wavelength = 750.0
        energy = float(photon_energy_ev(wavelength))
        epsilon_gold = complex(DEFAULT_AU_MATERIAL.epsilon_at(energy))
        contrast = epsilon_gold - self.profile.eps_environment
        radius_au = float(nm_to_au(self.profile.mnp_semiminor_nm))
        expected = (
            self.profile.eps_environment
            * radius_au**3
            / 3.0
            * contrast
            / (self.profile.eps_environment + contrast / 3.0)
        )
        actual = complex(
            longitudinal_alpha_au(
                self.profile,
                aspect_ratio=1.0,
                wavelength_nm=wavelength,
            )
        )

        self.assertAlmostEqual(float(longitudinal_depolarization_factor(1.0)), 1.0 / 3.0)
        near_sphere = float(longitudinal_depolarization_factor(1.0 + 1.0e-10))
        self.assertTrue(np.isfinite(near_sphere))
        self.assertAlmostEqual(near_sphere, 1.0 / 3.0, places=9)
        self.assertAlmostEqual(actual.real / expected.real, 1.0, places=14)
        self.assertAlmostEqual(actual.imag / expected.imag, 1.0, places=14)

    def test_cached_core_modal_audits_are_passive_stable_and_match_carriers(self) -> None:
        for aspect_ratio in (1.0, 3.0, 4.0):
            fit = build_cheng_modal_fit(self.profile, aspect_ratio)
            poles = modal_poles_au(fit)
            direct = np.asarray(
                [
                    longitudinal_alpha_au(
                        self.profile,
                        aspect_ratio=aspect_ratio,
                        wavelength_nm=wavelength,
                    )
                    for wavelength in self.profile.wavelengths_nm
                ],
                dtype=complex,
            )
            modal = np.asarray(
                cheng_modal_alpha_au(
                    self.profile,
                    aspect_ratio=aspect_ratio,
                    wavelength_nm=np.asarray(self.profile.wavelengths_nm),
                    fit=fit,
                ),
                dtype=complex,
            )
            complex_error = np.abs(modal - direct) / np.abs(direct)
            loss_channel_error = (
                np.max(np.abs(modal.imag - direct.imag))
                / np.max(np.abs(direct.imag))
            )

            with self.subTest(aspect_ratio=aspect_ratio):
                self.assertTrue(fit.passive_on_fit_window)
                self.assertTrue(fit.passive_for_all_positive_frequencies)
                self.assertGreaterEqual(float(np.min(fit.strengths_au2)), 0.0)
                self.assertGreater(float(np.min(fit.gamma_modes_au)), 0.0)
                self.assertLess(float(np.max(poles.real)), 0.0)
                self.assertGreaterEqual(float(np.min(modal.imag)), 0.0)
                self.assertLessEqual(
                    float(np.max(complex_error)),
                    MAX_MODAL_AUDIT_RELATIVE_ERROR,
                )
                self.assertLessEqual(
                    float(loss_channel_error),
                    MAX_MODAL_AUDIT_LOSS_CHANNEL_ERROR,
                )

    def test_johnson_christy_longitudinal_alpha_is_passive_on_figure_grid(self) -> None:
        aspect_ratios = np.linspace(1.0, 9.0, 401)
        for wavelength in self.profile.wavelengths_nm:
            with self.subTest(wavelength_nm=wavelength):
                alpha = longitudinal_alpha_au(
                    self.profile,
                    aspect_ratio=aspect_ratios,
                    wavelength_nm=wavelength,
                )
                self.assertGreaterEqual(float(np.min(alpha.imag)), 0.0)

    def test_large_q_sweep_records_quasistatic_retardation_warning_parameter(self) -> None:
        point = evaluate_rate_point(
            self.profile,
            aspect_ratio=9.0,
            wavelength_nm=600.0,
            gap_nm=2.0,
            geometry="side",
        )

        self.assertGreater(point.host_wavenumber_times_semimajor, 0.8)
        self.assertLess(point.host_wavenumber_times_semimajor, 0.9)

    def test_profile_records_missing_constants_and_material_mismatch(self) -> None:
        provenance = self.profile.provenance()

        self.assertIn("numerical QD transition dipole", provenance["missing_from_article"])
        self.assertIn(
            "Figure 4 pulse amplitude, duration and temporal shape",
            provenance["missing_from_article"],
        )
        self.assertEqual(
            provenance["project_model_choices"]["gold_data"],
            "Johnson-Christy table bundled with the project",
        )
        self.assertEqual(provenance["project_model_choices"]["article_gold_data"], "Palik")


class ChengRateAndZeroFeedbackProxyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = Cheng2007Profile()

    def test_radiative_dipole_profile_reconstructs_gamma_rad_zero(self) -> None:
        for wavelength in self.profile.wavelengths_nm:
            external_debye, internal_debye = radiatively_consistent_dipoles_debye(
                self.profile,
                wavelength,
            )
            external_au = external_debye * DEBYE_C_M / AU_DIPOLE_C_M
            gamma_au = homogeneous_radiative_decay_rate_au(
                external_au,
                float(eV_to_au(photon_energy_ev(wavelength))),
                self.profile.eps_environment,
            )
            gamma_ns_inv = gamma_au / AU_TIME_S * 1.0e-9

            self.assertAlmostEqual(
                gamma_ns_inv,
                self.profile.isolated_radiative_rate_ns_inv,
                places=13,
            )
            self.assertAlmostEqual(
                external_debye,
                internal_debye * self.profile.qd_local_field_factor,
                places=13,
            )

    def test_rates_are_nonnegative_and_cp_coherence_rate_is_gamma1_over_two(self) -> None:
        for geometry in ("end", "side"):
            for dipole_profile in ("radiative_consistent", "figure_calibrated"):
                point = evaluate_rate_point(
                    self.profile,
                    aspect_ratio=3.2,
                    wavelength_nm=750.0,
                    gap_nm=2.0,
                    geometry=geometry,
                    dipole_profile=dipole_profile,
                )
                with self.subTest(geometry=geometry, dipole_profile=dipole_profile):
                    self.assertGreaterEqual(point.alpha_mnp_au.imag, 0.0)
                    self.assertGreaterEqual(point.gamma_radiative_cheng_ns_inv, 0.0)
                    self.assertGreaterEqual(
                        point.radiative_rate_from_native_field_ratio_estimate_ns_inv,
                        0.0,
                    )
                    self.assertGreaterEqual(point.gamma_nonradiative_ns_inv, 0.0)
                    self.assertAlmostEqual(
                        point.gamma2_cp_ns_inv,
                        0.5 * point.gamma1_with_zeta_ns_inv,
                        places=14,
                    )

    def test_large_distance_reduces_to_isolated_qd(self) -> None:
        point = evaluate_rate_point(
            self.profile,
            aspect_ratio=3.0,
            wavelength_nm=750.0,
            gap_nm=1.0e6,
            geometry="end",
            dipole_profile="figure_calibrated",
        )

        self.assertAlmostEqual(
            point.gamma_radiative_cheng_ns_inv,
            self.profile.isolated_radiative_rate_ns_inv,
            places=12,
        )
        self.assertAlmostEqual(
            point.radiative_rate_from_native_field_ratio_estimate_ns_inv,
            self.profile.isolated_radiative_rate_ns_inv,
            places=12,
        )
        self.assertLess(point.gamma_nonradiative_ns_inv, 1.0e-20)

    def test_nonradiative_feedback_scales_exactly_as_distance_minus_six(self) -> None:
        near = evaluate_rate_point(
            self.profile,
            aspect_ratio=2.5,
            wavelength_nm=750.0,
            gap_nm=2.0,
            geometry="end",
            dipole_profile="figure_calibrated",
        )
        far = evaluate_rate_point(
            self.profile,
            aspect_ratio=2.5,
            wavelength_nm=750.0,
            gap_nm=20.0,
            geometry="end",
            dipole_profile="figure_calibrated",
        )
        expected = (far.center_distance_nm / near.center_distance_nm) ** 6

        self.assertAlmostEqual(
            near.gamma_nonradiative_ns_inv / far.gamma_nonradiative_ns_inv,
            expected,
            places=12,
        )
        self.assertAlmostEqual(
            abs(near.feedback_ns_inv) / abs(far.feedback_ns_inv),
            expected,
            places=12,
        )

    def test_figure_calibrated_profile_recovers_cheng_figure_3_scale(self) -> None:
        aspect_ratios = np.linspace(2.9, 3.5, 241)
        peaks: dict[str, tuple[float, float]] = {}
        for geometry in ("end", "side"):
            rates = np.asarray(
                [
                    evaluate_rate_point(
                        self.profile,
                        aspect_ratio=float(q),
                        wavelength_nm=750.0,
                        gap_nm=2.0,
                        geometry=geometry,
                        dipole_profile="figure_calibrated",
                    ).gamma_nonradiative_ns_inv
                    for q in aspect_ratios
                ]
            )
            peak_index = int(np.argmax(rates))
            peaks[geometry] = (float(aspect_ratios[peak_index]), float(rates[peak_index]))

        self.assertGreater(peaks["end"][0], 3.1)
        self.assertLess(peaks["end"][0], 3.3)
        self.assertGreater(peaks["end"][1], 0.19)
        self.assertLess(peaks["end"][1], 0.24)
        self.assertGreater(peaks["side"][0], 3.1)
        self.assertLess(peaks["side"][0], 3.3)
        self.assertGreater(peaks["side"][1], 2.5)
        self.assertLess(peaks["side"][1], 2.7)

    def test_figure_calibrated_dipole_is_not_radiatively_self_consistent(self) -> None:
        rate = homogeneous_rate_from_internal_dipole_ns_inv(
            self.profile,
            750.0,
            self.profile.calibrated_internal_dipole_debye,
        )

        self.assertTrue(np.isclose(rate, 8.0e-4, rtol=1.0e-3, atol=0.0))
        self.assertTrue(np.isclose(
            rate / self.profile.isolated_radiative_rate_ns_inv,
            0.01,
            rtol=1.0e-3,
            atol=0.0,
        ))

    def test_zero_feedback_area_proxy_has_the_exported_period(self) -> None:
        point = evaluate_rate_point(
            self.profile,
            aspect_ratio=1.0,
            wavelength_nm=750.0,
            gap_nm=6.0,
            geometry="end",
            dipole_profile="figure_calibrated",
        )
        period_pi = zero_feedback_period_pi_units(point.cheng_field_factor)
        theta = np.asarray([0.0, period_pi * np.pi, 0.5 * period_pi * np.pi])
        population = zero_feedback_area_population(theta, point.cheng_field_factor)

        np.testing.assert_allclose(population[:2], [0.0, 0.0], atol=2.0e-30)
        self.assertAlmostEqual(population[2], 1.0, places=14)
        self.assertAlmostEqual(period_pi, 1.9145637657284418, places=12)

    def test_smoke_run_writes_provenance_and_both_field_conventions(self) -> None:
        with TemporaryDirectory() as directory:
            stale_plot = Path(directory) / "figure_2_radiative_rates.png"
            stale_plot.write_bytes(b"stale plot from an earlier run")
            run_dir = run_comparison(
                output_dir=directory,
                q_points=21,
                theta_points=21,
                dipole_profiles=("figure_calibrated",),
                make_plots=False,
            )
            metadata_path = Path(run_dir) / "metadata.json"
            rates_path = Path(run_dir) / "figures_2_3_rates.csv"
            modal_audit_path = Path(run_dir) / "modal_audit.csv"
            area_path = Path(run_dir) / "figure_4_zero_feedback_area_proxy.csv"
            periods_path = Path(run_dir) / "figure_4_zero_feedback_periods.csv"

            self.assertTrue(metadata_path.exists())
            self.assertTrue(rates_path.exists())
            self.assertTrue(modal_audit_path.exists())
            self.assertTrue(area_path.exists())
            self.assertTrue(periods_path.exists())
            self.assertFalse(stale_plot.exists())
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertFalse(metadata["mathematical_core_modified"])
            self.assertTrue(
                metadata["geometry_adapter"]["longitudinal_alpha_for_both_positions"]
            )
            self.assertTrue(metadata["local_field_conventions"]["both_are_exported"])
            self.assertEqual(
                metadata["figure_4_scope"]["calculated"],
                "strong-drive zero-feedback area-theorem proxy",
            )
            self.assertTrue(
                metadata["modal_audit"]["all_carrier_samples_accepted"]
            )
            self.assertEqual(metadata["numerical_settings"]["q_points"], 21)
            self.assertEqual(metadata["numerical_settings"]["theta_points"], 21)
            self.assertEqual(
                metadata["numerical_settings"]["figures_2_3_q_interval"],
                [1.0, 9.0],
            )
            self.assertFalse(
                metadata["numerical_settings"]["refit_modal_audit"]
            )
            self.assertFalse(metadata["numerical_settings"]["make_plots"])
            self.assertIsNone(
                metadata["numerical_settings"]["resolved_plot_dipole_profile"]
            )
            calibrated = metadata[
                "figure_calibrated_dipole_consistency_diagnostic"
            ]["750"]
            self.assertTrue(np.isclose(
                calibrated["ratio_to_article_gamma_rad0"],
                0.01,
                rtol=1.0e-3,
                atol=0.0,
            ))


if __name__ == "__main__":
    unittest.main()
