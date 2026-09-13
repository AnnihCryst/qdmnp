"""Regression tests for common article spectral measurements and sampling."""

import unittest

import numpy as np

from article_observables.qd_mnp_gap_metrics_common import extract_feature
from article_observables.qd_mnp_calculate_excitation_spectrum_material_comparison import extract_fwhm_feature
from article_observables.qd_mnp_spectral_features import sampling_diagnostics


def lorentzian(energy, center=2.0, hwhm=0.02):
    return 1 / (1 + ((energy - center) / hwhm) ** 2)


class SharedSpectralFeatureTests(unittest.TestCase):
    def setUp(self):
        self.energy = np.linspace(1.8, 2.2, 4001)
        self.kw = dict(center_eV=2.0, half_window_eV=0.15)

    def test_both_article_paths_use_identical_baseline_invariant_width(self):
        spectrum = lorentzian(self.energy, 2.01)
        bare = extract_feature(self.energy, spectrum, **self.kw)
        raised = extract_feature(self.energy, spectrum + 0.6, **self.kw)
        material = extract_fwhm_feature(self.energy, spectrum + 0.6, **self.kw)
        self.assertEqual(raised.status, "ok")
        self.assertAlmostEqual(bare.width_eV, raised.width_eV, places=13)
        self.assertEqual(raised.width_eV, material["fwhm_eV"])
        self.assertEqual(raised.energy_eV, material["energy_eV"])

    def test_nearly_clipped_peak_never_becomes_artificial_narrow_line(self):
        spectrum = lorentzian(self.energy, 2.1495)
        for feature in (
            extract_feature(self.energy, spectrum, **self.kw),
            extract_feature(self.energy, spectrum + 0.6, **self.kw),
        ):
            self.assertEqual(feature.status, "edge_truncated")
            self.assertTrue(np.isnan(feature.width_eV))
        material = extract_fwhm_feature(self.energy, spectrum, **self.kw)
        self.assertEqual(material["status"], "edge_truncated")
        self.assertTrue(np.isnan(material["fwhm_eV"]))

    def test_nearest_credible_peak_selected_even_when_another_is_higher(self):
        spectrum = lorentzian(self.energy, 2.01, 0.003) + 3 * lorentzian(self.energy, 2.08, 0.003)
        feature = extract_fwhm_feature(self.energy, spectrum, **self.kw)
        self.assertAlmostEqual(feature["energy_eV"], 2.01, delta=0.0002)
        self.assertEqual(feature["status"], "split_or_ambiguous")
        for key in ("fwhm_eV", "left_eV", "right_eV"):
            self.assertTrue(np.isnan(feature[key]))

    def test_sampling_checks_narrow_hybrid_even_when_isolated_is_resolved(self):
        energy = np.linspace(1.8, 2.2, 2001)
        isolated = sampling_diagnostics(energy, lorentzian(energy), **self.kw)
        narrow = sampling_diagnostics(energy, lorentzian(energy, hwhm=0.0008), **self.kw)
        self.assertTrue(isolated["accepted"])
        self.assertFalse(narrow["accepted"])
        self.assertGreater(narrow["step_over_component_width"], 0.05)
        refined_energy = np.linspace(1.8, 2.2, 16001)
        refined = sampling_diagnostics(refined_energy, lorentzian(refined_energy, hwhm=0.0008), **self.kw)
        self.assertTrue(refined["accepted"])

    def test_resolved_split_can_pass_sampling_without_single_width(self):
        spectrum = lorentzian(self.energy, 1.97, 0.006) + lorentzian(self.energy, 2.03, 0.006)
        self.assertTrue(sampling_diagnostics(self.energy, spectrum, **self.kw)["accepted"])
        self.assertTrue(np.isnan(extract_feature(self.energy, spectrum, **self.kw).width_eV))

    def test_clipped_feature_cannot_receive_sampling_certificate(self):
        spectrum = lorentzian(self.energy, 2.1495)
        self.assertFalse(sampling_diagnostics(self.energy, spectrum, **self.kw)["accepted"])

    def test_symmetric_clipping_fails_window_gate_despite_dense_sampling(self):
        energy = np.linspace(1.999, 2.001, 2001)
        for baseline in (0.0, 0.6):
            for half_window in (0.001 + 1e-14, 0.15):
                check = sampling_diagnostics(
                    energy, lorentzian(energy) + baseline,
                    center_eV=2.0, half_window_eV=half_window,
                )
                self.assertLess(check["step_over_component_width"], 0.05)
                self.assertFalse(check["window_accepted"])
                self.assertFalse(check["accepted"])
                self.assertGreater(check["window_relative_change"], 0.05)
        # Expanding the actual sampled window resolves the same physical line.
        check = sampling_diagnostics(self.energy, lorentzian(self.energy), **self.kw)
        self.assertTrue(check["window_accepted"])
        self.assertTrue(check["accepted"])


if __name__ == "__main__":
    unittest.main()
