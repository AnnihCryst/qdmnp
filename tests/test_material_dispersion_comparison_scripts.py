"""Tests for the direct/one/multi material-dispersion article workflow."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")
import numpy as np

from article_observables import qd_mnp_calculate_material_dispersion_comparison as calc
from article_observables import qd_mnp_plot_material_dispersion_comparison as plot
from qd_mnp_rational_fit import AU_ENERGY_EV


class MaterialDispersionMetricTests(unittest.TestCase):
    def test_common_scale_residual_metrics_match_definition(self) -> None:
        reference = np.asarray([1.0 + 2.0j, 2.0 - 1.0j, -0.5 + 0.2j])
        candidate = reference + np.asarray([0.1j, -0.2, 0.05 + 0.02j])
        residual, normalized, absolute, nrms, maximum = (
            calc._robust_residual_metrics(candidate, reference)
        )
        expected_residual = candidate - reference
        peak = np.max(np.abs(reference))
        expected_nrms = np.sqrt(np.mean(np.abs(expected_residual) ** 2)) / np.sqrt(
            np.mean(np.abs(reference) ** 2)
        )
        np.testing.assert_allclose(residual, expected_residual)
        np.testing.assert_allclose(normalized, expected_residual / peak)
        np.testing.assert_allclose(absolute, np.abs(expected_residual) / peak)
        self.assertAlmostEqual(nrms, float(expected_nrms), places=14)
        self.assertAlmostEqual(maximum, float(np.max(np.abs(expected_residual)) / peak))

    def test_residual_normalization_remains_finite_for_zero_reference(self) -> None:
        outputs = calc._robust_residual_metrics(
            np.asarray([1.0 + 0.0j, 0.0 + 1.0j]),
            np.zeros(2, dtype=complex),
        )
        for output in outputs:
            self.assertTrue(np.all(np.isfinite(output)))

    def test_lspr_fwhm_is_interpolated_on_nearest_crossings(self) -> None:
        energy = np.linspace(0.5, 3.5, 3001)
        sigma = 0.2
        signal = 0.3 + 4.0 * np.exp(-0.5 * ((energy - 2.0) / sigma) ** 2)
        metrics = calc._lspr_peak_metrics(energy, signal)
        expected = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma
        self.assertEqual(metrics["status"], "ok")
        self.assertAlmostEqual(float(metrics["peak_energy_eV"]), 2.0, places=12)
        self.assertAlmostEqual(float(metrics["fwhm_eV"]), expected, places=5)


class MaterialDispersionWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temporary.name)
        cls.artifact = cls.directory / "material_comparison.npz"
        calc.main(
            [
                "--output",
                str(cls.artifact),
                "--energy-points",
                "81",
            ]
        )
        cls.data = plot.load_material_dispersion_artifact(cls.artifact)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_artifact_is_pickle_free_self_contained_and_versioned(self) -> None:
        with np.load(self.artifact, allow_pickle=False) as archive:
            self.assertNotIn("object", {array.dtype.name for array in archive.values()})
            self.assertTrue(all(not archive[key].dtype.hasobject for key in archive.files))
            metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
            self.assertEqual(
                metadata["schema_name"], "qd_mnp.material_dispersion_comparison"
            )
            self.assertEqual(metadata["schema_version"], 1)
            self.assertEqual(metadata["resolved_inputs"]["mode_counts"], {"one": 1, "multi": 9})
            self.assertIn("requested_inputs", metadata)
            self.assertIn("constants", metadata)
            self.assertIn("source_sha256", metadata["provenance"])
            for key in (
                "material_energy_eV",
                "material_n",
                "material_k",
                "alpha_complex_au3",
                "inverse_alpha_complex_au_minus3",
                "fit_strengths_au2",
                "fit_omega_modes_au",
                "fit_gamma_modes_au",
            ):
                self.assertIn(key, archive.files)

    def test_shapes_common_grid_errors_and_fit_certificates(self) -> None:
        data = self.data
        self.assertEqual(data["alpha_complex_au3"].shape, (2, 3, 81))
        self.assertEqual(data["inverse_alpha_complex_au_minus3"].shape, (2, 3, 81))
        np.testing.assert_array_equal(data["fit_mode_count"], [[1, 9], [1, 9]])
        self.assertTrue(np.all(data["fit_passive_on_window"]))
        self.assertTrue(np.all(data["fit_nonnegative_imag_all_positive"]))
        self.assertTrue(np.all(data["fit_linear_stable"]))
        self.assertTrue(np.all(data["fit_accuracy_gate_pass"][:, 1]))
        self.assertTrue(np.all(~data["fit_accuracy_gate_pass"][:, 0]))
        np.testing.assert_allclose(data["nrms_alpha"][:, 0], 0.0, atol=0.0)
        np.testing.assert_allclose(data["nrms_inverse_alpha"][:, 0], 0.0, atol=0.0)

        alpha = np.asarray(data["alpha_complex_au3"])
        for orientation in range(2):
            reference = alpha[orientation, 0]
            reference_rms = np.sqrt(np.mean(np.abs(reference) ** 2))
            reference_peak = np.max(np.abs(reference))
            for branch in (1, 2):
                difference = alpha[orientation, branch] - reference
                expected_nrms = np.sqrt(np.mean(np.abs(difference) ** 2)) / reference_rms
                expected_maximum = np.max(np.abs(difference)) / reference_peak
                self.assertAlmostEqual(
                    data["nrms_alpha"][orientation, branch],
                    float(expected_nrms),
                    places=13,
                )
                self.assertAlmostEqual(
                    data["max_normalized_alpha_error"][orientation, branch],
                    float(expected_maximum),
                    places=13,
                )

    def test_saved_lorentz_coefficients_reconstruct_both_fit_curves(self) -> None:
        energy_au = np.asarray(self.data["energy_eV"]) / AU_ENERGY_EV
        alpha_saved = np.asarray(self.data["alpha_dimensionless_complex"])
        for orientation in range(2):
            for fit_index, branch_index in enumerate((1, 2)):
                valid = np.asarray(
                    self.data["fit_coefficient_valid"][orientation, fit_index],
                    dtype=bool,
                )
                strengths = self.data["fit_strengths_au2"][orientation, fit_index, valid]
                frequencies = self.data["fit_omega_modes_au"][orientation, fit_index, valid]
                dampings = self.data["fit_gamma_modes_au"][orientation, fit_index, valid]
                reconstructed = np.full(
                    energy_au.shape,
                    self.data["fit_alpha_inf_dimensionless"][orientation, fit_index],
                    dtype=complex,
                )
                for strength, frequency, damping in zip(
                    strengths, frequencies, dampings
                ):
                    reconstructed += strength / (
                        frequency**2
                        - energy_au**2
                        - 1j * damping * energy_au
                    )
                np.testing.assert_allclose(
                    reconstructed,
                    alpha_saved[orientation, branch_index],
                    rtol=2.0e-13,
                    atol=2.0e-13,
                )

    def test_plotter_reads_only_artifact_and_writes_figure(self) -> None:
        output = self.directory / "material_comparison.png"
        result = plot.main([str(self.artifact), "--output", str(output), "--dpi", "80"])
        self.assertEqual(result, output)
        self.assertTrue(output.is_file())
        self.assertGreater(output.stat().st_size, 1000)

        source = Path(plot.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".")[0])
        self.assertNotIn("scipy", imported_roots)
        self.assertNotIn("qd_mnp_rational_fit", imported_roots)
        self.assertNotIn("qd_mnp_full_qs_model", imported_roots)

    def test_cli_rejects_comparisons_that_are_not_one_versus_many(self) -> None:
        args = calc.parse_args(["--one-modes", "2"])
        with self.assertRaisesRegex(ValueError, "one-oscillator"):
            calc.calculate_material_dispersion_comparison(args)
        args = calc.parse_args(["--multi-modes", "1"])
        with self.assertRaisesRegex(ValueError, "at least 2"):
            calc.calculate_material_dispersion_comparison(args)

    def test_cli_rejects_spectrum_outside_fit_window_before_fitting(self) -> None:
        args = calc.parse_args(
            [
                "--energy-min-ev",
                "0.7",
                "--fit-min-ev",
                "0.8",
            ]
        )
        with self.assertRaisesRegex(ValueError, "inside the fit interval"):
            calc.calculate_material_dispersion_comparison(args)


if __name__ == "__main__":
    unittest.main()
