"""Tests for the direct/one-/multi-mode excitation-spectrum script pair."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import matplotlib
import numpy as np

matplotlib.use("Agg")

from article_observables.qd_mnp_calculate_excitation_spectrum_material_comparison import (
    SCHEMA_NAME,
    _validate_args,
    calculate_payload,
    extract_fwhm_feature,
    parse_args,
)
from article_observables.qd_mnp_material_modes_artifact import atomic_write_npz
from article_observables.qd_mnp_plot_excitation_spectrum_material_comparison import (
    plot_artifact,
)


def lorentzian(energy: np.ndarray, center: float, hwhm: float) -> np.ndarray:
    return 1.0 / (1.0 + ((energy - center) / hwhm) ** 2)


class SpectrumMaterialFeatureTests(unittest.TestCase):
    def test_fwhm_extraction_on_lorentzian(self) -> None:
        energy = np.linspace(1.8, 2.2, 4001)
        feature = extract_fwhm_feature(
            energy,
            lorentzian(energy, 2.01, 0.02),
            center_eV=2.0,
            half_window_eV=0.15,
        )
        self.assertEqual(feature["status"], "ok")
        self.assertAlmostEqual(float(feature["energy_eV"]), 2.01, places=5)
        self.assertAlmostEqual(float(feature["fwhm_eV"]), 0.04, delta=2.0e-4)

    def test_multi_representation_really_requires_multiple_modes(self) -> None:
        args = parse_args(
            [
                "--output",
                "unused.npz",
                "--preset",
                "quick",
                "--multi-fit-modes",
                "1",
            ]
        )
        with self.assertRaisesRegex(ValueError, "at least two"):
            _validate_args(args)


class SpectrumMaterialCalculationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.args = parse_args(
            [
                "--output",
                "unused.npz",
                "--preset",
                "quick",
                "--channels",
                "axis_long",
                "--gap-nm",
                "2",
                "--energy-min-ev",
                "1.90",
                "--energy-max-ev",
                "2.18",
                "--energy-points",
                "101",
                "--gamma2-coherence-mev",
                "20",
                "--spatial-order-max",
                "2",
                "--modal-audit-points",
                "101",
                "--spatial-convergence-policy",
                "ignore",
                "--energy-resolution-policy",
                "ignore",
                "--radiative-consistency-policy",
                "ignore",
                "--one-fit-quality-policy",
                "ignore",
            ]
        )
        cls.payload, cls.metadata = calculate_payload(cls.args)

    def test_current_fqs_apis_produce_three_absolute_spectra(self) -> None:
        payload = self.payload
        self.assertEqual(payload["material_model_id"].tolist(), ["direct", "one", "multi"])
        self.assertEqual(payload["excitation_spectrum_au6"].shape, (3, 1, 101))
        self.assertEqual(payload["qd_transfer_au3"].shape, (3, 1, 101))
        self.assertEqual(payload["alpha_effective_au3"].shape, (3, 1, 101))
        self.assertTrue(np.all(np.isfinite(payload["interaction_A_au3"])))
        self.assertTrue(np.all(np.isfinite(payload["interaction_B"])))
        self.assertTrue(np.all(np.isfinite(payload["interaction_K_au_minus3"])))
        self.assertGreater(
            float(
                np.max(
                    np.abs(
                        payload["excitation_spectrum_au6"][1]
                        - payload["excitation_spectrum_au6"][0]
                    )
                )
            ),
            0.0,
        )
        np.testing.assert_array_equal(
            payload["excitation_spectrum_au6_residual_vs_direct"][0],
            np.zeros((1, 101)),
        )

    def test_fit_and_spatial_certificates_are_persisted(self) -> None:
        payload = self.payload
        self.assertEqual(self.metadata["schema_name"], SCHEMA_NAME)
        self.assertEqual(payload["fit_mode_count"].tolist(), [[1], [9]])
        self.assertTrue(np.all(payload["fit_passive_on_fit_window"]))
        self.assertTrue(np.all(payload["fit_passive_for_all_positive_frequencies"]))
        self.assertTrue(np.all(payload["bright_stability_stable"]))
        self.assertTrue(np.all(payload["coupled_stability_stable"]))
        self.assertTrue(bool(payload["modal_fit_accepted"][1, 0]))
        self.assertIn("material_energy_eV", payload)
        self.assertEqual(payload["spatial_channel_offsets"].shape, (2,))
        self.assertEqual(
            self.metadata["model_api"]["fitted_FQS"],
            "FullQSSpheroidPulseModel.frequency_response_from_fit",
        )


class SpectrumMaterialPlotTests(unittest.TestCase):
    @staticmethod
    def _write_artifact(path: Path, *, certified: bool = True) -> None:
        energy = np.linspace(1.9, 2.2, 301)
        isolated = lorentzian(energy, 2.04, 0.02)
        spectra = np.empty((3, 2, energy.size), dtype=float)
        for model_index in range(3):
            for channel_index in range(2):
                spectra[model_index, channel_index] = (
                    1.0 + 0.4 * model_index + 0.15 * channel_index
                ) * lorentzian(
                    energy,
                    2.04 + 0.004 * model_index,
                    0.02 + 0.002 * channel_index,
                )
        residual = spectra - spectra[0:1]
        residual /= np.max(spectra[0], axis=-1)[None, :, None]
        spatial_certificate = np.ones((2, 2), dtype=bool)
        if not certified:
            spatial_certificate[1, 0] = False
        payload = {
            "material_model_id": np.asarray(["direct", "one", "multi"], dtype="U16"),
            "channel_id": np.asarray(["axis_long", "axis_trans"], dtype="U32"),
            "energy_eV": energy,
            "surface_gap_nm": np.asarray(2.0),
            "isolated_excitation_spectrum_au6": isolated,
            "excitation_spectrum_au6": spectra,
            "excitation_spectrum_residual_normalized_to_direct_peak": residual,
            "peak_energy_eV": np.asarray([[2.04, 2.04], [2.044, 2.044], [2.048, 2.048]]),
            "feature_status": np.full((3, 2), "ok", dtype="U40"),
            "energy_step_over_isolated_fwhm": np.asarray(0.01),
            "fit_passive_on_fit_window": np.ones((2, 2), dtype=bool),
            "fit_passive_for_all_positive_frequencies": np.ones(
                (2, 2), dtype=bool
            ),
            "bright_stability_stable": np.ones((2, 2), dtype=bool),
            "modal_fit_passive_on_audit_grid": np.ones((2, 2), dtype=bool),
            "modal_fit_accepted": np.ones((2, 2), dtype=bool),
            "spatial_convergence_accepted": spatial_certificate,
            "coupled_stability_stable": np.ones((2, 2), dtype=bool),
        }
        metadata = {
            "schema_name": SCHEMA_NAME,
            "schema_version": 1,
            "multi_fit_mode_count": 9,
            "resolved_arguments": {
                "max_energy_step_over_isolated_fwhm": 0.05,
            },
            "channels": [
                {"channel_id": "axis_long", "label": "tip, longitudinal"},
                {"channel_id": "axis_trans", "label": "tip, transverse"},
            ],
        }
        atomic_write_npz(path, payload, metadata)

    def test_plotter_reads_saved_data_without_solver_imports(self) -> None:
        plotter_path = (
            Path(__file__).resolve().parents[1]
            / "article_observables"
            / "qd_mnp_plot_excitation_spectrum_material_comparison.py"
        )
        source = plotter_path.read_text(encoding="utf-8")
        self.assertNotIn("import scipy", source)
        self.assertNotIn("qd_mnp_full_qs_model", source)
        self.assertNotIn("qd_mnp_rational_fit", source)
        self.assertNotIn("qd_mnp_spheroid_green", source)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "spectrum.npz"
            figure = root / "spectrum.png"
            self._write_artifact(artifact)
            output = plot_artifact(artifact, figure, dpi=72)
            self.assertEqual(output, figure)
            self.assertTrue(figure.is_file())
            self.assertGreater(figure.stat().st_size, 1000)
            with np.load(artifact, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata_json"].item()))
            self.assertEqual(
                metadata["schema_name"],
                "qd_mnp.material_excitation_spectrum_comparison",
            )

    def test_plotter_refuses_failed_certificate_without_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "uncertified.npz"
            figure = root / "diagnostic.png"
            self._write_artifact(artifact, certified=False)
            with self.assertRaisesRegex(ValueError, "spatial_convergence_accepted"):
                plot_artifact(artifact, figure, dpi=72)
            output = plot_artifact(
                artifact,
                figure,
                dpi=72,
                allow_unconverged=True,
            )
            self.assertEqual(output, figure)
            self.assertTrue(figure.is_file())


if __name__ == "__main__":
    unittest.main()
