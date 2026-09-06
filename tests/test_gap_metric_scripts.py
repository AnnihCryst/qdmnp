"""Tests for the DD/FQS article gap-metric calculation/plot split."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import matplotlib
import numpy as np

matplotlib.use("Agg")

from article_observables.qd_mnp_gap_metrics_common import (
    MODEL_IDS,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    _atomic_write_npz,
    _spectral_metrics,
    compute_spectral_payload,
    dd_validity_distance,
    extract_feature,
    load_gap_metric_artifact,
    parse_spectral_calculation_args,
    plot_gap_metric,
    run_spectral_calculation,
    threshold_from_curve,
)


def lorentzian(energy: np.ndarray, center: float, gamma_hwhm: float) -> np.ndarray:
    return 1.0 / (1.0 + ((energy - center) / gamma_hwhm) ** 2)


class GapMetricExtractionTests(unittest.TestCase):
    def test_feature_peak_and_half_prominence_width(self) -> None:
        energy = np.linspace(1.8, 2.2, 4001)
        spectrum = lorentzian(energy, 2.01, 0.02)
        feature = extract_feature(
            energy,
            spectrum,
            center_eV=2.0,
            half_window_eV=0.15,
        )
        self.assertEqual(feature.status, "ok")
        self.assertAlmostEqual(feature.energy_eV, 2.01, places=4)
        # scipy peak prominence uses the finite feature window as its local
        # baseline, so the width is close to, but slightly below, 2*HWHM.
        self.assertAlmostEqual(feature.width_eV, 0.0393, delta=6.0e-4)

    def test_comparable_split_peak_marks_width_ambiguous(self) -> None:
        energy = np.linspace(1.8, 2.2, 4001)
        spectrum = lorentzian(energy, 1.97, 0.006) + lorentzian(
            energy, 2.03, 0.006
        )
        feature = extract_feature(
            energy,
            spectrum,
            center_eV=2.0,
            half_window_eV=0.15,
        )
        self.assertEqual(feature.status, "split_or_ambiguous")
        self.assertTrue(np.isnan(feature.width_eV))
        self.assertGreaterEqual(feature.competing_peak_count, 1)

    def test_threshold_uses_first_rabi_lobe_only(self) -> None:
        fluence = np.asarray([1.0, 4.0, 9.0, 16.0, 25.0]) * 1.0e-8
        population = np.asarray([0.05, 0.35, 0.60, 0.40, 0.80])
        threshold, status, bracket = threshold_from_curve(fluence, population, 0.5)
        self.assertEqual(status, "resolved")
        self.assertEqual(bracket, (1, 2))
        self.assertGreater(threshold, fluence[1])
        self.assertLess(threshold, fluence[2])

        threshold, status, bracket = threshold_from_curve(fluence, population, 0.7)
        self.assertTrue(np.isnan(threshold))
        self.assertEqual(status, "not_reached_first_lobe")
        self.assertIsNone(bracket)

    def test_threshold_censoring_is_explicit(self) -> None:
        fluence = np.asarray([1.0, 4.0, 9.0]) * 1.0e-8
        threshold, status, _ = threshold_from_curve(
            fluence, np.asarray([0.6, 0.7, 0.8]), 0.5
        )
        self.assertEqual(threshold, fluence[0])
        self.assertEqual(status, "left_censored")

    def test_dd_validity_distance_requires_all_larger_gaps(self) -> None:
        gaps = np.asarray([1.0, 2.0, 5.0, 10.0, 20.0])
        discrepancy = np.asarray([0.5, 0.08, 0.12, 0.04, 0.02])
        self.assertEqual(dd_validity_distance(gaps, discrepancy, 0.1), 10.0)
        self.assertTrue(
            np.isnan(dd_validity_distance(gaps, discrepancy, 0.01))
        )
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            dd_validity_distance(gaps, discrepancy, 0.0)

    def test_spectral_metrics_keep_absolute_dd_fqs_scale(self) -> None:
        energy = np.linspace(1.8, 2.2, 2001)
        isolated = lorentzian(energy, 2.0, 0.02)
        spectra = np.empty((2, 1, 2, energy.size))
        spectra[0, 0, 0] = 2.0 * lorentzian(energy, 2.005, 0.022)
        spectra[1, 0, 0] = 4.0 * lorentzian(energy, 2.010, 0.024)
        spectra[0, 0, 1] = 1.1 * lorentzian(energy, 2.001, 0.020)
        spectra[1, 0, 1] = 1.0 * lorentzian(energy, 2.000, 0.020)
        metrics = _spectral_metrics(
            energy,
            isolated,
            spectra,
            center_eV=2.0,
            half_window_eV=0.15,
        )
        self.assertAlmostEqual(metrics["excitation_gain_optimized"][1, 0, 0], 4.0, places=5)
        self.assertGreater(metrics["spectral_l2_relative_dd_vs_fqs"][0, 0], 0.4)
        self.assertGreater(
            metrics["spectral_l2_relative_dd_vs_fqs"][0, 0],
            metrics["spectral_l2_relative_dd_vs_fqs"][0, 1],
        )


class GapMetricArtifactAndApiTests(unittest.TestCase):
    @staticmethod
    def _spectral_artifact(path: Path) -> None:
        energy = np.linspace(1.8, 2.2, 801)
        gaps = np.asarray([1.0, 5.0, 20.0])
        isolated = lorentzian(energy, 2.0, 0.02)
        spectra = np.empty((2, 2, gaps.size, energy.size))
        for mi in range(2):
            for ci in range(2):
                for gi, gap in enumerate(gaps):
                    scale = 1.0 + 0.4 * (mi + 1) / gap + 0.1 * ci
                    shift = 0.006 * (mi + 1) / gap
                    spectra[mi, ci, gi] = scale * lorentzian(
                        energy, 2.0 + shift, 0.02 + 0.002 / gap
                    )
        derived = _spectral_metrics(
            energy,
            isolated,
            spectra,
            center_eV=2.0,
            half_window_eV=0.15,
        )
        payload = {
            "channel_id": np.asarray(["axis_long", "side_long"], dtype="U32"),
            "model_id": MODEL_IDS,
            "gap_nm": gaps,
            "energy_eV": energy,
            "isolated_qd_spectrum": isolated,
            "qd_excitation_spectrum": spectra,
            **derived,
        }
        metadata = {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "primary_metric": "excitation_gain",
            "computation_family": "spectral_gap_scan",
            "requested_arguments": {
                "feature_center_ev": 2.0,
                "feature_half_window_ev": 0.15,
            },
            "channels": [
                {"channel_id": "axis_long", "label": "tip, longitudinal"},
                {"channel_id": "side_long", "label": "side, longitudinal"},
            ],
        }
        _atomic_write_npz(path, payload, metadata, overwrite=False)

    @staticmethod
    def _threshold_artifact(path: Path) -> None:
        fluence = np.asarray([1.0, 4.0, 9.0, 16.0, 25.0]) * 1.0e-8
        isolated = np.asarray([0.05, 0.25, 0.55, 0.75, 0.62])
        hybrid = np.empty((2, 1, 2, fluence.size))
        hybrid[0, 0, 0] = [0.10, 0.40, 0.70, 0.65, 0.50]
        hybrid[1, 0, 0] = [0.15, 0.55, 0.80, 0.60, 0.40]
        hybrid[0, 0, 1] = [0.06, 0.28, 0.58, 0.74, 0.60]
        hybrid[1, 0, 1] = [0.055, 0.27, 0.57, 0.74, 0.61]
        iso_threshold, _, _ = threshold_from_curve(fluence, isolated, 0.5)
        thresholds = np.empty((2, 1, 2))
        for index in np.ndindex(thresholds.shape):
            thresholds[index], _, _ = threshold_from_curve(
                fluence, hybrid[index], 0.5
            )
        payload = {
            "channel_id": np.asarray(["axis_long"], dtype="U32"),
            "model_id": MODEL_IDS,
            "gap_nm": np.asarray([1.0, 10.0]),
            "fluence_j_cm2": fluence,
            "isolated_qd_population": isolated,
            "hybrid_population": hybrid,
            "threshold_population": np.asarray(0.5),
            "threshold_fluence_j_cm2": thresholds,
            "threshold_fluence_ratio_to_isolated": thresholds / iso_threshold,
            "absolute_threshold_discrepancy_dd_vs_fqs": np.abs(
                (thresholds[0] - thresholds[1]) / thresholds[1]
            ),
        }
        metadata = {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "primary_metric": "threshold_fluence",
            "computation_family": "threshold_gap_scan",
            "channels": [
                {"channel_id": "axis_long", "label": "tip, longitudinal"}
            ],
        }
        _atomic_write_npz(path, payload, metadata, overwrite=False)

    def test_artifact_round_trip_and_every_plotter_metric(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spectral = root / "spectral.npz"
            threshold = root / "threshold.npz"
            self._spectral_artifact(spectral)
            self._threshold_artifact(threshold)
            payload, metadata = load_gap_metric_artifact(spectral)
            self.assertEqual(metadata["schema_name"], SCHEMA_NAME)
            self.assertIn("qd_excitation_spectrum", payload)

            for metric in (
                "excitation_gain",
                "resonance_shift",
                "spectral_width",
                "model_discrepancy",
            ):
                output = root / f"{metric}.png"
                plot_gap_metric(
                    metric,
                    spectral,
                    output,
                    feature_center_ev=None,
                    feature_half_window_ev=None,
                    target_population=None,
                    dd_tolerance=0.1,
                    gain_kind="optimized",
                    x_scale="log",
                    dpi=72,
                    show=False,
                )
                self.assertTrue(output.is_file())
                self.assertGreater(output.stat().st_size, 1000)

            output = root / "threshold.png"
            with self.assertWarnsRegex(RuntimeWarning, "not Brent-refined"):
                plot_gap_metric(
                    "threshold_fluence",
                    threshold,
                    output,
                    feature_center_ev=None,
                    feature_half_window_ev=None,
                    target_population=0.45,
                    dd_tolerance=0.1,
                    gain_kind="optimized",
                    x_scale="log",
                    dpi=72,
                    show=False,
                )
            self.assertTrue(output.is_file())

            output = root / "threshold_discrepancy.png"
            plot_gap_metric(
                "model_discrepancy",
                threshold,
                output,
                feature_center_ev=None,
                feature_half_window_ev=None,
                target_population=None,
                dd_tolerance=0.1,
                gain_kind="optimized",
                x_scale="log",
                dpi=72,
                show=False,
            )
            self.assertTrue(output.is_file())

    def test_spectral_metric_can_be_derived_without_recalculation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.npz"
            output = root / "shift.npz"
            self._spectral_artifact(source)
            run_spectral_calculation(
                "resonance_shift",
                Path(__file__),
                [
                    "--source-artifact",
                    str(source),
                    "--output",
                    str(output),
                ],
            )
            payload, metadata = load_gap_metric_artifact(output)
            self.assertEqual(metadata["primary_metric"], "resonance_shift")
            self.assertEqual(metadata["derived_from_artifact"], str(source.resolve()))
            np.testing.assert_allclose(
                payload["qd_excitation_spectrum"],
                load_gap_metric_artifact(source)[0]["qd_excitation_spectrum"],
            )

    def test_threshold_discrepancy_can_be_derived_without_ode_solve(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "threshold.npz"
            output = root / "threshold_delta.npz"
            self._threshold_artifact(source)
            run_spectral_calculation(
                "model_discrepancy",
                Path(__file__),
                [
                    "--source-artifact",
                    str(source),
                    "--output",
                    str(output),
                ],
            )
            payload, metadata = load_gap_metric_artifact(output)
            self.assertEqual(metadata["primary_metric"], "model_discrepancy")
            self.assertEqual(metadata["computation_family"], "threshold_gap_scan")
            self.assertIn("absolute_threshold_discrepancy_dd_vs_fqs", payload)

    def test_frequency_calculation_calls_current_dd_and_fqs_apis(self) -> None:
        args = parse_spectral_calculation_args(
            "excitation_gain",
            [
                "--output",
                "unused.npz",
                "--preset",
                "quick",
                "--channels",
                "axis_long",
                "--gaps-nm",
                "2",
                "--energy-min-ev",
                "1.95",
                "--energy-max-ev",
                "2.13",
                "--energy-points",
                "121",
                "--gamma2-coherence-mev",
                "20",
                "--spatial-convergence-policy",
                "ignore",
                "--energy-resolution-policy",
                "ignore",
                "--radiative-consistency-policy",
                "ignore",
            ],
        )
        payload, metadata = compute_spectral_payload(args)
        self.assertEqual(payload["qd_excitation_spectrum"].shape, (2, 1, 1, 121))
        self.assertTrue(np.all(np.isfinite(payload["interaction_B"])))
        self.assertTrue(np.all(np.isfinite(payload["interaction_K_au_minus3"])))
        self.assertGreater(
            float(
                np.max(
                    np.abs(
                        payload["qd_excitation_spectrum"][0]
                        - payload["qd_excitation_spectrum"][1]
                    )
                )
            ),
            0.0,
        )
        self.assertEqual(metadata["spectrum_observable"], "linear_qd_response")


if __name__ == "__main__":
    unittest.main()
