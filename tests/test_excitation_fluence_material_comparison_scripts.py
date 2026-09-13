from __future__ import annotations

import ast
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from article_observables import qd_mnp_calculate_excitation_fluence_material_comparison as calculate
from article_observables import qd_mnp_plot_excitation_fluence_material_comparison as plot
from article_observables.qd_mnp_material_modes_artifact import atomic_write_npz


def _branch_payload(scale: float) -> dict[str, np.ndarray]:
    channel_id = np.asarray(
        [
            "bare_qd",
            "axis_long",
            "axis_trans",
            "side_long",
            "side_trans_radial",
            "side_trans_tangential",
        ],
        dtype="U32",
    )
    fluence = np.asarray([1.0e-8, 4.0e-8, 9.0e-8])
    bare = np.asarray([0.1, 0.4, 0.8])
    population = np.vstack([bare, *(np.clip(scale * bare, 0.0, 1.0) for _ in range(5))])
    return {
        "channel_id": channel_id,
        "hybrid_channel_id": channel_id[1:],
        "fluence_j_cm2": fluence,
        "pulse_e0_au": np.sqrt(fluence),
        "pulse_e0_v_m": 2.0 * np.sqrt(fluence),
        "peak_intensity_w_cm2": 3.0 * fluence,
        "isolated_qd_pulse_area_rad": 4.0 * np.sqrt(fluence),
        "p_exc_read": population,
        "p_exc_max": np.minimum(population + 0.05, 1.0),
        "time_of_p_exc_max_fs": np.full_like(population, 1.0),
        "read_time_fs": np.full(fluence.size, 250.0),
    }


class ExcitationFluenceMaterialComparisonTests(unittest.TestCase):
    def test_first_threshold_uses_sqrt_fluence(self) -> None:
        fluence = np.asarray([1.0, 4.0, 9.0])
        population = np.asarray([0.1, 0.4, 0.8])
        value, status, bracket = calculate._first_threshold(fluence, population, 0.6)
        self.assertEqual(status, "resolved")
        self.assertEqual(bracket, 1)
        self.assertAlmostEqual(value, 6.25)

    def test_calculator_and_plotter_stop_before_second_rabi_lobe(self) -> None:
        fluence = np.arange(1.0, 7.0) ** 2
        population = np.asarray([0.1, 0.6, 0.6, 0.4, 0.5, 0.8])
        value, status, bracket = calculate._first_threshold(fluence, population, 0.7)
        plotted_value, plotted_status = plot._first_threshold(fluence, population, 0.7)
        self.assertTrue(np.isnan(value))
        self.assertTrue(np.isnan(plotted_value))
        self.assertEqual(status, "not_reached_first_lobe")
        self.assertEqual(plotted_status, status)
        self.assertEqual(bracket, -1)

    def test_left_censored_thresholds_do_not_enter_ratios_or_intensities(self) -> None:
        args = calculate.parse_args(["--preset", "quick", "--points", "3", "--target-population", "0.5"])
        one, multi = _branch_payload(1.2), _branch_payload(1.5)
        one["p_exc_read"][1] = [0.6, 0.7, 0.8]
        metadata = {"resolved_settings": {}, "channels": []}
        with (
            patch.object(calculate, "calculate_excitation_fluence", side_effect=[(one, metadata), (multi, metadata)]),
            patch.object(calculate, "validate_excitation_comparison"),
            patch.object(calculate, "source_hashes", return_value={}),
            patch.object(calculate, "git_provenance", return_value={}),
        ):
            payload, _ = calculate.calculate_material_fluence_comparison(args)
        self.assertEqual(payload["threshold_status"][0, 1], "left_censored")
        self.assertEqual(payload["threshold_fluence_j_cm2"][0, 1], one["fluence_j_cm2"][0])
        self.assertTrue(np.isnan(payload["threshold_peak_intensity_w_cm2"][0, 1]))
        self.assertTrue(np.isnan(payload["hybrid_threshold_ratio_one_to_multi"][0]))
        self.assertTrue(np.all(np.isfinite(payload["hybrid_threshold_ratio_one_to_multi"][1:])))

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(plot, "save_figure") as saved:
                plot.plot_comparison(payload, {}, Path(directory) / "censored.png", channel_filter=["axis_long"])
            figure = saved.call_args.args[0]
            labels = [item.get_text() for axis in figure.axes for item in axis.texts]
            self.assertTrue(any("left_censored" in label for label in labels))
            self.assertFalse(any("mathcal" in label for label in labels))
            plot.plt.close(figure)

    def test_calculation_stacks_two_existing_api_results(self) -> None:
        args = calculate.parse_args(
            [
                "--preset",
                "quick",
                "--points",
                "3",
                "--target-population",
                "0.5",
            ]
        )
        one = _branch_payload(1.2)
        multi = _branch_payload(1.5)
        one_metadata = {"resolved_settings": {}, "channels": []}
        multi_metadata = {"resolved_settings": {}, "channels": []}
        with (
            patch.object(
                calculate,
                "calculate_excitation_fluence",
                side_effect=[(one, one_metadata), (multi, multi_metadata)],
            ) as mocked_calculation,
            patch.object(calculate, "validate_excitation_comparison") as mocked_validation,
            patch.object(calculate, "source_hashes", return_value={}),
            patch.object(calculate, "git_provenance", return_value={}),
        ):
            payload, metadata = calculate.calculate_material_fluence_comparison(args)
        self.assertEqual(mocked_calculation.call_count, 2)
        mocked_validation.assert_called_once()
        self.assertEqual(payload["p_exc_read"].shape, (2, 6, 3))
        self.assertTrue(np.array_equal(payload["branch_material_mode_count"], [1, 9]))
        self.assertTrue(np.allclose(payload["p_exc_read"][0, 0], payload["p_exc_read"][1, 0]))
        self.assertEqual(metadata["schema_name"], calculate.SCHEMA_NAME)
        self.assertIn("one__p_exc_read", payload)
        self.assertIn("multi__p_exc_read", payload)

    def test_same_mode_count_is_rejected(self) -> None:
        args = calculate.parse_args(
            [
                "--preset",
                "quick",
                "--one-material-fit-modes",
                "9",
            ]
        )
        with self.assertRaisesRegex(ValueError, "different mode counts"):
            calculate.calculate_material_fluence_comparison(args)

    def test_branch_names_enforce_one_versus_many(self) -> None:
        one_args = calculate.parse_args(
            [
                "--preset",
                "quick",
                "--one-material-fit-modes",
                "2",
            ]
        )
        with self.assertRaisesRegex(ValueError, "must equal 1"):
            calculate.calculate_material_fluence_comparison(one_args)

        multi_args = calculate.parse_args(
            [
                "--preset",
                "quick",
                "--material-fit-modes",
                "1",
            ]
        )
        with self.assertRaisesRegex(ValueError, "at least 2"):
            calculate.calculate_material_fluence_comparison(multi_args)

    def test_pickle_free_roundtrip_and_plot_from_npz(self) -> None:
        one = _branch_payload(1.2)
        multi = _branch_payload(1.5)
        p_read = np.stack([one["p_exc_read"], multi["p_exc_read"]])
        p_max = np.stack([one["p_exc_max"], multi["p_exc_max"]])
        payload: dict[str, np.ndarray] = {
            "branch_id": np.asarray(["one", "multi"]),
            "branch_material_mode_count": np.asarray([1, 9]),
            "channel_id": one["channel_id"],
            "hybrid_channel_id": one["hybrid_channel_id"],
            "fluence_j_cm2": one["fluence_j_cm2"],
            "p_exc_read": p_read,
            "p_exc_max": p_max,
            "read_time_fs": one["read_time_fs"],
            "target_population": np.asarray(0.5),
            "threshold_fluence_j_cm2": np.ones((2, 6)),
            "hybrid_threshold_ratio_one_to_multi": np.ones(5),
        }
        for branch, source in (("one", one), ("multi", multi)):
            for key, value in source.items():
                payload[f"{branch}__{key}"] = value
        metadata = {
            "schema_name": plot.SCHEMA_NAME,
            "schema_version": 1,
            "branch_metadata": {
                "one": {"resolved_settings": {}, "channels": []},
                "multi": {"resolved_settings": {}, "channels": []},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "comparison.npz"
            figure = Path(directory) / "comparison.png"
            atomic_write_npz(artifact, payload, metadata)
            loaded, loaded_metadata, failures = plot.load_comparison_artifact(
                artifact,
                allow_unconverged=True,
            )
            self.assertTrue(failures)
            plot.plot_comparison(
                loaded,
                loaded_metadata,
                figure,
                diagnostic_failures=failures,
            )
            self.assertTrue(figure.is_file())
            with np.load(artifact, allow_pickle=False) as archive:
                self.assertFalse(any(archive[key].dtype.hasobject for key in archive.files))

    def test_plotter_has_no_solver_imports(self) -> None:
        source = Path(plot.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        forbidden = (
            "qd_mnp_rational_fit",
            "qd_mnp_full_qs_model",
            "qd_mnp_spheroid_green",
            "qd_mnp_calculate_excitation_fluence",
            "scipy",
        )
        self.assertFalse(any(name.startswith(forbidden) for name in imported))


if __name__ == "__main__":
    unittest.main()
