"""Focused tests for the separated population-dynamics calculation/plot path."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import matplotlib

matplotlib.use("Agg")
import numpy as np

from article_observables.qd_mnp_calculate_population_dynamics import (
    CHANNELS,
    _surface_gap_separation_nm,
    calculate_population_dynamics,
    parse_args as parse_calculation_args,
)
from article_observables.qd_mnp_plot_population_dynamics import (
    load_population_artifact,
    plot_population_dynamics,
)


class PopulationDynamicsScriptTests(unittest.TestCase):
    def test_tip_and_side_use_the_same_surface_gap(self) -> None:
        common = dict(c_nm=15.0, a_nm=7.0, qd_radius_nm=2.0, gap_nm=1.5)
        tip = _surface_gap_separation_nm(CHANNELS["axis_long"], **common)
        side = _surface_gap_separation_nm(CHANNELS["side_long"], **common)
        self.assertAlmostEqual(tip, 18.5)
        self.assertAlmostEqual(side, 10.5)

    def test_calculation_npz_round_trip_and_plot_only_consumer(self) -> None:
        with TemporaryDirectory() as directory:
            artifact = Path(directory) / "population.npz"
            figure = Path(directory) / "population.png"
            args = parse_calculation_args(
                [
                    "--output",
                    str(artifact),
                    "--channels",
                    "axis_long",
                    "--spatial-order-max",
                    "1",
                    "--fluence-j-cm2",
                    "1e-12",
                    "--pulse-tau-fs",
                    "10",
                    "--post-fs",
                    "80",
                    "--common-time-points",
                    "201",
                    "--rtol",
                    "1e-6",
                    "--atol",
                    "1e-8",
                    "--points-per-fastest-cycle",
                    "8",
                    "--radiative-consistency-policy",
                    "ignore",
                    "--fit-quality-policy",
                    "ignore",
                    "--spatial-convergence-policy",
                    "ignore",
                    "--spectral-window-policy",
                    "ignore",
                    "--response-tail-policy",
                    "ignore",
                ]
            )
            calculate_population_dynamics(args)
            with self.assertRaisesRegex(
                ValueError, "uncertified population-dynamics artifact"
            ) as rejected:
                load_population_artifact(artifact)
            self.assertIn(
                "full-QS spatial-convergence certificate",
                str(rejected.exception),
            )
            loaded = load_population_artifact(artifact, allow_unconverged=True)
            self.assertEqual(loaded["rho22"].shape[0], 2)
            self.assertEqual(list(loaded["channel_ids"]), ["bare_qd", "axis_long"])
            metadata = loaded["metadata"]
            self.assertEqual(
                metadata["schema"],
                {"name": "qd_mnp_population_dynamics", "version": 1},
            )
            self.assertIn("atomic_unit_time_s", metadata["fundamental_and_conversion_constants"])
            self.assertAlmostEqual(
                metadata["physical_parameters_by_channel"]["axis_long"]["surface_gap_nm"],
                1.0,
            )
            hybrid_diagnostics = metadata["diagnostics_by_channel"]["axis_long"]
            for name in (
                "solver_success",
                "t_final_reached",
                "state_is_finite",
                "min_density_eigenvalue",
                "pulse_spectral_leakage",
                "qd_source_spectral_leakage",
                "mnp_drive_spectral_leakage",
                "mnp_dipole_spectral_leakage",
                "mnp_field_spectral_leakage",
                "work_nonnegative_within_tolerance",
                "work_passivity_tolerance_au",
            ):
                self.assertIn(name, hybrid_diagnostics)
            model_certificate = metadata["model_by_channel"]["axis_long"]
            self.assertIn("spatial_convergence", model_certificate)
            self.assertIn("modal_transform", model_certificate)
            self.assertIn("coupled_stability", model_certificate)
            self.assertIn("dark_reduction", model_certificate)
            self.assertFalse(model_certificate["spatial_convergence"]["accepted"])
            self.assertTrue(model_certificate["modal_transform"]["accepted"])
            self.assertTrue(model_certificate["coupled_stability"]["stable"])
            self.assertFalse(model_certificate["dark_reduction"]["applied"])
            self.assertTrue(
                model_certificate["material_fit"]["passive_on_fit_window"]
            )
            self.assertTrue(model_certificate["material_fit"]["globally_passive"])
            with np.load(artifact, allow_pickle=False) as stored:
                self.assertIn("material_energy_eV", stored.files)
                parsed = json.loads(str(stored["metadata_json"].item()))
                self.assertEqual(parsed["calculation_script"], artifact_script_name())

            built = plot_population_dynamics(
                loaded,
                output_path=figure,
                dpi=72,
                show=False,
            )
            self.assertTrue(figure.is_file())
            self.assertGreater(figure.stat().st_size, 1000)
            matplotlib.pyplot.close(built)

        plot_source = Path(
            "article_observables/qd_mnp_plot_population_dynamics.py"
        ).read_text(encoding="utf-8")
        imported_roots = {
            alias.name.split(".")[0]
            for node in ast.walk(ast.parse(plot_source))
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported_roots.update(
            node.module.split(".")[0]
            for node in ast.walk(ast.parse(plot_source))
            if isinstance(node, ast.ImportFrom) and node.module
        )
        self.assertFalse(any(name.startswith("qd_mnp") for name in imported_roots))
        self.assertNotIn("scipy", imported_roots)

    def test_strict_tail_policy_rejects_short_explicit_window(self) -> None:
        with TemporaryDirectory() as directory:
            artifact = Path(directory) / "must_not_exist.npz"
            args = parse_calculation_args(
                [
                    "--output",
                    str(artifact),
                    "--channels",
                    "axis_long",
                    "--spatial-order-max",
                    "1",
                    "--fluence-j-cm2",
                    "1e-12",
                    "--pulse-tau-fs",
                    "10",
                    "--post-fs",
                    "80",
                    "--common-time-points",
                    "201",
                    "--rtol",
                    "1e-6",
                    "--atol",
                    "1e-8",
                    "--points-per-fastest-cycle",
                    "8",
                    "--radiative-consistency-policy",
                    "ignore",
                    "--fit-quality-policy",
                    "ignore",
                    "--spatial-convergence-policy",
                    "ignore",
                    "--spectral-window-policy",
                    "ignore",
                    "--response-tail-policy",
                    "raise",
                ]
            )
            with self.assertRaisesRegex(RuntimeError, "response-tail"):
                calculate_population_dynamics(args)
            self.assertFalse(artifact.exists())


def artifact_script_name() -> str:
    return "article_observables/qd_mnp_calculate_population_dynamics.py"


if __name__ == "__main__":
    unittest.main()
