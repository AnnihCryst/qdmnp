"""End-to-end old/new laser-pulse comparison artifact test."""

import csv
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

from qd_mnp_spheroid_pulse_comparison import (
    _create_unique_run_dir,
    parse_args,
    run_pulse_comparison,
)


class SpheroidPulseComparisonTests(unittest.TestCase):
    def test_concurrent_smoke_run_exports_common_grid_and_diagnostics(self) -> None:
        with TemporaryDirectory() as directory:
            run_dir = Path(
                run_pulse_comparison(
                    output_dir=directory,
                    orientation="long",
                    spatial_order_max=2,
                    material_fit_modes=9,
                    pulse_E0_au=1.0e-7,
                    post_fs=20.0,
                    common_time_points=101,
                    spectral_window_policy="ignore",
                    tail_policy="ignore",
                    fit_quality_policy="ignore",
                    spatial_convergence_policy="ignore",
                    concurrent=True,
                    make_plots=True,
                    show=False,
                )
            )
            for name in (
                "metadata.json",
                "pulse_summary.csv",
                "pulse_traces_common_grid.csv",
                "pulse_traces.npz",
                "pulse_traces_adaptive.npz",
                "pulse_dynamics.png",
            ):
                with self.subTest(name=name):
                    self.assertGreater((run_dir / name).stat().st_size, 0)

            metadata = json.loads(
                (run_dir / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["execution"], "concurrent_threads")
            self.assertIn("legacy", metadata["implementation"])
            self.assertEqual(metadata["full_qs"]["spatial_order_max"], 2)
            self.assertIn("carrier_half_order_relative_change", metadata["full_qs"])
            self.assertIn("rho22_rms_difference", metadata["common_grid_metrics"])
            self.assertEqual(
                set(metadata["coupling_by_model"]),
                {"legacy", "spheroid_full"},
            )
            self.assertEqual(
                metadata["common_solver_policies"]["positivity_policy"],
                "raise",
            )
            self.assertFalse(metadata["time_window"]["post_was_automatic"])
            self.assertEqual(metadata["time_window"]["tail_policy"], "ignore")
            self.assertTrue(
                metadata["full_qs"]["coupled_spectral_abscissa_available"]
            )
            self.assertTrue(
                metadata["full_qs"]["decay_rate_estimate_is_exact"]
            )

            with (run_dir / "pulse_summary.csv").open(
                newline="",
                encoding="utf-8",
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(
                {row["model"] for row in rows},
                {"legacy", "spheroid_full"},
            )
            self.assertTrue(
                all(np.isfinite(float(row["excited_population_max"])) for row in rows)
            )

            resolved_points = metadata["time_window"][
                "common_time_points_resolved"
            ]
            self.assertGreaterEqual(resolved_points, 101)
            with np.load(run_dir / "pulse_traces.npz") as traces:
                self.assertEqual(traces["time_fs"].size, resolved_points)
                self.assertEqual(
                    traces["legacy_rho22"].shape,
                    (resolved_points,),
                )
                self.assertEqual(
                    traces["full_rho22"].shape,
                    (resolved_points,),
                )
            with np.load(run_dir / "pulse_traces_adaptive.npz") as traces:
                self.assertGreater(traces["legacy_t_au"].size, 1)
                self.assertGreater(traces["full_t_au"].size, 1)
                self.assertEqual(
                    traces["full_modal_outputs_au"].shape[1],
                    traces["full_t_au"].size,
                )

    def test_cli_defaults_to_automatic_common_post_window(self) -> None:
        with patch.object(sys, "argv", ["pulse"]):
            args = parse_args()
        self.assertIsNone(args.post_fs)
        self.assertFalse(args.auto_post)
        self.assertEqual(args.spatial_order_max, 80)
        self.assertEqual(args.positivity_policy, "raise")
        self.assertEqual(args.tail_policy, "raise")
        self.assertEqual(args.tail_ratio_tolerance, 1.0e-4)

    def test_runner_does_not_close_unrelated_matplotlib_figures(self) -> None:
        sentinel = plt.figure()
        try:
            with TemporaryDirectory() as directory:
                run_pulse_comparison(
                    output_dir=directory,
                    orientation="long",
                    spatial_order_max=1,
                    material_fit_modes=9,
                    pulse_E0_au=1.0e-7,
                    post_fs=20.0,
                    common_time_points=101,
                    spectral_window_policy="ignore",
                    tail_policy="ignore",
                    fit_quality_policy="ignore",
                    spatial_convergence_policy="ignore",
                    concurrent=False,
                    make_plots=False,
                    show=False,
                )
            self.assertTrue(plt.fignum_exists(sentinel.number))
        finally:
            plt.close(sentinel)

    def test_timestamp_collision_gets_a_unique_suffix(self) -> None:
        with TemporaryDirectory() as directory:
            fixed = Path(directory) / "20260101_000000"
            with patch(
                "qd_mnp_spheroid_pulse_comparison.timestamped_run_dir",
                return_value=fixed,
            ):
                first = _create_unique_run_dir(directory)
                second = _create_unique_run_dir(directory)
            self.assertEqual(first, fixed)
            self.assertEqual(second.name, "20260101_000000_001")


if __name__ == "__main__":
    unittest.main()
