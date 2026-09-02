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
            self.assertEqual(metadata["physical_parameters"]["qd_placement"], "axis")
            self.assertIsNone(
                metadata["physical_parameters"]["side_transverse_alignment"]
            )
            self.assertEqual(metadata["full_qs"]["spatial_order_max"], 2)
            self.assertFalse(metadata["full_qs"]["dark_reduction"]["applied"])
            self.assertEqual(metadata["full_qs"]["exact_spatial_mode_count"], 2)
            self.assertEqual(metadata["full_qs"]["dynamic_spatial_mode_count"], 2)
            direct_rows = metadata["full_qs"]["full_modal_outputs_au_rows"]
            self.assertEqual(len(direct_rows), 2)
            self.assertEqual(
                [row["full_modal_outputs_au_row"] for row in direct_rows],
                [0, 1],
            )
            self.assertEqual(
                [row["role"] for row in direct_rows],
                ["bright", "exact_dark"],
            )
            self.assertEqual(
                [row["source_mode_indices"] for row in direct_rows],
                [[0], [1]],
            )
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
        self.assertEqual(args.qd_placement, "axis")
        self.assertIsNone(args.side_transverse_alignment)
        self.assertEqual(args.spatial_order_max, 80)
        self.assertEqual(args.reduction_fit_grid_points, 1001)
        self.assertEqual(args.reduction_audit_grid_points, 1601)
        self.assertEqual(args.reduction_policy, "raise")
        self.assertEqual(args.max_modal_normalized_rms, 0.03)
        self.assertEqual(args.max_modal_relative_error, 0.06)
        self.assertEqual(args.positivity_policy, "raise")
        self.assertEqual(args.tail_policy, "raise")
        self.assertEqual(args.tail_ratio_tolerance, 1.0e-4)

    def test_side_long_n80_uses_certified_positive_reduction(self) -> None:
        with TemporaryDirectory() as directory:
            run_dir = Path(
                run_pulse_comparison(
                    output_dir=directory,
                    orientation="long",
                    qd_placement="side",
                    spatial_order_max=80,
                    material_fit_modes=9,
                    reduction_rms_tolerance=2.0e-6,
                    reduction_max_tolerance=2.0e-4,
                    pulse_E0_au=1.0e-7,
                    post_fs=20.0,
                    common_time_points=101,
                    spectral_window_policy="ignore",
                    tail_policy="ignore",
                    concurrent=False,
                    make_plots=False,
                    show=False,
                )
            )
            metadata = json.loads(
                (run_dir / "metadata.json").read_text(encoding="utf-8")
            )
            full_qs = metadata["full_qs"]
            certificate = full_qs["dark_reduction"]["certificate"]
            self.assertEqual(
                metadata["physical_parameters"]["qd_placement"],
                "side",
            )
            self.assertIsNone(
                metadata["physical_parameters"]["side_transverse_alignment"]
            )
            self.assertTrue(full_qs["dark_reduction"]["applied"])
            self.assertTrue(certificate["accepted"])
            self.assertTrue(certificate["passive_on_audit_grid"])
            reaudit = certificate["current_transfer_reaudit"]
            self.assertTrue(reaudit["accepted"])
            self.assertTrue(reaudit["passive_on_audit_grid"])
            self.assertEqual(
                reaudit["rms_tolerance"],
                full_qs["dark_reduction"]["rms_tolerance"],
            )
            self.assertEqual(
                reaudit["max_tolerance"],
                full_qs["dark_reduction"]["max_tolerance"],
            )
            self.assertLessEqual(
                reaudit["normalized_rms"],
                reaudit["rms_tolerance"],
            )
            self.assertLessEqual(
                reaudit["max_normalized_error"],
                reaudit["max_tolerance"],
            )
            self.assertEqual(full_qs["spatial_order_max"], 80)
            self.assertEqual(full_qs["modal_fit_normalized_rms_limit"], 0.03)
            self.assertEqual(full_qs["modal_fit_relative_error_limit"], 0.06)
            self.assertEqual(full_qs["exact_spatial_mode_count"], 1640)
            self.assertEqual(
                full_qs["dynamic_spatial_mode_count"],
                1 + full_qs["reduced_dark_node_count"],
            )
            self.assertLessEqual(
                certificate["max_normalized_rms"],
                full_qs["dark_reduction"]["rms_tolerance"],
            )
            self.assertLessEqual(
                certificate["max_normalized_error"],
                full_qs["dark_reduction"]["max_tolerance"],
            )
            mode_table = metadata["coupling_by_model"]["spheroid_full"][
                "exact_equatorial_mode_table"
            ]
            self.assertEqual(len(mode_table), 1640)
            self.assertEqual(
                mode_table[0],
                {"index": 0, "n": 1, "m": 0, "sector": "cos"},
            )
            dynamic_rows = full_qs["full_modal_outputs_au_rows"]
            self.assertEqual(
                len(dynamic_rows),
                full_qs["dynamic_spatial_mode_count"],
            )
            self.assertEqual(
                [row["full_modal_outputs_au_row"] for row in dynamic_rows],
                list(range(full_qs["dynamic_spatial_mode_count"])),
            )
            self.assertEqual(dynamic_rows[0]["role"], "bright")
            self.assertEqual(dynamic_rows[0]["source_mode_indices"], [0])
            self.assertEqual(
                [row["depolarization"] for row in dynamic_rows[1:]],
                certificate["depolarization_nodes"],
            )
            self.assertEqual(
                [row["reaction_weight_au_minus3"] for row in dynamic_rows[1:]],
                certificate["weights_au_minus3"],
            )
            self.assertEqual(
                [row["source_mode_indices"] for row in dynamic_rows[1:]],
                certificate["source_mode_indices"],
            )
            self.assertEqual(
                sorted(
                    index
                    for group in certificate["source_mode_indices"]
                    for index in group
                ),
                list(range(1, full_qs["exact_spatial_mode_count"])),
            )
            self.assertTrue(full_qs["modal_fit_accepted"])
            self.assertTrue(full_qs["modal_fit_passive_on_audit_grid"])
            self.assertIn("modal_fit_K_normalized_rms", full_qs)
            self.assertIn("modal_fit_K_max_relative_error", full_qs)

            legacy_coupling = metadata["coupling_by_model"]["legacy"]
            legacy_B = complex(
                legacy_coupling["B_at_carrier"]["real"],
                legacy_coupling["B_at_carrier"]["imag"],
            )
            legacy_K = complex(
                legacy_coupling["K_at_carrier_au_minus3"]["real"],
                legacy_coupling["K_at_carrier_au_minus3"]["imag"],
            )
            self.assertTrue(
                np.isclose(
                    legacy_K,
                    legacy_B * legacy_coupling["J_au_minus3"],
                    rtol=5.0e-15,
                    atol=0.0,
                )
            )
            with np.load(run_dir / "pulse_traces_adaptive.npz") as traces:
                self.assertEqual(
                    traces["full_modal_outputs_au"].shape[0],
                    full_qs["dynamic_spatial_mode_count"],
                )

    def test_side_tangential_low_order_keeps_direct_reference_backend(self) -> None:
        with TemporaryDirectory() as directory:
            run_dir = Path(
                run_pulse_comparison(
                    output_dir=directory,
                    orientation="trans",
                    qd_placement="side",
                    side_transverse_alignment="tangential",
                    spatial_order_max=2,
                    material_fit_modes=9,
                    pulse_E0_au=1.0e-7,
                    post_fs=20.0,
                    common_time_points=101,
                    spectral_window_policy="ignore",
                    tail_policy="ignore",
                    spatial_convergence_policy="ignore",
                    concurrent=False,
                    make_plots=False,
                    show=False,
                )
            )
            metadata = json.loads(
                (run_dir / "metadata.json").read_text(encoding="utf-8")
            )
            full_qs = metadata["full_qs"]
            self.assertFalse(full_qs["dark_reduction"]["applied"])
            self.assertEqual(full_qs["reduced_dark_node_count"], 0)
            self.assertEqual(
                full_qs["exact_spatial_mode_count"],
                full_qs["dynamic_spatial_mode_count"],
            )
            self.assertEqual(
                metadata["physical_parameters"]["side_transverse_alignment"],
                "tangential",
            )

    def test_side_radial_does_not_hide_the_default_material_fit_gate(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                RuntimeError,
                "transformed common-material realization does not meet",
            ):
                run_pulse_comparison(
                    output_dir=directory,
                    orientation="trans",
                    qd_placement="side",
                    side_transverse_alignment="radial",
                    spatial_order_max=80,
                    material_fit_modes=9,
                    pulse_E0_au=1.0e-7,
                    post_fs=20.0,
                    common_time_points=101,
                    spectral_window_policy="ignore",
                    tail_policy="ignore",
                    concurrent=False,
                    make_plots=False,
                    show=False,
                )

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
