"""Artifact and branch contracts for the old-vs-full-QS comparison runner."""

import csv
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt

from qd_mnp_spheroid_green_comparison import (
    OWNED_PLOTS,
    _create_unique_run_dir,
    parse_args,
    run_comparison,
)


class SpheroidGreenComparisonTests(unittest.TestCase):
    def test_smoke_run_exports_three_models_and_convergence_metadata(self) -> None:
        with TemporaryDirectory() as directory:
            run_dir = run_comparison(
                output_dir=directory,
                orientations=("long",),
                energy_window_eV=(2.0, 2.08),
                energy_points=41,
                target_energy_eV=2.042,
                n_max=20,
                multipole_orders=(1, 2, 4, 8, 16, 20),
                gaps_nm=(1.0, 5.0, 20.0),
                convergence_policy="ignore",
                make_plots=False,
            )
            run_path = Path(run_dir)
            expected = {
                "metadata.json",
                "linear_spectrum.csv",
                "response_coefficients.csv",
                "multipole_convergence.csv",
                "gap_sweep.csv",
            }
            self.assertTrue(expected.issubset({path.name for path in run_path.iterdir()}))
            self.assertTrue(all(not (run_path / name).exists() for name in OWNED_PLOTS))

            metadata = json.loads(
                (run_path / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                set(metadata["implementation"]),
                {"legacy", "spheroid_n1", "spheroid_full"},
            )
            self.assertEqual(
                metadata["normalization"]["reciprocal_identity"],
                "K_1=B^2/A without complex conjugation",
            )
            self.assertIn("long", metadata["orientation_diagnostics"])
            self.assertEqual(metadata["numerical_settings"]["n_max"], 20)
            self.assertEqual(
                metadata["numerical_settings"]["convergence_policy"],
                "ignore",
            )
            self.assertEqual(
                set(metadata["orientation_diagnostics"]["long"]
                    ["coupling_at_target_energy"])
                - {"energy_eV", "material_response"},
                {"legacy", "spheroid_n1", "spheroid_full"},
            )

            with (run_path / "linear_spectrum.csv").open(
                newline="",
                encoding="utf-8",
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(
                {row["model"] for row in rows},
                {"legacy", "spheroid_n1", "spheroid_full"},
            )
            self.assertEqual(len(rows), 3 * 41)

            with (run_path / "multipole_convergence.csv").open(
                newline="",
                encoding="utf-8",
            ) as stream:
                convergence_rows = list(csv.DictReader(stream))
            self.assertIn(
                "K_cumulative_au_minus3_real",
                convergence_rows[0],
            )
            self.assertIn(
                "K_cumulative_au_minus3_imag",
                convergence_rows[0],
            )
            self.assertIn(
                "K_order_contribution_au_minus3_real",
                convergence_rows[0],
            )

            with (run_path / "gap_sweep.csv").open(
                newline="",
                encoding="utf-8",
            ) as stream:
                gap_rows = list(csv.DictReader(stream))
            self.assertEqual(len(gap_rows), 3 * 3)
            self.assertTrue(
                all("full_series_converged" in row for row in gap_rows)
            )
            for row in gap_rows:
                self.assertEqual(
                    int(row["exact_mode_count"]),
                    20 if row["model"] == "spheroid_full" else 1,
                )

    def test_plotting_run_creates_every_owned_figure(self) -> None:
        with TemporaryDirectory() as directory:
            run_dir = Path(
                run_comparison(
                    output_dir=directory,
                    orientations=("trans",),
                    energy_window_eV=(2.0, 2.08),
                    energy_points=21,
                    target_energy_eV=2.042,
                    n_max=8,
                    multipole_orders=(1, 2, 4, 8),
                    gaps_nm=(1.0, 5.0),
                    convergence_policy="ignore",
                    make_plots=True,
                    show=False,
                )
            )
            for name in OWNED_PLOTS:
                with self.subTest(name=name):
                    self.assertGreater((run_dir / name).stat().st_size, 0)

    def test_n_max_uses_automatic_orders_and_cli_leaves_them_automatic(self) -> None:
        with patch.object(sys, "argv", ["green", "--n-max", "7"]):
            args = parse_args()
        self.assertEqual(args.n_max, 7)
        self.assertIsNone(args.multipole_orders)
        self.assertEqual(args.qd_placement, "axis")
        self.assertIsNone(args.side_transverse_alignment)

        with TemporaryDirectory() as directory:
            run_dir = run_comparison(
                output_dir=directory,
                orientations=("long",),
                energy_window_eV=(2.03, 2.05),
                energy_points=3,
                target_energy_eV=2.042,
                n_max=7,
                gaps_nm=(5.0,),
                convergence_policy="ignore",
                make_plots=False,
            )
            metadata = json.loads(
                (Path(run_dir) / "metadata.json").read_text(encoding="utf-8")
            )
            settings = metadata["numerical_settings"]
            self.assertEqual(settings["multipole_orders_source"], "automatic")
            self.assertEqual(settings["multipole_orders"][-1], 7)
            self.assertTrue(
                all(order <= 7 for order in settings["multipole_orders"])
            )

    def test_side_long_and_radial_transverse_export_exact_mode_metadata(self) -> None:
        with TemporaryDirectory() as directory:
            run_dir = Path(
                run_comparison(
                    output_dir=directory,
                    orientations=("long", "trans"),
                    qd_placement="side",
                    side_transverse_alignment="radial",
                    energy_window_eV=(2.03, 2.05),
                    energy_points=3,
                    target_energy_eV=2.042,
                    n_max=4,
                    gaps_nm=(1.0,),
                    convergence_policy="ignore",
                    make_plots=False,
                )
            )
            metadata = json.loads(
                (run_dir / "metadata.json").read_text(encoding="utf-8")
            )
            settings = metadata["numerical_settings"]
            self.assertEqual(settings["qd_placement"], "side")
            self.assertEqual(settings["side_transverse_alignment"], "radial")

            long = metadata["orientation_diagnostics"]["long"]
            radial = metadata["orientation_diagnostics"]["trans"]
            self.assertIsNone(long["side_transverse_alignment"])
            self.assertEqual(radial["side_transverse_alignment"], "radial")
            self.assertEqual(long["exact_mode_count"], 6)
            self.assertEqual(radial["exact_mode_count"], 8)
            self.assertEqual(
                long["mode_metadata"]["bright_mode"],
                {"index": 0, "n": 1, "m": 0, "sector": "cos"},
            )
            self.assertEqual(
                radial["mode_metadata"]["bright_mode"],
                {"index": 0, "n": 1, "m": 1, "sector": "cos"},
            )
            self.assertEqual(
                long["coupling_at_target_energy"]["spheroid_full"][
                    "exact_mode_count"
                ],
                6,
            )
            self.assertEqual(
                radial["coupling_at_target_energy"]["spheroid_full"][
                    "exact_mode_count"
                ],
                8,
            )
            self.assertAlmostEqual(
                long["common_physical_parameters"]["directional_mnp_radius_nm"],
                7.0,
            )

            with (run_dir / "gap_sweep.csv").open(
                newline="",
                encoding="utf-8",
            ) as stream:
                gap_rows = list(csv.DictReader(stream))
            self.assertEqual(len(gap_rows), 6)
            self.assertTrue(
                all(float(row["center_distance_nm"]) == 10.0 for row in gap_rows)
            )
            self.assertTrue(all(row["qd_placement"] == "side" for row in gap_rows))
            self.assertEqual(
                {row["side_transverse_alignment"] for row in gap_rows},
                {"", "radial"},
            )
            for row in gap_rows:
                expected_mode_count = {
                    "legacy": 1,
                    "spheroid_n1": 1,
                    "spheroid_full": 6 if row["orientation"] == "long" else 8,
                }[row["model"]]
                self.assertEqual(int(row["exact_mode_count"]), expected_mode_count)

    def test_side_tangential_cli_and_mode_sector(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "green",
                "--qd-placement",
                "side",
                "--orientations",
                "trans",
                "--side-transverse-alignment",
                "tangential",
            ],
        ):
            args = parse_args()
        self.assertEqual(args.qd_placement, "side")
        self.assertEqual(args.orientations, ["trans"])
        self.assertEqual(args.side_transverse_alignment, "tangential")

        with TemporaryDirectory() as directory:
            run_dir = Path(
                run_comparison(
                    output_dir=directory,
                    orientations=("trans",),
                    qd_placement="side",
                    side_transverse_alignment="tangential",
                    energy_window_eV=(2.03, 2.05),
                    energy_points=3,
                    target_energy_eV=2.042,
                    n_max=4,
                    gaps_nm=(1.0,),
                    convergence_policy="ignore",
                    make_plots=False,
                )
            )
            metadata = json.loads(
                (run_dir / "metadata.json").read_text(encoding="utf-8")
            )
            diagnostic = metadata["orientation_diagnostics"]["trans"]
            self.assertEqual(diagnostic["exact_mode_count"], 6)
            self.assertEqual(
                diagnostic["mode_metadata"]["bright_mode"],
                {"index": 0, "n": 1, "m": 1, "sector": "sin"},
            )
            self.assertEqual(
                diagnostic["common_physical_parameters"]["G"],
                -1.0,
            )

    def test_invalid_side_alignment_combinations_fail_before_export(self) -> None:
        invalid = (
            {
                "orientations": ("trans",),
                "qd_placement": "side",
                "side_transverse_alignment": None,
            },
            {
                "orientations": ("long",),
                "qd_placement": "side",
                "side_transverse_alignment": "radial",
            },
            {
                "orientations": ("long",),
                "qd_placement": "axis",
                "side_transverse_alignment": "tangential",
            },
        )
        for options in invalid:
            with self.subTest(options=options), TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ValueError, "side_transverse_alignment"):
                    run_comparison(
                        output_dir=directory,
                        energy_points=3,
                        gaps_nm=(1.0,),
                        make_plots=False,
                        **options,
                    )
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_runner_does_not_close_unrelated_matplotlib_figures(self) -> None:
        sentinel = plt.figure()
        try:
            with TemporaryDirectory() as directory:
                run_comparison(
                    output_dir=directory,
                    orientations=("long",),
                    energy_window_eV=(2.03, 2.05),
                    energy_points=3,
                    target_energy_eV=2.042,
                    n_max=2,
                    gaps_nm=(5.0,),
                    convergence_policy="ignore",
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
                "qd_mnp_spheroid_green_comparison.timestamped_run_dir",
                return_value=fixed,
            ):
                first = _create_unique_run_dir(directory)
                second = _create_unique_run_dir(directory)
            self.assertEqual(first, fixed)
            self.assertEqual(second.name, "20260101_000000_001")

    def test_raise_convergence_policy_fails_before_export(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "not converged"):
                run_comparison(
                    output_dir=directory,
                    orientations=("long",),
                    energy_window_eV=(2.03, 2.05),
                    energy_points=3,
                    target_energy_eV=2.042,
                    n_max=2,
                    gaps_nm=(1.0,),
                    convergence_policy="raise",
                    make_plots=False,
                )
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_invalid_grid_is_rejected_before_creating_output(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "at least 2"):
                run_comparison(
                    output_dir=directory,
                    energy_points=1,
                    make_plots=False,
                )
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
