"""Contracts for the Yan et al. Fig. 2(a) full-QS reconstruction."""

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import matplotlib.pyplot as plt
import numpy as np

from literature_reproductions.yan2008_fig2a import (
    CSV_FILENAME,
    METADATA_FILENAME,
    PLOT_FILENAME,
    _paper_raster_comparison_mode,
    _plot_fig2a,
    calculate_fig2a,
    parse_args,
    run_reproduction,
)


class Yan2008Fig2AReproductionTests(unittest.TestCase):
    def test_full_qs_spherical_limit_matches_yan_equation_15(self) -> None:
        result = calculate_fig2a(
            energy_window_eV=(1.5, 3.5),
            energy_points=81,
            n_max=10,
        )
        self.assertEqual(result.g_full_qs_meV.shape, (10, 81))
        self.assertLess(result.max_absolute_difference_meV, 1.0e-12)
        self.assertLess(result.max_relative_difference, 1.0e-12)

        peak_index = int(np.argmax(result.g_full_qs_meV[-1].real))
        ratio = (
            result.g_full_qs_meV[-1, peak_index].real
            / result.g_full_qs_meV[0, peak_index].real
        )
        # Yan's text describes the N=10 shift as almost seven times N=1.
        self.assertGreater(ratio, 6.5)
        self.assertLess(ratio, 7.2)

    def test_runner_exports_traceable_data_and_metadata(self) -> None:
        with TemporaryDirectory() as directory:
            run_dir = Path(
                run_reproduction(
                    output_dir=directory,
                    article_pdf=Path(directory) / "missing.pdf",
                    energy_points=31,
                    n_max=4,
                    make_plot=False,
                )
            )
            self.assertTrue((run_dir / CSV_FILENAME).is_file())
            self.assertTrue((run_dir / METADATA_FILENAME).is_file())
            self.assertFalse((run_dir / PLOT_FILENAME).exists())

            metadata = json.loads(
                (run_dir / METADATA_FILENAME).read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["article"]["figure"], "2(a)")
            self.assertIsNone(metadata["article"]["pdf_sha256"])
            self.assertEqual(metadata["model"]["orientation"], "long")
            self.assertIn("not restated by Yan", metadata["parameter_provenance"]["eps_host"])
            self.assertLess(
                metadata["comparison"]["max_relative_full_qs_vs_eq15"],
                1.0e-12,
            )
            self.assertEqual(
                metadata["comparison"][
                    "paper_raster_peak_comparison_available"
                ],
                False,
            )
            self.assertIsNone(
                metadata["comparison"]["paper_raster_peak_comparison"]
            )
            self.assertFalse(
                metadata["comparison"]["paper_text_target_evaluated"]
            )
            self.assertIsNone(
                metadata["comparison"]["paper_text_target_comparison"]
            )
            self.assertIn(
                "N=10",
                metadata["comparison"][
                    "paper_raster_peak_comparison_unavailable_reason"
                ],
            )

            with (run_dir / CSV_FILENAME).open(
                newline="",
                encoding="utf-8",
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 31)
            self.assertIn("G_N4_full_qs_real_meV", rows[0])
            self.assertIn("G_N4_yan_eq15_imag_meV", rows[0])

    def test_canonical_profile_records_only_graphical_peak_spot_checks(self) -> None:
        with TemporaryDirectory() as directory:
            run_dir = Path(
                run_reproduction(
                    output_dir=directory,
                    article_pdf=Path(directory) / "missing.pdf",
                    energy_window_eV=(2.2, 2.65),
                    energy_points=181,
                    n_max=10,
                    make_plot=False,
                )
            )
            metadata = json.loads(
                (run_dir / METADATA_FILENAME).read_text(encoding="utf-8")
            )
            comparison = metadata["comparison"]
            self.assertTrue(comparison["paper_raster_peak_comparison_available"])
            self.assertTrue(comparison["paper_text_target_evaluated"])
            self.assertGreater(
                comparison["paper_text_target_comparison"][
                    "real_G_N10_over_N1_at_N10_peak"
                ],
                6.5,
            )
            self.assertIsNone(
                comparison["paper_raster_peak_comparison_unavailable_reason"]
            )
            peaks = comparison["paper_raster_peak_comparison"]
            self.assertEqual(set(peaks), {"N10_real_peak", "N10_imag_peak"})
            for peak in peaks.values():
                self.assertIn("G_difference_meV", peak)
                self.assertIn("G_difference_in_readout_uncertainties", peak)
                self.assertIsInstance(peak["inside_readout_intervals"], bool)

    def test_raster_check_rejects_changed_profile_and_cropped_window(self) -> None:
        canonical = calculate_fig2a(
            energy_window_eV=(2.2, 2.65),
            energy_points=91,
            n_max=10,
        )
        mode_index, reason = _paper_raster_comparison_mode(
            canonical,
            sphere_radius_nm=15.0,
            centre_distance_nm=20.0,
            eps_host=1.2,
            eps_qd=6.0,
            transition_dipole_e_nm=0.65,
        )
        self.assertIsNone(mode_index)
        self.assertIn("eps_host", reason or "")

        with TemporaryDirectory() as directory:
            run_dir = run_reproduction(
                output_dir=directory,
                article_pdf=Path(directory) / "missing.pdf",
                energy_window_eV=(2.2, 2.65),
                energy_points=91,
                n_max=10,
                eps_host=1.2,
                make_plot=False,
            )
            metadata = json.loads(
                (run_dir / METADATA_FILENAME).read_text(encoding="utf-8")
            )
            self.assertIn(
                "runtime override",
                metadata["parameter_provenance"]["eps_host"],
            )
            self.assertNotIn(
                "completed with eps_e=1",
                metadata["parameter_provenance"]["eps_host"],
            )

        cropped = calculate_fig2a(
            energy_window_eV=(2.3, 2.55),
            energy_points=101,
            n_max=10,
        )
        mode_index, reason = _paper_raster_comparison_mode(
            cropped,
            sphere_radius_nm=15.0,
            centre_distance_nm=20.0,
            eps_host=1.0,
            eps_qd=6.0,
            transition_dipole_e_nm=0.65,
        )
        self.assertIsNone(mode_index)
        self.assertIn("search window", reason or "")

    def test_plot_contains_real_and_imag_eq15_markers_for_both_limits(self) -> None:
        result = calculate_fig2a(energy_points=41, n_max=3)
        with TemporaryDirectory() as directory:
            figure = _plot_fig2a(
                Path(directory),
                result,
                show_paper_anchors=False,
            )
            try:
                plotted_lines = figure.axes[0].lines
                real_markers = [
                    line
                    for line in plotted_lines
                    if line.get_marker() == "o" and len(line.get_xdata()) > 0
                ]
                imag_markers = [
                    line
                    for line in plotted_lines
                    if line.get_marker() == "s" and len(line.get_xdata()) > 0
                ]
                self.assertEqual(len(real_markers), 2)
                self.assertEqual(len(imag_markers), 2)
            finally:
                plt.close(figure)

    def test_plotting_creates_owned_figure_without_closing_unrelated_figures(self) -> None:
        sentinel = plt.figure()
        try:
            with TemporaryDirectory() as directory:
                run_dir = Path(
                    run_reproduction(
                        output_dir=directory,
                        article_pdf=Path(directory) / "missing.pdf",
                        energy_points=41,
                        n_max=3,
                        make_plot=True,
                        show=False,
                    )
                )
                self.assertGreater((run_dir / PLOT_FILENAME).stat().st_size, 0)
            self.assertTrue(plt.fignum_exists(sentinel.number))
        finally:
            plt.close(sentinel)

    def test_invalid_noninteger_grid_and_inside_sphere_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "integer"):
            calculate_fig2a(energy_points=10.5)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "outside"):
            calculate_fig2a(sphere_radius_nm=15.0, centre_distance_nm=15.0)

    def test_cli_defaults_match_article_target(self) -> None:
        args = parse_args([])
        self.assertEqual(args.n_max, 10)
        self.assertEqual(args.sphere_radius_nm, 15.0)
        self.assertEqual(args.centre_distance_nm, 20.0)
        self.assertEqual(args.eps_qd, 6.0)
        self.assertEqual(args.transition_dipole_e_nm, 0.65)


if __name__ == "__main__":
    unittest.main()
