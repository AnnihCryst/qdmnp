"""Tests for the plot-only transient work-loss spectrum program."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")
import numpy as np

from article_observables.qd_mnp_material_modes_artifact import atomic_write_npz
from article_observables.qd_mnp_plot_work_loss_spectrum_material_comparison import (
    SCHEMA_NAME,
    create_work_loss_spectrum_figure,
    load_work_loss_spectrum_artifact,
    plot_work_loss_spectrum_material_comparison,
)


class WorkLossSpectrumPlotterTests(unittest.TestCase):
    @staticmethod
    def _payload(*, certified: bool = True) -> dict[str, np.ndarray]:
        branch_count = 2
        channel_count = 2
        fluence_count = 3
        energy = np.linspace(1.90, 2.18, 101)
        fluence = np.asarray([5.0e-9, 5.0e-7, 5.0e-6])
        bare = np.empty((branch_count, channel_count, energy.size), dtype=float)
        sigma = np.empty(
            (branch_count, channel_count, fluence_count, energy.size),
            dtype=float,
        )
        for branch_index in range(branch_count):
            for channel_index in range(channel_count):
                background = (
                    1.1e-11
                    + 0.12e-11 * branch_index
                    + 0.06e-11 * channel_index
                )
                bare[branch_index, channel_index] = background * (
                    1.0
                    + 0.25
                    / (1.0 + ((energy - 2.02) / (0.07 + 0.005 * branch_index)) ** 2)
                )
                for fluence_index in range(fluence_count):
                    narrow = np.exp(-((energy - 2.042) / 0.018) ** 2)
                    sigma[branch_index, channel_index, fluence_index] = (
                        bare[branch_index, channel_index]
                        + (fluence_index - 0.7)
                        * (1.0 + 0.25 * branch_index)
                        * 8.0e-13
                        * narrow
                    )

        support = np.ones((fluence_count, energy.size), dtype=bool)
        support[:, :2] = False
        support[:, -2:] = False
        sigma[..., ~support[0]] = np.nan
        delta = sigma - bare[:, :, None, :]
        solve_shape = (branch_count, channel_count, fluence_count)
        model_shape = (branch_count, channel_count)
        solve_certificate = np.ones(solve_shape, dtype=bool)
        if not certified:
            solve_certificate[1, 0, 1] = False
        modal = np.ones(model_shape, dtype=bool)
        # An N=1 accuracy miss is intentional and must not invalidate an
        # otherwise passive and stable comparison artifact.
        modal[0] = False
        return {
            "branch_id": np.asarray(["one", "multi"], dtype="U16"),
            "branch_material_mode_count": np.asarray([1, 9], dtype=np.int64),
            "channel_key": np.asarray(["axis_long", "side_long"], dtype="U32"),
            "channel_label": np.asarray(
                ["tip, longitudinal", "side, longitudinal"], dtype="U48"
            ),
            "selected_fluence_j_cm2": fluence,
            "selected_fluence_label": np.asarray(
                ["linear", "threshold", "nonlinear"], dtype="U16"
            ),
            "carrier_energy_eV": np.asarray(2.042),
            "energy_eV": energy,
            "sigma_qs_work_cm2": sigma,
            "bare_mnp_sigma_qs_work_cm2": bare,
            "delta_sigma_qs_work_cm2": delta,
            "spectrum_support_mask": support,
            "solver_success": np.ones(solve_shape, dtype=bool),
            "t_final_reached": np.ones(solve_shape, dtype=bool),
            "state_is_finite": np.ones(solve_shape, dtype=bool),
            "response_tail_converged": solve_certificate,
            "spectrum_window_converged": np.ones(solve_shape, dtype=bool),
            "work_nonnegative_within_tolerance": np.ones(solve_shape, dtype=bool),
            "density_matrix_positive": np.ones(solve_shape, dtype=bool),
            "incident_ft_converged": np.ones(solve_shape, dtype=bool),
            "energy_grid_converged": np.ones(solve_shape, dtype=bool),
            "spatial_convergence_accepted": np.ones(model_shape, dtype=bool),
            "modal_fit_accepted": modal,
            "fit_passive": np.ones(model_shape, dtype=bool),
            "bright_stable": np.ones(model_shape, dtype=bool),
            "coupled_stable": np.ones(model_shape, dtype=bool),
        }

    @staticmethod
    def _metadata() -> dict[str, object]:
        return {
            "schema_name": SCHEMA_NAME,
            "schema_version": 1,
            "multi_fit_mode_count": 9,
            "observable": (
                "operational local-QS work-loss estimate; not separately "
                "calculated metal heating"
            ),
        }

    def _write(
        self,
        path: Path,
        *,
        certified: bool = True,
        modifier=None,
    ) -> None:
        payload = self._payload(certified=certified)
        if modifier is not None:
            modifier(payload)
        atomic_write_npz(path, payload, self._metadata())

    def test_loader_accepts_intentional_one_mode_accuracy_miss(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "spectra.npz"
            self._write(artifact)
            loaded = load_work_loss_spectrum_artifact(artifact)
            self.assertEqual(loaded["quality_failures"], [])
            self.assertEqual(
                loaded["arrays"]["sigma_qs_work_cm2"].shape,
                (2, 2, 3, 101),
            )

    def test_plotter_uses_saved_npz_without_solver_imports(self) -> None:
        plotter = (
            Path(__file__).resolve().parents[1]
            / "article_observables"
            / "qd_mnp_plot_work_loss_spectrum_material_comparison.py"
        )
        source = plotter.read_text(encoding="utf-8")
        self.assertNotIn("import scipy", source)
        self.assertNotIn("qd_mnp_full_qs_model", source)
        self.assertNotIn("qd_mnp_rational_fit", source)
        self.assertNotIn("qd_mnp_spheroid_green", source)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "spectra.npz"
            figure = root / "spectra.png"
            self._write(artifact)
            returned = plot_work_loss_spectrum_material_comparison(
                artifact,
                figure,
                channel="axis_long",
                dpi=72,
            )
            self.assertEqual(returned, figure)
            self.assertTrue(figure.is_file())
            self.assertGreater(figure.stat().st_size, 1000)

    def test_loader_rejects_corrupted_hybrid_minus_bare_identity(self) -> None:
        def corrupt(payload: dict[str, np.ndarray]) -> None:
            payload["delta_sigma_qs_work_cm2"] = np.array(
                payload["delta_sigma_qs_work_cm2"], copy=True
            )
            payload["delta_sigma_qs_work_cm2"][1, 0, 1, 50] += 2.0e-13

        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "bad_delta.npz"
            self._write(artifact, modifier=corrupt)
            with self.assertRaisesRegex(ValueError, "hybrid minus bare MNP"):
                load_work_loss_spectrum_artifact(artifact)

    def test_failed_certificate_requires_override_and_adds_watermark(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "uncertified.npz"
            self._write(artifact, certified=False)
            with self.assertRaisesRegex(ValueError, "uncertified work-loss spectrum"):
                load_work_loss_spectrum_artifact(artifact)
            loaded = load_work_loss_spectrum_artifact(
                artifact,
                allow_unconverged=True,
            )
            self.assertTrue(loaded["quality_failures"])
            figure = create_work_loss_spectrum_figure(loaded)
            self.assertIn(
                "UNCERTIFIED DIAGNOSTIC",
                [text.get_text() for text in figure.texts],
            )
            matplotlib.pyplot.close(figure)

    def test_boolean_certificate_requires_exact_boolean_dtype(self) -> None:
        def corrupt(payload: dict[str, np.ndarray]) -> None:
            payload["solver_success"] = payload["solver_success"].astype(np.int8)

        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "bad_bool.npz"
            self._write(artifact, modifier=corrupt)
            with self.assertRaisesRegex(ValueError, "exact NumPy boolean dtype"):
                load_work_loss_spectrum_artifact(artifact)

    def test_nonfinite_sample_inside_supported_band_is_rejected(self) -> None:
        def corrupt(payload: dict[str, np.ndarray]) -> None:
            payload["sigma_qs_work_cm2"] = np.array(
                payload["sigma_qs_work_cm2"], copy=True
            )
            payload["sigma_qs_work_cm2"][0, 0, 0, 50] = np.nan
            payload["delta_sigma_qs_work_cm2"] = (
                payload["sigma_qs_work_cm2"]
                - payload["bare_mnp_sigma_qs_work_cm2"][:, :, None, :]
            )

        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "bad_supported_sample.npz"
            self._write(artifact, modifier=corrupt)
            with self.assertRaisesRegex(ValueError, "Supported work-loss samples"):
                load_work_loss_spectrum_artifact(artifact)


if __name__ == "__main__":
    unittest.main()
