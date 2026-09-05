from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import numpy as np
from scipy.integrate import solve_ivp

from article_observables.qd_mnp_calculate_work_loss_fluence import (
    ARTICLE_CHANNELS,
    GRID_AUDIT_OBSERVABLES,
    PUBLICATION_GRID,
    WORK_LOSS_FLUENCE_SCHEMA,
    WORK_LOSS_FLUENCE_SCHEMA_VERSION,
    _bare_mnp_pulse_work_spectral_average,
    calculate_work_loss_fluence,
    fluence_grid_resolution_diagnostics,
    main as calculation_main,
    pulse_for_fluence,
    resolved_center_distance_nm,
)
from article_observables.qd_mnp_plot_work_loss_fluence import (
    load_work_loss_artifact,
    plot_work_loss_fluence,
)
from qd_mnp_rational_fit import AU_ENERGY_J, GaussianPulse, eV_to_au, fs_to_au


class WorkLossFluenceGeometryTests(unittest.TestCase):
    def test_publication_grid_is_dense_and_uniform_in_field_amplitude(self) -> None:
        self.assertGreaterEqual(int(PUBLICATION_GRID["points"]), 65)
        self.assertEqual(PUBLICATION_GRID["scale"], "sqrt")
        grid = np.linspace(
            np.sqrt(PUBLICATION_GRID["fluence_min_j_cm2"]),
            np.sqrt(PUBLICATION_GRID["fluence_max_j_cm2"]),
            int(PUBLICATION_GRID["points"]),
        ) ** 2
        np.testing.assert_allclose(
            np.diff(np.sqrt(grid)),
            np.diff(np.sqrt(grid))[0],
            rtol=2.0e-13,
            atol=0.0,
        )

    def test_cli_presets_resolve_to_publication_and_diagnostic_grids(self) -> None:
        with patch(
            "article_observables.qd_mnp_calculate_work_loss_fluence."
            "calculate_work_loss_fluence",
            return_value=Path("unused.npz"),
        ) as mocked:
            calculation_main(["--preset", "publication"])
            publication = mocked.call_args.kwargs
            self.assertEqual(publication["fluence_grid_scale"], "sqrt")
            self.assertGreaterEqual(publication["fluence_j_cm2"].size, 65)
            self.assertEqual(publication["fluence_grid_convergence_policy"], "raise")

            calculation_main(["--preset", "quick"])
            quick = mocked.call_args.kwargs
            self.assertEqual(quick["fluence_grid_scale"], "log")
            self.assertEqual(quick["fluence_j_cm2"].size, 3)
            self.assertEqual(quick["fluence_grid_convergence_policy"], "warn")

    def test_grid_certificate_detects_coarsening_error_and_area_step(self) -> None:
        amplitude = np.linspace(0.1, 2.1, 65)
        fluence = amplitude**2
        curves = np.stack(
            [
                1.0 + 0.2 * np.sin(amplitude),
                1.1 + 0.1 * np.cos(amplitude),
                0.2 * np.sin(amplitude),
                0.1 * np.cos(amplitude),
            ],
            axis=0,
        )[None, :, :]
        diagnostics = fluence_grid_resolution_diagnostics(
            fluence,
            curves,
            0.1 * amplitude,
            max_midpoint_normalized_error=1.0e-3,
            max_isolated_pulse_area_step_rad=0.01,
        )
        self.assertEqual(
            np.asarray(diagnostics["midpoint_normalized_error"]).shape,
            (1, len(GRID_AUDIT_OBSERVABLES)),
        )
        self.assertTrue(diagnostics["accepted"])

    def test_common_surface_gap_resolves_placement_specific_center_distance(self) -> None:
        common = dict(c_nm=15.0, a_nm=7.0, qd_radius_nm=2.0, gap_nm=1.5)
        tip_R = resolved_center_distance_nm(ARTICLE_CHANNELS["axis_long"], **common)
        side_R = resolved_center_distance_nm(ARTICLE_CHANNELS["side_long"], **common)

        self.assertAlmostEqual(tip_R, 18.5)
        self.assertAlmostEqual(side_R, 10.5)
        self.assertAlmostEqual(tip_R - common["c_nm"] - common["qd_radius_nm"], 1.5)
        self.assertAlmostEqual(side_R - common["a_nm"] - common["qd_radius_nm"], 1.5)

    def test_requested_fluence_is_reconstructed_by_exact_real_pulse_formula(self) -> None:
        requested = 2.5e-6
        pulse = pulse_for_fluence(
            requested,
            energy_eV=2.042,
            tau_fs=20.0,
            tau_kind="fwhm_intensity",
            eps_m=1.77,
        )
        self.assertAlmostEqual(
            pulse.fluence_j_cm2(eps_m=1.77),
            requested,
            delta=1.0e-14 * requested,
        )


class BareMnpSpectralReferenceTests(unittest.TestCase):
    def test_spectral_average_matches_independent_time_domain_lorentz_ade(self) -> None:
        eps_m = 1.77
        oscillator_frequency = float(eV_to_au(2.1))
        oscillator_damping = float(eV_to_au(0.45))
        oscillator_strength = 900.0 * oscillator_frequency**2
        alpha_infinity = 0.35
        polarizability_scale = 1.7e6

        class SyntheticBrightModel:
            C = polarizability_scale
            params = SimpleNamespace(eps_m=eps_m)
            fit = SimpleNamespace(
                omega_modes_au=np.asarray([oscillator_frequency], dtype=float)
            )

            @staticmethod
            def alpha_from_fit(
                energies_eV: np.ndarray,
                *,
                allow_extrapolation: bool = False,
            ) -> np.ndarray:
                del allow_extrapolation
                omega = np.asarray(eV_to_au(energies_eV), dtype=float)
                denominator = (
                    oscillator_frequency**2
                    - omega**2
                    - 1j * oscillator_damping * omega
                )
                return alpha_infinity + oscillator_strength / denominator

        pulse = GaussianPulse(
            E0_au=2.0e-5,
            omegaL_au=float(eV_to_au(2.042)),
            tau_au=float(fs_to_au(8.0)),
            tau_kind="fwhm_intensity",
        )
        spectral = _bare_mnp_pulse_work_spectral_average(
            SyntheticBrightModel(),
            pulse,
            max_relative_change=1.0e-7,
            convergence_policy="raise",
            work_passivity_policy="raise",
        )

        def rhs(time_au: float, state: np.ndarray) -> np.ndarray:
            coordinate, velocity, accumulated_work = state
            del accumulated_work
            incident = float(pulse.field(time_au))
            incident_dot = float(pulse.field_dot(time_au))
            dipole_dot = polarizability_scale * (
                velocity + alpha_infinity * incident_dot
            )
            return np.asarray(
                [
                    velocity,
                    oscillator_strength * incident
                    - oscillator_damping * velocity
                    - oscillator_frequency**2 * coordinate,
                    incident * dipole_dot,
                ]
            )

        start_au = -10.0 * pulse.sigma_t_au
        end_au = float(fs_to_au(120.0))
        solution = solve_ivp(
            rhs,
            (start_au, end_au),
            np.zeros(3),
            method="DOP853",
            rtol=2.0e-10,
            atol=1.0e-12,
            max_step=(2.0 * np.pi / oscillator_frequency) / 16.0,
        )
        self.assertTrue(solution.success)
        time_domain_sigma = float(
            solution.y[2, -1]
            * AU_ENERGY_J
            / pulse.fluence_j_cm2(eps_m=eps_m)
        )
        self.assertAlmostEqual(
            time_domain_sigma / spectral.sigma_energy_transfer_cm2,
            1.0,
            delta=2.0e-8,
        )


class WorkLossFluenceArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.artifact = cls.root / "work_loss.npz"
        calculate_work_loss_fluence(
            cls.artifact,
            channel_keys=("axis_long",),
            fluence_j_cm2=np.asarray([1.0e-8, 1.0001e-8, 1.0002e-8]),
            spatial_order_max=1,
            material_fit_modes=9,
            radiative_consistency_policy="ignore",
            fit_quality_policy="ignore",
            spatial_convergence_policy="ignore",
            spectral_window_policy="ignore",
            post_fs=100.0,
            tail_policy="ignore",
            observable_convergence_policy="ignore",
            fluence_grid_convergence_policy="ignore",
            rtol=1.0e-6,
            atol=1.0e-8,
            points_per_fastest_cycle=8.0,
            verbose=False,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_npz_contains_inputs_constants_diagnostics_and_primary_observables(self) -> None:
        loaded = load_work_loss_artifact(self.artifact, allow_unconverged=True)
        metadata = loaded["metadata"]
        arrays = loaded["arrays"]

        self.assertEqual(metadata["schema"], WORK_LOSS_FLUENCE_SCHEMA)
        self.assertEqual(metadata["schema_version"], WORK_LOSS_FLUENCE_SCHEMA_VERSION)
        self.assertEqual(metadata["inputs"]["channel_keys"], ["axis_long"])
        self.assertEqual(metadata["inputs"]["common_surface_gap_nm"], 1.0)
        self.assertIn("atomic_unit_time_s", metadata["physical_constants"])
        self.assertIn("resolved_physical_parameters", metadata["channels"][0])
        self.assertIn("response_tail_ratio", arrays)
        self.assertIn("solver_nfev", arrays)
        self.assertIn("sigma_energy_transfer_cm2", arrays)
        self.assertIn("sigma_bare_mnp_energy_transfer_cm2", arrays)
        self.assertIn("delta_sigma_energy_transfer_cm2", arrays)
        self.assertIn("observable_window_converged", arrays)
        self.assertTrue(arrays["bare_mnp_energy_integration_converged"].all())
        self.assertIn("bare_mnp_pulse_work", metadata["channels"][0])
        self.assertEqual(metadata["pulse_definition"]["envelope_center_fs"], 0.0)
        self.assertEqual(metadata["pulse_definition"]["carrier_phase_rad"], 0.0)
        self.assertEqual(metadata["initial_conditions"]["qd_bloch"]["W"], -1.0)
        self.assertIn("delta_interpretation_limit", metadata["observables"])
        self.assertIn("fluence_grid", metadata)
        self.assertIn("fluence_grid_midpoint_normalized_error", arrays)
        self.assertIn("maximum_isolated_pulse_area_step_rad", arrays)
        self.assertEqual(arrays["sigma_spectral_qs_work_loss_cm2"].shape, (1, 3))
        self.assertTrue(np.isfinite(arrays["sigma_spectral_qs_work_loss_cm2"]).all())
        np.testing.assert_allclose(
            arrays["delta_sigma_spectral_qs_work_loss_cm2"],
            arrays["sigma_spectral_qs_work_loss_cm2"]
            - arrays["sigma_bare_mnp_qs_work_loss_cm2"],
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            arrays["delta_sigma_energy_transfer_cm2"],
            arrays["sigma_energy_transfer_cm2"]
            - arrays["sigma_bare_mnp_energy_transfer_cm2"],
            rtol=0.0,
            atol=0.0,
        )
        self.assertNotAlmostEqual(
            float(arrays["sigma_bare_mnp_qs_work_loss_cm2"][0, 0]),
            float(arrays["sigma_bare_mnp_energy_transfer_cm2"][0, 0]),
            delta=1.0e-15,
        )

    def test_plotter_creates_figure_from_npz_only(self) -> None:
        output = self.root / "styled.png"
        returned = plot_work_loss_fluence(
            self.artifact,
            output,
            dpi=72,
            allow_unconverged=True,
        )
        self.assertEqual(returned, output)
        self.assertTrue(output.is_file())
        self.assertGreater(output.stat().st_size, 1000)

    def test_plotter_rejects_uncertified_result_by_default(self) -> None:
        with self.assertRaisesRegex(ValueError, "uncertified work-loss artifact"):
            load_work_loss_artifact(self.artifact)

    def test_plotter_rejects_incomplete_artifact(self) -> None:
        path = self.root / "incomplete.npz"
        metadata = {
            "schema": WORK_LOSS_FLUENCE_SCHEMA,
            "schema_version": WORK_LOSS_FLUENCE_SCHEMA_VERSION,
        }
        np.savez_compressed(
            path,
            metadata_json=np.asarray(json.dumps(metadata)),
            channel_key=np.asarray(["axis_long"]),
        )
        with self.assertRaisesRegex(ValueError, "missing required array"):
            load_work_loss_artifact(path)

    def _write_modified_artifact(self, name: str, modifier) -> Path:
        path = self.root / name
        with np.load(self.artifact, allow_pickle=False) as stored:
            payload = {key: np.array(stored[key], copy=True) for key in stored.files}
        modifier(payload)
        np.savez_compressed(path, **payload)
        return path

    def test_loader_requires_exact_boolean_storage_dtype(self) -> None:
        path = self._write_modified_artifact(
            "bad_bool.npz",
            lambda payload: payload.__setitem__(
                "solver_success", payload["solver_success"].astype(np.int8)
            ),
        )
        with self.assertRaisesRegex(ValueError, "exact NumPy boolean dtype"):
            load_work_loss_artifact(path, allow_unconverged=True)

    def test_loader_rejects_inconsistent_bare_reference(self) -> None:
        def corrupt(payload) -> None:
            payload["bare_mnp_energy_transfer_by_channel_cm2"] *= 1.01

        path = self._write_modified_artifact("bad_bare.npz", corrupt)
        with self.assertRaisesRegex(ValueError, "fluence-independent copy"):
            load_work_loss_artifact(path, allow_unconverged=True)

    def test_loader_recomputes_grid_certificate(self) -> None:
        def corrupt(payload) -> None:
            payload["fluence_grid_curve_scale_cm2"] = np.asarray(
                payload["fluence_grid_curve_scale_cm2"], dtype=float
            ) * 2.0

        path = self._write_modified_artifact("bad_grid.npz", corrupt)
        with self.assertRaisesRegex(ValueError, "saved work curves"):
            load_work_loss_artifact(path, allow_unconverged=True)

    def test_approximate_modal_accuracy_does_not_require_unconverged_override(self) -> None:
        def make_numerically_certified_approximation(payload) -> None:
            for name in (
                "solver_success",
                "t_final_reached",
                "response_tail_converged",
                "observable_window_converged",
                "work_nonnegative_within_tolerance",
                "bare_mnp_work_nonnegative_within_tolerance",
            ):
                payload[name][...] = True
            for name in (
                "response_tail_ratio",
                "sigma_spectral_half_window_relative_change",
                "sigma_energy_half_window_relative_change",
                "pulse_spectral_leakage",
                "qd_source_spectral_leakage",
                "mnp_dipole_spectral_leakage",
                "mnp_drive_spectral_leakage",
                "mnp_field_spectral_leakage",
                "min_density_eigenvalue",
            ):
                payload[name][...] = 0.0
            metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
            full_qs = metadata["channels"][0]["full_qs"]
            full_qs["spatial_convergence_accepted"] = True
            full_qs["linearized_ground_state_stable"] = True
            full_qs["modal_fit_passive_on_audit_grid"] = True
            full_qs["modal_fit_accepted"] = False
            material = metadata["channels"][0]["material_fit"]
            material["normalized_rms_alpha"] = 0.5
            material["normalized_rms_inverse_alpha"] = 0.5
            material["max_normalized_alpha_error"] = 0.8
            payload["metadata_json"] = np.asarray(json.dumps(metadata))

        path = self._write_modified_artifact(
            "approximate_material.npz", make_numerically_certified_approximation
        )
        with self.assertRaisesRegex(ValueError, "modal response"):
            load_work_loss_artifact(path)
        loaded = load_work_loss_artifact(
            path,
            allow_approximate_material_fit=True,
        )
        self.assertEqual(str(loaded["arrays"]["channel_key"][0]), "axis_long")

        def make_nonpassive_approximation(payload) -> None:
            make_numerically_certified_approximation(payload)
            metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
            metadata["channels"][0]["full_qs"][
                "modal_fit_passive_on_audit_grid"
            ] = False
            payload["metadata_json"] = np.asarray(json.dumps(metadata))

        nonpassive = self._write_modified_artifact(
            "nonpassive_approximation.npz", make_nonpassive_approximation
        )
        with self.assertRaisesRegex(ValueError, "modal passivity"):
            load_work_loss_artifact(
                nonpassive,
                allow_approximate_material_fit=True,
            )


if __name__ == "__main__":
    unittest.main()
