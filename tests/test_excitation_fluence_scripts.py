"""Focused tests for the separated excitation-fluence calculation/plot path."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import matplotlib
import numpy as np

matplotlib.use("Agg")

from article_observables.qd_mnp_calculate_excitation_fluence import (
    HYBRID_CHANNELS,
    _channel_center_distance_nm,
    _resolved_settings,
    fluence_grid_resolution_diagnostics,
    parse_args as parse_calculation_args,
    write_excitation_artifact,
)
from article_observables.qd_mnp_plot_excitation_fluence import (
    load_excitation_artifact,
    plot_excitation_comparison,
    plot_excitation_fluence,
    threshold_fluence,
    validate_excitation_comparison,
)


def _certified_metadata(channel_id: np.ndarray, *, n_modes: int = 9) -> dict:
    settings = {
        "max_population_decay_fraction_at_read": 1.0e-3,
        "positivity_tolerance": 1.0e-7,
        "max_spectral_leakage": 1.0e-3,
        "tail_ratio_tolerance": 1.0e-4,
        "max_fluence_grid_midpoint_error": 0.01,
        "max_isolated_pulse_area_step_rad": 0.25,
        "pulse_energy_ev": 2.042,
        "pulse_tau_fs": 20.0,
        "pulse_tau_kind": "fwhm_intensity",
        "c_nm": 15.0,
        "a_nm": 7.0,
        "qd_radius_nm": 2.0,
        "gap_nm": 1.0,
        "eps_m": 1.0,
        "eps_qd": 6.0,
        "d_debye": 0.65,
        "omega0_ev": 2.042,
        "gamma_population_mev": 0.1,
        "gamma2_coherence_mev": 0.15,
        "qd_dipole_convention": "effective_external",
        "fit_min_ev": 0.8,
        "fit_max_ev": 3.0,
        "weight_center_ev": None,
        "weight_sigma_ev": None,
        "spatial_order_max": 80,
        "post_fs": 250.0,
        "start_sigma": 10.0,
    }
    channels = [{"channel_id": "bare_qd", "label": "isolated QD"}]
    for value in channel_id[1:]:
        channels.append(
            {
                "channel_id": str(value),
                "label": str(value).replace("_", " "),
                "material_fit": {
                    "n_modes": n_modes,
                    "max_fit_normalized_rms_gate": 0.025,
                    "max_fit_pointwise_relative_error_gate": 0.05,
                    "normalized_rms_alpha": 0.01 if n_modes > 1 else 0.5,
                    "normalized_rms_inv_alpha": 0.01 if n_modes > 1 else 0.4,
                    "max_normalized_alpha_error": 0.02 if n_modes > 1 else 0.8,
                    "passive_for_all_positive_frequencies": True,
                },
                "modal_transform_diagnostics": {
                    "accepted": n_modes > 1,
                    "passive_on_audit_grid": True,
                },
                "spatial_convergence_diagnostics": {"accepted": True},
                "coupled_stability_diagnostics": {"stable": True},
                "dark_reduction": None,
            }
        )
    return {
        "schema_name": "qd_mnp_excitation_fluence",
        "schema_version": 1,
        "resolved_settings": settings,
        "channels": channels,
        "array_units": {"fluence_j_cm2": "J cm^-2", "p_exc_read": "1"},
    }


def _certified_payload(channel_id: np.ndarray) -> dict[str, np.ndarray]:
    fluence = np.asarray([1.0e-8, 1.0e-7, 1.0e-6])
    base = np.asarray([0.01, 0.10, 0.60])
    population = np.stack(
        [
            np.clip(base * factor, 0.0, 1.0)
            for factor in (1.0, 1.4, 0.8, 1.1, 1.2, 0.7)
        ]
    )
    n_hybrid = channel_id.size - 1
    full_shape = (n_hybrid, fluence.size)
    pulse_area = np.sqrt(fluence)
    pulse_area *= 0.1 / float(np.max(np.diff(pulse_area)))
    payload = {
        "channel_id": channel_id,
        "fluence_j_cm2": fluence,
        "p_exc_read": population,
        "p_exc_max": np.minimum(population + 0.02, 1.0),
        "read_time_fs": np.full(fluence.size, 250.0),
        "bare_diagnostic__response_tail_converged": np.ones(fluence.size, dtype=bool),
        "full_diagnostic__response_tail_converged": np.ones(full_shape, dtype=bool),
        "population_decay_fraction_at_read": np.full(fluence.size, 1.0e-5),
        "fluence_grid_midpoint_interpolation_error": np.full(channel_id.size, 1.0e-3),
        "fluence_grid_converged_by_channel": np.ones(channel_id.size, dtype=bool),
        "maximum_isolated_pulse_area_step_rad": np.asarray(
            float(np.max(np.diff(pulse_area)))
        ),
        "isolated_qd_pulse_area_rad": pulse_area,
    }
    for prefix, shape in (
        ("bare_diagnostic__", fluence.shape),
        ("full_diagnostic__", full_shape),
    ):
        for name in ("solver_success", "t_final_reached", "state_is_finite"):
            payload[prefix + name] = np.ones(shape, dtype=bool)
        payload[prefix + "min_density_eigenvalue"] = np.zeros(shape)
        payload[prefix + "boundary_envelope_fraction"] = np.full(shape, 1.0e-12)
        payload[prefix + "pulse_spectral_leakage"] = np.full(shape, 1.0e-5)
        payload[prefix + "response_tail_ratio"] = np.full(shape, 1.0e-5)
        payload[prefix + "response_tail_tolerance"] = np.full(shape, 1.0e-4)
    payload["full_diagnostic__work_nonnegative_within_tolerance"] = np.ones(
        full_shape, dtype=bool
    )
    for name in (
        "qd_source_spectral_leakage",
        "mnp_drive_spectral_leakage",
        "mnp_dipole_spectral_leakage",
        "mnp_field_spectral_leakage",
    ):
        payload[f"full_diagnostic__{name}"] = np.full(full_shape, 1.0e-5)
    return payload


class ExcitationFluenceScriptTests(unittest.TestCase):
    def test_sqrt_fluence_grid_audits_rabi_like_curve_against_coarsening(self) -> None:
        amplitude = np.linspace(0.0, 2.0 * np.pi, 65)
        fluence = (amplitude + 1.0e-6) ** 2
        population = np.stack(
            [np.sin(0.5 * amplitude) ** 2, np.sin(0.65 * amplitude) ** 2]
        )
        diagnostics = fluence_grid_resolution_diagnostics(
            fluence,
            population,
            amplitude,
            max_midpoint_error=0.01,
            max_pulse_area_step_rad=0.11,
        )
        self.assertTrue(diagnostics["accepted"])
        self.assertLess(
            float(np.max(diagnostics["midpoint_interpolation_error_by_channel"])),
            0.01,
        )

    def test_all_five_channels_share_surface_gap_not_center_distance(self) -> None:
        settings = _resolved_settings(
            parse_calculation_args(["--preset", "quick", "--gap-nm", "1.5"])
        )
        distances = {
            channel.channel_id: _channel_center_distance_nm(channel, settings)
            for channel in HYBRID_CHANNELS
        }
        self.assertEqual(len(distances), 5)
        self.assertAlmostEqual(distances["axis_long"], 18.5)
        self.assertAlmostEqual(distances["axis_trans"], 18.5)
        self.assertAlmostEqual(distances["side_long"], 10.5)
        self.assertAlmostEqual(distances["side_trans_radial"], 10.5)
        self.assertAlmostEqual(distances["side_trans_tangential"], 10.5)

    def test_one_npz_round_trip_and_plotter_needs_no_model_import(self) -> None:
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
        payload = _certified_payload(channel_id)
        population = payload["p_exc_read"]
        metadata = _certified_metadata(channel_id)
        with TemporaryDirectory() as directory:
            artifact = Path(directory) / "calculation.npz"
            figure = Path(directory) / "figure.png"
            write_excitation_artifact(artifact, payload, metadata)
            loaded, loaded_metadata = load_excitation_artifact(artifact)
            np.testing.assert_array_equal(loaded["p_exc_read"], population)
            self.assertEqual(loaded_metadata["schema_version"], 1)
            with np.load(artifact, allow_pickle=False) as archive:
                self.assertIn("metadata_json", archive.files)
                self.assertEqual(json.loads(str(archive["metadata_json"].item()))["schema_name"], "qd_mnp_excitation_fluence")
            plot_excitation_fluence(
                loaded,
                loaded_metadata,
                figure,
                target_population=0.5,
                include_maximum=True,
                ratio_panel=True,
            )
            self.assertGreater(figure.stat().st_size, 0)

        plot_source = Path(
            "article_observables/qd_mnp_plot_excitation_fluence.py"
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

    def test_threshold_is_first_sqrt_fluence_crossing(self) -> None:
        fluence = np.asarray([1.0, 10.0, 100.0])
        population = np.asarray([0.1, 0.5, 0.9])
        self.assertAlmostEqual(threshold_fluence(fluence, population, 0.5), 10.0)
        expected = (1.0 + 0.5 * (np.sqrt(10.0) - 1.0)) ** 2
        self.assertAlmostEqual(threshold_fluence(fluence, population, 0.3), expected)
        self.assertTrue(np.isnan(threshold_fluence(fluence, population, 0.95)))
        self.assertTrue(
            np.isnan(
                threshold_fluence(
                    fluence, np.asarray([0.6, 0.7, 0.8]), 0.5
                )
            )
        )

    def test_multi_artifact_plot_requires_explicit_one_pole_label(self) -> None:
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
        with TemporaryDirectory() as directory:
            root = Path(directory)
            n9 = root / "n9.npz"
            n1 = root / "n1.npz"
            bad_modal = root / "n9_bad_modal.npz"
            figure = root / "comparison.png"
            write_excitation_artifact(
                n9, _certified_payload(channel_id), _certified_metadata(channel_id, n_modes=9)
            )
            write_excitation_artifact(
                n1, _certified_payload(channel_id), _certified_metadata(channel_id, n_modes=1)
            )
            bad_modal_metadata = _certified_metadata(channel_id, n_modes=9)
            bad_modal_metadata["channels"][1]["modal_transform_diagnostics"][
                "accepted"
            ] = False
            write_excitation_artifact(
                bad_modal,
                _certified_payload(channel_id),
                bad_modal_metadata,
            )
            loaded_n9 = load_excitation_artifact(n9)
            with self.assertRaisesRegex(ValueError, "material polarizability fit"):
                load_excitation_artifact(n1)
            loaded_n1 = load_excitation_artifact(
                n1, allow_approximate_material_fit=True
            )
            with self.assertRaisesRegex(ValueError, "modal-transform accuracy"):
                load_excitation_artifact(
                    bad_modal, allow_approximate_material_fit=True
                )
            mismatched_metadata = json.loads(json.dumps(loaded_n1[1]))
            mismatched_metadata["resolved_settings"]["rtol"] = 1.0e-5
            with self.assertRaisesRegex(ValueError, "rtol"):
                validate_excitation_comparison(
                    [loaded_n9, (loaded_n1[0], mismatched_metadata)]
                )
            plot_excitation_comparison(
                [loaded_n9, loaded_n1],
                figure,
                channel_filter=["bare_qd", "axis_long", "side_long"],
                ratio_panel=True,
            )
            self.assertGreater(figure.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
