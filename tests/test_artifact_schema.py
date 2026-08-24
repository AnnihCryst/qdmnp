"""Regression tests for schema-3 pulse artifacts and schema-1 aliases."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from qd_mnp_pulse_absorption_sweep import save_artifact_run
from tests._fixtures import make_zero_mode_model


class PulseArtifactSchemaTests(unittest.TestCase):
    def test_schema3_records_and_preserves_every_observable_alias(self) -> None:
        canonical = {
            "sigma_energy_transfer_cm2": 1.0,
            "sigma_spectral_qs_work_loss_cm2": 2.0,
            "sigma_bare_mnp_qs_work_loss_cm2": 3.0,
            "delta_sigma_spectral_qs_work_loss_cm2": -1.0,
            "work_from_incident_field_j": 4.0,
        }
        aliases = {
            "sigma_energy_cm2": canonical["sigma_energy_transfer_cm2"],
            "sigma_spectral_cm2": canonical["sigma_spectral_qs_work_loss_cm2"],
            "sigma_bare_mnp_cm2": canonical["sigma_bare_mnp_qs_work_loss_cm2"],
            "sigma_spectral_minus_bare_cm2": canonical[
                "delta_sigma_spectral_qs_work_loss_cm2"
            ],
            "absorbed_energy_j": canonical["work_from_incident_field_j"],
        }
        row = {
            "peak_intensity_w_cm2": 5.0,
            "fluence_j_cm2": 6.0,
            "pulse_area_isolated_qd": 7.0,
            **canonical,
            # Historical schema fields retained in the artifact.
            "sigma_spectral_ext_cm2": 2.0,
            "sigma_bare_mnp_ext_cm2": 3.0,
            "delta_sigma_spectral_ext_cm2": -1.0,
            "sigma_spectral_sca_cm2": 0.2,
            "sigma_spectral_abs_cm2": 1.8,
            "sigma_spectral_formal_k_im_alpha_cm2": 2.0,
            "sigma_spectral_rayleigh_sca_estimate_cm2": 0.2,
            "sigma_spectral_optical_theorem_residual_cm2": 1.8,
            "sigma_bare_mnp_sca_cm2": 0.3,
            "sigma_bare_mnp_abs_cm2": 2.7,
            "delta_sigma_spectral_sca_cm2": -0.1,
            "delta_sigma_spectral_abs_cm2": -0.9,
            "post_fs_effective": 8.0,
            "response_tail_ratio": 1.0e-8,
            "tail_below_tolerance": True,
            "max_bloch_radius": 1.0,
            "min_density_eigenvalue": 0.0,
            "solver_nfev": 9,
            "t_final_reached": True,
            "solver_n_steps": 10,
            "solver_success": True,
            "linearized_ground_state_stable": True,
            "linearized_ground_state_spectral_abscissa_au": -1.0e-4,
            "applicability_diagnostic_energy_ev": 3.0,
            "medium_size_parameter_kc": 0.1,
            "medium_separation_parameter_kR": 0.2,
            "mnp_size_to_separation_ratio_c_over_R": 0.25,
            "qd_size_to_separation_ratio_rqd_over_R": 0.05,
            "quasistatic_guide_satisfied": True,
            "point_dipole_guide_satisfied": True,
            "mnp_fit_n_modes": 9,
            "mnp_fit_normalized_rms_alpha": 0.01,
            "mnp_fit_normalized_rms_inv_alpha": 0.02,
            "mnp_fit_max_relative_alpha_error": 0.03,
            "mnp_fit_globally_passive": True,
            "pulse_spectral_fraction_in_fit_window": 0.9999,
            "pulse_spectral_leakage": 0.0001,
            "mnp_drive_spectral_fraction_in_fit_window": 1.0,
            "mnp_drive_spectral_leakage": 0.0,
            "mnp_dipole_spectral_fraction_in_fit_window": 1.0,
            "mnp_dipole_spectral_leakage": 0.0,
            "work_passivity_checked": True,
            "work_passivity_tolerance_au": 1.0e-12,
            "work_nonnegative_within_tolerance": True,
            **aliases,
        }
        args = SimpleNamespace(
            tau_fs=[2.0],
            n_modes=9,
            fit_min_ev=0.8,
            fit_max_ev=3.0,
            weight_center_ev=None,
            weight_sigma_ev=None,
            omega_l_ev=2.0,
            pre_sigma=None,
            post_fs=None,
            orientation="long",
            spectral_window_policy="raise",
            max_spectral_leakage=1.0e-3,
            x_axis="fluence",
            method="DOP853",
            rtol=1.0e-8,
            atol=1.0e-10,
        )

        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            save_artifact_run(
                rows=[row],
                traces=[],
                params=make_zero_mode_model().params,
                args=args,
                e0_values_v_m=np.asarray([1.0]),
                run_dir=run_dir,
            )

            metadata = json.loads((run_dir / "params.json").read_text(encoding="utf-8"))
            alias_map = metadata["observables"]["legacy_aliases"]
            self.assertEqual(metadata["schema_version"], 3)
            physical = metadata["physical"]
            self.assertEqual(physical["model_profile"], "quasistatic_ellipsoid_tls")
            self.assertEqual(physical["orientation"], "long")
            self.assertEqual(physical["G"], 2.0)
            self.assertGreater(physical["surface_gap_nm"], 0.0)
            self.assertEqual(physical["qd_dipole_convention"], "effective_external")
            self.assertEqual(physical["qd_local_field_factor"], 1.0)
            self.assertTrue(metadata["fit"]["structurally_passive"])
            self.assertEqual(metadata["fit"]["normalized_rms_alpha"], 0.01)
            self.assertEqual(
                metadata["applicability"]["diagnostic_energy_ev"],
                3.0,
            )
            self.assertTrue(metadata["validation"]["all_tail_below_tolerance"])
            self.assertTrue(metadata["validation"]["all_work_passivity_checked"])
            self.assertTrue(
                metadata["validation"]["all_work_nonnegative_within_tolerance"]
            )
            self.assertEqual(alias_map, {
                "sigma_energy_cm2": "sigma_energy_transfer_cm2",
                "sigma_spectral_cm2": "sigma_spectral_qs_work_loss_cm2",
                "sigma_bare_mnp_cm2": "sigma_bare_mnp_qs_work_loss_cm2",
                "sigma_spectral_minus_bare_cm2": (
                    "delta_sigma_spectral_qs_work_loss_cm2"
                ),
                "absorbed_energy_j": "work_from_incident_field_j",
            })

            with np.load(run_dir / "data.npz") as data:
                for old_key, new_key in alias_map.items():
                    np.testing.assert_array_equal(data[old_key], data[new_key])
                self.assertEqual(int(data["mnp_fit_n_modes"]), 9)
                self.assertTrue(bool(data["work_passivity_checked"][0, 0]))
                self.assertTrue(
                    bool(data["work_nonnegative_within_tolerance"][0, 0])
                )
                self.assertEqual(
                    float(data["mnp_dipole_spectral_leakage"][0, 0]),
                    0.0,
                )


if __name__ == "__main__":
    unittest.main()
