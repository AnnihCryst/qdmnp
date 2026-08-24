"""Regression tests for schema-2 pulse artifacts and schema-1 aliases."""

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
    def test_schema2_records_and_preserves_every_observable_alias(self) -> None:
        canonical = {
            "sigma_energy_transfer_cm2": 1.0,
            "sigma_spectral_ext_cm2": 2.0,
            "sigma_bare_mnp_ext_cm2": 3.0,
            "delta_sigma_spectral_ext_cm2": -1.0,
            "work_from_incident_field_j": 4.0,
        }
        aliases = {
            "sigma_energy_cm2": canonical["sigma_energy_transfer_cm2"],
            "sigma_spectral_cm2": canonical["sigma_spectral_ext_cm2"],
            "sigma_bare_mnp_cm2": canonical["sigma_bare_mnp_ext_cm2"],
            "sigma_spectral_minus_bare_cm2": canonical["delta_sigma_spectral_ext_cm2"],
            "absorbed_energy_j": canonical["work_from_incident_field_j"],
        }
        row = {
            "peak_intensity_w_cm2": 5.0,
            "fluence_j_cm2": 6.0,
            "pulse_area_isolated_qd": 7.0,
            **canonical,
            "sigma_spectral_sca_cm2": 0.2,
            "sigma_spectral_abs_cm2": 1.8,
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
            **aliases,
        }
        args = SimpleNamespace(
            tau_fs=[2.0],
            n_modes=0,
            fit_min_ev=0.8,
            fit_max_ev=3.0,
            weight_center_ev=None,
            weight_sigma_ev=None,
            omega_l_ev=2.0,
            pre_sigma=None,
            post_fs=None,
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
            self.assertEqual(metadata["schema_version"], 2)
            self.assertEqual(alias_map, {
                "sigma_energy_cm2": "sigma_energy_transfer_cm2",
                "sigma_spectral_cm2": "sigma_spectral_ext_cm2",
                "sigma_bare_mnp_cm2": "sigma_bare_mnp_ext_cm2",
                "sigma_spectral_minus_bare_cm2": "delta_sigma_spectral_ext_cm2",
                "absorbed_energy_j": "work_from_incident_field_j",
            })

            with np.load(run_dir / "data.npz") as data:
                for old_key, new_key in alias_map.items():
                    np.testing.assert_array_equal(data[old_key], data[new_key])


if __name__ == "__main__":
    unittest.main()
